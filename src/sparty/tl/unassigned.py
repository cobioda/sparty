import numpy as np
import pandas as pd 
from scipy.sparse import coo_matrix
import anndata as ad
import scanpy as sc
from spatialdata.transformations import Identity, Scale, Sequence
from spatialdata.models import ShapesModel
import geopandas as gpd
import spatialdata as sd
import shapely 

from ..pp.transcripts import subset_transcripts

def compute_matrix_from_transcripts(
    data,
    cell_id: str = 'bin_id',
    feature_key: str = 'feature_name'
):
    bin_codes, bin_uniques = pd.factorize(data[cell_id])
    feat_codes, feat_uniques = pd.factorize(data[feature_key])

    n_bins = len(bin_uniques)
    n_feat = len(feat_uniques)

    matrix = coo_matrix((np.ones(
        len(bin_codes), dtype=np.int32), (bin_codes, feat_codes)),
        shape=(n_bins, n_feat)
    ).tocsr()
    return matrix, bin_uniques, feat_uniques


def min_max_round(
    data, precision = 1e-3, ndigits= 5
):
    xmin= data["x"].min()
    ymin= data["y"].min()
    xmax= data["x"].max()
    ymax= data["y"].max()

    return (
        round(xmin, ndigits) - precision,
        round(ymin, ndigits) - precision,
        round(xmax, ndigits) + precision,
        round(ymax, ndigits) + precision
        )

def assign_bins(
    data,
    bin_key: str = 'bin_id',
    bin_size_um: int = 10,
):
    xmin, ymin, xmax, ymax = min_max_round(data)

    x_bins = int((xmax - xmin) / bin_size_um)
    y_bins = int((ymax - ymin) / bin_size_um)

    x_edges = np.linspace(xmin, xmax, x_bins+1, endpoint=True)
    y_edges = np.linspace(ymin, ymax, y_bins+1, endpoint=True)

    data["x_bin"] = pd.cut(
        data["x"],
        bins=x_edges,
        labels=False,
        precision=5,
        right=False,          # intervalle (a, b]
        include_lowest=True  # inclut xmin
    )

    data["y_bin"] = pd.cut(
        data["y"],
        bins=y_edges,
        labels=False,precision=5,
        right=False,          # intervalle (a, b]
        include_lowest=True  # inclut xmin
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    data["center_x"] = x_centers[data["x_bin"].values]
    data["center_y"] = y_centers[data["y_bin"].values]

    n_y = data["y_bin"].nunique()  # e.g., 5
    data[bin_key] = data["x_bin"] * n_y + data["y_bin"]

    return data, x_edges, y_edges

def bin_to_gpd_grid(
    x_edges, 
    y_edges,
    n_y,
    cell_key: str = 'bin_id',
):
    grid, cell_ids, ix_list, iy_list = [], [], [], []

    for j in range(len(y_edges) - 1):
        ymin, ymax = y_edges[j], y_edges[j+1]

        for i in range(len(x_edges) - 1):
            xmin, xmax = x_edges[i], x_edges[i+1]
            
            poly = shapely.Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ])
            grid.append(poly)
            cell_ids.append(str(i * n_y + j))
            
            ix_list.append(i)
            iy_list.append(j)

    gdf = gpd.GeoDataFrame({
        cell_key: cell_ids,
        "geometry": grid,
        "ix": ix_list, # optionel
        "iy": iy_list, # optionel
    })
    gdf.index = gdf[cell_key].values
    
    return gdf


def unassigned_RNA(
    sdata,
    bin_size_um: int = 10,
    # genes = None,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    shape_key: str = 'bin_boundaries',
    table_key: str = 'unassigned',
    bin_key: str = 'bin_id',
    techno: str = "Xenium",
    min_counts: int = 0,
    add_bin_shape: bool = False,
    only_scale: bool = False,
    target_coordinates: str ='global',
):
    print('1) Start by subseting all valid transcripts...')
    data = subset_transcripts(
        sdata=sdata,
        genes= None,
        feature_key=feature_key,
        transcript_key= transcript_key,
        techno=techno,
        only_in_cell = False, # a modifier pour prendre uniquement "cell", "outside" or "all" ==> ex : which_transcripts = "cell" or "outside" or "all"
        only_outside = True,
        scale=False,
        return_gpd = False,
        ) 
    
    print("2) Create bins and assign transcripts to a bin identifiant...")
    data, x_edges, y_edges = assign_bins(data, bin_size_um=bin_size_um)
    
    print('3) Create metadata...')
    centroid_dict = (
        data.groupby(bin_key)[["center_x", "center_y"]]
        .first()
        .apply(tuple, axis=1)
        .to_dict()
    )
    metadata = pd.DataFrame(centroid_dict, index=['center_x', 'center_y']).T
    
    print('3) Compute matrix bin by gene...')
    matrix, bin_uniques, feat_uniques = compute_matrix_from_transcripts(
        data,    
        cell_id= bin_key,
        feature_key = 'feature_name'
    )
    
    print('4) Create anndata object...')
    bin_adata = ad.AnnData(
        X = matrix, 
        obs = metadata.loc[bin_uniques],
        var = pd.DataFrame(index=feat_uniques)
        )
    bin_adata.obs[bin_key] = bin_adata.obs.index
    bin_adata.obsm['spatial'] = bin_adata.obs[['center_x', 'center_y']].values
    bin_adata.uns['spatialdata_attrs'] = {
        'feature_key': feature_key,
        'instance_key': bin_key,
    }
    # print(bin_adata.n_obs)

    if min_counts:
        sc.pp.filter_cells(bin_adata, min_counts=min_counts)
        # print(bin_adata.n_obs)

    if add_bin_shape:
        print('5) Add bin shape...')
        n_y = data["y_bin"].nunique()  
        gdf = bin_to_gpd_grid(
            x_edges=x_edges, 
            y_edges=y_edges,
            cell_key=bin_key,
            n_y=n_y, 
        )
        # print(len(gdf))
        gdf = gdf.loc[bin_adata.obs_names]
        # print(len(gdf))
        sdata.shapes[shape_key] = ShapesModel.parse(
            gdf, 
            transformations = {target_coordinates: Identity()} #si deja dans coordinate_system ne pas le mettre !!
        )
        if only_scale:
            transfo = sd.transformations.get_transformation(
                sdata["cell_boundaries"], target_coordinates
            )
            if isinstance(transfo, Sequence):
                for t in transfo.transformations:
                    if isinstance(t, Scale):
                        transfo = t
            # if isinstance(transfo, Sequence):
            #     transfo = next(
            #         (t for t in transfo.transformations if isinstance(t, Scale)),
            #         None
            #     )

            if isinstance(transfo, Scale):
                sd.transformations.set_transformation(
                    sdata.shapes[shape_key],
                    transfo,
                    to_coordinate_system=target_coordinates,
                )

        bin_adata.obs['region'] = shape_key
        bin_adata.obs['region'] = bin_adata.obs['region'].astype("category")

        bin_adata.uns['spatialdata_attrs']['region'] = shape_key
        bin_adata.uns['spatialdata_attrs']['region_key'] = 'region'

        sdata.tables[table_key] = bin_adata
    else:
        return bin_adata


