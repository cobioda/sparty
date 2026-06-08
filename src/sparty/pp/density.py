import spatialdata as sd
import numpy as np
import shapely
import geopandas as gpd
from scipy.stats import gaussian_kde
from spatialdata.transformations import get_transformation 
# import dask.dataframe as dd

from ..pp.transformations import compute_bounds_gpd
from ..pp.transcripts import subset_transcripts


def _count_dens(
    x, y, xmax, xmin, ymax, ymin, 
    bin_size_um 
):
    x_bins = int((xmax - xmin) / bin_size_um)
    y_bins = int((ymax - ymin) / bin_size_um)

    x_edges = np.linspace(xmin, xmax, x_bins + 1)
    y_edges = np.linspace(ymin, ymax, y_bins + 1)

    heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    heatmap = heatmap.T 

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)

    return heatmap, xx, yy


def _kde_dens(
    x, 
    y,
    xmax, xmin, ymax, ymin, 
    nb_grid: int = 100j, 
    smooth: float = 1.0,
):
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy, bw_method=lambda s: s.scotts_factor() * smooth)

    yy, xx = np.mgrid[ymin:ymax:nb_grid, xmin:xmax:nb_grid]  # nb_grid x nb_grid
    positions = np.vstack([xx.ravel(),yy.ravel()])
    density = np.reshape(kde(positions), xx.shape)

    return density, xx, yy


def density_count_genes(
    sdata: sd.SpatialData,
    genes: list | str,
    polygon: shapely.Polygon = None,
    shape_key: str = "cell_boundaries",
    transcript_key: str = 'transcripts',
    nb_grid: int = 200j, 
    techno: str = "Xenium", # 'Xenium' or 'Merscope'
    smooth: float = 1.0,
    feature_key: str = 'feature_name',
    bin_size_um: float = 10.0,
    box_bounds: list | tuple = None,
    only_in_cell: bool = False,
    only_outside: bool = False,
    density_kde: bool = False,
    scale: bool = False,
    pct_max: int = 99,
    clip_outside: bool = False,
    by_codeword: bool = False,
    aggregate: bool = False,
):
    if type(genes) == str:
        genes = [genes]

    if box_bounds or polygon:
        return_gpd = True
    else: 
        return_gpd = False # faster
    
    data = subset_transcripts(
        sdata=sdata,
        genes=genes,
        feature_key=feature_key,
        transcript_key= transcript_key,
        techno=techno,
        only_in_cell=only_in_cell,
        only_outside=only_outside,
        scale=scale,
        return_gpd = return_gpd,
        )   
    
    if box_bounds and len(box_bounds) == 4:
        print("bbox")
        xmin, ymin, xmax, ymax = box_bounds 
        polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)
        data = data[data.within(polygon)]
    elif polygon:
        print("pol")
        xmin, ymin, xmax, ymax = polygon.bounds 
        if clip_outside:
            data = data[data.within(polygon)]
    elif not polygon and shape_key:
        print('Shape key default')
        xmin, ymin, xmax, ymax = compute_bounds_gpd(
            shape=sdata[shape_key], 
            transfo=get_transformation(sdata[shape_key]), 
            scale=scale)
        polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)

    if by_codeword and len(genes) == 1:
        feature_key = 'codeword_index'
        global_gene = genes[0]
        genes = data[feature_key].unique().tolist()
        print(f'Gene {" ".join(global_gene)} will be split by codewords : {" ".join([str(x) for x in genes])}.')
    
    
    results = {}

    if (aggregate and len(genes) > 1) or (only_outside):
        gene_iter = [None]   # only one
    else:
        gene_iter = genes

    for gene in gene_iter:
        if only_outside: # and not gene: # ajoutee 04/03/26
            sub_data = data
            key = 'Unassigned RNA'

        elif aggregate and len(genes) > 1:
            sub_data = data
            key = " + ".join(genes)
        else:
            sub_data = data[data[feature_key] == gene]
            key = f'{global_gene}_{gene}' if by_codeword else gene

        x = sub_data['x']
        y = sub_data['y']
        # x = sub_data.geometry.centroid.x
        # y = sub_data.geometry.centroid.y

        if density_kde:
            heatmap, xx, yy = _kde_dens(
                x, y, xmax, xmin, ymax, ymin, nb_grid, smooth
            )
        else:
            heatmap, xx, yy = _count_dens(
                x, y, xmax, xmin, ymax, ymin, bin_size_um
            )

        vmax = np.max([1, np.percentile(heatmap, pct_max)])
        # if clip_outside:
        #     data = data[data.within(polygon)]
        #     heatmap[~data.values.reshape(xx.shape)] = np.nan  # mettre à 0 en dehors
        results[key] = (heatmap, vmax, xx, yy)
    return results, (xmin, ymin, xmax, ymax)



def compute_coloc(
    sdata,
    genes: list,
    transcript_key: str = "transcripts",
    feature_key: str = "feature_name",
    table_key: str = "table",
    only_in_cell: bool = False,
    bin_size_um: int = 20,
):
    """
    Generate density heatmaps for 2 genes and return:
        heatmap1, heatmap2, xmin, xmax, ymin, ymax
    """

    if (not isinstance(genes, list)) or (len(genes) != 2):
        raise ValueError("Please provide exactly 2 genes in a list.")

    for g in genes:
        if g not in sdata[table_key].var_names:
            raise ValueError(f"Gene {g} not found in dataset.")

    df_transcripts = subset_transcripts(
        sdata=sdata,
        genes=genes,
        only_in_cell=only_in_cell,
        transcript_key=transcript_key,
        return_gpd = True,
    )

    multi = shapely.MultiPoint(df_transcripts.geometry.values)
    xmin, ymin, xmax, ymax = multi.bounds

    heatmaps = {}

    for gene in genes:

        sub = df_transcripts[
            df_transcripts[feature_key] == gene
        ].copy()

        heatmap, _, _ = _count_dens(
            sub["x"], sub["y"],
            xmax, xmin, ymax, ymin,
            bin_size_um
        )

        heatmaps[gene] = heatmap

    return heatmaps, (xmin, xmax, ymin, ymax)



    # results = {}
    # for gene in genes:
    #     sub_data = data[data[feature_key] == gene]
    #     x = sub_data.geometry.centroid.x
    #     y = sub_data.geometry.centroid.y

    #     if density_kde:
    #         heatmap, xx, yy = _kde_dens(x, y, xmax, xmin, ymax, ymin, nb_grid, smooth)
    #         vmax = np.max([1, np.percentile(heatmap, pct_max)])
    #     else:
    #         heatmap, xx, yy = _count_dens(x, y, xmax, xmin, ymax, ymin, bin_size_um)
    #         vmax = np.max([1, np.percentile(heatmap, pct_max)])

    #     if by_codeword:
    #         results[f'{global_gene}_{str(gene)}'] = (heatmap, vmax, xx, yy)
    #     else:
    #         results[gene] = (heatmap, vmax, xx, yy)

    # return results, (xmin, ymin, xmax, ymax)




# def density_count_genes(
#     sdata: sd.SpatialData,
#     genes: list | str,
#     polygon: shapely.Polygon = None,
#     shape_key: str = "cell_boundaries",
#     transcript_key: str = 'transcripts',
#     nb_grid: int = 200j, 
#     smooth: float = 1.0,
#     feature_key: str = 'feature_name',
#     bin_size_um: float = 10.0,
#     figsize: tuple = (8,8),
#     density_kde: bool = False,
#     pct_max: int = 99,
#     cmap = plt.cm.viridis,
#     clip_outside: bool = False,
#     origin: str = 'upper', # or lower 
#     ax = None, 
# ):
#     if type(genes) == str:
#         genes = [genes]

#     data = _subset_transcripts(
#         sdata=sdata,
#         genes=genes,
#         feature_key=feature_key,
#         transcript_key= transcript_key
#         )   
#     if not isinstance(data, gpd.GeoDataFrame):
#         data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['x'], data['y']))

#     if not polygon and shape_key:
#         xmin, ymin, xmax, ymax = sdata[shape_key].total_bounds
#         polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)
#     elif polygon:
#         xmin, ymin, xmax, ymax = polygon.bounds 
#         if clip_outside:
#             data = data[data.within(polygon)]
    
#     x = data.geometry.centroid.x
#     y = data.geometry.centroid.y

#     if density_kde:
#         print('kde')
#         heatmap, xx, yy = _kde_dens(x, y, xmax, xmin, ymax, ymin, nb_grid, smooth)
#         vmax = None
#     else:
#         print('histo')
#         heatmap, xx, yy = _count_dens(x, y, xmax, xmin, ymax, ymin, bin_size_um)
#         vmax = np.max([1, np.percentile(heatmap, pct_max)])
    
#     cmap.set_under('white')  

#     created_fig = False
#     if ax is None:
#         _, ax = plt.subplots(figsize=figsize)
#         # ax.set_title(f"Density plot - {':'.join(genes)}")
#         created_fig = True

#     if clip_outside:
#         coords = shapely.get_coordinates(polygon.boundary)
#         grid_points = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()))
#         mask = grid_points.within(polygon).to_numpy().reshape(heatmap.shape)
#         heatmap = np.where(mask, heatmap, np.nan) # or -1
#         ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2)

#     img = ax.imshow(
#         heatmap,
#         cmap=cmap,
#         origin="lower",
#         extent=[xmin, xmax, ymin, ymax],
#         vmax=vmax
#     )
#     ax.set_title(f"De nsity plot - {':'.join(genes)}")

#     if created_fig:
#         if origin == 'upper':
#             ax.invert_yaxis()
#         plt.colorbar(img, ax=ax, label="Counts", shrink=0.5)
#         plt.show()
#         return
#     else:
#         return img, vmax



# def density_count_genes(
#     sdata: sd.SpatialData,
#     genes: list | str,
#     shape: shapely.Polygon = None,
#     shape_key: str = "cell_boundaries",
#     transcript_key: str = 'transcripts',
#     # nb_grid: int = 100j, 
#     bin_size_um: float = 10.0,
#     figsize: tuple = (8,8),
#     pct_max: int = 99,
#     cmap = plt.cm.viridis,
#     clip_outside: bool = True,
#     ax = None, 
# ):
#     data = subset_transcripts(
#         sdata=sdata,
#         genes=genes,
#         transcript_key= transcript_key
#         )   
#     if not isinstance(data, gpd.GeoDataFrame):
#         data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['x'], data['y']))

#     if not shape and shape_key:
#         xmin, ymin, xmax, ymax = sdata[shape_key].total_bounds
#     elif shape:
#         xmin, ymin, xmax, ymax = shape.bounds 
#         if clip_outside:
#             data = data[data.within(shape)]
    
#     x = data.geometry.centroid.x
#     y = data.geometry.centroid.y

#     x_bins = int((xmax - xmin) / bin_size_um)
#     y_bins = int((ymax - ymin) / bin_size_um)

#     x_edges = np.linspace(xmin, xmax, x_bins + 1)
#     y_edges = np.linspace(ymin, ymax, y_bins + 1)

#     heatmap, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
#     heatmap = heatmap.T 

#     vmax = np.percentile(heatmap, pct_max)
#     if ax is None:
#         _, ax = plt.subplots(figsize=figsize)
#         ax.set_title(f"Density plot - {':'.join(genes)}")
#         img = ax.imshow(
#             heatmap,
#             cmap=cmap,
#             origin="lower",
#             extent=[xmin, xmax, ymin, ymax],
#             # interpolation="nearest"
#             # alpha=1.0
#             vmax=vmax
#         )
#         plt.colorbar(img, ax=ax, label="Counts", shrink=0.5)
#         plt.show()
#     else: 
#         img = ax.imshow(
#             heatmap,
#             cmap=cmap,
#             origin="lower",
#             extent=[xmin, xmax, ymin, ymax],
#             # interpolation="nearest"
#             # alpha=1.0
#             vmax=vmax
#         )
#         # ax.set_title("Heatmap brute - Comptes de points par pixel")
#         # plt.colorbar(img, ax=ax, label="Counts", shrink=0.5)
#         return img, vmax
        
