import pandas as pd
import scanpy as sc
import spatialdata as sd
from matplotlib import pyplot as plt
import math
import numpy as np

# import scvi
# import anndata as an

def metrics_summary(
    sdata: sd.SpatialData,
    scale_factor: float = 0.2125,
    table_key: str = 'table',
    transcripts_key: str = 'transcripts',
    image_key: str = 'morphology_focus',
) -> 'dict':
    """    Generate a summary of metrics for a SpatialData object that come from the Xenium technology.

    Parameters
    ----------
    sdata
        SpatialData object.
    scale_factor
        Scale factor to convert pixel dimensions to microns.
    Returns
    -------
    dict
        Dictionary containing various summary metrics.
    """
    adata = sdata[table_key].copy()
    transcripts = sdata[transcripts_key]
    img_dims = sdata[image_key]['scale0'].dims

    number_of_cells_detected = adata.n_obs
    median_transcripts_per_cell = adata.obs['transcript_counts'][adata.obs['transcript_counts'] != 0].median()
    adata.obs['genes_per_cell'] = np.sum(adata.X > 0, axis=1)
    median_genes_per_cell = adata.obs['genes_per_cell'][adata.obs['genes_per_cell'] != 0].median()

    total_cell_area = round(adata.obs['cell_area'].sum(), ndigits = 1)
    segmentation_method = adata.obs['segmentation_method'].value_counts()
    # percent_segmentation_method = dict(round(segmentation_method / number_of_cells_detected * 100, ndigits=1))
    percent_segmentation_method = round(segmentation_method / number_of_cells_detected * 100, ndigits=1)
    
    total_high_quality_decoded_transcripts = len(transcripts[
                                                 (transcripts['qv'] >= 20) &
                                                 (transcripts['is_gene'])
                                                 ])
    # is_gene = TRUE -> custom_gene + predesigned_gene
    number_of_fov_selected = len(transcripts['fov_name'].unique())
    transcripts_within_cells = len(transcripts[
                                   (transcripts['qv'] >= 20) &
                                   (transcripts['is_gene']) &
                                   (transcripts['cell_id'] != "UNASSIGNED")
                                   ])
    
    percent_of_transcripts_within_cells = round(transcripts_within_cells / total_high_quality_decoded_transcripts * 100, 
                                                ndigits=1)
    
    nuclear_transcript_count = len(transcripts[
                                   (transcripts['qv'] >= 20) &
                                   (transcripts['is_gene']) &
                                   (transcripts['cell_id'] != "UNASSIGNED") &
                                   (transcripts['overlaps_nucleus'] == 1)
                                   ])
    # The density of cells per 100 microns squared.
    nuclear_transcripts_per_100µm = round(nuclear_transcript_count / 
                                          adata.obs['nucleus_area'].sum() *100,
                                          ndigits=1)
    
    # TO improve
    region_area = round((img_dims['x'] * scale_factor) * (img_dims['y'] * scale_factor), 
                        ndigits=1) 
    # 3591422.2 ERROR trouver la bonne technique
    # 3,131,787.3
    cells_per_100µm = round(number_of_cells_detected / region_area * 100,
                            ndigits=2)
    # The density of cells per 100 microns squared.
    # 0,22 ERROR -> 0,25

    dict_columns = {'Number of cells detected': number_of_cells_detected,
                    'Median transcripts per cell': median_transcripts_per_cell,
                    'Nuclear transcripts per 100 µm²': nuclear_transcripts_per_100µm,
                    'Total high quality decoded transcripts': total_high_quality_decoded_transcripts,
                    'Number of FOV selected': number_of_fov_selected,
                    'Region area (µm²)': region_area,
                    'Total cell area (µm²)': total_cell_area,
                    'Percent of transcripts within cells': percent_of_transcripts_within_cells,	
                    'Cells per 100 µm²': cells_per_100µm,
                    'Median genes per cell': median_genes_per_cell,
                   } | dict(segmentation_method)
    return dict_columns



def subsetSamples(
    sdata: sd.SpatialData,
    sample: str,
    regions: list,
    manip: dict,
    fov_locations: pd.DataFrame,
    scale_factor: float = 0.2125,
    region: str = 'cell_boundaries',
    techno: str = 'Xenium',
) -> sd.SpatialData:
    """    Subset SpatialData object based on specified regions and sample.

    Parameters
    ----------
    sdata
        SpatialData object.
    sample
        Sample identifier.
    regions
        List of regions to subset.
    manip  
        Dictionary containing manipulation metadata.
    fov_locations
        DataFrame containing field of view locations.
    scale_factor
        Scale factor to convert pixel dimensions to microns.
    region
        Region key in SpatialData object.
    techno
        Technology type (e.g., 'Xenium').
        
    Returns
    -------
    sd.SpatialData
        Subsetted SpatialData object.
    """
    manip['Region name'] = sample
    width = fov_locations.iloc[0]['width']
    height = fov_locations.iloc[0]['height']

    print(f'Start with: {sample}')
    x_min = math.ceil(fov_locations.loc[regions,]['x'].min() / scale_factor)
    x_max = math.ceil((fov_locations.loc[regions,]['x'].max() + width) / scale_factor)   
    y_min = math.ceil(fov_locations.loc[regions,]['y'].min() / scale_factor)
    y_max = math.ceil((fov_locations.loc[regions,]['y'].max() + height) / scale_factor) 
    print(f'xmin: {x_min}, xmax: {x_max}')
    print(f'ymin: {y_min}, ymax: {y_max}')
        
    print("Start subset region...")
    sub_sdata = sd.bounding_box_query(sdata,
                                      min_coordinate=[x_min, y_min],
                                      max_coordinate=[x_max, y_max],
                                      axes=("x", "y"),
                                      target_coordinate_system="global")
    sub_sdata['table'].uns["spatialdata_attrs"]["region"] = region
    sub_sdata['table'].obs['region'] = region
    print(sub_sdata['table'].uns["spatialdata_attrs"])
    print("Done subset region")
    return sub_sdata



def prepro_qc_scanpy(
    sdata: sd.SpatialData,
    min_counts: int = 20,
    min_genes: int = 0,
    pct_negative: float | None = 5.0,
    layer: str = "counts",
    table_key: str = "table",
    positive_coord_only: bool = False,
):
    """Preprocess and QC-filter SpatialData table before Scanpy analysis.

    Parameters
    ----------
    sdata
        SpatialData object.
    min_counts
        Minimum transcript count to keep a cell.
    min_genes
        Minimum genes detected to keep a cell.
    pct_negative
        Maximum allowed percentage of negative probes.
        If None, skip negative probe filtering.
    layer
        Layer to use for counts.
    table_key
        Key for table in SpatialData.
    positive_coord_only
        If True, keep only cells with positive center_x and center_y.

    Returns
    -------
    adata
        Filtered AnnData object.
    """
    adata = sdata[table_key].copy()
    print("Total cells =", adata.n_obs)

    if layer not in adata.layers:
        adata.layers[layer] = adata.X

    adata.obs["n_genes_by_counts"] = (adata.layers[layer] > 0).sum(axis=1)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    adata.obs["pct_negative_probes"] = (
        (adata.obs["total_counts"] - adata.obs["transcript_counts"])
        / adata.obs["total_counts"]
        * 100
    )
    if pct_negative is not None:
        adata = adata[adata.obs["pct_negative_probes"] <= pct_negative].copy()

    if positive_coord_only:
        adata = adata[
            (adata.obs["center_x"] > 0) & (adata.obs["center_y"] > 0)
        ].copy()

    print("Remaining cells =", adata.n_obs)

    # del sdata.tables[table_key]
    sdata[table_key] = adata
    return 


def run_scanpy(
    sdata: sd.SpatialData,
    resolution: float = 0.5,
    layer: str = "counts",
    key: str = "leiden",
    table_key: str = "table",
    scale: bool = False,
):
    """Run Scanpy normalization, dimensionality reduction, clustering, and plotting.

    Parameters
    ----------
    sdata
        SpatialData object (already QC-filtered).
    resolution
        Leiden clustering resolution.
    layer
        Layer to use for counts.
    key
        Key to store clustering labels.
    table_key
        Key for table in SpatialData.
    scale
        If True, scale data before downstream analysis.

    Returns
    -------
    adata
        Processed AnnData object.
    """
    adata = sdata[table_key].copy()

    # Normalize + log transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata

    # Optional scaling
    if scale:
        sc.pp.scale(adata, max_value=10)

    # PCA + neighbors + UMAP + clustering
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added=key)

    # Plot
    _, axs = plt.subplots(1, 2, figsize=(20, 6))
    sc.pl.embedding(adata, "umap", color=key, ax=axs[0], show=False)
    sc.pl.embedding(adata, "spatial", color=key, ax=axs[1])
    plt.tight_layout()

    # Synchronize shapes
    sync_shape(sdata)
    sdata[table_key] = adata

    # Save updated table back
    # del sdata.tables[table_key]
    # sdata[table_key] = adata
    return 


# def run_scanpy(
#     sdata: sd.SpatialData,
#     min_counts: int = 20,
#     min_genes: int = 0,
#     resolution: float = 0.5,
#     pct_negative: float = 5.0,
#     layer: str ='counts',
#     key: str = "leiden",
#     table_key: str = 'table',
#     scale: bool = False,
#     positive_coord_only: bool = False,
# ):
#     """Filter and run scanpy analysis.

#     Parameters
#     ----------
#     sdata
#         SpatialData object.
#     min_counts
#         minimum transcript count to keep cell.
#     resolution
#         resolution for clustering.
#     key
#         key to add for clusters.

#     """
#     adata = sdata[table_key].copy()
#     print("total cells=", adata.n_obs)
#     if layer not in adata.layers:
#         adata.layers[layer] = adata.X
#     adata.obs["n_genes_by_counts"] =  (adata.layers['counts'] > 0).sum(axis=1)

#     sc.pp.filter_cells(adata, min_counts=min_counts)
#     sc.pp.filter_cells(adata, min_genes=min_genes)

#     adata.obs["pct_negative_probes"] = (adata.obs['total_counts'] - adata.obs['transcript_counts'])/adata.obs['total_counts'] * 100
#     adata = adata[adata.obs["pct_negative_probes"] <= pct_negative]

#     # filter also cells with negative coordinate center_x and center_y
#     # sdata['table'].obs.loc[sdata['table'].obs['center_x'] < 0, 'n_counts'] = 0
#     # sdata['table'].obs.loc[sdata['table'].obs['center_y'] < 0, 'n_counts'] = 0
#     # sc.pp.calculate_qc_metrics(
#     #     adata, inplace=True, percent_top=[20], log1p=True
#     # )

#     if positive_coord_only is True:
#         adata = adata[adata.obs.center_x > 0]
#         adata2 = adata[adata.obs.center_y > 0]
#         del sdata.tables["table"]
#         sdata['table'] = adata2

#     print("remaining cells=", adata.n_obs)

#     sc.pp.normalize_total(adata)
#     sc.pp.log1p(adata)
#     adata.raw = adata

#     sc.tl.pca(adata, svd_solver="arpack")
#     sc.pp.neighbors(adata, n_neighbors=10)
#     sc.tl.umap(adata)
#     sc.tl.leiden(adata, resolution=resolution, key_added=key)

#     if scale:
#         sc.pp.scale(adata, max_value=10)
        
#     _, axs = plt.subplots(1, 2, figsize=(20, 6))
#     sc.pl.embedding(adata, "umap", color=key, ax=axs[0], show=False)
#     sc.pl.embedding(adata, "spatial", color=key, ax=axs[1])
#     plt.tight_layout()

#     # synchronize current shapes with filtered table
#     sync_shape(sdata)

#     # for vizgen previous analysis
    # sdata.shapes['cell_boundaries'] = sdata.shapes['cell_boundaries'].loc[sdata['table'].obs.index.tolist()]


# def scvi_annotate(
#     ad_spatial: an.AnnData,
#     ad_ref: an.AnnData,
#     label_ref: str = "celltype",
#     label_key: str = "celltype",
#     layer: str = "counts",
#     batch_size: int = 128,
#     metaref2add: tuple = [],
#     filter_under_score: float = 0.5,
# ):
#     """Annotate anndata spatial cells using anndata cells reference using SCVI.

#     Parameters
#     ----------
#     ad_spatial
#         Anndata spatial object.
#     ad_ref
#         Anndata single-cell reference object.
#     label_ref
#         .obs key in single-cell reference object.
#     label_key
#         .obs key in spatial object.
#     layer
#         layer in which we can find the raw count values.
#     metaref2add
#         .obs key in single-cell reference object to transfert to spatial.
#     filter_under_score
#         remove cells having a scvi assignment score under this cutoff

#     """
#     ad_spatial.var.index = ad_spatial.var.index.str.upper()
#     ad_ref.var.index = ad_ref.var.index.str.upper()

#     print("ref. ", ad_ref.shape)
#     print("viz. ", ad_spatial.shape)

#     # Select shared gene panel genes only
#     genes_Vizgen = ad_spatial.var.index
#     genes_10x = ad_ref.var.index
#     genes_shared = genes_Vizgen.intersection(genes_10x)  # List of shared genes
#     ad_emb = ad_spatial[:, genes_Vizgen.isin(genes_shared)].copy()
#     ad_ref = ad_ref[:, genes_10x.isin(genes_shared)]

#     print(len(genes_shared), "common genes")

#     # missed = list(set(genes_Vizgen) - set(genes_shared))
#     # print("gene missed = ", missed)

#     # Concatenate the datasets
#     # both needs to have the count values in the layer "counts" or layer received
#     concat = ad_ref.concatenate(ad_emb, batch_key="tech", batch_categories=["10x", "MERFISH"]).copy()
#     # concat.layers[layer] = concat.raw.X.copy()

#     # Use the annotations from the 10x, and treat the MERFISH as unlabeled
#     concat.obs[f"{label_key}"] = "nan"
#     mask = concat.obs["tech"] == "10x"
#     concat.obs[f"{label_key}"][mask] = ad_ref.obs[label_ref].values

#     # Create the scVI latent space
#     scvi.model.SCVI.setup_anndata(concat, layer=layer, batch_key="tech")
#     vae = scvi.model.SCVI(concat)
#     # Train the model
#     vae.train(batch_size=batch_size)

#     # Register the object and run scANVI
#     scvi.model.SCANVI.setup_anndata(
#         concat,
#         layer=layer,
#         batch_key="tech",
#         labels_key=label_key,
#         unlabeled_category="nan",
#     )

#     lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=label_key, unlabeled_category="nan", adata=concat)
#     lvae.train(max_epochs=20, n_samples_per_label=100, batch_size=batch_size)

#     concat.obs["C_scANVI"] = lvae.predict(concat)
#     concat.obsm["X_scANVI"] = lvae.get_latent_representation(concat)

#     # add score
#     df_soft = lvae.predict(concat, soft=True)
#     concat.obs["score"] = df_soft.max(axis=1)

#     merfish_mask = concat.obs["tech"] == "MERFISH"
#     ad_spatial.obs[f"{label_key}"] = concat.obs["C_scANVI"][merfish_mask].values
#     ad_spatial.obs[f"{label_key}_score"] = concat.obs["score"][merfish_mask].values
#     ad_spatial.obs[f"{label_key}"] = ad_spatial.obs[f"{label_key}"].astype("category")

#     for i in range(0, len(metaref2add)):
#         d = pd.Series(ad_ref.obs[f"{metaref2add[i]}"].values, index=ad_ref.obs[f"{label_ref}"]).to_dict()
#         ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{label_key}"].map(d)
#         ad_spatial.obs[f"{metaref2add[i]}"] = ad_spatial.obs[f"{metaref2add[i]}"].astype("category")

#     # remove cells having a bad score
#     # nb_cells = ad_spatial.shape[0]
#     # ad_spatial = ad_spatial[ad_spatial.obs[f"{label_key}_score"] >= filter_under_score]
#     # filtered_cells = nb_cells - ad_spatial.shape[0]
#     # print("low assignment score filtering ", filtered_cells)


def sync_shape(
    sdata: sd.SpatialData,
    shape_key: str = None,
    table_key: str = 'table',
    region_key: str = 'region',

):
    """Synchronize shapes with table

    Parameters
    ----------
    sdata
        SpatialData object.
    shape_key
        key of shapes to synchronize

    """
    if shape_key is None:
        shape_key = sdata[table_key].uns["spatialdata_attrs"][region_key]

    _cells = sdata[table_key].obs[sdata[table_key].uns["spatialdata_attrs"]["instance_key"]].to_list()
    sdata[shape_key] = sdata[shape_key].loc[sdata[shape_key].index.isin(_cells)]


# def switch_region(
#    sdata: sd.SpatialData,
#    region: str = "cell_boundaries",  # could be cell_boundaries, nucleus_boundaries or cell_circles for xenium
# ):
#    """Swith to region of SpatialData object
#
#    Parameters
#    ----------
#    sdata
#        SpatialData object.
#    region
#        region (need to be a valid shape element).
#    """
#    # if i switch region
#    sdata['table'].obs.region = region
#    sdata['table'].uns["spatialdata_attrs"]["region"] = [region]
#    # i need to sync it to table
#    instance_key = sdata['table'].uns["spatialdata_attrs"]["instance_key"]
#    sdata.shapes[region] = sdata.shapes[region].loc[sdata['table'].obs[instance_key].tolist()]
