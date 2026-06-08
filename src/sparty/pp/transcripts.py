from spatialdata.transformations import get_transformation
import geopandas as gpd

from ..registry import TECHNO_REGISTRY
from ..constants import GENE_EXCLUDE_PATTERN #, XeniumKeys, MerscopeKeys, 
from ..pp.transformations import compute_bounds_dask

def subset_transcripts(
    sdata,
    genes: str | list = None,
    qv: int = 20,
    transcript_key: str= "transcripts",
    techno = "Xenium", # 'Xenium' or 'Merscope'
    only_in_cell: bool = True,
    only_outside: bool = False,
    gene_exclude_pattern = GENE_EXCLUDE_PATTERN,
    feature_key: str = 'feature_name',
    transform: bool = True,
    scale: str = False,
    copy: bool = True,
    return_gpd: bool = False,
):
    if techno not in TECHNO_REGISTRY:
        raise ValueError(f"Techno '{techno}' not supported. Available: {list(TECHNO_REGISTRY.keys())}")
    
    if type(genes) == str:
        genes = [genes]

    config = TECHNO_REGISTRY[techno]
    Keys = config["keys"]
    filter_fn = config["filter_fn"]

    ## WARNINGS FOR MERSCOPE TRANSCRIPT_KEY not always the same
    # df_transcripts = sdata[Keys.TRANSCRIPT_KEY].copy() if copy else sdata[Keys.TRANSCRIPT_KEY]
    df_transcripts = sdata[transcript_key].copy() if copy else sdata[transcript_key]
    
    if transform:
        df_transcripts = compute_bounds_dask(
            transcripts=df_transcripts, 
            transfo=get_transformation(df_transcripts), 
            scale=scale)

    df_transcripts = filter_fn(
            df=df_transcripts,
            genes=genes,
            qv=qv,
            only_in_cell=only_in_cell,
            only_outside=only_outside,
            gene_exclude_pattern=gene_exclude_pattern
        )
    
    df_transcripts = df_transcripts.compute() if hasattr(df_transcripts, "compute") else df_transcripts
    df_transcripts[Keys.FEATURE_KEY] = df_transcripts[Keys.FEATURE_KEY].cat.remove_unused_categories()
    
    if (return_gpd) and (not isinstance(df_transcripts, gpd.GeoDataFrame)):
        print("Create geopandas...")
        df_transcripts = gpd.GeoDataFrame(
            df_transcripts,
            geometry=gpd.points_from_xy(
                df_transcripts["x"],
                df_transcripts["y"]
            )
        )
        
        
    return df_transcripts
 


# def _subset_transcripts(
#     sdata,
#     genes: list,
#     qv: int = 20,
#     transcript_key: str= "transcript",
#     feature_key: str = 'feature_name',
#     gene_exclude_pattern = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
#     copy: bool = True,
# ):
#     if copy: 
#         df_transcripts = sdata[transcript_key].copy()
#     else:
#         df_transcripts = sdata[transcript_key]

#     df_transcripts = df_transcripts[(df_transcripts['qv'] >= qv) & 
#                                     (df_transcripts.is_gene) & 
#                                     (df_transcripts.cell_id != "UNASSIGNED") &
#                                     (df_transcripts[feature_key].isin(genes))
#                                     ].dropna(subset=[feature_key])
#     df_transcripts = df_transcripts[~(df_transcripts[feature_key].str.contains(gene_exclude_pattern, regex=True))].compute()
#     df_transcripts[feature_key] = df_transcripts[feature_key].cat.remove_unused_categories()
#     return df_transcripts