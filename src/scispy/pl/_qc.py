import seaborn as sns 
import matplotlib.pyplot as plt
import spatialdata as sd 
import anndata as ad

def plot_hist_QC(
    data: ad.AnnData | sd.SpatialData,
    table = 'table',
    figsize=(20, 4),
):  
    if type(data) == sd.SpatialData:
        adata = data[table].copy
    elif type(data) == ad.AnnData:
        adata = data.copy()
    else:
        Warning("Type unknown, please provide a anndata or a spatialdata object !")
        return
    if n_genes_by_counts not in adata.obs.columns:
        adata.obs['n_genes_by_counts'] = (adata.layers['counts'] > 0).sum(axis=0) 
    if 'nucleus_ration' not in adata.obs.columns:
        adata.obs['nucleus_ration'] = adata.obs['nucleus_area'] / adata.obs['cell_area']

    _, axs = plt.subplots(1, 4, figsize=figsize)
    axs[0].set_title("Total transcripts per cell")
    sns.histplot(
        adata.obs["transcript_counts"],
        kde=False,
        ax=axs[0],
    )

    axs[1].set_title("Unique transcripts per cell")
    sns.histplot(
        adata.obs["n_genes_by_counts"],
        kde=False,
        ax=axs[1],
    )

    axs[2].set_title("Area of segmented cells")
    sns.histplot(
        adata.obs["cell_area"],
        kde=False,
        ax=axs[2],
    )

    axs[3].set_title("Nucleus ratio")
    sns.histplot(
        adata.obs["nucleus_ration"],
        kde=False,
        ax=axs[3],
    )
    plt.show()


