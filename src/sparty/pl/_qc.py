import seaborn as sns 
import matplotlib.pyplot as plt
import spatialdata as sd 
import anndata as ad
import pandas as pd
import scanpy as sc 

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
    if 'n_genes_by_counts' not in adata.obs.columns:
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



def top_genes_expressed(
    data: ad.AnnData | sd.SpatialData,
    table: str = 'table',
    n_top: int = 20,
    figsize: tuple = (15, 6),
):
    if isinstance(data, sd.SpatialData):
        adata = data[table].copy
    elif isinstance(data, ad.AnnData):
        adata = data.copy()
    else:
        Warning("Type unknown, please provide a anndata or a spatialdata object !")
        return
    
    X = adata.X

    if not isinstance(X, np.ndarray):
        X = X.toarray()

    cell_total = adata.obs['transcript_counts'].values
    valid_cells = cell_total > 0
    X = X[valid_cells, :]

    cell_total = cell_total[valid_cells]
    gene_mean = X.mean(axis=0)

    
    df = pd.DataFrame({
        'gene': adata.var_names,
        'mean_expr': gene_mean,
    })
    top_genes = df.nlargest(n_top, 'mean_expr')
    # df.sort_values('mean_expr', ascending=False)
    # Sélection des 20 gènes les plus exprimés selon la moyenne

    _, axes = plt.subplots(1, 2, figsize=figsize)
    sns.barplot(top_genes, x="mean_expr", y="gene", legend=False, palette='coolwarm', ax = axes[0])
    # axes[0].set_xlabel('Mean expression per cell (raw counts)')
    axes[0].set_title('Top 20 genes by mean expression')

    sc.pl.highest_expr_genes(adata, n_top=n_top,ax=axes[1])
    # axes[1].set_xlabel('Mean % of total expression per cell')
    axes[1].set_title('Top 20 genes by relative expression')
    plt.tight_layout()
    plt.show()
    