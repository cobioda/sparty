import re
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def gene_heatmaps(
    data: ad.AnnData,
    x: str,
    y: str,
    genes: list | str ,
    isoform: bool = False,
    subset_obs: list = None,
    shared_scale: bool = False,
    layer: str = "counts",
    target_sum: int = 1000,
    cmap: str = "cividis",
    figsize_per_gene: tuple = (3, 4),
    save: str = None,
) -> None:
    if not isinstance(data, ad.AnnData):
        raise TypeError("`data` must be AnnData.")
    # if not isinstance(data, (ad.AnnData, pd.DataFrame)):
    #     raise TypeError("`data` must be AnnData or pd.DataFrame.")

    if isoform:
        if not isinstance(genes, str):
            raise ValueError("When isoform=True, `genes` must be a string prefix/regex.")
        # all_names = data.var_names if isinstance(data, ad.AnnData) else data.columns
        gene_list = [g for g in data.var_names if re.search(genes, g)]
        if not gene_list:
            raise ValueError(f"No genes matched pattern '{genes}'.")
        print(f"Isoforms matched for '{genes}': {gene_list}")
    else:
        gene_list = [genes] if isinstance(genes, str) else list(genes)
        missing = [g for g in gene_list if g not in data.var_names]
        if missing:
            raise ValueError(f"Genes not found in data: {missing}")

    # adata = data[:, gene_list].copy()
    adata = data.copy()
    adata.X = adata.layers[layer].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata = adata[:, gene_list]  

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    expr = pd.DataFrame(X, index=adata.obs.index, columns=adata.var_names)
    meta = adata.obs[[x, y]].copy()

    df = pd.concat([expr, meta], axis=1)

    if subset_obs is not None:
        df = df[df[y].isin(subset_obs)]
        df[y] = df[y].cat.remove_unused_categories() if hasattr(df[y], "cat") else df[y]

    matrices = {
        g: df.groupby([x, y])[g].mean().unstack()
        for g in gene_list
    }

    if shared_scale:
        all_vals = np.concatenate([m.values.flatten() for m in matrices.values()])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    else:
        vmin, vmax = None, None

    n = len(gene_list)
    _, axs = plt.subplots(
        1, n,
        sharey=True,
        figsize=(figsize_per_gene[0] * n, figsize_per_gene[1]),
    )
    if n == 1:
        axs = [axs]

    for i, (ax, g) in enumerate(zip(axs, gene_list)):
        is_last = (i == n -1)
        sns.heatmap(
            matrices[g].T,
            linewidths=0,
            cmap=cmap,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=is_last if shared_scale else True,
        )
        ax.set_title(g)

    # plt.tight_layout(w_pad=0.15)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight")
        print(f"Saved → {save}")
    plt.show()
  