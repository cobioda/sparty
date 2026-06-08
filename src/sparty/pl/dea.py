import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import anndata as ad
import matplotlib.pyplot as plt
from spatialdata import SpatialData
import spatialdata_plot
import adjustText as at
from matplotlib.transforms import Bbox
import decoupler as dc 
import matplotlib.gridspec as gridspec
import PyComplexHeatmap as pch
from scipy.cluster.hierarchy import linkage


sc.set_figure_params(vector_friendly=True, dpi=300, dpi_save=300) 
plt.rcParams.update(
    {'ps.fonttype':42,
    'ps.fonttype': 42, 
    'pdf.fonttype': 42, 
    'font.size': 10, 
    # 'font.family': 'Arial', 
    'mathtext.fontset': 'cm', 
    # 'mathtext.rm': 'Arial',
    'lines.linewidth': .2, 
    'xtick.top': False, 
    'ytick.right': False}
)


def barplotDE(
    adata: ad.AnnData,
    groupby: str = 'cell_type',
    splitby: str = 'condition',
    # y_key: str = 'log2FoldChange',
    palette: tuple | str = 'deep',
    key: str = 'results',
    # uns_key: str = 'pseudobulk',
    # title: str = None,
    padj: float = 0.05,
    groups: list | None = None,
    logFC: float = 0.5,
    figsize: tuple = (8,3),
    alpha: float = 0.5,
    join_by: str = '..',
    # save_format: str = "pdf",
    save: str | None = None,
    xticks_rotation: int = 90, 
    dpi: int = 300,
    ):
    if isinstance(adata, ad.AnnData):
        if key is None:
            raise ValueError("`key` must be provided when `adata` is an AnnData object.")
        res_de = adata.uns['sparty'][key].copy()
    elif isinstance(adata, pd.DataFrame):
        res_de = adata.copy()
    else:
        raise TypeError("`adata` must be either a pandas DataFrame or an AnnData object.")

    if groups:
        print("Filtrer groups...")
        res_de = res_de[res_de[groupby].isin(groups)]

    res_de["updown"] = np.where(
            (res_de["padj"] <= padj) & (res_de["log2FoldChange"] >= logFC), "Up",
            np.where((res_de["padj"] <= padj) & (res_de["log2FoldChange"] <= -logFC), "Down", "NS")
        )
    genes_DE_signif = res_de[res_de["updown"] != "NS"]
    # cell_types = genes_DE_signif[groupby].unique()
    # all_combinations = pd.MultiIndex.from_product([cell_types, ["Up", "Down"]], names=[groupby, "updown"])
    # df_m = genes_DE_signif.groupby([groupby, "updown"]).size().reindex(all_combinations, fill_value=0).reset_index(name="value") 
     
    # up_df = df_m[df_m["updown"] == "Up"].reset_index(drop=True)
    # down_df = df_m[df_m["updown"] == "Down"].reset_index(drop=True)
    
    if len(genes_DE_signif[splitby].unique()) > 1 : 
        print("More than one pairwise condition.")
    for cond in genes_DE_signif[splitby].unique():
        print(cond)
        sub_cond = genes_DE_signif[genes_DE_signif[splitby] == cond]
            
        cell_types = sub_cond[groupby].unique()
        all_combinations = pd.MultiIndex.from_product([cell_types, ["Up", "Down"]], names=[groupby, "updown"])
        df_m = sub_cond.groupby([groupby, "updown"]).size().reindex(all_combinations, fill_value=0).reset_index(name="value") 
            
        up_df = df_m[df_m["updown"] == "Up"].reset_index(drop=True)
        down_df = df_m[df_m["updown"] == "Down"].reset_index(drop=True)
    
        plt.figure(figsize=figsize)
        sns.barplot(data=up_df, x=groupby, y="value", 
                    hue=groupby, dodge="auto", palette=palette)
        sns.barplot(data=down_df, x=groupby, y=-down_df["value"], 
                    hue=groupby, dodge="auto", palette=palette, alpha=alpha)
            
        # Add labels
        for i, row in up_df.iterrows():
            if row["value"] != 0:
                plt.text(i, row["value"]  , str(row["value"]), ha='center', va='bottom', fontsize=10)
            
        for i, row in down_df.iterrows():
            if row["value"] != 0:
                plt.text(i, -row["value"] , str(row["value"]), ha='center', va='top', fontsize=10)
            
        # Customize plot
        plt.axhline(0, color="grey", linestyle="--")
        plt.xticks(rotation=xticks_rotation, fontsize=12) # ha='center',
        plt.xlabel("Cell types")
        plt.ylabel("Number of genes DE")
        cond1, cond2 = cond.split(join_by)
        # if not title:
        title = f'Number of genes DE up and down in {cond1} versus {cond2} for each cell type'
        plt.title(title, fontsize=12) #, fontweight='bold')
        if isinstance(save, str):
            plt.savefig(save, bbox_inches="tight", dpi=dpi)
        plt.show()
        
    

def stripPlotDE(
    adata: ad.AnnData | pd.DataFrame, 
    x_key: str = 'cell_type',
    y_key: str = 'log2FoldChange',
    splitby = "condition",
    palette: tuple | str = "deep",
    key: str = 'results',
    groups: list | None = None,
    padj: float = 0.05,
    logFC: float = 0.5,
    baseMean: float = 50.0,
    figsize: tuple = (8,3),
    top: int = 5,
    # order: list | None = None,
    join_by: str = '..',
    # save_format: str = 'pdf',
    save: str | None = None,
    xticks_rotation: int = 90, 
    title: str = None,
    dpi: int = 300
):
    """Plot DEG dataframe from pseudobulk analysis

    Parameters
    ----------
    adata
        anndata object
    x_key
        x key
    y_key
        y key
    key
        key in adata.uns['sparty'] storing the results to plot
    padj
        p adjusted to be significant
    log2FoldChange
        log2FoldChange to be significant
    figsize
        figure size
    save
        wether or not to save the figure
    save_format
        pdf or png
    """
    if isinstance(adata, ad.AnnData):
        df = adata.uns['sparty'][key].copy()
    else:
        df = adata.copy()

    df["significative"] = (
        (df["padj"] <= padj) &
        (df["log2FoldChange"].abs() >= logFC) &
        (df["baseMean"] >= baseMean)
    )

    for cond in df[splitby].unique():

        sub_df = df[df[splitby] == cond].copy()

        if groups is not None:
            sub_df = sub_df[sub_df[x_key].isin(groups)]
            order = list(groups)
        else:
            order = list(sub_df[x_key].unique())

        # enforce order
        sub_df[x_key] = pd.Categorical(sub_df[x_key], categories=order, ordered=True)
        tmp = sub_df[sub_df['significative']].reset_index(drop=True)
        
        _, ax = plt.subplots(figsize=figsize)

        sns.stripplot(
            data=sub_df[~sub_df["significative"]],
            x=x_key, y=y_key,
            order=order,
            color="gray",
            # orient='v',
            alpha=0.5,
            size=2,
            jitter=0.4,
            linewidth=0, 
            ax=ax
        )

        sns.stripplot(
            data=sub_df[sub_df["significative"]],
            x=x_key, y=y_key,
            order=order,
            # orient='v',
            hue=x_key,
            palette=palette,
            alpha=0.8,
            size=5,
            jitter=0.4,
            linewidth=1, 
            ax=ax
        )
        tmp["rank"] = tmp.groupby(x_key)["log2FoldChange"].rank(method="first", ascending=False)
        tmp["top_bottom"] = tmp.groupby(x_key).apply(
            lambda g: pd.Series(
                np.where(
                    g["rank"] <= min(top, len(g)), "Top",
                    np.where(g["rank"] > len(g) - min(top, len(g)), "Bottom", None)
                ), index=g.index
            )
        ).reset_index(level=0, drop=True)

        texts = []
        # Build a mapping from category label → integer x-position
        x_positions = {cat: i for i, cat in enumerate(order)}

        to_label = tmp[tmp["top_bottom"].notna()].copy()

        for _, row in to_label.iterrows():
            x_pos = x_positions.get(row[x_key])
            if x_pos is None:
                continue
            # Use a tiny deterministic offset so labels don't all stack at x_pos
            # adjust_text will handle the fine-grained repositioning
            texts.append(ax.text(
                x_pos, row[y_key], row["gene"],
                ha="center", va="bottom", color="black", size="x-small"
            ))
        # for collection, (_, group_data) in zip(
        #     ax.collections[-len(tmp[x_key].unique()):], tmp.groupby(x_key)):
        #     for i, (x, y) in enumerate(collection.get_offsets()):
        #         if group_data.iloc[i]['top_bottom'] is not None:
        #             texts.append(ax.text(
        #                 x, y, group_data.iloc[i]['gene'],
        #                 ha='center', va='bottom', color='black', size='x-small'
        #             ))

        at.adjust_text(
            texts,
            ax=ax,
            expand=(2,2),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
            )
        )
        
        if title:
            ax.set_title(title)
        else:
            cond1, cond2 = cond.split(join_by)
            ax.set_title(f"{cond1} vs {cond2}")

        ax.tick_params(axis='x', rotation=xticks_rotation)

        if save:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()

def maplot(
    adata: ad.AnnData,
    genes: list | None = None,
    key: str = 'results',
    thr_stat: float =0.5,
    thr_sign: float= 0.05,
    top: int = 10,
    x: str = "baseMean",
    y: str = "log2FoldChange",
    color_pos: str = "#D62728",
    color_neg: str = "#1F77B4",
    color_null: str = "gray",
    figsize: tuple = (8,6),
    fig_title: str = "MA plot",
):  
    if 'sparty' not in adata.uns.keys():
        print('Run DEA before plotting...')
        return
    df = adata.uns['sparty'][key].copy()
    df["log2_mean"] = np.log2(df[x])

    up_msk = (df[y] >= thr_stat) & (df["padj"] <= thr_sign) 
    dw_msk = (df[y] <= -thr_stat) & (df["padj"] <= thr_sign)
    not_sign = ~(up_msk | dw_msk)
    
    if type(genes) == list:
        signs = df[df['gene'].isin(genes)]
    else:
        signs = df[up_msk | dw_msk].sort_values("padj", ascending=False)
        signs = signs.iloc[:top]

    plt.figure(figsize=figsize)
    sc_mid = plt.scatter(df.loc[not_sign, "log2_mean"], df.loc[not_sign, y], color=color_null, alpha=0.6, s=20, label=f"Non-significant ({not_sign.sum()})")
    sc_up = plt.scatter(df.loc[up_msk, "log2_mean"], df.loc[up_msk, y], color=color_pos, alpha=0.7, s=20, label=f"Up ({up_msk.sum()})")
    sc_down = plt.scatter(df.loc[dw_msk, "log2_mean"], df.loc[dw_msk, y], color=color_neg, alpha=0.7, s=20, label=f"Down ({dw_msk.sum()})")

    ymax =  df[y].abs().max() + 0.5
    xmax = df["log2_mean"].max() + 0.5

    plt.ylim(-ymax, ymax)
    plt.xlim(0, xmax)
    plt.axhline(y=thr_stat, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-thr_stat, color='black', linestyle='--', linewidth=1)

    texts = []
    for x, y, s in zip(signs["log2_mean"], signs[y], signs['gene'], strict=False):
        texts.append(
            plt.text(x, y, s, 
                    bbox=dict(boxstyle="round",
                            facecolor='white', edgecolor='black',             
                    ),
                    fontweight = 'bold',
                    size='small'))
    if len(texts) > 0:
        at.adjust_text(
            texts, expand=(4, 4),
            arrowprops={"arrowstyle": "->", "color": "black"})
    plt.legend(
        handles=[sc_up, sc_mid, sc_down],
        loc='best', 
        bbox_to_anchor=(0.81, 0., 0.5, 0.5))
    plt.grid(True)
    plt.title(fig_title)
    plt.xlabel("Log2 mean expression")
    plt.ylabel('Log2 fold change')
    plt.show()

# def extract_sample(name, samples):
#     for s in samples:
#         if name.startswith(s):
#             return s
#     return None  # si rien ne matche

# def extract_sample(name: str, samples: list[str]) -> str | None:
#     """Returns the first prefix of `samples` that matches `name`."""
#     return next((s for s in samples if name.startswith(s)), None)
    

# def filter_genes(df: pd.DataFrame, thres: dict) -> pd.DataFrame:
#     """Filters genes based on the padj, log2FoldChange, baseMean, and pct thresholds."""
#     masks = [
#         df["padj"] <= thres["padj"],
#         df["log2FoldChange"].abs() >= thres["log2FoldChange"],
#         df["baseMean"] >= thres["baseMean"],
#         (df["pct_1"] >= thres["pct"]) | (df["pct_2"] >= thres["pct"]),
#     ]
#     return df[np.logical_and.reduce(masks)]
    
def filter_genes(df: pd.DataFrame, thres: dict) -> pd.DataFrame:
    padj_mask = df["padj"] <= thres["padj"]
    lfc_mask = np.abs(df["log2FoldChange"]) >= thres["log2FoldChange"]
    base_mask = df["baseMean"] >= thres["baseMean"]
    pct_mask = (df["pct_1"] >= thres["pct"]) | (df["pct_2"] >= thres["pct"])

    mask = padj_mask & lfc_mask & base_mask & pct_mask
    return df[mask]
    # return df.loc[mask, "gene"].unique() ==> RETURN ONLY gene NAME !!!


def filter_genes_query(df: pd.DataFrame, query: str | None) -> pd.DataFrame:
    """Applies a Pandas query if one is provided; otherwise, returns the dataframe unchanged."""
    return df.query(query) if query else df

def _plot_heatmap(
    df_plot: pd.DataFrame,
    df_colors: pd.DataFrame,
    colors: dict,
    show_rownames: bool = True,
    col_dendrogram: bool = True,
    row_dendrogram: bool = True,
    row_cluster: bool = True,
    col_cluster: bool = True,
    cmap: str = "bwr",
    linkage = None,
) -> pch.ClusterMapPlotter:
    """Creates and returns a PyComplexHeatmap ClusterMapPlotter."""
    valid_cols = [col for col in df_colors.columns if col in colors]
    
    if valid_cols:
        ann_kwargs = {
            col: pch.anno_simple(df_colors[col], colors=colors[col], add_text=False, legend=True)
            for col in valid_cols
        }
        col_ha = pch.HeatmapAnnotation(
            **ann_kwargs,
            legend=True,
            legend_gap=1, #5
            hgap=0.5,
            axis=1,
        )
    else:
        col_ha = None

    if linkage is not None:
        row_dendrogram_kws={'linkage': linkage}
    else:
        row_dendrogram_kws=None      

    return pch.ClusterMapPlotter(
        data=df_plot,
        top_annotation=col_ha,
        label='values',
        col_dendrogram=col_dendrogram,
        row_dendrogram=row_dendrogram,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        show_rownames=show_rownames,
        show_colnames=True,
        row_dendrogram_kws=row_dendrogram_kws,
        verbose=0,
        legend_hpad=2,
        legend_vpad=25, # 10??
        cmap=cmap,  
        plot =False,
        plot_legend=False,
        center=0, 
        xticklabels_kws={'labelrotation': 90},
        # col_split=adata.obs[condition],
        # col_split_gap=0.8,
    ) 


def _resolve_groups_col(groups_col: list) -> tuple[str | None, str, list]:
    """
    Returns (groups_key, condition, col_to_add) based on the size of groups_col.
    Raise a ValueError if groups_col has fewer than 2 elements.
    """
    match len(groups_col):
        case 1:
            condition  = groups_col[0]
            groups_key = None
            col_to_add = [None, condition]          # replicate will be inserted after
        case 2:
            groups_key, condition = groups_col
            col_to_add = [None, groups_key, condition]
        case _:
            raise ValueError("'groups_col' must contain 1 or 2 elements.")
    return groups_key, condition, col_to_add


def _resolve_params(adata, replicate, groups_col):
    """Merges the explicit parameters with those stored in adata.uns."""
    params     = adata.uns["sparty"]["params"]
    replicate  = replicate  or params["replicate"]
    groups_col = groups_col or params["groups_col"]

    if not isinstance(groups_col, (list, tuple)):
        raise ValueError("'groups_col' must be a list or tuple.")

    groups_key, condition, col_to_add = _resolve_groups_col(list(groups_col))
    col_to_add[0] = replicate          # insert the resolved replicate
    return replicate, groups_key, condition, col_to_add


def _build_heatmap_data(
    adata_hm: ad.AnnData,
    colors: dict,
    replicate: str,
    condition: str,
    paired: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Constructs df_plot and df_colors from a subselected AnnData.
    Also returns the effective replicate key (original or paired).
    """
    if paired:
        new_rep = f"{replicate}_{condition}"
        adata_hm.obs[new_rep] = (
            adata_hm.obs[replicate].astype(str) + "_" +
            adata_hm.obs[condition].astype(str)
        )
    else:
        new_rep = replicate

    df_plot = pd.DataFrame(
        adata_hm.X.T,
        index=adata_hm.var_names,
        columns=adata_hm.obs[new_rep],
    )
    df_colors = pd.DataFrame(
        adata_hm.obs[list(colors.keys())].values,
        index=adata_hm.obs[new_rep].values,
        columns=list(colors.keys()),
    )
    return df_plot, df_colors, new_rep


def _plot_volcano(
    ax, 
    sub_all: pd.DataFrame, 
    genes_sig, 
    genes: list | None = None, 
    x: str = "log2FoldChange", 
    y: str = "padj",
    top_volcano: int = 10,
    thr_stat: float = 1,
    thr_sign: float= 0.05,
) -> None:
    """Plots the volcano plot, with optional gene highlighting."""
    if genes is not None:
        dc.pl.volcano(sub_all, x=x, y=y, ax=ax,
                      top=1, thr_stat=thr_stat, thr_sign=thr_sign)

        highlight = sub_all.loc[genes_sig]

        texts = []
        for gene, row in highlight.iterrows():
            texts.append(
                ax.text(
                    row[x], -np.log10(row[y]), gene, 
                    ha="center", va="bottom", 
                    color="black", fontsize=10
                    # bbox=dict(boxstyle="round",
                    #         facecolor='white', edgecolor='black',             
                    # ), fontweight = 'bold',# size='small',
                    )
            )

        if len(texts) > 0:
            at.adjust_text(texts,ax=ax,
                           expand=(2,2),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                    )
                )
    else:
        dc.pl.volcano(
            sub_all, 
            x=x, y=y,
            ax=ax,
            top=top_volcano, 
            thr_stat=thr_stat, 
            thr_sign=thr_sign
        )
    ax.set_title("Volcano")



# def _plot_volcano(
#     ax, 
#     sub_all: pd.DataFrame, 
#     genes_sig, 
#     # genes, 
#     top_volcano: int,
#     thr_stat: float = 1,
#     thr_sign: float= 0.05,
#     ) -> None:
#     """Plots the volcano plot, with optional gene highlighting."""
#     if genes_sig is not None:
#         dc.pl.volcano(sub_all, x="log2FoldChange", y="padj", ax=ax,
#                       top=0, thr_stat=thr_stat, thr_sign=thr_sign)

#         highlight = sub_all.loc[genes_sig]
#         ax.scatter(highlight["log2FoldChange"],
#                    -np.log10(highlight["padj"]),
#                    color="red", zorder=10)
#         for gene, row in highlight.iterrows():
#             ax.text(row["log2FoldChange"], -np.log10(row["padj"]), gene, fontsize=8)
#     else:
#         dc.pl.volcano(
#             sub_all, 
#             x="log2FoldChange", 
#             y="padj", 
#             ax=ax,
#             top=top_volcano, 
#             thr_stat=thr_stat, 
#             thr_sign=thr_sign
#         )
#     ax.set_title("Volcano")


def plot_DE(
    adata: ad.AnnData,
    colors: dict,
    cell_type: str = "cell_type",
    groups_col: list | None = None,
    replicate: str | None = None,
    groups: list | None = None,
    query: str | None = None,
    cmap: str = "bwr",
    top_volcano: int = 10,
    genes: list | None = None,
    figsize: tuple = (20, 10),
    replicate_order: list | None = None, # a ajouter pour controler l'ordre des échantillons dans la heatmap et pas de dendogrammme
    join_by: str = "..",
    paired: bool = False,
    only_heatmap: bool = False,
    only_volcano: bool = False,
    save: str | None = None,
    nb_to_show: int = 50,
    sort_by: list | None = None,
    method: str = 'ward', # average
    metric: str = 'euclidean', #correlation
    thr_stat: float = 1,
    thr_sign: float= 0.05,
    dpi: int = 300,
) -> None:
    """
    Plot a heatmap and/or volcano plot of differentially expressed genes
    for each cell type present in adata.uns["sparty"].
    """
    df = adata.uns['sparty']['results'].copy()
    mtx = adata.uns["sparty"]["matrice"].copy()

    replicate, groups_key, condition, col_to_add = _resolve_params(
        adata, replicate, groups_col
    )

    df_sig = filter_genes_query(df, query)

    if groups:
        df = df[df[cell_type].isin(groups)]
        df_sig = df_sig[df_sig[cell_type].isin(groups)]

    for cell, sub_sig in df_sig.groupby(cell_type):
        if genes is not None:
            genes_sig = np.intersect1d(genes, df[df[cell_type] == cell]["gene"].unique())
        else:
            genes_sig = sub_sig["gene"].unique()

        if len(genes_sig) < 2:
            continue
    
        print(f"{cell} -> {len(genes_sig)} genes DE")

        column_ct = mtx.columns.str.contains(f'{join_by}{cell}{join_by}', regex=False)
        adata_tmp = ad.AnnData(mtx.loc[:, column_ct].T)
        adata_tmp.obs[col_to_add] =  adata_tmp.obs.index.str.split(join_by, regex=False).tolist() 
  
        for col in colors:
            if (col not in adata_tmp.obs.columns) and (col in adata.obs.columns):
                mapping = adata.obs[[replicate, col]].drop_duplicates().set_index(replicate)[col]
                adata_tmp.obs[col] = adata_tmp.obs[replicate].map(mapping)

        sc.pp.normalize_total(adata_tmp)
        sc.pp.log1p(adata_tmp)
        sc.pp.scale(adata_tmp, max_value=10)

        adata_hm = adata_tmp[:, genes_sig]
        df_plot, df_colors, _ = _build_heatmap_data(
            adata_hm, colors, replicate, condition, paired
        )

        fig = plt.figure(figsize=figsize)
        show_both = not only_heatmap and not only_volcano
        # gs = gridspec.GridSpec(1, 1 + show_both, wspace=0.5) #, width_ratios=[1, 1])
        if show_both:
            gs = gridspec.GridSpec(1, 2, wspace=0.6, width_ratios=[1.5, 1])
        else:
            gs = gridspec.GridSpec(1, 1)

        i = 0

        # ===== HEATMAP =====
        if not only_volcano:
            if replicate_order is not None:
                df_colors = df_colors[df_colors[replicate].isin(replicate_order)]
                df_colors[replicate] = pd.Categorical(
                    df_colors[replicate],
                    categories=replicate_order,
                    ordered=True
                )
                if sort_by:
                    df_colors = df_colors.sort_values(sort_by)
                else:
                    # df_colors = df_colors.sort_values(replicate)
                    df_colors = df_colors.sort_values([condition, replicate])

                df_plot = df_plot[df_colors.index]
                col_cluster = col_dendrogram = False 
                
                # ordered = [c for c in replicate_order if c in df_plot.columns]
                # df_plot = df_plot[ordered]
                # df_colors = df_colors.loc[ordered]
            else:
                col_cluster = col_dendrogram = True

            if (method != 'ward') & (metric != 'euclidean'):
                Z = linkage(df_plot.values, method=method, metric=metric)    
            else:
                Z = None
            ax = fig.add_subplot(gs[i])
            cluster = _plot_heatmap(
                df_plot, df_colors, colors,
                show_rownames=(len(genes_sig) <= nb_to_show),
                cmap=cmap,
                linkage=Z,
                col_cluster=col_cluster,
                col_dendrogram=col_dendrogram,
            )
            cluster.plot(ax=ax, subplot_spec=gs[i])
            cluster.plot_legends(ax=ax)
            ax.set_title("Heatmap")
            i += 1

        # ===== VOLCANO =====
        if not only_heatmap:
            ax = fig.add_subplot(gs[i])
            sub_all = df[df[cell_type] == cell].set_index("gene")
            _plot_volcano(
                ax=ax, 
                sub_all=sub_all, 
                genes_sig=genes_sig,
                genes=genes, 
                top_volcano=top_volcano, 
                thr_stat=thr_stat,
                thr_sign=thr_sign)

        fig.suptitle(cell, fontsize=18)
        if isinstance(save, str):
            plt.savefig(save, bbox_inches="tight", dpi=dpi)
        plt.show()

        


# def plot_DE(
#     adata: ad.AnnData,
#     colors: dict,
#     cell_type: str = "cell_type",
#     groups_col: list | None = None,
#     replicate: str | None = None,
#     groups: list | None = None,
#     query: str | None = None,
#     cmap: str = "bwr",
#     top_volcano: int = 10,
#     genes: list | None = None,
#     figsize: tuple = (20, 10),
#     replicate_order: list | None = None, # a ajouter pour controler l'ordre des échantillons dans la heatmap et pas de dendogrammme
#     join_by: str = "..",
#     paired: bool = False,
#     only_heatmap: bool = False,
#     only_volcano: bool = False,
# ) -> None:
#     """
#     Plot a heatmap and/or volcano plot of differentially expressed genes
#     for each cell type present in adata.uns['sparty'].
#     """
#     df = adata.uns['sparty']['results'].copy()
#     mtx = adata.uns["sparty"]["matrice"].copy()
#     params = adata.uns['sparty']['params']

#     if not replicate:
#         replicate = params['replicate']

#     if not groups_col:
#         groups_col = params['groups_col']

#     if not isinstance(groups_col, (list, tuple)):
#         raise ValueError("'groups_col' must be a list or tuple.")

#     if len(groups_col) not in (1, 2):
#         raise ValueError("Expected 'groups_col' to contain 1 or 2 elements.")

#     if len(groups_col) == 2:
#         groups_key, condition = groups_col
#         col_to_add = [replicate, groups_key, condition]
#     else:
#         condition = groups_col[0]
#         groups_key = None
#         col_to_add = [replicate, condition]


#     if not replicate:
#         replicate = adata.uns['sparty']['params']['replicate']
#     if not groups_key:
#         groups_key, condition = adata.uns['sparty']['params']['groups_col']

#     if isinstance(groups_col, list) and len(groups_col) == 2:
#             groups_key, condition = groups_col
#             col_to_add = [replicate, groups_key, condition]
#     elif isinstance(groups_col, list) and len(groups_col) == 1:
#         condition = groups_col[0]
#         col_to_add = [replicate, condition]
#     else:
#         raise ValueError("Expected exactly one or two grouping columns in 'groups_col'.")
    
#     # samples = adata.obs[replicate].unique().tolist()

#     df_sig = filter_genes_query(df, query)

#     if groups:
#         df = df[df[cell_type].isin(groups)]
#         df_sig = df_sig[df_sig[cell_type].isin(groups)]

#     for cell, sub_sig in df_sig.groupby(cell_type):
#         if genes is not None:
#             genes_sig = np.intersect1d(genes, df[df[cell_type] == cell]["gene"].unique())
#         else:
#             genes_sig = sub_sig["gene"].unique()

#         if len(genes_sig) < 2:
#             continue
    
#         print(f"{cell} -> {len(genes_sig)} genes DE")
#         column_ct = mtx.columns.str.contains(f'{join_by}{cell}{join_by}', regex=False)
#         adata_tmp = ad.AnnData(mtx.loc[:, column_ct].T)
#         adata_tmp.obs[col_to_add] =  adata_tmp.obs.index.str.split(join_by, regex=False).tolist() 
  
#         for col in colors.keys():
#             if (col not in adata_tmp.obs.columns) and (col in adata.obs.columns):
#                 mapping = adata.obs[[replicate, col]].drop_duplicates().set_index(replicate)[col]
#                 adata_tmp.obs[col] = adata_tmp.obs[replicate].map(mapping)

#         sc.pp.normalize_total(adata_tmp)
#         sc.pp.log1p(adata_tmp)
#         sc.pp.scale(adata_tmp, max_value=10)

#         adata_hm = adata_tmp[:, genes_sig]

#         if paired:
#             new_replicate = f'{replicate}_{condition}'
#             adata_hm.obs[new_replicate] = adata_hm.obs[replicate].astype(str) + "_" + adata_hm.obs[condition].astype(str)
#         else:
#             new_replicate = replicate

#         df_plot = pd.DataFrame(
#             adata_hm.X.T,
#             index=adata_hm.var_names,
#             columns=adata_hm.obs[new_replicate]
#         )

#         df_colors = pd.DataFrame(
#             adata_hm.obs[list(colors.keys())].values, 
#             index=adata_hm.obs[new_replicate],
#             columns=list(colors.keys())
#         )
#         show_rownames = len(genes_sig) <= 50

#         fig = plt.figure(figsize=figsize)

#         ncols = 1
#         if not only_heatmap and not only_volcano:
#             ncols = 2

#         gs = gridspec.GridSpec(1, ncols, wspace=0.5) #, width_ratios=[1, 1])
#         # gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.5, width_ratios=[1, 1])

#         i = 0

#         # ===== HEATMAP =====
#         if not only_volcano:
#             if replicate_order is not None:
#                 ordered_cols = [c for c in replicate_order if c in df_plot.columns]
#                 df_plot = df_plot[ordered_cols]
#                 df_colors = df_colors.loc[ordered_cols]

#                 col_cluster_flag = False
#                 col_dendrogram_flag = False
#             else:
#                 col_cluster_flag = True
#                 col_dendrogram_flag = True

#             ax = fig.add_subplot(gs[i])

#             cluster = plot_heatmap(
#                 df_plot,
#                 df_colors,
#                 colors,
#                 show_rownames=show_rownames,
#                 cmap=cmap,
#                 col_cluster=col_cluster_flag,
#                 col_dendrogram=col_dendrogram_flag,
#             )

#             cluster.plot(ax=ax, subplot_spec=gs[i])
#             cluster.plot_legends(ax=ax)
#             ax.set_title("Heatmap")

#             i += 1

#         # ===== VOLCANO =====
#         if not only_heatmap:
#             ax = fig.add_subplot(gs[i])
#             sub_all = df[df[cell_type] == cell].set_index("gene")
            
#             if genes is not None:
#                 sub_all["highlight"] = sub_all.index.isin(genes_sig)

#                 dc.pl.volcano(
#                     sub_all,
#                     x="log2FoldChange",
#                     y="padj",
#                     ax=ax,
#                     top=0,  # désactive top
#                     thr_stat=0.5,
#                     thr_sign=0.05,
#                 )

#                 highlight_df = sub_all.loc[genes_sig]
#                 ax.scatter(
#                     highlight_df["log2FoldChange"],
#                     -np.log10(highlight_df["padj"]),
#                     color="red",
#                     zorder=10
#                 )

#                 for g, row in highlight_df.iterrows():
#                     ax.text(
#                         row["log2FoldChange"],
#                         -np.log10(row["padj"]),
#                         g,
#                         fontsize=8
#                     )
#             else:
#                 dc.pl.volcano(
#                     sub_all,
#                     x="log2FoldChange",
#                     y="padj",
#                     ax=ax,
#                     top=top_volcano,
#                     thr_stat=0.5,
#                     thr_sign=0.05,
#                 )
#             ax.set_title("Volcano")

#         fig.suptitle(cell, fontsize=18)
#         plt.show()

        
# def plot_DE(
#     adata: ad.AnnData,
#     colors: dict,
#     signs_thres: dict = {
#         'padj': 0.05,
#         'log2FoldChange': 0.5,
#         'baseMean': 50.0,
#         'pct': 0.0
#     },
#     groups_col: str | None = None,
#     replicate: str | None = None,
#     cell_type: str = 'cell_type',
#     groups: list | None = None,
#     # fill_na: str = 'grey',
#     # condition = 'condition',
#     cmap: str ='bwr',
#     top_volcano: int = 10,
#     figsize: tuple =  (20, 10),
#     join_by: str = "..",
# ):
#     df = adata.uns['sparty']['results'].copy()
#     mtx = adata.uns["sparty"]["matrice"].copy()
    
#     if not replicate:
#         replicate = adata.uns['sparty']['params']['replicate']
#     if not groups_col:
#         groups_key, condition = adata.uns['sparty']['params']['groups_col']
#     if not groups:
#         groups = df[cell_type].unique().tolist()

#     samples = adata.obs[replicate].unique().to_list()

#     for cell in groups:
#         print(cell)
#         sub_df = df[df[cell_type] == cell]
#         sub_df.index = sub_df['gene']
#         signs = filter_genes(sub_df, signs_thres)

#         if len(signs) > 1:
#             sub_mtx = mtx.loc[:, mtx.columns.str.contains(f'{join_by}{cell}{join_by}', regex=False)].T

#             adata = ad.AnnData(sub_mtx)
#             adata.obs["sample"] = adata.obs.index.map(lambda x: extract_sample(x, samples))
#             adata.obs["groups"] = adata.obs.apply(
#                 lambda row: row.name.replace(f"{row['sample']}{join_by}", ""),
#                 axis=1
#             )
#             adata.obs[[groups_key, condition]] = (
#                 adata.obs["groups"].str.split(join_by, expand=True, regex=False)
#             )
         
#             sc.pp.normalize_total(adata)
#             sc.pp.log1p(adata)
#             sc.pp.scale(adata, max_value=10)
#             adata = adata[:, signs]

#             df_plot = pd.DataFrame(
#                 adata.X.T, 
#                 index=adata.var_names, 
#                 columns=adata.obs['sample']) 
#             df_plot = df_plot.apply(pd.to_numeric, errors="coerce")

#             df_colors = pd.DataFrame(adata.obs[list(colors.keys())].values, 
#                         index=adata.obs['sample'],
#                         columns=list(colors.keys())) 
            
#             if len(signs) <= 50:
#                 show_rownames = True
#             else:
#                 show_rownames = False

#             # PLOT
#             fig = plt.figure(figsize=figsize)
#             gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.5, width_ratios=[1, 1])

#             ax1 = fig.add_subplot(gs[0])
#             cluster = plot_heatmap(
#                     df_plot,
#                     df_colors,
#                     colors,
#                     show_rownames=show_rownames,
#                     cmap=cmap
#                 )
#             cluster.plot(ax=ax1, subplot_spec = gs[0])
#             cluster.plot_legends(ax=ax1)
#             ax1.set_title('Heatmap')

#             ax2 = fig.add_subplot(gs[1])
#             dc.pl.volcano(
#                 sub_df, 
#                 # figsize=(5,5),
#                 x="log2FoldChange", 
#                 y="padj", 
#                 ax=ax2,
#                 thr_stat=signs_thres['log2FoldChange'], 
#                 thr_sign=signs_thres['padj'], 
#                 top=top_volcano
#             )
#             ax2.set_title('Volcano plot')

#             fig.suptitle(cell, fontsize=18) #, y=0.99)
#             plt.show()


# def heatmap_volcano(
#     adata,
#     sub_cell,
#     signs,
#     colors,
#     title,
#     thr_stat=0.5,
#     thr_sign = 0.05,
#     top=5,
#     figsize=(20, 10),
# ):
#     if len(signs) >1 : 
#         row_dendrogram =True
#         col_dendrogram=True
#         row_cluster=True
#         col_cluster=True
#     else:
#         row_dendrogram=False
#         col_dendrogram=False
#         row_cluster=False
#         col_cluster=False

#     df_test = pd.DataFrame(adata[:,signs].X.T, index=adata[:,signs].var_names, columns=adata.obs_names) 
#     df_test = df_test.apply(pd.to_numeric, errors="coerce")

#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.4, width_ratios=[1.2, 0.9])

#     ax1 = fig.add_subplot(gs[0])
#     col_ha = pch.HeatmapAnnotation(df=adata.obs, 
#                                 colors=colors,
#                                 legend=True,
#                                 legend_gap=5,
#                                 hgap=0.5,
#                                 axis=1)
#     cluster = pch.ClusterMapPlotter(data=df_test,
#                                 top_annotation=col_ha,
#                                 # col_split=adata.obs[condition],
#                                 # col_split_gap=0.8,
#                                 label='values',
#                                 col_dendrogram=col_dendrogram,
#                                 row_dendrogram=row_dendrogram,
#                                 row_cluster=row_cluster,
#                                 col_cluster=col_cluster,
#                                 show_rownames=True,
#                                 show_colnames=True,
#                                 verbose=0,
#                                 legend_gap=5, 
#                                 cmap="bwr",  
#                                 plot =False,
#                                 plot_legend=False,
#                                 center=0, 
#                                 xticklabels_kws={'labelrotation':-90}) 
#     cluster.plot(ax=ax1, subplot_spec = gs[0])
#     cluster.plot_legends(ax=ax1)
#     ax1.set_title('Heatmap')

#     ax2 = fig.add_subplot(gs[1])
#     dc.pl.volcano(
#         sub_cell, 
#         # figsize=(5,5),
#         x="log2FoldChange", 
#         y="padj", 
#         ax=ax2,
#         thr_stat=thr_stat, 
#         thr_sign=thr_sign, 
#         top=top
#     )
#     ax2.set_title('Volcano plot')

#     fig.suptitle(title, fontsize=18) #, y=0.99)
#     plt.show()

 # plots: bool = False,
 # save: bool = False,
# save_prefix: str = "decoupler",
# figsize: tuple = (8,3),
#     if save is True:
# #         results_df.to_csv(save_prefix + "_" + ct + ".csv")
# #         fig.savefig(save_prefix + "_" + ct + ".pdf", bbox_inches="tight")

# def plot_DE(
#     adata: ad.AnnData,
#     colors: dict,
#     # replicate = 'condition'
#     condition = 'condition',
#     cell_type: str = 'cell_type',
#     top_volcano: int = 5,
#     thr_stat: float = 0.5, 
#     thr_sign: float = 0.05,
#     min_pct: float = 0.3,
#     min_base_mean: float = 50.0,
#     fill_na: str = 'grey',
#     cmap: str ='bwr',
#     # col_cluster: bool = False,
# ):
#     """Heatmap and volcano plot from pseudobulk analysis

#     Parameters
#     ----------
#     adata
#         anndata object
#     colors
#         dict of colors for each metadata to plot in the heatmap
#     condition
#         column that refers to the condition comparison. Default, set to condition
#     cell_type
#         column that refers to the celltype column. Default, set to cell_type  
#     key
#         key in adata.uns['sparty'] storing the results to plot
#     thr_sign
#         p adjusted to be significant
#     thr_stat
#         log2FoldChange to be significant
#     min_pct
#         min pct.1 or pct.2 to be significant
#     fill_na
#         if colors not provide for one condition, put it in grey by default
#     """
#     results = adata.uns['sparty']['results']
#     matrix = adata.uns["sparty"]["matrice"]
#     celltypes = results[cell_type].unique()
    
#     for cell in celltypes:
#         sub_mtx = matrix.loc[:,matrix.columns.str.contains(f'_{cell}_')].T
#         adata = ad.AnnData(sub_mtx)
#         adata.obs[['celltype','condition']] = adata.obs_names.str.split('_', expand=True).to_frame(index=False)[[2,3]].values
#         # print(cell)
#         sub_cell = results[results[cell_type] == cell]
#         sub_cell.index = sub_cell['gene']

#         signs = sub_cell.loc[
#             (sub_cell['padj'] <= thr_sign) & 
#             (np.abs(sub_cell['log2FoldChange']) >= thr_stat) &
#             (sub_cell['baseMean'] >=min_base_mean) & 
#             ((sub_cell['pct_1'] >= min_pct) | (sub_cell['pct_2']  >= min_pct)), 'gene'].unique()
    
#         col_colors = pd.DataFrame(adata.obs[condition].map(colors[condition]))
#         col_colors['celltype'] = "Blue"
#         col_colors = col_colors.fillna(fill_na)
#         # row_colors #= row_colors[['celltype', 'condition']] 

#         if len(signs) > 0:        
#             sc.pp.normalize_total(adata)
#             sc.pp.log1p(adata)
#             sc.pp.scale(adata, max_value=10)

#             # df_test = pd.DataFrame(adata[:,signs].X.T, index=adata[:,signs].var_names, columns=adata.obs_names) 
#             # df_test = df_test.apply(pd.to_numeric, errors="coerce")

#             heatmap_volcano(
#                 adata,
#                 sub_cell,
#                 signs=signs,
#                 colors=colors,
#                 title = cell,
#                 figsize=(20, 10),
#                 thr_stat = thr_stat, 
#                 thr_sign=thr_sign,
#                 top=top_volcano,
#             )


# def plot_pseudobulk(
#     adata: ad.AnnData,
#     x_key: str = 'scmusk',
#     y_key: str = 'log2FoldChange',
#     palette: tuple = None,
#     key: str = 'results',
#     title: str = None,
#     padj: float = 0.05,
#     log2FoldChange: float = 1,
#     baseMean: float = 50,
#     figsize: tuple = (8,3),
#     save: bool = False,
#     save_format: str = 'pdf',
# ):
#     """Plot DEG dataframe from pseudobulk analysis

#     Parameters
#     ----------
#     adata
#         anndata object
#     x_key
#         x key
#     y_key
#         y key
#     key
#         key in adata.uns['sparty'] storing the results to plot
#     padj
#         p adjusted to be significant
#     log2FoldChange
#         log2FoldChange to be significant
#     figsize
#         figure size
#     save
#         wether or not to save the figure
#     save_format
#         pdf or png
#     """
#     if palette is None:
#         palette="deep"

#     df = adata.uns['sparty'][key].copy()

#     df['significative'] = 0
#     df.loc[df.padj < padj, 'significative'] = 1
#     df.loc[abs(df.log2FoldChange) < log2FoldChange, 'significative'] = 0
#     df.loc[abs(df.baseMean) < baseMean, 'significative'] = 0

#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     tmp = df[df.significative==1]
#     tmp = tmp.reset_index(drop=True)
#     order = list(tmp.groupby([x_key]).groups.keys())

#     sns.stripplot(data=tmp, x=x_key, y=y_key, hue=x_key, palette=palette,
#                 orient='v', order=order, alpha=0.6, size=5, linewidth=1, edgecolor='black', jitter=0.4)

#     grouped = tmp.groupby([x_key])
#     # Label the points within each group
#     for collection, (group_key, group_data) in zip(ax.collections, grouped):
#         for i, (x, y) in enumerate(collection.get_offsets()):
#             y_value = group_data.iloc[i]['index']
#             ax.text(x, y+0.1, y_value, ha='left', va='bottom', color='grey', size='x-small')

#     tmp = df[df.significative==0]
#     sns.stripplot(data=tmp, x=x_key, y=y_key, color="grey", 
#                 orient='v', alpha=0.6, size=2, jitter=0.4)

#     ax.tick_params(axis='x', rotation=90)
#     ax.set_title(title)

#     # save figure
#     if save is True:
#         if save_format == "pdf":
#             print("saving plot_pseudobulk.pdf")
#             plt.savefig("plot_pseudobulk.pdf", bbox_inches="tight")
#         elif save_format == "png":
#             print("saving plot_pseudobulk.png")
#             plt.savefig("plot_pseudobulk.png", dpi=300, bbox_inches="tight")

# def stripPlotDE(
#     adata: ad.AnnData,
#     x_key: str = 'cell_type',
#     y_key: str = 'log2FoldChange',
#     splitby = "condition",
#     palette: tuple | str = "deep",
#     key: str = 'results',
#     groups: list | None = None,
#     # title: str = None,
#     padj: float = 0.05,
#     logFC: float = 0.5,
#     baseMean: float = 50.0,
#     figsize: tuple = (8,3),
#     top: int = 5,
#     order: list | None = None,
#     join_by: str = '..',
#     # save_format: str = 'pdf',
#     save: str | None = None,
#     xticks_rotation: int = 90, 
#     dpi: int = 300
# ):
#     """Plot DEG dataframe from pseudobulk analysis

#     Parameters
#     ----------
#     adata
#         anndata object
#     x_key
#         x key
#     y_key
#         y key
#     key
#         key in adata.uns['sparty'] storing the results to plot
#     padj
#         p adjusted to be significant
#     log2FoldChange
#         log2FoldChange to be significant
#     figsize
#         figure size
#     save
#         wether or not to save the figure
#     save_format
#         pdf or png
#     """
#     if isinstance(adata, ad.AnnData):
#         if key is None:
#             raise ValueError("`key` must be provided when `adata` is an AnnData object.")
#         df = adata.uns['sparty'][key].copy()
#     elif isinstance(adata, pd.DataFrame):
#         df = adata.copy()
#     else:
#         raise TypeError("`adata` must be either a pandas DataFrame or an AnnData object.")

#     df["significative"] = np.where(
#             (df["padj"] <= padj) & 
#             (df["log2FoldChange"].abs() >= logFC) & 
#             (df["baseMean"] >= baseMean),
#             1, 0)
#     # df["significative"] = np.where(
#     #     (df["padj"] < padj) & (df["log2FoldChange"].abs() > logFC) & (df["baseMean"].abs() > baseMean),
#     #     "Sig", "NS"
#     # )
    
#     if len(df[splitby].unique()) > 1 : 
#         print("More than one pairwise condition.")

#     for cond in df[splitby].unique():
#         print(cond)
#         sub_df = df[df[splitby] == cond]
        
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         if groups:
#             groups_2= groups
#             sub_df = sub_df[sub_df[x_key].isin(groups)]
#         else:
#             groups_2 = sub_df.loc[sub_df["significative"] == 1, x_key].unique()
#             sub_df = sub_df[sub_df[x_key].isin(groups_2)]
        
#         tmp = sub_df[sub_df['significative']==1].reset_index(drop=True)
#         order = list(tmp.groupby([x_key]).groups.keys())
        
#         sns.stripplot(
#             data=sub_df[sub_df['significative']==0], 
#             x=x_key, y=y_key, # hue=x_key, palette=palette,
#             color='gray',  # fixe pour tous
#             orient='v', order=order, alpha=0.6, size=2, linewidth=0, 
#             # edgecolor='black', 
#             jitter=0.4)
    
#         sns.stripplot(
#             data=sub_df[sub_df['significative']==1], 
#             x=x_key, y=y_key, hue=x_key, palette=palette,
#             orient='v', order=order, alpha=0.8, size=5, linewidth=1, 
#             # edgecolor='black', 
#             jitter=0.4)
        
#         tmp["rank"] = tmp.groupby(x_key)["log2FoldChange"].rank(method="first", ascending=False)
#         tmp["top_bottom"] = tmp.groupby(x_key).apply(
#             lambda g: pd.Series(
#                 np.where(
#                     g["rank"] <= min(top, len(g)), "Top",
#                     np.where(g["rank"] > len(g) - min(top, len(g)), "Bottom", None)
#                 ), index=g.index
#             )
#         ).reset_index(level=0, drop=True)

#         # tmp["top_bottom"] = np.where(
#         #                 tmp["rank"] <= top, "Top",
#         #                 np.where(tmp["rank"] > (tmp.groupby(x_key)["log2FoldChange"].transform("count") - top), 
#         #                         "Bottom", None))
#         # Label the top points within each group and adjust them
#         texts = []
#         for collection, (_, group_data) in zip(ax.collections[-len(tmp[x_key].unique()):], tmp.groupby(x_key)):
#             for i, (x, y) in enumerate(collection.get_offsets()):
#                 if group_data.iloc[i]['top_bottom'] is not None:
#                     texts.append(ax.text(
#                         x, y, group_data.iloc[i]['gene'],
#                         ha='center', va='bottom', color='black', size='x-small'
#                     ))
#         if texts:
#             at.adjust_text(
#                 texts, expand=(2,2),
#                 arrowprops=dict(arrowstyle='->', color='black'), ax=ax
#             )
                
#         # # Label the top points within each group and adjust them
#         # texts = []
#         # for collection, (_, group_data) in zip(ax.collections, tmp.groupby([x_key])):
#         #     for i, (x, y) in enumerate(collection.get_offsets()):
#         #         if group_data.iloc[i]['top_bottom'] is not None:
#         #             texts.append(ax.text(x, y, 
#         #                     group_data.iloc[i]['gene'], 
#         #                     ha='center', 
#         #                     va='bottom', color='black', 
#         #                     size='x-small' # "small"
#         #                     ))
#         # if len(texts) > 0:
#         #     at.adjust_text(texts, 
#         #                 expand=(2, 2),
#         #                 arrowprops=dict(arrowstyle='->', color='black'), ax=ax)

#         ax.tick_params(
#             axis='x', rotation=xticks_rotation, size=12) # , ha='center') #  ha='right'
#         cond1, cond2 = cond.split(join_by)
#         ax.set_title(f'Stripplot of the genes DE in {cond1} versus {cond2} for each cell type')
        
#         # save figure
#         if isinstance(save, str):
#             plt.savefig(save, bbox_inches="tight", dpi=dpi)
#         #     plt.savefig(save, bbox_inches="tight", dpi=dpi)
#         # if isinstance(save, str):
#         #     if save_format == "pdf":
#         #         # print(cond + "saving plot_pseudobulk.pdf")
#         #         plt.savefig(save, bbox_inches="tight",, dpi=dpi)
#         #     elif save_format == "png":
#         #         # print(cond + "saving plot_pseudobulk.png")
#         #         plt.savefig(save + "stripplot.png", dpi=300, bbox_inches="tight")
#         plt.show()
    