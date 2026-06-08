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
from statannotations.Annotator import Annotator
from itertools import combinations
from ..colors import get_palette

from ..tl.basic import sdata_rotate, add_to_points

sc.set_figure_params(vector_friendly=True, dpi=300, dpi_save=300) 
plt.rcParams.update(
    {'ps.fonttype':42,
    'ps.fonttype': 42, 
    'pdf.fonttype': 42, 
    'font.size': 10, 
    'font.family': 'Arial', 
    'mathtext.fontset': 'cm', 
    'mathtext.rm': 'Arial',
    'lines.linewidth': .2, 
    'xtick.top': False, 
    'ytick.right': False}
)

def plot_mecr(
        adata: ad.AnnData,
        g1: str, 
        g2: str,
        groupby: str = None,
        layer: str = 'counts',
        color: 'str' = 'black',
        figsize: tuple = (5,5),
):
    c1 = adata[:,g1].layers[layer].toarray()
    c2 = adata[:,g2].layers[layer].toarray()

    mecr = ((c1 > 0) & (c2 > 0)).sum() / ((c1 > 0) | (c2 > 0)).sum()

    if groupby:
        df = adata.obs[groupby].copy()

    plt.figure(figsize=figsize)
    plt.scatter(
        x=c1,
        y=c2, 
        s=2,
        c=color,
    )
    plt.xlabel(g1)
    plt.ylabel(g2)
    plt.title(f'MECR score between {g1} - {g2} : {round(mecr, 3)}')
    plt.show()


# def plot_shapes(
#     sdata: sd.SpatialData,
#     group_lst: tuple = None,  # the cell types to consider
#     shapes_lst: tuple = None,  # the shapes to plot
#     color_key: str = "celltype_spatial",
#     shape_key: str = "arteries",
#     target_coordinates: str = "microns",
#     figsize: tuple = (12, 6),
#     palette: tuple = None,
#     save: bool = False,
# ):
#     """Plot list of shapes

#     Parameters
#     ----------
#     sdata
#         SpatialData object obtained by tl.get_sdata_polygon()
#     group_lst
#         group list to consider (related to label_obs_key)
#     shapes_lst
#         shapes list to plot
#     color_key
#         label_key in sdata['table'].obs to consider
#     shape_key
#         SpatialData shape element to consider
#     palette
#         dictionary of colors to use
#     target_coordinates
#         target_coordinates system of sdata object
#     figsize
#         figure size
#     save
#         wether or not to save the figure

#     """
#     region_key = sdata['table'].uns["spatialdata_attrs"]["region"]
#     my_shapes = {region_key: sdata[region_key], shape_key: sdata[shape_key]}
#     my_tables = {"table": sdata["table"]}
#     sdata2 = SpatialData(shapes=my_shapes, tables=my_tables)

#     fig, axs = plt.subplots(ncols=len(shapes_lst), nrows=1, figsize=figsize)
#     for i in range(0, len(shapes_lst)):
#         poly = sdata2[shape_key][sdata2[shape_key].name == shapes_lst[i]].geometry.item()
#         sdata3 = sd.polygon_query(
#             sdata2,
#             poly,
#             target_coordinate_system=target_coordinates,
#             filter_table=True,
#         )

#         # sdata3.pl.render_images().pl.show(ax=axs[i])
#         if group_lst is None:
#             group_lst = sdata2['table'].obs[color_key].unique().tolist()

#         if palette is not None:
#             mypal = [palette[x] for x in group_lst]
#             sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst, palette=mypal).pl.show(
#                 ax=axs[i]
#             )
#         else:
#             sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst).pl.show(ax=axs[i])

#         axs[i].set_title(shapes_lst[i])
#         if i < len(shapes_lst) - 1:
#             axs[i].get_legend().remove()

#     plt.tight_layout()


def plot_shape_along_axis(
    sdata: sd.SpatialData,
    group_lst: tuple = [],  # the cell types to consider
    gene_lst: tuple = [],  # the genes to consider
    color_key: str = "scmusk",
    sdata_group_key: str = "scmusk",
    target_coordinates: str = "global",
    palette: tuple = [],
    scale_expr: bool = False,
    bin_size: int = 50,
    rotation_angle: int = 0,
    heatmap: bool = False,
    save: bool = False,
    figsize: tuple = (10, 6),
):
    """Analyse cell types occurence and gene expression along x axis of a sdata polygon shape after an evenual rotation

    Parameters
    ----------
    sdata
        SpatialData object obtained by tl.get_sdata_polygon()
    group_lst
        group list to consider (related to label_obs_key)
    gene_lst
        gene list to consider
    color_key
        label_key in sdata['table'].obs to consider
    sdata_group_key
        SpatialData element where to find the group label
    target_coordinates
        target_coordinates system of sdata object
    scale_expr
        wether or not to scale the gene expression plot
    bin_size
        size of bins for plotting (µm)
    rotation_angle
        horary rotation angle of the shape before computing along x axis
    save
        wether or not to save the figure
    """
    if group_lst is None:
        group_lst = sdata['table'].obs[color_key].unique().tolist()

    sdata_rotate(sdata, target_coordinates=target_coordinates, rotation_angle=rotation_angle)
    sdata2 = sdata.transform_to_coordinate_system(target_coordinates)
    
    dataset_id = sdata2['table'].obs.dataset_id.unique()[0]
    feature_key = sdata2['table'].uns["spatialdata_attrs"]["feature_key"]
    polygon_key = sdata2['table'].uns["spatialdata_attrs"]["region"][0]
    transcript_key = list(sdata.points.keys())[0]  # need to be in spatialdata_attrs

    # compute dataframes
    df_transcripts = sdata2[transcript_key].compute()
    df_celltypes = sdata2[sdata_group_key].compute()

    # parametrage
    x_min = df_transcripts.x.min()
    total_x = df_transcripts.x.max() - df_transcripts.x.min()
    step_number = int(total_x / bin_size)

    # init color palette
    #cats = sdata['table'].obs[color_key].cat.categories.tolist()
    #colors = list(sdata['table'].uns[color_key + "_colors"])
    #mypal = dict(zip(cats, colors))
    if palette is not None:
        mypal = palette
        mypal = {x: mypal[x] for x in group_lst}

    # compute values dataframes
    vals = pd.DataFrame({target_coordinates: [], "count": [], feature_key: []})
    for g in range(0, len(gene_lst)):
        df2 = df_transcripts[df_transcripts[feature_key] == gene_lst[g]]
        df2.shape[0] / step_number
        for i in range(0, step_number):
            new_row = {
                target_coordinates: (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                feature_key: gene_lst[g],
            }
            vals = pd.concat([vals, pd.DataFrame([new_row])], ignore_index=True)

    valct = pd.DataFrame({target_coordinates: [], "count": [], "celltype": []})
    for ct in range(0, len(group_lst)):
        df2 = df_celltypes[df_celltypes["ct"] == group_lst[ct]]
        for i in range(0, step_number):
            new_row = {
                target_coordinates: (x_min + (i + 0.5) * bin_size),
                "count": df2[(df2.x > (x_min + i * bin_size)) & (df2.x < (x_min + (i + 1) * bin_size))].shape[0],
                "celltype": group_lst[ct],
            }
            valct = pd.concat([valct, pd.DataFrame([new_row])], ignore_index=True)

    # draw figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=figsize, sharex=True)

    sdata.pl.render_shapes(
        elements=polygon_key, color=color_key, palette=list(mypal.values()), groups=group_lst,
        fill_alpha=1, outline_width=0.5, outline_color='#000000', outline_alpha=1,
    ).pl.show(ax=ax1)
    #ax1.legend(bbox_to_anchor=(1.0, 1.0))
    ax1.get_legend().remove()

    sns.lineplot(data=valct, x=target_coordinates, y="count", hue="celltype", linewidth=0.9, palette=mypal, ax=ax2)
    ax2.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    if scale_expr is True:
        means_stds = vals.groupby([feature_key])["count"].agg(["mean", "std", "max"]).reset_index()
        vals = vals.merge(means_stds, on=[feature_key])
        vals["norm_count"] = vals["count"] / vals["max"]

        if heatmap is True:
            #sns.set(font_scale=0.5)
            sns.heatmap(vals.pivot(index="gene", columns=target_coordinates, values="norm_count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x=target_coordinates,
                y="norm_count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    else:
        if heatmap is True:
            sns.heatmap(vals.pivot(index="gene", columns=target_coordinates, values="count"), cmap="viridis", ax=ax3)
        else:
            sns.lineplot(
                data=vals,
                x=target_coordinates,
                y="count",
                hue=feature_key,
                linewidth=0.9,
                palette=sns.color_palette("Paired"),
                ax=ax3,
            )
            ax3.legend(bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    ax1.set_title(dataset_id)
    ax2.set_title("Number of cells per cell types (" + str(int(bin_size)) + " µm bins)")
    ax3.set_title("Gene expression (" + str(int(bin_size)) + " µm bins)")
    #plt.legend(fontsize='small', title_fontsize='6')
    plt.tight_layout()

    # save figure
    if save is True:
        print("saving " + dataset_id + ".pdf")
        plt.savefig(dataset_id + ".pdf", format="pdf", bbox_inches="tight")


def plot_sdata(
    sdata: sd.SpatialData,
    color_key: str = "celltype",
    feature_key: str = "gene",  # or feature_name for xenium
    point_size: int = 1,
    figsize: tuple = (12, 6),
    outline: bool = False,
    outline_width: float = 1.0,
    outline_color: str = "red",
    shape_keys: str = None,
    shape_palette: tuple = None,
    shape_groups: tuple = None,  # need shape_palette to be defined
    cmap: tuple = None,
    point_groups: tuple = None,  # 10 maximum
    grid: bool = False,
    save: bool = False,
    target_coordinates: str = "microns",
    save_format: str = "pdf",
):
    """Plot sdata object (i.e. embedding and polygons). This should always works if well synchronized sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    color_key
        color key from ['table'].obs
    """
    if shape_keys is None:
        shape_keys = sdata['table'].uns['spatialdata_attrs']['region']
        #shape_keys = list(sdata.shapes.keys())  # better would be to get the list of spatialdata_attrs 'cell_regions'
    if feature_key is None:
        feature_key = sdata.tables[list(sdata.tables.keys())[0]].uns["spatialdata_attrs"]["feature_key"]

    args_shapes = {"element": shape_keys, "method": "matplotlib", "color": color_key}
    args_points = {"size": point_size, "color": feature_key}

    # size=0.01, linewidth=None, marker=".", edgecolor = 'none', markeredgewidth=0.0

    fig, ax = plt.subplots(figsize=figsize)
    if shape_palette is not None:
        args_shapes["palette"] = list(shape_palette.values())
        args_shapes["groups"] = list(shape_palette.keys())
        if shape_groups is not None:
            mypal = {x: shape_palette[x] for x in shape_groups}
            args_shapes["palette"] = list(mypal.values())
            args_shapes["groups"] = list(mypal.keys())
    if outline is True:
        args_shapes["outline"] = True
        args_shapes["outline_width"] = outline_width
        args_shapes["outline_color"] = outline_color
    if cmap is not None:  # case on continuous value like gene expression or .obs metadata
        args_shapes["cmap"] = cmap

    args_shapes["coordinate_system"] = target_coordinates
    sdata.pl.render_shapes(**args_shapes).pl.show(ax=ax)

    if point_groups is not None:
        point_dict = get_palette("fluo")
        if len(point_groups) > 10:
            point_groups = point_groups[0:10]
        point_pal = list(point_dict.values())[0 : len(point_groups)]
        sdata.pl.render_points(**args_points, groups=point_groups, palette=point_pal).pl.show(ax=ax)

    if grid is True:
        ax.grid(which="both", linestyle="dashed")
        ax.minorticks_on()
        ax.tick_params(which="minor", bottom=False, left=False)

    legend_without_duplicate_labels(ax)
    plt.tight_layout()

    # save figure
    if save is True:
        if save_format == "pdf":
            print("saving plot_sdata.pdf")
            plt.savefig("plot_sdata.pdf", bbox_inches="tight")
        elif save_format == "png":
            print("saving plot_sdata.png")
            plt.savefig("plot_sdata.png", dpi=300, bbox_inches="tight")


def plot_multi_sdata(
    sdata: sd.SpatialData,
    color_key: str = "celltype",
    save: bool = False,
):
    """Plot sdata object (i.e. embedding and polygons). This should always works if well synchronized sdata object

    Parameters
    ----------
    sdata
        SpatialData object.
    color_key
        color key from ['table'].obs
    """
    sdata['table'].obs[color_key] = sdata['table'].obs[color_key].cat.remove_unused_categories()
    sdata['table'].uns.pop(color_key + "_colors", None)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    sns.scatterplot(x="center_x", y="center_y", data=sdata['table'].obs, s=1, hue=color_key, ax=ax1)
    ax1.axis("equal")
    ax1.get_legend().remove()
    ax1.set_title("anndata.obs coordinates")
    ax1.invert_yaxis()

    sdata.pl.render_shapes(element=sdata['table'].uns["spatialdata_attrs"]["region"][0], color=color_key).pl.show(ax=ax2)
    ax2.get_legend().remove()
    ax2.set_title("spatialdata polygons")

    sc.pl.embedding(sdata['table'], "spatial", color=color_key, size=1, ax=ax3)
    #sq.pl.spatial_scatter(sdata['table'], basis='spatial', color=color_key, shape=None, size=1, ax=ax3)
    #ax2.get_legend().remove()
    #ax3.set_title("squidpy spatial")

    plt.tight_layout()

    # save figure
    if save is True:
        print("saving plot_multi_sdata.pdf")
        plt.savefig("plot_multi_sdata.pdf", format="pdf", bbox_inches="tight")


def plot_qc(sdata: sd.SpatialData):
    """Plot quality control analysis.

    Parameters
    ----------
    sdata
        SpatialData object.

    """
    dataset_id = sdata['table'].obs.dataset_id.unique().tolist()[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplot(2, 2, 1)
    bins = np.logspace(0, 4, 100)
    plt.hist(sdata['table'].obs["volume"], alpha=0.2, bins=bins, label=dataset_id, color="red")
    plt.xlabel("Volume")
    plt.ylabel("Cell count")
    plt.xscale("log")
    # Transcript count by cell
    plt.subplot(2, 2, 2)
    bins = np.logspace(0, 4, 100)
    plt.hist(sdata['table'].obs["transcript_count"], alpha=0.2, bins=bins, label=dataset_id, color="red")
    plt.xlabel("Transcript count")
    plt.ylabel("Cell count")
    plt.xscale("log")
    plt.yscale("log")
    plt.subplot(2, 2, 3)
    barcodeCount = sdata['table'].obs["transcript_count"]
    sns.distplot(barcodeCount, label=dataset_id, color="red")
    ax1 = plt.subplot(2, 2, 4)
    sc.pl.violin(sdata['table'].obs, keys="transcript_count", ax=ax1)
    plt.tight_layout()


def plot_per_groups(adata, clust_key, size=60, is_spatial=False, frameon=False, legend_loc=None, **kwargs):
    """Plot UMAP splitted by clust_key

    Parameters
    ----------
    adata
        Anndata object.
    clust_key
        key to plot
    is_spatial
        UMAP plot if False,

    """
    tmp = adata.copy()

    for i, clust in enumerate(adata.obs[clust_key].cat.categories):
        tmp.obs[clust] = adata.obs[clust_key].isin([clust]).astype("category")
        tmp.uns[clust + "_colors"] = ["#d3d3d3", adata.uns[clust_key + "_colors"][i]]

    if is_spatial is False:
        sc.pl.umap(
            tmp,
            groups=tmp.obs[clust].cat.categories[1:].values,
            color=adata.obs[clust_key].cat.categories.tolist(),
            size=size,
            frameon=frameon,
            legend_loc=legend_loc,
            **kwargs,
        )
    else:
        # not working !....
        tmp.uns["spatial"] = tmp.obsm["spatial"]
        
        #sc.pl.embedding(sdata['table'], "spatial", color=key, size=1, ax=axs[1])
        
        #sq.pl.spatial_scatter(
        #    tmp,
        #    groups=tmp.obs[clust].cat.categories[1:].values,
        #    color=adata.obs[clust_key].cat.categories.tolist(),
        #    size=size,
        #    frameon=frameon,
        #    legend_loc=legend_loc,
        #    **kwargs,
        #)
        
def scis_prop(
    adata: ad.AnnData,
    sample_col: str = "sample",
    condition_col: str = 'condition',
    group_col: str = "cell_type",
    group_only: str | list = None,
    group_top: int = None,
    strip_hue_col: str = None,
    palette_box: dict = None,
    palette_strip: dict = None,
    condition_order: list = None, # ["CTRL", "PAH"],  # might be possible to provide more conditions
    stat_test: str = "t-test_ind", #t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel
    ncols: int = 4,
    sub_figsize: tuple = (5,5),
    hspace: float = 0.5, 
    wspace: float = 0.5,
    save: str | None = None,
    rotate:int = 0,
    bbox_to_anchor=(0.98, 0.5),
):
    """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        group
    group_only
        just plot this group
    split_by
        x value split_by
    split_only
        focus on this split_by
    split_by_top
        top split_by to consider
    replicate
        replicate key in adata.obs
    condition
        condition key in adata.obs
    condition_order
        tuple of the x conditions to test
    
    figsize
        figure size
    Returns
    -------

    """
    def make_pairs(data, condition_col, condition_order=None):
        levels = data[condition_col].dropna().unique().tolist()
        if condition_order is not None:
            levels = [x for x in condition_order if x in levels]
        if len(levels) < 2:
            return []
        return list(combinations(levels, 2))

    df = adata.obs[[sample_col, condition_col, group_col]]
    nb_cells = df.groupby([sample_col, condition_col, group_col]).size().unstack()
    nb_cells = nb_cells.div(nb_cells.sum(axis=1), axis=0).reset_index()
    nb_cells = nb_cells.melt(id_vars=[sample_col, condition_col])
    nb_cells = nb_cells.dropna()

    if isinstance(group_only, str):
        group_only = [group_only]

    if group_only:
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    elif group_top:
        group_only = list(df[group_col].value_counts().head(group_top).index)
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    else:
        group_only = list(df[group_col].unique())

    if not condition_order:
        condition_order = list(adata.obs[condition_col].unique())

    nrows = (len(group_only) + ncols - 1) // ncols

    fig = plt.figure(figsize=(sub_figsize[0]*ncols, sub_figsize[1]*nrows))
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    legend_handles = None
    legend_labels = None

    for i, group in enumerate(group_only, 1):
        ax = fig.add_subplot(nrows, ncols, i)
        data_sub = nb_cells[nb_cells[group_col] == group]

        sns.boxplot(
            data=data_sub,
            x=condition_col,
            y="value",
            order=condition_order,
            width=0.7,
            # gap=0.1,
            showfliers=False,
            linewidth=0.2,
            ax=ax,
            palette=palette_box
        )

        if strip_hue_col:
            strip_df = adata.obs[[sample_col, strip_hue_col]].drop_duplicates()
            st_dict = dict(zip(strip_df[sample_col], strip_df[strip_hue_col].astype(str)))
            data_sub = data_sub.copy()
            data_sub[strip_hue_col] = data_sub[sample_col].map(st_dict)
            sns.stripplot(
                data=data_sub,
                x=condition_col,
                y="value",
                hue=strip_hue_col,
                palette=palette_strip,
                dodge=False,
                jitter=0.12,
                size=6,
                linewidth=0.2,
                edgecolor="black",
                ax=ax,
                legend=(legend_handles is None)
            )
        else:
            sns.stripplot(
                data=data_sub,
                x=condition_col,
                y="value",
                palette=palette_box,
                dodge=False,
                jitter=0.12,
                size=6,
                linewidth=0.2,
                edgecolor="black",
                ax=ax,
                legend=(legend_handles is None)
            )
        if strip_hue_col and i==1:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
        pairs = make_pairs(data_sub, condition_col, condition_order)
        
        if pairs:
            annotator = Annotator(
                ax, pairs, data=data_sub,
                x=condition_col, y="value",
                order=condition_order
            )
            annotator.configure(
                test=stat_test,
                text_format="star",
                hide_non_significant=True,
                line_width=0.2,
                text_offset=0.15,
                pvalue_thresholds=[
                    [1e-4, "****"],
                    [1e-3, "***"],
                    [1e-2, "**"],
                    [0.05, "*"]]
            )
            annotator.apply_and_annotate()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(group)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate) #, size=6)

        if i%ncols == 1:
            ax.set_ylabel("Proportion")
        else:
            ax.set_ylabel("")
    
    if strip_hue_col and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title=strip_hue_col,
            loc="center right",
            frameon=True,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=12,    
            title_fontsize=14  
        )
    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight")
    plt.show()



def scis_prop_OLD(
    adata: ad.AnnData,
    sample_col: str = "sample",
    condition_col: str = 'condition',
    group_col: str = "cell_type",
    group_only: str | list = None,
    group_top: int = None,
    strip_hue_col: str = None,
    palette_box: dict = None,
    palette_strip: dict = None,
    condition_order: list = None, # ["CTRL", "PAH"],  # might be possible to provide more conditions
    stat_test: str = "t-test_ind", #t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel
    figsize: tuple = (6, 3),
    title: str = None,
    save: str | None = None,
):
    """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        group
    group_only
        just plot this group
    split_by
        x value split_by
    split_only
        focus on this split_by
    split_by_top
        top split_by to consider
    replicate
        replicate key in adata.obs
    condition
        condition key in adata.obs
    condition_order
        tuple of the x conditions to test
    
    figsize
        figure size
    Returns
    -------

    """
    # sns.set_theme(style="whitegrid", palette="pastel")
    df = adata.obs[[sample_col, condition_col, group_col]]

    nb_cells = df.groupby([sample_col, condition_col, group_col]).size().unstack()
    nb_cells = nb_cells.div(nb_cells.sum(axis=1), axis=0).reset_index()
    nb_cells = nb_cells.melt(id_vars=[sample_col, condition_col])
    nb_cells = nb_cells.dropna()

    if isinstance(group_only, str):
        group_only = [group_only]

    if group_only:
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    elif group_top:
        group_only = list(df[group_col].value_counts().head(group_top).index)
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    else:
        group_only = list(df[group_col].unique())

    if not condition_order:
        condition_order = list(adata.obs[condition_col].unique())

    pairs_comb = list(combinations(condition_order,2))
    pairs = []

    for grp in group_only:
        for prs in pairs_comb:
            conds = nb_cells[nb_cells[group_col] == grp][condition_col].unique()
            if (prs[0] in conds) and (prs[1] in conds):
                pairs.append(((grp, prs[0]), (grp, prs[1])))
            else:
                print(f'paire {prs} in {grp} not exists')

    hue_plot_params = {
        "data": nb_cells,
        "x": group_col,
        "y": "value",
        "order": group_only,
        "hue": condition_col,
        "hue_order": condition_order,
        "palette": palette_box,
    }

    if len(pairs) > 0:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.boxplot(ax=ax, **hue_plot_params, boxprops={"alpha": 0.8}, showfliers=False, linecolor= 'black', linewidth=0.2, gap=0.3, width=.5)
        if strip_hue_col:
            strip_df = adata.obs[[sample_col, strip_hue_col]].drop_duplicates()
            st_dict = dict(zip(strip_df[sample_col], strip_df[strip_hue_col].astype(str)))
            nb_cells[strip_hue_col] = nb_cells[sample_col].map(st_dict)
            hue_striplot_params = {
                "data": nb_cells,
                "x": group_col,
                "y": "value",
                "order": group_only,
                "hue": strip_hue_col,
                # "hue_order": strip_order,
                "palette": palette_strip,
            }
            sns.stripplot(ax=ax, **hue_striplot_params, dodge=True, edgecolor="black", linewidth=0.2, size=5)
        else: 
            sns.stripplot(ax=ax, **hue_plot_params, dodge=True, edgecolor="black", linewidth=0.2, size=5)

        annotator = Annotator(ax, pairs, **hue_plot_params)
        annotator.configure(
            test=stat_test, 
            text_format="star",
            hide_non_significant=True,
            text_offset=0.2,
            line_width = 0.2,
            pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"]],  # removing ns as won't be shown
        )
        annotator.apply_and_annotate()
        
        if strip_hue_col:
            plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.0)
        else:
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(
                handles[0:len(condition_order)], labels[0:len(condition_order)], 
                bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.0)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=6)
        # ax.set_yticklabels(ax.get_yticklabels(), size=6)
        # ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set(ylabel="")
        ax.set_title(title)
        plt.tight_layout()

        if isinstance(save, str):
            plt.savefig(save, bbox_inches="tight")
        plt.show()



# def scis_prop(
#     adata: ad.AnnData,
#     group_by: str = "scmusk_T4",
#     group_only: str = "",
#     split_by: str = "anatomy",
#     split_only: str = "",
#     split_by_top: int = 5,
#     test="Mann-Whitney",
#     replicate: str = "sample",
#     condition: str = "group",
#     condition_order: tuple = ["CTRL", "PAH"],  # might be possible to provide more conditions
#     figsize: tuple = (6, 3),
#     rotation = 0,
#     size = 16,
#     xlegend = 'Cell types',
#     **kwargs,
# ):
#     """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

#     Parameters
#     ----------
#     adata
#         AnnData object.
#     group_by
#         group
#     group_only
#         just plot this group
#     split_by
#         x value split_by
#     split_only
#         focus on this split_by
#     split_by_top
#         top split_by to consider
#     replicate
#         replicate key in adata.obs
#     condition
#         condition key in adata.obs
#     condition_order
#         tuple of the x conditions to test
    
#     figsize
#         figure size
#     Returns
#     -------

#     """
#     sns.set_theme(style="whitegrid", palette="pastel")
#     l = list(adata.obs[group_by].unique())
#     if group_only:
#         l = [group_only]

#     for n in l:
#         print(n)
#         df = adata[adata.obs[group_by] == n].obs[[replicate, condition, split_by]]
#         df2 = df.groupby([replicate, condition, split_by])[split_by].count().unstack()
#         df2 = df2.div(df2.sum(axis=1), axis=0).reset_index()
#         df2 = df2.melt(id_vars=[replicate, condition])
#         df2 = df2.dropna()
#         df2 = df2[df2.value > 0]
        
#         hits = list(df[split_by].value_counts().head(split_by_top).keys())
#         df2 = df2[df2[split_by].isin(hits)]

#         if split_only:
#             df2 = df2[df2[split_by].isin(split_only)]
#             hits = split_only

#         split_order = hits

#         pairs = []
#         for s in split_order:
#             if len(df2[df2[split_by] == s][condition].unique()) > 1:
#                 pairs.append([(s, condition_order[0]), (s, condition_order[1])])
#                 if(len(condition_order) > 2):
#                     pairs.append([(s, condition_order[0]), (s, condition_order[2])])
#                     pairs.append([(s, condition_order[1]), (s, condition_order[2])])

#         hue_plot_params = {
#             "data": df2,
#             "x": split_by,
#             "y": "value",
#             "order": split_order,
#             "hue": condition,
#             "hue_order": condition_order,
#             # "palette": pal_group,
#         }

#         if len(pairs) > 0:
#             fig, ax = plt.subplots(1, 1, figsize=figsize)
#             sns.boxplot(ax=ax, **hue_plot_params, showfliers=False, linewidth=0.5, **kwargs)
#             # sns.boxplot(ax=ax, **hue_plot_params, boxprops={"alpha": 0.8}, showfliers=False, linewidth=0.5)
#             sns.stripplot(ax=ax, **hue_plot_params, dodge=True, edgecolor="black", linewidth=0.5, size=3, **kwargs)

#             annotator = Annotator(ax, pairs, **hue_plot_params)
#             annotator.configure(test= test, text_format="star")
#             annotator.apply_and_annotate()

#             handles, labels = ax.get_legend_handles_labels()
#             l = plt.legend(handles[0:len(condition_order)], 
#                            labels[0:len(condition_order)], 
#                            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

#             ax.set_xticklabels(ax.get_xticklabels(), 
#                                rotation=rotation, 
#                                size=size)
#             ax.set_yticklabels(ax.get_yticklabels(), size=size)
#             ax.xaxis.grid(True)
#             ax.yaxis.grid(True)
#             ax.set(ylabel="")
#             ax.set_title(str(n))
#             plt.xlabel(xlegend, fontsize=size)
#             plt.ylabel('Proportions', fontsize=size)
#             plt.tight_layout()


def legend_without_duplicate_labels(figure):
    """Remove duplicated labels in figure legend

    Parameters
    ----------
    figure
        matplotlib figure.
    """
    # code from here
    # https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=6, ncol=1)
