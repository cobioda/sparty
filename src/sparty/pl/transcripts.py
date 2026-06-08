import numpy as np
import matplotlib.pyplot as plt
import shapely
import geopandas as gpd
import spatialdata as sd
import re
from spatialdata.transformations import get_transformation #, Identity, Sequence
# from spatialdata.transformations.transformations import , Sequence, Identity
# import pandas as pd 
from matplotlib import gridspec
import matplotlib.colors as mcolors
import math

from ..pp.density import density_count_genes, _count_dens, compute_coloc  
from ..pp.transcripts import subset_transcripts
# from matplotlib.patches import FancyArrow
# from matplotlib.transforms import blended_transform_factory


def plot_density(
    sdatas: sd.SpatialData | list[sd.SpatialData], 
    genes: list | str | None,
    sample_key: str,
    isoform: str = None,
    sample_list: list = None,
    polygon: shapely.Polygon = None,
    shape_key: str = 'cell_boundaries',
    transcript_key: str = 'transcripts',
    table_key: str = 'table',
    nb_grid: int = 200j, 
    feature_key: str = 'feature_name',
    density_kde: bool = False,
    smooth: float = 1.0,
    pct_max: int = 99,
    bin_size_um: float = 10.0,
    box_bounds: dict | list | tuple = None,
    only_in_cell: bool = False,
    only_outside: bool = False,
    techno: str = "Xenium", # 'Xenium' or 'Merscope'
    clip_outside: bool = False,
    colorbar: str = None, #= "rows",  #"indiv", "rows", "global"
    by_codeword: bool = False,
    cmap = plt.cm.Reds,
    scale: bool = False, # str = 'microns', # pixels # True /False
    origin: str = 'upper', # or lower 
    hspace: float = 0.5, 
    wspace: float = 0.5,
    aggregate: bool = False,
    save: str | None = None,
    dpi: int = 300
):
    if not isinstance(sdatas, list): 
        sdatas = [sdatas]
    # if not isinstance(genes, list): 
    #     genes = [genes]
    # probleme avec genes d= none

    if density_kde:
        print(f"Using KDE density estimation with a smooth parameter of {str(smooth)}...")
    else:
        print("Using histogram density estimation...")

    if by_codeword and len(genes) != 1:
        raise ValueError("Please provide only one gene when using by_codeword.")
    
    if isoform and not colorbar:
        colorbar = 'global'
        print("Isoform plotting: setting colorbar to 'global'")
    elif genes and by_codeword:
        colorbar = 'global'
        print("Gene plotting by codeword: setting colorbar to 'global'")
    elif genes and not colorbar:
        colorbar = 'rows'
        print("Gene plotting: setting colorbar to 'rows'")
    elif colorbar:
        print(f"The colobar has been defined as '{colorbar}'")
    elif only_outside:
        colorbar = 'rows'
        print("Unassigned RNA plotting: setting colorbar to 'rows'")
    else:
        raise ValueError("Either 'genes' or 'isoform' must be provided.")
    
    res = {}
    global_vmax = 0

    for sdata in sdatas:
        if techno == "Merscope":
            shape_key = list(sdata.shapes.keys())[0]
            transcript_key = list(sdata.points.keys())[0]
        sample = sdata[table_key].obs[sample_key].unique()[0]

        if sample_list and sample in sample_list:
            if isinstance(box_bounds, dict):
                sample_box_bounds = box_bounds.get(sample) # None or list positions
            elif isinstance(box_bounds, (list, tuple)):
                sample_box_bounds = box_bounds
            else:
                sample_box_bounds = None
            
            if isoform:
                genes = list(list(
                    filter(lambda x: re.search(isoform, x), 
                    sdata[table_key].var_names)))
                print("Found those isoforms for sample", sample, ":", genes)

            results, _ = density_count_genes(    
                sdata=sdata, 
                genes=genes, 
                polygon=polygon, 
                shape_key=shape_key, 
                transcript_key=transcript_key,
                nb_grid=nb_grid,
                techno=techno,
                smooth=smooth,
                feature_key=feature_key,
                bin_size_um=bin_size_um, 
                box_bounds = sample_box_bounds,
                only_in_cell=only_in_cell,
                only_outside=only_outside,
                density_kde=density_kde,
                scale=scale,
                pct_max=pct_max,
                clip_outside=clip_outside,
                by_codeword=by_codeword,
                aggregate=aggregate,
                )
            vmax = max([v[1] for v in results.values()])
            global_vmax = max(global_vmax, vmax)
            res[sample] = results
    
    cmap.set_under('white')

    gene_names = list(list(res.values())[0].keys())
    
    if genes and (len(set(genes) - set(gene_names)) != 0) and (not aggregate) and (not only_outside):
        print(f"Warning: Some genes ({set(genes) - set(gene_names)}) were not found in the data and will be skipped")
    
    ncols = len(sample_list)
    nrows = len(gene_names)

    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    axes = [[None]*ncols for _ in range(nrows)]
    images = [[None]*ncols for _ in range(nrows)]

    if colorbar == 'rows':
        global_vmax = {}
        for gene in gene_names:
            vmax = max([res[sample][gene][1] for sample in sample_list])
            global_vmax[gene] = vmax

    for i, gene in enumerate(gene_names):          
        for j, sample in enumerate(sample_list):  
            heatmap = res[sample][gene][0]
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1)
            axes[i][j] = ax

            vmax = global_vmax[gene] if colorbar == 'rows' else global_vmax

            img = ax.imshow(
                heatmap,
                cmap=cmap,
                origin=origin, # upper or lower
                vmax=vmax, 
                vmin=0
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            if i == 0:
                ax.set_title(sample)
            if j == 0:
                ax.set_ylabel(gene, fontsize=12, rotation=90, labelpad=10)
                
            images[i][j] = img

    if colorbar == "global":
        fig.colorbar(
            images[0][0],
            ax=[ax for row in axes for ax in row],
            location="right",
            fraction=0.03,
            pad=0.02,
            label="Counts"
        )
    else: #  colorbar == "rows":
        for i, gene in enumerate(gene_names):
            fig.colorbar(
                images[i][0],
                ax=axes[i],
                location="right",
                fraction=0.03,
                pad=0.02,
                label="Counts"
        ) 
    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight", dpi=dpi)
    plt.show()



def BlendMatrix(
    n=10,
    col_threshold=0.5,
    two_colors=("#ff0000", "#00ff00"),
    background="black"
):
    """
    Python translation of Seurat's BlendMatrix.

    Returns:
        blend_matrix: (n, n, 4) RGBA array scaled 0-1 for matplotlib.imshow
    """

    if not (0 <= col_threshold <= 1):
        raise ValueError("col_threshold must be between 0 and 1")

    C0 = np.array(mcolors.to_rgba(background)) * 255
    C1 = np.array(mcolors.to_rgba(two_colors[0])) * 255
    C2 = np.array(mcolors.to_rgba(two_colors[1])) * 255

    blend_alpha = (C1[3] + C2[3]) / 2
    C0, C1, C2 = C0[:3], C1[:3], C2[:3]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def blend_color(i, j):
        c_min = sigmoid(5 * (1 / n - col_threshold))
        c_max = sigmoid(5 * (1 - col_threshold))
        _c1 = sigmoid(5 * (i / n - col_threshold))
        _c2 = sigmoid(5 * (j / n - col_threshold))
        _c0 = sigmoid(5 * ((i + j) / (2 * n) - col_threshold))
        c1_weight = (_c1 - c_min) / (c_max - c_min)
        c2_weight = (_c2 - c_min) / (c_max - c_min)
        c0_weight = (_c0 - c_min) / (c_max - c_min)

        C1_length = np.sqrt(np.sum((C1 - C0) ** 2))
        C2_length = np.sqrt(np.sum((C2 - C0) ** 2))
        C1_unit = (C1 - C0) / C1_length
        C2_unit = (C2 - C0) / C2_length
        C1_weight_vec = C1_unit * c1_weight
        C2_weight_vec = C2_unit * c2_weight

        C_blend = (
            C1_weight_vec * (i - 1) * C1_length / (n - 1)
            + C2_weight_vec * (j - 1) * C2_length / (n - 1)
            + (i - 1) * (j - 1) * c0_weight * C0 / (n - 1) ** 2
            + C0
        )

        C_blend = np.clip(C_blend, 0, 255)
        rgba = np.concatenate([C_blend, [blend_alpha]]) / 255

        return rgba

    blend_matrix = np.zeros((n, n, 4))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            blend_matrix[i - 1, j - 1] = blend_color(i, j)

    return blend_matrix


def draw_blend_matrix(
    ax,
    blend_img,
    xlabel="gene2",
    ylabel="gene1",
    wedge_color="black",
    fontsize=10,
):
    """
    Dessine une BlendMatrix sur un axe matplotlib.

    Toutes les annotations (wedges, labels) utilisent ax.transAxes
    (coordonnées 0→1), indépendantes de la taille du subplot ou des
    unités data de l'image. Zéro bidouillage nécessaire.

    Paramètres
    ----------
    ax         : matplotlib Axes sur lequel dessiner
    blend_img  : (n, n, 4) array retourné par BlendMatrix()
    xlabel     : label de l'axe X (gène en bas)
    ylabel     : label de l'axe Y (gène à gauche)
    wedge_color: couleur des flèches/wedges
    fontsize   : taille de police pour les labels
    
    """
    ax.imshow(blend_img, origin="lower", interpolation="nearest")
    ax.set_axis_off()

    fig = ax.get_figure()
    # ax.transAxes : (0,0)=bottom-left, (1,1)=top-right
    trans = ax.transAxes

    # Épaisseur et position des wedges en fraction de l'axe
    gap   = 0.04   # space between image and wedge start
    thick = 0.12   # thinkness wedge max (côté large)
    pad   = 0.04   # space betweene wedge and label

    # Wedge 
    x_tri = plt.Polygon(
        [(0, -(gap)),
         (1, -(gap + thick)),
         (1, -(gap))],
        transform=trans,
        clip_on=False,
        facecolor=wedge_color,
        linewidth=0,
    )
    ax.add_patch(x_tri)

    y_tri = plt.Polygon(
        [(-(gap),         0),
         (-(gap + thick), 1),
         (-(gap),         1)],
        transform=trans,
        clip_on=False,
        facecolor=wedge_color,
        linewidth=0,
    )
    ax.add_patch(y_tri)

    ax.text(
        0.5, -(gap + thick + pad),
        xlabel,
        transform=trans,
        ha="center", va="top",
        fontsize=fontsize,
        clip_on=False,
    )
    ax.text(
        -(gap + thick + pad), 0.5,
        ylabel,
        transform=trans,
        ha="right", va="center",
        rotation=90,
        fontsize=fontsize,
        clip_on=False,
    )


def colocalization(
    sdatas,
    genes: list,
    sample_key: str,
    sample_list: list,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    table_key: str = "table",
    pct_max: int = 99,
    only_in_cell: bool = False,
    bin_size_um: int = 20,
    
    scale: str="common", # or "independent"
    ncols: int = 4,
    channels: tuple = ('red', 'green'),
    figsize_per_subplots: tuple = (5,5),
    save: str | None = None,
    dpi: int = 300, 
    background: str = "black",
):
    """
    Plot colocalization of two genes across multiple samples.
    
    scale="common"      → divise les deux heatmaps par max(pct1, pct2).
                              Préserve l'abondance relative. Un gène 10× moins
                              exprimé sera 10× moins visible.
    scale="independent" → divise chaque heatmap par son propre percentile.
                              Met les deux gènes sur le même pied visuel.
                              Utile pour comparer des patterns spatiaux
                              indépendamment de l'expression absolue.
    """
    if not isinstance(sdatas, list):
        sdatas = [sdatas]

    if (not isinstance(genes, list)) or (len(genes) != 2):
        raise ValueError("Please provide exactly 2 genes in a list.")
    
    color_map = {
        "red": [0, "#ff0000"],
        "green": [1, "#00ff00"],
        "blue": [2, '#0000FF']
    }

    res = {}

    for sdata in sdatas:
        sample = sdata[table_key].obs[sample_key].unique()[0]

        if sample not in sample_list:
            continue
        
        heatmaps, _ = compute_coloc(
            sdata=sdata,
            genes=genes,
            transcript_key=transcript_key,
            feature_key=feature_key,
            table_key=table_key,
            only_in_cell=only_in_cell,
            bin_size_um=bin_size_um,
        )
        res[sample] = heatmaps

    # Plot
    if len(sample_list) == 1:
        ncols=2
    nrows = math.ceil((len(sample_list) + 1 )/ncols)
    
    figsize=(figsize_per_subplots[0]*ncols, figsize_per_subplots[1]*nrows)
    fig = plt.figure(figsize=figsize)

    for j, sample in enumerate(sample_list):
        heatmap1 = res[sample][genes[0]]
        heatmap2 = res[sample][genes[1]]

        # Indep percentile
        pct1 = np.max([1, np.nanpercentile(heatmap1, pct_max)]) 
        pct2 = np.max([1, np.nanpercentile(heatmap2, pct_max)]) 

        if scale == "common":
            denom1 = denom2 = max(pct1, pct2)
        elif scale == "independent":
            denom1, denom2 = pct1, pct2
        else:
            raise ValueError("scale must be 'common' or 'independent'")
    
        Z_A = np.clip(heatmap1 / denom1, 0, 1) # val > p95 => clip to 1 
        Z_B = np.clip(heatmap2 / denom2, 0, 1)

        # Normalize
        # Z_A_norm = heatmap1 / p95_A 
        # Z_B_norm = heatmap2 / p95_B 
        # or with max
        # Z_A_norm = heatmap1 / np.nanmax(heatmap1) 
        # Z_B_norm = heatmap2  / np.nanmax(heatmap2) 
        
        RGB = np.zeros((*heatmap1.shape, 3), dtype=float)
        RGB[..., color_map[channels[0]][0]] = Z_A 
        RGB[..., color_map[channels[1]][0]] = Z_B 

        if background == 'white':
            background_mask = (Z_A == 0) & (Z_B == 0)
            RGB[background_mask] = [1, 1, 1]

        ax = fig.add_subplot(nrows, ncols, j + 1)
        ax.imshow(RGB, origin="upper", rasterized=True, interpolation="nearest")
        ax.set_title(sample)
        ax.axis("off")
        # ax.set_rasterized(True)
    # fig.subplots_adjust(right=0.88)

    ax = fig.add_subplot(nrows, ncols, j + 2)
    blend_img = BlendMatrix(
        n=10, two_colors=(color_map[channels[0]][1], color_map[channels[1]][1]),
        background=background)
    draw_blend_matrix(
        ax, #axes[-1],
        blend_img,
        xlabel=genes[1],
        ylabel=genes[0],
        fontsize=12,
    )

    plt.tight_layout()

    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight", dpi=dpi) 
    plt.show()



def one_samp_colocalization(
    sdata,
    genes: list,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    pct_max: int = 99,
    only_in_cell: bool = False,
    bin_size_um: int = 20,
    figsize: tuple = (10, 6),
):
    if len(genes) != 2:
        raise ValueError("Please, provide 2 genes only !")
    else:
        if (not genes[0] in sdata['table'].var_names):
            raise ValueError(f"Gene 1 {genes[0]} is unvalid, please provide a valid gene name.")
        elif (not genes[1] in sdata['table'].var_names):
            raise ValueError(f"Gene 2 {genes[1]} is unvalid, please provide a valid gene name.")
    
    df_transcripts = subset_transcripts(
        sdata=sdata, 
        genes=genes, 
        only_in_cell = only_in_cell,
        transcript_key=transcript_key,
        return_gpd=True)

    multi = shapely.MultiPoint(df_transcripts.geometry.values)
    xmin, ymin, xmax, ymax = multi.bounds

    # GENE 1
    sub_genes = df_transcripts[df_transcripts[feature_key].isin([genes[0]])].copy()
    x = sub_genes['x']
    y = sub_genes['y']

    heatmap1, xx, yy = _count_dens(x, y, xmax, xmin, ymax, ymin, bin_size_um)
    vmax = np.max([1, np.percentile(heatmap1, pct_max)])
    
    # GENE 2
    sub_genes = df_transcripts[df_transcripts[feature_key].isin([genes[1]])].copy()
    x = sub_genes['x']
    y = sub_genes['y']

    heatmap2, xx, yy = _count_dens(x, y, xmax, xmin, ymax, ymin, bin_size_um)
    vmax = np.max([1, np.percentile(heatmap2, pct_max)])


    # Indep percentile
    p95_A = np.nanpercentile(heatmap1, pct_max)
    p95_B = np.nanpercentile(heatmap2, pct_max)

    # Normalize
    Z_A_norm = (heatmap1 - np.nanmin(heatmap1)) / (p95_A - np.nanmin(heatmap1))
    Z_B_norm = (heatmap2 - np.nanmin(heatmap2)) / (p95_B - np.nanmin(heatmap2))
    # or with max
    # Z_A_norm = (heatmap1 - np.nanmin(heatmap1)) / (np.nanmax(heatmap1) - np.nanmin(heatmap1))
    # Z_B_norm = (heatmap2 - np.nanmin(heatmap2)) / (np.nanmax(heatmap2) - np.nanmin(heatmap2))
    # or 
    # Z_A = np.clip(heatmap1 / p95_A, 0, 1)
    # Z_B = np.clip(heatmap2 / p95_B, 0, 1)

    RGB = np.zeros((*heatmap1.shape, 3), dtype=float)
    RGB[..., 0] = Z_A_norm  # Red
    RGB[..., 1] = Z_B_norm  # Green
    RGB[..., 2] = 0  # Blue

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[20, 1, 1], wspace=0.3)

    # Main image
    ax_main = plt.subplot(gs[0])
    img = ax_main.imshow(RGB,
            # extent=[xmin, xmax, ymin, ymax],
            origin='upper',
            vmax=0.5,
            )
    plt.axis("off")

    ax_main.set_title(f'Density map for {" - ".join(genes)}')
    ax_main.set_xlabel("X")
    ax_main.set_ylabel("Y")

    # Colorbar for Gene A
    ax_cbA = plt.subplot(gs[1])
    fig.colorbar(plt.cm.ScalarMappable(cmap='Reds'), 
                        cax=ax_cbA, 
                        label=f"Gene 1 : {genes[0]}",
                        shrink = 0.3)

    # Colorbar for Gene B
    ax_cbB = plt.subplot(gs[2])
    fig.colorbar(plt.cm.ScalarMappable(cmap='Greens'),
                        cax=ax_cbB, label=f"Gene 2 : {genes[1]}",
                        shrink = 0.3)

    plt.tight_layout()
    plt.show()