import numpy as np
import matplotlib.pyplot as plt
import shapely
import geopandas as gpd
import spatialdata as sd
import seaborn as sns
from matplotlib.path import Path
from scipy.stats import gaussian_kde

# from scipy.stats import gaussian_kde
# import pandas as pd 

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

    # Create a regular grid over the area
    yy, xx = np.mgrid[ymin:ymax:nb_grid, xmin:xmax:nb_grid]  # 100x100 grid
    positions = np.vstack([xx.ravel(),yy.ravel()])
    density = np.reshape(kde(positions), xx.shape)

    return density, xx, yy


def _subset_transcripts(
    sdata,
    genes: list,
    qv: int = 20,
    transcript_key: str= "transcript",
    copy: bool = True,
    feature_key: str = 'feature_name',
    gene_exclude_pattern = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
):
    if copy: 
        df_transcripts = sdata[transcript_key].copy()
    else:
        df_transcripts = sdata[transcript_key]

    df_transcripts = df_transcripts[(df_transcripts['qv'] >= qv) & 
                                    (df_transcripts.is_gene) & 
                                    (df_transcripts.cell_id != "UNASSIGNED") &
                                    (df_transcripts[feature_key].isin(genes))
                                    ].dropna(subset=[feature_key])
    df_transcripts = df_transcripts[~(df_transcripts[feature_key].str.contains(gene_exclude_pattern, regex=True))].compute()
    df_transcripts[feature_key] = df_transcripts[feature_key].cat.remove_unused_categories()
    return df_transcripts


def density_count_genes(
    sdata: sd.SpatialData,
    genes: list | str,
    polygon: shapely.Polygon = None,
    shape_key: str = "cell_boundaries",
    transcript_key: str = 'transcripts',
    nb_grid: int = 200j, 
    smooth: float = 1.0,
    feature_key: str = 'feature_name',
    bin_size_um: float = 10.0,
    figsize: tuple = (8,8),
    density_kde: bool = False,
    pct_max: int = 99,
    cmap = plt.cm.viridis,
    clip_outside: bool = False,
    ax = None, 
):
    data = _subset_transcripts(
        sdata=sdata,
        genes=genes,
        feature_key=feature_key,
        transcript_key= transcript_key
        )   
    if not isinstance(data, gpd.GeoDataFrame):
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['x'], data['y']))

    if not polygon and shape_key:
        xmin, ymin, xmax, ymax = sdata[shape_key].total_bounds
        polygon = shapely.geometry.box(xmin, ymin, xmax, ymax)
    elif polygon:
        xmin, ymin, xmax, ymax = polygon.bounds 
        if clip_outside:
            data = data[data.within(polygon)]
    
    x = data.geometry.centroid.x
    y = data.geometry.centroid.y

    if density_kde:
        print('kde')
        heatmap, xx, yy = _kde_dens(x, y, xmax, xmin, ymax, ymin, nb_grid, smooth)
        vmax = None
    else:
        print('histo')
        heatmap, xx, yy = _count_dens(x, y, xmax, xmin, ymax, ymin, bin_size_um )
        vmax = np.percentile(heatmap, pct_max)
    vmax=heatmap.max()
    print(vmax)
    if clip_outside:
        grid_points = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()))
        mask = grid_points.within(polygon).to_numpy().reshape(heatmap.shape)
        heatmap = np.where(mask, heatmap, np.nan) # or -1

    cmap.set_under('white')  

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Density plot - {':'.join(genes)}")
        ax.plot(
            shapely.get_coordinates(polygon.boundary)[:, 0], 
            shapely.get_coordinates(polygon.boundary)[:, 1], 'r-', linewidth=2) 
        img = ax.imshow(
            heatmap,
            cmap=cmap,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            vmax=vmax
        )
        plt.colorbar(img, ax=ax, label="Counts", shrink=0.5)
        plt.show()
    else: 
        ax.plot(
            shapely.get_coordinates(polygon.boundary)[:, 0], 
            shapely.get_coordinates(polygon.boundary)[:, 1], 'r-', linewidth=2) 
        img = ax.imshow(
            heatmap,
            cmap=cmap,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            vmax=vmax
        )
        # ax.set_title("Heatmap brute - Comptes de points par pixel")
        # plt.colorbar(img, ax=ax, label="Counts", shrink=0.5)
        return img, vmax
        


# def subset_transcripts(
#     sdata,
#     genes: list,
#     qv: int = 20,
#     transcript_key: str= "transcript",
#     copy: bool = True,
#     feature_key: str = 'feature_name',
#     gene_exclude_pattern = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
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
#     # print(df_transcripts.shape)
#     return df_transcripts


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
        
