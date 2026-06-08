import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon
import math
import numpy as np


def plot_cells_dict(
    cells: dict,
    nucls: dict,
    positions: dict | None = None,
    gene_name: str | None = None,
    ncols: int = 4,
    marker='o',
    figsize_per_subplot=(5, 5),
):
    """
    Plots cells, nuclei, and optional positions from dictionaries.

    Parameters
    ----------
    cells : dict
        {cell_id: Polygon or array-like of (x, y)}
    nucls : dict
        {cell_id: Polygon or array-like of (x, y)}
    positions : dict, optional
        {cell_id: list of (x, y)}
    ncols : int
        Number of columns in subplot grid
    figsize_per_subplot : tuple
        Size of each subplot
    """

    # cellules communes (sécurité)
    cell_ids = sorted(set(cells.keys()) & set(nucls.keys()))
    n = len(cell_ids)

    if n == 0:
        raise ValueError("No common cell_id between cells and nucls")

    nrows = math.ceil(n / ncols)
    figsize = (figsize_per_subplot[0] * ncols,
               figsize_per_subplot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, cell_id in enumerate(cell_ids):
        ax = axes[i]

        cell = cells[cell_id]
        nucl = nucls[cell_id]

        cell_poly = Polygon(cell) if not isinstance(cell, Polygon) else cell
        nucl_poly = Polygon(nucl) if not isinstance(nucl, Polygon) else nucl

        _, points1 = plot_polygon( # patch1, points1
            cell_poly,
            ax=ax,
            linewidth=2,
            edgecolor='#75747A',
            color='#75747A',
            facecolor='#FED9AF',
            add_points=True
        )

        _, points2 = plot_polygon( # patch2, points2
            nucl_poly,
            ax=ax,
            linewidth=1,
            edgecolor='#5D3E1E',
            color='#5D3E1E',
            facecolor='#D1EAC3',
            add_points=True
        )

        points1.set_markersize(5)
        points2.set_markersize(3)

        if positions is not None and cell_id in positions:
            pos = np.asarray(positions[cell_id])
            ax.scatter(pos[:, 0], pos[:, 1],
                s=20,
                c="#EF3E36", #"#EF3E36","#1904FF" #'#336699' ,
                zorder=5,
                marker=marker,
                label="positions"
            )
        ax.set_title(cell_id, fontsize=12)
        ax.set_aspect("equal")
        ax.set_axis_off()

    # Remove unused axes
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    if gene_name is not None:
        fig.suptitle(
            gene_name,
            fontsize=18,
            fontweight="bold",
            y=1.02
        )
    plt.tight_layout(rect=[0, 0, 1, 1])
    # plt.tight_layout()
    # fig.set_constrained_layout(True)

    plt.show()


def sub_positions_of_gene(
    df,
    gene,
    feature_key = 'feature_name',
    cell_id = 'cell_id',
    coordinates = ('x', 'y'),
) -> dict:
    x = coordinates[0]
    y = coordinates[1]

    sub_gene = df[df[feature_key] == gene]
    cell_dict = {
        cell: list(zip(group[x], group[y]))
        for cell, group in sub_gene.groupby(cell_id)
    }
    return cell_dict


def plot_gene_in_cells(
    sdata,
    gene: str,
    group: str,
    list_of_cells: list = None,
    groupby: str = 'cell_type',
    cell_id: str = 'cell_id',
    shape_cells_key: str = 'cell_boundaries',
    shape_nucleus_key: str = 'nucleus_boundaries',
    feature_key: str = 'feature_name',
    transcript_key: str = 'transcripts',
    table_key: str = 'table',  
    qv: int = 20,
    max_cells_to_plot: int = 50,
    ncols: int = 4,
    marker:str = 'o',
    figsize_per_subplot:tuple = (5, 5),
    # GENE_EXCLUDE_PATTERN = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
):
    if list_of_cells is None:
        list_of_cells = sdata[table_key].obs.loc[sdata[table_key].obs[groupby] == group, cell_id].tolist()

    if len(list_of_cells) > max_cells_to_plot:
        print(f"The group contains {len(list_of_cells)} cells and max_cells_to_plot has been set to {max_cells_to_plot}. We will randomly select cells to plot.")
        list_of_cells = np.random.choice(list_of_cells, size=max_cells_to_plot, replace=False)
    
    df_transcripts = sdata[transcript_key][
        (sdata[transcript_key]['qv'] >= qv) &
        (sdata[transcript_key][cell_id].isin(list_of_cells)) &
        (sdata[transcript_key][feature_key] == gene)
        ].dropna(subset=feature_key).compute()
    # df_transcripts = df_transcripts[
    #         ~df_transcripts[feature_key].str.contains(GENE_EXCLUDE_PATTERN, regex=True)
    #     ]
    df_transcripts[feature_key] = df_transcripts[feature_key].cat.remove_unused_categories()
    print(len(df_transcripts[feature_key].unique()))

    cell_dict = sub_positions_of_gene(
        df=df_transcripts,
        gene=gene)

    sub_shape_cells = sdata[shape_cells_key].loc[list_of_cells]['geometry'].to_dict()
    sub_shape_nucleus = sdata[shape_nucleus_key].loc[list_of_cells]['geometry'].to_dict()
    # {cell_id: polygon} 

    plot_cells_dict(
        cells=sub_shape_cells,     
        nucls=sub_shape_nucleus,   
        positions=cell_dict,
        gene_name=gene,
        ncols=ncols,
        marker=marker,
        figsize_per_subplot=figsize_per_subplot
    )


def plot_cells_list(
    cells, 
    nucls, 
    ncols = 4,
    figsize_per_subplot=(5, 5),
    # figsize=(15, 5),
):
    """
    Plots a list of cells and nuclei in subplots.
    
    Parameters:
    - cells: list of arrays/lists/Polygons
    - nucls: list of arrays/lists/Polygons
    - ncols: number of columns in the grid
    - figsize_per_subplot: size of each subplot (tuple)
    #- figsize: figure size (tuple)
    """
    n = len(cells)
    nrows = math.ceil(n / ncols)
    # nrows = n // ncols + (1 if n % ncols != 0 else 0)

    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (cell, nucl) in enumerate(zip(cells, nucls)):
        ax = axes[i]
        
        # Convert to Polygon if needed
        if isinstance(cell, (np.ndarray, list, tuple)):
            cell_poly = Polygon(cell)
        else:
            cell_poly = cell

        if isinstance(nucl, (np.ndarray, list, tuple)):
            nucl_poly = Polygon(nucl)
        else:
            nucl_poly = nucl
        
        # Plot polygons
        patch1, points1 = plot_polygon(
            cell_poly,
            ax=ax,
            linewidth=2,
            edgecolor='#75747A',
            color='#75747A',
            facecolor='#FED9AF',
            add_points=True
        )
        patch2, points2 = plot_polygon(
            nucl_poly,
            ax=ax,
            linewidth=1,
            edgecolor='#5D3E1E',
            color='#5D3E1E',
            facecolor='#D1EAC3',
            add_points=True
        )
        points1.set_markersize(5)
        points2.set_markersize(3)

        ax.set_axis_off()
        ax.set_aspect('equal')

    # Turn off any extra axes if total plots < nrows*ncols
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_shapes(shapes: dict, ncols: int = 4):
    """
    Plot shapely polygons stored in a dictionary.

    Parameters
    ----------
    shapes : dict
        Dictionary {shape_name: shapely.Polygon}
    ncols : int
        Number of columns for subplot grid.
    """
    n_shapes = len(shapes)
    nrows = n_shapes // ncols + (n_shapes % ncols > 0)

    plt.figure(figsize=(5 * ncols, 5 * nrows))
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    for i, (name, polygon) in enumerate(shapes.items(), start=1):
        x, y = polygon.exterior.xy
        ax = plt.subplot(nrows, ncols, i)
        ax.plot(x, y, linewidth=2)
        ax.set_title(name, fontsize=20)
        # ax.set_aspect('equal')
        # ax.axis('off')          # Turn off axes ticks and labels
        # ax.set_frame_on(True)  # Remove the box around the plot
    plt.show()


def plot_shape(
    shape, 
    title = "",
    only_max = False,
    plot_points = False,
    return_y_axis = True,
    figsize=(5,5),
):  
    """
    Plot one shape
    """
    _, ax = plt.subplots(figsize=figsize)
    
    if (shape.geom_type == 'MultiPolygon') &  (only_max):
        shape = max(shape.geoms, key=lambda g: g.area)
    
    if shape.geom_type == 'Polygon':
        x, y = shape.exterior.xy
        ax.plot(x, y, color="blue")
        ax.fill(x, y, alpha=0.3, color="lightblue")

        if plot_points:
            ax.scatter(x, y, c="blue", s=5)
    elif shape.geom_type == 'MultiPolygon':
        for pol in shape.geoms:
            x, y = pol.exterior.xy
            ax.plot(x, y, color="blue")
            ax.fill(x, y, alpha=0.3, color="lightblue")

            if plot_points:
                ax.scatter(x, y, c="blue", s=5)

    if return_y_axis:
        ax.invert_yaxis()
    
    plt.title(title)
    plt.show()



def plot_cells(
    cell1, 
    cell2, 
    figtitle,
    figsize=(6,6), 
    pt_cell=25, 
    pt_nucl=7
):
    plt.figure(figsize=figsize)
    plt.plot(cell1[:pt_cell,0], cell1[:pt_cell,1], color='blue', label='cell1')
    plt.plot(cell2[:pt_cell,0], cell2[:pt_cell,1], color='red', label='cell2')
    
    plt.plot(cell1[-pt_nucl:,0], cell1[-pt_nucl:,1], color='blue', marker='x', label='nucl1')
    plt.plot(cell2[-pt_nucl:,0], cell2[-pt_nucl:,1], color='red', marker='x', label='nucl2')

    # plt.axhline(0, color='grey', lw=1)
    # plt.axvline(0, color='grey', lw=1)
    plt.legend()
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.title(figtitle)
    plt.show()




def plot_cell(
    cell,
    nucl,
    figsize=(5,5),
):
    if isinstance(cell, (np.ndarray, list, tuple)):
        cell_poly = Polygon(cell)
    else:
        cell_poly = cell

    if isinstance(nucl, (np.ndarray, list, tuple)):
        nucl_poly = Polygon(nucl)
    else:
        nucl_poly = nucl

    plt.figure(figsize=figsize)

    patch1, points1 = plot_polygon(
        cell_poly, 
        edgecolor= '#75747A',
        linewidth=2, 
        color='#75747A',
        facecolor = '#FED9AF',
        )
    patch2, points2 = plot_polygon(
        nucl_poly, 
        edgecolor= '#5D3E1E',
        linewidth=1, 
        color='#5D3E1E',
        facecolor = '#D1EAC3',
        )
    points1.set_markersize(5)
    points2.set_markersize(3)

    ax = plt.gca()
    ax.set_axis_off()          # hides x & y axes
    plt.axis('off')
    plt.show()
    # ax.set_frame_on(False)     # removes border frame
    # plt.margins(0)             # removes padding
