import numpy as np
from spatialdata.transformations import Identity, Scale, Sequence
# import dask.dataframe as dd

def _transform_coords(ddf, M):
    """Transformation matricielle pour une partition Dask."""
    coords = np.vstack([ddf['x'], ddf['y'], np.ones(len(ddf))])
    transformed = M @ coords
    ddf['x'] = transformed[0]
    ddf['y'] = transformed[1]
    return ddf


def _to_affine_shapely(M):
    a, b, xoff = M[0]
    d, e, yoff = M[1]
    return (a, b, d, e, xoff, yoff)


def compute_bounds_dask(transcripts, transfo, scale=False):
    """
    Applique une transformation à un Dask DataFrame si nécessaire.
    """
    def apply_affine_dask(transcripts, transfo):
        M = transfo.to_affine_matrix(
            input_axes=("x", "y"), 
            output_axes=("x", "y")) # NO .T transpose else we get issue !! .T # .T transpose or not ???
        return transcripts.map_partitions(_transform_coords, M)

    if not (isinstance(transfo, Identity) or isinstance(transfo, Scale) and not scale):
        if isinstance(transfo, Sequence):
            if scale:
                return apply_affine_dask(transcripts, transfo)
            else:
                # Ignorer les Scale dans la sequence
                new_transfo = [t for t in transfo.transformations if not isinstance(t, Scale)]
                if len(new_transfo) == 1:
                    transfo_to_apply = new_transfo[0]
                elif len(new_transfo) > 1:
                    transfo_to_apply = Sequence(new_transfo)
                else:
                    transfo_to_apply = None  

                if transfo_to_apply is not None:
                    return apply_affine_dask(transcripts, transfo_to_apply)
        else:
            return apply_affine_dask(transcripts, transfo)
    return transcripts


def compute_bounds_gpd(shape, transfo, scale=False):
    """
    Calcule les bounds xmin, ymin, xmax, ymax selon la transformation appliquée.
    """
    # IF apply_affine is only used with compute bounds ==> stay here !! 
    # otherwise ==> new functions outside compute_bounds ++
    def apply_affine_gpd(shape, transfo):
        """Applique une transformation affine à un GeoDataFrame / GeoSeries."""
        M = transfo.to_affine_matrix(
            input_axes=("x", "y"), 
            output_axes=("x", "y")) # .T transpose or not ???
        A = _to_affine_shapely(M)
        return shape.affine_transform(A).total_bounds

    if (isinstance(transfo, Identity)) or (isinstance(transfo, Scale) and not scale):
        return shape.total_bounds

    elif isinstance(transfo, Sequence):
        if scale:
            return apply_affine_gpd(shape, transfo)
        else:
            # ignored scale in sequence
            new_tr = [t for t in transfo.transformations if not isinstance(t, Scale)]
            if len(new_tr) == 0:
                return shape.total_bounds
            elif len(new_tr) == 1:
                transfo_to_apply = new_tr[0]
            else:
                transfo_to_apply = Sequence(new_tr)
            return apply_affine_gpd(shape, transfo_to_apply)
    else:
        return apply_affine_gpd(shape, transfo)

