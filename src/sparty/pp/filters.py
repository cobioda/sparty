from ..constants import XeniumKeys, MerscopeKeys, GENE_EXCLUDE_PATTERN

def filter_xenium(
    df, 
    genes: list | None = None, 
    qv:int = 20, 
    only_in_cell: bool = True, 
    only_outside: bool = False,
    gene_exclude_pattern=GENE_EXCLUDE_PATTERN,
):
    mask = (
        (df[XeniumKeys.QV_KEY] >= qv)
        & (df[XeniumKeys.IS_GENE])
    )

    if genes:
        mask &= df[XeniumKeys.FEATURE_KEY].isin(genes)

    df = df[mask].dropna(subset=[XeniumKeys.FEATURE_KEY])

    if only_in_cell and only_outside:
        raise ValueError("Invalid combination: cannot be both 'only_in_cell' and 'only_outside'.")

    elif only_in_cell:
        df = df[df['cell_id'] != 'UNASSIGNED']

    elif only_outside:
        df = df[df['cell_id'] == 'UNASSIGNED']

    # if only_in_cell:
    #     df = df[df[XeniumKeys.CELL_ID] != XeniumKeys.UNASSIGNED_CELL_ID]

    df = df[
        ~df[XeniumKeys.FEATURE_KEY].str.contains(
            gene_exclude_pattern, regex=True)
    ]

    return df


def filter_merscope(
    df, 
    genes: list | None = None, 
    qv=None, 
    only_in_cell: bool = True,
    only_outside: bool = False, 
    gene_exclude_pattern=None,
):  
    if genes:
        df = df[df[MerscopeKeys.FEATURE_KEY].isin(genes)]

    if only_in_cell:
        df = df[(df[MerscopeKeys.CELL_ID] != -1.0) & (~df[MerscopeKeys.CELL_ID].isna())]

    return df