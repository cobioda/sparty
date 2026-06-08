class XeniumKeys:
    FEATURE_KEY = "feature_name"
    TRANSCRIPT_KEY = "transcript"
    QV_KEY = "qv"
    CELL_ID = "cell_id"
    IS_GENE = "is_gene"
    UNASSIGNED_CELL_ID = "UNASSIGNED"

class MerscopeKeys:
    FEATURE_KEY = "gene"
    TRANSCRIPT_KEY = "transcript_sample_name" # TO MODIFIED
    CELL_ID = "cell_id"


class CosmxKeys:
    FEATURE_KEY = "gene"
    TRANSCRIPT_KEY = "transcript_sample_name" # TO MODIFIED
    CELL_ID = "cell_id"

GENE_EXCLUDE_PATTERN = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*"

# GENE_EXCLUDE_PATTERN = "nan|<NA>|.*control.*|blank.*|antisense.*|unassigned.*|deprecated.*|intergenic.*|false.*|neg.*"
# VALID_DIMENSIONS = ("c", "y", "x")
# LOW_AVERAGE_COUNT = 0.01
# ATTRS_KEY = "spatialdata_attrs"