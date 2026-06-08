from .pp.filters import filter_xenium, filter_merscope
from .constants import XeniumKeys, MerscopeKeys

TECHNO_REGISTRY = {
    "Xenium": {
        "keys": XeniumKeys,
        "filter_fn": filter_xenium
    },
    "Merscope": {
        "keys": MerscopeKeys,
        "filter_fn": filter_merscope
    },
    # "Cosmx": CosmxKeys,  # to add,  
    # # Cosmx uses the same keys as Merscope
}