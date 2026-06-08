import seaborn as sns

def get_palette(color_key: str) -> dict:
    """Palette definition for specific projects.

    Parameters
    ----------
    color_key
        color key (might be 'group', 'population' or 'celltype').

    Returns
    -------
    Return palette dictionary.
    """
    if color_key == "group":
        palette = {"CTRL": "#006E82", "PAH": "#AA0A3C"}
    elif color_key == "population":
        palette = {"Endothelial": "#0077b6", "Epithelial": "#5e548e", "Immune": "#606c38", "Stroma": "#bb3e03"}
    elif color_key == "compartment":
        palette = {
            "cartilage nasal": "#fb8500",
            "vascular lymphatic": "#ef233c",
            "olfactory epithelium": "#344966",
            "migrating neuron": "#606c38",
        }
    elif color_key == "HTAP":
        palette = {
            # htap
            "AT2": "#3E8F91",
            "AT1": "#6F5D85",
            "Basal": "#E41A1C",
            "Multiciliated": "#1f618d",
            "Pre-TB secretory": "#3b683f",
            "Secretory": "#E6AB02",
            "AT0": "#BA6866",
            "AT1-AT2": "#F2920D",
            "Rare": "#DF8CC4",
            "EC general capillary": "#4EA2D7",
            "Plasma cells": "#78281f",
            "EC venous pulmonary": "#1E6275",
            "EC venous systemic": "#2FA679",
            "EC aerocyte capillary": "#95D286",
            "Lymphatic EC": "#2d7687",
            "EC arterial": "#C9CE46",
            "Smooth muscle": "#ec7063",
            "Alveolar fibroblasts": "#af801d",
            "Adventitial fibroblasts": "#D6217C",
            "Myofibroblasts": "#426F8E",
            "Pericytes": "#7b241c",
            "Mast cells": "#F79F80",
            "Alveolar macrophages": "#BC6399",
            "C1Q+ macrophages": "#B22070",
            "CD4 T cells": "#674A9C",
            "CD8 T cells": "#79838A",
            "B cells": "#668C61",
            "NK cells": "#8AA20A",
            "Monocytes": "#D1DC1F",
            "DC": "#AB674F",
            "Interstitial Mph perivascular": "#ff00a2",
            "Megakaryocytes": "#d68a1c",
        }
    elif color_key == "ann_level_2":
        # paolo
        palette = {
            "Cartilages": "#0B4B19",
            "Stromal0": "#99D6A9",
            "Stromal1": "#1B8F76",
            "Stromal2": "#9DAF07",
            "Osteoblasts": "#4CAD4C",
            "Progenitor cells": "#03045e",
            "Schwann cells": "#95ccff",
            "Lymphatic EC": "#F78896",
            "Vascular EC": "#E788C2",
            "Pericytes": "#BBD870",
            "Satellites": "#CB7647",
            "Skeletal muscle": "#926B54",
            "Neural crest": "#E3D9AC",
            "Olf. ensh. glia": "#cd6889",
            "Glia progenitors": "#FF4500",
            "ALK neurons": "#95819F",
            "NOS1 neurons": "#95819F",
            "Olfactory HBCs": "#E41A1C",
            "Respiratory HBCs": "#C82C73",
            "Olf. microvillars": "#efe13c",
            "Multiciliated": "#1f618d",
            "Deuterosomal": "#3498db",
            "Sustentaculars": "#C09ACA",
            "GBCs": "#F48B5A",
            "preOSNs": "#E69F00",
            "iOSNs": "#f05b43",
            "mOSNs": "#33b8ff",
            "Neural progenitors": "#6A0B78",
            "Excitatory neurons": "#706fd3",
            "Inhibitory neurons": "#800EF1",
            "GnRH neurons": "#2EECDB",
            "Myeloid": "#736376",
            "Microglia": "#91BFB7",

            #"Cycling HBCs": "#C2A523",
            #"Tufts": "#eb10fd",
            #"Duct": "#efe13c",
        }
    elif color_key == "ann_level_1":
        palette = {
            "Progenitor cells": "#03045e",
            "Olfactory epithelium": "#EF1B4F",
            "Respiratory epithelium": "#5562B7",
            "Neurons": "#6E5489",
            "Glial": "#919976",
            "Stroma": "#009E73",
            "Immune": "#2EECDB",
            "Vasculars": "#CC79A7",
            "Myocytes": "#803800",
            "Immune": "#736376",
            "Pericytes": "#BBD870",
        }
    elif color_key == "fluo":
        palette = {
            "blue": "#382aff",
            "green": "#82ff78",
            "purple": "#a900d7",
            "orange": "#ffa421",
            "darkblue": "#006b94",
            "red": "#c70015",
            "cyan": "#00b5b9",
            "brown": "#954600",
            "yellow": "#e9ffae",
            "pink": "#ff9eda",
        }
    elif color_key == "leiden":  # default is 40 colors returned
        l = list(range(0, 39, 1))
        ll = list(map(str, l))
        palette = dict(zip(ll, sns.color_palette("husl", 40).as_hex()))

    palette["others"] = "#ffffff"

    return palette