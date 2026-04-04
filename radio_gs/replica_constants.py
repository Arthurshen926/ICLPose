"""Authoritative Replica semantic class mapping.

Built from exhaustive visual inspection of semantic_class PNG mask overlays
across room_0, room_1, and room_2. These are Replica-native sparse IDs,
NOT NYU40 contiguous indices.

Each ID was verified by generating per-class mask overlays (red-tinted
regions on darkened RGB) and visually identifying the highlighted object
category across multiple rooms and viewpoints.
"""

import numpy as np

# ── Replica semantic class ID → human-readable name ──────────────────────────
# IDs are sparse (0..98) as stored in semantic_class_*.png files.
REPLICA_CLASSES = {
    0:  "undefined",
    3:  "ottoman",
    7:  "headboard",
    11: "blanket",
    12: "blinds",
    13: "vase",
    14: "ceiling light",
    15: "box",
    16: "plate",
    18: "chest of drawers",
    19: "figurine",
    20: "table",
    26: "bedding",
    29: "door",
    31: "ceiling",
    37: "nightstand",
    40: "chair",
    44: "plant",
    47: "cabinet",
    54: "floor",
    59: "picture",
    60: "mirror",
    61: "pillow",
    63: "candle",
    64: "book",
    65: "lantern",
    70: "sculpture",
    71: "bookcase",
    76: "sofa",
    78: "stool",
    79: "light switch",
    80: "tv",
    91: "pot",
    92: "lamp",
    93: "wall",
    95: "outlet",
    97: "window",
    98: "rug",
}

# ── Text grounding queries → class IDs ───────────────────────────────────────
# Only major, visually distinct classes suitable for text grounding.
# Query names must exist in the SigLIP2 text embedding bank.
# Ordered by class ID for deterministic evaluation.
GROUNDING_QUERIES = {
    "blanket":          11,
    "table":            20,
    "door":             29,
    "ceiling":          31,
    "chair":            40,
    "plant":            44,
    "cabinet":          47,
    "picture":          59,
    "mirror":           60,
    "sofa":             76,
    "television":       80,
    "lamp":             92,
    "wall":             93,
    "window":           97,
    "rug":              98,
}

# ── Per-class colors for segmentation visualization ──────────────────────────
# Deterministic random colors seeded for reproducibility.
SEG_COLORS = {}
_rng = np.random.RandomState(42)
for _cid in REPLICA_CLASSES:
    if _cid == 0:
        SEG_COLORS[_cid] = (40, 40, 40)
    else:
        SEG_COLORS[_cid] = tuple(_rng.randint(60, 255, 3).tolist())

# ── Per-room class ID sets (for filtering scene-specific queries) ────────────
ROOM_CLASS_IDS = {
    "room_0": {0, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47,
               59, 60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98},
    "room_1": {0, 3, 7, 11, 12, 13, 18, 26, 31, 37, 40, 44, 47, 54,
               59, 61, 64, 79, 91, 92, 93, 95, 97, 98},
    "room_2": {0, 12, 14, 15, 16, 20, 31, 37, 40, 44, 47, 64, 70, 71,
               79, 80, 91, 92, 93, 95, 97, 98},
}
