"""ScanNet semantic classes and class-name query fallback."""

import numpy as np


SCANNET20_CLASSES = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "bookshelf",
    10: "picture",
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "shower curtain",
    16: "toilet",
    17: "sink",
    18: "bathtub",
    19: "otherfurniture",
}

GROUNDING_QUERIES = {
    "cabinet": 2,
    "bed": 3,
    "chair": 4,
    "sofa": 5,
    "table": 6,
    "door": 7,
    "window": 8,
    "bookshelf": 9,
    "picture": 10,
    "counter": 11,
    "desk": 12,
    "curtain": 13,
    "refrigerator": 14,
    "toilet": 16,
    "sink": 17,
    "bathtub": 18,
}

SEG_COLORS = {}
_rng = np.random.RandomState(7)
for _cid in SCANNET20_CLASSES:
    SEG_COLORS[_cid] = tuple(_rng.randint(60, 255, 3).tolist())
