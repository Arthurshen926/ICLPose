"""Small, deterministic artifact-lineage helpers for Goal-Maplet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


FORBIDDEN_DEPLOYMENT_FLAGS = (
    "stores_mapping_rgb",
    "stores_mapping_image_paths",
    "stores_mapping_image_ids",
    "uses_alike_descriptors",
    "uses_radio_intermediate",
    "uses_sfm_points",
    "uses_sfm_tracks",
    "uses_point_correspondences",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays independent of NPZ compression and dictionary order."""

    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_deployment_metadata(metadata: Mapping[str, object]) -> None:
    for key in FORBIDDEN_DEPLOYMENT_FLAGS:
        if bool(metadata.get(key, False)):
            raise ValueError(f"Goal-Maplet deployment artifact violates contract: {key}")
    if metadata.get("vfm_layer", "radio_final") != "radio_final":
        raise ValueError("Goal-Maplet permits only RADIO-final canonical features")

