"""Mapping-only learned projection for chart-local RADIO descriptors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lineage import arrays_sha256, canonical_json_sha256


def load_chart_local_radio_projection(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {"weight": np.asarray(data["weight"], np.float32)}
    weight = arrays["weight"]
    content = dict(metadata)
    claimed_content = content.pop("content_sha256", None)
    if (
        metadata.get("artifact_type") != "goal_maplet_chart_local_radio_projection_v1"
        or metadata.get("query_pose_depth_or_ground_truth_read") is not False
        or metadata.get("mapping_rgb_stored") is not False
        or metadata.get("source_view_identity_retained_at_runtime") is not False
        or canonical_json_sha256(content) != claimed_content
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or weight.ndim != 2
        or weight.shape != (
            int(metadata.get("output_dimension", -1)),
            int(metadata.get("input_dimension", -1)),
        )
        or not np.all(np.isfinite(weight))
    ):
        raise ValueError("chart-local RADIO projection contract differs")
    return weight, metadata


def project_chart_local_radio(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    value = np.asarray(features, np.float32)
    matrix = np.asarray(weight, np.float32)
    if value.ndim != 2 or matrix.ndim != 2 or value.shape[1] != matrix.shape[1]:
        raise ValueError("RADIO feature/projection dimensions differ")
    projected = value @ matrix.T
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    if not np.all(np.isfinite(projected)):
        raise ValueError("projected RADIO descriptors are not finite")
    return projected.astype(np.float32)


__all__ = ["load_chart_local_radio_projection", "project_chart_local_radio"]
