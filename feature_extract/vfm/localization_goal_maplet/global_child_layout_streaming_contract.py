"""Artifact contract for parent/child/fusion global streaming layout scores."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from .lineage import arrays_sha256
from .streaming_child_layout_guide_gpu import MODES


SCORE_SCHEMA = "goal_maplet_global_hierarchy_layout_streaming_score_v1"
SHARD_RUN_SCHEMA = "goal_maplet_global_hierarchy_layout_streaming_phase1_shard_v1"
RUN_SCHEMA = "goal_maplet_global_hierarchy_layout_streaming_phase1_run_v1"
GATE_SCHEMA = "goal_maplet_global_hierarchy_layout_streaming_gpu_gate_v1"
TOPK = 4096
MAXIMUM_QUERY_PARENTS = 32
MAXIMUM_SCENE_CHILDREN = 64
POSITION_CHUNK_SIZE = 1024
TORCH_DTYPE = "float64"

BASE_ARRAY_NAMES = (
    "image_id", "selected_parent_ids", "selected_child_rows",
)
MODE_SUFFIXES = (
    "top_scores", "top_position_factor_indices", "top_orientation_factor_indices",
    "top_parent_scores", "top_child_scores", "top_parent_visible_counts",
    "top_child_visible_counts", "top_child_front_facing_counts",
    "top_child_positive_depth_counts", "top_child_center_in_image_counts",
    "top_child_projected_token_footprint_mass", "top_child_sqrt_overlap_mass",
)
SCORE_ARRAY_NAMES = BASE_ARRAY_NAMES + tuple(
    f"{mode}_{suffix}" for mode in MODES for suffix in MODE_SUFFIXES
)


def hierarchy_result_arrays(image_id: str, result: object) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "image_id": np.asarray(str(image_id)),
        "selected_parent_ids": np.asarray(result.selected_parent_ids, dtype=np.int64),
        "selected_child_rows": np.asarray(result.selected_child_rows, dtype=np.int64),
    }
    source_names = {
        "top_scores": "top_scores",
        "top_position_factor_indices": "top_position_rows",
        "top_orientation_factor_indices": "top_orientation_rows",
        "top_parent_scores": "top_parent_scores",
        "top_child_scores": "top_child_scores",
        "top_parent_visible_counts": "top_parent_visible_counts",
        "top_child_visible_counts": "top_child_visible_counts",
        "top_child_front_facing_counts": "top_child_front_facing_counts",
        "top_child_positive_depth_counts": "top_child_positive_depth_counts",
        "top_child_center_in_image_counts": "top_child_center_in_image_counts",
        "top_child_projected_token_footprint_mass": (
            "top_child_projected_token_footprint_mass"
        ),
        "top_child_sqrt_overlap_mass": "top_child_sqrt_overlap_mass",
    }
    for mode in MODES:
        ranking = result.rankings[mode]
        for suffix, source in source_names.items():
            value = np.asarray(getattr(ranking, source))
            if suffix.endswith("indices"):
                value = value.astype(np.int64, copy=False)
            elif suffix.endswith("counts"):
                value = value.astype(np.int16, copy=False)
            else:
                value = value.astype(np.float64, copy=False)
            arrays[f"{mode}_{suffix}"] = value
    if set(arrays) != set(SCORE_ARRAY_NAMES):
        raise AssertionError("hierarchy score array construction differs")
    return arrays


def hierarchy_topk_arrays_sha256(result: object) -> str:
    arrays = hierarchy_result_arrays("hash-placeholder", result)
    arrays.pop("image_id")
    arrays.pop("selected_parent_ids")
    arrays.pop("selected_child_rows")
    return arrays_sha256(arrays)


def load_hierarchy_score(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifact = Path(path)
    expected = {*(f"{name}.npy" for name in SCORE_ARRAY_NAMES), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("hierarchy streaming score is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("hierarchy streaming score exact NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*SCORE_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("hierarchy streaming score arrays differ")
        arrays = {name: np.asarray(data[name]) for name in SCORE_ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if (
        metadata.get("artifact_type") != SCORE_SCHEMA
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or arrays["image_id"].shape != ()
        or arrays["selected_parent_ids"].dtype != np.int64
        or arrays["selected_parent_ids"].shape != (MAXIMUM_QUERY_PARENTS,)
        or np.unique(arrays["selected_parent_ids"]).size != MAXIMUM_QUERY_PARENTS
        or arrays["selected_child_rows"].dtype != np.int64
        or arrays["selected_child_rows"].shape != (MAXIMUM_SCENE_CHILDREN,)
        or np.unique(arrays["selected_child_rows"]).size != MAXIMUM_SCENE_CHILDREN
        or metadata.get("maximum_query_parents") != MAXIMUM_QUERY_PARENTS
        or metadata.get("maximum_scene_children") != MAXIMUM_SCENE_CHILDREN
        or metadata.get("returned_topk") != TOPK
        or metadata.get("position_chunk_size") != POSITION_CHUNK_SIZE
        or metadata.get("torch_dtype") != TORCH_DTYPE
        or metadata.get("phase2_labels_opened") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("parent_score_semantics") is None
        or metadata.get("child_score_semantics") is None
        or metadata.get("child_geometry_contract") is None
        or metadata.get("fusion_semantics") is None
        or metadata.get("streaming_semantics") is None
    ):
        raise ValueError("hierarchy streaming score contract differs")
    for mode in MODES:
        score = arrays[f"{mode}_top_scores"]
        position = arrays[f"{mode}_top_position_factor_indices"]
        orientation = arrays[f"{mode}_top_orientation_factor_indices"]
        if (
            score.dtype != np.float64 or score.shape != (TOPK,)
            or position.dtype != np.int64 or position.shape != score.shape
            or orientation.dtype != np.int64 or orientation.shape != score.shape
            or np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0 + 1e-12))
            or np.unique(np.stack([position, orientation], axis=1), axis=0).shape[0]
            != TOPK
            or np.any(position < 0) or np.any(position >= int(metadata["position_count"]))
            or np.any(orientation < 0)
            or np.any(orientation >= int(metadata["orientation_count"]))
            or not np.array_equal(
                np.lexsort((orientation, position, -score)), np.arange(TOPK),
            )
        ):
            raise ValueError(f"hierarchy {mode} Top-K contract differs")
        for suffix in ("top_parent_scores", "top_child_scores"):
            value = arrays[f"{mode}_{suffix}"]
            if (
                value.dtype != np.float64 or value.shape != score.shape
                or np.any(~np.isfinite(value))
                or np.any((value < 0.0) | (value > 1.0 + 1e-12))
            ):
                raise ValueError(f"hierarchy {mode} component scores differ")
        for suffix, maximum in (
            ("top_parent_visible_counts", MAXIMUM_QUERY_PARENTS),
            ("top_child_visible_counts", MAXIMUM_SCENE_CHILDREN),
            ("top_child_front_facing_counts", MAXIMUM_SCENE_CHILDREN),
            ("top_child_positive_depth_counts", MAXIMUM_SCENE_CHILDREN),
            ("top_child_center_in_image_counts", MAXIMUM_SCENE_CHILDREN),
        ):
            value = arrays[f"{mode}_{suffix}"]
            if (
                value.dtype != np.int16 or value.shape != score.shape
                or np.any(value < 0) or np.any(value > maximum)
            ):
                raise ValueError(f"hierarchy {mode} visibility counts differ")
        for suffix in (
            "top_child_projected_token_footprint_mass", "top_child_sqrt_overlap_mass",
        ):
            value = arrays[f"{mode}_{suffix}"]
            if value.dtype != np.float64 or value.shape != score.shape or np.any(~np.isfinite(value)) or np.any(value < 0.0):
                raise ValueError(f"hierarchy {mode} child diagnostics differ")
        parent_component = arrays[f"{mode}_top_parent_scores"]
        child_component = arrays[f"{mode}_top_child_scores"]
        expected_score = {
            "parent": parent_component,
            "child": child_component,
            "geometric_mean": np.sqrt(parent_component * child_component),
        }[mode]
        if not np.allclose(score, expected_score, atol=1.0e-12, rtol=0.0):
            raise ValueError(f"hierarchy {mode} score/component algebra differs")
    return arrays, metadata


__all__ = [
    "GATE_SCHEMA", "MAXIMUM_QUERY_PARENTS", "MAXIMUM_SCENE_CHILDREN", "MODES",
    "POSITION_CHUNK_SIZE", "RUN_SCHEMA", "SCORE_ARRAY_NAMES", "SCORE_SCHEMA",
    "SHARD_RUN_SCHEMA", "TOPK", "TORCH_DTYPE", "hierarchy_result_arrays",
    "hierarchy_topk_arrays_sha256", "load_hierarchy_score",
]
