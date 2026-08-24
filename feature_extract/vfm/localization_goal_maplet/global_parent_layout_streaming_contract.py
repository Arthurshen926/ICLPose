"""Phase-separated artifact contract for global streaming layout scores."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from .lineage import arrays_sha256, file_sha256
from .natural_pose_transport_bridge import camera_intrinsics_content_sha256
from .parent_support_layout_guide import ParentLayoutCamera
from .pure_retrieval import PureRadioPhysicalRetrieval


SCORE_SCHEMA = "goal_maplet_global_parent_layout_streaming_score_v1"
RUN_SCHEMA = "goal_maplet_global_parent_layout_streaming_phase1_run_v1"
GATE_SCHEMA = "goal_maplet_global_parent_layout_streaming_gpu_gate_v1"
EXPECTED_SEQ10_QUERY_COUNT = 88
MAXIMUM_QUERY_PARENTS = 32
RETURNED_TOPK = 4096
POSITION_CHUNK_SIZE = 1024
TORCH_DTYPE = "float64"
RANKING_KEY = "(-score,position_row,orientation_row)"

SCORE_ARRAY_NAMES = (
    "image_id",
    "selected_query_parent_ids",
    "selected_query_parent_probability_mass",
    "top_scores",
    "top_position_factor_indices",
    "top_orientation_factor_indices",
    "top_visible_parent_counts",
    "top_front_facing_parent_counts",
    "top_positive_depth_parent_counts",
    "top_center_in_image_parent_counts",
    "top_projected_token_footprint_mass",
    "top_sqrt_overlap_mass",
)


def load_json_no_duplicate_keys(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON repeats key {key!r}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_pose_free_camera_manifest(
    path: Path,
) -> tuple[dict[str, ParentLayoutCamera], dict[str, str]]:
    manifest_path = Path(path).resolve()
    manifest = load_json_no_duplicate_keys(manifest_path)
    if set(manifest) != {
        "format", "query_count", "cameras", "intrinsic_audit", "production_contract",
    }:
        raise ValueError("pose-free camera manifest fields differ")
    contract = manifest["production_contract"]
    if (
        manifest["format"] != "per_query_colmap_calibration_only_v1"
        or not isinstance(contract, dict)
        or set(contract) != {
            "contains_camera_calibration", "contains_camera_pose",
            "contains_sfm_points", "contains_sfm_tracks",
        }
        or contract.get("contains_camera_calibration") is not True
        or contract.get("contains_camera_pose") is not False
        or contract.get("contains_sfm_points") is not False
        or contract.get("contains_sfm_tracks") is not False
        or not isinstance(manifest["cameras"], dict)
        or int(manifest["query_count"]) != len(manifest["cameras"])
    ):
        raise ValueError("camera manifest is not calibration-only")
    cameras: dict[str, ParentLayoutCamera] = {}
    bindings: dict[str, str] = {}
    for image_id, raw in manifest["cameras"].items():
        if not isinstance(raw, dict) or set(raw) != {"model_id", "width", "height", "params"}:
            raise ValueError("camera manifest query fields differ")
        camera = ParentLayoutCamera(
            int(raw["model_id"]), int(raw["width"]), int(raw["height"]),
            tuple(float(value) for value in raw["params"]),
        )
        if not str(image_id) or str(image_id) in cameras:
            raise ValueError("camera manifest query identity differs")
        cameras[str(image_id)] = camera
        bindings[str(image_id)] = camera_intrinsics_content_sha256(
            str(image_id), camera.model_id, camera.width, camera.height, camera.params,
        )
    return cameras, bindings


def load_seq10_control_summary(path: Path) -> dict[str, Any]:
    summary = load_json_no_duplicate_keys(Path(path))
    rows = summary.get("rows")
    blockers = [
        "query_route_used_for_validity_calibration",
        "layout_allocator_tuning_route_overlaps_query_route",
    ]
    if (
        summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1"
        or int(summary.get("query_count", -1)) != EXPECTED_SEQ10_QUERY_COUNT
        or not isinstance(rows, list) or len(rows) != EXPECTED_SEQ10_QUERY_COUNT
        or summary.get("control_only") is not True
        or summary.get("promotion_eligible") is not False
        or summary.get("eligible_for_held_route_promotion") is not False
        or summary.get("promotion_blockers") != blockers
        or summary.get("layout_child_allocator_tuning_route") != "seq10"
        or any(summary.get(key) is not False for key in (
            "uses_query_ground_truth", "uses_query_pose", "uses_alike", "uses_pnp",
        ))
    ):
        raise ValueError("seq10 layout retrieval control summary differs")
    image_ids = [str(row.get("image_id", "")) for row in rows]
    if (
        image_ids != sorted(image_ids)
        or len(set(image_ids)) != len(image_ids)
        or any(not value.startswith("seq10/") for value in image_ids)
    ):
        raise ValueError("seq10 retrieval query inventory differs")
    return summary


def load_bound_retrieval(
    row: dict[str, Any], *, physical_file_sha256: str, physical_content_sha256: str,
) -> tuple[PureRadioPhysicalRetrieval, Path]:
    artifact = Path(str(row.get("artifact", ""))).resolve()
    if file_sha256(artifact) != str(row.get("artifact_sha256", "")):
        raise ValueError("retrieval row file hash differs")
    retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
    if (
        retrieval.image_id != str(row.get("image_id", ""))
        or retrieval.content_sha256 != str(row.get("content_sha256", ""))
        or retrieval.physical_map_sha256 != str(physical_content_sha256)
        or retrieval.metadata.get("physical_map_file_sha256") != str(physical_file_sha256)
        or retrieval.metadata.get("uses_query_ground_truth") is not False
        or retrieval.metadata.get("uses_query_pose") is not False
    ):
        raise ValueError("retrieval row content/lineage differs")
    return retrieval, artifact


def atomic_save_score(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any],
) -> None:
    if set(arrays) != set(SCORE_ARRAY_NAMES):
        raise ValueError("streaming score public arrays differ")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_streaming_score(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    artifact = Path(path)
    expected = {*(f"{name}.npy" for name in SCORE_ARRAY_NAMES), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("streaming score is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("streaming score exact NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*SCORE_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("streaming score arrays differ")
        arrays = {name: np.asarray(data[name]) for name in SCORE_ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    score = arrays["top_scores"]
    position = arrays["top_position_factor_indices"]
    orientation = arrays["top_orientation_factor_indices"]
    count = int(score.size)
    one_d = SCORE_ARRAY_NAMES[3:]
    selected_ids = arrays["selected_query_parent_ids"]
    selected_mass = arrays["selected_query_parent_probability_mass"]
    if (
        metadata.get("artifact_type") != SCORE_SCHEMA
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or arrays["image_id"].shape != ()
        or selected_ids.dtype != np.int64
        or selected_ids.shape != (MAXIMUM_QUERY_PARENTS,)
        or np.unique(selected_ids).size != MAXIMUM_QUERY_PARENTS
        or selected_mass.dtype != np.float64
        or selected_mass.shape != selected_ids.shape
        or np.any(~np.isfinite(selected_mass)) or np.any(selected_mass <= 0.0)
        or score.dtype != np.float64 or score.shape != (RETURNED_TOPK,)
        or position.dtype != np.int64 or position.shape != score.shape
        or orientation.dtype != np.int64 or orientation.shape != score.shape
        or any(np.asarray(arrays[name]).shape != (count,) for name in one_d)
        or any(arrays[name].dtype != np.int16 for name in (
            "top_visible_parent_counts", "top_front_facing_parent_counts",
            "top_positive_depth_parent_counts", "top_center_in_image_parent_counts",
        ))
        or any(arrays[name].dtype != np.float64 for name in (
            "top_projected_token_footprint_mass", "top_sqrt_overlap_mass",
        ))
        or np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0 + 1e-12))
        or np.unique(np.stack([position, orientation], axis=1), axis=0).shape[0] != count
        or np.any(position < 0)
        or np.any(position >= int(metadata.get("position_count", -1)))
        or np.any(orientation < 0)
        or np.any(orientation >= int(metadata.get("orientation_count", -1)))
        or not np.array_equal(
            np.lexsort((orientation, position, -score)), np.arange(count),
        )
        or metadata.get("ranking_key") != RANKING_KEY
        or metadata.get("maximum_query_parents") != MAXIMUM_QUERY_PARENTS
        or metadata.get("returned_topk") != RETURNED_TOPK
        or metadata.get("position_chunk_size") != POSITION_CHUNK_SIZE
        or metadata.get("torch_dtype") != TORCH_DTYPE
        or metadata.get("phase2_labels_opened") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
    ):
        raise ValueError("streaming score contract differs")
    return arrays, metadata


__all__ = [
    "EXPECTED_SEQ10_QUERY_COUNT", "GATE_SCHEMA", "MAXIMUM_QUERY_PARENTS",
    "POSITION_CHUNK_SIZE", "RANKING_KEY", "RETURNED_TOPK", "RUN_SCHEMA",
    "SCORE_ARRAY_NAMES", "SCORE_SCHEMA", "TORCH_DTYPE", "atomic_save_score",
    "load_bound_retrieval", "load_json_no_duplicate_keys",
    "load_pose_free_camera_manifest", "load_seq10_control_summary",
    "load_streaming_score",
]
