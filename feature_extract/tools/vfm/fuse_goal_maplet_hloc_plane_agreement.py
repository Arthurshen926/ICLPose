"""Refine an HLoc pose only when a frozen planar pose agrees with its basin.

HLoc is authoritative and provides complete coarse localization.  A usable
RADIO/SIFT planar pose may refine it only when the two independently estimated
poses satisfy the same fixed 0.5 m / 5 degree agreement rule already used by
the planar pipeline.  Eligible poses are fused at their equal SE(3) midpoint.
No pose labels, query depth, or post-label confidence is consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_hloc_low_support import (
    _load_plane,
)
from feature_extract.tools.vfm.fuse_goal_maplet_direct_plane_pnp_radio_sift_agreement import (
    _interpolate_pose,
    _pose_distance,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


HLOC_LINEAGE_KEYS = (
    "names", "hloc_pose_w2c", "hloc_fallback_selected", "pose_w2c", "usable",
)


def _load_hloc_lineage(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
    if (
        metadata.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_hloc_low_support_fallback_v1"
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or metadata.get("hloc_mapping_query_name_overlap_count") != 0
        or metadata.get("hloc_mapping_query_route_overlap_count") != 0
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
        or any(key not in complete for key in HLOC_LINEAGE_KEYS)
    ):
        raise ValueError("HLoc lineage inventory is not strict and disjoint")
    return complete, metadata


def _fuse_hloc_with_plane_agreement(
    hloc_pose: np.ndarray,
    plane_pose: np.ndarray,
    plane_usable: np.ndarray,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hloc = np.asarray(hloc_pose, np.float64)
    plane = np.asarray(plane_pose, np.float64)
    usable = np.asarray(plane_usable, bool)
    if hloc.shape != plane.shape or hloc.ndim != 3 or hloc.shape[1:] != (4, 4):
        raise ValueError("HLoc and planar pose arrays differ")
    if usable.shape != (len(hloc),) or not np.all(np.isfinite(hloc)):
        raise ValueError("HLoc poses or planar usability differ")
    output = hloc.copy()
    selected = np.zeros(len(hloc), bool)
    translation = np.full(len(hloc), np.inf, np.float64)
    rotation = np.full(len(hloc), np.inf, np.float64)
    for index in range(len(hloc)):
        if not usable[index] or not np.all(np.isfinite(plane[index])):
            continue
        translation[index], rotation[index] = _pose_distance(hloc[index], plane[index])
        if (
            translation[index] <= float(maximum_translation_m)
            and rotation[index] <= float(maximum_rotation_deg)
        ):
            output[index] = _interpolate_pose(hloc[index], plane[index], 0.5)
            selected[index] = True
    return output, selected, translation, rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_pose_inventory", type=Path, required=True)
    parser.add_argument("--hloc_lineage_inventory", type=Path, required=True)
    parser.add_argument("--maximum_agreement_translation_m", type=float, default=0.5)
    parser.add_argument("--maximum_agreement_rotation_deg", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite HLoc-plane agreement fusion")
    if (
        float(args.maximum_agreement_translation_m) <= 0.0
        or float(args.maximum_agreement_rotation_deg) <= 0.0
    ):
        raise ValueError("pose-agreement thresholds must be positive")

    plane, plane_meta = _load_plane(args.plane_pose_inventory)
    hloc, hloc_meta = _load_hloc_lineage(args.hloc_lineage_inventory)
    if (
        not np.array_equal(plane["names"].astype(str), hloc["names"].astype(str))
        or hloc_meta.get("plane_pose_inventory_file_sha256")
        != file_sha256(args.plane_pose_inventory)
        or hloc_meta.get("plane_pose_inventory_content_sha256")
        != plane_meta.get("content_sha256")
    ):
        raise ValueError("HLoc lineage and planar pose inventories differ")

    output_pose, selected, translation, rotation = _fuse_hloc_with_plane_agreement(
        hloc["hloc_pose_w2c"], plane["pose_w2c"], plane["usable"],
        float(args.maximum_agreement_translation_m),
        float(args.maximum_agreement_rotation_deg),
    )
    count = len(plane["names"])
    arrays = {
        "names": plane["names"].astype(str),
        "pose_w2c": output_pose,
        "usable": np.ones(count, bool),
        "selected_branch": np.where(selected, 111, 110).astype(np.int16),
        "selected_confidence": np.ones(count, np.float64),
        "selected_candidate_correspondence_count": np.asarray(
            plane["selected_candidate_correspondence_count"], np.int64
        ),
        "selected_pnp_inlier_count": np.asarray(
            plane["selected_pnp_inlier_count"], np.int64
        ),
        "hloc_pose_w2c": np.asarray(hloc["hloc_pose_w2c"], np.float64),
        "plane_pose_w2c": np.asarray(plane["pose_w2c"], np.float64),
        "plane_usable": np.asarray(plane["usable"], bool),
        "hloc_plane_agreement_translation_m": translation,
        "hloc_plane_agreement_rotation_deg": rotation,
        "hloc_plane_midpoint_selected": selected,
    }
    metadata = {
        "artifact_type": "goal_maplet_hloc_plane_agreement_midpoint_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(count),
        "selection_rule": "hloc_default_else_equal_se3_midpoint_if_frozen_planar_pose_same_basin",
        "maximum_agreement_translation_m": float(args.maximum_agreement_translation_m),
        "maximum_agreement_rotation_deg": float(args.maximum_agreement_rotation_deg),
        "interpolation_fraction": 0.5,
        "midpoint_selected_count": int(np.sum(selected)),
        "source_branch_codes": {"110": "hloc", "111": "hloc_plane_midpoint"},
        "selected_confidence_semantics": "binary_hloc_pose_availability_not_calibrated_pose_quality",
        "plane_pose_inventory_file_sha256": file_sha256(args.plane_pose_inventory),
        "plane_pose_inventory_content_sha256": plane_meta.get("content_sha256"),
        "hloc_lineage_inventory_file_sha256": file_sha256(args.hloc_lineage_inventory),
        "hloc_lineage_inventory_content_sha256": hloc_meta.get("content_sha256"),
        "hloc_artifact_file_sha256": hloc_meta.get("hloc_artifact_file_sha256"),
        "hloc_mapping_routes": hloc_meta.get("hloc_mapping_routes"),
        "hloc_query_routes": hloc_meta.get("hloc_query_routes"),
        "hloc_mapping_query_name_overlap_count": 0,
        "hloc_mapping_query_route_overlap_count": 0,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "strict_runtime_phase_separation_eligible": True,
        "configuration_role": "historical_cross_scene_hybrid_validation_fixed_existing_agreement_rule",
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
