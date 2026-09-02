"""Tiered HLoc refinement using a frozen RADIO/SIFT planar pose.

The tight 0.5 m / 5 degree basin uses an equal SE(3) midpoint.  A pose in the
standard 2 m coarse translation basin but still within 5 degrees is moved only
one quarter toward the planar estimate, reusing the planar refiner's existing
0.25 damping.  HLoc remains unchanged outside these frozen basins.  The tool
never opens query pose labels or query depth.
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
from feature_extract.tools.vfm.fuse_goal_maplet_hloc_plane_agreement import (
    _load_hloc_lineage,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _fuse_hloc_with_plane_tiered_agreement(
    hloc_pose: np.ndarray,
    plane_pose: np.ndarray,
    plane_usable: np.ndarray,
    tight_translation_m: float,
    tight_rotation_deg: float,
    damped_translation_m: float,
    damped_rotation_deg: float,
    damped_plane_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hloc = np.asarray(hloc_pose, np.float64)
    plane = np.asarray(plane_pose, np.float64)
    usable = np.asarray(plane_usable, bool)
    if hloc.shape != plane.shape or hloc.ndim != 3 or hloc.shape[1:] != (4, 4):
        raise ValueError("HLoc and planar pose arrays differ")
    if usable.shape != (len(hloc),) or not np.all(np.isfinite(hloc)):
        raise ValueError("HLoc poses or planar usability differ")
    if not 0.0 < float(damped_plane_fraction) < 0.5:
        raise ValueError("damped planar fraction must lie in (0,0.5)")
    if float(damped_translation_m) < float(tight_translation_m):
        raise ValueError("damped translation basin cannot be tighter than midpoint basin")

    output = hloc.copy()
    branch = np.full(len(hloc), 110, np.int16)
    fraction = np.zeros(len(hloc), np.float64)
    translation = np.full(len(hloc), np.inf, np.float64)
    rotation = np.full(len(hloc), np.inf, np.float64)
    for index in range(len(hloc)):
        if not usable[index] or not np.all(np.isfinite(plane[index])):
            continue
        translation[index], rotation[index] = _pose_distance(hloc[index], plane[index])
        if (
            translation[index] <= float(tight_translation_m)
            and rotation[index] <= float(tight_rotation_deg)
        ):
            fraction[index] = 0.5
            branch[index] = 111
        elif (
            translation[index] <= float(damped_translation_m)
            and rotation[index] <= float(damped_rotation_deg)
        ):
            fraction[index] = float(damped_plane_fraction)
            branch[index] = 112
        if fraction[index] > 0.0:
            output[index] = _interpolate_pose(hloc[index], plane[index], fraction[index])
    return output, branch, fraction, translation, rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_pose_inventory", type=Path, required=True)
    parser.add_argument("--hloc_lineage_inventory", type=Path, required=True)
    parser.add_argument("--tight_translation_m", type=float, default=0.5)
    parser.add_argument("--tight_rotation_deg", type=float, default=5.0)
    parser.add_argument("--damped_translation_m", type=float, default=2.0)
    parser.add_argument("--damped_rotation_deg", type=float, default=5.0)
    parser.add_argument("--damped_plane_fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite tiered HLoc-plane fusion")

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

    output_pose, branch, fraction, translation, rotation = (
        _fuse_hloc_with_plane_tiered_agreement(
            hloc["hloc_pose_w2c"], plane["pose_w2c"], plane["usable"],
            float(args.tight_translation_m), float(args.tight_rotation_deg),
            float(args.damped_translation_m), float(args.damped_rotation_deg),
            float(args.damped_plane_fraction),
        )
    )
    count = len(plane["names"])
    arrays = {
        "names": plane["names"].astype(str),
        "pose_w2c": output_pose,
        "usable": np.ones(count, bool),
        "selected_branch": branch,
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
        "hloc_plane_interpolation_fraction": fraction,
    }
    metadata = {
        "artifact_type": "goal_maplet_hloc_plane_tiered_agreement_v2",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(count),
        "selection_rule": "hloc_default_tight_equal_midpoint_else_coarse_quarter_damped_planar_refinement",
        "tight_translation_m": float(args.tight_translation_m),
        "tight_rotation_deg": float(args.tight_rotation_deg),
        "damped_translation_m": float(args.damped_translation_m),
        "damped_rotation_deg": float(args.damped_rotation_deg),
        "damped_plane_fraction": float(args.damped_plane_fraction),
        "hloc_count": int(np.sum(branch == 110)),
        "midpoint_count": int(np.sum(branch == 111)),
        "damped_count": int(np.sum(branch == 112)),
        "source_branch_codes": {
            "110": "hloc", "111": "hloc_plane_midpoint", "112": "hloc_plane_quarter_damped"
        },
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
        "configuration_role": "historical_cross_scene_hybrid_validation_existing_contract_constants",
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
