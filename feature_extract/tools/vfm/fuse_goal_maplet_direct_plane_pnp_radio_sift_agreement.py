"""Fuse a RADIO-plane pose with a nearby high-resolution SIFT-plane pose.

The RADIO result remains authoritative unless the best frozen SIFT candidate
has the previously fixed minimum support and the two independently estimated
poses lie in the same bounded localization basin.  Eligible poses are fused at
their SE(3) midpoint.  The tool never opens query pose labels or query depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


RADIO_KEYS = (
    "names", "pose_w2c", "usable", "selected_inlier_ratio",
    "selected_candidate_correspondence_count", "selected_pnp_inlier_count",
)
SIFT_KEYS = (
    "names", "candidate_offsets", "candidate_pose_w2c",
    "candidate_inlier_count", "candidate_origin",
)


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, np.float64)
    return -pose[:3, :3].T @ pose[:3, 3]


def _pose_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(_camera_center(left) - _camera_center(right)))
    rotation = float(
        Rotation.from_matrix(left[:3, :3] @ right[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation, rotation


def _interpolate_pose(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("pose interpolation fraction must lie in [0,1]")
    rotations = Rotation.from_matrix(np.asarray([left[:3, :3], right[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([float(fraction)]).as_matrix()[0]
    center = (
        (1.0 - float(fraction)) * _camera_center(left)
        + float(fraction) * _camera_center(right)
    )
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -rotation @ center
    return output


def _load_radio(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        complete = {
            key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
        }
    if (
        metadata.get("artifact_type")
        != "goal_maplet_direct_plane_pnp_multiscale_reliability_cascade_v7"
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("strict_runtime_phase_separation_eligible") is not True
        or arrays_sha256(complete) != metadata.get("arrays_sha256")
        or any(key not in complete for key in RADIO_KEYS)
    ):
        raise ValueError("RADIO plane pose inventory is not strict and frozen")
    return complete, metadata


def _load_sift(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    shards: dict[str, list[np.ndarray]] = {key: [] for key in SIFT_KEYS}
    metadata_rows: list[dict[str, object]] = []
    candidate_base = 0
    query_base = 0
    adjusted_offsets: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            complete = {
                key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
            }
        if (
            metadata.get("artifact_type")
            != "goal_maplet_masked_sift_plane_pnp_candidate_inventory_v1"
            or metadata.get("pose_or_ground_truth_opened") is not False
            or metadata.get("query_depth_used") is not False
            or arrays_sha256(complete) != metadata.get("arrays_sha256")
            or any(key not in complete for key in SIFT_KEYS)
        ):
            raise ValueError("SIFT plane candidate inventory is not pose-free")
        offsets = np.asarray(complete["candidate_offsets"], np.int64)
        if (
            offsets.shape != (len(complete["names"]) + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(complete["candidate_pose_w2c"])
            or complete["candidate_pose_w2c"].shape[1:] != (4, 4)
            or len(complete["candidate_inlier_count"]) != int(offsets[-1])
            or len(complete["candidate_origin"]) != int(offsets[-1])
            or not np.all(np.isfinite(complete["candidate_pose_w2c"]))
        ):
            raise ValueError("SIFT candidate offsets differ")
        for lower, upper in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            inliers = np.asarray(
                complete["candidate_inlier_count"][int(lower):int(upper)], np.int64
            )
            if np.any(np.diff(inliers) > 0):
                raise ValueError("SIFT candidates are not support-ranked")
        adjusted = offsets + candidate_base
        if query_base:
            adjusted = adjusted[1:]
        adjusted_offsets.append(adjusted)
        candidate_base += int(offsets[-1])
        query_base += len(complete["names"])
        for key in SIFT_KEYS:
            if key != "candidate_offsets":
                shards[key].append(complete[key])
        metadata_rows.append(metadata)
    arrays = {
        key: np.concatenate(values, axis=0)
        for key, values in shards.items() if key != "candidate_offsets"
    }
    arrays["candidate_offsets"] = np.concatenate(adjusted_offsets)
    return arrays, metadata_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_pose_inventory", type=Path, required=True)
    parser.add_argument("--sift_candidate_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum_sift_inliers", type=int, default=16)
    parser.add_argument("--maximum_agreement_translation_m", type=float, default=0.5)
    parser.add_argument("--maximum_agreement_rotation_deg", type=float, default=5.0)
    parser.add_argument("--sift_interpolation_fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite RADIO-SIFT agreement fusion")
    if int(args.minimum_sift_inliers) < 1:
        raise ValueError("minimum SIFT inliers must be positive")
    if (
        float(args.maximum_agreement_translation_m) <= 0.0
        or float(args.maximum_agreement_rotation_deg) <= 0.0
    ):
        raise ValueError("pose-agreement thresholds must be positive")

    radio, radio_meta = _load_radio(args.radio_pose_inventory)
    sift, sift_meta = _load_sift(args.sift_candidate_inventory)
    names = radio["names"].astype(str)
    if (
        not np.array_equal(names, sift["names"].astype(str))
        or len(set(names.tolist())) != len(names)
    ):
        raise ValueError("RADIO and SIFT query inventories differ")

    output_pose = np.asarray(radio["pose_w2c"], np.float64).copy()
    selected = np.zeros(len(names), bool)
    sift_available = np.zeros(len(names), bool)
    sift_inliers = np.zeros(len(names), np.int64)
    agreement_translation = np.full(len(names), np.inf, np.float64)
    agreement_rotation = np.full(len(names), np.inf, np.float64)
    offsets = np.asarray(sift["candidate_offsets"], np.int64)
    for index in range(len(names)):
        lo, hi = map(int, offsets[index:index + 2])
        if lo == hi:
            continue
        sift_available[index] = True
        sift_inliers[index] = int(sift["candidate_inlier_count"][lo])
        candidate = np.asarray(sift["candidate_pose_w2c"][lo], np.float64)
        translation, rotation = _pose_distance(output_pose[index], candidate)
        agreement_translation[index] = translation
        agreement_rotation[index] = rotation
        eligible = (
            bool(radio["usable"][index])
            and sift_inliers[index] >= int(args.minimum_sift_inliers)
            and translation <= float(args.maximum_agreement_translation_m)
            and rotation <= float(args.maximum_agreement_rotation_deg)
        )
        if eligible:
            output_pose[index] = _interpolate_pose(
                output_pose[index], candidate, float(args.sift_interpolation_fraction)
            )
            selected[index] = True

    arrays = {
        "names": names,
        "pose_w2c": output_pose,
        "usable": np.asarray(radio["usable"], bool),
        "selected_branch": np.where(selected, 91, 90).astype(np.int16),
        "selected_inlier_ratio": np.asarray(radio["selected_inlier_ratio"], np.float64),
        "selected_candidate_correspondence_count": np.asarray(
            radio["selected_candidate_correspondence_count"], np.int64
        ),
        "selected_pnp_inlier_count": np.asarray(
            radio["selected_pnp_inlier_count"], np.int64
        ),
        "sift_candidate_available": sift_available,
        "sift_candidate_inlier_count": sift_inliers,
        "radio_sift_agreement_translation_m": agreement_translation,
        "radio_sift_agreement_rotation_deg": agreement_rotation,
        "radio_sift_midpoint_selected": selected,
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_radio_sift_agreement_midpoint_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selection_rule": (
            "radio_default_else_supported_sift_same_basin_equal_se3_midpoint"
        ),
        "minimum_sift_inliers": int(args.minimum_sift_inliers),
        "maximum_agreement_translation_m": float(args.maximum_agreement_translation_m),
        "maximum_agreement_rotation_deg": float(args.maximum_agreement_rotation_deg),
        "sift_interpolation_fraction": float(args.sift_interpolation_fraction),
        "midpoint_selected_count": int(np.sum(selected)),
        "source_branch_codes": {"90": "radio", "91": "radio_sift_midpoint"},
        "radio_pose_inventory_file_sha256": file_sha256(args.radio_pose_inventory),
        "radio_pose_inventory_content_sha256": radio_meta.get("content_sha256"),
        "sift_candidate_inventory_file_sha256_in_order": [
            file_sha256(path) for path in args.sift_candidate_inventory
        ],
        "sift_candidate_inventory_content_sha256_in_order": [
            row.get("content_sha256") for row in sift_meta
        ],
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "strict_runtime_phase_separation_eligible": True,
        "configuration_role": "historical_validation_cross_scene_fixed_rule_not_pristine_blind",
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
