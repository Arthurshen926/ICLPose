"""Select baseline or sparse-occlusion plane PnP without pose labels.

The carrier branch is deliberately auxiliary. It replaces the connected-
region baseline only when it strictly increases the RANSAC inlier count while
not decreasing the inlier ratio. Ties and all incomparable cases stay on the
baseline. This keeps the observed connected planes as the default and avoids
turning a successful foreground bridge into an unconditional appearance merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _merge(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    values: dict[str, list[np.ndarray]] = {}
    metadata: list[dict[str, object]] = []
    for path in paths:
        arrays, meta = _load_frozen_poses(path)
        metadata.append(meta)
        for key, value in arrays.items():
            values.setdefault(key, []).append(np.asarray(value))
    return {key: np.concatenate(rows, axis=0) for key, rows in values.items()}, metadata


def _carrier_pareto_mask(
    baseline_usable: np.ndarray,
    baseline_candidates: np.ndarray,
    baseline_inliers: np.ndarray,
    carrier_usable: np.ndarray,
    carrier_candidates: np.ndarray,
    carrier_inliers: np.ndarray,
) -> np.ndarray:
    """Return a conservative label-free carrier selection mask."""

    baseline_usable = np.asarray(baseline_usable, bool)
    carrier_usable = np.asarray(carrier_usable, bool)
    baseline_candidates = np.asarray(baseline_candidates, np.int64)
    carrier_candidates = np.asarray(carrier_candidates, np.int64)
    baseline_inliers = np.asarray(baseline_inliers, np.int64)
    carrier_inliers = np.asarray(carrier_inliers, np.int64)
    arrays = (
        baseline_usable, carrier_usable, baseline_candidates,
        carrier_candidates, baseline_inliers, carrier_inliers,
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("baseline/carrier selection arrays differ in shape")
    baseline_ratio = baseline_inliers / np.maximum(baseline_candidates, 1)
    carrier_ratio = carrier_inliers / np.maximum(carrier_candidates, 1)
    return carrier_usable & (
        ~baseline_usable
        | ((carrier_inliers > baseline_inliers) & (carrier_ratio >= baseline_ratio))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--carrier_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline_pose_inventory) != len(args.carrier_pose_inventory):
        raise ValueError("baseline/carrier shard counts differ")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sparse-occlusion PnP selection")

    baseline, baseline_meta = _merge(args.baseline_pose_inventory)
    carrier, carrier_meta = _merge(args.carrier_pose_inventory)
    names = baseline["names"].astype(str)
    if (
        not np.array_equal(names, carrier["names"].astype(str))
        or len(set(names.tolist())) != len(names)
    ):
        raise ValueError("baseline/carrier query names differ")
    if any(meta.get("query_moge3_role") != "plane_segmentation_only" for meta in baseline_meta):
        raise ValueError("baseline is not the connected plane-segmentation branch")
    if any(
        meta.get("query_moge3_role")
        != "plane_segmentation_and_sparse_foreground_carrier_only"
        for meta in carrier_meta
    ):
        raise ValueError("carrier branch lacks sparse-foreground semantics")
    lineage_keys = (
        "query_camera_only_inventory_file_sha256",
        "source_observation_bank_file_sha256",
        "token_grid",
    )
    for key in lineage_keys:
        baseline_values = [meta.get(key) for meta in baseline_meta]
        carrier_values = [meta.get(key) for meta in carrier_meta]
        if baseline_values != carrier_values or any(value is None for value in baseline_values):
            raise ValueError(f"baseline/carrier {key} lineage differs")

    choose_carrier = _carrier_pareto_mask(
        baseline["usable"], baseline["candidate_correspondence_count"],
        baseline["pnp_inlier_count"], carrier["usable"],
        carrier["candidate_correspondence_count"], carrier["pnp_inlier_count"],
    )
    baseline_ratio = baseline["pnp_inlier_count"] / np.maximum(
        baseline["candidate_correspondence_count"], 1,
    )
    carrier_ratio = carrier["pnp_inlier_count"] / np.maximum(
        carrier["candidate_correspondence_count"], 1,
    )
    arrays = {
        "names": names,
        "pose_w2c": np.where(
            choose_carrier[:, None, None], carrier["pose_w2c"], baseline["pose_w2c"],
        ).astype(np.float64),
        "usable": np.where(choose_carrier, carrier["usable"], baseline["usable"]).astype(bool),
        "selected_branch": np.where(choose_carrier, 1, 0).astype(np.int8),
        "selected_candidate_correspondence_count": np.where(
            choose_carrier, carrier["candidate_correspondence_count"],
            baseline["candidate_correspondence_count"],
        ).astype(np.int64),
        "selected_pnp_inlier_count": np.where(
            choose_carrier, carrier["pnp_inlier_count"], baseline["pnp_inlier_count"],
        ).astype(np.int64),
        "baseline_inlier_ratio": baseline_ratio.astype(np.float64),
        "carrier_inlier_ratio": carrier_ratio.astype(np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_sparse_occlusion_pnp_pareto_selection_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "selected_carrier_count": int(np.sum(choose_carrier)),
        "selection_rule": (
            "carrier_if_baseline_unusable_or_strictly_more_inliers_and_"
            "nondecreasing_inlier_ratio;_tie_baseline"
        ),
        "baseline_connected_regions_always_available": True,
        "carrier_is_auxiliary_pose_hypothesis": True,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "selection_designed_after_oldhospital_a1_a2_mechanism_audit": True,
        "eligible_use": "historical_mechanism_control_only_then_freeze_for_new_unseen_window",
        "baseline_source_file_sha256_in_order": [
            file_sha256(path) for path in args.baseline_pose_inventory
        ],
        "carrier_source_file_sha256_in_order": [
            file_sha256(path) for path in args.carrier_pose_inventory
        ],
        "query_camera_only_inventory_file_sha256_in_order": [
            meta["query_camera_only_inventory_file_sha256"] for meta in baseline_meta
        ],
        "source_observation_bank_file_sha256_in_order": [
            meta["source_observation_bank_file_sha256"] for meta in baseline_meta
        ],
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
