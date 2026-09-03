"""Seal paired point/continuous-coordinate poses as a Top-2 hypothesis set.

The first pose is the retained V5 operating point and the second is the V11
continuous chart-coordinate proposal.  No score is invented to collapse the
set: downstream geometry may verify either candidate.  Query labels are opened
only after the complete hypothesis-set NPZ has been written and hashed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import THRESHOLDS
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _seal_hypotheses(
    primary_path: Path,
    alternate_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    primary, primary_meta = _load_pose_candidate(primary_path)
    alternate, alternate_meta = _load_pose_candidate(alternate_path)
    names = primary["names"].astype(str)
    if not np.array_equal(names, alternate["names"].astype(str)):
        raise ValueError("coordinate-pose query order differs")
    if (
        primary_meta.get("query_pose_or_ground_truth_read") is not False
        or alternate_meta.get("query_pose_or_ground_truth_read") is not False
    ):
        raise ValueError("coordinate-pose input is not pose-free")
    pose = np.stack([
        np.asarray(primary["pose_w2c"], np.float64),
        np.asarray(alternate["pose_w2c"], np.float64),
    ], axis=1)
    usable = np.stack([
        np.asarray(primary["usable"], bool),
        np.asarray(alternate["usable"], bool),
    ], axis=1)
    arrays = {
        "names": primary["names"],
        "pose_w2c": pose,
        "usable": usable,
        "branch_names": np.asarray(["point_coordinate_V5", "continuous_chart_coordinate_V11"]),
    }
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_coordinate_pose_hypothesis_set_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "maximum_hypotheses_per_query": 2,
        "primary_branch_index": 0,
        "candidate_order": "retained_V5_then_continuous_V11",
        "candidate_collapse_or_label_based_selection": False,
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "primary_file_sha256": file_sha256(primary_path),
        "primary_content_sha256": primary_meta.get("content_sha256"),
        "alternate_file_sha256": file_sha256(alternate_path),
        "alternate_content_sha256": alternate_meta.get("content_sha256"),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    return arrays, metadata


def _load_hypotheses(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    expected_content = canonical_json_sha256({
        key: value for key, value in metadata.items() if key != "content_sha256"
    })
    count = len(arrays.get("names", []))
    if (
        metadata.get("artifact_type") != "goal_maplet_coordinate_pose_hypothesis_set_v1"
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("candidate_collapse_or_label_based_selection") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or metadata.get("content_sha256") != expected_content
        or arrays.get("pose_w2c", np.empty(0)).shape != (count, 2, 4, 4)
        or arrays.get("usable", np.empty(0)).shape != (count, 2)
        or arrays.get("branch_names", np.empty(0)).tolist()
        != ["point_coordinate_V5", "continuous_chart_coordinate_V11"]
    ):
        raise ValueError("coordinate-pose hypothesis-set contract differs")
    return arrays, metadata


def _pose_errors(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    center = -pose[:3, :3].T @ pose[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation = float(np.linalg.norm(center - target_center))
    rotation = float(
        Rotation.from_matrix(pose[:3, :3] @ target[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation, rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary_pose", type=Path, required=True)
    parser.add_argument("--alternate_pose", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_hypotheses.exists():
        raise FileExistsError("refusing to overwrite coordinate-pose hypotheses")

    arrays, metadata = _seal_hypotheses(args.primary_pose, args.alternate_pose)
    args.output_frozen_hypotheses.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_hypotheses,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    arrays, metadata = _load_hypotheses(args.output_frozen_hypotheses)

    # Phase 2: labels are opened only after the exact candidate bytes exist.
    query_count = len(arrays["names"])
    translation = np.full((query_count, 2), np.inf, np.float64)
    rotation = np.full((query_count, 2), np.inf, np.float64)
    for query, name in enumerate(arrays["names"].astype(str).tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        for branch in range(2):
            if bool(arrays["usable"][query, branch]):
                translation[query, branch], rotation[query, branch] = _pose_errors(
                    arrays["pose_w2c"][query, branch], target,
                )

    primary_hits: dict[str, int] = {}
    top2_hits: dict[str, int] = {}
    gains: dict[str, int] = {}
    for translation_limit, rotation_limit in THRESHOLDS:
        key = f"{translation_limit:g}m_{rotation_limit:g}deg"
        hit = (
            arrays["usable"]
            & (translation <= translation_limit)
            & (rotation <= rotation_limit)
        )
        primary_hits[key] = int(np.sum(hit[:, 0]))
        top2_hits[key] = int(np.sum(np.any(hit, axis=1)))
        gains[key] = top2_hits[key] - primary_hits[key]

    # A continuous diagnostic chooses the lower normalized joint error only
    # after labels open; it is not stored as a deployable branch selection.
    normalized = np.hypot(translation / 0.1, rotation / 1.0)
    oracle_branch = np.argmin(normalized, axis=1)
    row = np.arange(query_count)
    finite = np.isfinite(normalized[row, oracle_branch])
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_coordinate_pose_hypothesis_set_evaluation_v1",
        "hypotheses_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_hypothesis_file_sha256": file_sha256(args.output_frozen_hypotheses),
        "frozen_hypothesis_content_sha256": metadata["content_sha256"],
        "query_count": int(query_count),
        "primary_threshold_hit_counts": primary_hits,
        "top2_threshold_hit_counts": top2_hits,
        "top2_net_gain_counts": gains,
        "postlabel_continuous_oracle_median_translation_m": float(
            np.median(translation[row[finite], oracle_branch[finite]])
        ),
        "postlabel_continuous_oracle_median_rotation_deg": float(
            np.median(rotation[row[finite], oracle_branch[finite]])
        ),
        "postlabel_branch1_selected_count": int(np.sum(oracle_branch[finite] == 1)),
        "top2_is_proposal_recall_not_single_pose_accuracy": True,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
