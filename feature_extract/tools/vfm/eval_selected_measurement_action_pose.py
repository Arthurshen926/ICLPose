"""Evaluate frozen selected-measurement actions with an unchanged PnP backend."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.probe_local_assignment_support_views import (
    _evaluate_pose_strategy,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument(
        "--action_predictions_csv",
        default="",
        help="optional action CSV; omit to replay the immutable selected policy",
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument(
        "--split_name", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_policy(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected-policy artifact")
    required = {
        "selected_rows",
        "query_ids",
        "query_xy",
        "selected_track_ids",
        "selected_prototype_ids",
        "selected_canonical_rows",
        "selected_assignment_scores",
        "selected_pose_selection_scores",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"selected-policy artifact is missing arrays: {sorted(missing)}")
    count = len(arrays["selected_rows"])
    if any(len(arrays[key]) != count for key in required):
        raise ValueError("selected-policy arrays have inconsistent lengths")
    return arrays, metadata


def _load_actions(path: Path) -> dict[int, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        policy_row = int(row["policy_row_index"])
        if policy_row in output:
            raise ValueError(f"duplicate action for policy row {policy_row}")
        action = str(row["action"])
        if action not in {"KEEP", "UPDATE_MEAN", "UPDATE_MODE", "DROP"}:
            raise ValueError(f"unsupported measurement action: {action}")
        output[policy_row] = row
    return output


def _pose_gate(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "success_rate": float(candidate["success_rate"])
        >= float(baseline["success_rate"]),
        "median_translation": float(candidate["median_translation_m_success"])
        <= float(baseline["median_translation_m_success"]),
        "p90_translation": float(candidate["p90_translation_m_success"])
        <= float(baseline["p90_translation_m_success"]),
        "median_rotation": float(candidate["median_rotation_deg_success"])
        <= float(baseline["median_rotation_deg_success"]),
        "recall_25cm_2deg": float(candidate["recall_25cm_2deg"])
        >= float(baseline["recall_25cm_2deg"]),
        "recall_10cm_5deg": float(candidate["recall_10cm_5deg"])
        >= float(baseline["recall_10cm_5deg"]),
        "recall_5cm_5deg": float(candidate["recall_5cm_5deg"])
        >= float(baseline["recall_5cm_5deg"]),
    }
    return {"passes": bool(all(checks.values())), "checks": checks}


def _write_pose_rows(path: Path, baseline_rows, action_rows) -> None:
    by_query = {str(row["query_id"]): row for row in action_rows}
    fields = [
        "query_id",
        "baseline_success",
        "action_success",
        "baseline_translation_m",
        "action_translation_m",
        "baseline_rotation_deg",
        "action_rotation_deg",
        "baseline_match_count",
        "action_match_count",
        "baseline_inlier_count",
        "action_inlier_count",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline in baseline_rows:
            action = by_query[str(baseline["query_id"])]
            writer.writerow(
                {
                    "query_id": baseline["query_id"],
                    "baseline_success": baseline.get("success"),
                    "action_success": action.get("success"),
                    "baseline_translation_m": baseline.get("translation_m"),
                    "action_translation_m": action.get("translation_m"),
                    "baseline_rotation_deg": baseline.get("rotation_deg"),
                    "action_rotation_deg": action.get("rotation_deg"),
                    "baseline_match_count": baseline.get("match_count"),
                    "action_match_count": action.get("match_count"),
                    "baseline_inlier_count": baseline.get("inlier_count"),
                    "action_inlier_count": action.get("inlier_count"),
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    policy_path = Path(args.policy_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    arrays, metadata = _load_policy(policy_path)
    expected_hashes = {
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    mismatches = {
        key: {"policy": metadata.get(key), "input": value}
        for key, value in expected_hashes.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"stale selected-policy pose inputs: {mismatches}")
    actions = (
        {}
        if not str(args.action_predictions_csv)
        else _load_actions(Path(args.action_predictions_csv))
    )
    split = json.loads(split_path.read_text())
    split_query_ids = {str(value) for value in split[str(args.split_name)]}
    query_ids_all = np.asarray(arrays["query_ids"]).astype(str)
    selected_policy_rows = np.flatnonzero(
        np.isin(query_ids_all, np.asarray(sorted(split_query_ids), dtype=np.str_))
    )
    query_ids = query_ids_all[selected_policy_rows]
    baseline_xy = np.asarray(arrays["query_xy"], dtype=np.float64)[
        selected_policy_rows
    ]
    action_xy = baseline_xy.copy()
    dropped = np.zeros((len(selected_policy_rows),), dtype=bool)
    applied_counts = {name: 0 for name in ("KEEP", "UPDATE_MEAN", "UPDATE_MODE", "DROP")}
    for local_row, policy_row in enumerate(selected_policy_rows.tolist()):
        action_row = actions.get(int(policy_row))
        if action_row is None:
            applied_counts["KEEP"] += 1
            continue
        if str(action_row["query_id"]) != str(query_ids[local_row]):
            raise ValueError("action query identity does not match selected policy")
        expected_track = int(np.asarray(arrays["selected_track_ids"])[policy_row])
        if int(action_row["track_id"]) != expected_track:
            raise ValueError("action track identity does not match selected policy")
        action = str(action_row["action"])
        applied_counts[action] += 1
        if action == "UPDATE_MEAN":
            action_xy[local_row] = [
                float(action_row["updated_x"]),
                float(action_row["updated_y"]),
            ]
        elif action == "UPDATE_MODE":
            action_xy[local_row] = [
                float(action_row["mode_updated_x"]),
                float(action_row["mode_updated_y"]),
            ]
        elif action == "DROP":
            dropped[local_row] = True

    landmark_index, _bank_metadata = load_landmark_index_npz(bank_path)
    canonical_rows = np.asarray(arrays["selected_canonical_rows"], dtype=np.int64)[
        selected_policy_rows
    ]
    if np.any(canonical_rows < 0):
        raise ValueError("selected policy contains an invalid canonical landmark row")
    candidates = UniqueTrackCandidateSet(
        canonical_rows[:, None],
        np.asarray(arrays["selected_track_ids"], dtype=np.int64)[
            selected_policy_rows, None
        ],
        np.asarray(arrays["selected_prototype_ids"], dtype=np.int64)[
            selected_policy_rows, None
        ],
        np.asarray(arrays["selected_assignment_scores"], dtype=np.float32)[
            selected_policy_rows, None
        ],
    )
    assignment_scores = np.asarray(
        arrays["selected_assignment_scores"], dtype=np.float32
    )[selected_policy_rows, None]
    selection_scores = np.asarray(
        arrays["selected_pose_selection_scores"], dtype=np.float32
    )[selected_policy_rows, None]
    action_assignment_scores = assignment_scores.copy()
    action_selection_scores = selection_scores.copy()
    action_assignment_scores[dropped, 0] = -np.inf
    action_selection_scores[dropped, 0] = -np.inf
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)[
        selected_policy_rows
    ]
    selected_tracks = np.asarray(arrays["selected_track_ids"], dtype=np.int64)[
        selected_policy_rows
    ]

    def observations(xy: np.ndarray) -> list[ColmapTrackObservation]:
        return [
            ColmapTrackObservation(
                track_id=int(selected_tracks[row]),
                image_id=str(query_ids[row]),
                point2d_idx=int(selected_rows[row]),
                xy=(float(xy[row, 0]), float(xy[row, 1])),
                xyz=np.zeros((3,), dtype=np.float64),
                track_length=1,
                reprojection_error=0.0,
            )
            for row in range(len(query_ids))
        ]

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    max_matches = metadata.get("max_matches")
    selection_mode = str(metadata.get("selection_mode") or "score_topk")
    common = {
        "candidates": candidates,
        "query_ids": query_ids.tolist(),
        "landmark_index": landmark_index,
        "cameras": cameras,
        "images_by_name": images_by_name,
        "reprojection_error_px": float(args.pnp_reprojection_error_px),
        "iterations": int(args.pnp_iterations),
        "max_matches": None if max_matches is None else int(max_matches),
        "pose_selection_mode": selection_mode,
    }
    baseline_summary, baseline_rows = _evaluate_pose_strategy(
        strategy="selected_policy_keep_coordinates",
        scores=assignment_scores,
        selection_scores=selection_scores,
        query_observations=observations(baseline_xy),
        **common,
    )
    action_summary, action_rows = _evaluate_pose_strategy(
        strategy="selected_policy_frozen_measurement_actions",
        scores=action_assignment_scores,
        selection_scores=action_selection_scores,
        query_observations=observations(action_xy),
        **common,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pose_rows_path = output / "pose_rows.csv"
    _write_pose_rows(pose_rows_path, baseline_rows, action_rows)
    pose_rows_json_path = output / "pose_rows.json"
    pose_rows_json_path.write_text(
        json.dumps(
            {"baseline": baseline_rows, "action": action_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    gate = _pose_gate(action_summary, baseline_summary)
    summary = {
        "stage": "selected_measurement_action_pose_evaluation",
        "protocol": {
            "evaluation_role": str(args.evaluation_role),
            "split": str(args.split_name),
            "target_fields_read_from_action_csv": False,
            "identity_reassignment": False,
            "match_identity_set_reduced_by_drop": bool(
                applied_counts["DROP"] > 0
            ),
            "selection_scores_changed": False,
            "match_budget_changed": False,
            "ransac_seed_changed": False,
            "measurement_xy_changed": bool(
                applied_counts["UPDATE_MEAN"]
                + applied_counts["UPDATE_MODE"]
                > 0
            ),
            "match_set_changed_by_drop": bool(applied_counts["DROP"] > 0),
            "only_measurement_xy_changed": bool(
                applied_counts["DROP"] == 0
                and applied_counts["UPDATE_MEAN"]
                + applied_counts["UPDATE_MODE"]
                > 0
            ),
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "selected_policy_row_count": int(len(selected_policy_rows)),
        "action_rows_supplied": int(len(actions)),
        "applied_action_counts": applied_counts,
        "baseline_pose": baseline_summary,
        "action_pose": action_summary,
        "pose_gate": gate,
        "promotion_eligible": bool(
            gate["passes"] and str(args.evaluation_role) == "untouched_test"
        ),
        "inputs": {
            "policy_artifact": str(policy_path),
            "policy_artifact_sha256": file_sha256_short(policy_path),
            "action_predictions_csv": (
                None
                if not str(args.action_predictions_csv)
                else str(args.action_predictions_csv)
            ),
            "action_predictions_sha256": (
                None
                if not str(args.action_predictions_csv)
                else file_sha256_short(Path(args.action_predictions_csv))
            ),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": expected_hashes[
                "projected_landmark_bank_sha256"
            ],
            "split_json": str(split_path),
            "split_json_sha256": expected_hashes["split_json_sha256"],
        },
        "outputs": {
            "pose_rows": str(pose_rows_path),
            "pose_rows_sha256": file_sha256_short(pose_rows_path),
            "pose_rows_json": str(pose_rows_json_path),
            "pose_rows_json_sha256": file_sha256_short(pose_rows_json_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
