"""Choose baseline or frozen-DROP pose using common held-out matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    deterministic_spatial_holdout,
    verify_pose_hypothesis,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--action_predictions_csv", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument(
        "--split_name", choices=("validation", "test"), required=True
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--holdout_folds", type=int, default=4)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--verification_strict_px", type=float, default=2.0)
    parser.add_argument("--verification_loose_px", type=float, default=5.0)
    parser.add_argument("--verification_keep_only", action="store_true")
    parser.add_argument("--drop_min_strict_inlier_gain", type=int, default=0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _load_policy(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            key: np.asarray(payload[key])
            for key in (
                "selected_rows",
                "query_ids",
                "query_xy",
                "selected_track_ids",
                "selected_prototype_ids",
                "selected_canonical_rows",
                "selected_pose_selection_scores",
            )
        }
    if metadata.get("format") != "pose_safe_selected_policy_v1":
        raise ValueError("unsupported selected-policy artifact")
    count = len(arrays["query_ids"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("selected-policy arrays have inconsistent lengths")
    return arrays, metadata


def _load_actions(path: Path) -> dict[int, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        policy_row = int(row["policy_row_index"])
        if policy_row in output:
            raise ValueError(f"duplicate action policy row: {policy_row}")
        if str(row["action"]) not in {
            "KEEP",
            "UPDATE_MEAN",
            "UPDATE_MODE",
            "DROP",
        }:
            raise ValueError("unsupported measurement action")
        output[policy_row] = row
    return output


def _set_seed(query_id: str) -> None:
    try:
        import cv2

        seed = int.from_bytes(
            hashlib.sha256(str(query_id).encode("utf8")).digest()[:4],
            "little",
        )
        cv2.setRNGSeed(int(seed % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _pose_summary(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, object]:
    success = [row for row in rows if bool(row.get(f"{prefix}_success"))]
    translations = np.asarray(
        [float(row[f"{prefix}_translation_m"]) for row in success],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [float(row[f"{prefix}_rotation_deg"]) for row in success],
        dtype=np.float64,
    )
    output: dict[str, object] = {
        "query_count": len(rows),
        "success_count": len(success),
        "success_rate": 0.0 if not rows else float(len(success) / len(rows)),
        "median_translation_m_success": (
            None if not len(translations) else float(np.median(translations))
        ),
        "p90_translation_m_success": (
            None
            if not len(translations)
            else float(np.percentile(translations, 90))
        ),
        "median_rotation_deg_success": (
            None if not len(rotations) else float(np.median(rotations))
        ),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        output[f"recall_{name}"] = (
            0.0
            if not rows
            else float(
                np.mean(
                    [
                        bool(row.get(f"{prefix}_success"))
                        and float(row[f"{prefix}_translation_m"]) <= distance
                        and float(row[f"{prefix}_rotation_deg"]) <= angle
                        for row in rows
                    ]
                )
            )
        )
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


def _fit_pose(
    matches: Sequence[QueryTo3DMatch],
    camera,
    *,
    query_id: str,
    reprojection_error_px: float,
    iterations: int,
):
    values = stable_uniform_ransac_order(matches)
    _set_seed(query_id)
    return estimate_pose_pnp_ransac(
        values,
        camera,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
        refine_method="LM",
    )


def should_choose_drop_hypothesis(
    *,
    drop_solver_success: bool,
    baseline_verification,
    drop_verification,
    minimum_strict_inlier_gain: int,
) -> bool:
    return bool(
        drop_solver_success
        and drop_verification is not None
        and baseline_verification is not None
        and drop_verification.strict_inlier_count
        >= baseline_verification.strict_inlier_count
        + int(minimum_strict_inlier_gain)
        and drop_verification.rank_key() > baseline_verification.rank_key()
    )


def _error_fields(result, gt_pose: np.ndarray) -> tuple[bool, float | None, float | None]:
    error = pnp_pose_error(result.pose_w2c, gt_pose)
    success = bool(
        result.success
        and np.isfinite(error.translation_m)
        and np.isfinite(error.rotation_deg)
    )
    return (
        success,
        None if not np.isfinite(error.translation_m) else float(error.translation_m),
        None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.drop_min_strict_inlier_gain) < 0:
        raise ValueError("drop_min_strict_inlier_gain must be non-negative")
    policy_path = Path(args.policy_artifact)
    action_path = Path(args.action_predictions_csv)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    arrays, metadata = _load_policy(policy_path)
    actions = _load_actions(action_path)
    if any(
        str(row["action"]) in {"UPDATE_MEAN", "UPDATE_MODE"}
        for row in actions.values()
    ):
        raise ValueError("drop hypothesis verifier does not accept coordinate updates")
    expected_hashes = {
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    mismatches = {
        key: {"policy": metadata.get(key), "input": value}
        for key, value in expected_hashes.items()
        if str(metadata.get(key, "")) != value
    }
    if mismatches:
        raise ValueError(f"stale selected-policy verifier inputs: {mismatches}")
    split = json.loads(split_path.read_text())
    split_queries = {str(value) for value in split[str(args.split_name)]}
    landmark_index, _bank_metadata = load_landmark_index_npz(bank_path)
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    query_xy = np.asarray(arrays["query_xy"], dtype=np.float64)
    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    tracks = np.asarray(arrays["selected_track_ids"], dtype=np.int64)
    prototypes = np.asarray(arrays["selected_prototype_ids"], dtype=np.int64)
    canonical_rows = np.asarray(arrays["selected_canonical_rows"], dtype=np.int64)
    scores = np.asarray(arrays["selected_pose_selection_scores"], dtype=np.float64)
    policy_rows_by_query: dict[str, list[int]] = {}
    for policy_row, query_id in enumerate(query_ids.tolist()):
        if query_id in split_queries:
            policy_rows_by_query.setdefault(query_id, []).append(policy_row)
    if set(policy_rows_by_query) != split_queries:
        raise ValueError("selected policy does not exactly cover requested split")

    max_matches = int(metadata.get("max_matches") or 128)
    selection_mode = str(metadata.get("selection_mode") or "score_topk")
    output_rows: list[dict[str, object]] = []
    for query_id in sorted(policy_rows_by_query):
        policy_rows = policy_rows_by_query[query_id]
        image = images_by_name[query_id]
        camera = cameras[int(image.camera_id)]
        matches_by_policy_row: dict[int, QueryTo3DMatch] = {}
        for policy_row in policy_rows:
            matches_by_policy_row[policy_row] = QueryTo3DMatch(
                token_index=int(selected_rows[policy_row]),
                xy=query_xy[policy_row],
                track_id=int(tracks[policy_row]),
                xyz=np.asarray(
                    landmark_index.xyz[int(canonical_rows[policy_row])],
                    dtype=np.float64,
                ),
                similarity=float(scores[policy_row]),
                ratio=0.0,
                landmark_variance=float(
                    landmark_index.mean_variances[int(canonical_rows[policy_row])]
                ),
                source=f"selected_drop_verifier:{policy_row}",
                prototype_id=int(prototypes[policy_row]),
            )
        baseline_matches = select_pose_safe_matches(
            list(matches_by_policy_row.values()),
            max_matches=max_matches,
            image_width=int(camera.width),
            image_height=int(camera.height),
            mode=selection_mode,
        )
        match_to_policy = {
            (int(match.token_index), int(match.track_id)): policy_row
            for policy_row, match in matches_by_policy_row.items()
        }
        baseline_policy_rows = [
            match_to_policy[(int(match.token_index), int(match.track_id))]
            for match in baseline_matches
        ]
        fit_indices, verification_indices = deterministic_spatial_holdout(
            baseline_matches,
            image_width=int(camera.width),
            image_height=int(camera.height),
            folds=int(args.holdout_folds),
            fold=int(args.holdout_fold),
            grid_rows=4,
            grid_cols=4,
            salt=int.from_bytes(
                hashlib.sha256(query_id.encode("utf8")).digest()[:4],
                "little",
            ),
        )
        baseline_fit = [baseline_matches[int(index)] for index in fit_indices]
        drop_fit = [
            baseline_matches[int(index)]
            for index in fit_indices
            if str(
                actions.get(
                    baseline_policy_rows[int(index)], {"action": "KEEP"}
                )["action"]
            )
            != "DROP"
        ]
        verification_all = [
            baseline_matches[int(index)] for index in verification_indices
        ]
        verification = [
            baseline_matches[int(index)]
            for index in verification_indices
            if not bool(args.verification_keep_only)
            or str(
                actions.get(
                    baseline_policy_rows[int(index)], {"action": "KEEP"}
                )["action"]
            )
            != "DROP"
        ]
        baseline_hypothesis = _fit_pose(
            baseline_fit,
            camera,
            query_id=query_id,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        drop_hypothesis = _fit_pose(
            drop_fit,
            camera,
            query_id=query_id,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        expected_verification_count = len(verification_all)
        baseline_verification = verify_pose_hypothesis(
            baseline_hypothesis.pose_w2c,
            verification,
            camera,
            strict_threshold_px=float(args.verification_strict_px),
            loose_threshold_px=float(args.verification_loose_px),
            expected_count=expected_verification_count,
        )
        drop_verification = verify_pose_hypothesis(
            drop_hypothesis.pose_w2c,
            verification,
            camera,
            strict_threshold_px=float(args.verification_strict_px),
            loose_threshold_px=float(args.verification_loose_px),
            expected_count=expected_verification_count,
        )
        choose_drop = should_choose_drop_hypothesis(
            drop_solver_success=bool(drop_hypothesis.success),
            baseline_verification=baseline_verification,
            drop_verification=drop_verification,
            minimum_strict_inlier_gain=int(args.drop_min_strict_inlier_gain),
        )
        drop_full = [
            match
            for match, policy_row in zip(baseline_matches, baseline_policy_rows)
            if str(actions.get(policy_row, {"action": "KEEP"})["action"])
            != "DROP"
        ]
        baseline_final = _fit_pose(
            baseline_matches,
            camera,
            query_id=query_id,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        drop_final = _fit_pose(
            drop_full,
            camera,
            query_id=query_id,
            reprojection_error_px=float(args.pnp_reprojection_error_px),
            iterations=int(args.pnp_iterations),
        )
        chosen_final = drop_final if choose_drop else baseline_final
        gt_pose = np.eye(4, dtype=np.float64)
        gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
        gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
        baseline_success, baseline_t, baseline_r = _error_fields(
            baseline_final, gt_pose
        )
        drop_success, drop_t, drop_r = _error_fields(drop_final, gt_pose)
        chosen_success, chosen_t, chosen_r = _error_fields(chosen_final, gt_pose)
        output_rows.append(
            {
                "query_id": query_id,
                "chosen_policy": "drop" if choose_drop else "baseline",
                "fit_count_baseline": len(baseline_fit),
                "fit_count_drop": len(drop_fit),
                "verification_count_all": len(verification_all),
                "verification_count_used": len(verification),
                "baseline_verification_strict": (
                    None
                    if baseline_verification is None
                    else baseline_verification.strict_inlier_count
                ),
                "drop_verification_strict": (
                    None
                    if drop_verification is None
                    else drop_verification.strict_inlier_count
                ),
                "baseline_verification_soft": (
                    None
                    if baseline_verification is None
                    else baseline_verification.soft_consensus
                ),
                "drop_verification_soft": (
                    None
                    if drop_verification is None
                    else drop_verification.soft_consensus
                ),
                "baseline_success": baseline_success,
                "baseline_translation_m": baseline_t,
                "baseline_rotation_deg": baseline_r,
                "drop_success": drop_success,
                "drop_translation_m": drop_t,
                "drop_rotation_deg": drop_r,
                "chosen_success": chosen_success,
                "chosen_translation_m": chosen_t,
                "chosen_rotation_deg": chosen_r,
            }
        )

    baseline_summary = _pose_summary(output_rows, "baseline")
    drop_summary = _pose_summary(output_rows, "drop")
    chosen_summary = _pose_summary(output_rows, "chosen")
    gate = _pose_gate(chosen_summary, baseline_summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "pose_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "stage": "selected_drop_common_holdout_pose_hypothesis_verification",
        "protocol": {
            "gt_used_for_hypothesis_choice": False,
            "common_spatial_holdout": True,
            "holdout_excluded_from_hypothesis_fit": True,
            "action_context_was_computed_from_full_initial_pose": True,
            "final_pose_refit_after_policy_choice": True,
            "identity_reassignment": False,
            "coordinate_update": False,
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "split": str(args.split_name),
        "policy": {
            "holdout_folds": int(args.holdout_folds),
            "holdout_fold": int(args.holdout_fold),
            "verification_strict_px": float(args.verification_strict_px),
            "verification_loose_px": float(args.verification_loose_px),
            "verification_keep_only": bool(args.verification_keep_only),
            "drop_min_strict_inlier_gain": int(
                args.drop_min_strict_inlier_gain
            ),
        },
        "choice_counts": {
            "baseline": sum(
                row["chosen_policy"] == "baseline" for row in output_rows
            ),
            "drop": sum(row["chosen_policy"] == "drop" for row in output_rows),
        },
        "baseline_pose": baseline_summary,
        "drop_pose": drop_summary,
        "chosen_pose": chosen_summary,
        "pose_gate": gate,
        "inputs": {
            "policy_artifact": str(policy_path),
            "policy_artifact_sha256": file_sha256_short(policy_path),
            "action_predictions_csv": str(action_path),
            "action_predictions_sha256": file_sha256_short(action_path),
            "projected_landmark_bank": str(bank_path),
            "projected_landmark_bank_sha256": expected_hashes[
                "projected_landmark_bank_sha256"
            ],
            "split_json": str(split_path),
            "split_json_sha256": expected_hashes["split_json_sha256"],
            "colmap_cameras_sha256": file_sha256_short(
                model_dir / "cameras.bin"
            ),
            "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "outputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
