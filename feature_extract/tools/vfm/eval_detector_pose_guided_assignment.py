"""Two-stage detector landmark assignment using an inference-only coarse pose."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import candidate_reprojection_residuals
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import binary_average_precision
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)


def _float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--initial_strategy", default="alike_support_top2_mean")
    parser.add_argument("--geometry_mode", default="all_fit", choices=("all_fit", "cross_fit"))
    parser.add_argument("--crossfit_cell_size_px", type=float, default=64.0)
    parser.add_argument("--alpha_values", type=_float_list, default=(0.1, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--residual_cap_values", type=_float_list, default=(8.0, 16.0, 32.0, 64.0))
    parser.add_argument("--hard_threshold_values", type=_float_list, default=(0.0, 16.0, 32.0, 64.0))
    parser.add_argument("--train_query_count", type=int, default=60)
    parser.add_argument("--validation_query_count", type=int, default=15)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    return parser.parse_args(argv)


def _seed_opencv(query_id: str) -> None:
    try:
        import cv2

        seed = int.from_bytes(hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little")
        cv2.setRNGSeed(int(seed % (2**31 - 1)))
    except ImportError:
        pass


def _gt_pose(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def _matches_from_scores(
    *,
    rows: np.ndarray,
    scores: np.ndarray,
    payload: dict[str, np.ndarray],
    landmark_index,
    source: str,
) -> list[QueryTo3DMatch]:
    bank_rows = np.asarray(payload["bank_row_indices"], dtype=np.int64)
    tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
    prototypes = np.asarray(payload["candidate_prototype_ids"], dtype=np.int64)
    xy = np.asarray(payload["xy"], dtype=np.float64)
    values = np.asarray(scores, dtype=np.float32)
    matches = []
    for row in rows.tolist():
        valid = np.flatnonzero((bank_rows[row] >= 0) & np.isfinite(values[row]))
        if valid.size == 0:
            continue
        column = int(valid[np.argmax(values[row, valid])])
        bank_row = int(bank_rows[row, column])
        matches.append(
            QueryTo3DMatch(
                token_index=int(row),
                xy=xy[row].copy(),
                track_id=int(tracks[row, column]),
                xyz=landmark_index.xyz[bank_row].copy(),
                similarity=float(values[row, column]),
                ratio=0.0,
                landmark_variance=float(landmark_index.mean_variances[bank_row]),
                source=str(source),
                prototype_id=int(prototypes[row, column]),
            )
        )
    matches.sort(key=lambda match: float(match.similarity), reverse=True)
    return matches


def _estimate(
    *,
    query_id: str,
    rows: np.ndarray,
    scores: np.ndarray,
    payload: dict[str, np.ndarray],
    landmark_index,
    camera,
    reprojection_error_px: float,
    iterations: int,
    source: str,
) -> PnPResult:
    matches = _matches_from_scores(
        rows=rows,
        scores=scores,
        payload=payload,
        landmark_index=landmark_index,
        source=source,
    )
    _seed_opencv(query_id)
    return estimate_pose_pnp_ransac(
        matches,
        camera,
        reprojection_error_px=float(reprojection_error_px),
        iterations=int(iterations),
        refine_method="LM",
    )


def _summarize_results(results: dict[str, PnPResult], images_by_name) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for query_id, result in results.items():
        image = images_by_name[str(query_id)]
        error = pnp_pose_error(result.pose_w2c, _gt_pose(image))
        rows.append(
            {
                "query_id": str(query_id),
                "success": bool(result.success),
                "match_count": int(result.match_count),
                "inlier_count": int(result.inlier_count),
                "translation_m": None if not np.isfinite(error.translation_m) else float(error.translation_m),
                "rotation_deg": None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
            }
        )
    success = [row for row in rows if bool(row["success"])]
    translations = np.asarray([row["translation_m"] for row in success], dtype=np.float64)
    rotations = np.asarray([row["rotation_deg"] for row in success], dtype=np.float64)
    summary = {
        "query_count": int(len(rows)),
        "success_count": int(len(success)),
        "success_rate": float(len(success) / max(len(rows), 1)),
        "median_translation_m_success": None if not success else float(np.median(translations)),
        "p90_translation_m_success": None if not success else float(np.percentile(translations, 90.0)),
        "median_rotation_deg_success": None if not success else float(np.median(rotations)),
        "median_inliers_success": None if not success else float(np.median([row["inlier_count"] for row in success])),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
    ):
        summary[f"recall_{name}"] = float(
            np.mean(
                [
                    bool(row["success"])
                    and float(row["translation_m"]) <= distance
                    and float(row["rotation_deg"]) <= angle
                    for row in rows
                ]
            )
        )
    return summary, rows


def _strict_pose_gate(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    lower = ("median_translation_m_success", "p90_translation_m_success", "median_rotation_deg_success")
    higher = ("success_rate", "recall_25cm_2deg", "recall_10cm_5deg", "recall_5cm_5deg")
    tolerance = 1e-12
    no_regression = all(float(candidate[name]) <= float(baseline[name]) + tolerance for name in lower)
    no_regression &= all(float(candidate[name]) + tolerance >= float(baseline[name]) for name in higher)
    improved = any(float(candidate[name]) < float(baseline[name]) - tolerance for name in lower)
    improved |= any(float(candidate[name]) > float(baseline[name]) + tolerance for name in higher)
    return bool(no_regression and improved)


def _prepare_initial(
    *,
    query_ids: Sequence[str],
    payload: dict[str, np.ndarray],
    landmark_index,
    cameras,
    images_by_name,
    initial_scores: np.ndarray,
    reprojection_error_px: float,
    iterations: int,
    geometry_mode: str,
    crossfit_cell_size_px: float,
) -> tuple[dict[str, dict[str, object]], np.ndarray]:
    all_query_ids = np.asarray(payload["query_ids"]).astype(str)
    pose_keep = np.asarray(payload["pose_keep_mask"], dtype=bool)
    xy = np.asarray(payload["xy"], dtype=np.float32)
    bank_rows = np.asarray(payload["bank_row_indices"], dtype=np.int64)
    predicted_residuals = np.full(bank_rows.shape, np.inf, dtype=np.float32)
    prepared = {}
    for query_id in query_ids:
        rows = np.flatnonzero((all_query_ids == str(query_id)) & pose_keep)
        image = images_by_name[str(query_id)]
        camera = cameras[int(image.camera_id)]
        initial = _estimate(
            query_id=str(query_id), rows=rows, scores=initial_scores,
            payload=payload, landmark_index=landmark_index, camera=camera,
            reprojection_error_px=float(reprojection_error_px), iterations=int(iterations),
            source="detector_initial_assignment",
        )
        if str(geometry_mode) == "cross_fit":
            cell_size = max(float(crossfit_cell_size_px), 1.0)
            partition = (
                np.floor(xy[rows, 0] / cell_size).astype(np.int64)
                + np.floor(xy[rows, 1] / cell_size).astype(np.int64)
            ) % 2
            if np.sum(partition == 0) < 4 or np.sum(partition == 1) < 4:
                partition = np.arange(len(rows), dtype=np.int64) % 2
            fit_results = {}
            for fit_value in (0, 1):
                fit_rows = rows[partition == fit_value]
                fit_results[fit_value] = _estimate(
                    query_id=f"{query_id}:crossfit:{fit_value}",
                    rows=fit_rows,
                    scores=initial_scores,
                    payload=payload,
                    landmark_index=landmark_index,
                    camera=camera,
                    reprojection_error_px=float(reprojection_error_px),
                    iterations=int(iterations),
                    source=f"detector_crossfit_{fit_value}",
                )
            for target_value in (0, 1):
                target_rows = rows[partition == target_value]
                opposite = fit_results[1 - target_value]
                pose = opposite.pose_w2c if opposite.success else initial.pose_w2c
                if pose is not None:
                    predicted_residuals[target_rows] = candidate_reprojection_residuals(
                        xy[target_rows], bank_rows[target_rows], landmark_index, pose, camera
                    )
        elif initial.success and initial.pose_w2c is not None:
            predicted_residuals[rows] = candidate_reprojection_residuals(
                xy[rows], bank_rows[rows], landmark_index, initial.pose_w2c, camera
            )
        prepared[str(query_id)] = {"rows": rows, "camera": camera, "initial": initial}
    return prepared, predicted_residuals


def _refined_scores(
    initial_scores: np.ndarray,
    predicted_residuals: np.ndarray,
    *,
    alpha: float,
    residual_cap_px: float,
    hard_threshold_px: float | None,
) -> np.ndarray:
    scores = np.asarray(initial_scores, dtype=np.float32).copy()
    penalty = np.minimum(np.asarray(predicted_residuals, dtype=np.float32), float(residual_cap_px))
    scores -= float(alpha) * penalty / float(residual_cap_px)
    if hard_threshold_px is not None:
        scores[np.asarray(predicted_residuals) > float(hard_threshold_px)] = -np.inf
    return scores


def _evaluate_config(
    *,
    prepared: dict[str, dict[str, object]],
    scores: np.ndarray,
    payload: dict[str, np.ndarray],
    landmark_index,
    reprojection_error_px: float,
    iterations: int,
) -> dict[str, PnPResult]:
    output = {}
    for query_id, item in prepared.items():
        output[query_id] = _estimate(
            query_id=query_id,
            rows=np.asarray(item["rows"], dtype=np.int64),
            scores=scores,
            payload=payload,
            landmark_index=landmark_index,
            camera=item["camera"],
            reprojection_error_px=float(reprojection_error_px),
            iterations=int(iterations),
            source="detector_pose_guided_assignment",
        )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.proposals), allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    strategy_key = f"strategy__{args.initial_strategy}"
    if strategy_key not in payload:
        raise ValueError(f"proposal artifact is missing initial strategy: {args.initial_strategy}")
    initial_scores = np.asarray(payload[strategy_key], dtype=np.float32)
    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    valid = np.asarray(payload["bank_row_indices"], dtype=np.int64) >= 0
    if not np.array_equal(
        np.asarray(payload["candidate_track_ids"], dtype=np.int64)[valid],
        landmark_index.track_ids[np.asarray(payload["bank_row_indices"], dtype=np.int64)[valid]],
    ):
        raise ValueError("proposal and landmark bank rows differ")
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    all_ids = tuple(dict.fromkeys(np.asarray(payload["query_ids"]).astype(str).tolist()))
    train_end = int(args.train_query_count)
    validation_end = train_end + int(args.validation_query_count)
    if train_end <= 0 or validation_end >= len(all_ids):
        raise ValueError("split counts must leave a non-empty test block")
    split = {
        "strategy": "contiguous_temporal_blocks_v1",
        "train": list(all_ids[:train_end]),
        "validation": list(all_ids[train_end:validation_end]),
        "test": list(all_ids[validation_end:]),
    }
    validation_prepared, validation_predicted = _prepare_initial(
        query_ids=split["validation"], payload=payload, landmark_index=landmark_index,
        cameras=cameras, images_by_name=images_by_name, initial_scores=initial_scores,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        geometry_mode=str(args.geometry_mode), crossfit_cell_size_px=float(args.crossfit_cell_size_px),
    )
    validation_initial_results = {
        query_id: item["initial"] for query_id, item in validation_prepared.items()
    }
    validation_baseline, validation_baseline_rows = _summarize_results(
        validation_initial_results, images_by_name
    )
    trials = []
    for alpha, cap, hard_value in itertools.product(
        args.alpha_values,
        args.residual_cap_values,
        args.hard_threshold_values,
    ):
        hard = None if float(hard_value) <= 0.0 else float(hard_value)
        scores = _refined_scores(
            initial_scores,
            validation_predicted,
            alpha=float(alpha),
            residual_cap_px=float(cap),
            hard_threshold_px=hard,
        )
        results = _evaluate_config(
            prepared=validation_prepared, scores=scores, payload=payload,
            landmark_index=landmark_index,
            reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        )
        pose, _rows = _summarize_results(results, images_by_name)
        trials.append(
            {
                "alpha": float(alpha),
                "residual_cap_px": float(cap),
                "hard_threshold_px": hard,
                "pose": pose,
                "passes_strict_gate": _strict_pose_gate(pose, validation_baseline),
            }
        )
    eligible = [index for index, trial in enumerate(trials) if bool(trial["passes_strict_gate"])]
    if eligible:
        chosen_index = max(
            eligible,
            key=lambda index: (
                float(trials[index]["pose"]["recall_5cm_5deg"]),
                float(trials[index]["pose"]["recall_10cm_5deg"]),
                float(trials[index]["pose"]["recall_25cm_2deg"]),
                -float(trials[index]["pose"]["median_translation_m_success"]),
                -float(trials[index]["pose"]["p90_translation_m_success"]),
            ),
        )
        chosen = trials[chosen_index]
    else:
        chosen_index = -1
        chosen = {"alpha": 0.0, "residual_cap_px": 1.0, "hard_threshold_px": None, "pose": validation_baseline}

    test_prepared, test_predicted = _prepare_initial(
        query_ids=split["test"], payload=payload, landmark_index=landmark_index,
        cameras=cameras, images_by_name=images_by_name, initial_scores=initial_scores,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        geometry_mode=str(args.geometry_mode), crossfit_cell_size_px=float(args.crossfit_cell_size_px),
    )
    test_initial_results = {query_id: item["initial"] for query_id, item in test_prepared.items()}
    test_baseline, test_baseline_rows = _summarize_results(test_initial_results, images_by_name)
    if chosen_index >= 0:
        test_scores = _refined_scores(
            initial_scores,
            test_predicted,
            alpha=float(chosen["alpha"]),
            residual_cap_px=float(chosen["residual_cap_px"]),
            hard_threshold_px=chosen["hard_threshold_px"],
        )
        test_results = _evaluate_config(
            prepared=test_prepared, scores=test_scores, payload=payload,
            landmark_index=landmark_index,
            reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        )
    else:
        test_results = test_initial_results
    test_selected, test_selected_rows = _summarize_results(test_results, images_by_name)

    full_prepared, full_predicted = _prepare_initial(
        query_ids=all_ids, payload=payload, landmark_index=landmark_index,
        cameras=cameras, images_by_name=images_by_name, initial_scores=initial_scores,
        reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        geometry_mode=str(args.geometry_mode), crossfit_cell_size_px=float(args.crossfit_cell_size_px),
    )
    full_baseline_results = {query_id: item["initial"] for query_id, item in full_prepared.items()}
    full_baseline, full_baseline_rows = _summarize_results(full_baseline_results, images_by_name)
    if chosen_index >= 0:
        full_scores = _refined_scores(
            initial_scores,
            full_predicted,
            alpha=float(chosen["alpha"]),
            residual_cap_px=float(chosen["residual_cap_px"]),
            hard_threshold_px=chosen["hard_threshold_px"],
        )
        full_results = _evaluate_config(
            prepared=full_prepared, scores=full_scores, payload=payload,
            landmark_index=landmark_index,
            reprojection_error_px=float(args.pnp_reprojection_error_px), iterations=int(args.pnp_iterations),
        )
    else:
        full_results = full_baseline_results
    full_selected, full_selected_rows = _summarize_results(full_results, images_by_name)

    labels = np.asarray(payload["candidate_gt_residuals_px"], dtype=np.float32) <= 2.0
    diagnostics = {}
    query_ids_per_row = np.asarray(payload["query_ids"]).astype(str)
    for name, ids, predicted in (
        ("validation", split["validation"], validation_predicted),
        ("test", split["test"], test_predicted),
    ):
        row_mask = np.isin(query_ids_per_row, ids) & np.asarray(payload["pose_keep_mask"], dtype=bool)
        pair_mask = row_mask[:, None] & valid & np.isfinite(predicted)
        diagnostics[name] = {
            "pair_count": int(np.sum(pair_mask)),
            "geometry_validity_average_precision": binary_average_precision(
                labels[pair_mask], -predicted[pair_mask]
            ),
            "positive_predicted_residual_median_px": (
                None if not np.any(labels & pair_mask) else float(np.median(predicted[labels & pair_mask]))
            ),
            "negative_predicted_residual_median_px": (
                None if not np.any((~labels) & pair_mask) else float(np.median(predicted[(~labels) & pair_mask]))
            ),
        }
    residual_path = output_dir / "initial_pose_candidate_residuals.npz"
    np.savez(
        residual_path,
        validation_predicted_residuals_px=validation_predicted,
        test_predicted_residuals_px=test_predicted,
        full_predicted_residuals_px=full_predicted,
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(
        json.dumps(
            {
                "validation_baseline": validation_baseline_rows,
                "test_baseline": test_baseline_rows,
                "test_selected": test_selected_rows,
                "full90_baseline": full_baseline_rows,
                "full90_selected": full_selected_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "stage": "s4_l3_detector_pose_guided_assignment",
        "protocol": {
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "configuration_selected_on": "validation_only",
            "test_configuration_sweep": False,
            "initial_strategy": str(args.initial_strategy),
            "geometry_mode": str(args.geometry_mode),
            "crossfit_cell_size_px": float(args.crossfit_cell_size_px),
            "candidate_pool": "fixed_global_top20",
        },
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
        "split": split,
        "validation": {
            "baseline": validation_baseline,
            "trials": trials,
            "strict_gate_passed": bool(chosen_index >= 0),
            "chosen": chosen,
        },
        "test": {"baseline": test_baseline, "selected": test_selected},
        "full90": {"baseline": full_baseline, "selected": full_selected},
        "geometry_diagnostics": diagnostics,
        "runtime_seconds": float(time.time() - start_time),
        "limitations": [
            (
                "cross-fit geometry uses disjoint spatial point sets for fitting and candidate verification"
                if str(args.geometry_mode) == "cross_fit"
                else "candidate geometry is measured against the same correspondence set used for initial PnP and may self-confirm a wrong repeated-structure pose"
            ),
            "multi-hypothesis held-out verification is not active yet",
            "no local pixel measurement update is active",
        ],
        "outputs": {
            "candidate_residuals": str(residual_path),
            "pose_rows": str(pose_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
