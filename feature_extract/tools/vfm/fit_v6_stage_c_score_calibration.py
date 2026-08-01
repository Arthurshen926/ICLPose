"""Fit a non-negative Stage-C evidence ranker on calibration trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from feature_extract.vfm.localization_v6.stage_c_score_calibration import (
    FEATURE_NAMES,
    StageCScoreCalibration,
    candidate_rank_features,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--strict_holdout_trajectory_ids", nargs="+", required=True
    )
    parser.add_argument(
        "--minimum_basin_queries",
        type=int,
        default=2,
        help=(
            "Minimum calibration queries whose candidate pool contains a "
            "30 cm / 3 degree mode. A ranker cannot learn basin retention "
            "from an all-negative calibration set."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_cost(row: Mapping[str, object]) -> float:
    return (
        float(row["final_translation_m"]) / 0.30
        + float(row["final_rotation_deg"]) / 3.0
    )


def _pairs(groups: Sequence[Sequence[Mapping[str, object]]]) -> tuple[np.ndarray, np.ndarray]:
    differences = []
    importance = []
    for rows in groups:
        features = candidate_rank_features(rows)
        costs = np.asarray([_pose_cost(row) for row in rows], dtype=np.float64)
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                gap = float(costs[first] - costs[second])
                if abs(gap) < 0.05:
                    continue
                better, worse = (
                    (first, second) if gap < 0.0 else (second, first)
                )
                differences.append(features[better] - features[worse])
                importance.append(min(abs(gap), 3.0))
    if not differences:
        raise ValueError("calibration reports contain no ordered pose pairs")
    return (
        np.asarray(differences, dtype=np.float64),
        np.asarray(importance, dtype=np.float64),
    )


def _fit(
    groups: Sequence[Sequence[Mapping[str, object]]], regularization: float
) -> np.ndarray:
    differences, importance = _pairs(groups)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = differences @ weights
        loss = np.sum(importance * np.logaddexp(0.0, -logits))
        loss += 0.5 * float(regularization) * float(weights @ weights)
        probability = 1.0 / (1.0 + np.exp(np.clip(logits, -40.0, 40.0)))
        gradient = -(differences.T @ (importance * probability))
        gradient += float(regularization) * weights
        return float(loss), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.ones((len(FEATURE_NAMES),), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(FEATURE_NAMES),
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Stage-C calibration optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights)
    return weights / float(np.sum(weights))


def _ranked_rows(
    rows: Sequence[Mapping[str, object]], weights: np.ndarray
) -> list[Mapping[str, object]]:
    scores = candidate_rank_features(rows) @ weights
    order = np.argsort(-scores, kind="mergesort")
    return [rows[int(index)] for index in order.tolist()]


def _selected_row(
    rows: Sequence[Mapping[str, object]], weights: np.ndarray
) -> Mapping[str, object]:
    return _ranked_rows(rows, weights)[0]


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    translation = np.asarray(
        [float(row["final_translation_m"]) for row in rows], dtype=np.float64
    )
    rotation = np.asarray(
        [float(row["final_rotation_deg"]) for row in rows], dtype=np.float64
    )
    return {
        "query_count": len(rows),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "recall_30cm_3deg": float(
            np.mean((translation <= 0.30) & (rotation <= 3.0))
        ),
    }


def _ranking_metrics(
    groups: Sequence[Sequence[Mapping[str, object]]],
    weights: np.ndarray,
    *,
    retained_candidates: int = 8,
) -> dict[str, object]:
    ranked = [_ranked_rows(rows, weights) for rows in groups]
    top1 = [rows[0] for rows in ranked]
    retained = [
        min(
            rows[: max(int(retained_candidates), 1)],
            key=_pose_cost,
        )
        for rows in ranked
    ]
    return {
        **{f"top1_{key}": value for key, value in _metrics(top1).items()},
        **{
            f"retained_top{int(retained_candidates)}_oracle_{key}": value
            for key, value in _metrics(retained).items()
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = [Path(value) for value in args.input_json]
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report.get("stage") != "v6_stage_c_feature_atlas_replay" for report in reports):
        raise ValueError("every input must be a Stage-C replay report")
    if any(bool(report.get("deployable_result", True)) for report in reports):
        raise ValueError("score calibration requires labelled non-deployable reports")
    atlas_hashes = {str(report.get("radio_atlas_sha256", "")) for report in reports}
    if len(atlas_hashes) != 1 or not next(iter(atlas_hashes)):
        raise ValueError("calibration reports use different atlases")
    image_ids = [str(report["image_id"]) for report in reports]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("calibration query is duplicated")
    groups = [list(report.get("rows", ())) for report in reports]
    if any(len(rows) < 2 for rows in groups):
        raise ValueError("each calibration query needs at least two candidates")
    training_ids = sorted({value.split("/", 1)[0] for value in image_ids})
    holdout_ids = sorted(set(str(value) for value in args.strict_holdout_trajectory_ids))
    if set(training_ids) & set(holdout_ids):
        raise ValueError("score calibration overlaps strict holdout")
    if len(training_ids) < 2:
        raise ValueError(
            "score calibration needs at least two calibration trajectories "
            "for trajectory-level cross-validation"
        )
    pool_oracle = [min(rows, key=_pose_cost) for rows in groups]
    basin_query_count = sum(
        float(row["final_translation_m"]) <= 0.30
        and float(row["final_rotation_deg"]) <= 3.0
        for row in pool_oracle
    )
    if basin_query_count < max(int(args.minimum_basin_queries), 1):
        raise ValueError(
            "score calibration candidate pools contain too few 30 cm / 3 "
            f"degree modes: {basin_query_count}"
        )

    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_rows = []
    for regularization in candidates:
        heldout_rankings = []
        for heldout_trajectory in training_ids:
            training = [
                rows
                for image_id, rows in zip(image_ids, groups)
                if image_id.split("/", 1)[0] != heldout_trajectory
            ]
            weights = _fit(training, regularization)
            heldout_rankings.extend(
                _ranked_rows(rows, weights)
                for image_id, rows in zip(image_ids, groups)
                if image_id.split("/", 1)[0] == heldout_trajectory
            )
        selected = [rows[0] for rows in heldout_rankings]
        retained = [
            min(rows[:8], key=_pose_cost) for rows in heldout_rankings
        ]
        cv_rows.append(
            {
                "regularization": float(regularization),
                **{
                    f"top1_{key}": value
                    for key, value in _metrics(selected).items()
                },
                **{
                    f"retained_top8_oracle_{key}": value
                    for key, value in _metrics(retained).items()
                },
            }
        )
    best = min(
        cv_rows,
        key=lambda row: (
            -float(row["retained_top8_oracle_recall_30cm_3deg"]),
            float(row["retained_top8_oracle_translation_median_m"]),
            float(row["retained_top8_oracle_translation_p90_m"]),
            float(row["top1_translation_median_m"]),
            float(row["regularization"]),
        ),
    )
    weights = _fit(groups, float(best["regularization"]))
    calibration = StageCScoreCalibration(
        feature_names=FEATURE_NAMES,
        weights=weights,
        metadata={
            "calibration_trajectory_ids": training_ids,
            "strict_holdout_trajectory_ids": holdout_ids,
            "radio_atlas_sha256": next(iter(atlas_hashes)),
            "training_report_paths": [str(path) for path in paths],
            "training_report_sha256s": [_sha256(path) for path in paths],
            "query_normalization": "tie_aware_within_candidate_percentile",
            "fit_objective": "nonnegative_pairwise_logistic_pose_cost",
            "pose_cost": "translation_m/0.30+rotation_deg/3.0",
            "cross_validation": cv_rows,
            "cross_validation_unit": "leave_one_trajectory_out",
            "selected_regularization": float(best["regularization"]),
            "calibration_pool_oracle_metrics": _metrics(pool_oracle),
            "calibration_pool_basin_query_count": int(basin_query_count),
            "retained_candidate_budget": 8,
            "training_ranking_metrics": _ranking_metrics(
                groups, weights, retained_candidates=8
            ),
            "contains_query_images_or_target_poses": False,
        },
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_json(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "weights": dict(zip(FEATURE_NAMES, weights.tolist())),
                "selected_cross_validation": best,
                "training_ranking_metrics": calibration.metadata[
                    "training_ranking_metrics"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
