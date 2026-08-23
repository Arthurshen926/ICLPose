"""Evaluate physically distinct phase-ranked pose basins without refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _load_dataset,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCHEMA = "goal_maplet_phase_ranked_physical_basin_acquisition_v1"


def _physical_nms(
    poses_w2c: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_count: int,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> tuple[list[int], int]:
    candidates = np.flatnonzero(np.asarray(valid, dtype=bool))
    # Candidate zero is a diagnostic GT anchor and is never available to the
    # pose-free selector.
    candidates = candidates[candidates != 0]
    order = candidates[np.argsort(-np.asarray(scores)[candidates], kind="stable")]
    retained: list[int] = []
    duplicate_count = 0
    for candidate in order.tolist():
        duplicate = False
        for previous in retained:
            translation, rotation = _pose_errors(
                np.asarray(poses_w2c[candidate])[None], np.asarray(poses_w2c[previous])
            )
            if (
                float(translation[0]) <= float(translation_threshold_m) + 1.0e-6
                and float(rotation[0]) <= float(rotation_threshold_deg) + 1.0e-5
            ):
                duplicate = True
                duplicate_count += 1
                break
        if not duplicate:
            retained.append(int(candidate))
            if len(retained) >= int(maximum_count):
                break
    return retained, duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum_k", type=int, default=4)
    parser.add_argument("--translation_nms_m", type=float, default=0.5)
    parser.add_argument("--rotation_nms_deg", type=float, default=5.0)
    parser.add_argument("--query_start", type=int, default=0)
    parser.add_argument("--maximum_queries", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite phase basin evaluation")
    arrays, metadata = _load_dataset(Path(args.dataset))
    phase = json.loads(Path(args.phase_report).read_text())
    if (
        phase.get("dataset_file_sha256") != file_sha256(Path(args.dataset))
        or phase.get("score_semantics") != "conservative_phase"
        or not phase.get("all_candidate_scores_built_before_pose_error_metrics", False)
    ):
        raise ValueError("phase report does not bind the frozen dataset/scorer")
    scores = np.asarray(phase["candidate_score"], dtype=np.float64)
    if scores.shape != arrays["candidate_valid"].shape or np.any(~np.isfinite(scores)):
        raise ValueError("phase candidate score matrix differs from the dataset")
    begin = int(args.query_start)
    end = int(arrays["image_ids"].size)
    if args.maximum_queries is not None:
        end = min(end, begin + int(args.maximum_queries))
    if not 0 <= begin < end:
        raise ValueError("query range is empty")
    k_values = tuple(range(1, int(args.maximum_k) + 1))
    rows = []
    for query in range(begin, end):
        per_k = {}
        for k in k_values:
            retained, duplicates = _physical_nms(
                arrays["candidate_poses_w2c"][query], scores[query],
                arrays["candidate_valid"][query], maximum_count=k,
                translation_threshold_m=float(args.translation_nms_m),
                rotation_threshold_deg=float(args.rotation_nms_deg),
            )
            translation = arrays["translation_m"][query, retained]
            rotation = arrays["rotation_deg"][query, retained]
            per_k[str(k)] = {
                "retained_candidate_indices": retained,
                "retained_count": len(retained),
                "duplicate_rejections_before_completion": int(duplicates),
                "strict_acquired": bool(np.any(
                    (translation <= 0.5 + 1.0e-6) & (rotation <= 5.0 + 1.0e-5)
                )),
                "loose_acquired": bool(np.any(
                    (translation <= 1.0 + 1.0e-6) & (rotation <= 10.0 + 1.0e-5)
                )),
            }
        rows.append({"image_id": str(arrays["image_ids"][query]), "per_k": per_k})
    metrics = {}
    for k in k_values:
        values = [row["per_k"][str(k)] for row in rows]
        metrics[str(k)] = {
            "strict_acquisition_rate": float(np.mean([value["strict_acquired"] for value in values])),
            "loose_acquisition_rate": float(np.mean([value["loose_acquired"] for value in values])),
            "mean_retained_physical_basin_count": float(np.mean([value["retained_count"] for value in values])),
            "mean_duplicate_rejections_before_completion": float(np.mean([
                value["duplicate_rejections_before_completion"] for value in values
            ])),
        }
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "phase_report_file_sha256": file_sha256(Path(args.phase_report)),
        "energy_semantics": phase["energy_semantics"],
        "query_range": [begin, end],
        "physical_nms": {
            "translation_m": float(args.translation_nms_m),
            "rotation_deg": float(args.rotation_nms_deg),
            "same_basin_requires_both_thresholds": True,
        },
        "metrics_by_k": metrics,
        "rows": rows,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "production_eligible": False,
        "claim": "map_disjoint_phase_ranked_physical_basin_acquisition_not_top1_success",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
