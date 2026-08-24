"""Evaluate a frozen full-token pose ranker on a new pose-free candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _basin_recall_metrics,
    _load_feature_artifact,
    _score_queries,
    _tiered_energy_landscape_metrics,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _metrics,
    _spearman,
)
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FULLTOKEN_POSE_RANKER_SEMANTICS,
    FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS,
    load_fulltoken_candidate_pose_ranker,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)


REPORT_SCHEMA = "goal_maplet_fulltoken_pose_ranker_natural_candidate_evaluation_v1"


def _trajectory_monotonic_metrics(
    score: np.ndarray,
    seed_index: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(score, dtype=np.float64)
    seeds = np.asarray(seed_index, dtype=np.int64)
    fractions = np.asarray(alpha, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if not (values.shape == seeds.shape == fractions.shape == mask.shape):
        raise ValueError("trajectory metric arrays differ")
    consecutive_correct = consecutive_total = 0
    complete = []
    anchor_above_last = []
    correlations = []
    query_complete = []
    for query in range(values.shape[0]):
        local_complete = []
        for seed in np.unique(seeds[query, 1:]).tolist():
            if seed < 0:
                raise ValueError("trajectory metric seed groups differ")
            rows = np.flatnonzero(mask[query] & (seeds[query] == seed))
            order = rows[np.argsort(fractions[query, rows], kind="stable")]
            if order.size < 2 or np.any(np.diff(fractions[query, order]) <= 0.0):
                raise ValueError("trajectory metric alpha order differs")
            delta = np.diff(values[query, order])
            correct = delta >= -1.0e-8
            consecutive_correct += int(np.sum(correct))
            consecutive_total += int(correct.size)
            path_complete = bool(np.all(correct))
            complete.append(path_complete)
            local_complete.append(path_complete)
            anchor_above_last.append(bool(
                values[query, 0] >= values[query, order[-1]] - 1.0e-8
            ))
            correlations.append(_spearman(
                values[query, order], fractions[query, order],
            ))
        query_complete.append(bool(local_complete and all(local_complete)))
    return {
        "query_count": int(values.shape[0]),
        "seed_path_count": int(len(complete)),
        "consecutive_nondecreasing_rate": float(
            consecutive_correct / max(consecutive_total, 1)
        ),
        "complete_path_nondecreasing_rate": float(np.mean(complete)),
        "query_all_paths_nondecreasing_rate": float(np.mean(query_complete)),
        "anchor_above_last_path_sample_rate": float(np.mean(anchor_above_last)),
        "mean_score_alpha_spearman": float(np.mean(correlations)),
    }


def _domain_retention_metrics(
    score: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """Report broad basin retention separately from exact-centre success."""

    values = np.asarray(score, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation = np.asarray(rotation_deg, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != translation.shape or values.shape != rotation.shape or values.shape != mask.shape:
        raise ValueError("domain-retention arrays differ")
    order = []
    for row in range(values.shape[0]):
        indices = np.flatnonzero(mask[row])
        indices = indices[indices != 0]
        order.append(indices[np.lexsort((indices, -values[row, indices]))])
    maximum = max((value.size for value in order), default=0)
    result = {}
    for name, maximum_translation_m, maximum_rotation_deg in (
        ("medium_2m_20deg", 2.0, 20.0),
        ("declared_2m_45deg", 2.0, 45.0),
        ("wide_8m_45deg", 8.0, 45.0),
    ):
        raw_hits = []
        ranked_hits = {k: [] for k in (1, 4, 8, 16) if k <= maximum + 1}
        for row, ranked in enumerate(order):
            domain = (
                mask[row]
                & (translation[row] <= maximum_translation_m)
                & (rotation[row] <= maximum_rotation_deg)
            )
            domain[0] = False
            raw_hits.append(bool(np.any(domain)))
            for k in ranked_hits:
                ranked_hits[k].append(bool(np.any(domain[ranked[:k]])))
        result[name] = {
            "maximum_translation_m": maximum_translation_m,
            "maximum_rotation_deg": maximum_rotation_deg,
            "raw_candidate_domain_recall": float(np.mean(raw_hits)),
            **{
                f"ranked_domain_recall_at_{k}": float(np.mean(hits))
                for k, hits in ranked_hits.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--batch_queries", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output_report)
    if output.exists():
        raise FileExistsError("refusing to overwrite natural-candidate ranker evaluation")
    if int(args.batch_queries) <= 0:
        raise ValueError("batch_queries must be positive")
    dataset_path = Path(args.dataset)
    arrays, dataset_metadata = load_pose_candidate_dataset(
        dataset_path, require_rendered_targets=False,
    )
    if dataset_metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True:
        raise ValueError("evaluation dataset lacks a diagnostic GT anchor")
    features, feature_manifest = _load_feature_artifact(
        Path(args.features), Path(args.feature_manifest),
        dataset_path=dataset_path,
        dataset_content_sha256=str(dataset_metadata["content_sha256"]),
    )
    if features.shape[:2] != arrays["candidate_valid"].shape:
        raise ValueError("natural candidate features and pose inventory differ")

    device = torch.device(str(args.device))
    model, model_metadata = load_fulltoken_candidate_pose_ranker(
        Path(args.model), device=device,
    )
    if (
        model_metadata.get("model_semantics") != FULLTOKEN_POSE_RANKER_SEMANTICS
        or feature_manifest.get("feature_semantics") != FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS
    ):
        raise ValueError("model and feature semantics differ")
    rows = np.arange(arrays["image_ids"].size, dtype=np.int64)
    scores = _score_queries(
        model, features, rows, batch_queries=int(args.batch_queries), device=device,
    )
    retrieval_order_scores = np.broadcast_to(
        -np.arange(scores.shape[1], dtype=np.float32)[None], scores.shape,
    )
    learned_metrics = _metrics(
        scores, arrays["translation_m"], arrays["rotation_deg"],
        arrays["candidate_valid"], arrays["image_ids"],
    )
    retrieval_metrics = _metrics(
        retrieval_order_scores, arrays["translation_m"], arrays["rotation_deg"],
        arrays["candidate_valid"], arrays["image_ids"],
    )
    report = {
        "artifact_type": REPORT_SCHEMA,
        "dataset_file_sha256": file_sha256(dataset_path),
        "dataset_content_sha256": dataset_metadata["content_sha256"],
        "candidate_pool_frozen_before_target_pose_opened": dataset_metadata.get(
            "candidate_pool_frozen_before_target_pose_opened"
        ),
        "feature_manifest_file_sha256": file_sha256(Path(args.feature_manifest)),
        "feature_file_sha256": feature_manifest["feature_file_sha256"],
        "model_file_sha256": file_sha256(Path(args.model)),
        "model_content_sha256": model_metadata["model_content_sha256"],
        "model_training_dataset_content_sha256": model_metadata.get("dataset_content_sha256"),
        "evaluation_dataset_is_training_dataset": (
            model_metadata.get("dataset_content_sha256") == dataset_metadata["content_sha256"]
        ),
        "query_count": int(rows.size),
        "candidate_count": int(scores.shape[1]),
        "learned_metrics": learned_metrics,
        "learned_distinct_basin_recall": _basin_recall_metrics(scores, arrays, rows),
        "learned_domain_retention": _domain_retention_metrics(
            scores, arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"],
        ),
        "learned_tiered_energy_landscape": _tiered_energy_landscape_metrics(
            scores, arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"],
        ),
        "learned_trajectory_monotonic_metrics": (
            None if "trajectory_seed_candidate_index" not in arrays else
            _trajectory_monotonic_metrics(
                scores, arrays["trajectory_seed_candidate_index"],
                arrays["trajectory_alpha"], arrays["candidate_valid"],
            )
        ),
        "retrieval_order_metrics": retrieval_metrics,
        "retrieval_order_distinct_basin_recall": _basin_recall_metrics(
            retrieval_order_scores, arrays, rows,
        ),
        "retrieval_order_domain_retention": _domain_retention_metrics(
            retrieval_order_scores, arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"],
        ),
        "candidate_pose_values_are_model_inputs": False,
        "candidate_pose_errors_are_model_inputs": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "natural_pose_free_candidate_reranking_diagnostic_not_final_localization",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_report": str(output.resolve()),
        "learned": {key: value for key, value in learned_metrics.items() if key != "rows"},
        "retrieval_order": {
            key: value for key, value in retrieval_metrics.items() if key != "rows"
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
