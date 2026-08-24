"""Evaluate frozen natural q_pose scores against post-freeze pose labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    NATURAL_EVALUATION_SCHEMA,
    bind_scores_to_direct_labels,
    load_natural_score_artifact,
    ranked_natural_candidate_metrics,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    load_minimal_pose_transport_readout,
    pose_transport_model_content_sha256,
)


def _without_rows(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank_values", default="1,4,8,16,32,64")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite natural q_pose evaluation")
    ranks = tuple(sorted({int(value) for value in str(args.rank_values).split(",")}))
    if not ranks or ranks[0] <= 0:
        raise ValueError("natural q_pose rank values must be positive")

    score_path = Path(args.scores).resolve()
    direct_path = Path(args.direct_dataset).resolve()
    pool_path = Path(args.candidate_pool).resolve()
    model_path = Path(args.model).resolve()
    arrays, metadata = load_natural_score_artifact(score_path)
    labels = bind_scores_to_direct_labels(arrays, metadata, direct_path, pool_path)
    model, _ = load_minimal_pose_transport_readout(model_path, device="cpu")
    if (
        metadata.get("model_file_sha256") != file_sha256(model_path)
        or metadata.get("model_content_sha256") != pose_transport_model_content_sha256(model)
    ):
        raise ValueError("natural score artifact and frozen q_pose model differ")
    weights = model.edge_weights().detach().cpu().numpy().astype(np.float64)
    statistics = np.asarray(arrays["component_statistics"], dtype=np.float64)
    reconstructed = -1.0 + np.sum(statistics * weights[None, None], axis=2) / np.sum(weights)
    reconstruction_error = float(np.max(np.abs(
        reconstructed - np.asarray(arrays["scores"], dtype=np.float64)
    )))
    if reconstruction_error > 3.0e-6:
        raise ValueError("stored natural scores differ from sufficient statistics")

    image_ids = [str(value) for value in np.asarray(arrays["image_ids"]).tolist()]
    metrics = ranked_natural_candidate_metrics(
        arrays["scores"], arrays["candidate_poses_w2c"],
        labels["translation_m"], labels["rotation_deg"], arrays["candidate_valid"],
        image_ids=image_ids, ranks=ranks,
    )
    component_metrics = {}
    for name, channel in (("feature", 0), ("hierarchy", 4), ("layout", 5)):
        # Affine positive rescaling does not change a component's ranking, so
        # its exact sufficient statistic is the cleanest isolated diagnosis.
        component_metrics[name] = _without_rows(ranked_natural_candidate_metrics(
            statistics[..., channel], arrays["candidate_poses_w2c"],
            labels["translation_m"], labels["rotation_deg"], arrays["candidate_valid"],
            image_ids=image_ids, ranks=ranks,
        ))

    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    mean_mass = np.asarray(arrays["target_mean_rendered_mass"], dtype=np.float64)
    feature_fraction = np.asarray(
        arrays["target_feature_valid_mass_fraction"], dtype=np.float64
    )
    visible_fraction = np.asarray(arrays["target_visible_token_fraction"], dtype=np.float64)
    score = np.asarray(arrays["scores"], dtype=np.float64)
    full_missing = valid & (mean_mass <= 1.0e-8)
    feature_missing = valid & (
        (mean_mass <= 1.0e-8) | (feature_fraction <= 1.0e-8)
    )
    null = valid & (np.max(np.abs(statistics), axis=2) <= 1.0e-10) & (
        np.abs(score + 1.0) <= 1.0e-7
    )
    effective = valid & ~null
    selected = np.asarray([
        int(row["selected_candidate_index_zero_based_pose_free"])
        for row in metrics["rows"]
    ], dtype=np.int64)
    q = np.arange(valid.shape[0])
    score_spread = np.asarray([
        np.ptp(score[row, valid[row]]) for row in range(valid.shape[0])
    ])
    denominator = max(int(np.sum(valid)), 1)
    missingness = {
        "candidate_count": int(np.sum(valid)),
        "effective_non_null_candidate_count": int(np.sum(effective)),
        "effective_non_null_candidate_rate": float(np.sum(effective) / denominator),
        "fully_render_missing_candidate_count": int(np.sum(full_missing)),
        "fully_render_missing_candidate_rate": float(np.sum(full_missing) / denominator),
        "feature_missing_candidate_count": int(np.sum(feature_missing)),
        "feature_missing_candidate_rate": float(np.sum(feature_missing) / denominator),
        "exact_null_score_candidate_count": int(np.sum(null)),
        "exact_null_score_candidate_rate": float(np.sum(null) / denominator),
        "all_candidates_fully_missing_query_count": int(np.sum(np.all(~valid | full_missing, axis=1))),
        "all_candidates_null_query_count": int(np.sum(np.all(~valid | null, axis=1))),
        "constant_score_query_count_at_1e_8": int(np.sum(score_spread <= 1.0e-8)),
        "selected_candidate_fully_missing_rate": float(np.mean(full_missing[q, selected])),
        "selected_candidate_null_rate": float(np.mean(null[q, selected])),
        "mean_target_rendered_mass": float(np.mean(mean_mass[valid])),
        "mean_target_visible_token_fraction": float(np.mean(visible_fraction[valid])),
        "mean_target_feature_valid_mass_fraction": float(np.mean(feature_fraction[valid])),
        "minimum_source_retained_mass_fraction": float(np.min(
            np.asarray(arrays["source_retained_mass_fraction"], dtype=np.float64)
        )),
        "median_per_query_score_spread": float(np.median(score_spread)),
    }

    report: dict[str, object] = {
        "artifact_type": NATURAL_EVALUATION_SCHEMA,
        "score_artifact": str(score_path),
        "score_artifact_file_sha256": file_sha256(score_path),
        "score_artifact_content_sha256": str(metadata["content_sha256"]),
        "direct_dataset": str(direct_path),
        "direct_dataset_file_sha256": str(labels["direct_dataset_file_sha256"].item()),
        "direct_dataset_content_sha256": str(labels["direct_dataset_content_sha256"].item()),
        "candidate_pool": str(pool_path),
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_pool_content_sha256": str(metadata["candidate_pool_content_sha256"]),
        "candidate_pose_arrays_sha256": arrays_sha256({
            "candidate_poses_w2c": np.asarray(arrays["candidate_poses_w2c"]),
            "candidate_valid": np.asarray(arrays["candidate_valid"]),
        }),
        "model": str(model_path),
        "model_file_sha256": file_sha256(model_path),
        "model_content_sha256": pose_transport_model_content_sha256(model),
        "score_reconstruction_max_abs_error": reconstruction_error,
        "score_before_label_separation": {
            "candidate_zero_diagnostic_gt_anchor_absent_from_scores": True,
            "pose_error_labels_opened_only_by_this_posthoc_evaluator": True,
            "direct_candidate_dataset_opened_during_scoring": False,
            "candidate_pool_scores_consumed": False,
            "pose_errors_recomputed_from_phase2_contributor_gt": True,
            "direct_nonanchor_labels_consumed": False,
        },
        "direct_dataset_prefix_audit": {
            "exact_pose_free_pool_alignment_query_count": int(
                labels["direct_nonanchor_exact_pool_alignment_query_count"].item()
            ),
            "gt_conditioned_prefix_drift_query_count": int(
                labels[
                    "direct_nonanchor_gt_conditioned_prefix_drift_query_count"
                ].item()
            ),
            "existing_direct_nonanchor_rows_authoritative_for_k64": bool(
                int(labels[
                    "direct_nonanchor_gt_conditioned_prefix_drift_query_count"
                ].item()) == 0
            ),
        },
        "metrics": metrics,
        "component_ablation_diagnostics": component_metrics,
        "missingness_and_null": missingness,
        "strict_query_representation_route_disjoint": bool(
            metadata.get("strict_query_representation_route_disjoint")
        ),
        "non_strict_pretrained_mapper_control": bool(
            metadata.get("non_strict_pretrained_mapper_control")
        ),
        "claim": (
            "natural_pose_free_candidate_q_pose_control_not_production_localization"
            if metadata.get("non_strict_pretrained_mapper_control")
            else "natural_pose_free_candidate_q_pose_route_disjoint_evaluation"
        ),
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "query_count": metrics["query_count"],
        "q_pose_ranked_full_pool_basin_survival": metrics[
            "q_pose_ranked_full_pool_basin_survival"
        ],
        "mean_score_region_error_spearman": metrics[
            "mean_score_region_error_spearman"
        ],
        "ambiguity": metrics["ambiguity"],
        "missingness_and_null": missingness,
        "strict_query_representation_route_disjoint": report[
            "strict_query_representation_route_disjoint"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
