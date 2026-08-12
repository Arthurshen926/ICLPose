"""Low-capacity calibration features for selected GoalMaplet outputs.

The features in this module are deliberately measured *after* proposal,
pruning, exact ranking, refinement and winner selection.  They are neither
independent likelihood ratios nor a continuous SE(3) posterior.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES: tuple[str, ...] = (
    "selected_final_score",
    "selected_vs_best_other_margin",
    "selected_vs_baseline_gap",
    "selected_refinement_gain",
    "cross_splat_selected_margin",
    "cross_splat_argmax_agreement",
    "log_refined_candidate_count",
    "selected_common_initial_rank_log",
    "parent_out_of_map_mean",
    "parent_truncated_in_map_tail_mean",
    "mapping_view_typed_null_probability",
    "selected_exact_rendered_coverage",
    "selected_exact_feature_coverage",
    "query_geometry_confidence_mean",
    "mapping_view_anchor_entropy_normalized",
    "selected_vs_distinct_basin_margin",
    "log_distinct_refined_basin_count",
)


SOURCE_EVIDENCE_NAMES: tuple[str, ...] = (
    "parent_out_of_map_mean",
    "parent_truncated_in_map_tail_mean",
    "mapping_view_typed_null_probability",
    "selected_exact_rendered_coverage",
    "selected_exact_feature_coverage",
    "query_geometry_confidence_mean",
    "mapping_view_anchor_entropy_normalized",
)


def _normalized_softmax_entropy(scores: object) -> float:
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if value.size <= 1:
        return 0.0
    probability = np.exp(value - float(np.max(value)))
    probability /= float(np.sum(probability))
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1.0e-12))))
    return float(entropy / np.log(value.size))


def candidate_source_postselection_evidence(
    row: dict[str, object],
    *,
    mode_name: str = "actual_parent_actual_child",
) -> dict[str, float]:
    """Extract query-local evidence without reading any pose-error fields."""

    posterior = row.get("posterior_mass", {})
    geometry = row.get("query_geometry", {})
    diagnostics = row.get("proposal_diagnostics", {})
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get(mode_name), dict):
        diagnostics = diagnostics[mode_name]
    if not isinstance(posterior, dict):
        posterior = {}
    if not isinstance(geometry, dict):
        geometry = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    mapping = diagnostics.get("mapping_view_posterior", {})
    exact = diagnostics.get("disconnected_seed_vfm_likelihood", {})
    if not isinstance(mapping, dict):
        mapping = {}
    if not isinstance(exact, dict):
        exact = {}
    return {
        "parent_out_of_map_mean": float(posterior.get("out_of_map_mean", 0.0)),
        "parent_truncated_in_map_tail_mean": float(
            posterior.get("truncated_in_map_tail_mean", 0.0)
        ),
        "mapping_view_typed_null_probability": float(
            mapping.get("typed_null_probability", 0.0)
        ),
        "selected_exact_rendered_coverage": float(
            exact.get(
                "top1_rendered_coverage",
                exact.get("rendered_coverage_mean", 0.0),
            )
        ),
        "selected_exact_feature_coverage": float(
            exact.get(
                "top1_feature_coverage",
                exact.get("feature_coverage_mean", 0.0),
            )
        ),
        "query_geometry_confidence_mean": float(
            geometry.get("confidence_mean", 0.0)
        ),
        "mapping_view_anchor_entropy_normalized": _normalized_softmax_entropy(
            mapping.get("anchor_scores", [])
        ),
    }


def _rotation_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64)[:3, :3] @ np.asarray(
        right, dtype=np.float64
    )[:3, :3].T
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _distinct_basin_diagnostics(
    refinements: list[dict[str, object]], selected: dict[str, object],
) -> tuple[float, int]:
    if "pose_w2c" not in selected:
        return 0.0, 0
    selected_pose = np.asarray(selected["pose_w2c"], dtype=np.float64).reshape(4, 4)
    selected_center = -selected_pose[:3, :3].T @ selected_pose[:3, 3]
    distinct = []
    representatives: list[np.ndarray] = [selected_pose]
    for value in sorted(
        refinements, key=lambda item: float(item["final_score"]), reverse=True
    ):
        pose = np.asarray(value.get("pose_w2c"), dtype=np.float64).reshape(4, 4)
        center = -pose[:3, :3].T @ pose[:3, 3]
        if (
            np.linalg.norm(center - selected_center) >= 0.5
            or _rotation_distance_degrees(pose, selected_pose) >= 5.0
        ):
            distinct.append(value)
        if all(
            np.linalg.norm(center - (-other[:3, :3].T @ other[:3, 3])) >= 0.5
            or _rotation_distance_degrees(pose, other) >= 5.0
            for other in representatives
        ):
            representatives.append(pose)
    margin = (
        float(selected["final_score"])
        - max(float(value["final_score"]) for value in distinct)
        if distinct
        else 0.0
    )
    return float(margin), len(representatives)


def typed_null_diagnostics(row: dict[str, object]) -> dict[str, object]:
    """Expose typed null measurements without pretending they are probabilities."""

    source = row.get("postselection_source_evidence", {})
    if not isinstance(source, dict):
        source = {}
    rendered = float(source.get("selected_exact_rendered_coverage", 0.0))
    feature = float(source.get("selected_exact_feature_coverage", 0.0))
    refinements = list(row.get("candidate_refinements", []))
    selected_matches = [
        value for value in refinements
        if int(value.get("union_candidate_index", -1))
        == int(row.get("selected_union_candidate_index", -2))
    ]
    basin_margin, basin_count = (
        _distinct_basin_diagnostics(refinements, selected_matches[0])
        if len(selected_matches) == 1 else (0.0, 0)
    )
    return {
        "semantics": "typed_uncalibrated_measurements_not_independent_likelihoods",
        "measurements": {
            "out_of_map": float(source.get("parent_out_of_map_mean", 0.0)),
            "in_map_but_canonical_field_unsupported_fraction": float(
                np.clip((rendered - feature) / max(rendered, 1.0e-8), 0.0, 1.0)
            ),
            "low_quality_query": float(
                np.clip(1.0 - float(
                    source.get("query_geometry_confidence_mean", 0.0)
                ), 0.0, 1.0)
            ),
            "outside_mapping_view_manifold": float(
                source.get("mapping_view_typed_null_probability", 0.0)
            ),
            "repeated_or_symmetric_ambiguity_distinct_basin_margin": basin_margin,
            "distinct_refined_basin_count": int(basin_count),
        },
        "physically_resolved_not_scalar_calibrated": {
            "supported_but_occluded": (
                "all physical geometry participates in the z-buffer; hidden "
                "feature surfaces cannot contribute descriptor evidence"
            )
        },
    }


def postselection_feature_row(row: dict[str, object]) -> np.ndarray:
    refinements = list(row.get("candidate_refinements", []))
    source = row.get("postselection_source_evidence", {})
    if not isinstance(source, dict):
        source = {}
    source_values = [float(source.get(name, 0.0)) for name in SOURCE_EVIDENCE_NAMES]
    if not refinements:
        # A gate passthrough has no refinement competition. Missing diagnostic
        # values are encoded explicitly as zeros, not inferred from pose error.
        return np.asarray([
            float(row["final_score"]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            *source_values, 0.0, 0.0,
        ], dtype=np.float64)
    selected_index = int(row["selected_union_candidate_index"])
    selected_matches = [
        value
        for value in refinements
        if int(value["union_candidate_index"]) == selected_index
    ]
    baseline_matches = [
        value
        for value in refinements
        if int(value["union_candidate_index"]) == 0
    ]
    if len(selected_matches) != 1 or len(baseline_matches) != 1:
        raise ValueError("post-selection row lacks a unique selected/baseline state")
    selected = selected_matches[0]
    baseline = baseline_matches[0]
    other_scores = [
        float(value["final_score"])
        for value in refinements
        if int(value["union_candidate_index"]) != selected_index
    ]
    best_other = max(other_scores) if other_scores else float(selected["final_score"])
    has_validation = all("validation_score" in value for value in refinements)
    if has_validation:
        primary_argmax = int(max(
            refinements, key=lambda value: float(value["final_score"])
        )["union_candidate_index"])
        validation_argmax = int(max(
            refinements, key=lambda value: float(value["validation_score"])
        )["union_candidate_index"])
        validation_other = max(
            [
                float(value["validation_score"])
                for value in refinements
                if int(value["union_candidate_index"]) != selected_index
            ]
            or [float(selected["validation_score"])]
        )
        validation_margin = float(selected["validation_score"]) - validation_other
        agreement = float(primary_argmax == validation_argmax)
    else:
        validation_margin = 0.0
        agreement = 0.0
    distinct_margin, distinct_count = _distinct_basin_diagnostics(
        refinements, selected
    )
    return np.asarray([
            float(selected["final_score"]),
            float(selected["final_score"]) - best_other,
            float(selected["final_score"]) - float(baseline["final_score"]),
            float(selected["final_score"]) - float(selected["initial_score"]),
            validation_margin,
            agreement,
            np.log1p(len(refinements)),
            np.log1p(int(selected["common_initial_rank"])),
            *source_values,
            distinct_margin,
            np.log1p(distinct_count),
        ], dtype=np.float64)


def sigmoid_logistic_payload(model: object, mean: np.ndarray, scale: np.ndarray) -> dict[str, object]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "standardization_mean": np.asarray(mean, dtype=np.float64).tolist(),
        "standardization_scale": np.asarray(scale, dtype=np.float64).tolist(),
        "coefficient": np.asarray(model.coef_[0], dtype=np.float64).tolist(),
        "intercept": float(model.intercept_[0]),
        "inference": "sigmoid(intercept + coefficient @ ((z - mean) / scale))",
    }


def predict_sigmoid_logistic_payload(
    payload: dict[str, object], features: np.ndarray,
) -> np.ndarray:
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("success calibrator feature contract differs")
    value = np.asarray(features, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise ValueError("success calibrator feature shape differs")
    mean = np.asarray(payload["standardization_mean"], dtype=np.float64)
    scale = np.asarray(payload["standardization_scale"], dtype=np.float64)
    coefficient = np.asarray(payload["coefficient"], dtype=np.float64)
    if mean.shape != coefficient.shape or scale.shape != coefficient.shape:
        raise ValueError("success calibrator parameter shape differs")
    logit = float(payload["intercept"]) + (
        (value - mean[None, :]) / np.maximum(scale[None, :], 1.0e-12)
    ) @ coefficient
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))
