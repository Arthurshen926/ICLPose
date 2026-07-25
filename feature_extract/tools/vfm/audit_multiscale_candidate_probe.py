"""Audit frozen S1 multiscale candidate probabilities without held-out labels.

The S1 probe may use train-only geometric sets or train-only registered SfM
track identities.  This command deliberately joins validation/test geometry
and registered identity only after inference artifacts have been frozen.  It
reports the two target definitions separately because an exact registered
observation is stricter than a geometrically valid candidate at a detector
anchor that is not itself an SfM keypoint.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT,
    ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT,
    DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT,
    GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT,
    HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
)


FEATURE_ARTIFACT_FORMAT = "multiscale_candidate_probe_features_v1"
STRUCTURED_FEATURE_ARTIFACT_FORMAT = "structured_multiscale_candidate_probe_features_v1"
STRUCTURED_FEATURE_ARTIFACT_FORMAT_V2 = "structured_multiscale_candidate_probe_features_v2"
STRUCTURED_FEATURE_ARTIFACT_FORMATS = frozenset(
    {STRUCTURED_FEATURE_ARTIFACT_FORMAT, STRUCTURED_FEATURE_ARTIFACT_FORMAT_V2}
)
COST_VOLUME_FEATURE_ARTIFACT_FORMAT = "cost_volume_multiscale_candidate_probe_features_v1"
WIDE_FULL_CORRELATION_FEATURE_ARTIFACT_FORMAT = (
    "wide_full_correlation_multiscale_candidate_probe_features_v1"
)
GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT, GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT}
)
LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMATS = frozenset(
    {
        LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
        MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT,
    }
)
DENSE_LOCAL_MODE_FEATURE_ARTIFACT_FORMATS = frozenset(
    {DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT}
)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT}
)
ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT}
)
COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS = frozenset(
    {
        *STRUCTURED_FEATURE_ARTIFACT_FORMATS,
        COST_VOLUME_FEATURE_ARTIFACT_FORMAT,
        WIDE_FULL_CORRELATION_FEATURE_ARTIFACT_FORMAT,
        *GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS,
        *LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMATS,
        *DENSE_LOCAL_MODE_FEATURE_ARTIFACT_FORMATS,
        *ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMATS,
        *ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMATS,
    }
)
PREDICTION_ARTIFACT_FORMATS = frozenset(
    {
        "multiscale_candidate_probe_predictions_v1",
        "multiscale_candidate_probe_predictions_v2",
    }
)
GEOMETRIC_PROBABILITY_SEMANTICS = (
    "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one"
)
EXACT_IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
GEOMETRIC_SET_SUPERVISION_MODE = "geometric_set"
REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE = "registered_track_identity"
SUPPORTED_SUPERVISION_MODES = frozenset(
    {GEOMETRIC_SET_SUPERVISION_MODE, REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE}
)
AUDIT_SPLIT_NAMES = ("train", "validation", "test", "all")
_BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES = (
    (
        "bidirectional_absolute_v2",
        "bidirectional_absolute_visual_v2",
        "bidirectional_absolute_position_control_v2",
    ),
    (
        "bidirectional_absolute_raw_v3",
        "bidirectional_absolute_raw_visual_v3",
        "bidirectional_absolute_raw_position_control_v3",
    ),
    (
        "bidirectional_absolute_raw_layout_v4",
        "bidirectional_absolute_raw_layout_visual_v4",
        "bidirectional_absolute_raw_layout_position_control_v4",
    ),
    (
        "bidirectional_absolute_dual_head_raw_layout_v5",
        "bidirectional_absolute_dual_head_raw_layout_visual_v5",
        "bidirectional_absolute_dual_head_raw_layout_position_control_v5",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--base_prior_overlay", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geometric_positive_threshold_px", type=float, default=2.0)
    parser.add_argument("--exact_identity_radius_px", type=float, default=2.0)
    parser.add_argument(
        "--audit_splits",
        default=",".join(AUDIT_SPLIT_NAMES),
        help="comma-separated audit-only target splits; use validation to avoid test label materialization",
    )
    parser.add_argument(
        "--allow_diagnostic_feature_artifact",
        action="store_true",
        help="allow a --max_queries smoke artifact only for code diagnostics",
    )
    return parser.parse_args(argv)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} has no metadata_json")
    value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _parse_audit_splits(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = (
        tuple(item.strip() for item in str(value).split(","))
        if isinstance(value, str)
        else tuple(str(item).strip() for item in value)
    )
    splits = tuple(item for item in raw if item)
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(AUDIT_SPLIT_NAMES):
        raise ValueError("audit splits must be unique members of train, validation, test, all")
    return splits


def _allowed_soft_global_context(metadata: Mapping[str, Any]) -> bool:
    """Recognize only fixed candidate/support-view global context evidence."""

    artifact_format = str(metadata.get("format"))
    expected_usage = GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT.get(artifact_format)
    return bool(
        expected_usage is not None
        and metadata.get("whole_image_summary_or_global_used") is True
        and metadata.get("soft_global_context_factor") is True
        and metadata.get("global_context_usage") == expected_usage
        and metadata.get("global_context_hard_retrieval_or_candidate_reselection") is False
    )


def _allowed_prediction_soft_global_context(metadata: Mapping[str, Any]) -> bool:
    """Accept only fixed candidate/view region-token global evidence."""

    protocol = metadata.get("source_feature_protocol")
    if not isinstance(protocol, Mapping):
        return False
    return bool(
        protocol.get("whole_image_summary_or_global_used") is True
        and protocol.get("soft_global_context_factor_used") is True
        and protocol.get("candidate_conditioned_full_image_region_tokens") is True
        and protocol.get("global_context_hard_retrieval_or_candidate_reselection")
        is False
        and protocol.get("image_retrieval_or_submap_used") is False
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("average-precision inputs are invalid")
    positive_count = int(np.sum(target))
    if positive_count == 0:
        return None
    order = np.argsort(-values, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positive_count)


def _candidate_top_and_rank(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return top candidate indices and first-positive ranks (1-based, -1 absent)."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != candidate_valid.shape:
        raise ValueError("candidate score, label, and valid shapes differ")
    if np.any(~np.isfinite(values[candidate_valid])):
        raise ValueError("valid candidate scores must be finite")
    if np.any(positive & ~candidate_valid):
        raise ValueError("a positive candidate is invalid")
    if np.any(np.sum(candidate_valid, axis=1) == 0):
        raise ValueError("every audit row needs at least one valid candidate")
    ranked_scores = np.where(candidate_valid, values, -np.inf)
    order = np.argsort(-ranked_scores, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive, order, axis=1)
    has_positive = np.any(ranked_positive, axis=1)
    first_rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    first_rank[~has_positive] = -1
    top = order[:, 0].astype(np.int64)
    return top, first_rank


def _probability_contract(
    probability: np.ndarray, null_probability: np.ndarray, valid: np.ndarray
) -> None:
    candidate = np.asarray(probability, dtype=np.float64)
    null = np.asarray(null_probability, dtype=np.float64).reshape(-1)
    candidate_valid = np.asarray(valid, dtype=bool)
    if candidate.ndim != 2 or candidate.shape != candidate_valid.shape or null.shape != (
        len(candidate),
    ):
        raise ValueError("candidate plus null probability arrays are incompatible")
    if (
        np.any(~np.isfinite(candidate[candidate_valid]))
        or np.any(~np.isfinite(null))
        or np.any(candidate[candidate_valid] < 0.0)
        or np.any(null < 0.0)
        or np.any(np.abs(candidate[~candidate_valid]) > 1e-6)
    ):
        raise ValueError("candidate plus null probabilities are invalid")
    mass = candidate.sum(axis=1) + null
    if np.max(np.abs(mass - 1.0)) > 1e-4:
        raise ValueError("candidate plus null probabilities do not conserve mass")


def _set_valued_metrics(
    probability: np.ndarray,
    null_probability: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a geometry set target, with null correct on no-positive rows."""

    candidate = np.asarray(probability, dtype=np.float64)
    null = np.asarray(null_probability, dtype=np.float64).reshape(-1)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    if not (
        candidate.shape == positive.shape == candidate_valid.shape
        and null.shape == selected.shape == candidate.shape[:1]
    ):
        raise ValueError("set-valued audit arrays are incompatible")
    _probability_contract(candidate, null, candidate_valid)
    if not np.any(selected):
        raise ValueError("audit split has no rows")

    row_positive = np.any(positive, axis=1)
    target_mass = np.where(
        row_positive,
        np.sum(np.where(positive, candidate, 0.0), axis=1),
        null,
    )
    top, first_rank = _candidate_top_and_rank(candidate, positive, candidate_valid)
    combined = np.concatenate(
        [np.where(candidate_valid, candidate, -np.inf), null[:, None]], axis=1
    )
    prediction = np.argmax(combined, axis=1)
    group_correct = prediction == candidate.shape[1]
    # Positive rows require that the selected candidate belongs to the entire
    # valid-geometry set; no-positive rows require the explicit null choice.
    group_correct[row_positive] = (
        (prediction[row_positive] < candidate.shape[1])
        & positive[row_positive, prediction[row_positive].clip(max=candidate.shape[1] - 1)]
    )
    selected_positive = selected & row_positive
    selected_valid = candidate_valid[selected]
    selected_labels = positive[selected]
    selected_scores = candidate[selected]
    selected_null_label = ~row_positive[selected]
    return {
        "row_count": int(np.sum(selected)),
        "positive_row_count": int(np.sum(selected_positive)),
        "positive_row_rate": float(np.mean(row_positive[selected])),
        "positive_candidate_count": int(np.sum(selected_labels)),
        "candidate_edge_positive_rate": float(np.mean(selected_labels[selected_valid])),
        "candidate_pair_average_precision": _average_precision(
            selected_labels[selected_valid], selected_scores[selected_valid]
        ),
        "null_average_precision": _average_precision(selected_null_label, null[selected]),
        "group_target_nll": float(
            np.mean(-np.log(np.clip(target_mass[selected], 1e-12, None)))
        ),
        "group_argmax_correct_rate": float(np.mean(group_correct[selected])),
        "top1_geometry_valid_rate_given_positive": (
            None
            if not np.any(selected_positive)
            else float(
                np.mean(positive[selected_positive, top[selected_positive]])
            )
        ),
        "median_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.median(first_rank[selected_positive]))
        ),
        "p90_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.quantile(first_rank[selected_positive], 0.9))
        ),
        "mean_target_probability_mass": float(np.mean(target_mass[selected])),
    }


def _exact_identity_metrics(
    probability: np.ndarray,
    null_probability: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    supervised: np.ndarray,
) -> dict[str, Any]:
    """Evaluate strict registered-track rank without treating unsupervised rows as null."""

    candidate = np.asarray(probability, dtype=np.float64)
    null = np.asarray(null_probability, dtype=np.float64).reshape(-1)
    exact = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    observed = np.asarray(supervised, dtype=bool).reshape(-1)
    if (
        candidate.shape != exact.shape
        or exact.shape != candidate_valid.shape
        or observed.shape != (len(candidate),)
        or null.shape != (len(candidate),)
    ):
        raise ValueError("exact-identity audit arrays are incompatible")
    _probability_contract(candidate, null, candidate_valid)
    if np.any(exact & ~observed[:, None]):
        raise ValueError("exact labels require a registered query observation")
    top, first_rank = _candidate_top_and_rank(candidate, exact, candidate_valid)
    retrieved = observed & np.any(exact, axis=1)
    observed_count = int(np.sum(observed))
    exact_edge_mask = observed[:, None] & candidate_valid
    return {
        "registered_observation_row_count": observed_count,
        "registered_observation_row_rate": float(np.mean(observed)),
        "exact_candidate_retrieved_row_count": int(np.sum(retrieved)),
        "exact_candidate_recall_given_registered": (
            None if observed_count == 0 else float(np.mean(retrieved[observed]))
        ),
        "exact_candidate_pair_average_precision": _average_precision(
            exact[exact_edge_mask], candidate[exact_edge_mask]
        ),
        "top1_exact_rate_given_retrieved": (
            None
            if not np.any(retrieved)
            else float(np.mean(exact[retrieved, top[retrieved]]))
        ),
        "median_exact_rank_when_retrieved": (
            None
            if not np.any(retrieved)
            else float(np.median(first_rank[retrieved]))
        ),
        "p90_exact_rank_when_retrieved": (
            None
            if not np.any(retrieved)
            else float(np.quantile(first_rank[retrieved], 0.9))
        ),
    }


def _paired_rank_audit(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare fixed candidate rankings only where a positive exists."""

    base = np.asarray(baseline, dtype=np.float64)
    probe = np.asarray(candidate, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    requested = np.asarray(row_mask, dtype=bool).reshape(-1)
    if not (
        base.shape == probe.shape == positive.shape == candidate_valid.shape
        and requested.shape == base.shape[:1]
    ):
        raise ValueError("paired rank audit arrays are incompatible")
    _, baseline_rank = _candidate_top_and_rank(base, positive, candidate_valid)
    _, probe_rank = _candidate_top_and_rank(probe, positive, candidate_valid)
    selected = requested & (baseline_rank > 0) & (probe_rank > 0)
    if not np.any(selected):
        return {
            "positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
        }
    base_rank = baseline_rank[selected]
    new_rank = probe_rank[selected]
    return {
        "positive_row_count": int(np.sum(selected)),
        "rank_win_count": int(np.sum(new_rank < base_rank)),
        "rank_loss_count": int(np.sum(new_rank > base_rank)),
        "rank_tie_count": int(np.sum(new_rank == base_rank)),
        "median_rank_delta_baseline_minus_probe": float(
            np.median(base_rank - new_rank)
        ),
        "top1_rescue_count": int(np.sum((base_rank > 1) & (new_rank == 1))),
        "top1_harm_count": int(np.sum((base_rank == 1) & (new_rank > 1))),
    }


def _rank2_to_l_rescue_audit(
    baseline: np.ndarray,
    probe: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    """Audit candidates that the frozen baseline leaves below rank one.

    This is deliberately a *diagnostic* subset, not a new training target or
    a claim that every row is a repeated facade.  It directly answers whether
    the added appearance evidence rescues a geometrically valid alternative
    already present in the fixed top-L pool.
    """

    base = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(probe, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    requested = np.asarray(row_mask, dtype=bool).reshape(-1)
    if not (
        base.shape == candidate.shape == positive.shape == candidate_valid.shape
        and requested.shape == base.shape[:1]
    ):
        raise ValueError("rank2-to-L rescue audit arrays are incompatible")
    _, baseline_rank = _candidate_top_and_rank(base, positive, candidate_valid)
    _, probe_rank = _candidate_top_and_rank(candidate, positive, candidate_valid)
    selected = requested & (baseline_rank >= 2)
    if not np.any(selected):
        return {
            "eligible_positive_row_count": 0,
            "baseline": {
                "candidate_pair_average_precision": None,
                "top1_geometry_valid_rate": None,
                "median_first_positive_rank": None,
                "p90_first_positive_rank": None,
            },
            "probe": {
                "candidate_pair_average_precision": None,
                "top1_geometry_valid_rate": None,
                "median_first_positive_rank": None,
                "p90_first_positive_rank": None,
            },
            "paired_rank": _paired_rank_audit(
                base,
                candidate,
                labels=positive,
                valid=candidate_valid,
                row_mask=selected,
            ),
        }

    def metrics(probability: np.ndarray, ranks: np.ndarray) -> dict[str, Any]:
        top, _ = _candidate_top_and_rank(probability, positive, candidate_valid)
        edge_mask = selected[:, None] & candidate_valid
        return {
            "candidate_pair_average_precision": _average_precision(
                positive[edge_mask], probability[edge_mask]
            ),
            "top1_geometry_valid_rate": float(np.mean(positive[selected, top[selected]])),
            "median_first_positive_rank": float(np.median(ranks[selected])),
            "p90_first_positive_rank": float(np.quantile(ranks[selected], 0.9)),
        }

    return {
        "eligible_positive_row_count": int(np.sum(selected)),
        "baseline": metrics(base, baseline_rank),
        "probe": metrics(candidate, probe_rank),
        "paired_rank": _paired_rank_audit(
            base,
            candidate,
            labels=positive,
            valid=candidate_valid,
            row_mask=selected,
        ),
    }


def _visual_vs_position_control_pre_gate(
    *,
    position_probability: np.ndarray,
    position_null: np.ndarray,
    visual_probability: np.ndarray,
    visual_null: np.ndarray,
    geometry_labels: np.ndarray,
    exact_labels: np.ndarray,
    exact_supervised: np.ndarray,
    candidate_valid: np.ndarray,
    position_identity_probability: np.ndarray | None = None,
    position_identity_null: np.ndarray | None = None,
    visual_identity_probability: np.ndarray | None = None,
    visual_identity_null: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare actual visual evidence with its matched positional control.

    A model that merely learns image-coordinate priors can beat the immutable
    coarse distribution while adding no absolute appearance evidence.  This is
    deliberately a *candidate* pre-gate only: it never promotes a pose model,
    whose frozen-hypothesis rank/tail gate remains stricter and separate.
    """

    all_rows = np.ones((len(position_probability),), dtype=bool)
    geometry_position = _set_valued_metrics(
        position_probability,
        position_null,
        labels=geometry_labels,
        valid=candidate_valid,
        row_mask=all_rows,
    )
    geometry_visual = _set_valued_metrics(
        visual_probability,
        visual_null,
        labels=geometry_labels,
        valid=candidate_valid,
        row_mask=all_rows,
    )
    geometry_paired = _paired_rank_audit(
        position_probability,
        visual_probability,
        labels=geometry_labels,
        valid=candidate_valid,
        row_mask=all_rows,
    )
    rescue = _rank2_to_l_rescue_audit(
        position_probability,
        visual_probability,
        labels=geometry_labels,
        valid=candidate_valid,
        row_mask=all_rows,
    )
    identity_values = (
        position_identity_probability,
        position_identity_null,
        visual_identity_probability,
        visual_identity_null,
    )
    if any(value is not None for value in identity_values) and not all(
        value is not None for value in identity_values
    ):
        raise ValueError("identity control probabilities must be all present or all absent")
    if position_identity_probability is None:
        position_identity_probability = position_probability
        position_identity_null = position_null
        visual_identity_probability = visual_probability
        visual_identity_null = visual_null
    assert position_identity_null is not None
    assert visual_identity_probability is not None
    assert visual_identity_null is not None
    _probability_contract(
        np.asarray(position_identity_probability, dtype=np.float64),
        np.asarray(position_identity_null, dtype=np.float64),
        candidate_valid,
    )
    _probability_contract(
        np.asarray(visual_identity_probability, dtype=np.float64),
        np.asarray(visual_identity_null, dtype=np.float64),
        candidate_valid,
    )
    exact_position = _exact_identity_metrics(
        position_identity_probability,
        position_identity_null,
        labels=exact_labels,
        valid=candidate_valid,
        supervised=exact_supervised,
    )
    exact_visual = _exact_identity_metrics(
        visual_identity_probability,
        visual_identity_null,
        labels=exact_labels,
        valid=candidate_valid,
        supervised=exact_supervised,
    )
    exact_paired = _paired_rank_audit(
        position_identity_probability,
        visual_identity_probability,
        labels=exact_labels,
        valid=candidate_valid,
        row_mask=exact_supervised,
    )
    gate = {
        "geometry_nll_improved": bool(
            geometry_visual["group_target_nll"] < geometry_position["group_target_nll"]
        ),
        "geometry_candidate_ap_improved": bool(
            geometry_visual["candidate_pair_average_precision"]
            > geometry_position["candidate_pair_average_precision"]
        ),
        "geometry_paired_rank_wins_exceed_losses": bool(
            geometry_paired["rank_win_count"] > geometry_paired["rank_loss_count"]
        ),
        "geometry_top1_rescues_not_below_harms": bool(
            geometry_paired["top1_rescue_count"] >= geometry_paired["top1_harm_count"]
        ),
        "rank2_to_l_wins_exceed_losses": bool(
            rescue["paired_rank"]["rank_win_count"]
            > rescue["paired_rank"]["rank_loss_count"]
        ),
        "exact_identity_ap_not_degraded": bool(
            exact_visual["exact_candidate_pair_average_precision"]
            >= exact_position["exact_candidate_pair_average_precision"]
        ),
    }
    gate["passed"] = bool(all(gate.values()))
    gate["policy"] = (
        "validation-only visual-versus-position candidate pre-gate; passing is "
        "necessary but never sufficient for frozen hypothesis pose promotion"
    )
    return {
        "position_control": {
            "geometry_set": geometry_position,
            "exact_registered_identity": exact_position,
        },
        "visual": {
            "geometry_set": geometry_visual,
            "exact_registered_identity": exact_visual,
        },
        "geometry_paired_rank": geometry_paired,
        "exact_identity_paired_rank": exact_paired,
        "rank2_to_l_geometry_rescue": rescue,
        "exact_identity_probability_source": (
            "separate_identity_side_head"
            if identity_values[0] is not None
            else "geometry_probability_shared_legacy_head"
        ),
        "candidate_pre_gate": gate,
    }


def _masked_view_log_mean_numpy(
    values: np.ndarray, view_valid: np.ndarray
) -> np.ndarray:
    """Marginalize fixed support views without giving padded views a score.

    The context probe is trained per candidate/view.  A scale ablation must
    preserve that mixture rather than average support embeddings or substitute
    a zero logit for a missing view.  Invalid candidates receive a neutral
    residual; their immutable base-prior mass is already zero.
    """

    logits = np.asarray(values, dtype=np.float64)
    valid = np.asarray(view_valid, dtype=bool)
    if logits.ndim != 3 or logits.shape != valid.shape:
        raise ValueError("per-scale view logits and validity masks differ")
    count = np.sum(valid, axis=2)
    masked = np.where(valid, logits, -np.inf)
    maximum = np.max(masked, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shifted = np.where(np.isfinite(maximum), masked - maximum, -np.inf)
        log_mean = maximum[:, :, 0] + np.log(
            np.maximum(np.sum(np.exp(shifted), axis=2), 1e-300)
        ) - np.log(np.maximum(count, 1))
    return np.where(count > 0, log_mean, 0.0)


def _probabilities_from_base_prior_and_residual(
    *,
    base_candidate: np.ndarray,
    base_null: np.ndarray,
    candidate_residual: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a diagnostic posterior with a common immutable null reference.

    Per-scale outputs intentionally omit the learned joint null head: assigning
    that head to one scale would falsely attribute candidate-set evidence to
    that scale.  Every visual/control ablation therefore uses the same frozen
    base-null logit.  These probabilities are diagnostic only and cannot be
    promoted as a production overlay.
    """

    candidate = np.asarray(base_candidate, dtype=np.float64)
    null = np.asarray(base_null, dtype=np.float64).reshape(-1)
    residual = np.asarray(candidate_residual, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool)
    if (
        candidate.ndim != 2
        or candidate.shape != residual.shape
        or candidate.shape != valid.shape
        or null.shape != (len(candidate),)
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.any(~np.isfinite(residual[valid]))
        or np.any(candidate[valid] <= 0.0)
        or np.any(candidate[~valid] != 0.0)
    ):
        raise ValueError("base-prior residual diagnostic inputs are invalid")
    candidate_logits = np.where(valid, np.log(candidate) + residual, -np.inf)
    null_logits = np.log(np.maximum(null, 1e-300))
    logits = np.concatenate((candidate_logits, null_logits[:, None]), axis=1)
    maximum = np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits - maximum)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities[:, :-1], probabilities[:, -1]


def _per_scale_visual_control_diagnostic(
    *,
    visual_per_scale_view_logits: np.ndarray,
    position_per_scale_view_logits: np.ndarray,
    scale_names: Sequence[str],
    base_candidate: np.ndarray,
    base_null: np.ndarray,
    view_valid: np.ndarray,
    geometry_labels: np.ndarray,
    exact_labels: np.ndarray,
    exact_supervised: np.ndarray,
    candidate_valid: np.ndarray,
    visual_identity_per_scale_view_logits: np.ndarray | None = None,
    position_identity_per_scale_view_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Attribute a paired V2--V4 candidate result to frozen feature scales.

    This is deliberately an after-the-fact *validation diagnostic*, not a
    model-selection mechanism: its common base-null reference is not the
    learned joint null likelihood, and no resulting scale subset is allowed to
    enter pose scoring without a separately predeclared training run.
    """

    visual = np.asarray(visual_per_scale_view_logits, dtype=np.float64)
    position = np.asarray(position_per_scale_view_logits, dtype=np.float64)
    names = tuple(str(name) for name in scale_names)
    if (
        visual.ndim != 4
        or position.shape != visual.shape
        or visual.shape[:3] != np.asarray(view_valid, dtype=bool).shape
        or visual.shape[3] != len(names)
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("per-scale visual/control diagnostic inputs are incompatible")
    identity_per_scale_values = (
        visual_identity_per_scale_view_logits,
        position_identity_per_scale_view_logits,
    )
    if any(value is not None for value in identity_per_scale_values) and not all(
        value is not None for value in identity_per_scale_values
    ):
        raise ValueError("per-scale identity inputs must be both present or both absent")
    has_identity_side_head = visual_identity_per_scale_view_logits is not None
    identity_visual: np.ndarray | None = None
    identity_position: np.ndarray | None = None
    if has_identity_side_head:
        identity_visual = np.asarray(
            visual_identity_per_scale_view_logits, dtype=np.float64
        )
        identity_position = np.asarray(
            position_identity_per_scale_view_logits, dtype=np.float64
        )
        if identity_visual.shape != visual.shape or identity_position.shape != visual.shape:
            raise ValueError("per-scale identity tensors differ from geometry tensors")
    scale_residuals_visual = [
        _masked_view_log_mean_numpy(visual[..., index], view_valid)
        for index in range(len(names))
    ]
    scale_residuals_position = [
        _masked_view_log_mean_numpy(position[..., index], view_valid)
        for index in range(len(names))
    ]
    identity_scale_residuals_visual = (
        [
            _masked_view_log_mean_numpy(identity_visual[..., index], view_valid)
            for index in range(len(names))
        ]
        if identity_visual is not None
        else None
    )
    identity_scale_residuals_position = (
        [
            _masked_view_log_mean_numpy(identity_position[..., index], view_valid)
            for index in range(len(names))
        ]
        if identity_position is not None
        else None
    )
    result: dict[str, Any] = {
        "diagnostic_only": True,
        "null_reference": "immutable_base_null_only_no_learned_joint_null_head",
        "promotion_allowed": False,
        "scale_names": list(names),
        "combinations": {},
    }
    for width in range(1, len(names) + 1):
        for indices in combinations(range(len(names)), width):
            key = "+".join(names[index] for index in indices)
            visual_probability, visual_null = _probabilities_from_base_prior_and_residual(
                base_candidate=base_candidate,
                base_null=base_null,
                candidate_residual=np.sum(
                    [scale_residuals_visual[index] for index in indices], axis=0
                ),
                candidate_valid=candidate_valid,
            )
            position_probability, position_null = _probabilities_from_base_prior_and_residual(
                base_candidate=base_candidate,
                base_null=base_null,
                candidate_residual=np.sum(
                    [scale_residuals_position[index] for index in indices], axis=0
                ),
                candidate_valid=candidate_valid,
            )
            visual_identity_probability: np.ndarray | None = None
            visual_identity_null: np.ndarray | None = None
            position_identity_probability: np.ndarray | None = None
            position_identity_null: np.ndarray | None = None
            if identity_scale_residuals_visual is not None:
                assert identity_scale_residuals_position is not None
                visual_identity_probability, visual_identity_null = (
                    _probabilities_from_base_prior_and_residual(
                        base_candidate=base_candidate,
                        base_null=base_null,
                        candidate_residual=np.sum(
                            [
                                identity_scale_residuals_visual[index]
                                for index in indices
                            ],
                            axis=0,
                        ),
                        candidate_valid=candidate_valid,
                    )
                )
                position_identity_probability, position_identity_null = (
                    _probabilities_from_base_prior_and_residual(
                        base_candidate=base_candidate,
                        base_null=base_null,
                        candidate_residual=np.sum(
                            [
                                identity_scale_residuals_position[index]
                                for index in indices
                            ],
                            axis=0,
                        ),
                        candidate_valid=candidate_valid,
                    )
                )
            result["combinations"][key] = _visual_vs_position_control_pre_gate(
                position_probability=position_probability,
                position_null=position_null,
                visual_probability=visual_probability,
                visual_null=visual_null,
                geometry_labels=geometry_labels,
                exact_labels=exact_labels,
                exact_supervised=exact_supervised,
                candidate_valid=candidate_valid,
                position_identity_probability=position_identity_probability,
                position_identity_null=position_identity_null,
                visual_identity_probability=visual_identity_probability,
                visual_identity_null=visual_identity_null,
            )
    return result


def _validate_frozen_inputs(
    *,
    features: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    base_overlay: Mapping[str, np.ndarray],
    proposals: Mapping[str, np.ndarray],
    features_path: Path,
    predictions_path: Path,
    proposals_path: Path,
    base_overlay_path: Path,
    allow_diagnostic_feature_artifact: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate prediction lineage before loading any audit-only target arrays."""

    feature_metadata = _metadata(features, context="S1 feature artifact")
    prediction_metadata = _metadata(predictions, context="S1 prediction artifact")
    base_metadata = _metadata(base_overlay, context="base prior overlay")
    if feature_metadata.get("format") not in {
        FEATURE_ARTIFACT_FORMAT,
        *COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS,
    }:
        raise ValueError("unsupported S1 feature artifact format")
    if (
        int(feature_metadata.get("diagnostic_max_queries", 0)) > 0
        or int(feature_metadata.get("diagnostic_max_rows", 0)) > 0
    ) and not bool(allow_diagnostic_feature_artifact):
        raise ValueError("refusing to audit a diagnostic feature artifact")
    if feature_metadata.get("format") in COMPLETE_FROZEN_LAYOUT_FEATURE_ARTIFACT_FORMATS and feature_metadata.get(
        "is_complete_frozen_layout"
    ) is not True:
        raise ValueError("structured S1 feature artifact is not a fully merged frozen layout")
    if feature_metadata.get("contains_ground_truth") is not False or feature_metadata.get(
        "pose_or_ground_truth_used"
    ) is not False:
        raise ValueError("S1 feature artifact is not target-free")
    if bool(feature_metadata.get("image_retrieval_or_submap_used", True)):
        raise ValueError("S1 feature artifact violates the no-retrieval protocol")
    if bool(feature_metadata.get("whole_image_summary_or_global_used", True)) and not _allowed_soft_global_context(
        feature_metadata
    ):
        raise ValueError("S1 feature artifact has an unapproved whole-image context path")
    if feature_metadata.get("format") in GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMATS and not _allowed_soft_global_context(
        feature_metadata
    ):
        raise ValueError("global-context S1 feature artifact lacks its strict soft-factor manifest")
    if prediction_metadata.get("format") not in PREDICTION_ARTIFACT_FORMATS:
        raise ValueError("unsupported S1 prediction artifact format")
    if prediction_metadata.get("contains_ground_truth") is not False or prediction_metadata.get(
        "contains_target_errors"
    ) is not False:
        raise ValueError("S1 prediction artifact contains audit labels")
    supervision_mode = str(
        prediction_metadata.get("supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE)
    )
    probability_semantics = str(prediction_metadata.get("probability_semantics"))
    expected_semantics = {
        GEOMETRIC_SET_SUPERVISION_MODE: GEOMETRIC_PROBABILITY_SEMANTICS,
        REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE: EXACT_IDENTITY_PROBABILITY_SEMANTICS,
    }
    if supervision_mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError("S1 prediction has an unsupported supervision mode")
    if probability_semantics != expected_semantics[supervision_mode]:
        raise ValueError("S1 prediction supervision and probability semantics disagree")
    uses_separate_identity_head = bool(
        prediction_metadata.get("separate_exact_identity_head", False)
    )
    identity_array_names = {
        "identity_candidate_probabilities",
        "identity_null_probabilities",
        "identity_view_logits",
        "identity_per_scale_view_logits",
        "identity_view_log_probabilities",
        "identity_candidate_log_likelihood_ratios",
        "identity_null_log_likelihood_ratios",
    }
    if uses_separate_identity_head:
        if (
            str(prediction_metadata.get("identity_probability_semantics"))
            != EXACT_IDENTITY_PROBABILITY_SEMANTICS
            or prediction_metadata.get("identity_candidate_probability_allowed_for_pnp_overlay")
            is not False
        ):
            raise ValueError("S1 dual-head identity semantics are unsafe")
        missing_identity = identity_array_names - set(predictions)
        if missing_identity:
            raise ValueError(f"S1 dual-head prediction lacks {sorted(missing_identity)}")
    elif identity_array_names & set(predictions):
        raise ValueError("S1 legacy prediction unexpectedly carries a dual identity side head")
    prediction_source_protocol = prediction_metadata.get("source_feature_protocol")
    if isinstance(prediction_source_protocol, Mapping) and bool(
        prediction_source_protocol.get("whole_image_summary_or_global_used", False)
    ) and not _allowed_prediction_soft_global_context(prediction_metadata):
        raise ValueError("S1 prediction has an unapproved global evidence path")
    if str(prediction_metadata.get("features_sha256")) != str(
        file_sha256_short(features_path)
    ) or str(prediction_metadata.get("proposals_sha256")) != str(
        file_sha256_short(proposals_path)
    ) or str(prediction_metadata.get("base_prior_overlay_sha256")) != str(
        file_sha256_short(base_overlay_path)
    ):
        raise ValueError("S1 prediction artifact lineage does not match its inputs")
    if str(base_metadata.get("proposals_sha256")) != str(file_sha256_short(proposals_path)):
        raise ValueError("base prior overlay references different proposals")
    required_feature = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "candidate_track_ids",
        "candidate_view_valid",
    }
    required_prediction = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_view_valid",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
    }
    required_base = {"candidate_track_ids", "candidate_probabilities", "null_probabilities"}
    required_proposal = {"query_ids", "xy", "candidate_track_ids"}
    for name, payload, required in (
        ("features", features, required_feature),
        ("predictions", predictions, required_prediction),
        ("base overlay", base_overlay, required_base),
        ("proposals", proposals, required_proposal),
    ):
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{name} lacks {sorted(missing)}")
    rows = np.asarray(features["source_row_indices"], dtype=np.int64).reshape(-1)
    if len(rows) == 0 or np.unique(rows).size != len(rows):
        raise ValueError("S1 source rows are empty or duplicated")
    if not np.array_equal(rows, np.asarray(predictions["source_row_indices"], dtype=np.int64)):
        raise ValueError("prediction source rows differ from feature source rows")
    for key in ("query_ids", "split_names", "candidate_track_ids", "candidate_view_valid"):
        if not np.array_equal(features[key], predictions[key]):
            raise ValueError(f"prediction {key} differs from frozen feature artifact")
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    feature_tracks = np.asarray(features["candidate_track_ids"], dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(proposal_tracks)) or not np.array_equal(
        proposal_tracks[rows], feature_tracks
    ):
        raise ValueError("S1 feature candidates differ from proposal rows")
    base_tracks = np.asarray(base_overlay["candidate_track_ids"], dtype=np.int64)
    if not np.array_equal(base_tracks, proposal_tracks):
        raise ValueError("base overlay candidate tracks differ from proposals")
    valid = feature_tracks >= 0
    view_valid = np.asarray(features["candidate_view_valid"], dtype=bool)
    if view_valid.ndim != 3 or view_valid.shape[:2] != valid.shape:
        raise ValueError("S1 candidate support-view mask is invalid")
    if np.any(valid & ~np.any(view_valid, axis=2)):
        raise ValueError("S1 valid candidate has no real support view")
    family_names = np.asarray(predictions["family_names"]).astype(str).reshape(-1)
    probability = np.asarray(predictions["candidate_probabilities"], dtype=np.float64)
    null_probability = np.asarray(predictions["null_probabilities"], dtype=np.float64)
    if (
        probability.ndim != 3
        or probability.shape[0] != len(family_names)
        or probability.shape[1:] != valid.shape
        or null_probability.shape != (len(family_names), len(rows))
        or len(set(family_names.tolist())) != len(family_names)
    ):
        raise ValueError("S1 prediction probability arrays are invalid")
    for family_index in range(len(family_names)):
        _probability_contract(probability[family_index], null_probability[family_index], valid)
    if uses_separate_identity_head:
        identity_probability = np.asarray(
            predictions["identity_candidate_probabilities"], dtype=np.float64
        )
        identity_null_probability = np.asarray(
            predictions["identity_null_probabilities"], dtype=np.float64
        )
        identity_view_logits = np.asarray(predictions["identity_view_logits"], dtype=np.float64)
        identity_per_scale_view_logits = np.asarray(
            predictions["identity_per_scale_view_logits"], dtype=np.float64
        )
        identity_view_log_probabilities = np.asarray(
            predictions["identity_view_log_probabilities"], dtype=np.float64
        )
        identity_candidate_llr = np.asarray(
            predictions["identity_candidate_log_likelihood_ratios"], dtype=np.float64
        )
        identity_null_llr = np.asarray(
            predictions["identity_null_log_likelihood_ratios"], dtype=np.float64
        )
        if (
            identity_probability.shape != probability.shape
            or identity_null_probability.shape != null_probability.shape
            or identity_view_logits.shape
            != (len(family_names), len(rows), valid.shape[1], view_valid.shape[2])
            or identity_view_log_probabilities.shape != identity_view_logits.shape
            or identity_per_scale_view_logits.ndim != 5
            or identity_per_scale_view_logits.shape[:4] != identity_view_logits.shape
            or identity_candidate_llr.shape != probability.shape
            or identity_null_llr.shape != null_probability.shape
            or not np.all(np.isfinite(identity_view_logits))
            or not np.all(np.isfinite(identity_per_scale_view_logits))
            or not np.all(np.isfinite(identity_view_log_probabilities))
            or not np.all(np.isfinite(identity_candidate_llr))
            or not np.all(np.isfinite(identity_null_llr))
        ):
            raise ValueError("S1 dual-head identity prediction arrays are invalid")
        for family_index in range(len(family_names)):
            _probability_contract(
                identity_probability[family_index],
                identity_null_probability[family_index],
                valid,
            )
    _probability_contract(
        np.asarray(base_overlay["candidate_probabilities"], dtype=np.float64)[rows],
        np.asarray(base_overlay["null_probabilities"], dtype=np.float64)[rows],
        valid,
    )
    return feature_metadata, prediction_metadata, base_metadata


def audit_multiscale_candidate_probe(
    *,
    features_path: Path,
    predictions_path: Path,
    proposals_path: Path,
    base_overlay_path: Path,
    colmap_model_dir: Path,
    geometric_positive_threshold_px: float,
    exact_identity_radius_px: float,
    allow_diagnostic_feature_artifact: bool,
    audit_splits: Sequence[str] = AUDIT_SPLIT_NAMES,
) -> dict[str, Any]:
    if float(geometric_positive_threshold_px) <= 0.0 or float(exact_identity_radius_px) <= 0.0:
        raise ValueError("S1 audit thresholds must be positive")
    requested_splits = _parse_audit_splits(audit_splits)
    features = _load_npz(features_path)
    predictions = _load_npz(predictions_path)
    base_overlay = _load_npz(base_overlay_path)
    # This read is intentionally target-free.  GT residuals are loaded only
    # after the frozen inference artifact has passed every lineage check.
    with np.load(proposals_path, allow_pickle=False) as data:
        proposals = {
            key: np.asarray(data[key])
            for key in ("query_ids", "xy", "candidate_track_ids")
            if key in data.files
        }
    feature_metadata, prediction_metadata, base_metadata = _validate_frozen_inputs(
        features=features,
        predictions=predictions,
        base_overlay=base_overlay,
        proposals=proposals,
        features_path=features_path,
        predictions_path=predictions_path,
        proposals_path=proposals_path,
        base_overlay_path=base_overlay_path,
        allow_diagnostic_feature_artifact=allow_diagnostic_feature_artifact,
    )
    prediction_source_protocol = prediction_metadata.get("source_feature_protocol")
    prediction_global_context = _allowed_prediction_soft_global_context(prediction_metadata)
    soft_global_context = _allowed_soft_global_context(feature_metadata) or prediction_global_context

    rows = np.asarray(features["source_row_indices"], dtype=np.int64)
    query_ids = np.asarray(features["query_ids"]).astype(str)
    split_names = np.asarray(features["split_names"]).astype(str)
    query_xy = np.asarray(features["xy"], dtype=np.float32)
    candidate_tracks = np.asarray(features["candidate_track_ids"], dtype=np.int64)
    valid = candidate_tracks >= 0
    view_valid = np.asarray(features["candidate_view_valid"], dtype=bool)
    if view_valid.shape[:2] != candidate_tracks.shape or view_valid.ndim != 3:
        raise ValueError("frozen candidate support-view mask is invalid")
    family_names = np.asarray(predictions["family_names"]).astype(str)
    probe_probability = np.asarray(predictions["candidate_probabilities"], dtype=np.float64)
    probe_null = np.asarray(predictions["null_probabilities"], dtype=np.float64)
    uses_separate_identity_head = bool(
        prediction_metadata.get("separate_exact_identity_head", False)
    )
    identity_probe_probability = (
        np.asarray(predictions["identity_candidate_probabilities"], dtype=np.float64)
        if uses_separate_identity_head
        else None
    )
    identity_probe_null = (
        np.asarray(predictions["identity_null_probabilities"], dtype=np.float64)
        if uses_separate_identity_head
        else None
    )
    baseline_probability = np.asarray(base_overlay["candidate_probabilities"], dtype=np.float64)[rows]
    baseline_null = np.asarray(base_overlay["null_probabilities"], dtype=np.float64)[rows]

    split_masks = {
        split_name: (
            np.ones((len(rows),), dtype=bool)
            if split_name == "all"
            else split_names == split_name
        )
        for split_name in requested_splits
    }
    if any(not np.any(mask) for mask in split_masks.values()):
        raise ValueError("a requested audit split has no frozen rows")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    split_targets: dict[str, dict[str, object]] = {}
    # Labels are materialized only for requested splits after frozen prediction
    # lineage has been validated.  This allows a validation gate without
    # touching test correspondence targets.
    with np.load(proposals_path, allow_pickle=False) as data:
        if "candidate_gt_residuals_px" not in data.files:
            raise ValueError("audit proposals lack candidate_gt_residuals_px")
        residual_source = data["candidate_gt_residuals_px"]
        for split_name, mask in split_masks.items():
            selected_rows = rows[mask]
            selected_tracks = candidate_tracks[mask]
            selected_valid = valid[mask]
            residuals = np.asarray(residual_source[selected_rows], dtype=np.float32)
            if residuals.shape != selected_tracks.shape or np.any(np.isnan(residuals)):
                raise ValueError("audit geometric residuals do not align with candidate rows")
            exact_targets = registered_query_observation_targets(
                query_ids=query_ids[mask],
                query_xy=query_xy[mask],
                images_by_name=images_by_name,
                max_distance_px=float(exact_identity_radius_px),
            )
            split_targets[split_name] = {
                "mask": mask,
                "valid": selected_valid,
                "geometry_labels": selected_valid
                & np.isfinite(residuals)
                & (residuals <= float(geometric_positive_threshold_px)),
                "exact_targets": exact_targets,
                "exact_labels": registered_candidate_identity_labels(
                    selected_tracks, exact_targets
                ),
            }
    materialized_target_splits = sorted(
        {
            actual
            for requested, mask in split_masks.items()
            for actual in (
                ("train", "validation", "test") if requested == "all" else (requested,)
            )
            if np.any(split_names[mask] == actual)
        }
    )

    output: dict[str, Any] = {
        "stage": "S1_frozen_multiscale_candidate_probe_external_audit",
        "protocol": {
            "prediction_artifact_frozen_before_validation_test_label_join": True,
            "fit_uses_train_targets_only": bool(
                prediction_metadata.get("validation_or_test_labels_used_by_fit") is False
                and prediction_metadata.get("training_supervision_split") == "train"
            ),
            "fit_uses_train_geometric_targets_only": bool(
                prediction_metadata.get("validation_or_test_labels_used_by_fit") is False
                and prediction_metadata.get("training_supervision_split") == "train"
                and prediction_metadata.get("supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE)
                == GEOMETRIC_SET_SUPERVISION_MODE
            ),
            "fit_uses_train_registered_identity_targets_only": bool(
                prediction_metadata.get("validation_or_test_labels_used_by_fit") is False
                and prediction_metadata.get("training_supervision_split") == "train"
                and (
                    prediction_metadata.get("supervision_mode")
                    == REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE
                    or uses_separate_identity_head
                )
            ),
            "training_supervision_mode": prediction_metadata.get(
                "supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE
            ),
            "prediction_probability_semantics": prediction_metadata.get(
                "probability_semantics"
            ),
            "query_pose_or_target_used_by_feature_export": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": bool(
                feature_metadata.get("whole_image_summary_or_global_used")
                or (
                    isinstance(prediction_source_protocol, Mapping)
                    and prediction_source_protocol.get("whole_image_summary_or_global_used")
                )
            ),
            "soft_global_context_factor_used": bool(soft_global_context),
            "global_context_hard_retrieval_or_candidate_reselection": (
                prediction_source_protocol.get(
                    "global_context_hard_retrieval_or_candidate_reselection"
                )
                if isinstance(prediction_source_protocol, Mapping)
                else feature_metadata.get("global_context_hard_retrieval_or_candidate_reselection")
            ),
            "candidate_conditioned_full_image_region_tokens": bool(
                isinstance(prediction_source_protocol, Mapping)
                and prediction_source_protocol.get(
                    "candidate_conditioned_full_image_region_tokens"
                )
            ),
            "render": False,
            "exact_identity_is_strict_diagnostic_not_train_target": bool(
                prediction_metadata.get(
                    "supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE
                )
                != REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE
                and not uses_separate_identity_head
            ),
            "separate_exact_identity_side_head": bool(uses_separate_identity_head),
            "identity_side_head_allowed_for_pnp_overlay": bool(
                prediction_metadata.get(
                    "identity_candidate_probability_allowed_for_pnp_overlay", False
                )
            ),
            "test_used_for_model_selection": False,
            "audit_target_splits": list(requested_splits),
            "materialized_target_splits": materialized_target_splits,
            "test_target_labels_materialized": "test" in materialized_target_splits,
        },
        "thresholds": {
            "geometric_positive_px": float(geometric_positive_threshold_px),
            "exact_registered_identity_px": float(exact_identity_radius_px),
        },
        "inputs": {
            "features": str(features_path),
            "features_sha256": file_sha256_short(features_path),
            "predictions": str(predictions_path),
            "predictions_sha256": file_sha256_short(predictions_path),
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "base_prior_overlay": str(base_overlay_path),
            "base_prior_overlay_sha256": file_sha256_short(base_overlay_path),
            "colmap_model_dir": str(colmap_model_dir),
            "feature_protocol": {
                key: feature_metadata.get(key)
                for key in (
                    "support_view_selection",
                    "image_retrieval_or_submap_used",
                    "whole_image_summary_or_global_used",
                    "soft_global_context_factor",
                    "global_context_usage",
                    "global_context_hard_retrieval_or_candidate_reselection",
                    "diagnostic_max_queries",
                )
            },
            "prediction_source_feature_protocol": prediction_source_protocol,
            "base_probability_semantics": base_metadata.get("probability_semantics"),
        },
        "families": {},
    }
    for family_index, family_name in enumerate(family_names.tolist()):
        family_result: dict[str, Any] = {"splits": {}}
        for split_name in requested_splits:
            target = split_targets[split_name]
            mask = np.asarray(target["mask"], dtype=bool)
            selected_valid = np.asarray(target["valid"], dtype=bool)
            geometry_labels = np.asarray(target["geometry_labels"], dtype=bool)
            exact_targets = target["exact_targets"]
            exact_labels = np.asarray(target["exact_labels"], dtype=bool)
            baseline_candidate = baseline_probability[mask]
            baseline_split_null = baseline_null[mask]
            probe_candidate = probe_probability[family_index][mask]
            probe_split_null = probe_null[family_index][mask]
            identity_probe_candidate = (
                identity_probe_probability[family_index][mask]
                if identity_probe_probability is not None
                else probe_candidate
            )
            identity_probe_split_null = (
                identity_probe_null[family_index][mask]
                if identity_probe_null is not None
                else probe_split_null
            )
            all_rows = np.ones((len(baseline_candidate),), dtype=bool)
            exact_summary = summarize_registered_candidate_identity(
                exact_labels, exact_targets
            )
            family_result["splits"][split_name] = {
                "geometry_set": {
                    "baseline": _set_valued_metrics(
                        baseline_candidate,
                        baseline_split_null,
                        labels=geometry_labels,
                        valid=selected_valid,
                        row_mask=all_rows,
                    ),
                    "probe": _set_valued_metrics(
                        probe_candidate,
                        probe_split_null,
                        labels=geometry_labels,
                        valid=selected_valid,
                        row_mask=all_rows,
                    ),
                    "paired_rank": _paired_rank_audit(
                        baseline_candidate,
                        probe_candidate,
                        labels=geometry_labels,
                        valid=selected_valid,
                        row_mask=all_rows,
                    ),
                },
                "baseline_rank2_to_l_geometry_rescue": _rank2_to_l_rescue_audit(
                    baseline_candidate,
                    probe_candidate,
                    labels=geometry_labels,
                    valid=selected_valid,
                    row_mask=all_rows,
                ),
                "exact_registered_identity": {
                    "target_coverage": exact_summary,
                    "baseline": _exact_identity_metrics(
                        baseline_candidate,
                        baseline_split_null,
                        labels=exact_labels,
                        valid=selected_valid,
                        supervised=exact_targets.supervised,
                    ),
                    "probe": _exact_identity_metrics(
                        identity_probe_candidate,
                        identity_probe_split_null,
                        labels=exact_labels,
                        valid=selected_valid,
                        supervised=exact_targets.supervised,
                    ),
                    "paired_rank": _paired_rank_audit(
                        baseline_candidate,
                        identity_probe_candidate,
                        labels=exact_labels,
                        valid=selected_valid,
                        row_mask=np.asarray(exact_targets.supervised, dtype=bool),
                    ),
                },
            }
        output["families"][family_name] = family_result
    family_index = {str(name): index for index, name in enumerate(family_names.tolist())}
    paired_profiles = [
        profile
        for profile in _BIDIRECTIONAL_VISUAL_CONTROL_FAMILIES
        if {profile[1], profile[2]}.issubset(family_index)
    ]
    if len(paired_profiles) > 1:
        raise ValueError("prediction artifact mixes multiple paired visual/control profiles")
    if paired_profiles:
        profile_name, visual_family, position_control_family = paired_profiles[0]
        comparison: dict[str, Any] = {"splits": {}}
        comparison["family_profile"] = profile_name
        comparison["visual_family"] = visual_family
        comparison["position_control_family"] = position_control_family
        visual_index = family_index[visual_family]
        position_index = family_index[position_control_family]
        for split_name in requested_splits:
            target = split_targets[split_name]
            mask = np.asarray(target["mask"], dtype=bool)
            comparison["splits"][split_name] = _visual_vs_position_control_pre_gate(
                position_probability=probe_probability[position_index][mask],
                position_null=probe_null[position_index][mask],
                visual_probability=probe_probability[visual_index][mask],
                visual_null=probe_null[visual_index][mask],
                geometry_labels=np.asarray(target["geometry_labels"], dtype=bool),
                exact_labels=np.asarray(target["exact_labels"], dtype=bool),
                exact_supervised=np.asarray(target["exact_targets"].supervised, dtype=bool),
                candidate_valid=np.asarray(target["valid"], dtype=bool),
                position_identity_probability=(
                    identity_probe_probability[position_index][mask]
                    if identity_probe_probability is not None
                    else None
                ),
                position_identity_null=(
                    identity_probe_null[position_index][mask]
                    if identity_probe_null is not None
                    else None
                ),
                visual_identity_probability=(
                    identity_probe_probability[visual_index][mask]
                    if identity_probe_probability is not None
                    else None
                ),
                visual_identity_null=(
                    identity_probe_null[visual_index][mask]
                    if identity_probe_null is not None
                    else None
                ),
            )
        output["family_comparisons"] = {
            "visual_vs_position_control": comparison,
        }
        if "per_scale_view_logits" in predictions:
            per_scale = np.asarray(predictions["per_scale_view_logits"], dtype=np.float64)
            identity_per_scale = (
                np.asarray(predictions["identity_per_scale_view_logits"], dtype=np.float64)
                if uses_separate_identity_head
                else None
            )
            source_scales = (
                prediction_source_protocol.get("source_scales")
                if isinstance(prediction_source_protocol, Mapping)
                else None
            )
            if not isinstance(source_scales, Sequence) or isinstance(source_scales, (str, bytes)):
                raise ValueError("per-scale prediction lacks source-scale provenance")
            scale_names = tuple(
                str(item.get("name", ""))
                for item in source_scales
                if isinstance(item, Mapping)
            )
            if (
                len(scale_names) != len(source_scales)
                or not scale_names
                or any(not name for name in scale_names)
                or per_scale.shape
                != (
                    len(family_names),
                    len(rows),
                    candidate_tracks.shape[1],
                    view_valid.shape[2],
                    len(scale_names),
                )
            ):
                raise ValueError("per-scale prediction arrays differ from the frozen contract")
            diagnostic: dict[str, Any] = {
                "family_profile": profile_name,
                "visual_family": visual_family,
                "position_control_family": position_control_family,
                "splits": {},
            }
            for split_name in requested_splits:
                target = split_targets[split_name]
                mask = np.asarray(target["mask"], dtype=bool)
                diagnostic["splits"][split_name] = _per_scale_visual_control_diagnostic(
                    visual_per_scale_view_logits=per_scale[visual_index][mask],
                    position_per_scale_view_logits=per_scale[position_index][mask],
                    scale_names=scale_names,
                    base_candidate=baseline_probability[mask],
                    base_null=baseline_null[mask],
                    view_valid=view_valid[mask],
                    geometry_labels=np.asarray(target["geometry_labels"], dtype=bool),
                    exact_labels=np.asarray(target["exact_labels"], dtype=bool),
                    exact_supervised=np.asarray(target["exact_targets"].supervised, dtype=bool),
                    candidate_valid=np.asarray(target["valid"], dtype=bool),
                    visual_identity_per_scale_view_logits=(
                        identity_per_scale[visual_index][mask]
                        if identity_per_scale is not None
                        else None
                    ),
                    position_identity_per_scale_view_logits=(
                        identity_per_scale[position_index][mask]
                        if identity_per_scale is not None
                        else None
                    ),
                )
            output["per_scale_visual_vs_position_control_diagnostic"] = diagnostic
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    result = audit_multiscale_candidate_probe(
        features_path=Path(args.features),
        predictions_path=Path(args.predictions),
        proposals_path=Path(args.proposals),
        base_overlay_path=Path(args.base_prior_overlay),
        colmap_model_dir=Path(args.colmap_model_dir),
        geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
        exact_identity_radius_px=float(args.exact_identity_radius_px),
        allow_diagnostic_feature_artifact=bool(args.allow_diagnostic_feature_artifact),
        audit_splits=_parse_audit_splits(args.audit_splits),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
