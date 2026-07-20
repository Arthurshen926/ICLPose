"""Audit train-frozen appearance residual predictions on validation identities.

No model fitting occurs here.  The audit first verifies that every prediction
uses the same fixed top-L-plus-null posterior as its target-free input, then
joins only validation registered SfM identities.  Its gate is deliberately a
candidate-level gate: even a pass does not promote a model into pose scoring
until the same frozen-hypothesis pose audit is run separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    FrozenAppearanceProbeFeatures,
    load_frozen_appearance_probe_features,
)
from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
)
from feature_extract.tools.vfm.fit_frozen_multiscale_candidate_appearance_residual import (
    _validate_frozen_loftr_anchor_manifest_audit,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance_artifacts", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--frozen-loftr-anchor-manifest-audit")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("appearance artifacts must be non-empty and unique")
    return paths


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("rank average-precision inputs are invalid")
    count = int(np.sum(target))
    if count == 0:
        return None
    order = np.argsort(-values, kind="stable")
    ordered = target[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered]) / count)


def _top_and_rank(
    scores: np.ndarray, labels: np.ndarray, candidate_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != valid.shape:
        raise ValueError("rank arrays are incompatible")
    if np.any(~np.isfinite(values[valid])):
        raise ValueError("valid candidate score must be finite")
    if np.any(positive & ~valid):
        raise ValueError("positive identity candidate is absent from the fixed posterior")
    ranked = np.where(valid, values, -np.inf)
    order = np.argsort(-ranked, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive, order, axis=1)
    has_positive = np.any(ranked_positive, axis=1)
    rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    rank[~has_positive] = -1
    return order[:, 0].astype(np.int64), rank


def _rank_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1) & np.any(valid, axis=1)
    if not np.any(selected):
        return {"row_count": 0}
    top, rank = _top_and_rank(values, positive, valid)
    positive_rows = selected & np.any(positive, axis=1)
    flat = valid & selected[:, None]
    return {
        "row_count": int(np.sum(selected)),
        "candidate_edge_count": int(np.sum(flat)),
        "candidate_edge_positive_rate": float(np.mean(positive[flat])),
        "candidate_pair_average_precision": _average_precision(positive[flat], values[flat]),
        "positive_row_count": int(np.sum(positive_rows)),
        "top1_positive_rate_given_positive": (
            None
            if not np.any(positive_rows)
            else float(np.mean(positive[positive_rows, top[positive_rows]]))
        ),
        "median_first_positive_rank": (
            None
            if not np.any(positive_rows)
            else float(np.median(rank[positive_rows]))
        ),
        "p90_first_positive_rank": (
            None
            if not np.any(positive_rows)
            else float(np.quantile(rank[positive_rows], 0.9))
        ),
    }


def _paired_rank(
    *,
    baseline_scores: np.ndarray,
    probe_scores: np.ndarray,
    labels: np.ndarray,
    candidate_valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    probe = np.asarray(probe_scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1)
    if (
        baseline.shape != probe.shape
        or probe.shape != positive.shape
        or positive.shape != valid.shape
        or selected.shape != (len(baseline),)
    ):
        raise ValueError("paired rank arrays are incompatible")
    rows = selected & np.any(positive, axis=1) & np.any(valid, axis=1)
    if not np.any(rows):
        return {
            "positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
        }
    _top_base, base_rank = _top_and_rank(baseline, positive, valid)
    _top_probe, probe_rank = _top_and_rank(probe, positive, valid)
    base = base_rank[rows]
    updated = probe_rank[rows]
    return {
        "positive_row_count": int(len(base)),
        "rank_win_count": int(np.sum(updated < base)),
        "rank_loss_count": int(np.sum(updated > base)),
        "rank_tie_count": int(np.sum(updated == base)),
        "top1_rescue_count": int(np.sum((base > 1) & (updated == 1))),
        "top1_harm_count": int(np.sum((base == 1) & (updated > 1))),
        "median_rank_delta_baseline_minus_probe": float(np.median(base - updated)),
    }


def _conditional_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Require safe global behavior and a real rank2-to-L rescue signal."""

    overall = metrics.get("exact_registered_identity")
    rank2 = metrics.get("exact_registered_rank2_to_l")
    if not isinstance(overall, Mapping) or not isinstance(rank2, Mapping):
        raise ValueError("appearance residual gate lacks identity metrics")
    base = overall.get("baseline")
    probe = overall.get("probe")
    paired = overall.get("paired_rank")
    rank2_base = rank2.get("baseline")
    rank2_probe = rank2.get("probe")
    rank2_paired = rank2.get("paired_rank")
    if not all(isinstance(item, Mapping) for item in (base, probe, paired, rank2_base, rank2_probe, rank2_paired)):
        raise ValueError("appearance residual gate metrics are incomplete")
    required = (
        base.get("p90_first_positive_rank"),
        probe.get("p90_first_positive_rank"),
        base.get("top1_positive_rate_given_positive"),
        probe.get("top1_positive_rate_given_positive"),
        rank2_base.get("median_first_positive_rank"),
        rank2_probe.get("median_first_positive_rank"),
        rank2_base.get("p90_first_positive_rank"),
        rank2_probe.get("p90_first_positive_rank"),
    )
    comparable = all(value is not None for value in required)
    checks = {
        "comparable_validation_identity_rows": bool(comparable),
        "overall_p90_rank_not_worse": bool(
            comparable
            and float(probe["p90_first_positive_rank"])
            <= float(base["p90_first_positive_rank"])
        ),
        "overall_top1_not_worse": bool(
            comparable
            and float(probe["top1_positive_rate_given_positive"])
            >= float(base["top1_positive_rate_given_positive"])
        ),
        "overall_top1_rescues_exceed_harms": int(paired.get("top1_rescue_count", 0))
        > int(paired.get("top1_harm_count", 0)),
        "rank2_to_l_minimum_positive_rows": int(rank2_paired.get("positive_row_count", 0))
        >= 50,
        "rank2_to_l_median_strictly_improved": bool(
            comparable
            and float(rank2_probe["median_first_positive_rank"])
            < float(rank2_base["median_first_positive_rank"])
        ),
        "rank2_to_l_p90_not_worse": bool(
            comparable
            and float(rank2_probe["p90_first_positive_rank"])
            <= float(rank2_base["p90_first_positive_rank"])
        ),
        "rank2_to_l_wins_exceed_losses": int(rank2_paired.get("rank_win_count", 0))
        > int(rank2_paired.get("rank_loss_count", 0)),
        "rank2_to_l_has_top1_rescues": int(rank2_paired.get("top1_rescue_count", 0))
        > 0,
    }
    return {
        "policy": (
            "validation-only fixed-prior residual candidate gate; passing is necessary "
            "but not sufficient for frozen-hypothesis pose scoring or promotion"
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _load_predictions(
    *,
    features: FrozenAppearanceProbeFeatures,
    path: Path,
    frozen_loftr_anchor_manifest: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "query_ids",
            "split_names",
            "source_row_indices",
            "candidate_track_ids",
            "family_names",
            "candidate_probabilities",
            "null_probabilities",
            "per_view_residuals",
            "baseline_candidate_probabilities",
            "baseline_null_probabilities",
            "metadata_json",
        )
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: residual prediction lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FROZEN_APPEARANCE_RESIDUAL_PREDICTION_FORMAT
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("fixed_global_top_l") is not True
        or metadata.get("candidate_reselection") is not False
        or metadata.get("null_handling") != "immutable_input_null_log_prior_v1"
    ):
        raise ValueError(f"{path}: residual prediction violates the train-only contract")
    is_loftr = (
        features.metadata.get("format") == FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT
    )
    stored_loftr_manifest = metadata.get("frozen_loftr_anchor_manifest_audit")
    if is_loftr and stored_loftr_manifest != frozen_loftr_anchor_manifest:
        raise ValueError("LoFTR residual prediction does not bind the required manifest audit")
    if not is_loftr and stored_loftr_manifest is not None:
        raise ValueError("direct residual prediction unexpectedly binds a LoFTR manifest audit")
    expected_artifacts = [
        {"path": str(item), "sha256": file_sha256_short(item)} for item in features.paths
    ]
    if metadata.get("appearance_artifacts") != expected_artifacts:
        raise ValueError("residual prediction was fit from a different appearance artifact set")
    if (
        not np.array_equal(np.asarray(arrays["query_ids"]).astype(str), features.query_ids)
        or not np.array_equal(np.asarray(arrays["split_names"]).astype(str), features.split_names)
        or not np.array_equal(
            np.asarray(arrays["source_row_indices"], dtype=np.int64),
            features.source_row_indices,
        )
        or not np.array_equal(
            np.asarray(arrays["candidate_track_ids"], dtype=np.int64),
            features.candidate_track_ids,
        )
    ):
        raise ValueError("residual prediction identities differ from frozen appearance rows")
    baseline_candidate = np.asarray(arrays["baseline_candidate_probabilities"], dtype=np.float32)
    baseline_null = np.asarray(arrays["baseline_null_probabilities"], dtype=np.float32)
    if (
        baseline_candidate.shape != features.candidate_probabilities.shape
        or baseline_null.shape != features.null_probabilities.shape
        or np.max(np.abs(baseline_candidate - features.candidate_probabilities)) > 2e-6
        or np.max(np.abs(baseline_null - features.null_probabilities)) > 2e-6
    ):
        raise ValueError("residual prediction baseline is not the immutable input posterior")
    names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    residual = np.asarray(arrays["per_view_residuals"], dtype=np.float32)
    if (
        len(names) == 0
        or len(set(names)) != len(names)
        or candidate.shape != (len(names), *features.candidate_probabilities.shape)
        or null.shape != (len(names), len(features.query_ids))
        or residual.shape != (
            len(names),
            *features.candidate_view_weights.shape,
        )
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(candidate < 0.0)
        or np.any(null <= 0.0)
        or np.max(np.abs(candidate.sum(axis=2) + null - 1.0)) > 2e-5
    ):
        raise ValueError("residual prediction arrays are invalid")
    return candidate, null, residual, names, metadata


def audit_frozen_multiscale_candidate_appearance_residual(
    *,
    appearance_artifacts: Sequence[Path],
    predictions: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    registered_identity_radius_px: float,
    frozen_loftr_anchor_manifest_audit: Path | None = None,
) -> dict[str, Any]:
    """Join validation identities to a prediction that was frozen by train fit."""

    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    feature_paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_appearance_probe_features(feature_paths)
    is_loftr = (
        features.metadata.get("format") == FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT
    )
    if is_loftr and frozen_loftr_anchor_manifest_audit is None:
        raise ValueError("LoFTR residual audit requires the complete frozen anchor manifest audit")
    if not is_loftr and frozen_loftr_anchor_manifest_audit is not None:
        raise ValueError("direct residual audit must not consume a LoFTR manifest audit")
    loftr_manifest = (
        _validate_frozen_loftr_anchor_manifest_audit(
            features=features, path=Path(frozen_loftr_anchor_manifest_audit)
        )
        if is_loftr
        else None
    )
    candidate, _null, residual, families, prediction_metadata = _load_predictions(
        features=features,
        path=Path(predictions),
        frozen_loftr_anchor_manifest=loftr_manifest,
    )
    validation_mask = features.split_names == "validation"
    if not np.any(validation_mask):
        raise ValueError("appearance residual audit has no validation rows")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    validation_rows = np.flatnonzero(validation_mask)
    targets = registered_query_observation_targets(
        query_ids=features.query_ids[validation_rows],
        query_xy=features.xy[validation_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        features.candidate_track_ids[validation_rows], targets
    )
    identity_coverage = summarize_registered_candidate_identity(labels, targets)
    candidate_valid = (
        (features.candidate_track_ids[validation_rows] >= 0)
        & (features.candidate_probabilities[validation_rows] > 0.0)
    )
    if np.any(labels & ~candidate_valid):
        raise ValueError("validation identity target is absent from the fixed posterior")
    baseline_scores = np.log(
        np.maximum(features.candidate_probabilities[validation_rows], 1e-30)
    )
    result: dict[str, Any] = {
        "stage": "audit_frozen_multiscale_candidate_appearance_fixedprior_residual",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "appearance_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in feature_paths
        ],
        "predictions": {
            "path": str(Path(predictions)),
            "sha256": file_sha256_short(Path(predictions)),
        },
        "row_count": int(len(validation_rows)),
        "query_count": int(len(set(features.query_ids[validation_rows].tolist()))),
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "registered_identity_target_coverage": identity_coverage,
        "families": {},
        "predeclared_gate_summary": {},
        "protocol": {
            "targets_loaded_only_in_validation_audit": True,
            "model_fit_or_selection": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "pose_scoring": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "prediction_contract": {
            "training_supervision_split": prediction_metadata.get(
                "training_supervision_split"
            ),
            "null_handling": prediction_metadata.get("null_handling"),
            "support_view_marginalization": prediction_metadata.get(
                "support_view_marginalization"
            ),
            "frozen_loftr_anchor_manifest_audit": loftr_manifest,
        },
    }
    row_mask = targets.supervised & np.any(labels, axis=1)
    for family_index, family in enumerate(families):
        probe_scores = np.log(np.maximum(candidate[family_index, validation_rows], 1e-30))
        overall = {
            "baseline": _rank_metrics(
                scores=baseline_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=row_mask,
            ),
            "probe": _rank_metrics(
                scores=probe_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=row_mask,
            ),
            "paired_rank": _paired_rank(
                baseline_scores=baseline_scores,
                probe_scores=probe_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=row_mask,
            ),
        }
        _top, baseline_rank = _top_and_rank(baseline_scores, labels, candidate_valid)
        rank2_rows = row_mask & (baseline_rank > 1)
        rank2 = {
            "baseline": _rank_metrics(
                scores=baseline_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=rank2_rows,
            ),
            "probe": _rank_metrics(
                scores=probe_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=rank2_rows,
            ),
            "paired_rank": _paired_rank(
                baseline_scores=baseline_scores,
                probe_scores=probe_scores,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=rank2_rows,
            ),
        }
        payload = {
            "family": family,
            "exact_registered_identity": overall,
            "exact_registered_rank2_to_l": rank2,
            "residual_statistics": {
                "mean": float(np.mean(residual[family_index, validation_rows])),
                "std": float(np.std(residual[family_index, validation_rows])),
                "maximum_abs": float(
                    np.max(np.abs(residual[family_index, validation_rows]))
                ),
            },
        }
        payload["predeclared_incremental_gate"] = _conditional_gate(payload)
        result["families"][family] = payload
        result["predeclared_gate_summary"][family] = payload[
            "predeclared_incremental_gate"
        ]
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_frozen_multiscale_candidate_appearance_residual(
        appearance_artifacts=_paths(args.appearance_artifacts),
        predictions=Path(args.predictions),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        frozen_loftr_anchor_manifest_audit=(
            None
            if args.frozen_loftr_anchor_manifest_audit is None
            else Path(args.frozen_loftr_anchor_manifest_audit)
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
