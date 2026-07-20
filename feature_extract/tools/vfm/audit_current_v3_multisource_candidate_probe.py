"""Externally audit frozen current-V3 appearance predictions on validation.

This command never fits a model.  It first verifies that predictions are
target-free and lineage-compatible, then joins either V3 residual membership
or strict registered-track labels only for frozen validation rows.  Test rows
are rejected rather than reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    CURRENT_V3_PREDICTION_FORMAT,
    EXACT_IDENTITY_PROBABILITY_SEMANTICS,
    GEOMETRIC_SET_SUPERVISION_MODE,
    GEOMETRIC_PROBABILITY_SEMANTICS,
    REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
    SUPPORTED_SUPERVISION_MODES,
    align_current_v3_features_and_evidence,
    load_current_v3_evidence_inference,
    load_current_v3_features,
    metadata_from_npz,
    paired_rank_audit,
    rank2_to_l_rescue_audit,
    set_valued_metrics,
    validate_candidate_probability_contract,
    validation_geometric_labels,
    validation_registered_track_identity_labels,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--geometric_thresholds_px",
        default="2,5",
        help="comma-separated validation residual thresholds",
    )
    parser.add_argument("--primary_threshold_px", type=float, default=5.0)
    parser.add_argument(
        "--colmap_model_dir",
        default="",
        help="required only when the frozen prediction uses registered_track_identity",
    )
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    return parser.parse_args(argv)


def _thresholds(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not values or any(item <= 0.0 for item in values) or len(set(values)) != len(values):
        raise ValueError("validation thresholds must be unique positive values")
    return values


def _load_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_canonical_rows",
        "candidate_view_valid",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"current-V3 predictions lack {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
        metadata = metadata_from_npz(payload, context="current-V3 predictions")
    supervision_mode = str(
        metadata.get("supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE)
    )
    if supervision_mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError("current-V3 prediction has an unsupported supervision mode")
    expected_probability_semantics = (
        GEOMETRIC_PROBABILITY_SEMANTICS
        if supervision_mode == GEOMETRIC_SET_SUPERVISION_MODE
        else EXACT_IDENTITY_PROBABILITY_SEMANTICS
    )
    if (
        metadata.get("format") != CURRENT_V3_PREDICTION_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("probability_semantics") != expected_probability_semantics
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("render", True))
    ):
        raise ValueError("current-V3 predictions violate the frozen validation protocol")
    return arrays, metadata


def _validate_predictions(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    features_path: Path,
    candidate_evidence_path: Path,
    source_rows: np.ndarray,
    query_ids: np.ndarray,
    split_names: np.ndarray,
    candidate_tracks: np.ndarray,
    candidate_canonical_rows: np.ndarray,
    candidate_view_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if str(metadata.get("features_sha256", "")) != str(file_sha256_short(features_path)) or str(
        metadata.get("candidate_evidence_sha256", "")
    ) != str(file_sha256_short(candidate_evidence_path)):
        raise ValueError("current-V3 prediction lineage differs from audit inputs")
    for key, expected in (
        ("source_row_indices", source_rows),
        ("query_ids", query_ids),
        ("split_names", split_names),
        ("candidate_track_ids", candidate_tracks),
        ("candidate_canonical_rows", candidate_canonical_rows),
        ("candidate_view_valid", candidate_view_valid),
    ):
        if not np.array_equal(np.asarray(arrays[key]), expected):
            raise ValueError(f"current-V3 prediction {key} differs from frozen features")
    family_names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    if (
        not family_names
        or len(set(family_names)) != len(family_names)
        or candidate.ndim != 3
        or candidate.shape != (len(family_names), *candidate_tracks.shape)
        or null.shape != (len(family_names), len(source_rows))
    ):
        raise ValueError("current-V3 prediction probability arrays are invalid")
    for index in range(len(family_names)):
        validate_candidate_probability_contract(
            candidate[index], null[index], candidate_tracks >= 0
        )
    return candidate, null, family_names


def _candidate_gate(
    baseline: Mapping[str, Any],
    probe: Mapping[str, Any],
    paired: Mapping[str, Any],
    *,
    top1_metric_key: str = "top1_geometry_valid_rate",
    target_name: str = "geometry",
) -> dict[str, Any]:
    """A deliberately conservative gate before any pose-ranking experiment."""

    baseline_nll = float(baseline["group_target_nll"])
    probe_nll = float(probe["group_target_nll"])
    baseline_top1 = baseline.get(top1_metric_key)
    probe_top1 = probe.get(top1_metric_key)
    nll_improved = probe_nll < baseline_nll
    top1_improved = (
        baseline_top1 is not None
        and probe_top1 is not None
        and float(probe_top1) > float(baseline_top1)
    )
    rank_improved = int(paired["rank_win_count"]) > int(paired["rank_loss_count"])
    rescue_improved = int(paired["top1_rescue_count"]) > int(paired["top1_harm_count"])
    result = {
        "nll_improved": nll_improved,
        f"top1_{target_name}_rate_improved": top1_improved,
        "paired_rank_wins_exceed_losses": rank_improved,
        "top1_rescues_exceed_harms": rescue_improved,
        "passed": bool(nll_improved and top1_improved and rank_improved and rescue_improved),
        "policy": (
            "validation-only all-four gate; passing this single split is necessary "
            "but not sufficient for pose-ranking or hard-pose calibration"
        ),
    }
    if target_name == "geometry":
        # Preserve the original audit field for existing geometric artifacts.
        result["top1_geometry_valid_rate_improved"] = top1_improved
    return result


def _exact_identity_metric_names(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Expose strict-track metrics without relabelling them as geometry."""

    result = dict(metrics)
    result["top1_exact_identity_rate"] = result.pop("top1_geometry_valid_rate")
    result["mean_exact_track_probability_mass_when_present"] = result.pop(
        "mean_correct_probability_mass_when_present"
    )
    result["median_exact_track_rank"] = result.pop("median_first_positive_rank")
    result["p90_exact_track_rank"] = result.pop("p90_first_positive_rank")
    result["exact_track_top_l_recall"] = result.pop("positive_top_l_recall")
    return result


def audit_current_v3_multisource_candidate_probe(
    *,
    features_path: Path,
    candidate_evidence_path: Path,
    predictions_path: Path,
    thresholds_px: Sequence[float],
    primary_threshold_px: float,
    colmap_model_dir: Path | None = None,
    registered_identity_radius_px: float = 2.0,
) -> dict[str, Any]:
    thresholds = tuple(float(value) for value in thresholds_px)
    if not thresholds or any(value <= 0.0 for value in thresholds):
        raise ValueError("current-V3 validation thresholds must be positive")
    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("current-V3 registered identity radius must be positive")
    if float(primary_threshold_px) not in thresholds:
        raise ValueError("primary threshold must be one of geometric validation thresholds")
    features = load_current_v3_features(Path(features_path))
    evidence = load_current_v3_evidence_inference(Path(candidate_evidence_path))
    aligned = align_current_v3_features_and_evidence(features, evidence)
    prediction_arrays, prediction_metadata = _load_predictions(Path(predictions_path))
    candidate, null, families = _validate_predictions(
        arrays=prediction_arrays,
        metadata=prediction_metadata,
        features_path=Path(features_path),
        candidate_evidence_path=Path(candidate_evidence_path),
        source_rows=features.source_rows,
        query_ids=features.query_ids,
        split_names=features.split_names,
        candidate_tracks=features.candidate_tracks,
        candidate_canonical_rows=features.candidate_canonical_rows,
        candidate_view_valid=features.candidate_view_valid,
    )
    validation_mask = features.split_names == "validation"
    if not np.any(validation_mask) or np.any(features.split_names == "test"):
        raise ValueError("current-V3 audit accepts validation rows and rejects test rows")
    supervision_mode = str(
        prediction_metadata.get("supervision_mode", GEOMETRIC_SET_SUPERVISION_MODE)
    )
    result: dict[str, Any] = {
        "stage": "current_v3_multisource_candidate_probe_validation_audit",
        "protocol": {
            "fixed_global_top_l": True,
            "feature_export_target_free": True,
            "fit_uses_train_targets_only": True,
            "training_supervision_mode": supervision_mode,
            "prediction_frozen_before_validation_target_join": True,
            "validation_only": True,
            "test_used_for_model_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "inputs": {
            "features": str(Path(features_path)),
            "features_sha256": file_sha256_short(Path(features_path)),
            "candidate_evidence": str(Path(candidate_evidence_path)),
            "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence_path)),
            "predictions": str(Path(predictions_path)),
            "predictions_sha256": file_sha256_short(Path(predictions_path)),
        },
        "validation_row_count": int(np.sum(validation_mask)),
        "thresholds_px": list(thresholds),
        "primary_threshold_px": float(primary_threshold_px),
        "families": {},
    }
    valid = features.candidate_tracks >= 0
    if supervision_mode == REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE:
        if colmap_model_dir is None or not str(colmap_model_dir).strip():
            raise ValueError("registered-track validation audit requires --colmap_model_dir")
        identity_mask, identity_labels, identity_audit = (
            validation_registered_track_identity_labels(
                features,
                colmap_model_dir=Path(colmap_model_dir),
                identity_radius_px=float(registered_identity_radius_px),
            )
        )
        if np.any(identity_mask & ~validation_mask):
            raise RuntimeError("registered identity audit joined a non-validation target")
        baseline_metrics = _exact_identity_metric_names(
            set_valued_metrics(
                aligned.base_candidate_probabilities,
                aligned.base_null_probabilities,
                labels=identity_labels,
                valid=valid,
                row_mask=identity_mask,
            )
        )
        result["target_protocol"] = {
            "target": "registered_query_observation_exact_track_or_explicit_null",
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "validation_target_rows_only": True,
            "validation_target_row_count": int(np.sum(identity_mask)),
            "geometric_residual_labels_read": False,
            "audit": identity_audit,
        }
        for family_index, family in enumerate(families):
            probe_metrics = _exact_identity_metric_names(
                set_valued_metrics(
                    candidate[family_index],
                    null[family_index],
                    labels=identity_labels,
                    valid=valid,
                    row_mask=identity_mask,
                )
            )
            paired = paired_rank_audit(
                aligned.base_candidate_probabilities,
                candidate[family_index],
                labels=identity_labels,
                valid=valid,
                row_mask=identity_mask,
            )
            rescue = rank2_to_l_rescue_audit(
                aligned.base_candidate_probabilities,
                candidate[family_index],
                labels=identity_labels,
                valid=valid,
                row_mask=identity_mask,
            )
            result["families"][family] = {
                "exact_registered_identity": {
                    "baseline": baseline_metrics,
                    "probe": probe_metrics,
                    "paired_rank": paired,
                    "baseline_rank2_to_l_rescue": rescue,
                    "candidate_gate": _candidate_gate(
                        baseline_metrics,
                        probe_metrics,
                        paired,
                        top1_metric_key="top1_exact_identity_rate",
                        target_name="exact_identity",
                    ),
                }
            }
        return result
    for threshold in thresholds:
        mask, labels = validation_geometric_labels(
            Path(candidate_evidence_path), aligned, threshold_px=float(threshold)
        )
        if not np.array_equal(mask, validation_mask):
            raise RuntimeError("current-V3 validation target join changed the frozen split")
        baseline_metrics = set_valued_metrics(
            aligned.base_candidate_probabilities,
            aligned.base_null_probabilities,
            labels=labels,
            valid=valid,
            row_mask=validation_mask,
        )
        for family_index, family in enumerate(families):
            family_result = result["families"].setdefault(family, {"thresholds": {}})
            probe_metrics = set_valued_metrics(
                candidate[family_index],
                null[family_index],
                labels=labels,
                valid=valid,
                row_mask=validation_mask,
            )
            paired = paired_rank_audit(
                aligned.base_candidate_probabilities,
                candidate[family_index],
                labels=labels,
                valid=valid,
                row_mask=validation_mask,
            )
            rescue = rank2_to_l_rescue_audit(
                aligned.base_candidate_probabilities,
                candidate[family_index],
                labels=labels,
                valid=valid,
                row_mask=validation_mask,
            )
            block: dict[str, Any] = {
                "baseline": baseline_metrics,
                "probe": probe_metrics,
                "paired_rank": paired,
                "baseline_rank2_to_l_rescue": rescue,
            }
            if float(threshold) == float(primary_threshold_px):
                block["candidate_gate"] = _candidate_gate(
                    baseline_metrics, probe_metrics, paired
                )
            family_result["thresholds"][str(int(threshold) if threshold.is_integer() else threshold)] = block
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    result = audit_current_v3_multisource_candidate_probe(
        features_path=Path(args.features),
        candidate_evidence_path=Path(args.candidate_evidence),
        predictions_path=Path(args.predictions),
        thresholds_px=_thresholds(str(args.geometric_thresholds_px)),
        primary_threshold_px=float(args.primary_threshold_px),
        colmap_model_dir=(
            None if not str(args.colmap_model_dir).strip() else Path(args.colmap_model_dir)
        ),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
