"""Audit frozen mixed-point candidate predictions on validation only.

This command never fits a model.  It validates that candidate probabilities
were produced without validation targets, then joins validation SfM residuals
to measure whether fixed candidate-specific multi-scale evidence improves over
the immutable full-bank top-L coarse posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    paired_rank_audit,
    rank2_to_l_rescue_audit,
    set_valued_metrics,
)
from feature_extract.vfm.localization.mixed_multiscale_candidate_probe import (
    MIXED_GEOMETRIC_SET_SUPERVISION_MODE,
    MIXED_REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
    candidate_probe_gate,
    exact_identity_metric_names,
    geometric_membership_from_residuals,
    load_mixed_multiscale_candidate_predictions,
    load_mixed_multiscale_candidate_probe_features,
    materialize_mixed_candidate_reprojection_residuals,
    materialize_mixed_registered_identity_membership,
)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not np.isfinite(item) or item <= 0.0 for item in thresholds)
    ):
        raise ValueError("geometric thresholds must be a non-empty unique positive list")
    return thresholds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geometric_thresholds_px", default="2,5")
    parser.add_argument("--primary_threshold_px", type=float, default=5.0)
    return parser.parse_args(argv)


def _metrics_for_mask(
    *,
    baseline_candidate: np.ndarray,
    baseline_null: np.ndarray,
    probe_candidate: np.ndarray,
    probe_null: np.ndarray,
    labels: np.ndarray,
    candidate_valid: np.ndarray,
    row_mask: np.ndarray,
    exact_identity: bool = False,
) -> dict[str, Any]:
    baseline_raw = set_valued_metrics(
        baseline_candidate,
        baseline_null,
        labels=labels,
        valid=candidate_valid,
        row_mask=row_mask,
    )
    probe_raw = set_valued_metrics(
        probe_candidate,
        probe_null,
        labels=labels,
        valid=candidate_valid,
        row_mask=row_mask,
    )
    paired = paired_rank_audit(
        baseline_candidate,
        probe_candidate,
        labels=labels,
        valid=candidate_valid,
        row_mask=row_mask,
    )
    rank2_to_l = rank2_to_l_rescue_audit(
        baseline_candidate,
        probe_candidate,
        labels=labels,
        valid=candidate_valid,
        row_mask=row_mask,
    )
    if bool(exact_identity):
        baseline = exact_identity_metric_names(baseline_raw)
        probe = exact_identity_metric_names(probe_raw)
        if rank2_to_l.get("baseline") is not None:
            rank2_to_l = {
                **rank2_to_l,
                "baseline": exact_identity_metric_names(rank2_to_l["baseline"]),
                "probe": exact_identity_metric_names(rank2_to_l["probe"]),
            }
        gate = candidate_probe_gate(
            baseline,
            probe,
            paired,
            top1_metric_key="top1_exact_identity_rate",
            target_name="exact_identity",
        )
    else:
        baseline = baseline_raw
        probe = probe_raw
        gate = candidate_probe_gate(baseline, probe, paired)
    return {
        "baseline": baseline,
        "probe": probe,
        "paired_rank": paired,
        "rank2_to_l_rescue": rank2_to_l,
        "candidate_gate": gate,
    }


def audit_mixed_multiscale_candidate_probe(
    *,
    features_path: Path,
    verification_points_path: Path,
    predictions_path: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    thresholds_px: Sequence[float],
    primary_threshold_px: float,
) -> dict[str, Any]:
    thresholds = tuple(float(value) for value in thresholds_px)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    features = load_mixed_multiscale_candidate_probe_features(
        features_path=Path(features_path), verification_points_path=Path(verification_points_path)
    )
    candidates, nulls, families, prediction_metadata = load_mixed_multiscale_candidate_predictions(
        path=Path(predictions_path), features=features
    )
    supervision_mode = str(
        prediction_metadata.get("supervision_mode", MIXED_GEOMETRIC_SET_SUPERVISION_MODE)
    )
    if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE and float(
        primary_threshold_px
    ) not in thresholds:
        raise ValueError("primary threshold must be one of geometric thresholds")
    if supervision_mode not in {
        MIXED_GEOMETRIC_SET_SUPERVISION_MODE,
        MIXED_REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
    }:
        raise ValueError("mixed multi-scale prediction has an unsupported supervision mode")
    validation_rows = np.flatnonzero(features.split_names == "validation")
    if not len(validation_rows):
        raise ValueError("mixed multi-scale features contain no validation rows")
    valid = features.candidate_tracks >= 0
    if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE:
        validation_residuals, target_audit = materialize_mixed_candidate_reprojection_residuals(
            features=features,
            projected_landmark_bank=Path(projected_landmark_bank),
            colmap_model_dir=Path(colmap_model_dir),
            row_indices=validation_rows,
            required_split="validation",
        )
        labels_by_threshold: dict[str, np.ndarray] = {}
        for threshold in thresholds:
            labels = np.zeros_like(valid, dtype=bool)
            membership = geometric_membership_from_residuals(
                residuals=validation_residuals,
                candidate_valid=valid[validation_rows],
                threshold_px=float(threshold),
            )
            labels[validation_rows] = membership[:, :-1]
            labels_by_threshold[f"{float(threshold):g}"] = labels
        validation_mask = features.split_names == "validation"
        identity_labels = None
    else:
        radius = float(prediction_metadata.get("registered_identity_radius_px", 0.0))
        if radius <= 0.0:
            raise ValueError("registered identity prediction lacks its target radius")
        identity_rows, _membership, sparse_labels, target_audit = (
            materialize_mixed_registered_identity_membership(
                features=features,
                colmap_model_dir=Path(colmap_model_dir),
                split_name="validation",
                identity_radius_px=radius,
            )
        )
        validation_mask = np.zeros((len(features.source_point_ids),), dtype=bool)
        validation_mask[identity_rows] = True
        identity_labels = np.zeros_like(valid, dtype=bool)
        identity_labels[identity_rows] = sparse_labels
        labels_by_threshold = {}
    report: dict[str, Any] = {
        "stage": "audit_frozen_mixed_multiscale_candidate_probe_validation",
        "inputs": {
            "features": str(features.path),
            "features_sha256": file_sha256_short(features.path),
            "verification_points": str(Path(verification_points_path)),
            "verification_points_sha256": file_sha256_short(Path(verification_points_path)),
            "predictions": str(Path(predictions_path)),
            "predictions_sha256": file_sha256_short(Path(predictions_path)),
            **target_audit,
        },
        "prediction_metadata": {
            "prior_mode": prediction_metadata.get("prior_mode"),
            "base_coarse_posterior_role": prediction_metadata.get("base_coarse_posterior_role"),
            "training_target": prediction_metadata.get("training_target"),
            "supervision_mode": supervision_mode,
        },
        "validation_point_count": int(len(validation_rows)),
        "validation_query_count": int(len(set(features.query_ids[validation_rows].tolist()))),
        "validation_scored_point_count": int(np.sum(validation_mask)),
        "thresholds_px": (
            list(thresholds)
            if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
            else []
        ),
        "primary_threshold_px": (
            float(primary_threshold_px)
            if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
            else None
        ),
        "families": {},
        "protocol": {
            "feature_export_target_free": True,
            "fit_uses_train_targets_only": True,
            "prediction_frozen_before_validation_target_join": True,
            "validation_target_join_only": True,
            "test_target_labels_materialized": False,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    source_names = ("all", *sorted(set(features.point_sources[validation_mask].tolist())))
    for family_index, family in enumerate(families):
        family_report: dict[str, Any] = {"by_source": {}}
        for source in source_names:
            source_mask = (
                validation_mask
                if source == "all"
                else validation_mask & (features.point_sources == str(source))
            )
            if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE:
                threshold_report = {
                    key: _metrics_for_mask(
                        baseline_candidate=features.base_candidate_probabilities,
                        baseline_null=features.base_null_probabilities,
                        probe_candidate=candidates[family_index],
                        probe_null=nulls[family_index],
                        labels=labels,
                        candidate_valid=valid,
                        row_mask=source_mask,
                    )
                    for key, labels in labels_by_threshold.items()
                }
                payload = {
                    "point_count": int(np.sum(source_mask)),
                    "thresholds": threshold_report,
                }
            else:
                if identity_labels is None:
                    raise RuntimeError("registered identity labels were not materialized")
                payload = {
                    "point_count": int(np.sum(source_mask)),
                    "registered_identity": _metrics_for_mask(
                        baseline_candidate=features.base_candidate_probabilities,
                        baseline_null=features.base_null_probabilities,
                        probe_candidate=candidates[family_index],
                        probe_null=nulls[family_index],
                        labels=identity_labels,
                        candidate_valid=valid,
                        row_mask=source_mask,
                        exact_identity=True,
                    ),
                }
            if source == "all":
                family_report["all"] = payload
            else:
                family_report["by_source"][str(source)] = payload
        primary = (
            family_report["all"]["thresholds"][f"{float(primary_threshold_px):g}"]
            if supervision_mode == MIXED_GEOMETRIC_SET_SUPERVISION_MODE
            else family_report["all"]["registered_identity"]
        )
        family_report["primary_candidate_gate"] = primary["candidate_gate"]
        report["families"][family] = family_report
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            audit_mixed_multiscale_candidate_probe(
                features_path=Path(args.features),
                verification_points_path=Path(args.verification_points),
                predictions_path=Path(args.predictions),
                projected_landmark_bank=Path(args.projected_landmark_bank),
                colmap_model_dir=Path(args.colmap_model_dir),
                output_dir=Path(args.output_dir),
                thresholds_px=_parse_thresholds(args.geometric_thresholds_px),
                primary_threshold_px=float(args.primary_threshold_px),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
