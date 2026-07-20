"""Audit frozen mixed-point context-attention predictions on validation only.

The cross-attention fitter consumes a target-free mixed verification-point
artifact rather than the older detector-proposal/overlay pair.  This command
proves that lineage before it joins validation-only SfM targets, so candidate
rank results cannot be produced by a row-order or prior-source mismatch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_mixed_multiscale_candidate_probe import (
    _metrics_for_mask,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_frozen_layout,
)
from feature_extract.vfm.localization.current_v3_candidate_probe import (
    validate_candidate_probability_contract,
)
from feature_extract.vfm.localization.mixed_multiscale_candidate_probe import (
    geometric_membership_from_residuals,
    materialize_mixed_candidate_reprojection_residuals,
    materialize_mixed_registered_identity_membership,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MixedVerificationPoints,
    load_mixed_verification_points,
)


PREDICTION_ARTIFACT_FORMAT = "multiscale_candidate_probe_predictions_v2"
SUPERVISION_MODE = "registered_track_identity"
PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
MIXED_CANDIDATE_INPUT = "mixed_verification_points_embedded_coarse_prior_v1"


@dataclass(frozen=True)
class _MixedCandidateAuditRows:
    """Minimal target-free row view accepted by shared SfM target join helpers."""

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    xy: np.ndarray
    candidate_tracks: np.ndarray
    candidate_bank_rows: np.ndarray
    points: MixedVerificationPoints


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context} lacks metadata_json")
    value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError(f"{context} metadata is not an object")
    return value


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(not np.isfinite(item) or item <= 0.0 for item in thresholds)
    ):
        raise ValueError("geometric thresholds must be non-empty unique positive values")
    return thresholds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--verification_points", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geometric_thresholds_px", default="2,5")
    parser.add_argument("--primary_geometric_threshold_px", type=float, default=5.0)
    return parser.parse_args(argv)


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("context-attention contract is not an object")
    required_false = (
        "contains_ground_truth",
        "contains_target_errors",
        "pose_or_ground_truth_used",
        "image_retrieval_or_submap_used",
        "whole_image_summary_or_global_used",
        "render",
    )
    if any(value.get(key) is not False for key in required_false):
        raise ValueError("context-attention contract is not target-free local evidence")
    if value.get("candidate_input_kind") != MIXED_CANDIDATE_INPUT:
        raise ValueError("audit requires the explicit mixed verification-point input contract")
    return value


def _load_layout_rows(
    *, contract: Mapping[str, Any], verification_points_path: Path
) -> tuple[_MixedCandidateAuditRows, dict[str, Any], dict[str, np.ndarray]]:
    layout_path = Path(str(contract.get("frozen_layout_features", "")))
    if not layout_path.is_file() or str(contract.get("frozen_layout_features_sha256", "")) != file_sha256_short(
        layout_path
    ):
        raise ValueError("context-attention frozen layout is stale")
    layout, layout_metadata = load_context_attention_frozen_layout(layout_path)
    points_path = Path(verification_points_path)
    points_sha = file_sha256_short(points_path)
    if (
        str(contract.get("candidate_input_lineage_sha256", "")) != points_sha
        or str(contract.get("proposals_sha256", "")) != points_sha
        or str(layout_metadata.get("verification_points_sha256", "")) != points_sha
        or str(layout_metadata.get("proposals_sha256", "")) != points_sha
    ):
        raise ValueError("mixed verification points differ from the frozen context layout")
    points = load_mixed_verification_points(points_path)
    rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    query_ids = np.asarray(layout["query_ids"]).astype(str)
    split_names = np.asarray(layout["split_names"]).astype(str)
    xy = np.asarray(layout["xy"], dtype=np.float32)
    tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    canonical = np.asarray(layout["candidate_canonical_rows"], dtype=np.int64)
    view_valid = np.asarray(layout["candidate_view_valid"], dtype=bool)
    if not (
        np.array_equal(rows, points.source_point_ids)
        and np.array_equal(query_ids, points.query_ids)
        and np.array_equal(split_names, points.split_names)
        and np.allclose(xy, points.xy, rtol=0.0, atol=1e-4)
        and np.array_equal(tracks, points.candidate_track_ids)
        and np.array_equal(canonical, points.candidate_bank_rows)
    ):
        raise ValueError("context-attention layout differs from mixed verification points")
    valid = tracks >= 0
    if (
        not len(rows)
        or len(np.unique(rows)) != len(rows)
        or np.any(valid & ~np.any(view_valid, axis=2))
        or np.any(~valid & np.any(view_valid, axis=2))
    ):
        raise ValueError("context-attention candidate/support-view layout is invalid")
    return (
        _MixedCandidateAuditRows(
            source_point_ids=rows,
            query_ids=query_ids,
            split_names=split_names,
            xy=xy,
            candidate_tracks=tracks,
            candidate_bank_rows=canonical,
            points=points,
        ),
        layout_metadata,
        {
            "candidate_view_valid": view_valid,
            "candidate_probabilities": np.asarray(
                points.candidate_prior_probabilities, dtype=np.float32
            ),
            "null_probabilities": np.asarray(points.null_probabilities, dtype=np.float32),
        },
    )


def _load_predictions(
    *,
    predictions_path: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    rows: _MixedCandidateAuditRows,
    candidate_view_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_view_valid",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
        "metadata_json",
    }
    with np.load(Path(predictions_path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"context-attention predictions lack {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required}
    metadata = _metadata(arrays, context="context-attention predictions")
    if (
        metadata.get("format") != PREDICTION_ARTIFACT_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used_for_prediction") is not False
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("supervision_mode") != SUPERVISION_MODE
        or metadata.get("probability_semantics") != PROBABILITY_SEMANTICS
        or metadata.get("candidate_input_kind") != MIXED_CANDIDATE_INPUT
        or str(metadata.get("features_sha256", ""))
        != str(contract.get("frozen_layout_features_sha256", ""))
        or str(metadata.get("proposals_sha256", ""))
        != str(contract.get("candidate_input_lineage_sha256", ""))
        or str(metadata.get("candidate_input_sha256", ""))
        != str(contract.get("candidate_input_lineage_sha256", ""))
        or str(metadata.get("context_attention_contract_sha256", ""))
        != file_sha256_short(Path(contract_path))
    ):
        raise ValueError("context-attention predictions violate the frozen target-free contract")
    expected_arrays = {
        "source_row_indices": rows.source_point_ids,
        "query_ids": rows.query_ids,
        "split_names": rows.split_names,
        "candidate_track_ids": rows.candidate_tracks,
        "candidate_view_valid": candidate_view_valid,
    }
    for key, expected in expected_arrays.items():
        if not np.array_equal(arrays[key], expected):
            raise ValueError(f"context-attention prediction {key} differs from frozen layout")
    families = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    valid = rows.candidate_tracks >= 0
    if (
        not families
        or len(set(families)) != len(families)
        or candidate.shape != (len(families), *valid.shape)
        or null.shape != (len(families), len(rows.source_point_ids))
    ):
        raise ValueError("context-attention prediction probability arrays are invalid")
    for family_index in range(len(families)):
        validate_candidate_probability_contract(candidate[family_index], null[family_index], valid)
    return candidate, null, families, metadata


def _source_metrics(
    *,
    source_mask: np.ndarray,
    baseline_candidate: np.ndarray,
    baseline_null: np.ndarray,
    probe_candidate: np.ndarray,
    probe_null: np.ndarray,
    geometry_labels_by_threshold: Mapping[str, np.ndarray],
    identity_labels: np.ndarray,
    identity_mask: np.ndarray,
    candidate_valid: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(source_mask, dtype=bool)
    result: dict[str, Any] = {
        "point_count": int(np.sum(mask)),
        "geometry_by_threshold": {
            key: _metrics_for_mask(
                baseline_candidate=baseline_candidate,
                baseline_null=baseline_null,
                probe_candidate=probe_candidate,
                probe_null=probe_null,
                labels=labels,
                candidate_valid=candidate_valid,
                row_mask=mask,
            )
            for key, labels in geometry_labels_by_threshold.items()
        },
    }
    exact_mask = mask & np.asarray(identity_mask, dtype=bool)
    result["exact_registered_identity"] = (
        None
        if not np.any(exact_mask)
        else _metrics_for_mask(
            baseline_candidate=baseline_candidate,
            baseline_null=baseline_null,
            probe_candidate=probe_candidate,
            probe_null=probe_null,
            labels=identity_labels,
            candidate_valid=candidate_valid,
            row_mask=exact_mask,
            exact_identity=True,
        )
    )
    return result


def audit_mixed_context_attention_candidate_probe(
    *,
    contract_path: Path,
    verification_points_path: Path,
    predictions_path: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    thresholds_px: Sequence[float],
    primary_threshold_px: float,
) -> dict[str, Any]:
    output = Path(output_dir)
    thresholds = tuple(float(value) for value in thresholds_px)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if float(primary_threshold_px) not in thresholds:
        raise ValueError("primary geometric threshold must be one of --geometric_thresholds_px")
    contract = _load_contract(Path(contract_path))
    rows, layout_metadata, base = _load_layout_rows(
        contract=contract, verification_points_path=Path(verification_points_path)
    )
    candidate, null, families, prediction_metadata = _load_predictions(
        predictions_path=Path(predictions_path),
        contract_path=Path(contract_path),
        contract=contract,
        rows=rows,
        candidate_view_valid=np.asarray(base["candidate_view_valid"], dtype=bool),
    )
    valid = rows.candidate_tracks >= 0
    validate_candidate_probability_contract(
        np.asarray(base["candidate_probabilities"], dtype=np.float32),
        np.asarray(base["null_probabilities"], dtype=np.float32),
        valid,
    )
    validation_rows = np.flatnonzero(rows.split_names == "validation")
    if not len(validation_rows):
        raise ValueError("frozen context-attention layout has no validation rows")

    # All target materialization occurs after every target-free input and
    # prediction lineage check above.
    residuals, geometry_target_audit = materialize_mixed_candidate_reprojection_residuals(
        features=rows,  # type: ignore[arg-type]
        projected_landmark_bank=Path(projected_landmark_bank),
        colmap_model_dir=Path(colmap_model_dir),
        row_indices=validation_rows,
        required_split="validation",
    )
    geometry_labels_by_threshold: dict[str, np.ndarray] = {}
    for threshold in thresholds:
        labels = np.zeros_like(valid, dtype=bool)
        labels[validation_rows] = geometric_membership_from_residuals(
            residuals=residuals,
            candidate_valid=valid[validation_rows],
            threshold_px=float(threshold),
        )[:, :-1]
        geometry_labels_by_threshold[f"{threshold:g}"] = labels
    identity_rows, _membership, sparse_identity_labels, identity_target_audit = (
        materialize_mixed_registered_identity_membership(
            features=rows,  # type: ignore[arg-type]
            colmap_model_dir=Path(colmap_model_dir),
            split_name="validation",
            identity_radius_px=float(prediction_metadata.get("registered_identity_radius_px", 0.0)),
        )
    )
    identity_labels = np.zeros_like(valid, dtype=bool)
    identity_labels[identity_rows] = sparse_identity_labels
    identity_mask = np.zeros((len(rows.source_point_ids),), dtype=bool)
    identity_mask[identity_rows] = True
    validation_mask = rows.split_names == "validation"

    report: dict[str, Any] = {
        "stage": "audit_frozen_mixed_context_attention_candidate_probe_validation",
        "protocol": {
            "fit_uses_train_targets_only": True,
            "fit_uses_train_registered_identity_targets_only": True,
            "training_supervision_mode": SUPERVISION_MODE,
            "prediction_probability_semantics": PROBABILITY_SEMANTICS,
            "prediction_artifact_frozen_before_validation_test_label_join": True,
            "validation_target_join_only": True,
            "test_target_labels_materialized": False,
            "test_used_for_model_selection": False,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": False,
            "render": False,
        },
        "inputs": {
            "contract": str(Path(contract_path)),
            "contract_sha256": file_sha256_short(Path(contract_path)),
            "frozen_layout": str(contract["frozen_layout_features"]),
            "frozen_layout_sha256": contract["frozen_layout_features_sha256"],
            "verification_points": str(Path(verification_points_path)),
            "verification_points_sha256": file_sha256_short(Path(verification_points_path)),
            "predictions": str(Path(predictions_path)),
            "predictions_sha256": file_sha256_short(Path(predictions_path)),
            "candidate_input_kind": prediction_metadata.get("candidate_input_kind"),
            "layout_support_view_selection": layout_metadata.get("support_view_selection"),
            **geometry_target_audit,
            "identity_target": identity_target_audit,
        },
        "thresholds": {
            "geometric_thresholds_px": list(thresholds),
            "primary_geometric_threshold_px": float(primary_threshold_px),
            "registered_identity_radius_px": float(
                prediction_metadata["registered_identity_radius_px"]
            ),
        },
        "validation": {
            "point_count": int(len(validation_rows)),
            "query_count": int(len(set(rows.query_ids[validation_rows].tolist()))),
            "registered_identity_scored_point_count": int(np.sum(identity_mask)),
        },
        "families": {},
    }
    point_sources = np.asarray(rows.points.point_sources).astype(str)
    for family_index, family in enumerate(families):
        all_metrics = _source_metrics(
            source_mask=validation_mask,
            baseline_candidate=np.asarray(base["candidate_probabilities"], dtype=np.float32),
            baseline_null=np.asarray(base["null_probabilities"], dtype=np.float32),
            probe_candidate=candidate[family_index],
            probe_null=null[family_index],
            geometry_labels_by_threshold=geometry_labels_by_threshold,
            identity_labels=identity_labels,
            identity_mask=identity_mask,
            candidate_valid=valid,
        )
        geometry_key = f"{float(primary_threshold_px):g}"
        family_report: dict[str, Any] = {
            "splits": {
                "validation": {
                    "geometry_set": all_metrics["geometry_by_threshold"][geometry_key],
                    "geometry_set_by_threshold": all_metrics["geometry_by_threshold"],
                    "exact_registered_identity": all_metrics["exact_registered_identity"],
                }
            },
            "by_point_source": {},
        }
        for source in sorted(set(point_sources[validation_mask].tolist())):
            source_metrics = _source_metrics(
                source_mask=validation_mask & (point_sources == str(source)),
                baseline_candidate=np.asarray(base["candidate_probabilities"], dtype=np.float32),
                baseline_null=np.asarray(base["null_probabilities"], dtype=np.float32),
                probe_candidate=candidate[family_index],
                probe_null=null[family_index],
                geometry_labels_by_threshold=geometry_labels_by_threshold,
                identity_labels=identity_labels,
                identity_mask=identity_mask,
                candidate_valid=valid,
            )
            family_report["by_point_source"][str(source)] = source_metrics
        report["families"][family] = family_report
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_mixed_context_attention_candidate_probe(
        contract_path=Path(args.contract),
        verification_points_path=Path(args.verification_points),
        predictions_path=Path(args.predictions),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        thresholds_px=_parse_thresholds(args.geometric_thresholds_px),
        primary_threshold_px=float(args.primary_geometric_threshold_px),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
