"""Audit aligned independent-RGB candidate predictions without rerunning images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_training import (
    CandidateRGBTrainingData,
    _identity_metrics,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    fuse_candidate_log_likelihood_ratios,
)


def _load_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != "independent_rgb_candidate_predictions_v1":
        raise ValueError(f"unsupported RGB candidate prediction format: {path}")
    required = {
        "evidence_row_indices",
        "candidate_rgb_log_likelihood_ratios",
        "candidate_rgb_measured",
        "rgb_full_top_l_availability_available",
        "rgb_full_top_l_availability_probability",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"RGB candidate predictions lack arrays: {sorted(missing)}")
    return arrays, metadata


def _split_predictions(
    arrays: dict[str, np.ndarray], evidence_indices: np.ndarray
) -> dict[str, np.ndarray]:
    rows = np.asarray(arrays["evidence_row_indices"], dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("RGB prediction evidence rows are not unique")
    position_by_row = {int(row): position for position, row in enumerate(rows.tolist())}
    try:
        positions = np.asarray(
            [position_by_row[int(row)] for row in evidence_indices.tolist()],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"RGB predictions do not cover evidence row {error.args[0]}") from error
    return {
        key: np.asarray(value)[positions]
        for key, value in arrays.items()
        if np.asarray(value).ndim > 0 and int(np.asarray(value).shape[0]) == len(rows)
    }


def _identity_bundle(
    scores: np.ndarray,
    *,
    data: CandidateRGBTrainingData,
    indices: np.ndarray,
    identity_threshold_px: float,
    identity_negative_threshold_px: float,
) -> dict[str, object]:
    valid = data.candidate_valid[indices]
    geometric_labels = valid & np.isfinite(data.residuals[indices]) & (
        data.residuals[indices] <= float(identity_threshold_px)
    )
    appearance_labels = (
        valid
        & data.actual_query_observation[indices]
        & np.isfinite(data.actual_center_residuals[indices])
        & (data.actual_center_residuals[indices] <= float(identity_threshold_px))
    )
    appearance_valid = valid & (
        appearance_labels
        | (
            np.isfinite(data.residuals[indices])
            & (data.residuals[indices] >= float(identity_negative_threshold_px))
        )
    )
    prior = data.prior[indices]
    return {
        "geometric_identity": _identity_metrics(
            scores, labels=geometric_labels, valid=valid, prior=prior
        ),
        "appearance_supervised_identity": _identity_metrics(
            scores,
            labels=appearance_labels,
            valid=appearance_valid,
            prior=prior,
        ),
    }


def audit_predictions(
    *,
    prediction_paths: Sequence[Path],
    candidate_evidence: Path,
    availability_evidence: Path,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    test_rows_csv: Path,
    identity_threshold_px: float,
    identity_negative_threshold_px: float,
    max_views: int,
) -> dict[str, object]:
    if not prediction_paths:
        raise ValueError("at least one prediction artifact is required")
    data = CandidateRGBTrainingData(
        candidate_evidence=Path(candidate_evidence),
        availability_evidence=Path(availability_evidence),
        train_rows_csv=Path(train_rows_csv),
        validation_rows_csv=Path(validation_rows_csv),
        test_rows_csv=Path(test_rows_csv),
        max_views=int(max_views),
    )
    expected_candidate_hash = file_sha256_short(Path(candidate_evidence))
    expected_availability_hash = file_sha256_short(Path(availability_evidence))
    loaded = []
    manifests = []
    for path in prediction_paths:
        arrays, metadata = _load_predictions(Path(path))
        if metadata.get("candidate_evidence_sha256") != expected_candidate_hash:
            raise ValueError(f"candidate evidence hash mismatch: {path}")
        if metadata.get("availability_evidence_sha256") != expected_availability_hash:
            raise ValueError(f"availability evidence hash mismatch: {path}")
        loaded.append(arrays)
        manifests.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(Path(path)),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            }
        )

    split_metrics: dict[str, object] = {}
    for split_name in ("validation", "test"):
        indices = data.indices_by_split[split_name]
        blocks = [_split_predictions(arrays, indices) for arrays in loaded]
        measured = np.asarray(blocks[0]["candidate_rgb_measured"], dtype=bool)
        q_available = np.asarray(
            blocks[0]["rgb_full_top_l_availability_available"], dtype=bool
        )
        for block in blocks[1:]:
            if not np.array_equal(measured, block["candidate_rgb_measured"]):
                raise ValueError("RGB seeds disagree on candidate measurement availability")
            if not np.array_equal(
                q_available, block["rgb_full_top_l_availability_available"]
            ):
                raise ValueError("RGB seeds disagree on group measurement availability")
        llrs = [
            np.asarray(block["candidate_rgb_log_likelihood_ratios"], dtype=np.float32)
            for block in blocks
        ]
        q_probabilities = [
            np.asarray(
                block["rgb_full_top_l_availability_probability"], dtype=np.float32
            )
            for block in blocks
        ]
        ensemble_llr = np.mean(np.stack(llrs, axis=0), axis=0)
        ensemble_q = np.mean(np.stack(q_probabilities, axis=0), axis=0)
        availability_labels = data.availability_valid[indices] & np.isfinite(
            data.availability_residuals[indices]
        ) & (
            data.availability_residuals[indices] <= float(identity_threshold_px)
        )
        any_available = np.any(availability_labels, axis=1)
        valid = data.candidate_valid[indices]
        prior = data.prior[indices]
        unknown = data.unknown[indices]
        split_result: dict[str, object] = {
            "coarse_prior": _identity_bundle(
                np.log(prior.clip(min=1e-30)),
                data=data,
                indices=indices,
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(identity_negative_threshold_px),
            ),
            "seeds": [
                _identity_bundle(
                    llr,
                    data=data,
                    indices=indices,
                    identity_threshold_px=float(identity_threshold_px),
                    identity_negative_threshold_px=float(
                        identity_negative_threshold_px
                    ),
                )
                for llr in llrs
            ],
            "ensemble_rgb_only": _identity_bundle(
                ensemble_llr,
                data=data,
                indices=indices,
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(identity_negative_threshold_px),
            ),
            "full_top_l_availability": {
                "seeds": [
                    confidence_metrics(any_available[q_available], value[q_available])
                    for value in q_probabilities
                ],
                "ensemble": confidence_metrics(
                    any_available[q_available], ensemble_q[q_available]
                ),
                "measurable_group_count": int(np.count_nonzero(q_available)),
            },
            "fusion_weight_sweep": {},
        }
        for weight in (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
            fused, fused_unknown = fuse_candidate_log_likelihood_ratios(
                torch.from_numpy(prior),
                torch.from_numpy(unknown),
                torch.from_numpy(ensemble_llr),
                measured_mask=torch.from_numpy(measured),
                candidate_valid=torch.from_numpy(valid),
                evidence_weight=float(weight),
            )
            if not torch.equal(fused_unknown, torch.from_numpy(unknown)):
                raise RuntimeError("RGB identity fusion changed unknown probability mass")
            split_result["fusion_weight_sweep"][f"{weight:g}"] = _identity_bundle(
                np.log(fused.numpy().clip(min=1e-30)),
                data=data,
                indices=indices,
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(identity_negative_threshold_px),
            )
        split_metrics[split_name] = split_result
    return {
        "stage": "independent_rgb_candidate_prediction_audit",
        "protocol": {
            "identity_positive": "actual_query_observation_center_residual_le_threshold",
            "identity_hard_negative": "gt_projection_residual_ge_negative_threshold",
            "geometric_downstream_target": "gt_projection_residual_le_threshold",
            "ensemble": "mean_log_likelihood_ratio_and_mean_availability_probability",
            "missing_rgb_log_likelihood_ratio": 0.0,
            "candidate_availability_mass_preserved": True,
        },
        "config": {
            "identity_threshold_px": float(identity_threshold_px),
            "identity_negative_threshold_px": float(identity_negative_threshold_px),
            "max_views": int(max_views),
        },
        "inputs": {
            "candidate_evidence": str(candidate_evidence),
            "candidate_evidence_sha256": expected_candidate_hash,
            "availability_evidence": str(availability_evidence),
            "availability_evidence_sha256": expected_availability_hash,
            "predictions": manifests,
        },
        "metrics": split_metrics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="comma-separated NPZ paths")
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--availability_evidence", required=True)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--test_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity_threshold_px", type=float, default=2.0)
    parser.add_argument("--identity_negative_threshold_px", type=float, default=5.0)
    parser.add_argument("--max_views", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prediction_paths = tuple(
        Path(value.strip())
        for value in str(args.predictions).split(",")
        if value.strip()
    )
    summary = audit_predictions(
        prediction_paths=prediction_paths,
        candidate_evidence=Path(args.candidate_evidence),
        availability_evidence=Path(args.availability_evidence),
        train_rows_csv=Path(args.train_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        test_rows_csv=Path(args.test_rows_csv),
        identity_threshold_px=float(args.identity_threshold_px),
        identity_negative_threshold_px=float(args.identity_negative_threshold_px),
        max_views=int(args.max_views),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
