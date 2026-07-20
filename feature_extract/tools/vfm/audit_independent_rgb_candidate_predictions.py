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
    IDENTITY_TARGET_GEOMETRIC,
    _identity_metrics,
    _identity_training_masks,
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


def _first_positive_rank(
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top candidate and 1-based first-positive rank for each group."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    if (
        values.shape != positive.shape
        or values.shape != candidate_valid.shape
        or values.ndim != 2
    ):
        raise ValueError("identity rank arrays are incompatible")
    masked = np.where(candidate_valid, values, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    sorted_positive = np.take_along_axis(positive, order, axis=1)
    eligible = np.any(positive, axis=1) & np.any(candidate_valid, axis=1)
    rank = np.where(eligible, np.argmax(sorted_positive, axis=1) + 1, 0)
    return order[:, 0], rank.astype(np.int64, copy=False)


def _paired_identity_rank(
    *,
    baseline_scores: np.ndarray,
    probe_scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """Compare a probe against a baseline on the same candidate groups."""

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    probe = np.asarray(probe_scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    if (
        baseline.shape != probe.shape
        or baseline.shape != positive.shape
        or positive.shape != candidate_valid.shape
    ):
        raise ValueError("paired identity rank arrays are incompatible")
    baseline_top, baseline_rank = _first_positive_rank(
        baseline, labels=positive, valid=candidate_valid
    )
    probe_top, probe_rank = _first_positive_rank(
        probe, labels=positive, valid=candidate_valid
    )
    rows = np.any(positive, axis=1) & np.any(candidate_valid, axis=1)
    if not np.any(rows):
        return {
            "positive_group_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
        }
    base_rank = baseline_rank[rows]
    updated_rank = probe_rank[rows]
    baseline_correct = positive[rows, baseline_top[rows]]
    probe_correct = positive[rows, probe_top[rows]]
    return {
        "positive_group_count": int(np.count_nonzero(rows)),
        "rank_win_count": int(np.count_nonzero(updated_rank < base_rank)),
        "rank_loss_count": int(np.count_nonzero(updated_rank > base_rank)),
        "rank_tie_count": int(np.count_nonzero(updated_rank == base_rank)),
        "top1_rescue_count": int(np.count_nonzero(~baseline_correct & probe_correct)),
        "top1_harm_count": int(np.count_nonzero(baseline_correct & ~probe_correct)),
        "median_rank_delta_baseline_minus_probe": float(
            np.median(base_rank - updated_rank)
        ),
    }


def _coarse_prior_hard_negative_margins(
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    supervision_valid: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-minus-negative mass margins for supervised groups.

    ``scores`` are the deployed, per-candidate log probabilities.  Their
    groupwise normalizer cancels from this comparison, making the margin
    exactly equivalent to the coarse-prior-aware ranking objective when the
    scores were formed from ``log(prior) + evidence_weight * RGB_LLR``.
    """

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    supervised = np.asarray(supervision_valid, dtype=bool)
    candidate_prior = np.asarray(prior, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape != positive.shape
        or positive.shape != supervised.shape
        or supervised.shape != candidate_prior.shape
    ):
        raise ValueError("hard-negative mass arrays are incompatible")
    if not np.all(np.isfinite(candidate_prior)) or np.any(candidate_prior < 0.0):
        raise ValueError("hard-negative candidate priors must be finite and non-negative")

    positive = positive & supervised
    negative = ~positive & supervised
    active = np.any(positive, axis=1) & np.any(negative, axis=1)
    positive_log_mass = np.logaddexp.reduce(
        np.where(positive, values, -np.inf), axis=1
    )
    negative_log_mass = np.logaddexp.reduce(
        np.where(negative, values, -np.inf), axis=1
    )
    margin = np.full((len(values),), np.nan, dtype=np.float64)
    margin[active] = positive_log_mass[active] - negative_log_mass[active]

    prior_order = np.argsort(
        -np.where(supervised, candidate_prior, -np.inf), axis=1, kind="stable"
    )
    top_prior_candidate = prior_order[:, 0]
    coarse_top_is_hard_negative = negative[
        np.arange(len(negative)), top_prior_candidate
    ]
    return active, margin, coarse_top_is_hard_negative


def _coarse_prior_hard_negative_summary(
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    supervision_valid: np.ndarray,
    prior: np.ndarray,
) -> dict[str, object]:
    """Audit whether geometric positives overcome high-prior wrong tracks."""

    active, margins, coarse_top_is_hard_negative = (
        _coarse_prior_hard_negative_margins(
            scores,
            labels=labels,
            supervision_valid=supervision_valid,
            prior=prior,
        )
    )
    if not np.any(active):
        return {
            "active_group_count": 0,
            "coarse_top_hard_negative_group_count": 0,
            "positive_over_negative_count": 0,
            "positive_over_negative_fraction": None,
            "median_positive_minus_negative_log_mass": None,
            "mean_positive_minus_negative_log_mass": None,
            "mean_softplus_negative_margin": None,
            "coarse_top_hard_negative_positive_over_negative_fraction": None,
        }
    selected = margins[active]
    positive_wins = selected > 0.0
    coarse_wrong = active & coarse_top_is_hard_negative
    return {
        "active_group_count": int(np.count_nonzero(active)),
        "coarse_top_hard_negative_group_count": int(np.count_nonzero(coarse_wrong)),
        "positive_over_negative_count": int(np.count_nonzero(positive_wins)),
        "positive_over_negative_fraction": float(np.mean(positive_wins)),
        "median_positive_minus_negative_log_mass": float(np.median(selected)),
        "mean_positive_minus_negative_log_mass": float(np.mean(selected)),
        "mean_softplus_negative_margin": float(
            np.mean(np.logaddexp(0.0, -selected))
        ),
        "coarse_top_hard_negative_positive_over_negative_fraction": (
            None
            if not np.any(coarse_wrong)
            else float(np.mean(margins[coarse_wrong] > 0.0))
        ),
    }


def _paired_coarse_prior_hard_negative_mass(
    *,
    baseline_scores: np.ndarray,
    probe_scores: np.ndarray,
    labels: np.ndarray,
    supervision_valid: np.ndarray,
    prior: np.ndarray,
) -> dict[str, object]:
    """Paired change in the deployed coarse-prior-aware hard-negative margin."""

    baseline_active, baseline_margin, baseline_coarse_wrong = (
        _coarse_prior_hard_negative_margins(
            baseline_scores,
            labels=labels,
            supervision_valid=supervision_valid,
            prior=prior,
        )
    )
    probe_active, probe_margin, probe_coarse_wrong = (
        _coarse_prior_hard_negative_margins(
            probe_scores,
            labels=labels,
            supervision_valid=supervision_valid,
            prior=prior,
        )
    )
    if not np.array_equal(baseline_active, probe_active) or not np.array_equal(
        baseline_coarse_wrong, probe_coarse_wrong
    ):
        raise ValueError("paired hard-negative groups must be fixed")
    active = baseline_active
    if not np.any(active):
        return {
            "active_group_count": 0,
            "margin_win_count": 0,
            "margin_loss_count": 0,
            "margin_tie_count": 0,
            "positive_over_negative_rescue_count": 0,
            "positive_over_negative_harm_count": 0,
            "mean_margin_delta_probe_minus_baseline": None,
            "coarse_top_hard_negative_margin_win_count": 0,
            "coarse_top_hard_negative_margin_loss_count": 0,
        }
    delta = probe_margin[active] - baseline_margin[active]
    baseline_wins = baseline_margin[active] > 0.0
    probe_wins = probe_margin[active] > 0.0
    coarse_wrong = active & baseline_coarse_wrong
    return {
        "active_group_count": int(np.count_nonzero(active)),
        "margin_win_count": int(np.count_nonzero(delta > 0.0)),
        "margin_loss_count": int(np.count_nonzero(delta < 0.0)),
        "margin_tie_count": int(np.count_nonzero(delta == 0.0)),
        "positive_over_negative_rescue_count": int(
            np.count_nonzero(~baseline_wins & probe_wins)
        ),
        "positive_over_negative_harm_count": int(
            np.count_nonzero(baseline_wins & ~probe_wins)
        ),
        "mean_margin_delta_probe_minus_baseline": float(np.mean(delta)),
        "coarse_top_hard_negative_margin_win_count": int(
            np.count_nonzero(
                (probe_margin[coarse_wrong] - baseline_margin[coarse_wrong]) > 0.0
            )
        ),
        "coarse_top_hard_negative_margin_loss_count": int(
            np.count_nonzero(
                (probe_margin[coarse_wrong] - baseline_margin[coarse_wrong]) < 0.0
            )
        ),
    }


def _fused_scores(
    *,
    prior: np.ndarray,
    unknown: np.ndarray,
    llr: np.ndarray,
    measured: np.ndarray,
    valid: np.ndarray,
    weight: float,
) -> np.ndarray:
    fused, fused_unknown = fuse_candidate_log_likelihood_ratios(
        torch.from_numpy(prior),
        torch.from_numpy(unknown),
        torch.from_numpy(llr),
        measured_mask=torch.from_numpy(measured),
        candidate_valid=torch.from_numpy(valid),
        evidence_weight=float(weight),
    )
    if not torch.equal(fused_unknown, torch.from_numpy(unknown)):
        raise RuntimeError("RGB identity fusion changed unknown probability mass")
    return np.log(fused.numpy().clip(min=1e-30))


def audit_predictions(
    *,
    prediction_paths: Sequence[Path],
    baseline_prediction_paths: Sequence[Path] = (),
    candidate_evidence: Path,
    availability_evidence: Path,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    test_rows_csv: Path,
    identity_threshold_px: float,
    identity_negative_threshold_px: float,
    max_views: int,
    splits: Sequence[str] = ("validation", "test"),
) -> dict[str, object]:
    if not prediction_paths:
        raise ValueError("at least one prediction artifact is required")
    normalized_splits = tuple(str(name) for name in splits)
    invalid_splits = set(normalized_splits) - {"train", "validation", "test"}
    if invalid_splits or not normalized_splits:
        raise ValueError(
            "splits must contain train/validation/test: " f"{sorted(invalid_splits)}"
        )
    if baseline_prediction_paths and len(baseline_prediction_paths) != len(prediction_paths):
        raise ValueError(
            "baseline prediction count must equal probe prediction count for paired audit"
        )
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
    def load_many(paths: Sequence[Path], *, role: str) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
        loaded_arrays: list[dict[str, np.ndarray]] = []
        manifests: list[dict[str, object]] = []
        for path in paths:
            arrays, metadata = _load_predictions(Path(path))
            if metadata.get("candidate_evidence_sha256") != expected_candidate_hash:
                raise ValueError(f"candidate evidence hash mismatch ({role}): {path}")
            if metadata.get("availability_evidence_sha256") != expected_availability_hash:
                raise ValueError(f"availability evidence hash mismatch ({role}): {path}")
            loaded_arrays.append(arrays)
            manifests.append(
                {
                    "path": str(path),
                    "sha256": file_sha256_short(Path(path)),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                }
            )
        return loaded_arrays, manifests

    loaded, manifests = load_many(prediction_paths, role="probe")
    baseline_loaded, baseline_manifests = load_many(
        baseline_prediction_paths, role="baseline"
    )

    split_metrics: dict[str, object] = {}
    for split_name in normalized_splits:
        indices = data.indices_by_split[split_name]
        blocks = [_split_predictions(arrays, indices) for arrays in loaded]
        baseline_blocks = [
            _split_predictions(arrays, indices) for arrays in baseline_loaded
        ]
        measured = np.asarray(blocks[0]["candidate_rgb_measured"], dtype=bool)
        q_available = np.asarray(
            blocks[0]["rgb_full_top_l_availability_available"], dtype=bool
        )
        for block in blocks[1:] + baseline_blocks:
            if not np.array_equal(measured, block["candidate_rgb_measured"]):
                raise ValueError("RGB artifacts disagree on candidate measurement availability")
            if not np.array_equal(
                q_available, block["rgb_full_top_l_availability_available"]
            ):
                raise ValueError("RGB artifacts disagree on group measurement availability")
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
        baseline_llrs = [
            np.asarray(block["candidate_rgb_log_likelihood_ratios"], dtype=np.float32)
            for block in baseline_blocks
        ]
        baseline_ensemble_llr = (
            None
            if not baseline_llrs
            else np.mean(np.stack(baseline_llrs, axis=0), axis=0)
        )
        availability_labels = data.availability_valid[indices] & np.isfinite(
            data.availability_residuals[indices]
        ) & (
            data.availability_residuals[indices] <= float(identity_threshold_px)
        )
        any_available = np.any(availability_labels, axis=1)
        valid = data.candidate_valid[indices]
        prior = data.prior[indices]
        unknown = data.unknown[indices]
        geometric_labels, geometric_supervision_valid = _identity_training_masks(
            valid=valid,
            actual_query_observation=data.actual_query_observation[indices],
            actual_center_residuals=data.actual_center_residuals[indices],
            target_projection_residuals=data.residuals[indices],
            identity_target=IDENTITY_TARGET_GEOMETRIC,
            positive_threshold_px=float(identity_threshold_px),
            negative_threshold_px=float(identity_negative_threshold_px),
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
            fused_scores = _fused_scores(
                prior=prior,
                unknown=unknown,
                llr=ensemble_llr,
                measured=measured,
                valid=valid,
                weight=float(weight),
            )
            sweep_item: dict[str, object] = _identity_bundle(
                fused_scores,
                data=data,
                indices=indices,
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(identity_negative_threshold_px),
            )
            sweep_item["coarse_prior_hard_negative"] = (
                _coarse_prior_hard_negative_summary(
                    fused_scores,
                    labels=geometric_labels,
                    supervision_valid=geometric_supervision_valid,
                    prior=prior,
                )
            )
            if baseline_ensemble_llr is not None:
                baseline_scores = _fused_scores(
                    prior=prior,
                    unknown=unknown,
                    llr=baseline_ensemble_llr,
                    measured=measured,
                    valid=valid,
                    weight=float(weight),
                )
                sweep_item["paired_vs_baseline"] = {
                    "geometric_identity": _paired_identity_rank(
                        baseline_scores=baseline_scores,
                        probe_scores=fused_scores,
                        labels=geometric_labels,
                        valid=valid,
                    ),
                    "appearance_supervised_identity": _paired_identity_rank(
                        baseline_scores=baseline_scores,
                        probe_scores=fused_scores,
                        labels=appearance_labels,
                        valid=appearance_valid,
                    ),
                    "coarse_prior_hard_negative": (
                        _paired_coarse_prior_hard_negative_mass(
                            baseline_scores=baseline_scores,
                            probe_scores=fused_scores,
                            labels=geometric_labels,
                            supervision_valid=geometric_supervision_valid,
                            prior=prior,
                        )
                    ),
                }
            split_result["fusion_weight_sweep"][f"{weight:g}"] = sweep_item
        split_metrics[split_name] = split_result
    return {
        "stage": "independent_rgb_candidate_prediction_audit",
        "protocol": {
            "identity_positive": "actual_query_observation_center_residual_le_threshold",
            "identity_hard_negative": "gt_projection_residual_ge_negative_threshold",
            "geometric_downstream_target": "gt_projection_residual_le_threshold",
            "coarse_prior_hard_negative": (
                "same_group_geometric_positive_vs_residual_ge_negative_threshold_"
                "candidate_mass_under_deployed_coarse_prior_plus_rgb_evidence"
            ),
            "ensemble": "mean_log_likelihood_ratio_and_mean_availability_probability",
            "missing_rgb_log_likelihood_ratio": 0.0,
            "candidate_availability_mass_preserved": True,
            "paired_rank": bool(baseline_prediction_paths),
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
            "baseline_predictions": baseline_manifests,
        },
        "metrics": split_metrics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="comma-separated NPZ paths")
    parser.add_argument(
        "--baseline_predictions",
        default="",
        help=(
            "optional comma-separated NPZ paths aligned with --predictions; "
            "enables paired candidate-rank audit"
        ),
    )
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--availability_evidence", required=True)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--test_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity_threshold_px", type=float, default=2.0)
    parser.add_argument("--identity_negative_threshold_px", type=float, default=5.0)
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="comma-separated train/validation/test; defaults to validation,test",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prediction_paths = tuple(
        Path(value.strip())
        for value in str(args.predictions).split(",")
        if value.strip()
    )
    baseline_prediction_paths = tuple(
        Path(value.strip())
        for value in str(args.baseline_predictions).split(",")
        if value.strip()
    )
    splits = tuple(
        value.strip() for value in str(args.splits).split(",") if value.strip()
    )
    summary = audit_predictions(
        prediction_paths=prediction_paths,
        baseline_prediction_paths=baseline_prediction_paths,
        candidate_evidence=Path(args.candidate_evidence),
        availability_evidence=Path(args.availability_evidence),
        train_rows_csv=Path(args.train_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        test_rows_csv=Path(args.test_rows_csv),
        identity_threshold_px=float(args.identity_threshold_px),
        identity_negative_threshold_px=float(args.identity_negative_threshold_px),
        max_views=int(args.max_views),
        splits=splits,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
