"""Marginalize candidate RGB evidence and audit it before any PnP use."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    return arrays, metadata


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    result = np.where(
        finite,
        maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)),
        -np.inf,
    )
    return np.squeeze(result, axis=axis)


def marginalize_candidate_views(
    *,
    local_log_probabilities: np.ndarray,
    support_view_probabilities: np.ndarray,
    dustbin_probabilities: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return candidate validity and spatial likelihood without erasing missing mass."""

    log_probs = np.asarray(local_log_probabilities, dtype=np.float64)
    view = np.asarray(support_view_probabilities, dtype=np.float64).reshape(-1)
    dustbin = np.asarray(dustbin_probabilities, dtype=np.float64).reshape(-1)
    if log_probs.ndim != 2 or len(log_probs) != len(view) or len(view) != len(dustbin):
        raise ValueError("view likelihood arrays are not aligned")
    if len(view) == 0:
        raise ValueError("candidate has no support-view likelihood")
    if np.any(~np.isfinite(view)) or np.any(view < 0.0) or np.sum(view) > 1.0 + 2e-5:
        raise ValueError("available support-view probability mass is invalid")
    if np.any(~np.isfinite(dustbin)) or np.any(dustbin < 0.0) or np.any(dustbin > 1.0):
        raise ValueError("dustbin probabilities must lie in [0, 1]")
    normalizers = _logsumexp(log_probs, axis=1)
    if not np.allclose(normalizers, 0.0, rtol=0.0, atol=2e-3):
        raise ValueError("per-view spatial likelihood is not normalized")

    valid_view_mass = view * (1.0 - dustbin)
    invalid_probability = float(np.sum(view * dustbin))
    missing_probability = float(max(1.0 - np.sum(view), 0.0))
    valid_probability = float(np.sum(valid_view_mass))
    total = valid_probability + invalid_probability + missing_probability
    if not np.isclose(total, 1.0, rtol=0.0, atol=2e-5):
        raise RuntimeError("candidate RGB probability mass changed during marginalization")
    if valid_probability <= 0.0:
        conditional_log_probs = np.full((log_probs.shape[1],), -np.inf)
    else:
        joint = log_probs + np.log(np.clip(valid_view_mass, 1e-30, None))[:, None]
        conditional_log_probs = _logsumexp(joint, axis=0) - np.log(valid_probability)
    return {
        "conditional_local_log_probabilities": conditional_log_probs,
        "valid_probability": valid_probability,
        "invalid_probability": invalid_probability,
        "missing_probability": missing_probability,
    }


def _nearest_offset_indices(offsets_xy: np.ndarray, targets_xy: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    targets = np.asarray(targets_xy, dtype=np.float64)
    return np.argmin(
        np.sum((targets[:, None, :] - offsets[None, :, :]) ** 2, axis=2), axis=1
    )


def _identity_metrics(
    *,
    candidate_probabilities: np.ndarray,
    unknown_probabilities: np.ndarray,
    residuals_px: np.ndarray,
    threshold_px: float,
) -> dict[str, object]:
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    unknown = np.asarray(unknown_probabilities, dtype=np.float64).reshape(-1)
    correct = np.isfinite(residuals_px) & (residuals_px <= float(threshold_px))
    any_correct = np.any(correct, axis=1)
    correct_mass = np.sum(np.where(correct, probabilities, 0.0), axis=1)
    target_mass = np.where(any_correct, correct_mass, unknown)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked_correct = np.take_along_axis(correct, order, axis=1)
    first_correct = np.full((len(probabilities),), -1, dtype=np.int64)
    for rank in range(probabilities.shape[1]):
        take = (first_correct < 0) & ranked_correct[:, rank]
        first_correct[take] = rank + 1
    best = order[:, 0]
    candidate_top1_correct = correct[np.arange(len(correct)), best]
    state_top1_is_unknown = unknown >= probabilities[np.arange(len(probabilities)), best]
    state_top1_correct = np.where(
        any_correct, candidate_top1_correct & ~state_top1_is_unknown, state_top1_is_unknown
    )
    null_report = confidence_metrics(~any_correct, unknown)
    return {
        "threshold_px": float(threshold_px),
        "token_count": int(len(probabilities)),
        "tokens_with_correct_candidate": int(np.sum(any_correct)),
        "correct_candidate_availability_rate": float(np.mean(any_correct)),
        "identity_target_nll": float(
            np.mean(-np.log(np.clip(target_mass, 1e-12, 1.0)))
        ),
        "candidate_only_top1_correct_rate": float(np.mean(candidate_top1_correct)),
        "state_aware_top1_correct_rate": float(np.mean(state_top1_correct)),
        "unknown_top1_rate": float(np.mean(state_top1_is_unknown)),
        "mean_correct_probability_mass_when_present": (
            None
            if not np.any(any_correct)
            else float(np.mean(correct_mass[any_correct]))
        ),
        "mean_first_correct_rank_when_present": (
            None
            if not np.any(first_correct > 0)
            else float(np.mean(first_correct[first_correct > 0]))
        ),
        "first_correct_rank_counts": {
            str(rank): int(np.sum(first_correct == rank))
            for rank in range(1, probabilities.shape[1] + 1)
        },
        "null_detection": {
            **null_report,
            "null_count": int(np.sum(~any_correct)),
            "positive_prior": float(np.mean(~any_correct)),
        },
    }


def _spatial_metrics(
    *,
    offsets_xy: np.ndarray,
    conditional_log_probabilities: np.ndarray,
    valid_probabilities: np.ndarray,
    projected_offsets_xy: np.ndarray,
    residuals_px: np.ndarray,
) -> dict[str, object]:
    valid = (
        np.isfinite(projected_offsets_xy).all(axis=1)
        & np.isfinite(residuals_px)
        & np.all(
            np.abs(projected_offsets_xy)
            <= np.max(np.abs(offsets_xy), axis=0)[None, :] + 1e-5,
            axis=1,
        )
        & np.isfinite(conditional_log_probabilities).any(axis=1)
        & (valid_probabilities > 0.0)
    )
    if not np.any(valid):
        return {"count": 0}
    log_probs = conditional_log_probabilities[valid]
    targets = projected_offsets_xy[valid]
    nearest = _nearest_offset_indices(offsets_xy, targets)
    nll = -log_probs[np.arange(len(log_probs)), nearest]
    mode_indices = np.argmax(log_probs, axis=1)
    mode_offsets = offsets_xy[mode_indices]
    mode_epe = np.linalg.norm(mode_offsets - targets, axis=1)
    probabilities = np.exp(log_probs)
    mean_offsets = probabilities @ offsets_xy
    mean_epe = np.linalg.norm(mean_offsets - targets, axis=1)
    entropy = -np.sum(probabilities * log_probs, axis=1)
    baseline = np.asarray(residuals_px, dtype=np.float64)[valid]
    return {
        "count": int(np.sum(valid)),
        "spatial_nll_mean": float(np.mean(nll)),
        "spatial_nll_median": float(np.median(nll)),
        "mode_epe_median_px": float(np.median(mode_epe)),
        "mean_epe_median_px": float(np.median(mean_epe)),
        "center_baseline_epe_median_px": float(np.median(baseline)),
        "mode_improve_rate": float(np.mean(mode_epe < baseline)),
        "mean_improve_rate": float(np.mean(mean_epe < baseline)),
        "mode_recall_0p5px": float(np.mean(mode_epe <= 0.5)),
        "mode_recall_1px": float(np.mean(mode_epe <= 1.0)),
        "mode_recall_2px": float(np.mean(mode_epe <= 2.0)),
        "entropy_mean": float(np.mean(entropy)),
        "rgb_valid_probability_mean": float(np.mean(valid_probabilities[valid])),
    }


def build_and_audit_candidate_evidence_v3(
    *,
    candidate_evidence_path: Path,
    spatial_likelihood_path: Path,
    split_name: str,
    output_path: Path,
    validity_source: str = "inverse_dustbin",
) -> dict[str, object]:
    evidence, evidence_metadata = _load_npz(Path(candidate_evidence_path))
    likelihood, likelihood_metadata = _load_npz(Path(spatial_likelihood_path))
    if evidence_metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("unsupported candidate evidence artifact")
    if likelihood_metadata.get("format") not in {
        "candidate_spatial_likelihood_v3",
        "candidate_spatial_likelihood_v4",
        "candidate_spatial_likelihood_v5",
    }:
        raise ValueError("unsupported candidate spatial likelihood artifact")
    if str(split_name) not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation, or test")
    if str(validity_source) not in {"inverse_dustbin", "geometry_head"}:
        raise ValueError("validity_source must be inverse_dustbin or geometry_head")

    split_mask = np.asarray(evidence["split_names"]).astype(str) == str(split_name)
    if not np.any(split_mask):
        raise ValueError("candidate evidence contains no rows for requested split")
    selected_rows = np.asarray(evidence["selected_rows"], dtype=np.int64)[split_mask]
    query_ids = np.asarray(evidence["query_ids"]).astype(str)[split_mask]
    query_xy = np.asarray(evidence["query_xy"], dtype=np.float32)[split_mask]
    track_ids = np.asarray(evidence["candidate_track_ids"], dtype=np.int64)[split_mask]
    prototype_ids = np.asarray(
        evidence["candidate_prototype_ids"], dtype=np.int64
    )[split_mask]
    candidate_valid = np.asarray(evidence["candidate_valid"], dtype=bool)[split_mask]
    prior = np.asarray(
        evidence["candidate_prior_probabilities"], dtype=np.float64
    )[split_mask]
    prior_unknown = np.asarray(evidence["unknown_probability"], dtype=np.float64)[
        split_mask
    ]
    residuals = np.asarray(
        evidence["candidate_target_gt_residuals_px"], dtype=np.float64
    )[split_mask]
    candidate_shape = prior.shape
    offsets_xy = np.asarray(likelihood["offsets_xy"], dtype=np.float64)
    local_log_probs = np.asarray(
        likelihood["local_log_probabilities"], dtype=np.float64
    )
    likelihood_query_ids = np.asarray(likelihood["query_ids"]).astype(str)
    likelihood_source_rows = np.asarray(
        likelihood["source_query_rows"], dtype=np.int64
    )
    likelihood_track_ids = np.asarray(
        likelihood["candidate_track_ids"], dtype=np.int64
    )
    likelihood_prototype_ids = np.asarray(
        likelihood["candidate_prototype_ids"], dtype=np.int64
    )
    support_probabilities = np.asarray(
        likelihood["support_view_probabilities"], dtype=np.float64
    )
    dustbin_probabilities = np.asarray(
        likelihood["dustbin_probabilities"], dtype=np.float64
    )
    if str(validity_source) == "geometry_head":
        if "measurement_geometry_probabilities" not in likelihood:
            raise ValueError("spatial likelihood lacks measurement geometry probabilities")
        geometry_probabilities = np.asarray(
            likelihood["measurement_geometry_probabilities"], dtype=np.float64
        )
        if np.any(~np.isfinite(geometry_probabilities)):
            raise ValueError("measurement geometry probability contains missing values")
        per_view_invalid_probabilities = 1.0 - geometry_probabilities
    else:
        per_view_invalid_probabilities = dustbin_probabilities

    row_groups: dict[tuple[str, int, int, int], list[int]] = {}
    for row in range(len(likelihood_query_ids)):
        key = (
            str(likelihood_query_ids[row]),
            int(likelihood_source_rows[row]),
            int(likelihood_track_ids[row]),
            int(likelihood_prototype_ids[row]),
        )
        row_groups.setdefault(key, []).append(row)

    bin_count = int(offsets_xy.shape[0])
    conditional_log_probs = np.full(
        (*candidate_shape, bin_count), -np.inf, dtype=np.float32
    )
    rgb_valid = np.zeros(candidate_shape, dtype=np.float32)
    rgb_invalid = np.zeros(candidate_shape, dtype=np.float32)
    rgb_missing = np.ones(candidate_shape, dtype=np.float32)
    projected_offsets = np.full((*candidate_shape, 2), np.nan, dtype=np.float32)
    candidate_group_count = 0
    support_view_rows = []
    for row in range(candidate_shape[0]):
        for column in range(candidate_shape[1]):
            if not candidate_valid[row, column]:
                continue
            key = (
                str(query_ids[row]),
                int(selected_rows[row]),
                int(track_ids[row, column]),
                int(prototype_ids[row, column]),
            )
            view_rows = row_groups.pop(key, [])
            if not view_rows:
                continue
            all_indices = np.asarray(view_rows, dtype=np.int64)
            indices = all_indices[np.isfinite(support_probabilities[all_indices])]
            if len(indices) == 0:
                continue
            candidate_group_count += 1
            result = marginalize_candidate_views(
                local_log_probabilities=local_log_probs[indices],
                support_view_probabilities=support_probabilities[indices],
                dustbin_probabilities=per_view_invalid_probabilities[indices],
            )
            conditional_log_probs[row, column] = np.asarray(
                result["conditional_local_log_probabilities"], dtype=np.float32
            )
            rgb_valid[row, column] = float(result["valid_probability"])
            rgb_invalid[row, column] = float(result["invalid_probability"])
            rgb_missing[row, column] = float(result["missing_probability"])
            projected_xy = np.asarray(
                likelihood["target_gt_projected_xy"], dtype=np.float64
            )[indices]
            if not np.allclose(projected_xy, projected_xy[:1], rtol=0.0, atol=1e-5):
                raise ValueError("support views disagree on candidate GT projection target")
            projected_offsets[row, column] = projected_xy[0] - query_xy[row]
            nearest = _nearest_offset_indices(
                offsets_xy,
                np.broadcast_to(projected_offsets[row, column], (len(indices), 2)),
            )
            view_mode = offsets_xy[np.argmax(local_log_probs[indices], axis=1)]
            view_epe = np.linalg.norm(
                view_mode - projected_offsets[row, column][None, :], axis=1
            )
            view_nll = -local_log_probs[indices, nearest]
            if np.all(
                np.abs(projected_offsets[row, column])
                <= np.max(np.abs(offsets_xy), axis=0) + 1e-5
            ):
                available_mass = float(np.sum(support_probabilities[indices]))
                normalized_view = support_probabilities[indices] / max(
                    available_mass, 1e-12
                )
                support_view_rows.append(
                    {
                        "learned_expected_mode_epe_px": float(
                            np.sum(normalized_view * view_epe)
                        ),
                        "oracle_mode_epe_px": float(np.min(view_epe)),
                        "learned_expected_spatial_nll": float(
                            np.sum(normalized_view * view_nll)
                        ),
                        "oracle_spatial_nll": float(np.min(view_nll)),
                    }
                )
    if row_groups:
        examples = list(row_groups)[:3]
        raise ValueError(f"likelihood contains candidates absent from evidence: {examples}")

    posterior_candidates = prior * rgb_valid
    posterior_unknown = prior_unknown + np.sum(prior * (1.0 - rgb_valid), axis=1)
    posterior_total = np.sum(posterior_candidates, axis=1) + posterior_unknown
    if not np.allclose(posterior_total, 1.0, rtol=0.0, atol=5e-5):
        raise RuntimeError("candidate posterior and unknown mass do not sum to one")

    identity = {}
    for threshold in (1.0, 2.0, 5.0):
        identity[str(int(threshold))] = {
            "prior": _identity_metrics(
                candidate_probabilities=prior,
                unknown_probabilities=prior_unknown,
                residuals_px=residuals,
                threshold_px=threshold,
            ),
            "rgb_posterior": _identity_metrics(
                candidate_probabilities=posterior_candidates,
                unknown_probabilities=posterior_unknown,
                residuals_px=residuals,
                threshold_px=threshold,
            ),
        }
    flat_valid = candidate_valid.reshape(-1)
    spatial = _spatial_metrics(
        offsets_xy=offsets_xy,
        conditional_log_probabilities=conditional_log_probs.reshape(-1, bin_count)[
            flat_valid
        ],
        valid_probabilities=rgb_valid.reshape(-1)[flat_valid],
        projected_offsets_xy=projected_offsets.reshape(-1, 2)[flat_valid],
        residuals_px=residuals.reshape(-1)[flat_valid],
    )
    support_view_audit = {
        "candidate_count": int(len(support_view_rows)),
        "learned_expected_mode_epe_median_px": float(
            np.median(
                [row["learned_expected_mode_epe_px"] for row in support_view_rows]
            )
        ),
        "oracle_mode_epe_median_px": float(
            np.median([row["oracle_mode_epe_px"] for row in support_view_rows])
        ),
        "learned_expected_spatial_nll_mean": float(
            np.mean(
                [row["learned_expected_spatial_nll"] for row in support_view_rows]
            )
        ),
        "oracle_spatial_nll_mean": float(
            np.mean([row["oracle_spatial_nll"] for row in support_view_rows])
        ),
    }

    output_metadata = {
        "format": "candidate_evidence_v3_rgb_marginalized",
        "format_version": 3,
        "split": str(split_name),
        "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence_path)),
        "spatial_likelihood_sha256": file_sha256_short(Path(spatial_likelihood_path)),
        "measurement_checkpoint_sha256": likelihood_metadata.get(
            "measurement_checkpoint_sha256"
        ),
        "view_marginalization": "logsumexp_of_support_prior_times_validity_times_spatial_likelihood",
        "candidate_posterior": "identity_prior_times_rgb_valid_probability",
        "candidate_validity_source": str(validity_source),
        "unknown_posterior": "prior_unknown_plus_candidate_invalid_and_missing_mass",
        "ground_truth_used_for_posterior": False,
        "ground_truth_arrays_target_only": [
            "candidate_target_gt_residuals_px",
            "candidate_target_projected_offset_xy",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        selected_rows=selected_rows,
        query_ids=query_ids,
        query_xy=query_xy,
        candidate_track_ids=track_ids,
        candidate_prototype_ids=prototype_ids,
        candidate_valid=candidate_valid,
        candidate_prior_probabilities=prior.astype(np.float32),
        prior_unknown_probability=prior_unknown.astype(np.float32),
        candidate_rgb_valid_probabilities=rgb_valid,
        candidate_rgb_invalid_probabilities=rgb_invalid,
        candidate_rgb_missing_probabilities=rgb_missing,
        candidate_posterior_probabilities=posterior_candidates.astype(np.float32),
        posterior_unknown_probability=posterior_unknown.astype(np.float32),
        offsets_xy=offsets_xy.astype(np.float32),
        candidate_conditional_local_log_probabilities=conditional_log_probs.astype(
            np.float16
        ),
        candidate_target_projected_offset_xy=projected_offsets,
        candidate_target_gt_residuals_px=residuals.astype(np.float32),
        metadata_json=np.asarray(json.dumps(output_metadata, sort_keys=True), dtype=np.str_),
    )
    report = {
        "stage": "candidate_evidence_v3_pre_pnp_audit",
        "protocol": output_metadata,
        "coverage": {
            "token_count": int(candidate_shape[0]),
            "candidate_slot_count": int(np.sum(candidate_valid)),
            "candidate_with_rgb_count": int(candidate_group_count),
            "candidate_with_rgb_rate": float(
                candidate_group_count / max(int(np.sum(candidate_valid)), 1)
            ),
            "mean_rgb_valid_probability": float(np.mean(rgb_valid[candidate_valid])),
            "mean_rgb_missing_probability": float(np.mean(rgb_missing[candidate_valid])),
        },
        "identity": identity,
        "spatial": spatial,
        "support_view_oracle_gap": support_view_audit,
        "outputs": {
            "marginalized_evidence": str(output),
            "marginalized_evidence_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
