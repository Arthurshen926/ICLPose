"""Audit rank-binned coverage of frozen target-free RGB spatial modes.

The candidate posterior and its GT residuals are joined only in this external
diagnostic.  Spatial predictions remain target-free and are never modified by
the audit.  The report distinguishes a genuine top-K information plateau from
a plateau caused by candidates that were never materialized for RGB scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


SPATIAL_FORMAT = "candidate_spatial_likelihood_v7"
EVIDENCE_FORMAT = "candidate_evidence_v3"
RANK_BINS = ((1, 5), (6, 10), (11, 20))


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one artifact path is required")
    return paths


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: artifact has no metadata_json")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must contain an object")
    return arrays, metadata


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _spatial_view_index(
    paths: Sequence[Path],
) -> tuple[
    dict[tuple[int, int, int], list[dict[str, float | int]]],
    list[dict[str, Any]],
]:
    index: dict[tuple[int, int, int], list[dict[str, float | int]]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for path in paths:
        arrays, metadata = _load_npz(path)
        if metadata.get("format") != SPATIAL_FORMAT:
            raise ValueError(f"{path}: unsupported spatial artifact format")
        if (
            bool(metadata.get("contains_ground_truth_arrays"))
            or bool(metadata.get("ground_truth_loaded_by_inference_process"))
            or bool(metadata.get("pose_or_ground_truth_used_for_inference"))
            or not bool(metadata.get("prediction_frozen_before_target_join"))
        ):
            raise ValueError(f"{path}: spatial artifact is not target-free")
        required = {
            "source_query_rows",
            "candidate_measurement_ranks",
            "candidate_track_ids",
            "candidate_prototype_ids",
            "support_view_ranks",
            "support_view_probabilities",
            "local_log_probabilities",
            "dustbin_probabilities",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"{path}: missing required arrays {missing}")
        source_rows = np.asarray(arrays["source_query_rows"], dtype=np.int64)
        candidate_ranks = np.asarray(
            arrays["candidate_measurement_ranks"], dtype=np.int64
        )
        tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        prototypes = np.asarray(arrays["candidate_prototype_ids"], dtype=np.int64)
        view_ranks = np.asarray(arrays["support_view_ranks"], dtype=np.int64)
        view_probability = np.asarray(
            arrays["support_view_probabilities"], dtype=np.float64
        )
        local_log = np.asarray(arrays["local_log_probabilities"], dtype=np.float64)
        dustbin = np.asarray(arrays["dustbin_probabilities"], dtype=np.float64)
        count = len(source_rows)
        if not (
            candidate_ranks.shape
            == tracks.shape
            == prototypes.shape
            == view_ranks.shape
            == view_probability.shape
            == dustbin.shape
            == (count,)
            and local_log.shape[0] == count
        ):
            raise ValueError(f"{path}: spatial rows are not aligned")
        if np.any(candidate_ranks <= 0) or np.any(view_ranks < 0):
            raise ValueError(f"{path}: spatial ranks are invalid")
        if np.any(~np.isfinite(view_probability)) or np.any(
            (view_probability < 0.0) | (view_probability > 1.0)
        ):
            raise ValueError(f"{path}: support-view probabilities are invalid")
        if np.any(~np.isfinite(dustbin)) or np.any((dustbin < 0.0) | (dustbin > 1.0)):
            raise ValueError(f"{path}: dustbin probabilities are invalid")
        local_probability = np.exp(local_log)
        if not np.allclose(
            np.sum(local_probability, axis=1),
            1.0,
            rtol=0.0,
            atol=3e-3,
        ):
            raise ValueError(f"{path}: local RGB probability rows are not normalized")
        entropy = -np.sum(local_probability * local_log, axis=1)
        peak = np.max(local_probability, axis=1)
        for row in range(count):
            key = (int(source_rows[row]), int(tracks[row]), int(prototypes[row]))
            value = {
                "candidate_rank": int(candidate_ranks[row]),
                "support_view_rank": int(view_ranks[row]),
                "support_probability": float(view_probability[row]),
                "dustbin_probability": float(dustbin[row]),
                "entropy": float(entropy[row]),
                "mode_peak": float(peak[row]),
            }
            previous = index.setdefault(key, [])
            if any(
                int(existing["support_view_rank"])
                == int(value["support_view_rank"])
                for existing in previous
            ):
                raise ValueError(f"{path}: duplicate support-view rank for {key}")
            previous.append(value)
        metadata_rows.append(metadata)
    for rows in index.values():
        rows.sort(key=lambda row: int(row["support_view_rank"]))
    return index, metadata_rows


def _rank_bin_name(lower: int, upper: int) -> str:
    return f"rank_{int(lower)}_{int(upper)}"


def _summarize_records(
    records: Sequence[Mapping[str, float | int | bool]],
) -> dict[str, object]:
    if not records:
        return {
            "candidate_slot_count": 0,
            "correct_candidate_count_2px": 0,
            "correct_candidate_occurrence_rate_2px": None,
            "candidate_prior_mass": {"sum": 0.0, "mean": None},
            "spatial_candidate_coverage_rate": None,
            "correct_candidate_spatial_coverage_rate_2px": None,
            "valid_support_view_count": _distribution(()),
            "support_view_probability_mass": _distribution(()),
            "missing_support_view_probability": _distribution(()),
            "dustbin_probability": _distribution(()),
            "spatial_entropy": _distribution(()),
            "spatial_mode_peak": _distribution(()),
        }
    count = len(records)
    correct = np.asarray([bool(row["correct_2px"]) for row in records], dtype=bool)
    covered = np.asarray([bool(row["spatial_covered"]) for row in records], dtype=bool)
    priors = np.asarray([float(row["prior"]) for row in records], dtype=np.float64)
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in records]

    return {
        "candidate_slot_count": int(count),
        "correct_candidate_count_2px": int(np.count_nonzero(correct)),
        "correct_candidate_occurrence_rate_2px": float(np.mean(correct)),
        "candidate_prior_mass": {
            "sum": float(np.sum(priors)),
            "mean": float(np.mean(priors)),
        },
        "spatial_candidate_coverage_rate": float(np.mean(covered)),
        "correct_candidate_spatial_coverage_rate_2px": (
            None
            if not np.any(correct)
            else float(np.mean(covered[correct]))
        ),
        "valid_support_view_count": _distribution(values("support_view_count")),
        "support_view_probability_mass": _distribution(values("support_mass")),
        "missing_support_view_probability": _distribution(values("missing_mass")),
        "dustbin_probability": _distribution(values("dustbin")),
        "spatial_entropy": _distribution(values("entropy")),
        "spatial_mode_peak": _distribution(values("mode_peak")),
    }


def audit_candidate_spatial_rank_coverage(
    *,
    candidate_evidence_path: Path,
    candidate_spatial_likelihood_paths: Sequence[Path],
    output_path: Path,
    correct_threshold_px: float = 2.0,
) -> dict[str, object]:
    """Build an external coverage audit for a frozen spatial top-K artifact."""

    if float(correct_threshold_px) <= 0.0:
        raise ValueError("correct_threshold_px must be positive")
    evidence, evidence_metadata = _load_npz(Path(candidate_evidence_path))
    if evidence_metadata.get("format") != EVIDENCE_FORMAT:
        raise ValueError("candidate evidence must use the v3 probability contract")
    required = {
        "selected_rows",
        "split_names",
        "candidate_valid",
        "candidate_track_ids",
        "candidate_prototype_ids",
        "candidate_prior_probabilities",
        "candidate_score_ranks",
        "unknown_probability",
        "candidate_target_gt_residuals_px",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise ValueError(f"candidate evidence lacks {missing}")
    selected_rows = np.asarray(evidence["selected_rows"], dtype=np.int64)
    split_names = np.asarray(evidence["split_names"]).astype(str)
    valid = np.asarray(evidence["candidate_valid"], dtype=bool)
    tracks = np.asarray(evidence["candidate_track_ids"], dtype=np.int64)
    prototypes = np.asarray(evidence["candidate_prototype_ids"], dtype=np.int64)
    priors = np.asarray(evidence["candidate_prior_probabilities"], dtype=np.float64)
    ranks = np.asarray(evidence["candidate_score_ranks"], dtype=np.int64)
    unknown = np.asarray(evidence["unknown_probability"], dtype=np.float64).reshape(-1)
    residuals = np.asarray(
        evidence["candidate_target_gt_residuals_px"], dtype=np.float64
    )
    candidate_shape = valid.shape
    if not (
        selected_rows.shape == split_names.shape == unknown.shape == (candidate_shape[0],)
        and tracks.shape
        == prototypes.shape
        == priors.shape
        == ranks.shape
        == residuals.shape
        == candidate_shape
    ):
        raise ValueError("candidate evidence arrays are not aligned")
    if np.any(priors[valid] < 0.0) or np.any(~np.isfinite(priors[valid])):
        raise ValueError("candidate evidence priors are invalid")
    posterior_total = np.sum(np.where(valid, priors, 0.0), axis=1) + unknown
    if not np.allclose(posterior_total, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError("candidate evidence posterior mass is not conserved")

    spatial_index, spatial_metadata = _spatial_view_index(
        candidate_spatial_likelihood_paths
    )
    records_by_split_and_bin: dict[tuple[str, str], list[dict[str, float | int | bool]]] = {}
    materialized_mass = np.zeros((candidate_shape[0],), dtype=np.float64)
    unmaterialized_mass = np.zeros((candidate_shape[0],), dtype=np.float64)
    unmatched_spatial_keys = set(spatial_index)
    for row in range(candidate_shape[0]):
        for column in range(candidate_shape[1]):
            if not bool(valid[row, column]):
                continue
            key = (
                int(selected_rows[row]),
                int(tracks[row, column]),
                int(prototypes[row, column]),
            )
            views = spatial_index.get(key, [])
            if views:
                unmatched_spatial_keys.discard(key)
                artifact_ranks = {int(view["candidate_rank"]) for view in views}
                if artifact_ranks != {int(ranks[row, column])}:
                    raise ValueError(
                        f"spatial rank differs from frozen evidence rank for {key}"
                    )
                support_mass = float(
                    sum(float(view["support_probability"]) for view in views)
                )
                if support_mass > 1.0 + 2e-5:
                    raise ValueError(f"spatial support mass exceeds one for {key}")
                normalized = np.asarray(
                    [float(view["support_probability"]) for view in views],
                    dtype=np.float64,
                )
                normalizer = max(float(np.sum(normalized)), 1e-12)
                dustbin = float(
                    np.sum(
                        normalized
                        * np.asarray(
                            [float(view["dustbin_probability"]) for view in views]
                        )
                    )
                    / normalizer
                )
                entropy = float(
                    np.sum(
                        normalized
                        * np.asarray([float(view["entropy"]) for view in views])
                    )
                    / normalizer
                )
                peak = float(
                    np.sum(
                        normalized
                        * np.asarray([float(view["mode_peak"]) for view in views])
                    )
                    / normalizer
                )
                materialized_mass[row] += float(priors[row, column])
            else:
                support_mass = 0.0
                dustbin = float("nan")
                entropy = float("nan")
                peak = float("nan")
                unmaterialized_mass[row] += float(priors[row, column])
            candidate_rank = int(ranks[row, column])
            for lower, upper in RANK_BINS:
                if lower <= candidate_rank <= upper:
                    record = {
                        "prior": float(priors[row, column]),
                        "correct_2px": bool(
                            np.isfinite(residuals[row, column])
                            and residuals[row, column] <= float(correct_threshold_px)
                        ),
                        "spatial_covered": bool(views),
                        "support_view_count": int(len(views)),
                        "support_mass": float(support_mass),
                        "missing_mass": float(max(1.0 - support_mass, 0.0)),
                        "dustbin": dustbin,
                        "entropy": entropy,
                        "mode_peak": peak,
                    }
                    records_by_split_and_bin.setdefault(
                        (str(split_names[row]), _rank_bin_name(lower, upper)), []
                    ).append(record)
                    break
    if unmatched_spatial_keys:
        examples = sorted(unmatched_spatial_keys)[:3]
        raise ValueError(
            "spatial artifact contains candidates absent from frozen evidence: "
            f"{examples}"
        )
    effective_null = unknown + unmaterialized_mass
    topk_mass_total = materialized_mass + effective_null
    if not np.allclose(topk_mass_total, 1.0, rtol=0.0, atol=3e-5):
        raise RuntimeError("materialized candidate and effective null mass diverged")

    summaries: dict[str, object] = {}
    for split_name in sorted(set(split_names.tolist())):
        split_report: dict[str, object] = {}
        for lower, upper in RANK_BINS:
            bin_name = _rank_bin_name(lower, upper)
            split_report[bin_name] = _summarize_records(
                records_by_split_and_bin.get((str(split_name), bin_name), [])
            )
        split_mask = split_names == str(split_name)
        split_report["topk_mass_conservation"] = {
            "token_count": int(np.count_nonzero(split_mask)),
            "materialized_candidate_mass_mean": float(
                np.mean(materialized_mass[split_mask])
            ),
            "unmaterialized_candidate_mass_mean": float(
                np.mean(unmaterialized_mass[split_mask])
            ),
            "frozen_unknown_mass_mean": float(np.mean(unknown[split_mask])),
            "effective_null_mass_mean": float(np.mean(effective_null[split_mask])),
            "materialized_plus_effective_null_max_abs_error": float(
                np.max(np.abs(topk_mass_total[split_mask] - 1.0))
            ),
        }
        summaries[str(split_name)] = split_report

    report = {
        "stage": "candidate_spatial_rank_coverage_audit_v1",
        "protocol": {
            "spatial_predictions_target_free": True,
            "candidate_residuals_joined_externally_for_audit_only": True,
            "candidate_prior_source": "fixed_candidate_evidence_v3",
            "topk_omitted_candidate_mass_treated_as_effective_null": True,
            "rank_bins": [list(values) for values in RANK_BINS],
            "correct_threshold_px": float(correct_threshold_px),
        },
        "inputs": {
            "candidate_evidence": str(candidate_evidence_path),
            "candidate_evidence_sha256": file_sha256_short(candidate_evidence_path),
            "candidate_spatial_likelihood": [
                str(path) for path in candidate_spatial_likelihood_paths
            ],
            "candidate_spatial_likelihood_sha256": [
                file_sha256_short(path) for path in candidate_spatial_likelihood_paths
            ],
            "candidate_evidence_metadata": evidence_metadata,
            "candidate_spatial_metadata": spatial_metadata,
        },
        "splits": summaries,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument(
        "--candidate_spatial_likelihood",
        required=True,
        help="comma-separated candidate_spatial_likelihood_v7 NPZ files",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--correct_threshold_px", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_candidate_spatial_rank_coverage(
        candidate_evidence_path=Path(args.candidate_evidence),
        candidate_spatial_likelihood_paths=_paths(args.candidate_spatial_likelihood),
        output_path=Path(args.output),
        correct_threshold_px=float(args.correct_threshold_px),
    )
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "output": str(args.output),
                "splits": report["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
