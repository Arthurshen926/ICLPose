"""Audit raw intermediate candidate scores after their target-free export.

The input keeps the final descriptor-space global top-L identities fixed.  The
only new value is a raw RADIO-intermediate query-to-track cosine.  This command
is the first target boundary: it joins SfM geometry and registered point2D to
point3D identities only after the raw score artifact is frozen.  It neither
fits calibration nor scores pose hypotheses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.build_radio_intermediate_fixed_candidate_rerank import (
    ARTIFACT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.detector_landmark_proposals import (
    candidate_reprojection_residuals,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--context-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-splits", default="train,validation,test")
    parser.add_argument("--geometric-thresholds-px", default="2,5")
    parser.add_argument("--exact-identity-radius-px", type=float, default=2.0)
    return parser.parse_args(argv)


def _splits(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed) or set(parsed) - {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("audit splits must be unique train/validation/test names")
    return parsed


def _thresholds(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("geometric thresholds must be numeric") from error
    if (
        not parsed
        or len(set(parsed)) != len(parsed)
        or any(not np.isfinite(item) or item <= 0.0 for item in parsed)
    ):
        raise ValueError("geometric thresholds must be unique positive finite values")
    return tuple(sorted(parsed))


def _metadata(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError("intermediate rerank artifact lacks metadata_json")
    value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(value, dict):
        raise ValueError("intermediate rerank metadata must be an object")
    return value


def load_radio_intermediate_fixed_candidate_rerank(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the target-free artifact and reject stale/mixed descriptor spaces."""

    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "xy",
        "selected_candidate_columns",
        "candidate_track_ids",
        "candidate_bank_rows",
        "candidate_valid",
        "candidate_prior_probabilities",
        "null_prior_probabilities",
        "radio_intermediate_cosine",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        if "labels" in payload.files:
            raise ValueError("intermediate rerank artifact unexpectedly contains labels")
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"intermediate rerank artifact lacks {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required if key != "metadata_json"}
        metadata = _metadata(payload)
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_bank_rows"], dtype=np.int64)
    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    prior = np.asarray(arrays["candidate_prior_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_prior_probabilities"], dtype=np.float32).reshape(-1)
    cosine = np.asarray(arrays["radio_intermediate_cosine"], dtype=np.float32)
    columns = np.asarray(arrays["selected_candidate_columns"], dtype=np.int64)
    count = len(rows)
    if (
        metadata.get("format") != ARTIFACT_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or bool(metadata.get("image_retrieval_or_submap_used", True))
        or bool(metadata.get("render", True))
        or len(rows) == 0
        or np.unique(rows).size != len(rows)
        or query_ids.shape != splits.shape == (count,)
        or xy.shape != (count, 2)
        or tracks.ndim != 2
        or bank_rows.shape != valid.shape != prior.shape != cosine.shape != tracks.shape
        or columns.shape != tracks.shape
        or null.shape != (count,)
        or set(splits.tolist()) - {"train", "validation", "test"}
        or np.any(~np.isfinite(xy))
        or np.any(prior < 0.0)
        or np.any(null <= 0.0)
        or np.max(np.abs(prior.sum(axis=1) + null - 1.0)) > 3e-4
        or np.any(valid & ((tracks < 0) | (bank_rows < 0)))
        or np.any(~valid & ((tracks >= 0) | (bank_rows >= 0)))
        or np.any(~np.isfinite(cosine[valid]))
        or np.any(np.isfinite(cosine[~valid]))
        or not np.all(np.sort(columns, axis=1) == np.arange(tracks.shape[1]))
    ):
        raise ValueError("intermediate rerank artifact violates its frozen contract")
    return arrays, metadata


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("rank average-precision inputs are invalid")
    count = int(np.sum(positive))
    if not count:
        return None
    ranked = positive[np.argsort(-values, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / count)


def top_and_first_positive_rank(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    if (
        values.shape != positive.shape
        or values.shape != candidate_valid.shape
        or np.any(~np.isfinite(values[candidate_valid]))
    ):
        raise ValueError("rank inputs are incompatible")
    order = np.argsort(
        np.where(candidate_valid, -values, np.inf), axis=1, kind="stable"
    )
    ordered_positive = np.take_along_axis(positive, order, axis=1)
    ordered_valid = np.take_along_axis(candidate_valid, order, axis=1)
    rank = np.full((len(values),), -1, dtype=np.int64)
    for column in range(values.shape[1]):
        take = (rank < 0) & ordered_valid[:, column] & ordered_positive[:, column]
        rank[take] = column + 1
    return order[:, 0], rank


def rank_metrics(
    *, scores: np.ndarray, labels: np.ndarray, valid: np.ndarray, row_mask: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1).copy()
    if (
        values.shape != positive.shape
        or values.shape != candidate_valid.shape
        or selected.shape != values.shape[:1]
    ):
        raise ValueError("rank metrics inputs are incompatible")
    selected &= np.any(candidate_valid, axis=1)
    if not np.any(selected):
        return {"row_count": 0}
    top, rank = top_and_first_positive_rank(values, positive, candidate_valid)
    positive_present = np.any(positive & candidate_valid, axis=1)
    positive_rows = selected & positive_present
    edge_mask = selected[:, None] & candidate_valid
    return {
        "row_count": int(np.sum(selected)),
        "candidate_edge_count": int(np.sum(edge_mask)),
        "candidate_edge_positive_rate": float(np.mean(positive[edge_mask])),
        "candidate_pair_average_precision": _average_precision(
            positive[edge_mask], values[edge_mask]
        ),
        "positive_row_count": int(np.sum(positive_rows)),
        "positive_row_rate": float(np.mean(positive_present[selected])),
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


def paired_rank_audit(
    *,
    baseline_scores: np.ndarray,
    probe_scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    probe = np.asarray(probe_scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1).copy()
    if (
        baseline.shape != probe.shape
        or baseline.shape != positive.shape
        or positive.shape != candidate_valid.shape
        or selected.shape != baseline.shape[:1]
    ):
        raise ValueError("paired rank inputs are incompatible")
    _top, baseline_rank = top_and_first_positive_rank(
        baseline, positive, candidate_valid
    )
    _top, probe_rank = top_and_first_positive_rank(probe, positive, candidate_valid)
    eligible = selected & (baseline_rank > 0)
    if not np.any(eligible):
        return {
            "positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "median_rank_delta_baseline_minus_probe": None,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
        }
    return {
        "positive_row_count": int(np.sum(eligible)),
        "rank_win_count": int(np.sum(probe_rank[eligible] < baseline_rank[eligible])),
        "rank_loss_count": int(np.sum(probe_rank[eligible] > baseline_rank[eligible])),
        "rank_tie_count": int(np.sum(probe_rank[eligible] == baseline_rank[eligible])),
        "median_rank_delta_baseline_minus_probe": float(
            np.median(baseline_rank[eligible] - probe_rank[eligible])
        ),
        "top1_rescue_count": int(
            np.sum((baseline_rank[eligible] > 1) & (probe_rank[eligible] == 1))
        ),
        "top1_harm_count": int(
            np.sum((baseline_rank[eligible] == 1) & (probe_rank[eligible] > 1))
        ),
    }


def raw_identity_gate(
    baseline: Mapping[str, Any], probe: Mapping[str, Any], paired: Mapping[str, Any]
) -> dict[str, Any]:
    """A raw-score gate, deliberately insufficient for posterior promotion."""

    values = (
        baseline.get("candidate_pair_average_precision"),
        probe.get("candidate_pair_average_precision"),
        baseline.get("median_first_positive_rank"),
        probe.get("median_first_positive_rank"),
        baseline.get("p90_first_positive_rank"),
        probe.get("p90_first_positive_rank"),
        baseline.get("top1_positive_rate_given_positive"),
        probe.get("top1_positive_rate_given_positive"),
    )
    comparable = all(value is not None for value in values)
    checks = {
        "comparable_rows": bool(comparable),
        "candidate_ap_strictly_improved": bool(
            comparable and float(values[1]) > float(values[0])
        ),
        "median_rank_strictly_improved": bool(
            comparable and float(values[3]) < float(values[2])
        ),
        "p90_rank_not_worse": bool(
            comparable and float(values[5]) <= float(values[4])
        ),
        "top1_not_worse": bool(comparable and float(values[7]) >= float(values[6])),
        "paired_wins_exceed_losses": int(paired["rank_win_count"])
        > int(paired["rank_loss_count"]),
        "top1_rescues_exceed_harms": int(paired["top1_rescue_count"])
        > int(paired["top1_harm_count"]),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "policy": (
            "raw validation diagnostic only; a pass is necessary but not sufficient "
            "for train-only calibration, candidate fusion, or pose scoring"
        ),
    }


def _pose_w2c(image: object) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return pose


def _candidate_geometry_residuals(
    *,
    query_ids: np.ndarray,
    xy: np.ndarray,
    candidate_bank_rows: np.ndarray,
    bank: object,
    colmap_model_dir: Path,
) -> np.ndarray:
    model_dir = Path(colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    by_name = {str(image.image_name): image for image in images.values()}
    output = np.full(candidate_bank_rows.shape, np.inf, dtype=np.float32)
    for query_id in sorted(set(np.asarray(query_ids).astype(str).tolist())):
        rows = np.flatnonzero(np.asarray(query_ids).astype(str) == str(query_id))
        image = by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"COLMAP model is missing query image {query_id}")
        output[rows] = candidate_reprojection_residuals(
            xy[rows],
            candidate_bank_rows[rows],
            bank,
            _pose_w2c(image),
            cameras[int(image.camera_id)],
        )
    return output


def audit_radio_intermediate_fixed_candidate_rerank(
    *,
    features: Path,
    context_landmark_bank: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    audit_splits: Sequence[str],
    geometric_thresholds_px: Sequence[float],
    exact_identity_radius_px: float,
) -> dict[str, Any]:
    """Join targets only after raw intermediate candidate scores are frozen."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite intermediate rerank audit output")
    if float(exact_identity_radius_px) <= 0.0:
        raise ValueError("exact identity radius must be positive")
    arrays, metadata = load_radio_intermediate_fixed_candidate_rerank(Path(features))
    bank_path = Path(context_landmark_bank)
    bank, bank_metadata = load_landmark_index_npz(bank_path)
    if (
        str(metadata.get("context_landmark_bank_sha256", ""))
        != str(file_sha256_short(bank_path))
        or str(metadata.get("descriptor_space_id", ""))
        != str(bank_metadata.get("descriptor_space_id", ""))
    ):
        raise ValueError("intermediate rerank features and landmark bank differ")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    bank_rows = np.asarray(arrays["candidate_bank_rows"], dtype=np.int64)
    valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    safe_rows = np.maximum(bank_rows, 0)
    if np.any(bank.track_ids[safe_rows[valid]] != tracks[valid]):
        raise ValueError("intermediate rerank candidate tracks differ from landmark bank")
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    xy = np.asarray(arrays["xy"], dtype=np.float32)
    split_names = np.asarray(arrays["split_names"]).astype(str)
    model_dir = Path(colmap_model_dir)
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=query_ids,
        query_xy=xy,
        images_by_name=images_by_name,
        max_distance_px=float(exact_identity_radius_px),
    )
    exact_labels = registered_candidate_identity_labels(tracks, targets)
    residuals = _candidate_geometry_residuals(
        query_ids=query_ids,
        xy=xy,
        candidate_bank_rows=bank_rows,
        bank=bank,
        colmap_model_dir=model_dir,
    )
    if np.any(np.isnan(residuals)) or np.any(residuals[valid] < 0.0):
        raise RuntimeError("candidate reprojection target materialization is invalid")
    baseline = np.log(
        np.maximum(np.asarray(arrays["candidate_prior_probabilities"], dtype=np.float64), 1e-12)
    )
    intermediate = np.asarray(arrays["radio_intermediate_cosine"], dtype=np.float64)
    requested_splits = tuple(str(value) for value in audit_splits)
    result: dict[str, Any] = {
        "stage": "audit_radio_intermediate_fixed_candidate_rerank",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "inputs": {
            "features": str(Path(features)),
            "features_sha256": file_sha256_short(Path(features)),
            "context_landmark_bank": str(bank_path),
            "context_landmark_bank_sha256": file_sha256_short(bank_path),
            "colmap_model_dir": str(model_dir),
            "colmap_cameras_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "row_count": int(len(query_ids)),
        "query_count": int(len(set(query_ids.tolist()))),
        "geometric_thresholds_px": [float(value) for value in geometric_thresholds_px],
        "exact_identity_radius_px": float(exact_identity_radius_px),
        "registered_identity_target_coverage": summarize_registered_candidate_identity(
            exact_labels, targets
        ),
        "splits": {},
        "protocol": {
            "target_free_export_precedes_target_join": True,
            "fixed_final_top_l": True,
            "candidate_reselection": False,
            "calibration_or_fitting": False,
            "posterior_update": False,
            "pose_scoring": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    csv_rows: list[dict[str, Any]] = []
    geometry_labels = {
        float(threshold): valid & np.isfinite(residuals) & (residuals <= float(threshold))
        for threshold in geometric_thresholds_px
    }
    for split in requested_splits:
        row_mask = split_names == str(split)
        if not np.any(row_mask):
            raise ValueError(f"intermediate rerank has no rows for split {split!r}")
        split_result: dict[str, Any] = {
            "row_count": int(np.sum(row_mask)),
            "query_count": int(len(set(query_ids[row_mask].tolist()))),
            "exact_registered_identity": {},
            "geometry_by_threshold_px": {},
        }
        exact_rows = row_mask & targets.supervised
        exact_baseline = rank_metrics(
            scores=baseline,
            labels=exact_labels,
            valid=valid,
            row_mask=exact_rows,
        )
        exact_probe = rank_metrics(
            scores=intermediate,
            labels=exact_labels,
            valid=valid,
            row_mask=exact_rows,
        )
        exact_paired = paired_rank_audit(
            baseline_scores=baseline,
            probe_scores=intermediate,
            labels=exact_labels,
            valid=valid,
            row_mask=exact_rows,
        )
        split_result["exact_registered_identity"] = {
            "supervised_row_count": int(np.sum(exact_rows)),
            "baseline_final_prior": exact_baseline,
            "raw_radio_intermediate": exact_probe,
            "paired_rank": exact_paired,
            "raw_validation_gate": raw_identity_gate(
                exact_baseline, exact_probe, exact_paired
            )
            if str(split) == "validation"
            else None,
        }
        for threshold, labels in geometry_labels.items():
            split_result["geometry_by_threshold_px"][str(threshold)] = {
                "baseline_final_prior": rank_metrics(
                    scores=baseline,
                    labels=labels,
                    valid=valid,
                    row_mask=row_mask,
                ),
                "raw_radio_intermediate": rank_metrics(
                    scores=intermediate,
                    labels=labels,
                    valid=valid,
                    row_mask=row_mask,
                ),
                "paired_rank": paired_rank_audit(
                    baseline_scores=baseline,
                    probe_scores=intermediate,
                    labels=labels,
                    valid=valid,
                    row_mask=row_mask,
                ),
            }
        rank2_rows = exact_rows.copy()
        _top, baseline_rank = top_and_first_positive_rank(
            baseline, exact_labels, valid
        )
        rank2_rows &= baseline_rank >= 2
        split_result["exact_registered_rank2_to_l"] = paired_rank_audit(
            baseline_scores=baseline,
            probe_scores=intermediate,
            labels=exact_labels,
            valid=valid,
            row_mask=rank2_rows,
        )
        result["splits"][str(split)] = split_result
        csv_rows.append(
            {
                "split": str(split),
                "exact_baseline_ap": exact_baseline.get("candidate_pair_average_precision"),
                "exact_intermediate_ap": exact_probe.get("candidate_pair_average_precision"),
                "exact_baseline_top1": exact_baseline.get("top1_positive_rate_given_positive"),
                "exact_intermediate_top1": exact_probe.get("top1_positive_rate_given_positive"),
                "exact_baseline_median_rank": exact_baseline.get("median_first_positive_rank"),
                "exact_intermediate_median_rank": exact_probe.get("median_first_positive_rank"),
                "exact_wins": exact_paired.get("rank_win_count"),
                "exact_losses": exact_paired.get("rank_loss_count"),
                "exact_rescues": exact_paired.get("top1_rescue_count"),
                "exact_harms": exact_paired.get("top1_harm_count"),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "split_summary.csv").open("w", newline="") as handle:
        fieldnames = list(csv_rows[0]) if csv_rows else ["split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_radio_intermediate_fixed_candidate_rerank(
        features=Path(args.features),
        context_landmark_bank=Path(args.context_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        audit_splits=_splits(args.audit_splits),
        geometric_thresholds_px=_thresholds(args.geometric_thresholds_px),
        exact_identity_radius_px=float(args.exact_identity_radius_px),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
