"""Join frozen target-free RGB spatial predictions to pose targets externally."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.candidate_rgb_spatial_inference import (
    SPATIAL_INFERENCE_FORMAT,
)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(row: Mapping[str, object], key: str, default: float = np.nan) -> float:
    text = str(row.get(key, "")).strip()
    return float(default) if not text else float(text)


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array, initial=-np.inf))
    if not math.isfinite(maximum):
        return maximum
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"sample_count": 0, "mean": None, "median": None, "p90": None}
    return {
        "sample_count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _candidate_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    group_keys: Sequence[tuple[str, int]],
) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    finite = np.isfinite(score)
    finite_score = score[finite]
    if len(finite_score):
        standardized = (finite_score - np.median(finite_score)) / max(
            float(np.std(finite_score)), 1e-6
        )
        monotonic_probability = np.where(
            standardized >= 0.0,
            1.0 / (1.0 + np.exp(-standardized)),
            np.exp(standardized) / (1.0 + np.exp(standardized)),
        )
        binary = confidence_metrics(target[finite], monotonic_probability)
    else:
        binary = confidence_metrics(np.zeros((0,), dtype=bool), np.zeros((0,)))
    positions: dict[tuple[str, int], list[int]] = {}
    for index, key in enumerate(group_keys):
        positions.setdefault(key, []).append(index)
    top1_correct = 0
    eligible = 0
    for rows in positions.values():
        local = np.asarray(rows, dtype=np.int64)
        local = local[np.isfinite(score[local])]
        if len(local) == 0:
            continue
        eligible += 1
        top1_correct += int(target[local[int(np.argmax(score[local]))]])
    return {
        "ranking_auroc": binary["auroc"],
        "ranking_auprc": binary["auprc"],
        "positive_count": binary["positive_count"],
        "candidate_count": binary["sample_count"],
        "group_count": int(eligible),
        "top1_correct_count": int(top1_correct),
        "top1_correct_rate": (
            None if eligible == 0 else float(top1_correct / eligible)
        ),
    }


def audit_candidate_rgb_spatial_target_free(
    *,
    spatial_likelihood: Path,
    supervision_rows_csv: Path,
    output: Path,
    target_sigma_px: float | None = None,
    pose_residual_sigma_px: float = 2.0,
    pose_outlier_likelihood: float = 1e-3,
) -> dict[str, object]:
    spatial_path = Path(spatial_likelihood)
    with np.load(spatial_path, allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != SPATIAL_INFERENCE_FORMAT:
        raise ValueError("external RGB audit requires target-free v7 predictions")
    forbidden = {
        key
        for key in arrays
        if key.startswith("target_")
        or "ground_truth" in key
        or key in {"measurement_validity_supervision_weight", "dustbin_supervision_weight"}
    }
    if forbidden:
        raise ValueError(f"target-free RGB prediction exposes targets: {sorted(forbidden)}")
    if (
        bool(metadata.get("contains_ground_truth_arrays"))
        or bool(metadata.get("ground_truth_loaded_by_inference_process"))
        or not bool(metadata.get("prediction_frozen_before_target_join"))
    ):
        raise ValueError("RGB prediction metadata does not prove a target-free boundary")
    summary_path = spatial_path.parent / "summary.json"
    if not summary_path.exists():
        raise ValueError("RGB prediction summary is missing")
    inference_summary = json.loads(summary_path.read_text())
    if dict(inference_summary.get("outputs", {})).get(
        "spatial_likelihood_sha256"
    ) != file_sha256_short(spatial_path):
        raise ValueError("RGB prediction artifact is stale")

    inference_rows_path = Path(str(metadata.get("rows_csv", "")))
    inference_rows_summary_path = inference_rows_path.with_suffix(".summary.json")
    if not inference_rows_summary_path.exists():
        raise ValueError("RGB inference rows summary is missing")
    inference_rows_summary = json.loads(inference_rows_summary_path.read_text())
    inputs = dict(inference_rows_summary.get("inputs", {}))
    target_path = Path(supervision_rows_csv)
    if (
        inputs.get("supervision_rows_sha256") != file_sha256_short(target_path)
        or str(Path(str(inputs.get("supervision_rows", "")))) != str(target_path)
    ):
        raise ValueError("external audit targets differ from the sanitized-row manifest")
    with target_path.open(newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    source_indices = np.asarray(
        arrays["supervision_source_row_indices"], dtype=np.int64
    )
    if (
        len(np.unique(source_indices)) != len(source_indices)
        or np.any((source_indices < 0) | (source_indices >= len(target_rows)))
    ):
        raise ValueError("RGB prediction target join indices are invalid")
    rows = [target_rows[int(index)] for index in source_indices.tolist()]
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    source_query_rows = np.asarray(arrays["source_query_rows"], dtype=np.int64)
    candidate_ranks = np.asarray(
        arrays["candidate_measurement_ranks"], dtype=np.int64
    )
    track_ids = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    support_ids = np.asarray(arrays["support_image_ids"]).astype(str)
    for index, row in enumerate(rows):
        if (
            str(row.get("query_id", "")) != query_ids[index]
            or int(row.get("source_query_row", -1)) != source_query_rows[index]
            or int(row.get("candidate_measurement_rank", -1)) != candidate_ranks[index]
            or int(row.get("track_id", -1)) != track_ids[index]
            or str(row.get("support_image_id", "")) != support_ids[index]
        ):
            raise ValueError("RGB prediction row identity differs from external target row")

    local_log = np.asarray(arrays["local_log_probabilities"], dtype=np.float64)
    dustbin = np.asarray(arrays["dustbin_probabilities"], dtype=np.float64)
    offsets = np.asarray(arrays["offsets_xy"], dtype=np.float64)
    center_xy = np.asarray(arrays["center_xy"], dtype=np.float64)
    view_probability = np.asarray(
        arrays["support_view_probabilities"], dtype=np.float64
    )
    if (
        local_log.ndim != 2
        or offsets.shape != (local_log.shape[1], 2)
        or dustbin.shape != (len(local_log),)
        or center_xy.shape != (len(local_log), 2)
        or view_probability.shape != (len(local_log),)
        or len(rows) != len(local_log)
    ):
        raise ValueError("RGB prediction arrays have incompatible shapes")
    sigma = float(
        metadata.get("spatial_target_sigma_px")
        if target_sigma_px is None
        else target_sigma_px
    )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("RGB spatial target sigma must be positive")
    pose_sigma = float(pose_residual_sigma_px)
    outlier = float(pose_outlier_likelihood)
    if not math.isfinite(pose_sigma) or pose_sigma <= 0.0:
        raise ValueError("pose residual sigma must be positive")
    if not 0.0 < outlier <= 1.0:
        raise ValueError("pose outlier likelihood must be in (0, 1]")

    target_xy = np.asarray(
        [
            [_float(row, "target_gt_projected_x"), _float(row, "target_gt_projected_y")]
            for row in rows
        ],
        dtype=np.float64,
    )
    target_offset = target_xy - center_xy
    supervision_weight = np.asarray(
        [_float(row, "geometry_supervision_weight", 0.0) for row in rows],
        dtype=np.float64,
    )
    evaluable = (
        np.all(np.isfinite(target_offset), axis=1)
        & np.isfinite(supervision_weight)
        & (supervision_weight > 0.0)
    )
    physical = evaluable & np.asarray(
        [
            _bool(row.get("target_gt_projection_in_front"))
            and _bool(row.get("target_gt_projection_in_image"))
            for row in rows
        ],
        dtype=bool,
    )
    minimum = np.min(offsets, axis=0)
    maximum = np.max(offsets, axis=0)
    inside = (
        physical
        & np.all(target_offset >= minimum[None], axis=1)
        & np.all(target_offset <= maximum[None], axis=1)
    )
    safe_dustbin = np.clip(dustbin, 1e-8, 1.0 - 1e-8)
    joint_local_log = local_log + np.log1p(-safe_dustbin)[:, None]
    target_log_likelihood = np.full((len(rows),), np.nan, dtype=np.float64)
    mode_offset = offsets[np.argmax(local_log, axis=1)]
    mode_epe = np.linalg.norm(mode_offset - target_offset, axis=1)
    center_epe = np.linalg.norm(target_offset, axis=1)
    chunk_size = 2048
    for start in range(0, len(rows), chunk_size):
        end = min(len(rows), start + chunk_size)
        distance2 = np.sum(
            (offsets[None] - target_offset[start:end, None]) ** 2, axis=2
        )
        logits = -0.5 * distance2 / (sigma**2)
        logits -= np.max(logits, axis=1, keepdims=True)
        soft_target = np.exp(logits)
        soft_target /= np.sum(soft_target, axis=1, keepdims=True)
        local_score = np.sum(soft_target * joint_local_log[start:end], axis=1)
        target_log_likelihood[start:end] = np.where(
            inside[start:end], local_score, np.log(safe_dustbin[start:end])
        )
    target_log_likelihood[~evaluable] = np.nan

    center_kernel = np.exp(-0.5 * np.square(center_epe / pose_sigma))
    base_pose_likelihood = outlier + (1.0 - outlier) * center_kernel
    backend_spatial_view_likelihood = np.full(
        (len(rows),), np.nan, dtype=np.float64
    )
    for start in range(0, len(rows), chunk_size):
        end = min(len(rows), start + chunk_size)
        distance = np.linalg.norm(
            offsets[None] - target_offset[start:end, None], axis=2
        )
        geometric = np.exp(-0.5 * np.square(distance / pose_sigma))
        mode_evidence = np.sum(
            np.exp(local_log[start:end]) * geometric, axis=1
        )
        mode_likelihood = outlier + (1.0 - outlier) * mode_evidence
        reliability = np.clip(1.0 - dustbin[start:end], 0.0, 1.0)
        backend_spatial_view_likelihood[start:end] = (
            (1.0 - reliability) * base_pose_likelihood[start:end]
            + reliability * mode_likelihood
        )
    backend_spatial_view_likelihood[~evaluable] = np.nan

    audited = evaluable
    non_dustbin_metrics = confidence_metrics(
        inside[audited], (1.0 - dustbin[audited])
    )
    accepted = audited & (dustbin < 0.5)
    mode_improvement = center_epe - mode_epe
    spatial_metrics = {
        "evaluable_row_count": int(np.count_nonzero(audited)),
        "inside_local_support_count": int(np.count_nonzero(inside)),
        "normalized_density_nll": _distribution(-target_log_likelihood[audited]),
        "non_dustbin_calibration": non_dustbin_metrics,
        "inside_support_mode_epe_px": _distribution(mode_epe[inside]),
        "inside_support_center_epe_px": _distribution(center_epe[inside]),
        "inside_support_mode_improvement_px": _distribution(mode_improvement[inside]),
        "inside_support_mode_improves_rate": (
            None
            if not np.any(inside)
            else float(np.mean(mode_improvement[inside] > 0.0))
        ),
        "accepted_at_dustbin_lt_0_5": {
            "row_count": int(np.count_nonzero(accepted)),
            "inside_precision": (
                None
                if not np.any(accepted)
                else float(np.mean(inside[accepted]))
            ),
            "inside_recall": (
                None
                if not np.any(inside)
                else float(np.count_nonzero(accepted & inside) / np.count_nonzero(inside))
            ),
        },
    }

    candidate_records: dict[tuple[str, int, int], list[int]] = {}
    for row_index in range(len(rows)):
        key = (
            str(query_ids[row_index]),
            int(source_query_rows[row_index]),
            int(candidate_ranks[row_index]),
        )
        candidate_records.setdefault(key, []).append(row_index)
    candidate_keys = sorted(candidate_records)
    candidate_likelihood = []
    candidate_base_likelihood = []
    candidate_prior = []
    label_2px = []
    label_5px = []
    group_keys: list[tuple[str, int]] = []
    for key in candidate_keys:
        indices = np.asarray(candidate_records[key], dtype=np.int64)
        valid = indices[np.isfinite(target_log_likelihood[indices])]
        if len(valid) == 0:
            candidate_likelihood.append(float("nan"))
            candidate_base_likelihood.append(float("nan"))
        else:
            probabilities = np.clip(view_probability[valid], 0.0, None)
            available_mass = float(np.sum(probabilities))
            if available_mass > 1.0 + 2e-5:
                raise ValueError("RGB support-view probability mass exceeds one")
            base_value = float(base_pose_likelihood[int(valid[0])])
            if not np.allclose(
                base_pose_likelihood[valid], base_value, rtol=0.0, atol=1e-5
            ):
                raise ValueError("support views disagree on base pose likelihood")
            spatial_value = max(1.0 - available_mass, 0.0) * base_value
            spatial_value += float(
                np.sum(probabilities * backend_spatial_view_likelihood[valid])
            )
            candidate_likelihood.append(max(spatial_value, 1e-12))
            candidate_base_likelihood.append(max(base_value, 1e-12))
        first = rows[int(indices[0])]
        candidate_prior.append(
            max(_float(first, "candidate_assignment_probability", 0.0), 1e-30)
        )
        label_2px.append(_bool(first.get("target_geometry_correct_2px")))
        label_5px.append(_bool(first.get("target_geometry_correct_5px")))
        group_keys.append((key[0], key[1]))
    candidate_likelihood_array = np.asarray(candidate_likelihood, dtype=np.float64)
    candidate_base_array = np.asarray(
        candidate_base_likelihood, dtype=np.float64
    )
    candidate_log_prior = np.log(np.asarray(candidate_prior, dtype=np.float64))
    labels_by_threshold = {
        "2px": np.asarray(label_2px, dtype=bool),
        "5px": np.asarray(label_5px, dtype=bool),
    }
    candidate_identity: dict[str, object] = {}
    for label_name, labels in labels_by_threshold.items():
        result: dict[str, object] = {
            "coarse_prior": _candidate_metrics(
                candidate_log_prior, labels, group_keys
            ),
            "base_coordinate_true_pose_likelihood": _candidate_metrics(
                candidate_log_prior + np.log(candidate_base_array),
                labels,
                group_keys,
            ),
            "rgb_spatial_true_pose_likelihood": _candidate_metrics(
                candidate_log_prior + np.log(candidate_likelihood_array),
                labels,
                group_keys,
            ),
            "fusion_weight_sweep": {},
        }
        for weight in (0.0625, 0.125, 0.25, 0.5, 1.0):
            combined_likelihood = (
                (1.0 - float(weight)) * candidate_base_array
                + float(weight) * candidate_likelihood_array
            )
            fused = candidate_log_prior + np.log(
                np.maximum(combined_likelihood, 1e-12)
            )
            result["fusion_weight_sweep"][f"{weight:g}"] = _candidate_metrics(
                fused, labels, group_keys
            )
        candidate_identity[label_name] = result

    result = {
        "stage": "candidate_rgb_spatial_external_target_audit",
        "protocol": {
            "prediction_artifact_frozen_before_target_join": True,
            "inference_process_loaded_ground_truth": False,
            "target_join_key": "supervision_source_row_index",
            "candidate_identity_likelihood_evaluated_at_true_pose_TARGET_ONLY": True,
            "reused_test_is_diagnostic_not_untouched": True,
        },
        "config": {
            "spatial_target_sigma_px": sigma,
            "pose_residual_sigma_px": pose_sigma,
            "pose_outlier_likelihood": outlier,
        },
        "inputs": {
            "spatial_likelihood": str(spatial_path),
            "spatial_likelihood_sha256": file_sha256_short(spatial_path),
            "supervision_rows_csv": str(target_path),
            "supervision_rows_csv_sha256": file_sha256_short(target_path),
        },
        "spatial_density": spatial_metrics,
        "candidate_identity_at_true_pose_TARGET_ONLY": candidate_identity,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial_likelihood", required=True)
    parser.add_argument("--supervision_rows_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_sigma_px", type=float, default=None)
    parser.add_argument("--pose_residual_sigma_px", type=float, default=2.0)
    parser.add_argument("--pose_outlier_likelihood", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_candidate_rgb_spatial_target_free(
        spatial_likelihood=Path(args.spatial_likelihood),
        supervision_rows_csv=Path(args.supervision_rows_csv),
        output=Path(args.output),
        target_sigma_px=args.target_sigma_px,
        pose_residual_sigma_px=float(args.pose_residual_sigma_px),
        pose_outlier_likelihood=float(args.pose_outlier_likelihood),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
