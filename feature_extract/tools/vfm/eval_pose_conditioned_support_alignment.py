"""Join frozen support-alignment diagnostics with GT only after scoring.

The score artifact remains target-free.  This evaluator is intentionally the
only place that reads COLMAP query poses, and reports every raw feature family
instead of silently promoting a diagnostic score into the production selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import qvec_to_rotmat, read_colmap_images_binary
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_artifacts", required=True)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one artifact is required")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _load_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: score artifact has no metadata")
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
    if not isinstance(metadata, dict) or metadata.get("format") != POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT:
        raise ValueError(f"{path}: unsupported support-alignment score format")
    if (
        metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("selection_not_promoted") is not True
    ):
        raise ValueError(f"{path}: score artifact is not a target-free diagnostic")
    contract = metadata.get("strict_diagnostic_contract")
    required_contract = {
        "heldout_query_rows": True,
        "fixed_global_topl_support_pool": True,
        "fixed_support_images": True,
        "fixed_support_tracks": True,
        "candidate_anchor_tracks_excluded": True,
        "hypothesis_fit_tracks_excluded": True,
        "pose_local_support_reselection": False,
        "image_wide_normalized_position_denominator": True,
        "real_images_only": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "calibrated_probability": False,
    }
    if not isinstance(contract, Mapping) or any(contract.get(key) is not value for key, value in required_contract.items()):
        raise ValueError(f"{path}: score artifact violates the diagnostic protocol")
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "diagnostic_score_top1",
        "active_support_track_counts",
        "visible_support_observation_counts",
        "fixed_support_track_counts",
    }
    fields = tuple(str(value) for value in metadata.get("score_fields", ()))
    if not fields or not set(fields).issubset(arrays) or not required.issubset(arrays):
        raise ValueError(f"{path}: score schema is incomplete")
    row_count = int(metadata.get("row_count", -1))
    if row_count <= 0 or any(value.ndim == 0 or len(value) != row_count for value in arrays.values()):
        raise ValueError(f"{path}: score arrays are not row-aligned")
    keys = list(
        zip(
            arrays["query_ids"].astype(str).tolist(),
            arrays["evaluation_labels"].astype(str).tolist(),
            arrays["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: score artifact repeats query/hypothesis rows")
    if any(np.any(~np.isfinite(np.asarray(arrays[field], dtype=np.float64))) for field in fields):
        raise ValueError(f"{path}: diagnostic score has non-finite values")
    return arrays, metadata


def _compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    inputs = metadata.get("inputs")
    return {
        "version": metadata.get("version"),
        "score_fields": metadata.get("score_fields"),
        "temperatures": metadata.get("temperatures"),
        "support_layout_sha256": metadata.get("support_layout_sha256"),
        "strict_diagnostic_contract": metadata.get("strict_diagnostic_contract"),
        "inputs": inputs,
        "implementation": metadata.get("implementation"),
    }


def _merge_scores(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]], str]:
    loaded = [_load_score(path) for path in paths]
    fingerprints = [_canonical_hash(_compatibility(metadata)) for _arrays, metadata in loaded]
    if len(set(fingerprints)) != 1:
        raise ValueError(f"support-alignment score shards are incompatible: {fingerprints}")
    fields = set(loaded[0][0])
    if any(set(arrays) != fields for arrays, _metadata in loaded):
        raise ValueError("support-alignment score shards have different fields")
    merged = {
        field: np.concatenate([arrays[field] for arrays, _metadata in loaded], axis=0)
        for field in sorted(fields)
    }
    keys = list(
        zip(
            merged["query_ids"].astype(str).tolist(),
            merged["evaluation_labels"].astype(str).tolist(),
            merged["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("merged support-alignment scores overlap")
    return merged, [metadata for _arrays, metadata in loaded], fingerprints[0]


def _load_hypothesis_pose_lookup(
    paths: Sequence[Path], *, query_ids: set[str]
) -> dict[tuple[str, str, int], np.ndarray]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "poses_w2c",
    )
    lookup: dict[tuple[str, str, int], np.ndarray] = {}
    for path in paths:
        arrays, _metadata = load_inference_artifact_fields(path, fields)
        mask = np.isin(np.asarray(arrays["query_ids"]).astype(str), sorted(query_ids))
        for query_id, split, label, index, pose in zip(
            np.asarray(arrays["query_ids"]).astype(str)[mask],
            np.asarray(arrays["split_names"]).astype(str)[mask],
            np.asarray(arrays["evaluation_labels"]).astype(str)[mask],
            np.asarray(arrays["hypothesis_indices"], dtype=np.int64)[mask],
            np.asarray(arrays["poses_w2c"], dtype=np.float64)[mask],
        ):
            key = (str(query_id), str(label), int(index))
            if key in lookup:
                raise ValueError(f"hypothesis artifacts repeat {key}")
            lookup[key] = pose
    return lookup


def _gt_pose_w2c(image: object) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(np.asarray(getattr(image, "qvec"), dtype=np.float64))
    pose[:3, 3] = np.asarray(getattr(image, "tvec"), dtype=np.float64)
    return pose


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(np.percentile(array, float(percentile)))


def _pose_summary(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, object]:
    values = [
        (float(row[f"{prefix}_translation_error_m"]), float(row[f"{prefix}_rotation_error_deg"]))
        for row in rows
        if row.get(f"{prefix}_translation_error_m") is not None
        and row.get(f"{prefix}_rotation_error_deg") is not None
    ]
    return {
        "coverage": float(len(values) / max(len(rows), 1)),
        "median_translation_error_cm": _percentile([100.0 * value[0] for value in values], 50),
        "p90_translation_error_cm": _percentile([100.0 * value[0] for value in values], 90),
        "median_rotation_error_deg": _percentile([value[1] for value in values], 50),
        "p90_rotation_error_deg": _percentile([value[1] for value in values], 90),
        "recall_5cm_5deg": float(sum(value[0] <= 0.05 and value[1] <= 5.0 for value in values) / max(len(rows), 1)),
        "recall_10cm_5deg": float(sum(value[0] <= 0.10 and value[1] <= 5.0 for value in values) / max(len(rows), 1)),
        "recall_25cm_2deg": float(sum(value[0] <= 0.25 and value[1] <= 2.0 for value in values) / max(len(rows), 1)),
    }


def _best_correct_rank(
    scores: np.ndarray, translation: np.ndarray, rotation: np.ndarray, threshold_m: float
) -> int | None:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    correct = (np.asarray(translation) <= float(threshold_m)) & (np.asarray(rotation) <= 5.0)
    positions = np.flatnonzero(correct[order])
    return None if len(positions) == 0 else int(positions[0] + 1)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_paths = _paths(args.score_artifacts)
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    scores, score_metadata, compatibility_hash = _merge_scores(score_paths)
    query_ids = np.asarray(scores["query_ids"]).astype(str)
    split_names = np.asarray(scores["split_names"]).astype(str)
    labels = np.asarray(scores["evaluation_labels"]).astype(str)
    indices = np.asarray(scores["hypothesis_indices"], dtype=np.int64)
    score_fields = tuple(str(value) for value in score_metadata[0]["score_fields"])
    model_dir = Path(args.colmap_model_dir)
    images = {
        str(image.image_name): image
        for image in read_colmap_images_binary(model_dir / "images.bin").values()
    }
    pose_lookup = _load_hypothesis_pose_lookup(
        hypothesis_paths, query_ids=set(query_ids.tolist())
    )
    translation = np.empty((len(query_ids),), dtype=np.float64)
    rotation = np.empty((len(query_ids),), dtype=np.float64)
    for row, (query_id, label, index) in enumerate(zip(query_ids, labels, indices)):
        pose = pose_lookup.get((str(query_id), str(label), int(index)))
        image = images.get(str(query_id))
        if pose is None or image is None:
            raise ValueError(f"score row has no source hypothesis or GT image: {query_id}")
        error = pnp_pose_error(pose, _gt_pose_w2c(image))
        translation[row] = float(error.translation_m)
        rotation[row] = float(error.rotation_deg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, object] = {}
    all_rows: list[dict[str, object]] = []
    group_keys = sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist())))
    for score_field in score_fields:
        values = np.asarray(scores[score_field], dtype=np.float64)
        per_query: list[dict[str, object]] = []
        for split, label, query_id in group_keys:
            group = np.flatnonzero(
                (split_names == split) & (labels == label) & (query_ids == query_id)
            )
            selected = int(group[np.argmax(values[group])])
            oracle = int(min(group.tolist(), key=lambda row: (translation[row], rotation[row], row)))
            local_scores = values[group]
            row: dict[str, object] = {
                "score_field": score_field,
                "query_id": query_id,
                "split_name": split,
                "evaluation_label": label,
                "hypothesis_count": int(len(group)),
                "selected_translation_error_m": float(translation[selected]),
                "selected_rotation_error_deg": float(rotation[selected]),
                "oracle_translation_error_m": float(translation[oracle]),
                "oracle_rotation_error_deg": float(rotation[oracle]),
                "oracle_score_rank": int(1 + np.count_nonzero(local_scores > values[oracle])),
                "best_3cm_rank": _best_correct_rank(
                    local_scores, translation[group], rotation[group], 0.03
                ),
                "best_5cm_rank": _best_correct_rank(
                    local_scores, translation[group], rotation[group], 0.05
                ),
                "best_10cm_rank": _best_correct_rank(
                    local_scores, translation[group], rotation[group], 0.10
                ),
                "selected_active_support_tracks": int(scores["active_support_track_counts"][selected]),
                "selected_visible_support_observations": int(scores["visible_support_observation_counts"][selected]),
                "selected_fixed_support_tracks": int(scores["fixed_support_track_counts"][selected]),
            }
            per_query.append(row)
            all_rows.append(row)
        for split, label in sorted(set((str(row["split_name"]), str(row["evaluation_label"])) for row in per_query)):
            rows = [row for row in per_query if row["split_name"] == split and row["evaluation_label"] == label]
            key = f"{score_field}::{split}::{label}"
            all_metrics[key] = {
                "score_field": score_field,
                "split_name": split,
                "evaluation_label": label,
                "query_count": len(rows),
                "selected": _pose_summary(rows, "selected"),
                "shortlist_oracle": _pose_summary(rows, "oracle"),
                "median_oracle_score_rank": _percentile([float(row["oracle_score_rank"]) for row in rows], 50),
                "median_best_3cm_rank": _percentile(
                    [float(row["best_3cm_rank"]) for row in rows if row["best_3cm_rank"] is not None], 50
                ),
                "median_best_5cm_rank": _percentile(
                    [float(row["best_5cm_rank"]) for row in rows if row["best_5cm_rank"] is not None], 50
                ),
                "median_best_10cm_rank": _percentile(
                    [float(row["best_10cm_rank"]) for row in rows if row["best_10cm_rank"] is not None], 50
                ),
                "catastrophic_1m_count": int(sum(float(row["selected_translation_error_m"]) > 1.0 for row in rows)),
                "selected_active_track_median": _percentile(
                    [float(row["selected_active_support_tracks"]) for row in rows], 50
                ),
            }
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary = {
        "stage": "evaluate_pose_conditioned_support_alignment",
        "diagnostic_only": True,
        "score_artifacts": [str(path) for path in score_paths],
        "score_artifact_sha256": [file_sha256_short(path) for path in score_paths],
        "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
        "hypothesis_artifact_sha256": [file_sha256_short(path) for path in hypothesis_paths],
        "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        "score_compatibility_sha256": compatibility_hash,
        "targets_joined_after_diagnostic_scoring": True,
        "metrics": all_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
