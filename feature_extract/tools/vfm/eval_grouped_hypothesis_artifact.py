"""Join inference-only grouped pose hypotheses with GT in a separate process.

The input artifact is deliberately target-free. This evaluator validates that
contract before loading COLMAP ground truth, then writes targets and aggregate
metrics to separate files. It must never be imported by pose generation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import qvec_to_rotmat, read_colmap_images_binary
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


ARTIFACT_FORMAT = "grouped_pose_hypotheses_inference_only_v1"
TARGET_FORMAT = "grouped_pose_hypothesis_targets_v1"
REQUIRED_FIELDS = {
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "generation_profiles",
    "selection_modes",
    "shortlisted_for_verification",
    "chosen_for_optional_pose",
    "poses_w2c",
    "preliminary_log_likelihood_means",
    "shortlist_log_likelihood_means",
    "verification_log_likelihood_means",
    "metadata_json",
}
TARGET_FIELD_MARKERS = (
    "ground_truth",
    "gt_pose",
    "translation_error",
    "rotation_error",
    "target",
    "correct_",
)
TRANSLATION_THRESHOLDS_CM = (3, 5, 10, 25)
ROTATION_THRESHOLD_DEG = 5.0
RELATION_SCORE_FIELD = "verification_relation_log_likelihood_ratio_means"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis_artifacts",
        required=True,
        help="comma-separated inference-only grouped hypothesis NPZ files",
    )
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metadata_from_array(value: np.ndarray) -> dict[str, object]:
    if np.asarray(value).shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    payload = json.loads(str(np.asarray(value).item()))
    if not isinstance(payload, dict):
        raise ValueError("metadata_json must decode to an object")
    return payload


def validate_inference_artifact(
    arrays: Mapping[str, np.ndarray], *, source: str = "<memory>"
) -> dict[str, object]:
    """Validate the no-target schema and return decoded metadata."""

    missing = sorted(REQUIRED_FIELDS.difference(arrays))
    if missing:
        raise ValueError(f"{source}: missing required fields: {missing}")
    forbidden = sorted(
        key
        for key in arrays
        if key != "metadata_json"
        and any(marker in key.lower() for marker in TARGET_FIELD_MARKERS)
    )
    if forbidden:
        raise ValueError(f"{source}: inference artifact contains target fields: {forbidden}")

    metadata = _metadata_from_array(np.asarray(arrays["metadata_json"]))
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{source}: unsupported artifact format")
    if metadata.get("contains_target_fields") is not False:
        raise ValueError(f"{source}: metadata does not assert target-free content")
    if metadata.get("pose_or_ground_truth_used_for_generation") is not False:
        raise ValueError(f"{source}: generation is not declared pose/GT-free")

    row_count = int(np.asarray(arrays["poses_w2c"]).shape[0])
    if int(metadata.get("row_count", -1)) != row_count:
        raise ValueError(f"{source}: metadata row_count mismatch")
    for key, value in arrays.items():
        array = np.asarray(value)
        if key == "metadata_json":
            continue
        if array.ndim == 0 or int(array.shape[0]) != row_count:
            raise ValueError(f"{source}: field {key!r} is not row-aligned")
    poses = np.asarray(arrays["poses_w2c"], dtype=np.float64)
    if poses.shape != (row_count, 4, 4) or not np.all(np.isfinite(poses)):
        raise ValueError(f"{source}: poses_w2c must be finite [N,4,4]")
    if not np.allclose(poses[:, 3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"{source}: malformed homogeneous poses")

    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    evaluation_labels = np.asarray(arrays["evaluation_labels"]).astype(str)
    hypothesis_indices = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)
    keys = list(zip(query_ids.tolist(), evaluation_labels.tolist(), hypothesis_indices.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source}: duplicate query/evaluation/hypothesis keys")
    return metadata


def load_inference_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    metadata = validate_inference_artifact(arrays, source=str(path))
    return arrays, metadata


def _compatibility_payload(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "inputs": metadata.get("inputs"),
        "grouped_config": metadata.get("grouped_config"),
    }


def _gt_pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(np.asarray(image.qvec, dtype=np.float64))
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    return pose


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.percentile(finite, percentile))


def _best_index(indices: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> int | None:
    if len(indices) == 0:
        return None
    return int(
        min(
            indices.tolist(),
            key=lambda idx: (float(translation[idx]), float(rotation[idx]), int(idx)),
        )
    )


def _score_rank(indices: np.ndarray, scores: np.ndarray, target_index: int | None) -> int | None:
    if target_index is None or target_index not in set(indices.tolist()):
        return None
    valid = indices[np.isfinite(scores[indices])]
    if target_index not in set(valid.tolist()):
        return None
    target = float(scores[target_index])
    return 1 + int(np.count_nonzero(scores[valid] > target))


def _best_correct_score_rank(
    indices: np.ndarray,
    scores: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    threshold_cm: float = 10.0,
) -> int | None:
    valid = indices[np.isfinite(scores[indices])]
    correct = valid[
        (translation[valid] <= float(threshold_cm) / 100.0)
        & (rotation[valid] <= ROTATION_THRESHOLD_DEG)
    ]
    if len(correct) == 0:
        return None
    ranks = [_score_rank(valid, scores, int(idx)) for idx in correct]
    return min(int(rank) for rank in ranks if rank is not None)


def _top_score_index(indices: np.ndarray, scores: np.ndarray) -> int | None:
    valid = indices[np.isfinite(scores[indices])]
    if len(valid) == 0:
        return None
    return int(valid[int(np.argmax(scores[valid]))])


def _pose_fields(prefix: str, index: int | None, translation: np.ndarray, rotation: np.ndarray) -> dict[str, object]:
    if index is None:
        return {
            f"{prefix}_translation_error_m": None,
            f"{prefix}_rotation_error_deg": None,
        }
    return {
        f"{prefix}_translation_error_m": float(translation[index]),
        f"{prefix}_rotation_error_deg": float(rotation[index]),
    }


def build_per_query_rows(
    arrays: Mapping[str, np.ndarray],
    translation: np.ndarray,
    rotation: np.ndarray,
) -> list[dict[str, object]]:
    query_ids = np.asarray(arrays["query_ids"]).astype(str)
    split_names = np.asarray(arrays["split_names"]).astype(str)
    labels = np.asarray(arrays["evaluation_labels"]).astype(str)
    shortlisted = np.asarray(arrays["shortlisted_for_verification"], dtype=bool)
    chosen = np.asarray(arrays["chosen_for_optional_pose"], dtype=bool)
    preliminary = np.asarray(arrays["preliminary_log_likelihood_means"], dtype=np.float64)
    shortlist_score = np.asarray(arrays["shortlist_log_likelihood_means"], dtype=np.float64)
    verification = np.asarray(arrays["verification_log_likelihood_means"], dtype=np.float64)
    score_specs: list[tuple[str, np.ndarray]] = [
        ("preliminary", preliminary),
        ("shortlist", shortlist_score),
        ("verification", verification),
    ]
    if RELATION_SCORE_FIELD in arrays:
        relation = np.asarray(arrays[RELATION_SCORE_FIELD], dtype=np.float64)
        score_specs.extend(
            [
                ("relation_DIAGNOSTIC_ONLY", relation),
                (
                    "verification_plus_relation_DIAGNOSTIC_ONLY",
                    verification + relation,
                ),
            ]
        )

    group_keys = sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist())))
    rows: list[dict[str, object]] = []
    for split_name, label, query_id in group_keys:
        all_indices = np.flatnonzero(
            (split_names == split_name) & (labels == label) & (query_ids == query_id)
        )
        shortlist_indices = all_indices[shortlisted[all_indices]]
        chosen_indices = all_indices[chosen[all_indices]]
        if len(chosen_indices) > 1:
            raise ValueError(f"{query_id}/{label}: multiple chosen hypotheses")
        full_oracle = _best_index(all_indices, translation, rotation)
        shortlist_oracle = _best_index(shortlist_indices, translation, rotation)
        chosen_index = None if len(chosen_indices) == 0 else int(chosen_indices[0])
        row: dict[str, object] = {
            "query_id": query_id,
            "split_name": split_name,
            "evaluation_label": label,
            "hypothesis_count": int(len(all_indices)),
            "shortlisted_count": int(len(shortlist_indices)),
            "chosen_count": int(len(chosen_indices)),
        }
        row.update(_pose_fields("full_oracle", full_oracle, translation, rotation))
        row.update(_pose_fields("shortlist_oracle", shortlist_oracle, translation, rotation))
        row.update(_pose_fields("chosen", chosen_index, translation, rotation))
        full_error = row["full_oracle_translation_error_m"]
        shortlist_error = row["shortlist_oracle_translation_error_m"]
        chosen_error = row["chosen_translation_error_m"]
        row["shortlist_regret_m"] = (
            None if full_error is None or shortlist_error is None else float(shortlist_error) - float(full_error)
        )
        row["selection_regret_m"] = (
            None if full_error is None or chosen_error is None else float(chosen_error) - float(full_error)
        )
        for name, score in score_specs:
            scope = (
                all_indices
                if name in {"preliminary", "shortlist"}
                else shortlist_indices
            )
            top_index = _top_score_index(scope, score)
            row[f"{name}_full_oracle_rank"] = _score_rank(scope, score, full_oracle)
            for threshold_cm in (5.0, 10.0):
                row[f"{name}_best_{threshold_cm:g}cm_rank"] = (
                    _best_correct_score_rank(
                        scope,
                        score,
                        translation,
                        rotation,
                        threshold_cm=threshold_cm,
                    )
                )
            row.update(_pose_fields(f"{name}_top1", top_index, translation, rotation))
        rows.append(row)
    return rows


def _pose_summary(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, object]:
    translations = [row.get(f"{prefix}_translation_error_m") for row in rows]
    rotations = [row.get(f"{prefix}_rotation_error_deg") for row in rows]
    paired = [
        (float(t), float(r))
        for t, r in zip(translations, rotations)
        if t is not None and r is not None and np.isfinite(float(t)) and np.isfinite(float(r))
    ]
    output: dict[str, object] = {
        "coverage": float(len(paired) / max(len(rows), 1)),
        "median_translation_error_cm": _percentile([100.0 * item[0] for item in paired], 50.0),
        "p90_translation_error_cm": _percentile([100.0 * item[0] for item in paired], 90.0),
        "median_rotation_error_deg": _percentile([item[1] for item in paired], 50.0),
        "p90_rotation_error_deg": _percentile([item[1] for item in paired], 90.0),
    }
    for threshold_cm in TRANSLATION_THRESHOLDS_CM:
        output[f"recall_{threshold_cm}cm"] = float(
            sum(item[0] <= threshold_cm / 100.0 for item in paired) / max(len(rows), 1)
        )
        output[f"recall_{threshold_cm}cm_5deg"] = float(
            sum(
                item[0] <= threshold_cm / 100.0 and item[1] <= ROTATION_THRESHOLD_DEG
                for item in paired
            )
            / max(len(rows), 1)
        )
    return output


def summarize_per_query_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["split_name"]), str(row["evaluation_label"])), []).append(row)
    output: dict[str, object] = {}
    for (split_name, label), group in sorted(grouped.items()):
        key = f"{split_name}::{label}"
        score_names = ["preliminary", "shortlist", "verification"]
        if group and "relation_DIAGNOSTIC_ONLY_top1_translation_error_m" in group[0]:
            score_names.extend(
                [
                    "relation_DIAGNOSTIC_ONLY",
                    "verification_plus_relation_DIAGNOSTIC_ONLY",
                ]
            )
        entry: dict[str, object] = {
            "split_name": split_name,
            "evaluation_label": label,
            "query_count": int(len(group)),
            "median_hypothesis_count": _percentile(
                [float(row["hypothesis_count"]) for row in group], 50.0
            ),
            "median_shortlisted_count": _percentile(
                [float(row["shortlisted_count"]) for row in group], 50.0
            ),
            "full_oracle": _pose_summary(group, "full_oracle"),
            "shortlist_oracle": _pose_summary(group, "shortlist_oracle"),
            "chosen": _pose_summary(group, "chosen"),
            "preliminary_top1": _pose_summary(group, "preliminary_top1"),
            "shortlist_top1": _pose_summary(group, "shortlist_top1"),
            "verification_top1": _pose_summary(group, "verification_top1"),
            "median_shortlist_regret_cm": _percentile(
                [100.0 * float(row["shortlist_regret_m"]) for row in group if row["shortlist_regret_m"] is not None],
                50.0,
            ),
            "median_selection_regret_cm": _percentile(
                [100.0 * float(row["selection_regret_m"]) for row in group if row["selection_regret_m"] is not None],
                50.0,
            ),
        }
        for score_name in score_names:
            entry[f"{score_name}_top1"] = _pose_summary(
                group, f"{score_name}_top1"
            )
            for threshold_cm in (5, 10):
                ranks = [
                    float(row[f"{score_name}_best_{threshold_cm}cm_rank"])
                    for row in group
                    if row[f"{score_name}_best_{threshold_cm}cm_rank"] is not None
                ]
                entry[f"{score_name}_{threshold_cm}cm_rank_coverage"] = float(
                    len(ranks) / max(len(group), 1)
                )
                entry[
                    f"{score_name}_median_best_{threshold_cm}cm_rank"
                ] = _percentile(ranks, 50.0)
        output[key] = entry
    return output


def _write_query_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(value.strip()) for value in args.hypothesis_artifacts.split(",") if value.strip()]
    if not paths:
        raise ValueError("at least one hypothesis artifact is required")
    loaded = [load_inference_artifact(path) for path in paths]
    compatibility = [_compatibility_payload(metadata) for _arrays, metadata in loaded]
    fingerprints = [_canonical_sha256(item) for item in compatibility]
    if len(set(fingerprints)) != 1:
        raise ValueError(f"incompatible hypothesis sources: {fingerprints}")

    array_keys = set(loaded[0][0]) - {"metadata_json"}
    if any((set(arrays) - {"metadata_json"}) != array_keys for arrays, _metadata in loaded):
        raise ValueError("hypothesis shards have different schemas")
    merged = {
        key: np.concatenate([arrays[key] for arrays, _metadata in loaded], axis=0)
        for key in sorted(array_keys)
    }
    source_artifact_indices = np.concatenate(
        [np.full((len(arrays["query_ids"]),), index, dtype=np.int64) for index, (arrays, _metadata) in enumerate(loaded)]
    )
    source_row_indices = np.concatenate(
        [np.arange(len(arrays["query_ids"]), dtype=np.int64) for arrays, _metadata in loaded]
    )
    merged_metadata = dict(loaded[0][1])
    merged_metadata["row_count"] = int(len(merged["query_ids"]))
    merged_with_metadata = dict(merged)
    merged_with_metadata["metadata_json"] = np.asarray(
        json.dumps(merged_metadata, sort_keys=True)
    )
    validate_inference_artifact(merged_with_metadata, source="merged artifacts")

    model_dir = Path(args.colmap_model_dir)
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_ids = np.asarray(merged["query_ids"]).astype(str)
    missing_queries = sorted(set(query_ids.tolist()).difference(images_by_name))
    if missing_queries:
        raise ValueError(f"queries absent from COLMAP model: {missing_queries[:10]}")

    poses = np.asarray(merged["poses_w2c"], dtype=np.float64)
    translation = np.empty((len(poses),), dtype=np.float64)
    rotation = np.empty((len(poses),), dtype=np.float64)
    for index, (query_id, pose) in enumerate(zip(query_ids, poses)):
        error = pnp_pose_error(pose, _gt_pose_w2c(images_by_name[str(query_id)]))
        translation[index] = float(error.translation_m)
        rotation[index] = float(error.rotation_deg)

    per_query_rows = build_per_query_rows(merged, translation, rotation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "grouped_hypothesis_targets_v1.npz"
    target_metadata = {
        "format": TARGET_FORMAT,
        "contains_target_fields": True,
        "targets_joined_after_inference": True,
        "inference_artifacts": [str(path) for path in paths],
        "inference_artifact_sha256": [file_sha256_short(path) for path in paths],
        "inference_compatibility_sha256": fingerprints[0],
        "colmap_images_bin": str(model_dir / "images.bin"),
        "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        "row_count": int(len(poses)),
        "rotation_success_threshold_deg": ROTATION_THRESHOLD_DEG,
    }
    target_arrays: dict[str, np.ndarray] = {
        "query_ids": query_ids,
        "split_names": np.asarray(merged["split_names"]).astype(str),
        "evaluation_labels": np.asarray(merged["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(merged["hypothesis_indices"], dtype=np.int64),
        "source_artifact_indices": source_artifact_indices,
        "source_row_indices": source_row_indices,
        "translation_errors_m": translation,
        "rotation_errors_deg": rotation,
        "metadata_json": np.asarray(json.dumps(target_metadata, sort_keys=True)),
    }
    for threshold_cm in TRANSLATION_THRESHOLDS_CM:
        target_arrays[f"correct_{threshold_cm}cm_5deg"] = (
            (translation <= threshold_cm / 100.0) & (rotation <= ROTATION_THRESHOLD_DEG)
        )
    np.savez_compressed(target_path, **target_arrays)
    _write_query_csv(output_dir / "grouped_hypothesis_per_query.csv", per_query_rows)
    summary = {
        "stage": "post_inference_grouped_hypothesis_gt_join",
        "protocol": {
            "ground_truth_loaded_only_after_inference_artifact_validation": True,
            "targets_stored_separately": True,
            "translation_thresholds_cm": list(TRANSLATION_THRESHOLDS_CM),
            "rotation_threshold_deg": ROTATION_THRESHOLD_DEG,
            "pairwise_relation_score_is_diagnostic_only": bool(
                RELATION_SCORE_FIELD in merged
            ),
            "relation_combination_rule": (
                None
                if RELATION_SCORE_FIELD not in merged
                else "unary_log_likelihood_mean_plus_pair_relation_log_ratio_mean_unit_weight"
            ),
        },
        "inputs": target_metadata,
        "metrics": summarize_per_query_rows(per_query_rows),
        "outputs": {
            "targets": str(target_path),
            "per_query": str(output_dir / "grouped_hypothesis_per_query.csv"),
        },
    }
    summary_path = output_dir / "grouped_hypothesis_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
