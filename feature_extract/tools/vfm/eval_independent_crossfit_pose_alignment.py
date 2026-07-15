"""Join target-free independent cross-fit pose selections with GT for reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.run_independent_crossfit_pose_alignment import (
    ARTIFACT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import qvec_to_rotmat, read_colmap_images_binary
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


TARGET_ARTIFACT_FORMAT = "independent_crossfit_pose_alignment_targets_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment_artifacts", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("alignment metadata has no input manifest")
    return {
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "hypothesis_compatibility_sha256": metadata.get(
            "hypothesis_compatibility_sha256"
        ),
        "score_config": metadata.get("score_config"),
        "descriptor_evidence": metadata.get("descriptor_evidence"),
        "refinement_config": metadata.get("refinement_config"),
        "score_thresholds": metadata.get("score_thresholds"),
        "observability_thresholds": metadata.get("observability_thresholds"),
        "hypothesis_scope": metadata.get("hypothesis_scope"),
        "crossfit": metadata.get("crossfit"),
        "immutable_source_pose_artifact_sha256": inputs.get(
            "immutable_source_pose_artifact_sha256"
        ),
        "immutable_source_pose_evaluation_label": inputs.get(
            "immutable_source_pose_evaluation_label"
        ),
        "detector_query_cache_sha256": inputs.get("detector_query_cache_sha256"),
        "proposals_sha256": inputs.get("proposals_sha256"),
        "candidate_artifact_sha256": inputs.get("candidate_artifact_sha256"),
        "fixed_candidate_prior_overlay_sha256": inputs.get(
            "fixed_candidate_prior_overlay_sha256"
        ),
        "fixed_candidate_prior_overlay_metadata_sha256": inputs.get(
            "fixed_candidate_prior_overlay_metadata_sha256"
        ),
        "projected_landmark_bank_sha256": inputs.get(
            "projected_landmark_bank_sha256"
        ),
        "independent_verification_landmark_bank_sha256": inputs.get(
            "independent_verification_landmark_bank_sha256"
        ),
        "prototype_view_geometry_sha256": inputs.get(
            "prototype_view_geometry_sha256"
        ),
        "support_geometry_index_sha256": inputs.get(
            "support_geometry_index_sha256"
        ),
        "view_geometry_sha256": inputs.get("view_geometry_sha256"),
        "maplet_support_index_sha256": inputs.get("maplet_support_index_sha256"),
        "colmap_images_bin_sha256": inputs.get("colmap_images_bin_sha256"),
    }


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: alignment artifact has no metadata")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{path}: unsupported alignment artifact")
    if metadata.get("contains_target_fields") is not False:
        raise ValueError(f"{path}: alignment artifact is not target-free")
    if metadata.get("pose_or_ground_truth_used_for_scoring") is not False:
        raise ValueError(f"{path}: alignment scorer accessed target pose")
    row_count = int(metadata.get("row_count", -1))
    for key, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != row_count:
            raise ValueError(f"{path}: {key} is not row-aligned")
    keys = list(
        zip(
            arrays["query_ids"].astype(str).tolist(),
            arrays["evaluation_labels"].astype(str).tolist(),
            arrays["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate hypothesis rows")
    return arrays, metadata


def _merge(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], str]:
    loaded = [_load(path) for path in paths]
    fingerprints = [
        _canonical_hash(_compatibility(metadata)) for _arrays, metadata in loaded
    ]
    if len(set(fingerprints)) != 1:
        raise ValueError(f"alignment shards are incompatible: {fingerprints}")
    fields = set(loaded[0][0])
    if any(set(arrays) != fields for arrays, _metadata in loaded):
        raise ValueError("alignment shards have different schemas")
    merged = {
        key: np.concatenate([arrays[key] for arrays, _metadata in loaded], axis=0)
        for key in sorted(fields)
    }
    return merged, [metadata for _arrays, metadata in loaded], fingerprints[0]


def _gt_pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    return pose


def _pose_error(pose: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = pnp_pose_error(pose, target)
    return float(error.translation_m), float(error.rotation_deg)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(np.percentile(array, percentile))


def _pose_summary(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, object]:
    values = [
        (
            float(row[f"{prefix}_translation_error_m"]),
            float(row[f"{prefix}_rotation_error_deg"]),
        )
        for row in rows
    ]
    return {
        "median_translation_error_cm": _percentile(
            [100.0 * value[0] for value in values], 50.0
        ),
        "p90_translation_error_cm": _percentile(
            [100.0 * value[0] for value in values], 90.0
        ),
        "median_rotation_error_deg": _percentile(
            [value[1] for value in values], 50.0
        ),
        "p90_rotation_error_deg": _percentile(
            [value[1] for value in values], 90.0
        ),
        "recall_3cm_5deg": float(
            np.mean([value[0] <= 0.03 and value[1] <= 5.0 for value in values])
        ),
        "recall_5cm_5deg": float(
            np.mean([value[0] <= 0.05 and value[1] <= 5.0 for value in values])
        ),
        "recall_10cm_5deg": float(
            np.mean([value[0] <= 0.10 and value[1] <= 5.0 for value in values])
        ),
        "recall_25cm_2deg": float(
            np.mean([value[0] <= 0.25 and value[1] <= 2.0 for value in values])
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = tuple(
        Path(value.strip())
        for value in str(args.alignment_artifacts).split(",")
        if value.strip()
    )
    if not paths:
        raise ValueError("at least one alignment artifact is required")
    arrays, metadata, compatibility_hash = _merge(paths)
    model_dir = Path(args.colmap_model_dir)
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    query_ids = arrays["query_ids"].astype(str)
    split_names = arrays["split_names"].astype(str)
    labels = arrays["evaluation_labels"].astype(str)
    source_chosen = arrays["source_chosen"].astype(bool)
    optional_top1 = arrays["optional_rank_top1"].astype(bool)
    rank_shortlisted = np.asarray(
        arrays.get("rank_shortlisted", np.ones_like(optional_top1)), dtype=bool
    )
    source_poses = np.asarray(arrays["source_poses_w2c"], dtype=np.float64)
    candidate_poses = np.asarray(arrays["candidate_poses_w2c"], dtype=np.float64)
    returned_poses = np.asarray(arrays["returned_poses_w2c"], dtype=np.float64)
    source_translation = np.zeros((len(query_ids),), dtype=np.float64)
    source_rotation = np.zeros((len(query_ids),), dtype=np.float64)
    candidate_translation = np.zeros((len(query_ids),), dtype=np.float64)
    candidate_rotation = np.zeros((len(query_ids),), dtype=np.float64)
    for row, query_id in enumerate(query_ids):
        image = images_by_name.get(str(query_id))
        if image is None:
            raise ValueError(f"query absent from target model: {query_id}")
        target = _gt_pose_w2c(image)
        source_translation[row], source_rotation[row] = _pose_error(
            source_poses[row], target
        )
        candidate_translation[row], candidate_rotation[row] = _pose_error(
            candidate_poses[row], target
        )

    group_keys = sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist())))
    per_query: list[dict[str, object]] = []
    for split_name, label, query_id in group_keys:
        group = np.flatnonzero(
            (split_names == split_name) & (labels == label) & (query_ids == query_id)
        )
        source_rows = group[source_chosen[group]]
        optional_rows = group[optional_top1[group]]
        if len(source_rows) != 1 or len(optional_rows) != 1:
            raise ValueError(f"{query_id}: source/optional selection is not unique")
        source_row = int(source_rows[0])
        optional_row = int(optional_rows[0])
        if not all(
            np.array_equal(returned_poses[int(row)], returned_poses[int(group[0])])
            for row in group
        ):
            raise ValueError(f"{query_id}: returned pose differs across hypothesis rows")
        if len(np.unique(arrays["promoted"][group])) != 1:
            raise ValueError(f"{query_id}: promotion decision differs across rows")
        target = _gt_pose_w2c(images_by_name[query_id])
        returned_translation, returned_rotation = _pose_error(
            returned_poses[int(group[0])], target
        )
        source_oracle_row = min(
            group.tolist(),
            key=lambda row: (source_translation[row], source_rotation[row], row),
        )
        candidate_oracle_row = min(
            group.tolist(),
            key=lambda row: (candidate_translation[row], candidate_rotation[row], row),
        )
        shortlist_rows = group[rank_shortlisted[group]]
        if len(shortlist_rows) == 0:
            raise ValueError(f"{query_id}: rank shortlist is empty")
        shortlist_oracle_row = min(
            shortlist_rows.tolist(),
            key=lambda row: (candidate_translation[row], candidate_rotation[row], row),
        )
        row = {
            "query_id": query_id,
            "split_name": split_name,
            "evaluation_label": label,
            "hypothesis_count": int(len(group)),
            "promoted": bool(arrays["promoted"][source_row]),
            "promotion_failures": str(arrays["promotion_failures"][source_row]),
            "source_translation_error_m": float(source_translation[source_row]),
            "source_rotation_error_deg": float(source_rotation[source_row]),
            "optional_translation_error_m": float(candidate_translation[optional_row]),
            "optional_rotation_error_deg": float(candidate_rotation[optional_row]),
            "returned_translation_error_m": returned_translation,
            "returned_rotation_error_deg": returned_rotation,
            "source_oracle_translation_error_m": float(
                source_translation[source_oracle_row]
            ),
            "source_oracle_rotation_error_deg": float(source_rotation[source_oracle_row]),
            "candidate_oracle_translation_error_m": float(
                candidate_translation[candidate_oracle_row]
            ),
            "candidate_oracle_rotation_error_deg": float(
                candidate_rotation[candidate_oracle_row]
            ),
            "rank_shortlist_count": int(len(shortlist_rows)),
            "rank_shortlist_oracle_translation_error_m": float(
                candidate_translation[shortlist_oracle_row]
            ),
            "rank_shortlist_oracle_rotation_error_deg": float(
                candidate_rotation[shortlist_oracle_row]
            ),
            "candidate_oracle_improves_source_oracle": bool(
                candidate_translation[candidate_oracle_row]
                < source_translation[source_oracle_row]
            ),
            "returned_improves_source": bool(
                returned_translation < source_translation[source_row]
            ),
            "returned_worsens_source": bool(
                returned_translation > source_translation[source_row]
            ),
            "optional_improves_source": bool(
                candidate_translation[optional_row] < source_translation[source_row]
            ),
            "audit_score_delta": float(arrays["audit_score_deltas"][source_row]),
            "candidate_rank_score": float(arrays["candidate_rank_scores"][optional_row]),
            "refinement_used_count": int(np.count_nonzero(arrays["refinement_used"][group])),
            "refinement_success_count": int(
                np.count_nonzero(arrays["refinement_success"][group])
            ),
            "optional_information_match_count": int(
                arrays["information_match_counts"][optional_row]
            ),
            "optional_bearing_max_angle_deg": float(
                arrays["observability_bearing_max_angle_deg"][optional_row]
            ),
            "optional_depth_span_ratio": float(
                arrays["observability_camera_depth_span_ratio"][optional_row]
            ),
            "optional_xyz_second_ratio": float(
                arrays["observability_xyz_second_singular_ratio"][optional_row]
            ),
        }
        per_query.append(row)

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in per_query:
        grouped.setdefault(
            (str(row["split_name"]), str(row["evaluation_label"])), []
        ).append(row)
    metrics: dict[str, object] = {}
    for (split_name, label), values in sorted(grouped.items()):
        metrics[f"{split_name}::{label}"] = {
            "query_count": int(len(values)),
            "immutable_source": _pose_summary(values, "source"),
            "rank_selected_optional": _pose_summary(values, "optional"),
            "audit_gated_returned": _pose_summary(values, "returned"),
            "source_hypothesis_oracle": _pose_summary(values, "source_oracle"),
            "crossfit_refined_hypothesis_oracle": _pose_summary(
                values, "candidate_oracle"
            ),
            "rank_shortlist_hypothesis_oracle": _pose_summary(
                values, "rank_shortlist_oracle"
            ),
            "median_rank_shortlist_count": _percentile(
                [float(row["rank_shortlist_count"]) for row in values], 50.0
            ),
            "promotion_count": int(sum(bool(row["promoted"]) for row in values)),
            "returned_win_count": int(
                sum(bool(row["returned_improves_source"]) for row in values)
            ),
            "returned_loss_count": int(
                sum(bool(row["returned_worsens_source"]) for row in values)
            ),
            "optional_win_count": int(
                sum(bool(row["optional_improves_source"]) for row in values)
            ),
            "candidate_oracle_improvement_count": int(
                sum(
                    bool(row["candidate_oracle_improves_source_oracle"])
                    for row in values
                )
            ),
            "median_refinement_used_count": _percentile(
                [float(row["refinement_used_count"]) for row in values], 50.0
            ),
            "median_optional_information_match_count": _percentile(
                [float(row["optional_information_match_count"]) for row in values],
                50.0,
            ),
            "median_optional_bearing_max_angle_deg": _percentile(
                [float(row["optional_bearing_max_angle_deg"]) for row in values],
                50.0,
            ),
            "median_optional_depth_span_ratio": _percentile(
                [float(row["optional_depth_span_ratio"]) for row in values], 50.0
            ),
            "median_optional_xyz_second_ratio": _percentile(
                [float(row["optional_xyz_second_ratio"]) for row in values], 50.0
            ),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", per_query)
    target_path = output_dir / "independent_crossfit_pose_alignment_targets_v1.npz"
    target_metadata = {
        "format": TARGET_ARTIFACT_FORMAT,
        "contains_target_fields": True,
        "targets_joined_after_selection": True,
        "alignment_artifacts": [str(path) for path in paths],
        "alignment_artifact_sha256": [file_sha256_short(path) for path in paths],
        "alignment_compatibility_sha256": compatibility_hash,
        "colmap_images_bin": str(model_dir / "images.bin"),
        "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        "row_count": int(len(query_ids)),
    }
    np.savez_compressed(
        target_path,
        query_ids=query_ids,
        split_names=split_names,
        evaluation_labels=labels,
        hypothesis_indices=arrays["hypothesis_indices"],
        source_translation_errors_m=source_translation,
        source_rotation_errors_deg=source_rotation,
        candidate_translation_errors_m=candidate_translation,
        candidate_rotation_errors_deg=candidate_rotation,
        metadata_json=np.asarray(json.dumps(target_metadata, sort_keys=True)),
    )
    summary = {
        "stage": "independent_crossfit_pose_alignment_gt_join",
        "protocol": {
            "selection_frozen_before_gt_join": True,
            "target_free_alignment_artifact_validated": True,
            "rank_and_audit_roles_separate": True,
            "immutable_fallback_pose": True,
        },
        "metrics": metrics,
        "inputs": target_metadata,
        "alignment_metadata": metadata[0],
        "outputs": {
            "targets": str(target_path),
            "per_query": str(output_dir / "per_query.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
