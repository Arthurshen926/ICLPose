"""Join frozen independent pose scores with GT and report selection quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    SCORE_ARTIFACT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    qvec_to_rotmat,
    read_colmap_images_binary,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


TARGET_ARTIFACT_FORMAT = "independent_landmark_hypothesis_score_targets_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score_artifacts",
        required=True,
        help="comma-separated target-free independent score NPZ files",
    )
    parser.add_argument(
        "--hypothesis_artifacts",
        required=True,
        help="comma-separated grouped inference-only hypothesis NPZ shards",
    )
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _canonical_hash(payload: object) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()[:16]


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one artifact is required")
    return paths


def _validate_candidate_spatial_materialization_contract(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    strict_contract: Mapping[str, object],
    *,
    row_count: int,
) -> None:
    """Reject RGB score artifacts whose claimed absolute evidence was absent.

    A neutral all-missing spatial tensor is mathematically pose-independent, but
    it is not a usable independent RGB verifier.  Requiring the scorer's
    per-query materialization audit here prevents a legacy artifact from being
    ranked as though it carried RGB evidence on a split for which no modes were
    exported.
    """

    if strict_contract.get("candidate_specific_rgb_spatial_modes") is not True:
        return
    required_contract = {
        "candidate_spatial_dustbin_and_missing_pose_independent": True,
        "candidate_spatial_omitted_topk_mass_is_null": True,
        "candidate_spatial_query_materialization": (
            "required_at_least_one_heldout_verification_point"
        ),
    }
    if any(
        strict_contract.get(key) != expected
        for key, expected in required_contract.items()
    ):
        raise ValueError(
            f"{path}: candidate RGB spatial evidence lacks the strict "
            "materialization contract"
        )
    required_arrays = (
        "candidate_spatial_materialized_verification_point_counts",
        "candidate_spatial_materialized_candidate_view_counts",
    )
    missing = [key for key in required_arrays if key not in arrays]
    if missing:
        raise ValueError(
            f"{path}: candidate RGB spatial evidence lacks materialization "
            f"audit arrays: {missing}"
        )
    for key in required_arrays:
        values = np.asarray(arrays[key])
        if values.ndim != 1 or len(values) != int(row_count):
            raise ValueError(
                f"{path}: {key} must be a row-aligned one-dimensional audit"
            )
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                f"{path}: candidate RGB spatial evidence is unmaterialized "
                f"for at least one scored row ({key})"
            )


def _load_score_artifact(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: score artifact has no metadata")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != SCORE_ARTIFACT_FORMAT:
        raise ValueError(f"{path}: unsupported score artifact format")
    if metadata.get("contains_target_fields") is not False:
        raise ValueError(f"{path}: score artifact is not target-free")
    if metadata.get("pose_or_ground_truth_used_for_scoring") is not False:
        raise ValueError(f"{path}: score generation accessed pose targets")
    if metadata.get("supervision_arrays_loaded") is not False:
        raise ValueError(f"{path}: score generation loaded supervision arrays")
    strict_contract = metadata.get("strict_absolute_evidence_contract")
    required_contract = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
    }
    if not isinstance(strict_contract, Mapping) or any(
        strict_contract.get(key) is not expected
        for key, expected in required_contract.items()
    ):
        raise ValueError(f"{path}: score artifact is not strict absolute evidence")
    row_count = int(metadata.get("row_count", -1))
    for key, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != row_count:
            raise ValueError(f"{path}: {key} is not row-aligned")
    _validate_candidate_spatial_materialization_contract(
        path,
        arrays,
        strict_contract,
        row_count=row_count,
    )
    keys = list(
        zip(
            arrays["query_ids"].astype(str).tolist(),
            arrays["evaluation_labels"].astype(str).tolist(),
            arrays["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate score rows")
    return arrays, metadata


def _score_compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("score metadata has no input manifest")
    return {
        "version": metadata.get("version"),
        "hypothesis_compatibility_sha256": metadata.get(
            "hypothesis_compatibility_sha256"
        ),
        "config": metadata.get("config"),
        "selection": metadata.get("selection"),
        "implementation": metadata.get("implementation"),
        "query_point_selection": metadata.get("query_point_selection"),
        "hypothesis_scope": metadata.get("hypothesis_scope"),
        "crossfit": metadata.get("crossfit"),
        "strict_absolute_evidence_contract": metadata.get(
            "strict_absolute_evidence_contract"
        ),
        "supervision_arrays_loaded": metadata.get("supervision_arrays_loaded"),
        "view_geometry_mode": metadata.get("view_geometry_mode"),
        "detector_query_cache_sha256": inputs.get(
            "detector_query_cache_sha256"
        ),
        "proposals_sha256": inputs.get("proposals_sha256"),
        "candidate_artifact_sha256": inputs.get("candidate_artifact_sha256"),
        "fixed_candidate_prior_overlay_sha256": inputs.get(
            "fixed_candidate_prior_overlay_sha256"
        ),
        "fixed_candidate_prior_overlay_metadata_sha256": inputs.get(
            "fixed_candidate_prior_overlay_metadata_sha256"
        ),
        "candidate_spatial_likelihood_sha256": inputs.get(
            "candidate_spatial_likelihood_sha256"
        ),
        "candidate_spatial_likelihood_metadata_sha256": inputs.get(
            "candidate_spatial_likelihood_metadata_sha256"
        ),
        "projected_landmark_bank_sha256": inputs.get(
            "projected_landmark_bank_sha256"
        ),
        "independent_verification_landmark_bank_sha256": inputs.get(
            "independent_verification_landmark_bank_sha256"
        ),
        "support_geometry_index_sha256": inputs.get(
            "support_geometry_index_sha256"
        ),
        "prototype_view_geometry_sha256": inputs.get(
            "prototype_view_geometry_sha256"
        ),
        "maplet_support_index_sha256": inputs.get("maplet_support_index_sha256"),
        "detector_descriptor_space_id": inputs.get(
            "detector_descriptor_space_id"
        ),
        "landmark_descriptor_space_id": inputs.get(
            "landmark_descriptor_space_id"
        ),
        "projection_space_id": inputs.get("projection_space_id"),
        "projection_compatibility": inputs.get("projection_compatibility"),
    }


def _selection_score_contract(
    metadata: Mapping[str, object], score_fields: set[str]
) -> tuple[str, str]:
    """Resolve the target-free score array used for frozen pose ranking."""

    selection_metadata = metadata.get("selection")
    if selection_metadata is None:
        selection_score_field = "independent_log_likelihood_means"
        selection_statistic = "mean"
    elif isinstance(selection_metadata, Mapping):
        selection_score_field = str(
            selection_metadata.get(
                "score_field", "independent_log_likelihood_means"
            )
        )
        selection_statistic = str(selection_metadata.get("statistic", "mean"))
    else:
        raise ValueError("score metadata selection contract must be an object")
    if selection_score_field not in score_fields:
        raise ValueError(
            f"score artifact lacks configured selection field: {selection_score_field}"
        )
    return selection_score_field, selection_statistic


def _merge_score_artifacts(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], str]:
    loaded = [_load_score_artifact(path) for path in paths]
    fingerprints = [
        _canonical_hash(_score_compatibility(metadata))
        for _arrays, metadata in loaded
    ]
    if len(set(fingerprints)) != 1:
        raise ValueError(f"incompatible score shards: {fingerprints}")
    keys = set(loaded[0][0])
    if any(set(arrays) != keys for arrays, _ in loaded):
        raise ValueError("score shards have different schemas")
    merged = {
        key: np.concatenate([arrays[key] for arrays, _ in loaded], axis=0)
        for key in sorted(keys)
    }
    row_keys = list(
        zip(
            merged["query_ids"].astype(str).tolist(),
            merged["evaluation_labels"].astype(str).tolist(),
            merged["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("merged score shards contain duplicate rows")
    return merged, [metadata for _arrays, metadata in loaded], fingerprints[0]


def _merge_hypothesis_lookup(
    paths: Sequence[Path],
) -> dict[tuple[str, str, int], np.ndarray]:
    lookup: dict[tuple[str, str, int], np.ndarray] = {}
    for path in paths:
        arrays, _metadata = load_inference_artifact(path)
        for query_id, label, hypothesis_index, pose in zip(
            arrays["query_ids"].astype(str),
            arrays["evaluation_labels"].astype(str),
            arrays["hypothesis_indices"].astype(np.int64),
            np.asarray(arrays["poses_w2c"], dtype=np.float64),
        ):
            key = (str(query_id), str(label), int(hypothesis_index))
            if key in lookup:
                raise ValueError(f"duplicate hypothesis row: {key}")
            lookup[key] = pose
    return lookup


def _gt_pose_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = qvec_to_rotmat(image.qvec)
    pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    return pose


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
        if row.get(f"{prefix}_translation_error_m") is not None
        and row.get(f"{prefix}_rotation_error_deg") is not None
    ]
    return {
        "coverage": float(len(values) / max(len(rows), 1)),
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
        "recall_5cm_5deg": float(
            sum(value[0] <= 0.05 and value[1] <= 5.0 for value in values)
            / max(len(rows), 1)
        ),
        "recall_10cm_5deg": float(
            sum(value[0] <= 0.10 and value[1] <= 5.0 for value in values)
            / max(len(rows), 1)
        ),
        "recall_25cm_2deg": float(
            sum(value[0] <= 0.25 and value[1] <= 2.0 for value in values)
            / max(len(rows), 1)
        ),
    }


def _best_correct_rank(
    scores: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    threshold_m: float,
) -> int | None:
    order = np.argsort(-scores, kind="mergesort")
    correct = (translation <= float(threshold_m)) & (rotation <= 5.0)
    ranks = np.flatnonzero(correct[order])
    return None if ranks.size == 0 else int(ranks[0] + 1)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_paths = _paths(args.score_artifacts)
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    scores, score_metadata, score_compatibility_hash = _merge_score_artifacts(
        score_paths
    )
    hypothesis_lookup = _merge_hypothesis_lookup(hypothesis_paths)
    model_dir = Path(args.colmap_model_dir)
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}

    query_ids = scores["query_ids"].astype(str)
    split_names = scores["split_names"].astype(str)
    labels = scores["evaluation_labels"].astype(str)
    hypothesis_indices = scores["hypothesis_indices"].astype(np.int64)
    selection_score_field, selection_statistic = _selection_score_contract(
        score_metadata[0], set(scores)
    )
    likelihood = np.asarray(scores[selection_score_field], dtype=np.float64)
    likelihood_mean = np.asarray(
        scores["independent_log_likelihood_means"], dtype=np.float64
    )
    score_top1 = np.asarray(scores["independent_score_top1"], dtype=bool)
    source_chosen = np.asarray(
        scores["source_chosen_for_optional_pose"], dtype=bool
    )
    poses = np.zeros((len(query_ids), 4, 4), dtype=np.float64)
    translation = np.zeros((len(query_ids),), dtype=np.float64)
    rotation = np.zeros((len(query_ids),), dtype=np.float64)
    for row, (query_id, label, hypothesis_index) in enumerate(
        zip(query_ids, labels, hypothesis_indices)
    ):
        key = (str(query_id), str(label), int(hypothesis_index))
        pose = hypothesis_lookup.get(key)
        if pose is None:
            raise ValueError(f"score row has no source hypothesis: {key}")
        image = images_by_name.get(str(query_id))
        if image is None:
            raise ValueError(f"query absent from COLMAP GT model: {query_id}")
        poses[row] = pose
        error = pnp_pose_error(pose, _gt_pose_w2c(image))
        translation[row] = float(error.translation_m)
        rotation[row] = float(error.rotation_deg)

    group_keys = sorted(set(zip(split_names.tolist(), labels.tolist(), query_ids.tolist())))
    per_query: list[dict[str, object]] = []
    for split_name, label, query_id in group_keys:
        group = np.flatnonzero(
            (split_names == split_name)
            & (labels == label)
            & (query_ids == query_id)
        )
        selected = group[score_top1[group]]
        chosen = group[source_chosen[group]]
        if len(selected) != 1:
            raise ValueError(f"{query_id}: independent score must select one pose")
        if len(chosen) > 1:
            raise ValueError(f"{query_id}: source selected multiple poses")
        selected_row = int(selected[0])
        chosen_row = None if len(chosen) == 0 else int(chosen[0])
        oracle_row = int(
            min(
                group.tolist(),
                key=lambda row: (
                    float(translation[row]),
                    float(rotation[row]),
                    int(row),
                ),
            )
        )
        local_scores = likelihood[group]
        local_translation = translation[group]
        local_rotation = rotation[group]
        row: dict[str, object] = {
            "query_id": query_id,
            "split_name": split_name,
            "evaluation_label": label,
            "hypothesis_count": int(len(group)),
            "selected_translation_error_m": float(translation[selected_row]),
            "selected_rotation_error_deg": float(rotation[selected_row]),
            "selected_log_likelihood_mean": float(likelihood_mean[selected_row]),
            "selected_selection_score": float(likelihood[selected_row]),
            "chosen_translation_error_m": (
                None if chosen_row is None else float(translation[chosen_row])
            ),
            "chosen_rotation_error_deg": (
                None if chosen_row is None else float(rotation[chosen_row])
            ),
            "chosen_log_likelihood_mean": (
                None if chosen_row is None else float(likelihood_mean[chosen_row])
            ),
            "chosen_selection_score": (
                None if chosen_row is None else float(likelihood[chosen_row])
            ),
            "oracle_translation_error_m": float(translation[oracle_row]),
            "oracle_rotation_error_deg": float(rotation[oracle_row]),
            "oracle_score_rank": int(
                1 + np.count_nonzero(local_scores > likelihood[oracle_row])
            ),
            "best_3cm_rank": _best_correct_rank(
                local_scores, local_translation, local_rotation, 0.03
            ),
            "best_5cm_rank": _best_correct_rank(
                local_scores, local_translation, local_rotation, 0.05
            ),
            "best_10cm_rank": _best_correct_rank(
                local_scores, local_translation, local_rotation, 0.10
            ),
            "selection_regret_m": float(
                translation[selected_row] - translation[oracle_row]
            ),
            "selected_minus_chosen_translation_m": (
                None
                if chosen_row is None
                else float(translation[selected_row] - translation[chosen_row])
            ),
            "selected_improves_chosen": bool(
                chosen_row is not None
                and translation[selected_row] < translation[chosen_row]
            ),
            "selected_worsens_chosen": bool(
                chosen_row is not None
                and translation[selected_row] > translation[chosen_row]
            ),
            "selected_catastrophic_1m": bool(translation[selected_row] > 1.0),
            "verification_point_count": int(
                scores["verification_point_counts"][selected_row]
            ),
            "direct_excluded_track_count": int(
                scores["direct_excluded_track_counts"][selected_row]
            ),
            "total_excluded_track_count": int(
                scores["total_excluded_track_counts"][selected_row]
            ),
            "selected_effective_point_count": int(
                scores["independent_effective_point_counts"][selected_row]
            ),
            "selected_evidence_coverage": float(
                scores["independent_evidence_coverages"][selected_row]
            ),
        }
        per_query.append(row)

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in per_query:
        grouped.setdefault(
            (str(row["split_name"]), str(row["evaluation_label"])), []
        ).append(row)
    metrics: dict[str, object] = {}
    for (split_name, label), rows in sorted(grouped.items()):
        key = f"{split_name}::{label}"
        paired_delta = [
            float(row["selected_minus_chosen_translation_m"])
            for row in rows
            if row["selected_minus_chosen_translation_m"] is not None
        ]
        metrics[key] = {
            "split_name": split_name,
            "evaluation_label": label,
            "query_count": int(len(rows)),
            "selected": _pose_summary(rows, "selected"),
            "source_chosen": _pose_summary(rows, "chosen"),
            "shortlist_oracle": _pose_summary(rows, "oracle"),
            "median_oracle_score_rank": _percentile(
                [float(row["oracle_score_rank"]) for row in rows], 50.0
            ),
            "best_3cm_rank_coverage": float(
                sum(row["best_3cm_rank"] is not None for row in rows)
                / max(len(rows), 1)
            ),
            "median_best_3cm_rank": _percentile(
                [
                    float(row["best_3cm_rank"])
                    for row in rows
                    if row["best_3cm_rank"] is not None
                ],
                50.0,
            ),
            "best_5cm_rank_coverage": float(
                sum(row["best_5cm_rank"] is not None for row in rows)
                / max(len(rows), 1)
            ),
            "median_best_5cm_rank": _percentile(
                [
                    float(row["best_5cm_rank"])
                    for row in rows
                    if row["best_5cm_rank"] is not None
                ],
                50.0,
            ),
            "best_10cm_rank_coverage": float(
                sum(row["best_10cm_rank"] is not None for row in rows)
                / max(len(rows), 1)
            ),
            "median_best_10cm_rank": _percentile(
                [
                    float(row["best_10cm_rank"])
                    for row in rows
                    if row["best_10cm_rank"] is not None
                ],
                50.0,
            ),
            "mean_selected_minus_chosen_translation_cm": (
                None
                if not paired_delta
                else float(100.0 * np.mean(paired_delta))
            ),
            "improves_chosen_fraction": float(
                sum(bool(row["selected_improves_chosen"]) for row in rows)
                / max(len(rows), 1)
            ),
            "worsens_chosen_fraction": float(
                sum(bool(row["selected_worsens_chosen"]) for row in rows)
                / max(len(rows), 1)
            ),
            "catastrophic_1m_count": int(
                sum(bool(row["selected_catastrophic_1m"]) for row in rows)
            ),
            "median_selection_regret_cm": _percentile(
                [100.0 * float(row["selection_regret_m"]) for row in rows],
                50.0,
            ),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "independent_landmark_hypothesis_targets_v1.npz"
    target_metadata = {
        "format": TARGET_ARTIFACT_FORMAT,
        "contains_target_fields": True,
        "targets_joined_after_independent_score_selection": True,
        "score_artifacts": [str(path) for path in score_paths],
        "score_artifact_sha256": [
            file_sha256_short(path) for path in score_paths
        ],
        "score_compatibility_sha256": score_compatibility_hash,
        "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
        "hypothesis_artifact_sha256": [
            file_sha256_short(path) for path in hypothesis_paths
        ],
        "colmap_images_bin": str(model_dir / "images.bin"),
        "colmap_images_bin_sha256": file_sha256_short(
            model_dir / "images.bin"
        ),
        "row_count": int(len(query_ids)),
    }
    np.savez_compressed(
        target_path,
        query_ids=query_ids,
        split_names=split_names,
        evaluation_labels=labels,
        hypothesis_indices=hypothesis_indices,
        translation_errors_m=translation,
        rotation_errors_deg=rotation,
        metadata_json=np.asarray(json.dumps(target_metadata, sort_keys=True)),
    )
    _write_csv(output_dir / "per_query.csv", per_query)
    summary = {
        "stage": "independent_landmark_pose_score_gt_join",
        "protocol": {
            "score_selection_frozen_before_gt_join": True,
            "target_free_score_artifact_validated": True,
            "same_hypothesis_denominator_for_source_and_independent_selection": True,
        },
        "selection": {
            "statistic": selection_statistic,
            "score_field": selection_score_field,
        },
        "metrics": metrics,
        "inputs": target_metadata,
        "score_metadata": score_metadata[0],
        "outputs": {
            "targets": str(target_path),
            "per_query": str(output_dir / "per_query.csv"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
