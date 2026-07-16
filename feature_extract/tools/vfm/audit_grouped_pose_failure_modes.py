"""Decompose selected grouped-pose failures with target-side diagnostics.

This command is evaluation-only. It joins pose targets and proposal residuals
after inference selection and must never produce an inference input artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import (
    build_disjoint_maplet_cluster_ids,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz


ARTIFACT_FORMAT = "grouped_pose_failure_decomposition_targets_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--score_artifacts", required=True)
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--success_threshold_m", type=float, default=0.10)
    parser.add_argument("--identity_threshold_px", type=float, default=2.0)
    parser.add_argument("--minimum_available_fraction", type=float, default=0.5)
    parser.add_argument("--minimum_identity_fraction", type=float, default=0.5)
    parser.add_argument("--minimum_shift_pairs", type=int, default=3)
    parser.add_argument("--minimum_shift_m", type=float, default=0.10)
    parser.add_argument("--maximum_shift_dispersion_m", type=float, default=0.20)
    parser.add_argument("--maximum_shift_relative_dispersion", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in value.split(",") if item.strip())
    if not paths:
        raise ValueError("artifact list is empty")
    return paths


def _coherent_shift(
    deltas: np.ndarray,
    *,
    minimum_pairs: int,
    minimum_norm_m: float,
    maximum_dispersion_m: float,
    maximum_relative_dispersion: float,
) -> tuple[bool, float, float]:
    values = np.asarray(deltas, dtype=np.float64).reshape(-1, 3)
    if len(values) < int(minimum_pairs):
        return False, float("nan"), float("nan")
    center = np.median(values, axis=0)
    norm = float(np.linalg.norm(center))
    dispersion = float(np.median(np.linalg.norm(values - center, axis=1)))
    limit = max(float(maximum_dispersion_m), float(maximum_relative_dispersion) * norm)
    return bool(norm >= float(minimum_norm_m) and dispersion <= limit), norm, dispersion


def _is_degenerate(hypothesis: dict[str, float]) -> bool:
    return bool(
        hypothesis["translation_information_min_eigenvalue"] < 10.0
        or hypothesis["translation_information_condition"] > 30.0
        or hypothesis["bearing_max_angle_deg"] < 40.0
        or hypothesis["camera_depth_span_ratio"] < 0.25
        or hypothesis["xyz_third_singular_ratio"] < 0.04
    )


def _classify(
    *,
    translation_error_m: float,
    success_threshold_m: float,
    available_fraction: float,
    minimum_available_fraction: float,
    identity_fraction: float,
    minimum_identity_fraction: float,
    degenerate: bool,
    coherent_shift: bool,
    maplet_mismatch_fraction: float,
) -> str:
    if translation_error_m <= success_threshold_m:
        return "success_10cm"
    if available_fraction < minimum_available_fraction:
        return "E_correct_candidate_absent_from_sample_pool"
    if identity_fraction >= minimum_identity_fraction and degenerate:
        return "D_identity_right_geometry_degenerate"
    if identity_fraction >= minimum_identity_fraction:
        return "C_identity_right_spatial_or_pose_wrong"
    if coherent_shift:
        return "A_coherent_3d_identity_shift"
    if maplet_mismatch_fraction >= 0.5:
        return "B_wrong_disjoint_maplet"
    return "B_unstructured_wrong_identity"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    score_paths = _paths(args.score_artifacts)
    if len(hypothesis_paths) != len(score_paths):
        raise ValueError("hypothesis and score shard counts differ")
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_query.csv"
    if (summary_path.exists() or rows_path.exists()) and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    with np.load(Path(args.target_artifact), allow_pickle=False) as data:
        target_query_ids = data["query_ids"].astype(str)
        target_hypothesis_indices = np.asarray(data["hypothesis_indices"], dtype=np.int64)
        translation_errors = np.asarray(data["translation_errors_m"], dtype=np.float64)
        rotation_errors = np.asarray(data["rotation_errors_deg"], dtype=np.float64)
    with np.load(Path(args.proposals), allow_pickle=False) as data:
        candidate_track_ids = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        candidate_residuals = np.asarray(data["candidate_gt_residuals_px"], dtype=np.float64)
    bank, _bank_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    bank_position = {int(track): row for row, track in enumerate(bank.track_ids)}
    maplets, _maplet_metadata = load_local_maplet_support_index_npz(Path(args.maplet_index))
    cluster_ids = build_disjoint_maplet_cluster_ids(maplets)
    cluster_by_track = {
        int(track): int(cluster)
        for track, cluster in zip(maplets.anchor_track_ids, cluster_ids)
    }

    rows: list[dict[str, object]] = []
    offset = 0
    for hypothesis_path, score_path in zip(hypothesis_paths, score_paths):
        with np.load(score_path, allow_pickle=False) as score:
            selected = np.asarray(score["independent_score_top1"], dtype=bool)
            score_query_ids = score["query_ids"].astype(str)
            score_labels = score["evaluation_labels"].astype(str)
            score_indices = np.asarray(score["hypothesis_indices"], dtype=np.int64)
        with np.load(hypothesis_path, allow_pickle=False) as hypothesis:
            shard_count = len(selected)
            target_slice = slice(offset, offset + shard_count)
            if not np.array_equal(score_query_ids, target_query_ids[target_slice]):
                raise ValueError("target and inference query ordering differs")
            if not np.array_equal(score_indices, target_hypothesis_indices[target_slice]):
                raise ValueError("target and inference hypothesis identities differ")
            hypothesis_keys = list(
                zip(
                    hypothesis["query_ids"].astype(str).tolist(),
                    hypothesis["evaluation_labels"].astype(str).tolist(),
                    np.asarray(hypothesis["hypothesis_indices"], dtype=np.int64).tolist(),
                )
            )
            hypothesis_position = {key: row for row, key in enumerate(hypothesis_keys)}
            if len(hypothesis_position) != len(hypothesis_keys):
                raise ValueError("hypothesis identity keys are not unique")
            score_keys = list(
                zip(
                    score_query_ids.tolist(),
                    score_labels.tolist(),
                    score_indices.tolist(),
                )
            )
            if set(score_keys) != set(hypothesis_keys):
                raise ValueError("score and hypothesis identity sets differ")
            for score_row in np.flatnonzero(selected):
                local_row = hypothesis_position[score_keys[int(score_row)]]
                global_row = offset + int(score_row)
                sample_count = 0
                available_count = 0
                identity_count = 0
                maplet_comparisons = 0
                maplet_mismatches = 0
                deltas = []
                for proposal_row, selected_track in zip(
                    hypothesis["sample_token_indices"][local_row],
                    hypothesis["sample_track_ids"][local_row],
                ):
                    proposal_row = int(proposal_row)
                    selected_track = int(selected_track)
                    if proposal_row < 0 or selected_track < 0:
                        continue
                    sample_count += 1
                    tracks = candidate_track_ids[proposal_row]
                    residuals = candidate_residuals[proposal_row]
                    valid = tracks >= 0
                    correct = valid & (residuals <= float(args.identity_threshold_px))
                    selected_columns = np.flatnonzero(tracks == selected_track)
                    if len(selected_columns) and residuals[selected_columns[0]] <= float(
                        args.identity_threshold_px
                    ):
                        identity_count += 1
                    if not np.any(correct):
                        continue
                    available_count += 1
                    correct_columns = np.flatnonzero(correct)
                    correct_column = correct_columns[np.argmin(residuals[correct_columns])]
                    correct_track = int(tracks[correct_column])
                    if selected_track == correct_track:
                        continue
                    if selected_track in bank_position and correct_track in bank_position:
                        deltas.append(
                            bank.xyz[bank_position[selected_track]]
                            - bank.xyz[bank_position[correct_track]]
                        )
                    if selected_track in cluster_by_track and correct_track in cluster_by_track:
                        maplet_comparisons += 1
                        maplet_mismatches += int(
                            cluster_by_track[selected_track]
                            != cluster_by_track[correct_track]
                        )
                available_fraction = available_count / max(sample_count, 1)
                identity_fraction = identity_count / max(sample_count, 1)
                maplet_mismatch_fraction = maplet_mismatches / max(maplet_comparisons, 1)
                coherent, shift_norm, shift_dispersion = _coherent_shift(
                    np.asarray(deltas, dtype=np.float64).reshape(-1, 3),
                    minimum_pairs=int(args.minimum_shift_pairs),
                    minimum_norm_m=float(args.minimum_shift_m),
                    maximum_dispersion_m=float(args.maximum_shift_dispersion_m),
                    maximum_relative_dispersion=float(args.maximum_shift_relative_dispersion),
                )
                geometry = {
                    "translation_information_min_eigenvalue": float(
                        hypothesis["translation_information_min_eigenvalues"][local_row]
                    ),
                    "translation_information_condition": float(
                        hypothesis["translation_information_conditions"][local_row]
                    ),
                    "bearing_max_angle_deg": float(
                        hypothesis["bearing_max_angles_deg"][local_row]
                    ),
                    "camera_depth_span_ratio": float(
                        hypothesis["camera_depth_span_ratios"][local_row]
                    ),
                    "xyz_third_singular_ratio": float(
                        hypothesis["xyz_third_singular_ratios"][local_row]
                    ),
                }
                degenerate = _is_degenerate(geometry)
                translation_error = float(translation_errors[global_row])
                row = {
                    "query_id": str(score_query_ids[score_row]),
                    "split_name": str(hypothesis["split_names"][local_row]),
                    "hypothesis_index": int(score_indices[score_row]),
                    "translation_error_m": translation_error,
                    "rotation_error_deg": float(rotation_errors[global_row]),
                    "sample_count": int(sample_count),
                    "correct_candidate_available_fraction": float(available_fraction),
                    "selected_identity_correct_fraction": float(identity_fraction),
                    "wrong_track_delta_count": int(len(deltas)),
                    "coherent_shift": bool(coherent),
                    "coherent_shift_norm_m": shift_norm,
                    "coherent_shift_dispersion_m": shift_dispersion,
                    "maplet_comparison_count": int(maplet_comparisons),
                    "maplet_mismatch_fraction": float(maplet_mismatch_fraction),
                    "geometry_degenerate": bool(degenerate),
                    **geometry,
                }
                row["failure_class"] = _classify(
                    translation_error_m=translation_error,
                    success_threshold_m=float(args.success_threshold_m),
                    available_fraction=available_fraction,
                    minimum_available_fraction=float(args.minimum_available_fraction),
                    identity_fraction=identity_fraction,
                    minimum_identity_fraction=float(args.minimum_identity_fraction),
                    degenerate=degenerate,
                    coherent_shift=coherent,
                    maplet_mismatch_fraction=maplet_mismatch_fraction,
                )
                rows.append(row)
            offset += shard_count
    if offset != len(target_query_ids):
        raise ValueError("target rows remain after consuming inference shards")

    metrics = {}
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split_name"])].append(row)
    for split_name, split_rows in sorted(by_split.items()):
        failures = [
            row
            for row in split_rows
            if float(row["translation_error_m"]) > float(args.success_threshold_m)
        ]
        counts = Counter(str(row["failure_class"]) for row in split_rows)
        failure_counts = Counter(str(row["failure_class"]) for row in failures)
        errors = np.asarray(
            [float(row["translation_error_m"]) for row in split_rows], dtype=np.float64
        )
        metrics[split_name] = {
            "query_count": len(split_rows),
            "failure_count": len(failures),
            "translation_median_m": float(np.median(errors)),
            "translation_p90_m": float(np.quantile(errors, 0.9)),
            "class_counts": dict(counts),
            "failure_class_counts": dict(failure_counts),
            "failure_class_fractions": {
                key: float(value / max(len(failures), 1))
                for key, value in failure_counts.items()
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "grouped_pose_failure_decomposition",
        "format": ARTIFACT_FORMAT,
        "contains_target_fields": True,
        "evaluation_only_not_inference_input": True,
        "classification_priority": ["success", "E", "D", "C", "A", "B"],
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if key.startswith(("success_", "identity_", "minimum_", "maximum_"))
        },
        "inputs": {
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in hypothesis_paths],
            "score_artifact_sha256": [file_sha256_short(path) for path in score_paths],
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "proposals_sha256": file_sha256_short(Path(args.proposals)),
            "projected_landmark_bank_sha256": file_sha256_short(
                Path(args.projected_landmark_bank)
            ),
            "maplet_index_sha256": file_sha256_short(Path(args.maplet_index)),
        },
        "metrics": metrics,
        "outputs": {"per_query": str(rows_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
