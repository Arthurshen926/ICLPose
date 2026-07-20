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
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import (
    build_disjoint_maplet_cluster_ids,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz


ARTIFACT_FORMAT = "grouped_pose_failure_decomposition_targets_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--score_artifacts", required=True)
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument(
        "--candidate_artifact",
        required=True,
        help=(
            "inference-only candidate layout used to generate the hypotheses; "
            "its selected columns define the frozen top-L pool"
        ),
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--success_threshold_m", type=float, default=0.10)
    parser.add_argument("--success_rotation_threshold_deg", type=float, default=5.0)
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


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{context}: metadata_json is required")
    try:
        value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context}: metadata_json is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context}: metadata_json must contain an object")
    return value


def _require_arrays(
    payload: Mapping[str, np.ndarray], *, required: set[str], context: str
) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{context}: required arrays are missing: {missing}")


def _require_hash_list(
    metadata: Mapping[str, Any],
    *,
    key: str,
    expected: Sequence[str],
    context: str,
) -> None:
    value = metadata.get(key)
    if not isinstance(value, list) or [str(item) for item in value] != list(expected):
        raise ValueError(f"{context}: {key} does not match the supplied shard list")


def _actual_sample_candidates(
    *,
    proposal_row: int,
    selected_track: int,
    expected_query_id: str,
    proposal_query_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_residuals: np.ndarray,
    candidate_position_by_proposal_row: np.ndarray,
    candidate_selected_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidate columns actually available to one sampled match.

    `sample_token_indices` are global proposal-row indices.  The corresponding
    candidate artifact can reorder or remove proposal columns, so target-side
    auditing must not silently inspect the broader raw proposal row.  The
    frozen artifact does not serialize the smaller per-hypothesis PROSAC
    candidate limit, so this helper deliberately audits fixed top-L coverage,
    not eligibility under a particular minimal-set draw.
    """

    if proposal_row < 0 or proposal_row >= len(proposal_query_ids):
        raise ValueError("sample proposal row is outside the proposal artifact")
    if str(proposal_query_ids[proposal_row]) != str(expected_query_id):
        raise ValueError("sample proposal row belongs to a different query")
    candidate_position = int(candidate_position_by_proposal_row[proposal_row])
    if candidate_position < 0:
        raise ValueError("sample proposal row is absent from the candidate artifact")
    columns = np.asarray(candidate_selected_columns[candidate_position], dtype=np.int64)
    if np.any((columns < -1) | (columns >= candidate_track_ids.shape[1])):
        raise ValueError("candidate artifact contains an out-of-range proposal column")
    columns = columns[columns >= 0]
    if len(columns) == 0:
        raise ValueError("sample proposal row has no active candidate columns")
    tracks = np.asarray(candidate_track_ids[proposal_row, columns], dtype=np.int64)
    residuals = np.asarray(candidate_residuals[proposal_row, columns], dtype=np.float64)
    if selected_track < 0 or not np.any(tracks == int(selected_track)):
        raise ValueError("sampled track is absent from its actual candidate pool")
    return tracks, residuals


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
    rotation_error_deg: float,
    success_threshold_m: float,
    success_rotation_threshold_deg: float,
    available_fraction: float,
    minimum_available_fraction: float,
    identity_fraction: float,
    minimum_identity_fraction: float,
    degenerate: bool,
    coherent_shift: bool,
    maplet_mismatch_fraction: float,
) -> str:
    if (
        translation_error_m <= success_threshold_m
        and rotation_error_deg <= success_rotation_threshold_deg
    ):
        return "success_10cm"
    if available_fraction < minimum_available_fraction:
        return "E_correct_candidate_absent_from_frozen_topL_pool"
    if identity_fraction >= minimum_identity_fraction and degenerate:
        return "D_identity_right_geometry_degenerate"
    if identity_fraction >= minimum_identity_fraction:
        return "C_identity_right_spatial_or_pose_wrong"
    if coherent_shift:
        return "A_coherent_3d_identity_shift"
    if maplet_mismatch_fraction >= 0.5:
        return "B_wrong_disjoint_maplet"
    return "B_unstructured_wrong_identity"


def _finite_distribution(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"sample_count": 0, "median": None, "p90": None}
    return {
        "sample_count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    hypothesis_paths = _paths(args.hypothesis_artifacts)
    score_paths = _paths(args.score_artifacts)
    if len(hypothesis_paths) != len(score_paths):
        raise ValueError("hypothesis and score shard counts differ")
    target_path = Path(args.target_artifact)
    candidate_path = Path(args.candidate_artifact)
    proposal_path = Path(args.proposals)
    bank_path = Path(args.projected_landmark_bank)
    maplet_path = Path(args.maplet_index)
    hypothesis_hashes = [file_sha256_short(path) for path in hypothesis_paths]
    score_hashes = [file_sha256_short(path) for path in score_paths]
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_query.csv"
    if (summary_path.exists() or rows_path.exists()) and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    with np.load(target_path, allow_pickle=False) as data:
        _require_arrays(
            data,
            required={
                "metadata_json",
                "query_ids",
                "split_names",
                "evaluation_labels",
                "hypothesis_indices",
                "translation_errors_m",
                "rotation_errors_deg",
            },
            context="target artifact",
        )
        target_metadata = _metadata(data, context="target artifact")
        target_query_ids = data["query_ids"].astype(str)
        target_split_names = data["split_names"].astype(str)
        target_labels = data["evaluation_labels"].astype(str)
        target_hypothesis_indices = np.asarray(data["hypothesis_indices"], dtype=np.int64)
        translation_errors = np.asarray(data["translation_errors_m"], dtype=np.float64)
        rotation_errors = np.asarray(data["rotation_errors_deg"], dtype=np.float64)
    target_count = len(target_query_ids)
    if any(
        np.asarray(values).shape != (target_count,)
        for values in (
            target_split_names,
            target_labels,
            target_hypothesis_indices,
            translation_errors,
            rotation_errors,
        )
    ):
        raise ValueError("target artifact arrays are not row-aligned")
    if target_metadata.get("contains_target_fields") is not True:
        raise ValueError("target artifact is not explicitly target-side")
    _require_hash_list(
        target_metadata,
        key="hypothesis_artifact_sha256",
        expected=hypothesis_hashes,
        context="target artifact",
    )
    _require_hash_list(
        target_metadata,
        key="score_artifact_sha256",
        expected=score_hashes,
        context="target artifact",
    )

    with np.load(proposal_path, allow_pickle=False) as data:
        _require_arrays(
            data,
            required={"query_ids", "candidate_track_ids", "candidate_gt_residuals_px"},
            context="proposal artifact",
        )
        proposal_query_ids = np.asarray(data["query_ids"]).astype(str)
        candidate_track_ids = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        candidate_residuals = np.asarray(data["candidate_gt_residuals_px"], dtype=np.float64)
    if (
        candidate_track_ids.ndim != 2
        or candidate_residuals.shape != candidate_track_ids.shape
        or proposal_query_ids.shape != (candidate_track_ids.shape[0],)
    ):
        raise ValueError("proposal artifact candidate arrays are not aligned")

    with np.load(candidate_path, allow_pickle=False) as data:
        _require_arrays(
            data,
            required={"metadata_json", "selected_rows", "selected_columns"},
            context="candidate artifact",
        )
        candidate_metadata = _metadata(data, context="candidate artifact")
        candidate_selected_rows = np.asarray(data["selected_rows"], dtype=np.int64)
        candidate_selected_columns = np.asarray(data["selected_columns"], dtype=np.int64)
    if candidate_metadata.get("contains_ground_truth") is not False:
        raise ValueError("candidate artifact must be inference-only")
    proposal_hash = file_sha256_short(proposal_path)
    bank_hash = file_sha256_short(bank_path)
    maplet_hash = file_sha256_short(maplet_path)
    candidate_hash = file_sha256_short(candidate_path)
    for key, expected in (
        ("proposals_sha256", proposal_hash),
        ("projected_landmark_bank_sha256", bank_hash),
        ("maplet_support_index_sha256", maplet_hash),
    ):
        if str(candidate_metadata.get(key)) != expected:
            raise ValueError(f"candidate artifact {key} differs from the audit input")
    if (
        candidate_selected_rows.ndim != 1
        or candidate_selected_columns.ndim != 2
        or candidate_selected_columns.shape[0] != len(candidate_selected_rows)
        or len(np.unique(candidate_selected_rows)) != len(candidate_selected_rows)
        or np.any(
            (candidate_selected_rows < 0)
            | (candidate_selected_rows >= len(proposal_query_ids))
        )
    ):
        raise ValueError("candidate artifact selected rows/columns are invalid")
    candidate_position_by_proposal_row = np.full(
        (len(proposal_query_ids),), -1, dtype=np.int64
    )
    candidate_position_by_proposal_row[candidate_selected_rows] = np.arange(
        len(candidate_selected_rows), dtype=np.int64
    )

    bank, _bank_metadata = load_landmark_index_npz(bank_path)
    bank_position = {int(track): row for row, track in enumerate(bank.track_ids)}
    maplets, _maplet_metadata = load_local_maplet_support_index_npz(maplet_path)
    if set(map(int, bank.track_ids)) != set(map(int, maplets.anchor_track_ids)):
        raise ValueError("maplet index does not cover exactly the projected landmark bank")
    cluster_ids = build_disjoint_maplet_cluster_ids(maplets)
    cluster_by_track = {
        int(track): int(cluster)
        for track, cluster in zip(maplets.anchor_track_ids, cluster_ids)
    }

    rows: list[dict[str, object]] = []
    offset = 0
    for hypothesis_path, score_path in zip(hypothesis_paths, score_paths):
        with np.load(score_path, allow_pickle=False) as score:
            _require_arrays(
                score,
                required={
                    "metadata_json",
                    "query_ids",
                    "split_names",
                    "evaluation_labels",
                    "hypothesis_indices",
                    "independent_score_top1",
                    "independent_selection_scores",
                },
                context=f"score artifact {score_path}",
            )
            score_metadata = _metadata(score, context=f"score artifact {score_path}")
            if (
                score_metadata.get("contains_target_fields") is not False
                or score_metadata.get("pose_or_ground_truth_used_for_scoring") is not False
            ):
                raise ValueError(f"{score_path}: score artifact is not target-free")
            selected = np.asarray(score["independent_score_top1"], dtype=bool)
            score_query_ids = score["query_ids"].astype(str)
            score_split_names = score["split_names"].astype(str)
            score_labels = score["evaluation_labels"].astype(str)
            score_indices = np.asarray(score["hypothesis_indices"], dtype=np.int64)
            independent_scores = np.asarray(
                score["independent_selection_scores"], dtype=np.float64
            )
        with np.load(hypothesis_path, allow_pickle=False) as hypothesis:
            _require_arrays(
                hypothesis,
                required={
                    "metadata_json",
                    "query_ids",
                    "split_names",
                    "evaluation_labels",
                    "hypothesis_indices",
                    "sample_token_indices",
                    "sample_track_ids",
                    "translation_information_min_eigenvalues",
                    "translation_information_conditions",
                    "bearing_max_angles_deg",
                    "camera_depth_span_ratios",
                    "xyz_third_singular_ratios",
                },
                context=f"hypothesis artifact {hypothesis_path}",
            )
            hypothesis_metadata = _metadata(
                hypothesis, context=f"hypothesis artifact {hypothesis_path}"
            )
            if hypothesis_metadata.get("contains_target_fields") is not False:
                raise ValueError(f"{hypothesis_path}: hypothesis artifact is not inference-only")
            hypothesis_inputs = hypothesis_metadata.get("inputs")
            if not isinstance(hypothesis_inputs, dict):
                raise ValueError(f"{hypothesis_path}: hypothesis inputs are missing")
            for key, expected in (
                ("candidate_artifact_sha256", candidate_hash),
                ("proposals_sha256", proposal_hash),
                ("projected_landmark_bank_sha256", bank_hash),
                ("maplet_support_index_sha256", maplet_hash),
            ):
                if str(hypothesis_inputs.get(key)) != expected:
                    raise ValueError(f"{hypothesis_path}: {key} differs from the audit input")
            shard_count = len(selected)
            target_slice = slice(offset, offset + shard_count)
            if target_slice.stop > target_count:
                raise ValueError("score rows exceed the target artifact")
            if (
                not np.array_equal(score_query_ids, target_query_ids[target_slice])
                or not np.array_equal(score_split_names, target_split_names[target_slice])
                or not np.array_equal(score_labels, target_labels[target_slice])
                or not np.array_equal(score_indices, target_hypothesis_indices[target_slice])
            ):
                raise ValueError("target and inference score identities differ")
            if any(
                np.asarray(values).shape != (shard_count,)
                for values in (
                    score_query_ids,
                    score_split_names,
                    score_labels,
                    score_indices,
                    independent_scores,
                )
            ):
                raise ValueError(f"{score_path}: score arrays are not row-aligned")
            if np.any(~np.isfinite(independent_scores)):
                raise ValueError(f"{score_path}: independent score is non-finite")
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
            if len(score_keys) != len(set(score_keys)):
                raise ValueError("score hypothesis identity keys are not unique")
            if set(score_keys).difference(hypothesis_position):
                raise ValueError(
                    "score contains hypotheses absent from its frozen source shard"
                )
            score_rows_by_query: dict[tuple[str, str, str], list[int]] = defaultdict(list)
            for score_row, key in enumerate(
                zip(
                    score_query_ids.tolist(),
                    score_split_names.tolist(),
                    score_labels.tolist(),
                )
            ):
                score_rows_by_query[key].append(int(score_row))
            scope_diagnostics_by_score_row: dict[int, dict[str, object]] = {}
            for scope_key, local_rows in score_rows_by_query.items():
                local = np.asarray(local_rows, dtype=np.int64)
                selected_rows = local[selected[local]]
                if len(selected_rows) != 1:
                    raise ValueError(
                        f"{score_path}: {scope_key[0]} must have exactly one selected score"
                    )
                global_rows = offset + local
                scope_translation = translation_errors[global_rows]
                scope_rotation = rotation_errors[global_rows]
                scope_success = (
                    (scope_translation <= float(args.success_threshold_m))
                    & (
                        scope_rotation
                        <= float(args.success_rotation_threshold_deg)
                    )
                )
                score_order = local[
                    np.argsort(-independent_scores[local], kind="stable")
                ]
                best_position = int(
                    np.lexsort((scope_rotation, scope_translation))[0]
                )
                best_score_row = int(local[best_position])
                best_rank = int(
                    np.flatnonzero(score_order == best_score_row)[0] + 1
                )
                success_order = scope_success[
                    np.argsort(-independent_scores[local], kind="stable")
                ]
                success_positions = np.flatnonzero(success_order)
                first_success_rank = (
                    None
                    if len(success_positions) == 0
                    else int(success_positions[0] + 1)
                )
                scope_diagnostics_by_score_row[int(selected_rows[0])] = {
                    "score_scope_hypothesis_count": int(len(local)),
                    "score_scope_success_pose_count": int(np.count_nonzero(scope_success)),
                    "score_scope_success_pose_available": bool(np.any(scope_success)),
                    "score_scope_first_success_score_rank": first_success_rank,
                    "score_scope_best_translation_m": float(
                        scope_translation[best_position]
                    ),
                    "score_scope_best_translation_rotation_deg": float(
                        scope_rotation[best_position]
                    ),
                    "score_scope_best_translation_score_rank": best_rank,
                }
            for score_row in np.flatnonzero(selected):
                local_row = hypothesis_position[score_keys[int(score_row)]]
                global_row = offset + int(score_row)
                sample_count = 0
                available_count = 0
                identity_count = 0
                maplet_comparisons = 0
                maplet_mismatches = 0
                actual_candidate_column_count = 0
                deltas = []
                sample_rows = np.asarray(
                    hypothesis["sample_token_indices"][local_row], dtype=np.int64
                )
                sample_tracks = np.asarray(
                    hypothesis["sample_track_ids"][local_row], dtype=np.int64
                )
                if sample_rows.shape != sample_tracks.shape:
                    raise ValueError("hypothesis sample token/track arrays are not aligned")
                for proposal_row, selected_track in zip(sample_rows, sample_tracks):
                    proposal_row = int(proposal_row)
                    selected_track = int(selected_track)
                    if proposal_row < 0 or selected_track < 0:
                        if (proposal_row < 0) != (selected_track < 0):
                            raise ValueError("hypothesis sample token/track padding is inconsistent")
                        continue
                    sample_count += 1
                    tracks, residuals = _actual_sample_candidates(
                        proposal_row=proposal_row,
                        selected_track=selected_track,
                        expected_query_id=str(score_query_ids[score_row]),
                        proposal_query_ids=proposal_query_ids,
                        candidate_track_ids=candidate_track_ids,
                        candidate_residuals=candidate_residuals,
                        candidate_position_by_proposal_row=candidate_position_by_proposal_row,
                        candidate_selected_columns=candidate_selected_columns,
                    )
                    actual_candidate_column_count += int(len(tracks))
                    valid = (tracks >= 0) & np.isfinite(residuals)
                    correct = valid & (residuals <= float(args.identity_threshold_px))
                    selected_columns = np.flatnonzero(tracks == selected_track)
                    if np.any(
                        residuals[selected_columns]
                        <= float(args.identity_threshold_px)
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
                scope_diagnostics = scope_diagnostics_by_score_row.get(int(score_row))
                if scope_diagnostics is None:
                    raise RuntimeError("selected score lacks scoped oracle diagnostics")
                row = {
                    "query_id": str(score_query_ids[score_row]),
                    "split_name": str(hypothesis["split_names"][local_row]),
                    "hypothesis_index": int(score_indices[score_row]),
                    "translation_error_m": translation_error,
                    "rotation_error_deg": float(rotation_errors[global_row]),
                    "sample_count": int(sample_count),
                    "actual_candidate_column_count": int(actual_candidate_column_count),
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
                    **scope_diagnostics,
                }
                row["failure_class"] = _classify(
                    translation_error_m=translation_error,
                    rotation_error_deg=float(rotation_errors[global_row]),
                    success_threshold_m=float(args.success_threshold_m),
                    success_rotation_threshold_deg=float(
                        args.success_rotation_threshold_deg
                    ),
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
            if str(row["failure_class"]) != "success_10cm"
        ]
        counts = Counter(str(row["failure_class"]) for row in split_rows)
        failure_counts = Counter(str(row["failure_class"]) for row in failures)
        errors = np.asarray(
            [float(row["translation_error_m"]) for row in split_rows], dtype=np.float64
        )
        scope_success_available = np.asarray(
            [bool(row["score_scope_success_pose_available"]) for row in split_rows],
            dtype=bool,
        )
        scope_by_class: dict[str, dict[str, object]] = {}
        for failure_class in sorted(counts):
            class_rows = [
                row for row in split_rows if str(row["failure_class"]) == failure_class
            ]
            scope_by_class[failure_class] = {
                "query_count": int(len(class_rows)),
                "success_pose_available_fraction": float(
                    np.mean(
                        [bool(row["score_scope_success_pose_available"]) for row in class_rows]
                    )
                ),
                "best_translation_m": _finite_distribution(
                    [float(row["score_scope_best_translation_m"]) for row in class_rows]
                ),
                "first_success_score_rank": _finite_distribution(
                    [row["score_scope_first_success_score_rank"] for row in class_rows]
                ),
                "best_translation_score_rank": _finite_distribution(
                    [row["score_scope_best_translation_score_rank"] for row in class_rows]
                ),
            }
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
            "scored_scope_oracle": {
                "success_pose_available_fraction": float(
                    np.mean(scope_success_available)
                ),
                "best_translation_m": _finite_distribution(
                    [float(row["score_scope_best_translation_m"]) for row in split_rows]
                ),
                "first_success_score_rank": _finite_distribution(
                    [row["score_scope_first_success_score_rank"] for row in split_rows]
                ),
                "by_selected_failure_class": scope_by_class,
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
            "hypothesis_artifact_sha256": hypothesis_hashes,
            "score_artifact_sha256": score_hashes,
            "target_artifact_sha256": file_sha256_short(target_path),
            "candidate_artifact_sha256": candidate_hash,
            "proposals_sha256": proposal_hash,
            "projected_landmark_bank_sha256": bank_hash,
            "maplet_index_sha256": maplet_hash,
        },
        "contracts": {
            "target_hashes_match_supplied_inference_shards": True,
            "hypotheses_and_scores_are_target_free": True,
            "candidate_layout_matches_hypotheses": True,
            "sample_token_indices_are_global_proposal_rows": True,
            "sample_query_ids_match_hypothesis_query_ids": True,
            "frozen_candidate_layout_columns_used_for_identity_audit": True,
            "per_hypothesis_sampling_candidate_limit_not_serialized": True,
            "target_free_score_scope_is_allowed_hypothesis_subset": True,
        },
        "metrics": metrics,
        "outputs": {"per_query": str(rows_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
