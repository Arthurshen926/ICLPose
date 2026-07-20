"""Build a frozen, target-free real-image support pool for pose diagnostics.

The pool is derived only from held-out global top-L candidate tracks and their
already fixed SfM support observations.  It is not image retrieval or a
submap: every selected support image is an observation of a pre-existing
global candidate, and the later scorer never reselects it per pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    FrozenSupportAlignmentLayout,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_layout_features", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--fixed_candidate_prior_overlay", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--rank_band_ends",
        default="5,10,20",
        help="exclusive global top-L rank ends used to diversify fixed support images",
    )
    parser.add_argument(
        "--support_images_per_rank_band",
        default="3,3,2",
        help="number of distinct fixed support images retained from each rank band",
    )
    parser.add_argument("--tracks_per_support_image", type=int, default=96)
    parser.add_argument("--support_spatial_grid_size", type=int, default=8)
    parser.add_argument("--max_support_views_per_track", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _parse_positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not output or any(item <= 0 for item in output):
        raise ValueError(f"{name} must contain positive integers")
    return output


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = (
            {}
            if "metadata_json" not in payload.files
            else json.loads(str(payload["metadata_json"].item()))
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json is not an object")
    return arrays, metadata


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()[:16]


def _load_frozen_layout(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, metadata = _load_npz(path)
    required = {
        "source_row_indices",
        "query_ids",
        "split_names",
        "candidate_track_ids",
        "candidate_view_valid",
        "candidate_support_image_ids",
    }
    if not required.issubset(arrays):
        raise ValueError(f"frozen layout lacks {sorted(required - set(arrays))}")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or bool(metadata.get("pose_or_ground_truth_used", False))
        or bool(metadata.get("image_retrieval_or_submap_used", False))
        or bool(metadata.get("render", False))
    ):
        raise ValueError("frozen layout violates the target-free real-image protocol")
    if not str(metadata.get("source_row_selection", "")).startswith("heldout_"):
        raise ValueError("frozen layout does not declare held-out query rows")
    rows = np.asarray(arrays["source_row_indices"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    splits = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    valid = np.asarray(arrays["candidate_view_valid"], dtype=bool)
    image_ids = np.asarray(arrays["candidate_support_image_ids"]).astype(str)
    if (
        len(rows) == 0
        or len(np.unique(rows)) != len(rows)
        or query_ids.shape != splits.shape != (len(rows),)
        or tracks.ndim != 2
        or valid.shape[:2] != tracks.shape
        or image_ids.shape != valid.shape
        or np.any((tracks >= 0) & ~np.any(valid, axis=2))
        or np.any(valid & (image_ids == ""))
    ):
        raise ValueError("frozen layout arrays are incompatible")
    return {
        "source_row_indices": rows,
        "query_ids": query_ids,
        "split_names": splits,
        "candidate_track_ids": tracks,
        "candidate_view_valid": valid,
        "candidate_support_image_ids": image_ids,
    }, metadata


def _load_fixed_candidate_probabilities(
    path: Path,
    *,
    proposals: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    arrays, metadata = _load_npz(path)
    required = {"candidate_track_ids", "candidate_probabilities", "null_probabilities"}
    if set(arrays) != required:
        raise ValueError("candidate prior overlay fields differ from contract")
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or bool(metadata.get("pose_or_ground_truth_used", False))
        or bool(metadata.get("image_retrieval_or_submap_used", False))
        or bool(metadata.get("render", False))
    ):
        raise ValueError("candidate prior overlay is not target-free")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    if (
        not np.array_equal(tracks, proposal_tracks)
        or probabilities.shape != tracks.shape
        or null.shape != (len(tracks),)
        or np.any(~np.isfinite(probabilities))
        or np.any(~np.isfinite(null))
        or np.any(probabilities < 0.0)
        or np.any(null < 0.0)
    ):
        raise ValueError("candidate prior overlay is not proposal-aligned")
    valid = tracks >= 0
    if np.any(np.abs(probabilities[~valid]) > 1e-6) or np.any(
        np.abs(np.sum(np.where(valid, probabilities, 0.0), axis=1) + null - 1.0) > 2e-4
    ):
        raise ValueError("candidate prior overlay does not preserve explicit null mass")
    return probabilities, null, metadata


def _fixed_fit_tracks_by_query(
    *,
    proposals: Mapping[str, np.ndarray],
    candidate_artifact: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    selected_rows = np.asarray(candidate_artifact.get("selected_rows"), dtype=np.int64).reshape(-1)
    query_ids = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    if (
        selected_rows.size == 0
        or len(np.unique(selected_rows)) != len(selected_rows)
        or np.any((selected_rows < 0) | (selected_rows >= len(query_ids)))
        or tracks.shape[0] != len(query_ids)
    ):
        raise ValueError("candidate artifact fit rows are invalid")
    output: dict[str, np.ndarray] = {}
    for query_id in sorted(set(query_ids[selected_rows].tolist())):
        rows = selected_rows[query_ids[selected_rows] == query_id]
        values = tracks[rows].reshape(-1)
        output[str(query_id)] = np.unique(values[values >= 0])
    return output


def select_rank_band_support_images(
    *,
    candidate_probabilities: np.ndarray,
    candidate_view_valid: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    rank_band_ends: Sequence[int],
    images_per_rank_band: Sequence[int],
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Pick a deterministic, rank-diverse support-image pool before pose scoring."""

    probability = np.asarray(candidate_probabilities, dtype=np.float64)
    valid = np.asarray(candidate_view_valid, dtype=bool)
    image_ids = np.asarray(candidate_support_image_ids).astype(str)
    ends = tuple(int(value) for value in rank_band_ends)
    counts = tuple(int(value) for value in images_per_rank_band)
    if (
        probability.ndim != 2
        or valid.shape[:2] != probability.shape
        or image_ids.shape != valid.shape
        or len(ends) != len(counts)
        or not ends
        or any(end <= 0 for end in ends)
        or any(right <= left for left, right in zip((0, *ends[:-1]), ends))
        or ends[-1] > probability.shape[1]
        or any(count <= 0 for count in counts)
    ):
        raise ValueError("rank-band support selection inputs are invalid")
    total_scores: dict[str, float] = {}
    band_scores: list[dict[str, float]] = [dict() for _ in ends]
    starts = (0, *ends[:-1])
    for row in range(probability.shape[0]):
        for band_index, (start, end) in enumerate(zip(starts, ends)):
            for rank in range(int(start), int(end)):
                views = np.flatnonzero(valid[row, rank])
                if len(views) == 0:
                    continue
                mass = float(probability[row, rank]) / float(len(views))
                for view in views.tolist():
                    image_id = str(image_ids[row, rank, view])
                    total_scores[image_id] = total_scores.get(image_id, 0.0) + mass
                    band_scores[band_index][image_id] = (
                        band_scores[band_index].get(image_id, 0.0) + mass
                    )
    selected: list[str] = []
    selected_set: set[str] = set()
    for score_map, count in zip(band_scores, counts):
        ordered = sorted(score_map, key=lambda image_id: (-score_map[image_id], image_id))
        selected_from_band = 0
        for image_id in ordered:
            if image_id in selected_set:
                continue
            selected.append(image_id)
            selected_set.add(image_id)
            selected_from_band += 1
            if selected_from_band >= int(count):
                break
    if not selected:
        raise ValueError("fixed global candidates have no support images")
    return tuple(selected), total_scores


def _select_spatially_diverse_image_observations(
    *,
    geometry: SupportObservationGeometryIndex,
    image_id: str,
    xyz_by_track: Mapping[int, np.ndarray],
    excluded_tracks: set[int],
    image_width: int,
    image_height: int,
    grid_size: int,
    maximum_count: int,
) -> np.ndarray:
    """Select real SfM observations by fixed image-plane coverage and quality."""

    image_slice = geometry.image_slice(str(image_id))
    rows = np.arange(int(image_slice.start), int(image_slice.stop), dtype=np.int64)
    tracks = geometry.track_ids[image_slice]
    xy = geometry.xy[image_slice]
    errors = geometry.reprojection_errors[image_slice]
    if len(rows) == 0:
        return rows
    has_xyz = np.fromiter((int(track) in xyz_by_track for track in tracks), dtype=bool)
    valid = (
        has_xyz
        & ~np.isin(tracks, np.asarray(sorted(excluded_tracks), dtype=np.int64))
        & np.all(np.isfinite(xy), axis=1)
        & np.isfinite(errors)
        & (errors >= 0.0)
    )
    rows = rows[valid]
    tracks = tracks[valid]
    xy = xy[valid]
    errors = errors[valid]
    if len(rows) == 0:
        return rows
    columns = np.minimum(
        (xy[:, 0] * int(grid_size) / float(image_width)).astype(np.int64), int(grid_size) - 1
    )
    grid_rows = np.minimum(
        (xy[:, 1] * int(grid_size) / float(image_height)).astype(np.int64), int(grid_size) - 1
    )
    cells = np.clip(grid_rows, 0, int(grid_size) - 1) * int(grid_size) + np.clip(
        columns, 0, int(grid_size) - 1
    )
    quota = max(1, int(np.ceil(float(maximum_count) / float(grid_size * grid_size))))
    selected: list[int] = []
    selected_set: set[int] = set()
    for cell in range(int(grid_size) * int(grid_size)):
        local = np.flatnonzero(cells == cell)
        ordered = sorted(local.tolist(), key=lambda index: (float(errors[index]), int(tracks[index])))
        for index in ordered[:quota]:
            selected.append(int(rows[index]))
            selected_set.add(int(rows[index]))
    if len(selected) < int(maximum_count):
        ordered = sorted(
            range(len(rows)), key=lambda index: (float(errors[index]), int(tracks[index]))
        )
        for index in ordered:
            row = int(rows[index])
            if row in selected_set:
                continue
            selected.append(row)
            selected_set.add(row)
            if len(selected) >= int(maximum_count):
                break
    return np.asarray(selected[: int(maximum_count)], dtype=np.int64)


def build_layout(
    *,
    frozen_layout_features: Path,
    proposals_path: Path,
    candidate_artifact_path: Path,
    fixed_candidate_prior_overlay: Path,
    support_geometry_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    rank_band_ends: Sequence[int],
    images_per_rank_band: Sequence[int],
    tracks_per_support_image: int,
    support_spatial_grid_size: int,
    max_support_views_per_track: int,
) -> tuple[FrozenSupportAlignmentLayout, dict[str, object]]:
    if (
        int(tracks_per_support_image) <= 0
        or int(support_spatial_grid_size) <= 0
        or int(max_support_views_per_track) <= 0
    ):
        raise ValueError("support-pool selection parameters must be positive")
    layout, layout_metadata = _load_frozen_layout(frozen_layout_features)
    proposals, _proposals_metadata = _load_npz(proposals_path)
    if not {"query_ids", "candidate_track_ids"}.issubset(proposals):
        raise ValueError("proposals lack query IDs or candidate tracks")
    if str(layout_metadata.get("proposals_sha256", "")) != file_sha256_short(proposals_path):
        raise ValueError("frozen layout references different proposals")
    candidate_artifact, candidate_metadata = _load_npz(candidate_artifact_path)
    if "selected_rows" not in candidate_artifact:
        raise ValueError("candidate artifact lacks selected rows")
    if str(layout_metadata.get("candidate_artifact_sha256", "")) != file_sha256_short(
        candidate_artifact_path
    ):
        raise ValueError("frozen layout references a different hypothesis-fit artifact")
    if candidate_metadata.get("contains_ground_truth") is not False:
        raise ValueError("candidate artifact is not inference-only")
    probabilities, _null, overlay_metadata = _load_fixed_candidate_probabilities(
        fixed_candidate_prior_overlay, proposals=proposals
    )
    if str(overlay_metadata.get("proposals_sha256", "")) != file_sha256_short(proposals_path):
        raise ValueError("candidate prior overlay references different proposals")
    if str(layout_metadata.get("projected_landmark_bank_sha256", "")) != file_sha256_short(
        projected_landmark_bank
    ):
        raise ValueError("frozen layout references a different projected landmark bank")
    bank, bank_metadata = load_landmark_index_npz(projected_landmark_bank)
    if len(np.unique(bank.track_ids)) != len(bank.track_ids):
        raise ValueError("support layout requires one canonical XYZ row per physical track")
    xyz_by_track = {
        int(track_id): np.asarray(xyz, dtype=np.float64)
        for track_id, xyz in zip(bank.track_ids.tolist(), bank.xyz)
    }
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(
        support_geometry_index
    )
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("support geometry does not use real SfM observation coordinates")
    model_dir = Path(colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    image_camera_ids = read_colmap_image_camera_ids_binary(model_dir / "images.bin")
    proposal_query_ids = np.asarray(proposals["query_ids"]).astype(str).reshape(-1)
    proposal_tracks = np.asarray(proposals["candidate_track_ids"], dtype=np.int64)
    if proposal_tracks.shape != probabilities.shape or proposal_tracks.shape[0] != len(proposal_query_ids):
        raise ValueError("proposals and fixed candidate probabilities differ")
    source_rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
    if np.any((source_rows < 0) | (source_rows >= len(proposal_query_ids))) or not np.array_equal(
        layout["query_ids"], proposal_query_ids[source_rows]
    ) or not np.array_equal(layout["candidate_track_ids"], proposal_tracks[source_rows]):
        raise ValueError("frozen support layout is not proposal-aligned")
    fit_tracks_by_query = _fixed_fit_tracks_by_query(
        proposals=proposals, candidate_artifact=candidate_artifact
    )
    query_ids = tuple(sorted(set(np.asarray(layout["query_ids"]).astype(str).tolist())))
    query_splits: list[str] = []
    query_observation_offsets = [0]
    query_track_offsets = [0]
    track_observation_offsets = [0]
    observation_track_ids: list[int] = []
    observation_xyz: list[np.ndarray] = []
    support_image_ids: list[str] = []
    support_xy: list[np.ndarray] = []
    support_image_scores: list[float] = []
    support_reprojection_errors: list[float] = []
    support_image_counts: list[int] = []
    heldout_anchor_counts: list[int] = []
    fit_exclusion_counts: list[int] = []
    for query_id in query_ids:
        local_rows = np.flatnonzero(np.asarray(layout["query_ids"]).astype(str) == query_id)
        local_splits = np.unique(np.asarray(layout["split_names"]).astype(str)[local_rows])
        if len(local_splits) != 1:
            raise ValueError(f"{query_id}: frozen layout mixes splits")
        source = source_rows[local_rows]
        selected_images, image_score_map = select_rank_band_support_images(
            candidate_probabilities=probabilities[source],
            candidate_view_valid=np.asarray(layout["candidate_view_valid"])[local_rows],
            candidate_support_image_ids=np.asarray(layout["candidate_support_image_ids"])[local_rows],
            rank_band_ends=rank_band_ends,
            images_per_rank_band=images_per_rank_band,
        )
        anchor_tracks = np.unique(
            np.asarray(layout["candidate_track_ids"], dtype=np.int64)[local_rows].reshape(-1)
        )
        anchor_tracks = anchor_tracks[anchor_tracks >= 0]
        fit_tracks = fit_tracks_by_query.get(query_id)
        if fit_tracks is None:
            raise ValueError(f"{query_id}: hypothesis-fit artifact has no fit tracks")
        excluded_tracks = set(int(value) for value in np.concatenate([anchor_tracks, fit_tracks]))
        per_track: dict[int, list[tuple[str, np.ndarray, float, float]]] = {}
        for support_image_id in selected_images:
            if support_image_id == query_id:
                raise ValueError(f"{query_id}: support pool contains the query image")
            position = geometry.image_position(support_image_id)
            if position is None:
                raise ValueError(f"support image {support_image_id!r} is absent from geometry")
            camera_id = image_camera_ids.get(support_image_id)
            if camera_id is None or camera_id not in cameras:
                raise ValueError(f"support image {support_image_id!r} has no camera ownership")
            camera = cameras[camera_id]
            rows = _select_spatially_diverse_image_observations(
                geometry=geometry,
                image_id=support_image_id,
                xyz_by_track=xyz_by_track,
                excluded_tracks=excluded_tracks,
                image_width=int(camera.width),
                image_height=int(camera.height),
                grid_size=int(support_spatial_grid_size),
                maximum_count=int(tracks_per_support_image),
            )
            for row in rows.tolist():
                track_id = int(geometry.track_ids[row])
                per_track.setdefault(track_id, []).append(
                    (
                        support_image_id,
                        np.asarray(geometry.xy[row], dtype=np.float32),
                        float(image_score_map[support_image_id]),
                        float(geometry.reprojection_errors[row]),
                    )
                )
        if not per_track:
            raise ValueError(f"{query_id}: fixed support pool has no independent tracks")
        query_splits.append(str(local_splits[0]))
        support_image_counts.append(len(selected_images))
        heldout_anchor_counts.append(len(anchor_tracks))
        fit_exclusion_counts.append(len(fit_tracks))
        for track_id in sorted(per_track):
            views = sorted(
                per_track[track_id],
                key=lambda item: (-item[2], item[3], item[0]),
            )[: int(max_support_views_per_track)]
            for image_id, xy, image_score, reprojection_error in views:
                observation_track_ids.append(int(track_id))
                observation_xyz.append(xyz_by_track[int(track_id)])
                support_image_ids.append(image_id)
                support_xy.append(xy)
                support_image_scores.append(float(image_score))
                support_reprojection_errors.append(float(reprojection_error))
            track_observation_offsets.append(len(observation_track_ids))
        query_observation_offsets.append(len(observation_track_ids))
        query_track_offsets.append(len(track_observation_offsets) - 1)
    metadata: dict[str, object] = {
        "format": POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT,
        "version": "fixed_candidate_support_image_pool_v1",
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "support_pool_protocol": {
            "source": "heldout_global_topl_candidate_support_observation_union_v1",
            "support_images_fixed_before_pose_scoring": True,
            "support_tracks_fixed_before_pose_scoring": True,
            "candidate_anchor_tracks_excluded_from_context_evidence": True,
            "hypothesis_fit_tracks_excluded_from_context_evidence": True,
            "pose_local_support_reselection": False,
            "support_observation_coordinate_source": "sfm_observation_xy",
            "image_retrieval": False,
            "submap": False,
            "render": False,
        },
        "selection": {
            "rank_band_ends": [int(value) for value in rank_band_ends],
            "support_images_per_rank_band": [int(value) for value in images_per_rank_band],
            "tracks_per_support_image": int(tracks_per_support_image),
            "support_spatial_grid_size": int(support_spatial_grid_size),
            "max_support_views_per_track": int(max_support_views_per_track),
            "track_selection": "fixed_image_grid_quota_then_reprojection_error_v1",
            "view_marginalization": "fixed_per_track_support_observation_log_mixture_v1",
        },
        "inputs": {
            "frozen_layout_features": str(frozen_layout_features),
            "frozen_layout_features_sha256": file_sha256_short(frozen_layout_features),
            "proposals": str(proposals_path),
            "proposals_sha256": file_sha256_short(proposals_path),
            "candidate_artifact": str(candidate_artifact_path),
            "candidate_artifact_sha256": file_sha256_short(candidate_artifact_path),
            "fixed_candidate_prior_overlay": str(fixed_candidate_prior_overlay),
            "fixed_candidate_prior_overlay_sha256": file_sha256_short(fixed_candidate_prior_overlay),
            "support_geometry_index": str(support_geometry_index),
            "support_geometry_index_sha256": file_sha256_short(support_geometry_index),
            "projected_landmark_bank": str(projected_landmark_bank),
            "projected_landmark_bank_sha256": file_sha256_short(projected_landmark_bank),
            "bank_descriptor_space_id": bank_metadata.get("descriptor_space_id"),
            "colmap_model_dir": str(model_dir),
            "colmap_cameras_bin_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        },
        "frozen_layout_source_rows_sha256": _array_hash(source_rows),
        "query_count": len(query_ids),
        "support_observation_count": len(observation_track_ids),
        "support_track_count": len(track_observation_offsets) - 1,
        "support_image_counts_per_query": support_image_counts,
        "heldout_anchor_track_counts_per_query": heldout_anchor_counts,
        "fit_track_exclusion_counts_per_query": fit_exclusion_counts,
    }
    output = FrozenSupportAlignmentLayout(
        query_ids=np.asarray(query_ids, dtype=np.str_),
        split_names=np.asarray(query_splits, dtype=np.str_),
        query_observation_offsets=np.asarray(query_observation_offsets, dtype=np.int64),
        query_track_offsets=np.asarray(query_track_offsets, dtype=np.int64),
        track_observation_offsets=np.asarray(track_observation_offsets, dtype=np.int64),
        observation_track_ids=np.asarray(observation_track_ids, dtype=np.int64),
        observation_xyz=np.asarray(observation_xyz, dtype=np.float64),
        support_image_ids=np.asarray(support_image_ids, dtype=np.str_),
        support_xy=np.asarray(support_xy, dtype=np.float32),
        support_image_scores=np.asarray(support_image_scores, dtype=np.float32),
        support_reprojection_errors=np.asarray(support_reprojection_errors, dtype=np.float32),
        metadata=metadata,
    )
    return output, metadata


def save_layout(layout: FrozenSupportAlignmentLayout, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_ids=layout.query_ids,
        split_names=layout.split_names,
        query_observation_offsets=layout.query_observation_offsets,
        query_track_offsets=layout.query_track_offsets,
        track_observation_offsets=layout.track_observation_offsets,
        observation_track_ids=layout.observation_track_ids,
        observation_xyz=layout.observation_xyz,
        support_image_ids=layout.support_image_ids,
        support_xy=layout.support_xy,
        support_image_scores=layout.support_image_scores,
        support_reprojection_errors=layout.support_reprojection_errors,
        metadata_json=np.asarray(json.dumps(dict(layout.metadata), sort_keys=True)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    summary_path = Path(args.summary_json)
    if (output.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite a support-alignment layout")
    ends = _parse_positive_ints(args.rank_band_ends, name="rank_band_ends")
    per_band = _parse_positive_ints(
        args.support_images_per_rank_band, name="support_images_per_rank_band"
    )
    if len(ends) != len(per_band) or any(right <= left for left, right in zip((0, *ends[:-1]), ends)):
        raise ValueError("rank bands and image counts are incompatible")
    layout, metadata = build_layout(
        frozen_layout_features=Path(args.frozen_layout_features),
        proposals_path=Path(args.proposals),
        candidate_artifact_path=Path(args.candidate_artifact),
        fixed_candidate_prior_overlay=Path(args.fixed_candidate_prior_overlay),
        support_geometry_index=Path(args.support_geometry_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        rank_band_ends=ends,
        images_per_rank_band=per_band,
        tracks_per_support_image=int(args.tracks_per_support_image),
        support_spatial_grid_size=int(args.support_spatial_grid_size),
        max_support_views_per_track=int(args.max_support_views_per_track),
    )
    save_layout(layout, output)
    summary = {
        "stage": "build_pose_conditioned_support_alignment_layout",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "query_count": layout.query_count,
        "support_observation_count": layout.observation_count,
        "support_track_count": layout.track_count,
        "support_images_median_per_query": float(
            np.median(np.asarray(metadata["support_image_counts_per_query"], dtype=np.float64))
        ),
        "protocol": metadata["support_pool_protocol"],
        "selection": metadata["selection"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
