"""Score frozen pose hypotheses with fixed multi-observation real-image evidence.

This is a diagnostic scorer, not a production pose selector.  It never reads
COLMAP query poses or labels.  Each query's support images and independent
tracks are fixed by a separate layout; a hypothesis only changes 3D-to-query
projection locations inside a fixed full-image feature denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ContextPositionLikelihoodMaps,
    FrozenSupportAlignmentLayout,
    ImageGridFeatureSource,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_VERSION,
    aggregate_view_log_ratios,
    build_fixed_affine_maplet_topology,
    build_context_position_likelihood_maps,
    fit_fixed_affine_maplet_transforms_torch,
    project_simple_radial_torch,
    sample_context_position_log_ratios,
    score_affine_warped_maplet_patch_likelihoods,
    score_multiscale_observation_group_position_likelihoods,
    score_multiscale_position_likelihoods,
    summarize_track_log_ratios,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_layout", required=True)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--radio_final_context_cache", required=True)
    parser.add_argument("--radio_final_grid_size", type=int, default=16)
    parser.add_argument("--radio_intermediate_context_cache", required=True)
    parser.add_argument("--radio_intermediate_grid_size", type=int, default=16)
    parser.add_argument("--alike_context_cache", required=True)
    parser.add_argument("--alike_grid_size", type=int, default=64)
    parser.add_argument(
        "--radio_final_pca_context_cache",
        help="optional train-only PCA cache used only by context-NCC profiles",
    )
    parser.add_argument(
        "--radio_intermediate_pca_context_cache",
        help="optional train-only PCA cache used only by context-NCC profiles",
    )
    parser.add_argument(
        "--context_profiles",
        default="",
        help=(
            "semicolon-separated source:odd-window list, e.g. "
            "radio_final_pca64:3,5;radio_intermediate_pca64:5,9,13;alike:5,9"
        ),
    )
    parser.add_argument("--context_temperature", type=float, default=0.10)
    parser.add_argument("--context_template_batch_size", type=int, default=64)
    parser.add_argument("--context_minimum_support_fraction", type=float, default=0.75)
    parser.add_argument("--context_minimum_query_overlap_fraction", type=float, default=0.75)
    parser.add_argument("--radio_final_temperature", type=float, default=0.10)
    parser.add_argument("--radio_intermediate_temperature", type=float, default=0.10)
    parser.add_argument("--alike_temperature", type=float, default=0.10)
    parser.add_argument(
        "--coherent_support_block_grids",
        default="",
        help=(
            "comma-separated fixed support-image block grids for same-view "
            "joint evidence, e.g. 1,2,4; empty disables this diagnostic"
        ),
    )
    parser.add_argument(
        "--affine_maplet_profiles",
        default="",
        help=(
            "semicolon-separated source:odd-window profiles for candidate-conditioned "
            "affine maplet evidence, e.g. alike:5;radio_final_pca64:5"
        ),
    )
    parser.add_argument("--affine_maplet_anchor_block_grid", type=int, default=2)
    parser.add_argument("--affine_maplet_anchors_per_block", type=int, default=1)
    parser.add_argument("--affine_maplet_neighbor_radius_px", type=float, default=256.0)
    parser.add_argument("--affine_maplet_neighbor_sigma_px", type=float, default=128.0)
    parser.add_argument("--affine_maplet_max_neighbors", type=int, default=16)
    parser.add_argument("--affine_maplet_min_neighbors", type=int, default=4)
    parser.add_argument("--affine_maplet_max_condition_number", type=float, default=100.0)
    parser.add_argument("--affine_maplet_max_rmse_px", type=float, default=32.0)
    parser.add_argument("--affine_maplet_minimum_patch_fraction", type=float, default=0.75)
    parser.add_argument("--affine_maplet_temperature", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hypothesis_batch_size", type=int, default=16)
    parser.add_argument("--score_splits", default="validation,test")
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _constant_string_column(value: str, count: int) -> np.ndarray:
    """Create a non-truncating, row-aligned Unicode column."""

    if int(count) < 0:
        raise ValueError("string-column count must be non-negative")
    return np.full((int(count),), str(value))


def _load_layout(path: Path) -> FrozenSupportAlignmentLayout:
    required = {
        "query_ids",
        "split_names",
        "query_observation_offsets",
        "query_track_offsets",
        "track_observation_offsets",
        "observation_track_ids",
        "observation_xyz",
        "support_image_ids",
        "support_xy",
        "support_image_scores",
        "support_reprojection_errors",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"support layout lacks fields: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("support layout metadata is not an object")
        if metadata.get("format") != POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT:
            raise ValueError("unsupported support-alignment layout format")
        protocol = metadata.get("support_pool_protocol")
        if (
            metadata.get("contains_ground_truth") is not False
            or metadata.get("contains_target_errors") is not False
            or metadata.get("pose_or_ground_truth_used") is not False
            or metadata.get("image_retrieval_or_submap_used") is not False
            or metadata.get("render") is not False
            or not isinstance(protocol, Mapping)
            or protocol.get("support_images_fixed_before_pose_scoring") is not True
            or protocol.get("support_tracks_fixed_before_pose_scoring") is not True
            or protocol.get("pose_local_support_reselection") is not False
        ):
            raise ValueError("support layout does not satisfy the frozen evidence contract")
        return FrozenSupportAlignmentLayout(
            query_ids=np.asarray(payload["query_ids"]),
            split_names=np.asarray(payload["split_names"]),
            query_observation_offsets=np.asarray(payload["query_observation_offsets"]),
            query_track_offsets=np.asarray(payload["query_track_offsets"]),
            track_observation_offsets=np.asarray(payload["track_observation_offsets"]),
            observation_track_ids=np.asarray(payload["observation_track_ids"]),
            observation_xyz=np.asarray(payload["observation_xyz"]),
            support_image_ids=np.asarray(payload["support_image_ids"]),
            support_xy=np.asarray(payload["support_xy"]),
            support_image_scores=np.asarray(payload["support_image_scores"]),
            support_reprojection_errors=np.asarray(payload["support_reprojection_errors"]),
            metadata=metadata,
        )


def _validate_real_image_metadata(metadata: Mapping[str, object], *, expected_format: str) -> None:
    if str(metadata.get("format", "")) != str(expected_format):
        raise ValueError(f"expected real-image cache format {expected_format!r}")
    if (
        bool(metadata.get("pose_or_ground_truth_used", False))
        or bool(metadata.get("image_retrieval_or_submap_used", False))
        or bool(metadata.get("render", False))
    ):
        raise ValueError("image context cache violates the real-image target-free contract")


def _load_radio_final_source(
    path: Path,
    *,
    grid_size: int,
    image_sizes: Mapping[str, tuple[int, int]],
) -> ImageGridFeatureSource:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError("RADIO-final cache lacks metadata")
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_real_image_metadata(metadata, expected_format="radio_image_multiscale_context_v1")
        key = f"grid{int(grid_size)}_descriptors"
        if key not in payload.files or "image_ids" not in payload.files:
            raise ValueError("RADIO-final cache lacks the requested image grid")
        image_ids = np.asarray(payload["image_ids"]).astype(str)
        sizes: list[tuple[int, int]] = []
        for image_id in image_ids.tolist():
            size = image_sizes.get(str(image_id))
            if size is None:
                raise ValueError(f"RADIO-final cache image {image_id!r} has no COLMAP camera")
            sizes.append(size)
        descriptors = np.asarray(payload[key], dtype=np.float32)
    return ImageGridFeatureSource(
        name="radio_final",
        image_ids=image_ids,
        image_sizes=np.asarray(sizes, dtype=np.int64),
        grid_size=int(grid_size),
        descriptors=descriptors,
        metadata=metadata,
    )


def _load_spatial_source(
    path: Path,
    *,
    name: str,
    expected_format: str,
    grid_size: int,
) -> ImageGridFeatureSource:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {"image_ids", "image_sizes", "metadata_json", f"grid{int(grid_size)}_descriptors"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{name} cache lacks {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_real_image_metadata(metadata, expected_format=expected_format)
        image_ids = np.asarray(payload["image_ids"]).copy()
        image_sizes = np.asarray(payload["image_sizes"]).copy()
        descriptors = np.asarray(
            payload[f"grid{int(grid_size)}_descriptors"], dtype=np.float32
        ).copy()
    return ImageGridFeatureSource(
        name=name,
        image_ids=image_ids,
        image_sizes=image_sizes,
        grid_size=int(grid_size),
        descriptors=descriptors,
        metadata=metadata,
    )


def _load_final_pca_source(path: Path, *, grid_size: int) -> ImageGridFeatureSource:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "image_ids",
            "image_sizes",
            "metadata_json",
            f"grid{int(grid_size)}_descriptors",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"RADIO-final PCA cache lacks {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_real_image_metadata(
            metadata, expected_format="radio_final_context_pca_v1"
        )
        if metadata.get("pca_fit_scope") != "mapping_train_images_only":
            raise ValueError("RADIO-final PCA cache was not fit on mapping train images")
        return ImageGridFeatureSource(
            name="radio_final_pca64",
            image_ids=np.asarray(payload["image_ids"]),
            image_sizes=np.asarray(payload["image_sizes"]),
            grid_size=int(grid_size),
            descriptors=np.asarray(
                payload[f"grid{int(grid_size)}_descriptors"], dtype=np.float32
            ),
            metadata=metadata,
        )


def _load_intermediate_pca_source(path: Path, *, grid_size: int) -> ImageGridFeatureSource:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "image_ids",
            "image_sizes",
            "metadata_json",
            f"grid{int(grid_size)}_descriptors",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"RADIO-intermediate PCA cache lacks {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_real_image_metadata(
            metadata, expected_format="radio_intermediate_image_context_pca_v1"
        )
        if metadata.get("pca_fit_scope") != "mapping_train_images_only":
            raise ValueError("RADIO-intermediate PCA cache was not fit on mapping train images")
        return ImageGridFeatureSource(
            name="radio_intermediate_pca64",
            image_ids=np.asarray(payload["image_ids"]),
            image_sizes=np.asarray(payload["image_sizes"]),
            grid_size=int(grid_size),
            descriptors=np.asarray(
                payload[f"grid{int(grid_size)}_descriptors"], dtype=np.float32
            ),
            metadata=metadata,
        )


def _parse_context_profiles(
    value: str, *, sources: Mapping[str, ImageGridFeatureSource]
) -> tuple[tuple[str, str, int], ...]:
    """Resolve fixed source/window profiles before any query is scored."""

    profiles: list[tuple[str, str, int]] = []
    for group in (item.strip() for item in str(value).split(";") if item.strip()):
        if group.count(":") != 1:
            raise ValueError("each context profile must be source:window[,window]")
        source_name, windows_text = (item.strip() for item in group.split(":", 1))
        source = sources.get(source_name)
        if source is None:
            raise ValueError(f"context profile refers to an unavailable source: {source_name}")
        try:
            windows = tuple(int(item.strip()) for item in windows_text.split(",") if item.strip())
        except ValueError as error:
            raise ValueError("context profile windows must be integers") from error
        if not windows:
            raise ValueError("context profile has no windows")
        for window in windows:
            if window <= 0 or window % 2 != 1 or window > int(source.grid_size):
                raise ValueError(
                    f"{source_name}: context window {window} must be positive, odd, and fit the grid"
                )
            profiles.append((f"{source_name}_context{window}", source_name, int(window)))
    names = [profile[0] for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("context profiles repeat a source/window field")
    return tuple(profiles)


def _parse_coherent_support_block_grids(value: str) -> tuple[int, ...]:
    """Parse fixed support-image spatial partitions without pose-dependent choices."""

    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        return ()
    try:
        grids = tuple(int(item) for item in items)
    except ValueError as error:
        raise ValueError("coherent support block grids must be integers") from error
    if any(grid <= 0 or grid > 64 for grid in grids):
        raise ValueError("coherent support block grids must be in [1, 64]")
    if len(set(grids)) != len(grids):
        raise ValueError("coherent support block grids repeat a value")
    return grids


def _parse_affine_maplet_profiles(
    value: str, *, sources: Mapping[str, ImageGridFeatureSource]
) -> tuple[tuple[str, str, int], ...]:
    """Resolve candidate-conditioned affine patch profiles before pose scoring."""

    profiles: list[tuple[str, str, int]] = []
    for group in (item.strip() for item in str(value).split(";") if item.strip()):
        if group.count(":") != 1:
            raise ValueError("each affine maplet profile must be source:window[,window]")
        source_name, windows_text = (item.strip() for item in group.split(":", 1))
        source = sources.get(source_name)
        if source is None:
            raise ValueError(
                f"affine maplet profile refers to an unavailable source: {source_name}"
            )
        try:
            windows = tuple(int(item.strip()) for item in windows_text.split(",") if item.strip())
        except ValueError as error:
            raise ValueError("affine maplet profile windows must be integers") from error
        if not windows:
            raise ValueError("affine maplet profile has no windows")
        for window in windows:
            if window < 3 or window % 2 != 1 or window > int(source.grid_size):
                raise ValueError(
                    f"{source_name}: affine maplet window {window} must be odd, at least 3, and fit the grid"
                )
            profiles.append((f"affine_maplet_{source_name}_context{window}", source_name, int(window)))
    names = [profile[0] for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("affine maplet profiles repeat a source/window field")
    return tuple(profiles)


def _load_hypotheses(
    paths: Sequence[Path], *, selected_query_ids: set[str]
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Stream only the requested query rows from each inference shard.

    Grouped artifacts also retain large generation diagnostics.  Loading every
    array from all 21 shards before query filtering can consume tens of GB and
    does not strengthen this frozen scoring protocol.
    """

    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "poses_w2c",
    )
    retained: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    metadata: list[dict[str, object]] = []
    for path in paths:
        arrays, item = load_inference_artifact_fields(path, fields)
        if (
            item.get("contains_target_fields") is not False
            or item.get("pose_or_ground_truth_used_for_generation") is not False
        ):
            raise ValueError("hypothesis artifact is not inference-only")
        mask = np.isin(np.asarray(arrays["query_ids"]).astype(str), sorted(selected_query_ids))
        if np.any(mask):
            for field in fields:
                retained[field].append(np.asarray(arrays[field])[mask].copy())
        metadata.append(item)
        del arrays
    if not metadata:
        raise ValueError("at least one grouped hypothesis artifact is required")
    if any(not retained[field] for field in fields):
        raise ValueError("requested support-layout queries are absent from grouped hypotheses")
    arrays = {field: np.concatenate(retained[field], axis=0) for field in fields}
    row_keys = list(
        zip(
            arrays["query_ids"].astype(str).tolist(),
            arrays["evaluation_labels"].astype(str).tolist(),
            arrays["hypothesis_indices"].astype(np.int64).tolist(),
        )
    )
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("grouped hypothesis shards repeat rows")
    return arrays, metadata


def _local_track_groups(layout: FrozenSupportAlignmentLayout, query_index: int) -> tuple[slice, np.ndarray, np.ndarray]:
    observation_slice, track_slice = layout.query_slice(query_index)
    global_offsets = layout.track_observation_offsets[
        int(track_slice.start) : int(track_slice.stop) + 1
    ]
    local_offsets = global_offsets - int(observation_slice.start)
    counts = np.diff(local_offsets).astype(np.int64)
    groups = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if len(groups) != int(observation_slice.stop) - int(observation_slice.start):
        raise RuntimeError("query support-track group offsets are invalid")
    return observation_slice, groups, counts


def _fixed_same_view_observation_groups(
    *,
    support_image_ids: np.ndarray,
    support_xy: np.ndarray,
    geometry_source: ImageGridFeatureSource,
    block_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign observations to frozen ``(support image, spatial block)`` groups.

    The partition uses source-image pixel geometry only.  It is built once per
    query before any candidate pose is projected, and is shared by every
    feature family so the score cannot select a different support view per
    physical track.
    """

    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(support_xy, dtype=np.float32).reshape(-1, 2)
    grid = int(block_grid)
    if len(image_ids) == 0 or len(image_ids) != len(coordinates) or grid <= 0:
        raise ValueError("same-view support grouping inputs are invalid")
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("same-view support grouping has non-finite coordinates")
    keys: list[tuple[str, int, int]] = []
    for image_id, (x, y) in zip(image_ids.tolist(), coordinates.tolist()):
        position = geometry_source.image_position(str(image_id))
        width, height = (
            int(value)
            for value in np.asarray(geometry_source.image_sizes[position], dtype=np.int64)
        )
        if width <= 1 or height <= 1:
            raise ValueError("same-view support grouping has invalid image geometry")
        # Clamp only at the image edge.  COLMAP observations and cache sampling
        # use the same pixel convention, while this makes an exact final pixel
        # deterministically belong to the last block.
        normalized_x = float(np.clip(float(x) / float(width - 1), 0.0, 1.0))
        normalized_y = float(np.clip(float(y) / float(height - 1), 0.0, 1.0))
        column = min(grid - 1, int(np.floor(normalized_x * float(grid))))
        row = min(grid - 1, int(np.floor(normalized_y * float(grid))))
        keys.append((str(image_id), row, column))
    ordered_keys = sorted(set(keys))
    positions = {key: index for index, key in enumerate(ordered_keys)}
    groups = np.asarray([positions[key] for key in keys], dtype=np.int64)
    counts = np.bincount(groups, minlength=len(ordered_keys)).astype(np.int64)
    if len(groups) != len(image_ids) or len(counts) == 0 or np.any(counts <= 0):
        raise RuntimeError("same-view support grouping is not a complete partition")
    return groups, counts


def _is_audit_field(name: str) -> bool:
    """Separate coverage counters from candidate-ranking score fields."""

    return str(name) in {
        "active_support_track_counts",
        "visible_support_observation_counts",
        "fixed_support_track_counts",
    } or str(name).endswith("_active_support_group_counts") or str(name).endswith(
        "_fixed_support_group_counts"
    ) or (
        str(name).startswith("affine_maplet_")
        and str(name).endswith(("_active_counts", "_fixed_counts", "_template_usable_counts"))
    )


def _build_context_maps(
    *,
    query_grid: torch.Tensor,
    source: ImageGridFeatureSource,
    support_image_ids: np.ndarray,
    support_xy: np.ndarray,
    window_size: int,
    temperature: float,
    minimum_support_fraction: float,
    minimum_query_overlap_fraction: float,
    template_batch_size: int,
    device: torch.device,
) -> ContextPositionLikelihoodMaps:
    """Precompute fixed full-query likelihood maps in bounded template batches."""

    if int(template_batch_size) <= 0:
        raise ValueError("context template batch size must be positive")
    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(support_xy, dtype=np.float32).reshape(-1, 2)
    if len(image_ids) == 0 or len(image_ids) != len(coordinates):
        raise ValueError("context support observations are invalid")
    cached_grids: dict[str, torch.Tensor] = {}
    logits: list[torch.Tensor] = []
    valid_cells: list[torch.Tensor] = []
    normalizers: list[torch.Tensor] = []
    usable: list[torch.Tensor] = []
    for begin in range(0, len(image_ids), int(template_batch_size)):
        stop = min(begin + int(template_batch_size), len(image_ids))
        patches, patch_valid = source.context_patches_torch(
            image_ids[begin:stop],
            coordinates[begin:stop],
            window_size=int(window_size),
            device=device,
            source_grid_cache=cached_grids,
        )
        maps = build_context_position_likelihood_maps(
            query_grid=query_grid,
            support_patches=patches,
            support_patch_valid=patch_valid,
            temperature=float(temperature),
            minimum_support_fraction=float(minimum_support_fraction),
            minimum_query_overlap_fraction=float(minimum_query_overlap_fraction),
        )
        logits.append(maps.logits)
        valid_cells.append(maps.valid_cells)
        normalizers.append(maps.log_uniform_normalizers)
        usable.append(maps.template_usable)
    return ContextPositionLikelihoodMaps(
        logits=torch.cat(logits, dim=0),
        valid_cells=torch.cat(valid_cells, dim=0),
        log_uniform_normalizers=torch.cat(normalizers, dim=0),
        template_usable=torch.cat(usable, dim=0),
    )


def _score_query(
    *,
    query_id: str,
    query_index: int,
    poses_w2c: np.ndarray,
    camera: object,
    layout: FrozenSupportAlignmentLayout,
    sources: Mapping[str, ImageGridFeatureSource],
    temperatures: Mapping[str, float],
    context_sources: Mapping[str, ImageGridFeatureSource],
    context_profiles: Sequence[tuple[str, str, int]],
    coherent_support_block_grids: Sequence[int],
    affine_maplet_profiles: Sequence[tuple[str, str, int]],
    affine_maplet_anchor_block_grid: int,
    affine_maplet_anchors_per_block: int,
    affine_maplet_neighbor_radius_px: float,
    affine_maplet_neighbor_sigma_px: float,
    affine_maplet_max_neighbors: int,
    affine_maplet_min_neighbors: int,
    affine_maplet_max_condition_number: float,
    affine_maplet_max_rmse_px: float,
    affine_maplet_minimum_patch_fraction: float,
    affine_maplet_temperature: float,
    context_temperature: float,
    context_template_batch_size: int,
    context_minimum_support_fraction: float,
    context_minimum_query_overlap_fraction: float,
    device: torch.device,
    hypothesis_batch_size: int,
) -> dict[str, np.ndarray]:
    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("pose-conditioned support alignment v1 requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4:
        raise ValueError("SIMPLE_RADIAL camera has invalid parameters")
    observation_slice, observation_groups, group_counts = _local_track_groups(layout, query_index)
    xyz = np.asarray(layout.observation_xyz[observation_slice], dtype=np.float32)
    support_ids = np.asarray(layout.support_image_ids[observation_slice]).astype(str)
    support_xy = np.asarray(layout.support_xy[observation_slice], dtype=np.float32)
    support_scores = np.asarray(layout.support_image_scores[observation_slice], dtype=np.float32)
    group_count = int(len(group_counts))
    expected_size = np.asarray([int(getattr(camera, "width")), int(getattr(camera, "height"))])
    source_support = {
        name: torch.as_tensor(source.sample_numpy(support_ids, support_xy), device=device)
        for name, source in sources.items()
    }
    query_grids: dict[str, torch.Tensor] = {}
    for name, source in sources.items():
        grid, size = source.image_grid(query_id)
        if not np.array_equal(size, expected_size):
            raise ValueError(f"{query_id}: {name} cache image size differs from COLMAP camera")
        query_grids[name] = torch.as_tensor(grid, device=device)
    context_maps: dict[str, ContextPositionLikelihoodMaps] = {}
    for field_name, source_name, window_size in context_profiles:
        source = context_sources[source_name]
        grid, size = source.image_grid(query_id)
        if not np.array_equal(size, expected_size):
            raise ValueError(
                f"{query_id}: {source_name} context cache image size differs from COLMAP camera"
            )
        context_maps[field_name] = _build_context_maps(
            query_grid=torch.as_tensor(grid, dtype=torch.float32, device=device),
            source=source,
            support_image_ids=support_ids,
            support_xy=support_xy,
            window_size=int(window_size),
            temperature=float(context_temperature),
            minimum_support_fraction=float(context_minimum_support_fraction),
            minimum_query_overlap_fraction=float(context_minimum_query_overlap_fraction),
            template_batch_size=int(context_template_batch_size),
            device=device,
        )
    affine_maplet_topology = None
    affine_maplet_templates: dict[
        str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ] = {}
    if affine_maplet_profiles:
        affine_maplet_topology = build_fixed_affine_maplet_topology(
            support_image_ids=support_ids,
            support_xy=support_xy,
            support_reprojection_errors=np.asarray(
                layout.support_reprojection_errors[observation_slice], dtype=np.float32
            ),
            geometry_source=sources["alike"],
            anchor_block_grid=int(affine_maplet_anchor_block_grid),
            anchors_per_block=int(affine_maplet_anchors_per_block),
            neighbor_radius_px=float(affine_maplet_neighbor_radius_px),
            max_neighbors=int(affine_maplet_max_neighbors),
            min_neighbors=int(affine_maplet_min_neighbors),
        )
        anchor_indices = np.asarray(
            affine_maplet_topology.anchor_observation_indices, dtype=np.int64
        )
        anchor_ids = support_ids[anchor_indices]
        anchor_support_xy = support_xy[anchor_indices]
        geometry_sizes = np.asarray(
            [
                sources["alike"].image_sizes[
                    sources["alike"].image_position(image_id)
                ]
                for image_id in anchor_ids.tolist()
            ],
            dtype=np.int64,
        )
        for field_name, source_name, window_size in affine_maplet_profiles:
            source = context_sources[source_name]
            grid, size = source.image_grid(query_id)
            if not np.array_equal(size, expected_size):
                raise ValueError(
                    f"{query_id}: {source_name} affine-maplet cache image size differs from COLMAP camera"
                )
            source_sizes = np.asarray(
                [source.image_sizes[source.image_position(image_id)] for image_id in anchor_ids],
                dtype=np.int64,
            )
            if not np.array_equal(source_sizes, geometry_sizes):
                raise ValueError(
                    f"{query_id}: {source_name} affine-maplet support image geometry differs from ALIKE"
                )
            patches, patch_valid = source.context_patches_torch(
                anchor_ids,
                anchor_support_xy,
                window_size=int(window_size),
                device=device,
            )
            affine_maplet_templates[field_name] = (
                torch.as_tensor(grid, dtype=torch.float32, device=device),
                patches,
                patch_valid,
                torch.as_tensor(source_sizes, dtype=torch.float32, device=device),
                int(source.grid_size),
            )
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    group_tensor = torch.as_tensor(observation_groups, dtype=torch.long, device=device)
    count_tensor = torch.as_tensor(group_counts, dtype=torch.float32, device=device)
    coherent_group_tensors: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for block_grid in coherent_support_block_grids:
        coherent_groups, coherent_counts = _fixed_same_view_observation_groups(
            support_image_ids=support_ids,
            support_xy=support_xy,
            geometry_source=sources["alike"],
            block_grid=int(block_grid),
        )
        coherent_group_tensors[int(block_grid)] = (
            torch.as_tensor(coherent_groups, dtype=torch.long, device=device),
            torch.as_tensor(coherent_counts, dtype=torch.float32, device=device),
            int(len(coherent_counts)),
        )
    output: dict[str, list[np.ndarray]] = {}
    active_track_counts: list[np.ndarray] = []
    visible_observation_counts: list[np.ndarray] = []
    coherent_active_group_counts: dict[int, list[np.ndarray]] = {
        int(block_grid): [] for block_grid in coherent_support_block_grids
    }
    affine_support_xy = (
        None
        if affine_maplet_topology is None
        else torch.as_tensor(support_xy, dtype=torch.float32, device=device)
    )
    affine_maplet_image_groups = None
    affine_support_image_count = None
    affine_support_image_priors = None
    if affine_maplet_topology is not None:
        affine_anchor_indices = np.asarray(
            affine_maplet_topology.anchor_observation_indices, dtype=np.int64
        )
        fixed_images = np.unique(support_ids).astype(str)
        image_positions = {
            image_id: position for position, image_id in enumerate(fixed_images.tolist())
        }
        affine_maplet_image_groups = torch.as_tensor(
            [image_positions[str(image_id)] for image_id in support_ids[affine_anchor_indices]],
            dtype=torch.long,
            device=device,
        )
        image_scores = np.empty((len(fixed_images),), dtype=np.float64)
        for position, image_id in enumerate(fixed_images.tolist()):
            values = np.asarray(support_scores[support_ids == str(image_id)], dtype=np.float64)
            if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError(f"{query_id}: fixed support image has invalid prior mass")
            if not np.allclose(values, values[0], rtol=0.0, atol=1e-5):
                raise ValueError(f"{query_id}: fixed support image prior is not image-constant")
            image_scores[position] = float(values[0])
        affine_support_image_count = int(len(fixed_images))
        affine_support_image_priors = torch.as_tensor(
            image_scores / image_scores.sum(), dtype=torch.float32, device=device
        )
    affine_geometry_active_counts: list[np.ndarray] = []
    affine_profile_active_counts: dict[str, list[np.ndarray]] = {
        field_name: [] for field_name, _source_name, _window in affine_maplet_profiles
    }
    affine_profile_template_usable_counts: dict[str, list[np.ndarray]] = {
        field_name: [] for field_name, _source_name, _window in affine_maplet_profiles
    }
    poses = np.asarray(poses_w2c, dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(poses), int(hypothesis_batch_size)):
            stop = min(begin + int(hypothesis_batch_size), len(poses))
            pose_tensor = torch.as_tensor(poses[begin:stop], dtype=torch.float32, device=device)
            projected, projection_valid = project_simple_radial_torch(
                xyz_tensor,
                pose_tensor,
                focal_length=params[0],
                principal_x=params[1],
                principal_y=params[2],
                radial_k=params[3],
                image_width=int(getattr(camera, "width")),
                image_height=int(getattr(camera, "height")),
            )
            summaries, active_tracks, visible_observations = score_multiscale_position_likelihoods(
                query_grids=query_grids,
                support_descriptors=source_support,
                projected_xy=projected,
                projection_valid=projection_valid,
                image_width=int(getattr(camera, "width")),
                image_height=int(getattr(camera, "height")),
                temperatures=temperatures,
                observation_track_groups=group_tensor,
                track_group_count=group_count,
                group_observation_counts=count_tensor,
            )
            for family, family_summary in summaries.items():
                for statistic, values in family_summary.items():
                    output.setdefault(f"{family}_{statistic}", []).append(
                        values.detach().cpu().numpy().astype(np.float32)
                    )
            if affine_maplet_topology is not None:
                if affine_support_xy is None:  # pragma: no cover - structural guard
                    raise RuntimeError("affine maplet support coordinates are unavailable")
                (
                    affine_matrices,
                    affine_anchors,
                    affine_valid,
                    _affine_neighbor_counts,
                    _affine_rmse,
                ) = fit_fixed_affine_maplet_transforms_torch(
                    support_xy=affine_support_xy,
                    projected_xy=projected,
                    projection_valid=projection_valid,
                    topology=affine_maplet_topology,
                    neighbor_sigma_px=float(affine_maplet_neighbor_sigma_px),
                    minimum_neighbors=int(affine_maplet_min_neighbors),
                    maximum_condition_number=float(affine_maplet_max_condition_number),
                    maximum_rmse_px=float(affine_maplet_max_rmse_px),
                )
                affine_geometry_active_counts.append(
                    affine_valid.sum(dim=1).detach().cpu().numpy().astype(np.int32)
                )
                for field_name, (
                    affine_query_grid,
                    affine_patches,
                    affine_patch_valid,
                    affine_image_sizes,
                    affine_grid_size,
                ) in affine_maplet_templates.items():
                    affine_summaries, affine_active, affine_template_usable = (
                        score_affine_warped_maplet_patch_likelihoods(
                            query_grid=affine_query_grid,
                            support_patches=affine_patches,
                            support_patch_valid=affine_patch_valid,
                            support_image_sizes=affine_image_sizes,
                            support_grid_size=int(affine_grid_size),
                            affine_matrices=affine_matrices,
                            projected_anchor_xy=affine_anchors,
                            maplet_geometry_valid=affine_valid,
                            image_width=int(getattr(camera, "width")),
                            image_height=int(getattr(camera, "height")),
                            temperature=float(affine_maplet_temperature),
                            minimum_support_fraction=float(
                                affine_maplet_minimum_patch_fraction
                            ),
                            exclude_center=True,
                            maplet_support_image_groups=affine_maplet_image_groups,
                            support_image_count=affine_support_image_count,
                            fixed_support_image_priors=affine_support_image_priors,
                        )
                    )
                    for statistic, values in affine_summaries.items():
                        output.setdefault(f"{field_name}_{statistic}", []).append(
                            values.detach().cpu().numpy().astype(np.float32)
                        )
                    affine_profile_active_counts[field_name].append(
                        affine_active.detach().cpu().numpy().astype(np.int32)
                    )
                    affine_profile_template_usable_counts[field_name].append(
                        affine_template_usable.detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
            for block_grid, (coherent_groups, coherent_counts, coherent_group_count) in (
                coherent_group_tensors.items()
            ):
                coherent_summaries, coherent_active_groups = (
                    score_multiscale_observation_group_position_likelihoods(
                        query_grids=query_grids,
                        support_descriptors=source_support,
                        projected_xy=projected,
                        projection_valid=projection_valid,
                        image_width=int(getattr(camera, "width")),
                        image_height=int(getattr(camera, "height")),
                        temperatures=temperatures,
                        observation_groups=coherent_groups,
                        group_count=int(coherent_group_count),
                        group_observation_counts=coherent_counts,
                    )
                )
                for family, family_summary in coherent_summaries.items():
                    for statistic, values in family_summary.items():
                        output.setdefault(
                            f"same_view_block{int(block_grid)}_{family}_{statistic}", []
                        ).append(values.detach().cpu().numpy().astype(np.float32))
                coherent_active_group_counts[int(block_grid)].append(
                    coherent_active_groups.detach().cpu().numpy().astype(np.int32)
                )
            for field_name, maps in context_maps.items():
                ratios, evidence_valid = sample_context_position_log_ratios(
                    maps=maps,
                    projected_xy=projected,
                    projection_valid=projection_valid,
                    image_width=int(getattr(camera, "width")),
                    image_height=int(getattr(camera, "height")),
                )
                tracks, _active = aggregate_view_log_ratios(
                    ratios,
                    evidence_valid,
                    observation_track_groups=group_tensor,
                    track_group_count=group_count,
                    group_observation_counts=count_tensor,
                )
                for statistic, values in summarize_track_log_ratios(tracks).items():
                    output.setdefault(f"{field_name}_{statistic}", []).append(
                        values.detach().cpu().numpy().astype(np.float32)
                    )
            active_track_counts.append(active_tracks.detach().cpu().numpy().astype(np.int32))
            visible_observation_counts.append(
                visible_observations.detach().cpu().numpy().astype(np.int32)
            )
    flattened = {key: np.concatenate(values) for key, values in output.items()}
    flattened["active_support_track_counts"] = np.concatenate(active_track_counts)
    flattened["visible_support_observation_counts"] = np.concatenate(visible_observation_counts)
    flattened["fixed_support_track_counts"] = np.full(
        (len(poses),), group_count, dtype=np.int32
    )
    for block_grid, (_groups, _counts, coherent_group_count) in coherent_group_tensors.items():
        flattened[f"same_view_block{int(block_grid)}_active_support_group_counts"] = (
            np.concatenate(coherent_active_group_counts[int(block_grid)])
        )
        flattened[f"same_view_block{int(block_grid)}_fixed_support_group_counts"] = np.full(
            (len(poses),), int(coherent_group_count), dtype=np.int32
        )
    if affine_maplet_topology is not None:
        flattened["affine_maplet_geometry_active_counts"] = np.concatenate(
            affine_geometry_active_counts
        )
        flattened["affine_maplet_fixed_counts"] = np.full(
            (len(poses),), int(affine_maplet_topology.maplet_count), dtype=np.int32
        )
        for field_name in affine_maplet_templates:
            flattened[f"{field_name}_active_counts"] = np.concatenate(
                affine_profile_active_counts[field_name]
            )
            flattened[f"{field_name}_template_usable_counts"] = np.concatenate(
                affine_profile_template_usable_counts[field_name]
            )
    return flattened


def _parse_paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one hypothesis artifact is required")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.hypothesis_batch_size) <= 0:
        raise ValueError("hypothesis_batch_size must be positive")
    if int(args.context_template_batch_size) <= 0:
        raise ValueError("context_template_batch_size must be positive")
    if (
        not np.isfinite(float(args.context_temperature))
        or float(args.context_temperature) <= 0.0
        or not 0.0 < float(args.context_minimum_support_fraction) <= 1.0
        or not 0.0 < float(args.context_minimum_query_overlap_fraction) <= 1.0
    ):
        raise ValueError("context-NCC parameters are invalid")
    if (
        int(args.affine_maplet_anchor_block_grid) <= 0
        or int(args.affine_maplet_anchors_per_block) <= 0
        or int(args.affine_maplet_min_neighbors) < 2
        or int(args.affine_maplet_max_neighbors) < int(args.affine_maplet_min_neighbors)
        or not np.isfinite(
            [
                args.affine_maplet_neighbor_radius_px,
                args.affine_maplet_neighbor_sigma_px,
                args.affine_maplet_max_condition_number,
                args.affine_maplet_max_rmse_px,
                args.affine_maplet_minimum_patch_fraction,
                args.affine_maplet_temperature,
            ]
        ).all()
        or float(args.affine_maplet_neighbor_radius_px) <= 0.0
        or float(args.affine_maplet_neighbor_sigma_px) <= 0.0
        or float(args.affine_maplet_max_condition_number) <= 1.0
        or float(args.affine_maplet_max_rmse_px) <= 0.0
        or not 0.0 < float(args.affine_maplet_minimum_patch_fraction) <= 1.0
        or float(args.affine_maplet_temperature) <= 0.0
    ):
        raise ValueError("affine-maplet parameters are invalid")
    if int(args.query_shard_count) <= 0 or not 0 <= int(args.query_shard_index) < int(args.query_shard_count):
        raise ValueError("query shard is invalid")
    temperatures = {
        "radio_final": float(args.radio_final_temperature),
        "radio_intermediate": float(args.radio_intermediate_temperature),
        "alike": float(args.alike_temperature),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in temperatures.values()):
        raise ValueError("feature temperatures must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    output_dir = Path(args.output_dir)
    output_path = output_dir / "pose_conditioned_support_alignment_scores_v1.npz"
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    layout_path = Path(args.support_layout)
    layout = _load_layout(layout_path)
    score_splits = {item.strip() for item in str(args.score_splits).split(",") if item.strip()}
    if not score_splits or score_splits - set(layout.split_names.tolist()):
        raise ValueError("score_splits is absent from the frozen support layout")
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    image_camera_ids = read_colmap_image_camera_ids_binary(model_dir / "images.bin")
    image_sizes = {
        image_id: (int(cameras[camera_id].width), int(cameras[camera_id].height))
        for image_id, camera_id in image_camera_ids.items()
        if camera_id in cameras
    }
    selected_queries = [
        index
        for index, split in enumerate(layout.split_names.tolist())
        if split in score_splits and index % int(args.query_shard_count) == int(args.query_shard_index)
    ]
    if not selected_queries:
        raise ValueError("query shard selected no frozen support-layout queries")
    selected_query_ids = {str(layout.query_ids[index]) for index in selected_queries}
    sources: dict[str, ImageGridFeatureSource] = {
        "radio_final": _load_radio_final_source(
            Path(args.radio_final_context_cache),
            grid_size=int(args.radio_final_grid_size),
            image_sizes=image_sizes,
        ),
        "radio_intermediate": _load_spatial_source(
            Path(args.radio_intermediate_context_cache),
            name="radio_intermediate",
            expected_format="radio_intermediate_image_spatial_context_v1",
            grid_size=int(args.radio_intermediate_grid_size),
        ),
        "alike": _load_spatial_source(
            Path(args.alike_context_cache),
            name="alike",
            expected_format="alike_image_spatial_context_v1",
            grid_size=int(args.alike_grid_size),
        ),
    }
    context_sources: dict[str, ImageGridFeatureSource] = dict(sources)
    context_input_paths: dict[str, Path] = {
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_context_cache),
    }
    if args.radio_final_pca_context_cache:
        pca_path = Path(args.radio_final_pca_context_cache)
        pca_source = _load_final_pca_source(
            pca_path, grid_size=int(args.radio_final_grid_size)
        )
        if str(pca_source.metadata.get("source_context_sha256", "")) != file_sha256_short(
            Path(args.radio_final_context_cache)
        ):
            raise ValueError("RADIO-final PCA cache does not derive from the scored raw cache")
        context_sources["radio_final_pca64"] = pca_source
        context_input_paths["radio_final_pca64"] = pca_path
    if args.radio_intermediate_pca_context_cache:
        pca_path = Path(args.radio_intermediate_pca_context_cache)
        pca_source = _load_intermediate_pca_source(
            pca_path, grid_size=int(args.radio_intermediate_grid_size)
        )
        if str(pca_source.metadata.get("source_context_sha256", "")) != file_sha256_short(
            Path(args.radio_intermediate_context_cache)
        ):
            raise ValueError(
                "RADIO-intermediate PCA cache does not derive from the scored raw cache"
            )
        context_sources["radio_intermediate_pca64"] = pca_source
        context_input_paths["radio_intermediate_pca64"] = pca_path
    context_profiles = _parse_context_profiles(
        args.context_profiles, sources=context_sources
    )
    coherent_support_block_grids = _parse_coherent_support_block_grids(
        args.coherent_support_block_grids
    )
    affine_maplet_profiles = _parse_affine_maplet_profiles(
        args.affine_maplet_profiles, sources=context_sources
    )
    hypothesis_paths = _parse_paths(args.hypothesis_artifacts)
    hypotheses, hypothesis_metadata = _load_hypotheses(
        hypothesis_paths, selected_query_ids=selected_query_ids
    )
    query_ids = np.asarray(hypotheses["query_ids"]).astype(str)
    split_names = np.asarray(hypotheses["split_names"]).astype(str)
    labels = np.asarray(hypotheses["evaluation_labels"]).astype(str)
    indices = np.asarray(hypotheses["hypothesis_indices"], dtype=np.int64)
    poses = np.asarray(hypotheses["poses_w2c"], dtype=np.float64)
    if poses.shape != (len(query_ids), 4, 4) or not np.all(np.isfinite(poses)):
        raise ValueError("inference hypotheses have invalid pose matrices")
    started = time.monotonic()
    rows: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "split_names": [],
        "evaluation_labels": [],
        "hypothesis_indices": [],
    }
    value_names: set[str] | None = None
    score_names: set[str] | None = None
    for query_order, layout_index in enumerate(selected_queries, start=1):
        query_id = str(layout.query_ids[layout_index])
        split = str(layout.split_names[layout_index])
        camera_id = image_camera_ids.get(query_id)
        if camera_id is None or camera_id not in cameras:
            raise ValueError(f"{query_id}: no query camera ownership")
        group = np.flatnonzero((query_ids == query_id) & (split_names == split))
        local_labels = np.unique(labels[group])
        if len(group) == 0 or len(local_labels) != 1:
            raise ValueError(f"{query_id}: hypotheses are missing or mix evaluation labels")
        values = _score_query(
            query_id=query_id,
            query_index=layout_index,
            poses_w2c=poses[group],
            camera=cameras[camera_id],
            layout=layout,
            sources=sources,
            temperatures=temperatures,
            context_sources=context_sources,
            context_profiles=context_profiles,
            coherent_support_block_grids=coherent_support_block_grids,
            affine_maplet_profiles=affine_maplet_profiles,
            affine_maplet_anchor_block_grid=int(args.affine_maplet_anchor_block_grid),
            affine_maplet_anchors_per_block=int(args.affine_maplet_anchors_per_block),
            affine_maplet_neighbor_radius_px=float(args.affine_maplet_neighbor_radius_px),
            affine_maplet_neighbor_sigma_px=float(args.affine_maplet_neighbor_sigma_px),
            affine_maplet_max_neighbors=int(args.affine_maplet_max_neighbors),
            affine_maplet_min_neighbors=int(args.affine_maplet_min_neighbors),
            affine_maplet_max_condition_number=float(
                args.affine_maplet_max_condition_number
            ),
            affine_maplet_max_rmse_px=float(args.affine_maplet_max_rmse_px),
            affine_maplet_minimum_patch_fraction=float(
                args.affine_maplet_minimum_patch_fraction
            ),
            affine_maplet_temperature=float(args.affine_maplet_temperature),
            context_temperature=float(args.context_temperature),
            context_template_batch_size=int(args.context_template_batch_size),
            context_minimum_support_fraction=float(args.context_minimum_support_fraction),
            context_minimum_query_overlap_fraction=float(
                args.context_minimum_query_overlap_fraction
            ),
            device=device,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
        )
        current_names = set(values)
        current_score_names = {name for name in current_names if not _is_audit_field(name)}
        if value_names is None:
            value_names = current_names
            score_names = current_score_names
            rows.update({name: [] for name in sorted(value_names)})
        elif current_names != value_names or current_score_names != score_names:
            raise RuntimeError("support-alignment score schema changed between queries")
        # ``dtype=np.str_`` on ``np.full`` silently chooses ``<U1`` and would
        # truncate query IDs / policy labels in the diagnostic artifact.
        rows["query_ids"].append(_constant_string_column(query_id, len(group)))
        rows["split_names"].append(_constant_string_column(split, len(group)))
        rows["evaluation_labels"].append(
            _constant_string_column(str(local_labels[0]), len(group))
        )
        rows["hypothesis_indices"].append(indices[group].astype(np.int64))
        for name, value in values.items():
            rows[name].append(value)
        print(
            json.dumps(
                {
                    "stage": "score_pose_conditioned_support_alignment",
                    "query": query_id,
                    "completed_queries": query_order,
                    "assigned_queries": len(selected_queries),
                    "hypotheses": int(len(group)),
                    "fixed_support_tracks": int(values["fixed_support_track_counts"][0]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if value_names is None or score_names is None:
        raise RuntimeError("support-alignment scorer emitted no score fields")
    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    row_count = len(arrays["query_ids"])
    if any(len(value) != row_count for value in arrays.values()):
        raise RuntimeError("support-alignment output arrays are not row-aligned")
    top_field = "multiscale_equal_median"
    top1 = np.zeros((row_count,), dtype=bool)
    for query_id, split, label in sorted(
        set(zip(arrays["query_ids"].tolist(), arrays["split_names"].tolist(), arrays["evaluation_labels"].tolist()))
    ):
        group = np.flatnonzero(
            (arrays["query_ids"] == query_id)
            & (arrays["split_names"] == split)
            & (arrays["evaluation_labels"] == label)
        )
        top1[group[int(np.argmax(arrays[top_field][group]))]] = True
    layout_metadata = dict(layout.metadata)
    metadata: dict[str, Any] = {
        "format": POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT,
        "version": (
            "fixed_support_candidate_conditioned_affine_maplet_likelihood_v1"
            if affine_maplet_profiles
            else (
                "fixed_support_same_view_group_full_image_likelihood_v1"
                if coherent_support_block_grids
                else (
                    POSE_CONDITIONED_SUPPORT_ALIGNMENT_VERSION
                    if not context_profiles
                    else "fixed_support_full_image_normalized_context_ncc_likelihood_v1"
                )
            )
        ),
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "selection_not_promoted": True,
        "row_count": int(row_count),
        "score_fields": sorted(score_names),
        "audit_fields": sorted(value_names - score_names),
        "diagnostic_selection_field": top_field,
        "support_layout": str(layout_path),
        "support_layout_sha256": file_sha256_short(layout_path),
        "support_layout_protocol": layout_metadata.get("support_pool_protocol"),
        "strict_diagnostic_contract": {
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
            "per_view_context_templates": bool(context_profiles),
            "context_maps_fixed_before_pose_scoring": bool(context_profiles),
            "same_view_support_groups_fixed_before_pose_scoring": bool(
                coherent_support_block_grids
            ),
            "affine_maplet_topology_fixed_before_pose_scoring": bool(
                affine_maplet_profiles
            ),
            "affine_maplet_anchor_or_neighbor_reselection": False,
            "affine_maplet_invalid_or_missing_is_neutral": bool(affine_maplet_profiles),
        },
        "temperatures": temperatures,
        "context_ncc": {
            "profiles": [
                {
                    "score_field_prefix": field_name,
                    "source": source_name,
                    "window_size": int(window_size),
                }
                for field_name, source_name, window_size in context_profiles
            ],
            "temperature": float(args.context_temperature),
            "template_batch_size": int(args.context_template_batch_size),
            "minimum_support_fraction": float(args.context_minimum_support_fraction),
            "minimum_query_overlap_fraction": float(
                args.context_minimum_query_overlap_fraction
            ),
            "normalization": "per_support_view_full_query_grid_ncc_relative_to_uniform_v1",
        },
        "same_view_support_groups": {
            "block_grids": [int(value) for value in coherent_support_block_grids],
            "grouping": "support_image_id_plus_fixed_normalized_image_block_v1",
            "geometry_source": "alike",
            "aggregation": "mean_observation_log_likelihood_ratio_then_robust_group_summary_v1",
            "invalid_projection": "neutral_log_likelihood_ratio_zero",
        },
        "affine_maplets": {
            "profiles": [
                {
                    "score_field_prefix": field_name,
                    "source": source_name,
                    "window_size": int(window_size),
                }
                for field_name, source_name, window_size in affine_maplet_profiles
            ],
            "anchor_partition": {
                "source": "fixed_support_image_pixel_blocks_v1",
                "geometry_source": "alike",
                "block_grid": int(args.affine_maplet_anchor_block_grid),
                "anchors_per_block": int(args.affine_maplet_anchors_per_block),
            },
            "neighbor_topology": {
                "same_support_image_only": True,
                "neighbor_radius_px": float(args.affine_maplet_neighbor_radius_px),
                "neighbor_sigma_px": float(args.affine_maplet_neighbor_sigma_px),
                "max_neighbors": int(args.affine_maplet_max_neighbors),
                "min_neighbors": int(args.affine_maplet_min_neighbors),
                "maximum_condition_number": float(args.affine_maplet_max_condition_number),
                "maximum_rmse_px": float(args.affine_maplet_max_rmse_px),
                "candidate_pose_use": "fixed_track_projection_only_v1",
            },
            "likelihood": {
                "normalization": "per_patch_descriptor_full_query_grid_relative_to_uniform_v1",
                "exclude_center": True,
                "minimum_support_patch_fraction": float(
                    args.affine_maplet_minimum_patch_fraction
                ),
                "temperature": float(args.affine_maplet_temperature),
                "invalid_or_missing": "neutral_log_likelihood_ratio_zero",
            },
            "support_image_mixture": {
                "source": "fixed_support_image_scores_from_target_free_layout_v1",
                "unknown_support_image": "neutral_log_likelihood_ratio_zero",
                "uniform": "log_mean_exp_of_fixed_per_image_maplet_ratios_v1",
                "prior_weighted": "logsumexp_fixed_image_prior_plus_per_image_maplet_ratio_v1",
            },
        },
        "query_shard": {
            "count": int(args.query_shard_count),
            "index": int(args.query_shard_index),
            "score_splits": sorted(score_splits),
        },
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
            "hypothesis_artifact_sha256": [
                file_sha256_short(path) for path in hypothesis_paths
            ],
            "colmap_cameras_bin_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
            "radio_final_context_cache": str(args.radio_final_context_cache),
            "radio_final_context_cache_sha256": file_sha256_short(Path(args.radio_final_context_cache)),
            "radio_intermediate_context_cache": str(args.radio_intermediate_context_cache),
            "radio_intermediate_context_cache_sha256": file_sha256_short(Path(args.radio_intermediate_context_cache)),
            "alike_context_cache": str(args.alike_context_cache),
            "alike_context_cache_sha256": file_sha256_short(Path(args.alike_context_cache)),
            "context_profile_caches": {
                source_name: {
                    "path": str(context_input_paths[source_name]),
                    "sha256": file_sha256_short(context_input_paths[source_name]),
                    "grid_size": int(context_sources[source_name].grid_size),
                    "descriptor_dim": int(context_sources[source_name].descriptor_dim),
                }
                for source_name in sorted(
                    {profile[1] for profile in context_profiles}
                    | {profile[1] for profile in affine_maplet_profiles}
                )
            },
        },
        "implementation": {
            "scorer_source_sha256": file_sha256_short(Path(__file__)),
            "layout_format": POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT,
            "hypothesis_metadata_fingerprint": _canonical_hash(hypothesis_metadata),
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        diagnostic_score_top1=top1,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "stage": "score_pose_conditioned_support_alignment",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(row_count),
        "query_count": int(len(selected_queries)),
        "diagnostic_selection_field": top_field,
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
