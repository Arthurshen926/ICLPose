"""Score frozen pose hypotheses with lazy real-image MASt3R pair evidence.

This diagnostic intentionally evaluates only the support images already fixed
by ``pose_conditioned_support_alignment``.  For every query/support image
pair, MASt3R produces pair-conditioned dense descriptors in memory; no global
image retrieval, submap, render, pose-local support reselection, or dense disk
cache is used.  A tested pose only moves fixed SfM tracks through the query
image before sampling each pair's full-query normalized likelihood.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from feature_extract.tools.vfm.score_pose_conditioned_support_alignment import (
    _canonical_hash,
    _constant_string_column,
    _load_hypotheses,
    _load_layout,
    _local_track_groups,
    _parse_paths,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    ImageGridFeatureSource,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT,
    POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT,
    aggregate_fixed_support_image_log_ratios,
    build_fixed_affine_maplet_topology,
    fit_fixed_affine_maplet_transforms_torch,
    project_simple_radial_torch,
)


MAST3R_PAIRWISE_ALIGNMENT_VERSION = "mast3r_pair_conditioned_fixed_support_affine_maplet_v1"
_SCORE_PREFIX = "mast3r_affine_maplet_context"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_layout", required=True)
    parser.add_argument("--hypothesis_artifacts", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--mast3r_root", default="third_party/mast3r")
    parser.add_argument("--mast3r_checkpoint", required=True)
    parser.add_argument("--mast3r_image_size", type=int, default=512)
    parser.add_argument("--mast3r_grid_size", type=int, default=128)
    parser.add_argument("--mast3r_pair_batch_size", type=int, default=4)
    parser.add_argument("--maplet_window_size", type=int, default=9)
    parser.add_argument("--maplet_anchor_block_grid", type=int, default=2)
    parser.add_argument("--maplet_anchors_per_block", type=int, default=1)
    parser.add_argument("--maplet_neighbor_radius_px", type=float, default=256.0)
    parser.add_argument("--maplet_neighbor_sigma_px", type=float, default=128.0)
    parser.add_argument("--maplet_max_neighbors", type=int, default=16)
    parser.add_argument("--maplet_min_neighbors", type=int, default=4)
    parser.add_argument("--maplet_max_condition_number", type=float, default=100.0)
    parser.add_argument("--maplet_max_rmse_px", type=float, default=32.0)
    parser.add_argument("--maplet_minimum_support_fraction", type=float, default=0.75)
    parser.add_argument("--maplet_temperature", type=float, default=0.1)
    parser.add_argument("--hypothesis_batch_size", type=int, default=128)
    parser.add_argument("--score_splits", default="validation,test")
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_mast3r_runtime(root: Path) -> tuple[Any, Any]:
    resolved = Path(root).resolve()
    if not (resolved / "mast3r" / "model.py").is_file() or not (
        resolved / "dust3r" / "dust3r" / "inference.py"
    ).is_file():
        raise FileNotFoundError(f"MASt3R source tree is incomplete: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    return (AsymmetricMASt3R, (inference, load_images))


def _load_rgb_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        return (int(oriented.width), int(oriented.height))


def _image_path(image_root: Path, image_id: str) -> Path:
    relative = Path(str(image_id))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"image ID is not a safe relative path: {image_id!r}")
    path = Path(image_root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_linear_resize(
    *,
    image_id: str,
    original_size: tuple[int, int],
    expected_size: tuple[int, int],
    view: Mapping[str, object],
) -> None:
    """Reject MASt3R center-crops that cannot map back to COLMAP pixels."""

    original_width, original_height = (int(value) for value in original_size)
    camera_width, camera_height = (int(value) for value in expected_size)
    aspect_error = abs(
        float(original_width) / float(original_height)
        - float(camera_width) / float(camera_height)
    )
    if aspect_error > 2e-3:
        raise ValueError(
            f"{image_id}: RGB aspect {original_size} disagrees with COLMAP camera {expected_size}"
        )
    true_shape = np.asarray(view.get("true_shape"), dtype=np.int64).reshape(-1)
    if true_shape.shape != (2,) or np.any(true_shape <= 1):
        raise ValueError(f"{image_id}: MASt3R view has invalid true_shape")
    output_height, output_width = (int(value) for value in true_shape)
    scale_x = float(output_width - 1) / float(original_width - 1)
    scale_y = float(output_height - 1) / float(original_height - 1)
    if not np.isfinite([scale_x, scale_y]).all() or abs(scale_x - scale_y) > 2e-3:
        raise ValueError(
            f"{image_id}: MASt3R preprocessing center-cropped the image; "
            "its dense grid cannot be safely projected into the COLMAP frame"
        )


def _resample_pair_descriptor_grid(
    descriptors: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resample MASt3R's rectangular descriptor grid into a pixel-aware square grid."""

    values = torch.as_tensor(descriptors)
    size = int(grid_size)
    if (
        values.ndim != 3
        or values.shape[0] <= 1
        or values.shape[1] <= 1
        or values.shape[2] <= 0
        or size <= 1
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("MASt3R descriptor grid is invalid")
    tensor = values.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    square = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)[0]
    output = F.normalize(square.permute(1, 2, 0), p=2, dim=2)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("MASt3R descriptor resampling produced non-finite values")
    return output


def _sample_square_grid_patches(
    *,
    descriptor_grid: torch.Tensor,
    xy: np.ndarray,
    image_size: tuple[int, int],
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed support patches in the original image-pixel frame."""

    grid = torch.as_tensor(descriptor_grid)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    width, height = (int(value) for value in image_size)
    window = int(window_size)
    if (
        grid.ndim != 3
        or grid.shape[0] != grid.shape[1]
        or grid.shape[2] <= 0
        or len(coordinates) == 0
        or np.any(~np.isfinite(coordinates))
        or width <= 1
        or height <= 1
        or window < 3
        or window % 2 != 1
        or window > int(grid.shape[0])
    ):
        raise ValueError("square MASt3R support-patch inputs are invalid")
    radius = int(window // 2)
    offsets = torch.arange(-radius, radius + 1, dtype=grid.dtype, device=grid.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    local_xy = torch.as_tensor(coordinates, dtype=grid.dtype, device=grid.device)
    size = int(grid.shape[0])
    center_x = local_xy[:, 0] * float(size - 1) / float(width - 1)
    center_y = local_xy[:, 1] * float(size - 1) / float(height - 1)
    sample_x = center_x[:, None, None] + offset_x[None]
    sample_y = center_y[:, None, None] + offset_y[None]
    valid = (
        (sample_x >= 0.0)
        & (sample_x <= float(size - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(size - 1))
    )
    normalized = torch.stack(
        [
            2.0 * sample_x / float(size - 1) - 1.0,
            2.0 * sample_y / float(size - 1) - 1.0,
        ],
        dim=-1,
    )
    source = grid.permute(2, 0, 1)[None]
    patches = F.grid_sample(
        source.expand(len(local_xy), -1, -1, -1),
        normalized,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return F.normalize(patches, p=2, dim=1), valid


@dataclass(frozen=True)
class _PairMapletContext:
    query_grid: torch.Tensor
    templates: torch.Tensor
    template_valid: torch.Tensor
    template_counts: torch.Tensor
    template_usable: torch.Tensor
    log_uniform_normalizers: torch.Tensor
    support_image_sizes: torch.Tensor
    support_grid_size: int


def _build_pair_maplet_context(
    *,
    query_grid: torch.Tensor,
    support_patches: torch.Tensor,
    support_patch_valid: torch.Tensor,
    support_image_sizes: torch.Tensor,
    support_grid_size: int,
    temperature: float,
    minimum_support_fraction: float,
) -> _PairMapletContext:
    """Precompute each fixed pair's full-query descriptor normalizers once."""

    query = F.normalize(torch.as_tensor(query_grid, dtype=torch.float32), p=2, dim=2)
    patches = F.normalize(torch.as_tensor(support_patches, dtype=torch.float32), p=2, dim=1)
    valid = torch.as_tensor(support_patch_valid, dtype=torch.bool, device=query.device)
    sizes = torch.as_tensor(support_image_sizes, dtype=query.dtype, device=query.device)
    if (
        query.ndim != 3
        or query.shape[0] != query.shape[1]
        or patches.ndim != 4
        or patches.shape[0] == 0
        or patches.shape[1] != query.shape[2]
        or patches.shape[2] != patches.shape[3]
        or patches.shape[2] < 3
        or patches.shape[2] % 2 != 1
        or valid.shape != patches.shape[:1] + patches.shape[2:]
        or sizes.shape != (patches.shape[0], 2)
        or len({query.device, patches.device, valid.device, sizes.device}) != 1
        or int(support_grid_size) != int(query.shape[0])
        or not np.isfinite([temperature, minimum_support_fraction]).all()
        or float(temperature) <= 0.0
        or not 0.0 < float(minimum_support_fraction) <= 1.0
        or torch.any(sizes <= 1.0)
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(patches).all())
    ):
        raise ValueError("MASt3R pair-maplet context inputs are incompatible")
    window = int(patches.shape[2])
    radius = int(window // 2)
    context_valid = valid.clone()
    context_valid[:, radius, radius] = False
    slots = int(window * window - 1)
    counts = context_valid.sum(dim=(1, 2)).to(dtype=query.dtype)
    usable = counts >= float(minimum_support_fraction) * float(slots)
    templates = patches.permute(0, 2, 3, 1).reshape(patches.shape[0], -1, patches.shape[1])
    template_valid = context_valid.reshape(patches.shape[0], -1)
    flat_templates = templates.reshape(-1, templates.shape[-1])
    query_rows = query.reshape(-1, query.shape[-1])
    normalizers = torch.logsumexp(
        (flat_templates @ query_rows.T) / float(temperature), dim=1
    ) - float(np.log(float(query.shape[0] * query.shape[1])))
    normalizers = torch.where(
        (template_valid & usable[:, None]).reshape(-1),
        normalizers,
        torch.zeros_like(normalizers),
    ).reshape(template_valid.shape)
    return _PairMapletContext(
        query_grid=query,
        templates=templates,
        template_valid=template_valid,
        template_counts=counts,
        template_usable=usable,
        log_uniform_normalizers=normalizers,
        support_image_sizes=sizes,
        support_grid_size=int(support_grid_size),
    )


def _score_pair_maplet_context(
    *,
    context: _PairMapletContext,
    affine_matrices: torch.Tensor,
    projected_anchor_xy: torch.Tensor,
    maplet_geometry_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed MASt3R pair density at candidate-conditioned maplet warps."""

    affine = torch.as_tensor(affine_matrices, dtype=context.query_grid.dtype, device=context.query_grid.device)
    anchors = torch.as_tensor(projected_anchor_xy, dtype=context.query_grid.dtype, device=context.query_grid.device)
    geometry_valid = torch.as_tensor(
        maplet_geometry_valid, dtype=torch.bool, device=context.query_grid.device
    )
    template_count = int(context.templates.shape[0])
    point_count = int(context.templates.shape[1])
    window = int(round(np.sqrt(float(point_count))))
    if (
        affine.ndim != 4
        or affine.shape[1:] != (template_count, 2, 2)
        or anchors.shape != affine.shape[:2] + (2,)
        or geometry_valid.shape != affine.shape[:2]
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not np.isfinite(float(temperature))
        or float(temperature) <= 0.0
        or window * window != point_count
    ):
        raise ValueError("MASt3R pair-maplet score inputs are incompatible")
    radius = int(window // 2)
    offsets = torch.arange(-radius, radius + 1, dtype=affine.dtype, device=affine.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    grid_offsets = torch.stack([offset_x, offset_y], dim=2).reshape(-1, 2)
    source_scales = torch.stack(
        [
            (context.support_image_sizes[:, 0] - 1.0)
            / float(context.support_grid_size - 1),
            (context.support_image_sizes[:, 1] - 1.0)
            / float(context.support_grid_size - 1),
        ],
        dim=1,
    )
    source_offsets = source_scales[:, None, :] * grid_offsets[None]
    positions = anchors[:, :, None, :] + torch.einsum(
        "bmij,mpj->bmpi", affine, source_offsets
    )
    position_valid = (
        torch.isfinite(positions).all(dim=3)
        & (positions[..., 0] >= 0.0)
        & (positions[..., 0] <= float(image_width - 1))
        & (positions[..., 1] >= 0.0)
        & (positions[..., 1] <= float(image_height - 1))
        & geometry_valid[:, :, None]
        & context.template_valid[None]
        & context.template_usable[None, :, None]
    )
    grid = context.query_grid.permute(2, 0, 1)[None]
    normalized = torch.stack(
        [
            2.0 * positions[..., 0] / float(image_width - 1) - 1.0,
            2.0 * positions[..., 1] / float(image_height - 1) - 1.0,
        ],
        dim=-1,
    ).reshape(positions.shape[0], template_count * point_count, 1, 2)
    sampled = F.grid_sample(
        grid.expand(positions.shape[0], -1, -1, -1),
        normalized,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, :, 0].transpose(1, 2)
    sampled = F.normalize(sampled, p=2, dim=2).reshape(
        positions.shape[0], template_count, point_count, -1
    )
    numerator = (sampled * context.templates[None]).sum(dim=3) / float(temperature)
    ratios = numerator - context.log_uniform_normalizers[None]
    ratios = torch.where(position_valid, ratios, torch.zeros_like(ratios))
    maplet_scores = ratios.sum(dim=2) / context.template_counts[None].clamp_min(1.0)
    maplet_scores = torch.where(
        context.template_usable[None], maplet_scores, torch.zeros_like(maplet_scores)
    )
    active = (
        geometry_valid & context.template_usable[None] & position_valid.any(dim=2)
    )
    return maplet_scores, active


def _fixed_support_image_priors(
    *,
    support_ids: np.ndarray,
    support_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.unique(np.asarray(support_ids).astype(str))
    scores = np.asarray(support_scores, dtype=np.float64).reshape(-1)
    ids = np.asarray(support_ids).astype(str).reshape(-1)
    if len(images) == 0 or scores.shape != ids.shape:
        raise ValueError("fixed support-image prior inputs are invalid")
    per_image = np.empty((len(images),), dtype=np.float64)
    for position, image_id in enumerate(images.tolist()):
        values = scores[ids == str(image_id)]
        if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"fixed support image {image_id!r} has invalid prior mass")
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-5):
            raise ValueError(f"fixed support image {image_id!r} prior is not image-constant")
        per_image[position] = float(values[0])
    return images, (per_image / per_image.sum()).astype(np.float32)


def _geometry_source(
    *, image_ids: np.ndarray, image_sizes: np.ndarray
) -> ImageGridFeatureSource:
    """Provide only fixed real-image pixel geometry to the maplet builder."""

    return ImageGridFeatureSource(
        name="mast3r_fixed_support_geometry",
        image_ids=np.asarray(image_ids).astype(str),
        image_sizes=np.asarray(image_sizes, dtype=np.int64),
        grid_size=2,
        descriptors=np.ones((len(image_ids), 4, 1), dtype=np.float32),
        metadata={"source": "colmap_real_image_geometry_only_v1"},
    )


@dataclass(frozen=True)
class _SupportPairContext:
    image_position: int
    maplet_indices: torch.Tensor
    context: _PairMapletContext


def _build_pair_contexts(
    *,
    query_id: str,
    query_size: tuple[int, int],
    support_images: np.ndarray,
    support_sizes: Mapping[str, tuple[int, int]],
    anchor_image_ids: np.ndarray,
    anchor_xy: np.ndarray,
    mast3r_root: Path,
    mast3r_model: Any,
    inference: Any,
    load_images: Any,
    image_root: Path,
    image_size: int,
    grid_size: int,
    pair_batch_size: int,
    window_size: int,
    minimum_support_fraction: float,
    temperature: float,
    device: torch.device,
) -> tuple[tuple[_SupportPairContext, ...], int]:
    """Run pair-conditioned MASt3R lazily for this query's fixed support views."""

    del mast3r_root  # The import contract is checked before this function is called.
    query_path = _image_path(image_root, query_id)
    query_original_size = _load_rgb_size(query_path)
    pairs: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for image_id in support_images.tolist():
        support_path = _image_path(image_root, str(image_id))
        views = load_images([str(query_path), str(support_path)], size=int(image_size), verbose=False)
        if len(views) != 2:
            raise RuntimeError(f"{query_id}: MASt3R did not load one support pair")
        _validate_linear_resize(
            image_id=query_id,
            original_size=query_original_size,
            expected_size=query_size,
            view=views[0],
        )
        _validate_linear_resize(
            image_id=str(image_id),
            original_size=_load_rgb_size(support_path),
            expected_size=support_sizes[str(image_id)],
            view=views[1],
        )
        pairs.append((views[0], views[1]))
    output = inference(
        pairs,
        mast3r_model,
        str(device),
        batch_size=int(pair_batch_size),
        verbose=False,
    )
    query_descriptors = torch.as_tensor(output["pred1"]["desc"])
    support_descriptors = torch.as_tensor(output["pred2"]["desc"])
    if (
        query_descriptors.ndim != 4
        or support_descriptors.shape != query_descriptors.shape
        or query_descriptors.shape[0] != len(support_images)
    ):
        raise RuntimeError("MASt3R pair output shape is incompatible with fixed support images")
    contexts: list[_SupportPairContext] = []
    for position, image_id in enumerate(support_images.tolist()):
        maplet_rows = np.flatnonzero(anchor_image_ids == str(image_id)).astype(np.int64)
        if len(maplet_rows) == 0:
            continue
        query_grid = _resample_pair_descriptor_grid(
            query_descriptors[position], grid_size=int(grid_size), device=device
        )
        support_grid = _resample_pair_descriptor_grid(
            support_descriptors[position], grid_size=int(grid_size), device=device
        )
        patches, patch_valid = _sample_square_grid_patches(
            descriptor_grid=support_grid,
            xy=anchor_xy[maplet_rows],
            image_size=support_sizes[str(image_id)],
            window_size=int(window_size),
        )
        context = _build_pair_maplet_context(
            query_grid=query_grid,
            support_patches=patches,
            support_patch_valid=patch_valid,
            support_image_sizes=torch.as_tensor(
                np.repeat(
                    np.asarray(support_sizes[str(image_id)], dtype=np.float32)[None],
                    len(maplet_rows),
                    axis=0,
                ),
                device=device,
            ),
            support_grid_size=int(grid_size),
            temperature=float(temperature),
            minimum_support_fraction=float(minimum_support_fraction),
        )
        contexts.append(
            _SupportPairContext(
                image_position=int(position),
                maplet_indices=torch.as_tensor(maplet_rows, dtype=torch.long, device=device),
                context=context,
            )
        )
    if not contexts:
        raise RuntimeError(f"{query_id}: no fixed MASt3R maplets were materialized")
    return tuple(contexts), int(query_descriptors.shape[-1])


def _score_query(
    *,
    query_id: str,
    query_index: int,
    poses_w2c: np.ndarray,
    camera: object,
    layout: Any,
    support_image_sizes: Mapping[str, tuple[int, int]],
    mast3r_root: Path,
    mast3r_model: Any,
    inference: Any,
    load_images: Any,
    image_root: Path,
    mast3r_image_size: int,
    mast3r_grid_size: int,
    mast3r_pair_batch_size: int,
    maplet_window_size: int,
    maplet_anchor_block_grid: int,
    maplet_anchors_per_block: int,
    maplet_neighbor_radius_px: float,
    maplet_neighbor_sigma_px: float,
    maplet_max_neighbors: int,
    maplet_min_neighbors: int,
    maplet_max_condition_number: float,
    maplet_max_rmse_px: float,
    maplet_minimum_support_fraction: float,
    maplet_temperature: float,
    device: torch.device,
    hypothesis_batch_size: int,
) -> tuple[dict[str, np.ndarray], int]:
    if int(getattr(camera, "model_id")) != 2:
        raise ValueError("MASt3R support alignment requires COLMAP SIMPLE_RADIAL cameras")
    params = tuple(float(value) for value in getattr(camera, "params"))
    if len(params) != 4:
        raise ValueError("SIMPLE_RADIAL camera has invalid parameters")
    observation_slice, _groups, _counts = _local_track_groups(layout, query_index)
    xyz = np.asarray(layout.observation_xyz[observation_slice], dtype=np.float32)
    support_ids = np.asarray(layout.support_image_ids[observation_slice]).astype(str)
    support_xy = np.asarray(layout.support_xy[observation_slice], dtype=np.float32)
    support_scores = np.asarray(layout.support_image_scores[observation_slice], dtype=np.float32)
    fixed_images, fixed_priors = _fixed_support_image_priors(
        support_ids=support_ids, support_scores=support_scores
    )
    size_rows = []
    for image_id in fixed_images.tolist():
        size = support_image_sizes.get(str(image_id))
        if size is None:
            raise ValueError(f"{query_id}: support image {image_id!r} has no COLMAP camera")
        size_rows.append(size)
    topology = build_fixed_affine_maplet_topology(
        support_image_ids=support_ids,
        support_xy=support_xy,
        support_reprojection_errors=np.asarray(
            layout.support_reprojection_errors[observation_slice], dtype=np.float32
        ),
        geometry_source=_geometry_source(
            image_ids=fixed_images, image_sizes=np.asarray(size_rows, dtype=np.int64)
        ),
        anchor_block_grid=int(maplet_anchor_block_grid),
        anchors_per_block=int(maplet_anchors_per_block),
        neighbor_radius_px=float(maplet_neighbor_radius_px),
        max_neighbors=int(maplet_max_neighbors),
        min_neighbors=int(maplet_min_neighbors),
    )
    anchor_indices = np.asarray(topology.anchor_observation_indices, dtype=np.int64)
    contexts, descriptor_dim = _build_pair_contexts(
        query_id=query_id,
        query_size=(int(getattr(camera, "width")), int(getattr(camera, "height"))),
        support_images=fixed_images,
        support_sizes=support_image_sizes,
        anchor_image_ids=support_ids[anchor_indices],
        anchor_xy=support_xy[anchor_indices],
        mast3r_root=mast3r_root,
        mast3r_model=mast3r_model,
        inference=inference,
        load_images=load_images,
        image_root=image_root,
        image_size=int(mast3r_image_size),
        grid_size=int(mast3r_grid_size),
        pair_batch_size=int(mast3r_pair_batch_size),
        window_size=int(maplet_window_size),
        minimum_support_fraction=float(maplet_minimum_support_fraction),
        temperature=float(maplet_temperature),
        device=device,
    )
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    support_xy_tensor = torch.as_tensor(support_xy, dtype=torch.float32, device=device)
    priors = torch.as_tensor(fixed_priors, dtype=torch.float32, device=device)
    poses = np.asarray(poses_w2c, dtype=np.float32)
    primary_name = f"{_SCORE_PREFIX}{int(maplet_window_size)}_support_image_prior_mixture"
    uniform_name = f"{_SCORE_PREFIX}{int(maplet_window_size)}_support_image_uniform_mixture"
    values: dict[str, list[np.ndarray]] = {primary_name: [], uniform_name: []}
    active_counts: list[np.ndarray] = []
    usable_counts: list[np.ndarray] = []
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
            affine, anchors, affine_valid, _neighbors, _rmse = fit_fixed_affine_maplet_transforms_torch(
                support_xy=support_xy_tensor,
                projected_xy=projected,
                projection_valid=projection_valid,
                topology=topology,
                neighbor_sigma_px=float(maplet_neighbor_sigma_px),
                minimum_neighbors=int(maplet_min_neighbors),
                maximum_condition_number=float(maplet_max_condition_number),
                maximum_rmse_px=float(maplet_max_rmse_px),
            )
            image_ratios = torch.zeros(
                (len(pose_tensor), len(fixed_images)), dtype=torch.float32, device=device
            )
            image_active = torch.zeros(
                (len(pose_tensor), len(fixed_images)), dtype=torch.int32, device=device
            )
            usable = 0
            for pair_context in contexts:
                maplet_scores, active = _score_pair_maplet_context(
                    context=pair_context.context,
                    affine_matrices=affine.index_select(1, pair_context.maplet_indices),
                    projected_anchor_xy=anchors.index_select(1, pair_context.maplet_indices),
                    maplet_geometry_valid=affine_valid.index_select(
                        1, pair_context.maplet_indices
                    ),
                    image_width=int(getattr(camera, "width")),
                    image_height=int(getattr(camera, "height")),
                    temperature=float(maplet_temperature),
                )
                image_ratios[:, pair_context.image_position] = maplet_scores.mean(dim=1)
                image_active[:, pair_context.image_position] = active.sum(dim=1).to(
                    dtype=torch.int32
                )
                usable += int(pair_context.context.template_usable.sum().item())
            mixtures = aggregate_fixed_support_image_log_ratios(
                image_ratios,
                maplet_support_image_groups=torch.arange(
                    len(fixed_images), dtype=torch.long, device=device
                ),
                support_image_count=int(len(fixed_images)),
                fixed_support_image_priors=priors,
            )
            values[primary_name].append(
                mixtures["prior_mixture"].detach().cpu().numpy().astype(np.float32)
            )
            values[uniform_name].append(
                mixtures["uniform_mixture"].detach().cpu().numpy().astype(np.float32)
            )
            active_counts.append(image_active.sum(dim=1).detach().cpu().numpy().astype(np.int32))
            usable_counts.append(
                np.full((len(pose_tensor),), usable, dtype=np.int32)
            )
    output = {name: np.concatenate(items) for name, items in values.items()}
    output["mast3r_affine_maplet_active_counts"] = np.concatenate(active_counts)
    output["mast3r_affine_maplet_template_usable_counts"] = np.concatenate(usable_counts)
    output["mast3r_affine_maplet_fixed_counts"] = np.full(
        (len(poses),), int(topology.maplet_count), dtype=np.int32
    )
    output["mast3r_fixed_support_image_counts"] = np.full(
        (len(poses),), int(len(fixed_images)), dtype=np.int32
    )
    # Preserve the shared diagnostic schema. In this verifier a fixed maplet,
    # rather than a 3-D track observation, is the independently held-out
    # support unit reported by the evaluator.
    output["active_support_track_counts"] = output[
        "mast3r_affine_maplet_active_counts"
    ].copy()
    output["visible_support_observation_counts"] = output[
        "mast3r_affine_maplet_active_counts"
    ].copy()
    output["fixed_support_track_counts"] = output[
        "mast3r_affine_maplet_fixed_counts"
    ].copy()
    return output, int(descriptor_dim)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "mast3r_image_size": args.mast3r_image_size,
        "mast3r_grid_size": args.mast3r_grid_size,
        "mast3r_pair_batch_size": args.mast3r_pair_batch_size,
        "maplet_window_size": args.maplet_window_size,
        "maplet_anchor_block_grid": args.maplet_anchor_block_grid,
        "maplet_anchors_per_block": args.maplet_anchors_per_block,
        "maplet_max_neighbors": args.maplet_max_neighbors,
        "maplet_min_neighbors": args.maplet_min_neighbors,
        "hypothesis_batch_size": args.hypothesis_batch_size,
        "query_shard_count": args.query_shard_count,
    }
    if any(int(value) <= 0 for value in positive_ints.values()):
        raise ValueError("MASt3R scorer integer parameters must be positive")
    if (
        int(args.mast3r_grid_size) > int(args.mast3r_image_size)
        or int(args.maplet_window_size) < 3
        or int(args.maplet_window_size) % 2 != 1
        or int(args.maplet_window_size) > int(args.mast3r_grid_size)
        or int(args.maplet_min_neighbors) < 2
        or int(args.maplet_max_neighbors) < int(args.maplet_min_neighbors)
        or not 0 <= int(args.query_shard_index) < int(args.query_shard_count)
        or not np.isfinite(
            [
                args.maplet_neighbor_radius_px,
                args.maplet_neighbor_sigma_px,
                args.maplet_max_condition_number,
                args.maplet_max_rmse_px,
                args.maplet_minimum_support_fraction,
                args.maplet_temperature,
            ]
        ).all()
        or float(args.maplet_neighbor_radius_px) <= 0.0
        or float(args.maplet_neighbor_sigma_px) <= 0.0
        or float(args.maplet_max_condition_number) <= 1.0
        or float(args.maplet_max_rmse_px) <= 0.0
        or not 0.0 < float(args.maplet_minimum_support_fraction) <= 1.0
        or float(args.maplet_temperature) <= 0.0
    ):
        raise ValueError("MASt3R affine-maplet parameters are invalid")


def _image_manifest_hash(
    *, image_root: Path, query_ids: Sequence[str], layout: Any
) -> str:
    image_ids = set(str(value) for value in query_ids)
    for query_id in query_ids:
        query_index = layout.query_index(str(query_id))
        observation_slice, _track_slice = layout.query_slice(query_index)
        image_ids.update(layout.support_image_ids[observation_slice].astype(str).tolist())
    manifest = [
        f"{image_id}:{file_sha256_short(_image_path(image_root, image_id))}"
        for image_id in sorted(image_ids)
    ]
    return _canonical_hash(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
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
    selected_queries = [
        index
        for index, split in enumerate(layout.split_names.tolist())
        if split in score_splits and index % int(args.query_shard_count) == int(args.query_shard_index)
    ]
    if not selected_queries:
        raise ValueError("query shard selected no frozen support-layout queries")
    selected_query_ids = {str(layout.query_ids[index]) for index in selected_queries}
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    image_camera_ids = read_colmap_image_camera_ids_binary(model_dir / "images.bin")
    image_sizes = {
        image_id: (int(cameras[camera_id].width), int(cameras[camera_id].height))
        for image_id, camera_id in image_camera_ids.items()
        if camera_id in cameras
    }
    hypotheses, hypothesis_metadata = _load_hypotheses(
        _parse_paths(args.hypothesis_artifacts), selected_query_ids=selected_query_ids
    )
    hypothesis_paths = _parse_paths(args.hypothesis_artifacts)
    query_ids = np.asarray(hypotheses["query_ids"]).astype(str)
    split_names = np.asarray(hypotheses["split_names"]).astype(str)
    labels = np.asarray(hypotheses["evaluation_labels"]).astype(str)
    indices = np.asarray(hypotheses["hypothesis_indices"], dtype=np.int64)
    poses = np.asarray(hypotheses["poses_w2c"], dtype=np.float64)
    if poses.shape != (len(query_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("inference hypotheses have invalid pose matrices")
    mast3r_root = Path(args.mast3r_root)
    mast3r_checkpoint = Path(args.mast3r_checkpoint)
    if not mast3r_checkpoint.is_file():
        raise FileNotFoundError(mast3r_checkpoint)
    model_class, (inference, load_images) = _load_mast3r_runtime(mast3r_root)
    mast3r_model = model_class.from_pretrained(str(mast3r_checkpoint)).to(device).eval()
    image_root = Path(args.image_root)
    started = time.monotonic()
    rows: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "split_names": [],
        "evaluation_labels": [],
        "hypothesis_indices": [],
    }
    score_names: set[str] | None = None
    value_names: set[str] | None = None
    descriptor_dim: int | None = None
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
        values, local_descriptor_dim = _score_query(
            query_id=query_id,
            query_index=layout_index,
            poses_w2c=poses[group],
            camera=cameras[camera_id],
            layout=layout,
            support_image_sizes=image_sizes,
            mast3r_root=mast3r_root,
            mast3r_model=mast3r_model,
            inference=inference,
            load_images=load_images,
            image_root=image_root,
            mast3r_image_size=int(args.mast3r_image_size),
            mast3r_grid_size=int(args.mast3r_grid_size),
            mast3r_pair_batch_size=int(args.mast3r_pair_batch_size),
            maplet_window_size=int(args.maplet_window_size),
            maplet_anchor_block_grid=int(args.maplet_anchor_block_grid),
            maplet_anchors_per_block=int(args.maplet_anchors_per_block),
            maplet_neighbor_radius_px=float(args.maplet_neighbor_radius_px),
            maplet_neighbor_sigma_px=float(args.maplet_neighbor_sigma_px),
            maplet_max_neighbors=int(args.maplet_max_neighbors),
            maplet_min_neighbors=int(args.maplet_min_neighbors),
            maplet_max_condition_number=float(args.maplet_max_condition_number),
            maplet_max_rmse_px=float(args.maplet_max_rmse_px),
            maplet_minimum_support_fraction=float(args.maplet_minimum_support_fraction),
            maplet_temperature=float(args.maplet_temperature),
            device=device,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
        )
        current_names = set(values)
        current_scores = {
            name
            for name in current_names
            if name.endswith(("_support_image_prior_mixture", "_support_image_uniform_mixture"))
        }
        if len(current_scores) != 2:
            raise RuntimeError("MASt3R scorer did not emit exactly two fixed image mixtures")
        if value_names is None:
            value_names = current_names
            score_names = current_scores
            rows.update({name: [] for name in sorted(current_names)})
        elif current_names != value_names or current_scores != score_names:
            raise RuntimeError("MASt3R score schema changed between queries")
        if descriptor_dim is None:
            descriptor_dim = int(local_descriptor_dim)
        elif int(descriptor_dim) != int(local_descriptor_dim):
            raise RuntimeError("MASt3R descriptor dimension changed between pairs")
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
                    "stage": "score_mast3r_pairwise_support_alignment",
                    "query": query_id,
                    "completed_queries": query_order,
                    "assigned_queries": len(selected_queries),
                    "hypotheses": int(len(group)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if value_names is None or score_names is None or descriptor_dim is None:
        raise RuntimeError("MASt3R scorer emitted no rows")
    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    row_count = len(arrays["query_ids"])
    if any(len(value) != row_count for value in arrays.values()):
        raise RuntimeError("MASt3R score arrays are not row-aligned")
    primary_field = f"{_SCORE_PREFIX}{int(args.maplet_window_size)}_support_image_prior_mixture"
    if primary_field not in score_names:
        raise RuntimeError("MASt3R primary score field is absent")
    top1 = np.zeros((row_count,), dtype=bool)
    for query_id, split, label in sorted(
        set(
            zip(
                arrays["query_ids"].tolist(),
                arrays["split_names"].tolist(),
                arrays["evaluation_labels"].tolist(),
            )
        )
    ):
        group = np.flatnonzero(
            (arrays["query_ids"] == query_id)
            & (arrays["split_names"] == split)
            & (arrays["evaluation_labels"] == label)
        )
        top1[group[int(np.argmax(arrays[primary_field][group]))]] = True
    ordered_query_ids = [str(layout.query_ids[index]) for index in selected_queries]
    metadata: dict[str, Any] = {
        "format": POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT,
        "version": MAST3R_PAIRWISE_ALIGNMENT_VERSION,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "selection_not_promoted": True,
        "row_count": int(row_count),
        "score_fields": sorted(score_names),
        "audit_fields": sorted(value_names - score_names),
        "diagnostic_selection_field": primary_field,
        "support_layout": str(layout_path),
        "support_layout_sha256": file_sha256_short(layout_path),
        "support_layout_protocol": dict(layout.metadata).get("support_pool_protocol"),
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
            "pair_conditioned_dense_descriptors": True,
            "pair_features_materialized_lazily_per_query": True,
            "full_dense_pair_cache_written": False,
            "full_query_grid_normalizer_fixed_before_pose_sampling": True,
            "missing_maplet_or_support_image": "neutral_log_likelihood_ratio_zero",
        },
        "mast3r": {
            "root": str(mast3r_root),
            "checkpoint": str(mast3r_checkpoint),
            "checkpoint_sha256": file_sha256_short(mast3r_checkpoint),
            "image_size": int(args.mast3r_image_size),
            "square_grid_size": int(args.mast3r_grid_size),
            "descriptor_dim": int(descriptor_dim),
            "pair_batch_size": int(args.mast3r_pair_batch_size),
            "preprocessing": "aspect_preserving_resize_only_required_v1",
            "image_manifest_hash": _image_manifest_hash(
                image_root=image_root, query_ids=ordered_query_ids, layout=layout
            ),
        },
        "affine_maplets": {
            "anchor_partition": {
                "source": "fixed_support_image_pixel_blocks_v1",
                "block_grid": int(args.maplet_anchor_block_grid),
                "anchors_per_block": int(args.maplet_anchors_per_block),
            },
            "neighbor_topology": {
                "same_support_image_only": True,
                "neighbor_radius_px": float(args.maplet_neighbor_radius_px),
                "neighbor_sigma_px": float(args.maplet_neighbor_sigma_px),
                "max_neighbors": int(args.maplet_max_neighbors),
                "min_neighbors": int(args.maplet_min_neighbors),
                "maximum_condition_number": float(args.maplet_max_condition_number),
                "maximum_rmse_px": float(args.maplet_max_rmse_px),
                "candidate_pose_use": "fixed_track_projection_only_v1",
            },
            "likelihood": {
                "window_size": int(args.maplet_window_size),
                "exclude_center": True,
                "minimum_support_patch_fraction": float(args.maplet_minimum_support_fraction),
                "temperature": float(args.maplet_temperature),
                "normalization": "per_pair_support_descriptor_full_query_grid_relative_to_uniform_v1",
            },
            "support_image_mixture": {
                "source": "fixed_support_image_scores_from_target_free_layout_v1",
                "unknown_support_image": "neutral_log_likelihood_ratio_zero",
                "primary": "prior_mixture",
                "control": "uniform_mixture",
            },
        },
        "query_shard": {
            "count": int(args.query_shard_count),
            "index": int(args.query_shard_index),
            "score_splits": sorted(score_splits),
        },
        "inputs": {
            "hypothesis_artifacts": [str(path) for path in hypothesis_paths],
            "hypothesis_artifact_sha256": [file_sha256_short(path) for path in hypothesis_paths],
            "colmap_cameras_bin_sha256": file_sha256_short(model_dir / "cameras.bin"),
            "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
            "image_root": str(image_root),
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
        "stage": "score_mast3r_pairwise_support_alignment",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "row_count": int(row_count),
        "query_count": int(len(selected_queries)),
        "diagnostic_selection_field": primary_field,
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
