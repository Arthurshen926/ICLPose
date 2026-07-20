"""Frozen multi-observation image evidence for pose-hypothesis diagnostics.

This module deliberately sits outside the production selector.  A layout is
fixed from held-out global top-L candidates before poses are read.  At score
time, a pose can only move known SfM tracks through the query image; it cannot
change support images, tracks, query features, or the image-wide denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


POSE_CONDITIONED_SUPPORT_ALIGNMENT_LAYOUT_FORMAT = (
    "pose_conditioned_support_alignment_layout_v1"
)
POSE_CONDITIONED_SUPPORT_ALIGNMENT_SCORE_FORMAT = (
    "pose_conditioned_support_alignment_scores_v1"
)
POSE_CONDITIONED_SUPPORT_ALIGNMENT_VERSION = (
    "fixed_support_full_image_normalized_position_likelihood_v1"
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(~np.isfinite(array)) or np.any(norms <= 1e-8):
        raise ValueError("descriptor rows must be finite and non-zero")
    return array / norms


@dataclass(frozen=True)
class FrozenSupportAlignmentLayout:
    """Fixed support observations grouped by query and physical track."""

    query_ids: np.ndarray
    split_names: np.ndarray
    query_observation_offsets: np.ndarray
    query_track_offsets: np.ndarray
    track_observation_offsets: np.ndarray
    observation_track_ids: np.ndarray
    observation_xyz: np.ndarray
    support_image_ids: np.ndarray
    support_xy: np.ndarray
    support_image_scores: np.ndarray
    support_reprojection_errors: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        query_observation_offsets = np.asarray(
            self.query_observation_offsets, dtype=np.int64
        ).reshape(-1)
        query_track_offsets = np.asarray(self.query_track_offsets, dtype=np.int64).reshape(-1)
        track_observation_offsets = np.asarray(
            self.track_observation_offsets, dtype=np.int64
        ).reshape(-1)
        track_ids = np.asarray(self.observation_track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.observation_xyz, dtype=np.float64).reshape(-1, 3)
        image_ids = np.asarray(self.support_image_ids).astype(str).reshape(-1)
        xy = np.asarray(self.support_xy, dtype=np.float32).reshape(-1, 2)
        image_scores = np.asarray(self.support_image_scores, dtype=np.float32).reshape(-1)
        reprojection = np.asarray(self.support_reprojection_errors, dtype=np.float32).reshape(-1)
        query_count = len(query_ids)
        observation_count = len(track_ids)
        track_count = len(track_observation_offsets) - 1
        if (
            query_count == 0
            or len(set(query_ids.tolist())) != query_count
            or splits.shape != (query_count,)
            or query_observation_offsets.shape != (query_count + 1,)
            or query_track_offsets.shape != (query_count + 1,)
            or track_observation_offsets.shape != (track_count + 1,)
            or query_observation_offsets[0] != 0
            or query_observation_offsets[-1] != observation_count
            or query_track_offsets[0] != 0
            or query_track_offsets[-1] != track_count
            or track_observation_offsets[0] != 0
            or track_observation_offsets[-1] != observation_count
            or np.any(np.diff(query_observation_offsets) <= 0)
            or np.any(np.diff(query_track_offsets) <= 0)
            or np.any(np.diff(track_observation_offsets) <= 0)
        ):
            raise ValueError("support-alignment layout offsets are invalid")
        if not (
            xyz.shape == (observation_count, 3)
            and image_ids.shape == (observation_count,)
            and xy.shape == (observation_count, 2)
            and image_scores.shape == (observation_count,)
            and reprojection.shape == (observation_count,)
        ):
            raise ValueError("support-alignment observation arrays are misaligned")
        if (
            np.any(track_ids < 0)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(image_scores))
            or np.any(~np.isfinite(reprojection))
            or np.any(reprojection < 0.0)
            or np.any(image_ids == "")
        ):
            raise ValueError("support-alignment observations are invalid")
        for query_index in range(query_count):
            group_begin = int(query_track_offsets[query_index])
            group_end = int(query_track_offsets[query_index + 1])
            observation_begin = int(query_observation_offsets[query_index])
            observation_end = int(query_observation_offsets[query_index + 1])
            if (
                int(track_observation_offsets[group_begin]) != observation_begin
                or int(track_observation_offsets[group_end]) != observation_end
            ):
                raise ValueError("query and track observation offsets disagree")
            query_tracks: list[int] = []
            for group_index in range(group_begin, group_end):
                begin = int(track_observation_offsets[group_index])
                end = int(track_observation_offsets[group_index + 1])
                group_tracks = track_ids[begin:end]
                if not np.all(group_tracks == group_tracks[0]):
                    raise ValueError("a track group mixes physical track identities")
                if not np.allclose(xyz[begin:end], xyz[begin], rtol=0.0, atol=1e-8):
                    raise ValueError("a track group mixes XYZ coordinates")
                if len(set(image_ids[begin:end].tolist())) != end - begin:
                    raise ValueError("a track group repeats a support image")
                query_tracks.append(int(group_tracks[0]))
            if len(query_tracks) != len(set(query_tracks)):
                raise ValueError("a query repeats a physical support track")
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "query_observation_offsets", query_observation_offsets)
        object.__setattr__(self, "query_track_offsets", query_track_offsets)
        object.__setattr__(self, "track_observation_offsets", track_observation_offsets)
        object.__setattr__(self, "observation_track_ids", track_ids)
        object.__setattr__(self, "observation_xyz", xyz)
        object.__setattr__(self, "support_image_ids", image_ids)
        object.__setattr__(self, "support_xy", xy)
        object.__setattr__(self, "support_image_scores", image_scores)
        object.__setattr__(self, "support_reprojection_errors", reprojection)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def query_count(self) -> int:
        return int(len(self.query_ids))

    @property
    def observation_count(self) -> int:
        return int(len(self.observation_track_ids))

    @property
    def track_count(self) -> int:
        return int(len(self.track_observation_offsets) - 1)

    def query_index(self, query_id: str) -> int:
        matches = np.flatnonzero(self.query_ids == str(query_id))
        if len(matches) != 1:
            raise KeyError(f"support-alignment layout has no unique query {query_id!r}")
        return int(matches[0])

    def query_slice(self, query_index: int) -> tuple[slice, slice]:
        index = int(query_index)
        if not 0 <= index < self.query_count:
            raise IndexError("support-alignment query index is out of range")
        return (
            slice(
                int(self.query_observation_offsets[index]),
                int(self.query_observation_offsets[index + 1]),
            ),
            slice(
                int(self.query_track_offsets[index]),
                int(self.query_track_offsets[index + 1]),
            ),
        )


@dataclass(frozen=True)
class ImageGridFeatureSource:
    """A real-image dense descriptor grid with explicit pixel geometry."""

    name: str
    image_ids: np.ndarray
    image_sizes: np.ndarray
    grid_size: int
    descriptors: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        name = str(self.name)
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        image_sizes = np.asarray(self.image_sizes, dtype=np.int64).reshape(-1, 2)
        grid_size = int(self.grid_size)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        if (
            not name
            or grid_size <= 0
            or len(image_ids) == 0
            or len(set(image_ids.tolist())) != len(image_ids)
            or image_sizes.shape != (len(image_ids), 2)
            or np.any(image_sizes <= 1)
            or descriptors.ndim != 3
            or descriptors.shape[:2] != (len(image_ids), grid_size * grid_size)
            or descriptors.shape[2] <= 0
        ):
            raise ValueError("image-grid feature source shape is invalid")
        normalized = _normalize_rows(descriptors.reshape(-1, descriptors.shape[-1]))
        if np.max(np.abs(normalized - descriptors.reshape(-1, descriptors.shape[-1]))) > 5e-3:
            raise ValueError("image-grid descriptors must be L2 normalized")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", image_sizes)
        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def descriptor_dim(self) -> int:
        return int(self.descriptors.shape[-1])

    def image_position(self, image_id: str) -> int:
        positions = getattr(self, "_positions", None)
        if positions is None:
            positions = {value: index for index, value in enumerate(self.image_ids.tolist())}
            object.__setattr__(self, "_positions", positions)
        position = positions.get(str(image_id))
        if position is None:
            raise KeyError(f"{self.name}: no image grid for {image_id!r}")
        return int(position)

    def image_grid(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        position = self.image_position(image_id)
        return (
            self.descriptors[position].reshape(
                self.grid_size, self.grid_size, self.descriptor_dim
            ),
            self.image_sizes[position],
        )

    def sample_numpy(self, image_ids: Sequence[str], xy: np.ndarray) -> np.ndarray:
        requested = np.asarray(image_ids).astype(str).reshape(-1)
        coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if len(requested) != len(coordinates):
            raise ValueError("support image IDs and coordinates differ")
        output = np.empty((len(requested), self.descriptor_dim), dtype=np.float32)
        for image_id in sorted(set(requested.tolist())):
            rows = np.flatnonzero(requested == image_id)
            grid, image_size = self.image_grid(str(image_id))
            output[rows] = bilinear_sample_image_grid(
                grid, coordinates[rows], image_size=image_size
            )
        return _normalize_rows(output)

    def context_patches_torch(
        self,
        image_ids: Sequence[str],
        xy: np.ndarray,
        *,
        window_size: int,
        device: torch.device,
        source_grid_cache: MutableMapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-observation 2-D descriptor templates on a real image grid.

        The context window is measured in descriptor-grid cells, not pixels.
        Coordinates are sampled with the same ``align_corners=True`` convention
        as the center-descriptor path.  Cells outside a support image are
        marked invalid rather than synthesized by padding.
        """

        requested = np.asarray(image_ids).astype(str).reshape(-1)
        coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        window = int(window_size)
        if (
            len(requested) != len(coordinates)
            or window <= 0
            or window % 2 != 1
            or window > int(self.grid_size)
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("context-template sampling inputs are invalid")
        cache: MutableMapping[str, torch.Tensor] = (
            {} if source_grid_cache is None else source_grid_cache
        )
        count = int(len(requested))
        output = torch.zeros(
            (count, self.descriptor_dim, window, window),
            dtype=torch.float32,
            device=device,
        )
        valid = torch.zeros((count, window, window), dtype=torch.bool, device=device)
        radius = int(window // 2)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        for image_id in sorted(set(requested.tolist())):
            rows_np = np.flatnonzero(requested == image_id)
            rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
            grid_tensor = cache.get(str(image_id))
            if grid_tensor is None:
                grid, _image_size = self.image_grid(str(image_id))
                grid_tensor = torch.as_tensor(
                    grid.transpose(2, 0, 1)[None], dtype=torch.float32, device=device
                ).contiguous()
                cache[str(image_id)] = grid_tensor
            if grid_tensor.shape != (
                1,
                self.descriptor_dim,
                int(self.grid_size),
                int(self.grid_size),
            ):
                raise RuntimeError("cached context source grid has an invalid shape")
            _grid, image_size = self.image_grid(str(image_id))
            width, height = (int(value) for value in np.asarray(image_size, dtype=np.int64))
            local_xy = torch.as_tensor(coordinates[rows_np], dtype=torch.float32, device=device)
            center_x = local_xy[:, 0] * float(self.grid_size - 1) / float(width - 1)
            center_y = local_xy[:, 1] * float(self.grid_size - 1) / float(height - 1)
            sample_x = center_x[:, None, None] + offset_x[None]
            sample_y = center_y[:, None, None] + offset_y[None]
            local_valid = (
                (sample_x >= 0.0)
                & (sample_x <= float(self.grid_size - 1))
                & (sample_y >= 0.0)
                & (sample_y <= float(self.grid_size - 1))
            )
            normalized_xy = torch.stack(
                [
                    2.0 * sample_x / float(self.grid_size - 1) - 1.0,
                    2.0 * sample_y / float(self.grid_size - 1) - 1.0,
                ],
                dim=-1,
            )
            patches = F.grid_sample(
                grid_tensor.expand(len(rows), -1, -1, -1),
                normalized_xy,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            patches = F.normalize(patches, p=2, dim=1)
            output[rows] = patches
            valid[rows] = local_valid
        return output, valid


@dataclass(frozen=True)
class FixedAffineMapletTopology:
    """Frozen same-view anchor/neighbour topology for local affine evidence.

    Indices address one query's local support-observation array.  The topology
    is built solely from real support-image coordinates and reprojection
    quality before any candidate pose is evaluated.  ``-1`` pads short
    neighbour lists; it never denotes an evidence-bearing observation.
    """

    anchor_observation_indices: np.ndarray
    neighbor_observation_indices: np.ndarray
    neighbor_counts: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_observation_indices, dtype=np.int64).reshape(-1)
        neighbors = np.asarray(self.neighbor_observation_indices, dtype=np.int64)
        counts = np.asarray(self.neighbor_counts, dtype=np.int64).reshape(-1)
        if (
            anchors.size == 0
            or np.unique(anchors).size != anchors.size
            or np.any(anchors < 0)
            or neighbors.ndim != 2
            or neighbors.shape[0] != len(anchors)
            or neighbors.shape[1] == 0
            or counts.shape != (len(anchors),)
            or np.any(counts <= 0)
            or np.any(counts > neighbors.shape[1])
            or np.any(neighbors < -1)
            or not np.array_equal(counts, np.sum(neighbors >= 0, axis=1))
            or np.any(neighbors == anchors[:, None])
        ):
            raise ValueError("fixed affine maplet topology is invalid")
        object.__setattr__(self, "anchor_observation_indices", anchors)
        object.__setattr__(self, "neighbor_observation_indices", neighbors)
        object.__setattr__(self, "neighbor_counts", counts)

    @property
    def maplet_count(self) -> int:
        return int(len(self.anchor_observation_indices))

    @property
    def maximum_neighbor_count(self) -> int:
        return int(self.neighbor_observation_indices.shape[1])


def build_fixed_affine_maplet_topology(
    *,
    support_image_ids: Sequence[str],
    support_xy: np.ndarray,
    support_reprojection_errors: np.ndarray,
    geometry_source: ImageGridFeatureSource,
    anchor_block_grid: int,
    anchors_per_block: int,
    neighbor_radius_px: float,
    max_neighbors: int,
    min_neighbors: int,
) -> FixedAffineMapletTopology:
    """Build a pose-independent local-maplet partition from real support views.

    Each maplet is anchored in one support image and uses only nearby support
    observations from that *same* image.  Candidate poses are not available to
    this routine, so neither view selection nor local geometry can be adapted
    to a favored hypothesis.
    """

    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(support_xy, dtype=np.float32).reshape(-1, 2)
    reprojection = np.asarray(support_reprojection_errors, dtype=np.float32).reshape(-1)
    block_grid = int(anchor_block_grid)
    per_block = int(anchors_per_block)
    maximum = int(max_neighbors)
    minimum = int(min_neighbors)
    if (
        len(image_ids) == 0
        or len(image_ids) != len(coordinates)
        or reprojection.shape != (len(image_ids),)
        or np.any(~np.isfinite(coordinates))
        or np.any(~np.isfinite(reprojection))
        or np.any(reprojection < 0.0)
        or block_grid <= 0
        or per_block <= 0
        or not np.isfinite(float(neighbor_radius_px))
        or float(neighbor_radius_px) <= 0.0
        or maximum < minimum
        or minimum < 2
    ):
        raise ValueError("fixed affine maplet topology inputs are invalid")
    anchors: list[int] = []
    neighbor_rows: list[np.ndarray] = []
    for image_id in sorted(set(image_ids.tolist())):
        rows = np.flatnonzero(image_ids == str(image_id)).astype(np.int64)
        position = geometry_source.image_position(str(image_id))
        width, height = (
            int(value)
            for value in np.asarray(geometry_source.image_sizes[position], dtype=np.int64)
        )
        if width <= 1 or height <= 1:
            raise ValueError("affine maplet support image has invalid geometry")
        xy = coordinates[rows]
        columns = np.minimum(
            np.floor(np.clip(xy[:, 0], 0.0, float(width - 1)) / float(width - 1) * block_grid)
            .astype(np.int64),
            block_grid - 1,
        )
        grid_rows = np.minimum(
            np.floor(np.clip(xy[:, 1], 0.0, float(height - 1)) / float(height - 1) * block_grid)
            .astype(np.int64),
            block_grid - 1,
        )
        cells = grid_rows * block_grid + columns
        for cell in range(block_grid * block_grid):
            local_rows = rows[cells == cell]
            if len(local_rows) == 0:
                continue
            ordered = sorted(
                local_rows.tolist(),
                key=lambda row: (float(reprojection[row]), int(row)),
            )
            selected = 0
            for anchor in ordered:
                offsets = coordinates[rows] - coordinates[int(anchor)][None]
                distances = np.linalg.norm(offsets, axis=1)
                candidates = rows[
                    (rows != int(anchor)) & (distances <= float(neighbor_radius_px))
                ]
                if len(candidates) < minimum:
                    continue
                neighbor_order = sorted(
                    candidates.tolist(),
                    key=lambda row: (
                        float(np.linalg.norm(coordinates[row] - coordinates[int(anchor)])),
                        float(reprojection[row]),
                        int(row),
                    ),
                )
                anchors.append(int(anchor))
                neighbor_rows.append(
                    np.asarray(neighbor_order[:maximum], dtype=np.int64)
                )
                selected += 1
                if selected >= per_block:
                    break
    if not anchors:
        raise ValueError("fixed affine maplet topology has no geometrically supported anchors")
    width = max(len(rows) for rows in neighbor_rows)
    padded = np.full((len(anchors), width), -1, dtype=np.int64)
    for row, values in enumerate(neighbor_rows):
        padded[row, : len(values)] = values
    return FixedAffineMapletTopology(
        anchor_observation_indices=np.asarray(anchors, dtype=np.int64),
        neighbor_observation_indices=padded,
        neighbor_counts=np.asarray([len(rows) for rows in neighbor_rows], dtype=np.int64),
    )


def bilinear_sample_image_grid(
    grid: np.ndarray,
    xy: np.ndarray,
    *,
    image_size: Sequence[int],
) -> np.ndarray:
    """Sample an adaptive image grid with the ``align_corners=True`` convention."""

    values = np.asarray(grid, dtype=np.float32)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    width, height = (int(value) for value in np.asarray(image_size, dtype=np.int64).reshape(2))
    if values.ndim != 3 or values.shape[0] <= 0 or values.shape[1] <= 0 or width <= 1 or height <= 1:
        raise ValueError("image-grid sampling inputs are invalid")
    if np.any(~np.isfinite(coordinates)) or np.any(coordinates[:, 0] < 0.0) or np.any(
        coordinates[:, 0] > float(width - 1)
    ) or np.any(coordinates[:, 1] < 0.0) or np.any(coordinates[:, 1] > float(height - 1)):
        raise ValueError("image-grid coordinates are outside image bounds")
    grid_height, grid_width, dimension = values.shape
    x = coordinates[:, 0] * float(grid_width - 1) / float(width - 1)
    y = coordinates[:, 1] * float(grid_height - 1) / float(height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, grid_width - 1)
    y1 = np.minimum(y0 + 1, grid_height - 1)
    dx = (x - x0).astype(np.float32)[:, None]
    dy = (y - y0).astype(np.float32)[:, None]
    top = (1.0 - dx) * values[y0, x0] + dx * values[y0, x1]
    bottom = (1.0 - dx) * values[y1, x0] + dx * values[y1, x1]
    output = (1.0 - dy) * top + dy * bottom
    if output.shape != (len(coordinates), dimension) or np.any(~np.isfinite(output)):
        raise RuntimeError("image-grid sampling produced invalid descriptors")
    return output.astype(np.float32, copy=False)


@dataclass(frozen=True)
class ContextPositionLikelihoodMaps:
    """Fixed per-support full-image NCC likelihood maps for one context scale."""

    logits: torch.Tensor
    valid_cells: torch.Tensor
    log_uniform_normalizers: torch.Tensor
    template_usable: torch.Tensor

    def __post_init__(self) -> None:
        logits = torch.as_tensor(self.logits)
        valid = torch.as_tensor(self.valid_cells, dtype=torch.bool, device=logits.device)
        normalizers = torch.as_tensor(
            self.log_uniform_normalizers, dtype=logits.dtype, device=logits.device
        )
        usable = torch.as_tensor(
            self.template_usable, dtype=torch.bool, device=logits.device
        )
        if (
            logits.ndim != 3
            or logits.shape[0] == 0
            or logits.shape[1] <= 0
            or logits.shape[2] <= 0
            or valid.shape != logits.shape
            or normalizers.shape != (logits.shape[0],)
            or usable.shape != (logits.shape[0],)
            or not bool(torch.isfinite(logits).all())
            or not bool(torch.isfinite(normalizers).all())
            or torch.any(valid & ~usable[:, None, None])
        ):
            raise ValueError("context position-likelihood maps are invalid")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "valid_cells", valid)
        object.__setattr__(self, "log_uniform_normalizers", normalizers)
        object.__setattr__(self, "template_usable", usable)


def build_context_position_likelihood_maps(
    *,
    query_grid: torch.Tensor,
    support_patches: torch.Tensor,
    support_patch_valid: torch.Tensor,
    temperature: float,
    minimum_support_fraction: float,
    minimum_query_overlap_fraction: float,
) -> ContextPositionLikelihoodMaps:
    """Build image-wide normalized NCC maps for fixed support-view templates.

    Each support patch produces one likelihood map over the *entire* query
    feature grid.  Thus a tested pose only samples a precomputed map at its
    projection; it cannot affect the support template, map denominator, or
    set of eligible query locations.  The returned logits are normalized later
    against the uniform distribution over eligible full-image locations.
    """

    query = torch.as_tensor(query_grid)
    patches = torch.as_tensor(support_patches)
    patch_valid = torch.as_tensor(
        support_patch_valid, dtype=torch.bool, device=patches.device
    )
    if (
        query.ndim != 3
        or patches.ndim != 4
        or patch_valid.shape != patches.shape[:1] + patches.shape[2:]
        or query.shape[2] != patches.shape[1]
        or patches.shape[2] != patches.shape[3]
        or patches.shape[2] <= 0
        or patches.shape[2] % 2 != 1
        or query.device != patches.device
        or query.dtype != patches.dtype
        or not np.isfinite(
            [temperature, minimum_support_fraction, minimum_query_overlap_fraction]
        ).all()
        or float(temperature) <= 0.0
        or not 0.0 < float(minimum_support_fraction) <= 1.0
        or not 0.0 < float(minimum_query_overlap_fraction) <= 1.0
    ):
        raise ValueError("context position-likelihood inputs are incompatible")
    window = int(patches.shape[2])
    radius = int(window // 2)
    support = F.normalize(patches.to(dtype=torch.float32), p=2, dim=1)
    support_mask = patch_valid[:, None].to(dtype=support.dtype)
    support = support * support_mask
    support_cells = support_mask.sum(dim=(1, 2, 3))
    support_energy = support.square().sum(dim=(1, 2, 3))
    required_support_cells = float(minimum_support_fraction) * float(window * window)
    usable = (support_cells >= required_support_cells) & (support_energy > 1e-8)

    normalized_query = F.normalize(
        query.to(dtype=torch.float32).permute(2, 0, 1)[None], p=2, dim=1
    )
    dot = F.conv2d(normalized_query, support, padding=radius)[0]
    query_energy = F.conv2d(
        torch.ones(
            (1, 1, int(query.shape[0]), int(query.shape[1])),
            dtype=support.dtype,
            device=support.device,
        ),
        support_mask,
        padding=radius,
    )[0]
    denominator = torch.sqrt(
        support_energy[:, None, None].clamp_min(1e-8)
        * query_energy.clamp_min(1e-8)
    )
    cosine = dot / denominator
    required_query_cells = (
        float(minimum_query_overlap_fraction) * support_cells[:, None, None]
    )
    valid_cells = usable[:, None, None] & (query_energy >= required_query_cells)
    logits = cosine / float(temperature)
    masked_logits = torch.where(
        valid_cells,
        logits,
        torch.full_like(logits, -torch.inf),
    )
    valid_counts = valid_cells.sum(dim=(1, 2)).to(dtype=logits.dtype)
    normalizers = torch.logsumexp(masked_logits.reshape(len(logits), -1), dim=1)
    normalizers -= torch.log(valid_counts.clamp_min(1.0))
    normalizers = torch.where(usable & (valid_counts > 0.0), normalizers, torch.zeros_like(normalizers))
    return ContextPositionLikelihoodMaps(
        logits=torch.where(valid_cells, logits, torch.zeros_like(logits)),
        valid_cells=valid_cells,
        log_uniform_normalizers=normalizers,
        template_usable=usable,
    )


def _bilinear_sample_per_template_maps(
    *,
    maps: torch.Tensor,
    valid_cells: torch.Tensor,
    projected_xy: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one full-image map for each observation without map replication."""

    values = torch.as_tensor(maps)
    valid = torch.as_tensor(valid_cells, dtype=torch.bool, device=values.device)
    projected = torch.as_tensor(projected_xy, dtype=values.dtype, device=values.device)
    if (
        values.ndim != 3
        or valid.shape != values.shape
        or projected.ndim != 3
        or projected.shape[1:] != (values.shape[0], 2)
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("per-template map sampling inputs are invalid")
    batch, count = int(projected.shape[0]), int(projected.shape[1])
    height, width = int(values.shape[1]), int(values.shape[2])
    safe_projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.clamp(
        safe_projected[..., 0] * float(width - 1) / float(image_width - 1),
        0.0,
        float(width - 1),
    )
    y = torch.clamp(
        safe_projected[..., 1] * float(height - 1) / float(image_height - 1),
        0.0,
        float(height - 1),
    )
    x0 = torch.floor(x).to(dtype=torch.long)
    y0 = torch.floor(y).to(dtype=torch.long)
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    dx = x - x0.to(dtype=values.dtype)
    dy = y - y0.to(dtype=values.dtype)
    template_rows = torch.arange(count, device=values.device)[None].expand(batch, -1)
    top_left = values[template_rows, y0, x0]
    top_right = values[template_rows, y0, x1]
    bottom_left = values[template_rows, y1, x0]
    bottom_right = values[template_rows, y1, x1]
    top = (1.0 - dx) * top_left + dx * top_right
    bottom = (1.0 - dx) * bottom_left + dx * bottom_right
    sampled = (1.0 - dy) * top + dy * bottom
    sampled_valid = (
        valid[template_rows, y0, x0]
        & valid[template_rows, y0, x1]
        & valid[template_rows, y1, x0]
        & valid[template_rows, y1, x1]
        & torch.isfinite(projected).all(dim=2)
    )
    return sampled, sampled_valid


def sample_context_position_log_ratios(
    *,
    maps: ContextPositionLikelihoodMaps,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed context likelihood maps and return neutral unknown evidence."""

    projected = torch.as_tensor(projected_xy, dtype=maps.logits.dtype, device=maps.logits.device)
    geometric_valid = torch.as_tensor(
        projection_valid, dtype=torch.bool, device=maps.logits.device
    )
    if projected.shape[:2] != geometric_valid.shape or projected.shape[1:] != (
        maps.logits.shape[0],
        2,
    ):
        raise ValueError("context position-log-ratio projection inputs are invalid")
    sampled, sampled_valid = _bilinear_sample_per_template_maps(
        maps=maps.logits,
        valid_cells=maps.valid_cells,
        projected_xy=projected,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    evidence_valid = (
        geometric_valid
        & sampled_valid
        & maps.template_usable[None]
    )
    ratios = sampled - maps.log_uniform_normalizers[None]
    return torch.where(evidence_valid, ratios, torch.zeros_like(ratios)), evidence_valid


def project_simple_radial_torch(
    xyz: torch.Tensor,
    poses_w2c: torch.Tensor,
    *,
    focal_length: float,
    principal_x: float,
    principal_y: float,
    radial_k: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a fixed point set for batched COLMAP ``SIMPLE_RADIAL`` cameras."""

    points = torch.as_tensor(xyz)
    poses = torch.as_tensor(poses_w2c)
    if points.ndim != 2 or points.shape[1] != 3 or poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("batched SIMPLE_RADIAL projection tensors are incompatible")
    if points.device != poses.device or points.dtype != poses.dtype:
        raise ValueError("projection points and poses must share device and dtype")
    if int(image_width) <= 1 or int(image_height) <= 1 or not np.isfinite(
        [focal_length, principal_x, principal_y, radial_k]
    ).all():
        raise ValueError("SIMPLE_RADIAL camera parameters are invalid")
    camera = torch.einsum("bij,nj->bni", poses[:, :3, :3], points) + poses[:, None, :3, 3]
    z = camera[..., 2]
    safe_z = torch.where(torch.abs(z) > 1e-8, z, torch.ones_like(z))
    x = camera[..., 0] / safe_z
    y = camera[..., 1] / safe_z
    radial = 1.0 + float(radial_k) * (x.square() + y.square())
    projected = torch.stack(
        [
            float(focal_length) * x * radial + float(principal_x),
            float(focal_length) * y * radial + float(principal_y),
        ],
        dim=-1,
    )
    valid = (
        torch.isfinite(projected).all(dim=-1)
        & (z > 1e-8)
        & (projected[..., 0] >= 0.0)
        & (projected[..., 0] <= float(image_width - 1))
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] <= float(image_height - 1))
    )
    return projected, valid


def normalized_position_log_ratios_at_positions(
    *,
    query_grid: torch.Tensor,
    support_descriptors: torch.Tensor,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperature: float,
) -> torch.Tensor:
    """Return log likelihood ratios at arbitrary query positions.

    For support descriptor ``s`` the numerator is its softmax likelihood at a
    projected query position.  The denominator is the same descriptor's fixed
    full-query-grid partition function, divided by the number of grid cells.
    Invalid projections are an explicit neutral likelihood ratio of one.
    """

    grid = torch.as_tensor(query_grid)
    support = torch.as_tensor(support_descriptors)
    projected = torch.as_tensor(projected_xy)
    valid = torch.as_tensor(projection_valid)
    if (
        grid.ndim != 3
        or support.ndim != 2
        or projected.ndim != 3
        or valid.ndim != 2
        or grid.shape[2] != support.shape[1]
        or projected.shape[:2] != valid.shape
        or projected.shape[1] != support.shape[0]
        or len({grid.device, support.device, projected.device, valid.device}) != 1
        or len({grid.dtype, support.dtype, projected.dtype}) != 1
    ):
        raise ValueError("normalized position-likelihood tensors are incompatible")
    if int(image_width) <= 1 or int(image_height) <= 1 or float(temperature) <= 0.0:
        raise ValueError("normalized position-likelihood geometry is invalid")
    height, width, dimension = grid.shape
    normalized_grid = F.normalize(grid.reshape(-1, dimension), p=2, dim=1)
    normalized_support = F.normalize(support, p=2, dim=1)
    denominator_logits = normalized_support @ normalized_grid.T
    denominator = torch.logsumexp(denominator_logits / float(temperature), dim=1)
    denominator -= float(np.log(float(height * width)))
    map_tensor = grid.permute(2, 0, 1).unsqueeze(0)
    batch = projected.shape[0]
    normalized_xy = torch.stack(
        [
            2.0 * projected[..., 0] / float(image_width - 1) - 1.0,
            2.0 * projected[..., 1] / float(image_height - 1) - 1.0,
        ],
        dim=-1,
    ).reshape(batch, projected.shape[1], 1, 2)
    sampled = F.grid_sample(
        map_tensor.expand(batch, -1, -1, -1),
        normalized_xy,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, :, 0].transpose(1, 2)
    sampled = F.normalize(sampled, p=2, dim=2)
    numerator = torch.sum(sampled * normalized_support[None], dim=2) / float(temperature)
    ratios = numerator - denominator[None]
    # An off-image/depth-invalid observation is unknown.  It must not become
    # negative evidence merely because a tested pose cannot project it.
    return torch.where(valid, ratios, torch.zeros_like(ratios))


def normalized_position_log_ratios(
    *,
    query_grid: torch.Tensor,
    support_descriptors: torch.Tensor,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperature: float,
) -> torch.Tensor:
    """Return full-image-normalized ratios for one projected point per template."""

    return normalized_position_log_ratios_at_positions(
        query_grid=query_grid,
        support_descriptors=support_descriptors,
        projected_xy=projected_xy,
        projection_valid=projection_valid,
        image_width=int(image_width),
        image_height=int(image_height),
        temperature=float(temperature),
    )


def fit_fixed_affine_maplet_transforms_torch(
    *,
    support_xy: torch.Tensor,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    topology: FixedAffineMapletTopology,
    neighbor_sigma_px: float,
    minimum_neighbors: int,
    maximum_condition_number: float,
    maximum_rmse_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit candidate-conditioned local support-to-query affine maps.

    Source anchors and neighbour identities are fixed by ``topology``.  The
    tested pose only supplies projected locations for those known 3-D points.
    A failed/degenerate fit is unknown evidence, not a fallback to a
    query-center geometric score.
    """

    source = torch.as_tensor(support_xy)
    projected = torch.as_tensor(projected_xy)
    valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=source.device)
    if (
        source.ndim != 2
        or source.shape[1] != 2
        or projected.ndim != 3
        or projected.shape[1:] != (source.shape[0], 2)
        or valid.shape != projected.shape[:2]
        or len({source.device, projected.device, valid.device}) != 1
        or len({source.dtype, projected.dtype}) != 1
        or int(minimum_neighbors) < 2
        or not np.isfinite(
            [neighbor_sigma_px, maximum_condition_number, maximum_rmse_px]
        ).all()
        or float(neighbor_sigma_px) <= 0.0
        or float(maximum_condition_number) <= 1.0
        or float(maximum_rmse_px) <= 0.0
    ):
        raise ValueError("fixed affine maplet fit inputs are incompatible")
    anchors = torch.as_tensor(
        topology.anchor_observation_indices, dtype=torch.long, device=source.device
    )
    neighbors = torch.as_tensor(
        topology.neighbor_observation_indices, dtype=torch.long, device=source.device
    )
    if (
        int(anchors.max()) >= source.shape[0]
        or int(neighbors.max()) >= source.shape[0]
        or int(neighbors.min()) < -1
        or int(minimum_neighbors) > topology.maximum_neighbor_count
    ):
        raise ValueError("fixed affine maplet topology is outside support observations")
    safe_neighbors = neighbors.clamp_min(0)
    neighbor_present = neighbors >= 0
    source_anchor = source.index_select(0, anchors)
    source_neighbors = source[safe_neighbors]
    source_offsets = source_neighbors - source_anchor[:, None, :]
    safe_projected = torch.where(valid[..., None], projected, torch.zeros_like(projected))
    projected_anchor = safe_projected.index_select(1, anchors)
    projected_neighbors = safe_projected[:, safe_neighbors]
    target_offsets = projected_neighbors - projected_anchor[:, :, None, :]
    anchor_valid = valid.index_select(1, anchors)
    neighbor_valid = valid[:, safe_neighbors]
    usable = anchor_valid[:, :, None] & neighbor_valid & neighbor_present[None]
    distances = torch.linalg.vector_norm(source_offsets, dim=2)
    weights = torch.exp(-0.5 * (distances / float(neighbor_sigma_px)).square())
    weights = weights[None] * usable.to(dtype=source.dtype)
    normal = torch.einsum("bmk,mki,mkj->bmij", weights, source_offsets, source_offsets)
    rhs = torch.einsum("bmk,mki,bmkj->bmij", weights, source_offsets, target_offsets)
    eigvals = torch.linalg.eigvalsh(normal)
    maximum_eigenvalue = eigvals[..., 1].clamp_min(1.0)
    regularizer = maximum_eigenvalue * 1e-6
    identity = torch.eye(2, dtype=source.dtype, device=source.device)
    coefficients = torch.linalg.solve(normal + regularizer[..., None, None] * identity, rhs)
    affine = coefficients.transpose(-2, -1)
    predicted_offsets = torch.einsum("mki,bmij->bmkj", source_offsets, coefficients)
    residual_squared = (predicted_offsets - target_offsets).square().sum(dim=3)
    weight_sum = weights.sum(dim=2)
    rmse = torch.sqrt(
        (weights * residual_squared).sum(dim=2) / weight_sum.clamp_min(1e-12)
    )
    active_neighbor_counts = usable.sum(dim=2)
    condition = maximum_eigenvalue / eigvals[..., 0].clamp_min(1e-12)
    nondegenerate = eigvals[..., 0] > maximum_eigenvalue * 1e-5
    affine_valid = (
        anchor_valid
        & (active_neighbor_counts >= int(minimum_neighbors))
        & nondegenerate
        & (condition <= float(maximum_condition_number))
        & (rmse <= float(maximum_rmse_px))
        & torch.isfinite(affine).flatten(start_dim=2).all(dim=2)
        & torch.isfinite(rmse)
        & (torch.abs(torch.linalg.det(affine)) > 1e-6)
    )
    return affine, projected_anchor, affine_valid, active_neighbor_counts, rmse


def aggregate_fixed_support_image_log_ratios(
    maplet_log_ratios: torch.Tensor,
    *,
    maplet_support_image_groups: torch.Tensor,
    support_image_count: int,
    fixed_support_image_priors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Marginalize fixed per-image evidence without dropping unknown views.

    ``maplet_log_ratios`` is one score for each fixed maplet.  A support image
    with no viable maplet remains an explicit neutral likelihood-ratio term;
    it is never removed from a pose-dependent denominator.  This is useful for
    both image-local descriptors and pair-conditioned dense descriptors.
    """

    ratios = torch.as_tensor(maplet_log_ratios)
    groups = torch.as_tensor(
        maplet_support_image_groups, dtype=torch.long, device=ratios.device
    ).reshape(-1)
    count = int(support_image_count)
    priors = torch.as_tensor(
        fixed_support_image_priors, dtype=ratios.dtype, device=ratios.device
    ).reshape(-1)
    if (
        ratios.ndim != 2
        or ratios.shape[1] == 0
        or groups.shape != (ratios.shape[1],)
        or count <= 0
        or priors.shape != (count,)
        or torch.any(groups < 0)
        or torch.any(groups >= count)
        or torch.any(~torch.isfinite(ratios))
        or torch.any(~torch.isfinite(priors))
        or torch.any(priors <= 0.0)
        or not bool(
            torch.isclose(
                priors.sum(), priors.new_tensor(1.0), rtol=1e-5, atol=1e-5
            )
        )
    ):
        raise ValueError("fixed support-image log-ratio aggregation is incompatible")
    group_sums = torch.zeros(
        (ratios.shape[0], count), dtype=ratios.dtype, device=ratios.device
    )
    group_sums.scatter_add_(1, groups[None].expand(ratios.shape[0], -1), ratios)
    group_counts = torch.bincount(groups, minlength=count).to(dtype=ratios.dtype)
    image_ratios = torch.where(
        group_counts[None] > 0.0,
        group_sums / group_counts[None].clamp_min(1.0),
        torch.zeros_like(group_sums),
    )
    return {
        "uniform_mixture": torch.logsumexp(image_ratios, dim=1)
        - float(np.log(float(count))),
        "prior_mixture": torch.logsumexp(torch.log(priors)[None] + image_ratios, dim=1),
    }


def score_affine_warped_maplet_patch_likelihoods(
    *,
    query_grid: torch.Tensor,
    support_patches: torch.Tensor,
    support_patch_valid: torch.Tensor,
    support_image_sizes: torch.Tensor,
    support_grid_size: int,
    affine_matrices: torch.Tensor,
    projected_anchor_xy: torch.Tensor,
    maplet_geometry_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperature: float,
    minimum_support_fraction: float,
    exclude_center: bool = True,
    maplet_support_image_groups: torch.Tensor | None = None,
    support_image_count: int | None = None,
    fixed_support_image_priors: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Score affine-warped real-image support patches against a query grid.

    Every patch descriptor has its own fixed full-query-image normalizer.  The
    affine warp changes only the sampled query location under a candidate
    pose.  Source-edge loss, invalid fits, and off-image patch cells contribute
    a neutral log-likelihood ratio of zero.  Optional support-image groups
    preserve per-view evidence before a fixed support-view mixture.
    """

    query = torch.as_tensor(query_grid)
    patches = torch.as_tensor(support_patches)
    patch_valid = torch.as_tensor(
        support_patch_valid, dtype=torch.bool, device=patches.device
    )
    image_sizes = torch.as_tensor(
        support_image_sizes, dtype=patches.dtype, device=patches.device
    )
    affine = torch.as_tensor(affine_matrices, dtype=patches.dtype, device=patches.device)
    anchor_xy = torch.as_tensor(
        projected_anchor_xy, dtype=patches.dtype, device=patches.device
    )
    geometry_valid = torch.as_tensor(
        maplet_geometry_valid, dtype=torch.bool, device=patches.device
    )
    image_groups = None
    image_count = None
    image_priors = None
    if maplet_support_image_groups is not None:
        if support_image_count is None or fixed_support_image_priors is None:
            raise ValueError("support-image maplet aggregation requires count and priors")
        image_groups = torch.as_tensor(
            maplet_support_image_groups, dtype=torch.long, device=patches.device
        ).reshape(-1)
        image_count = int(support_image_count)
        image_priors = torch.as_tensor(
            fixed_support_image_priors, dtype=patches.dtype, device=patches.device
        ).reshape(-1)
    elif support_image_count is not None or fixed_support_image_priors is not None:
        raise ValueError("support-image maplet aggregation inputs are incomplete")
    if (
        query.ndim != 3
        or patches.ndim != 4
        or patches.shape[0] == 0
        or patches.shape[1] != query.shape[2]
        or patches.shape[2] != patches.shape[3]
        or patches.shape[2] < 3
        or patches.shape[2] % 2 != 1
        or patch_valid.shape != patches.shape[:1] + patches.shape[2:]
        or image_sizes.shape != (patches.shape[0], 2)
        or affine.ndim != 4
        or affine.shape[1:] != (patches.shape[0], 2, 2)
        or anchor_xy.shape != affine.shape[:2] + (2,)
        or geometry_valid.shape != affine.shape[:2]
        or len({query.device, patches.device, image_sizes.device, affine.device, anchor_xy.device, geometry_valid.device}) != 1
        or len({query.dtype, patches.dtype, affine.dtype, anchor_xy.dtype}) != 1
        or int(support_grid_size) <= 1
        or not np.isfinite([temperature, minimum_support_fraction]).all()
        or float(temperature) <= 0.0
        or not 0.0 < float(minimum_support_fraction) <= 1.0
        or int(image_width) <= 1
        or int(image_height) <= 1
        or torch.any(image_sizes <= 1.0)
        or (
            image_groups is not None
            and (
                image_count is None
                or image_priors is None
                or image_count <= 0
                or image_groups.shape != (patches.shape[0],)
                or image_priors.shape != (image_count,)
                or torch.any(image_groups < 0)
                or torch.any(image_groups >= image_count)
                or torch.any(~torch.isfinite(image_priors))
                or torch.any(image_priors <= 0.0)
                or not bool(
                    torch.isclose(
                        image_priors.sum(),
                        image_priors.new_tensor(1.0),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )
            )
        )
    ):
        raise ValueError("affine maplet patch likelihood inputs are incompatible")
    maplet_count = int(patches.shape[0])
    window = int(patches.shape[2])
    radius = int(window // 2)
    context_valid = patch_valid.clone()
    if bool(exclude_center):
        context_valid[:, radius, radius] = False
    slots = int(window * window - int(bool(exclude_center)))
    template_counts = context_valid.sum(dim=(1, 2))
    template_usable = template_counts.to(dtype=patches.dtype) >= (
        float(minimum_support_fraction) * float(slots)
    )
    offsets = torch.arange(-radius, radius + 1, dtype=patches.dtype, device=patches.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    grid_offsets = torch.stack([offset_x, offset_y], dim=2).reshape(-1, 2)
    source_scales = torch.stack(
        [
            (image_sizes[:, 0] - 1.0) / float(int(support_grid_size) - 1),
            (image_sizes[:, 1] - 1.0) / float(int(support_grid_size) - 1),
        ],
        dim=1,
    )
    source_offsets = source_scales[:, None, :] * grid_offsets[None]
    query_positions = anchor_xy[:, :, None, :] + torch.einsum(
        "bmij,mpj->bmpi", affine, source_offsets
    )
    position_valid = (
        torch.isfinite(query_positions).all(dim=3)
        & (query_positions[..., 0] >= 0.0)
        & (query_positions[..., 0] <= float(image_width - 1))
        & (query_positions[..., 1] >= 0.0)
        & (query_positions[..., 1] <= float(image_height - 1))
        & geometry_valid[:, :, None]
        & context_valid.reshape(maplet_count, -1)[None]
        & template_usable[None, :, None]
    )
    templates = patches.permute(0, 2, 3, 1).reshape(-1, patches.shape[1])
    ratios = normalized_position_log_ratios_at_positions(
        query_grid=query,
        support_descriptors=templates,
        projected_xy=query_positions.reshape(query_positions.shape[0], -1, 2),
        projection_valid=position_valid.reshape(position_valid.shape[0], -1),
        image_width=int(image_width),
        image_height=int(image_height),
        temperature=float(temperature),
    ).reshape(query_positions.shape[0], maplet_count, -1)
    maplet_scores = ratios.sum(dim=2) / template_counts.to(dtype=patches.dtype)[None].clamp_min(1.0)
    maplet_scores = torch.where(
        template_usable[None], maplet_scores, torch.zeros_like(maplet_scores)
    )
    active_maplets = (
        geometry_valid
        & template_usable[None]
        & position_valid.any(dim=2)
    )
    summaries = summarize_track_log_ratios(maplet_scores)
    if image_groups is not None:
        if image_count is None or image_priors is None:  # pragma: no cover - guarded above
            raise RuntimeError("support-image maplet aggregation is incomplete")
        image_summaries = aggregate_fixed_support_image_log_ratios(
            maplet_scores,
            maplet_support_image_groups=image_groups,
            support_image_count=image_count,
            fixed_support_image_priors=image_priors,
        )
        summaries["support_image_uniform_mixture"] = image_summaries["uniform_mixture"]
        summaries["support_image_prior_mixture"] = image_summaries["prior_mixture"]
    return (
        summaries,
        active_maplets.sum(dim=1),
        template_usable.sum().expand(maplet_scores.shape[0]),
    )


def aggregate_view_log_ratios(
    observation_log_ratios: torch.Tensor,
    projection_valid: torch.Tensor,
    *,
    observation_track_groups: torch.Tensor,
    track_group_count: int,
    group_observation_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize fixed support views and report whether each track is visible."""

    ratios = torch.as_tensor(observation_log_ratios)
    valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=ratios.device)
    groups = torch.as_tensor(observation_track_groups, dtype=torch.long, device=ratios.device)
    counts = torch.as_tensor(group_observation_counts, dtype=ratios.dtype, device=ratios.device)
    if (
        ratios.ndim != 2
        or valid.shape != ratios.shape
        or groups.shape != (ratios.shape[1],)
        or counts.shape != (int(track_group_count),)
        or int(track_group_count) <= 0
        or int(groups.min()) < 0
        or int(groups.max()) >= int(track_group_count)
        or torch.any(counts <= 0)
    ):
        raise ValueError("support-view likelihood groups are invalid")
    # Ratios are clipped only for numerical stability.  The bounded range is
    # far beyond values produced by the feature temperatures used in practice.
    values = torch.exp(torch.clamp(ratios, min=-30.0, max=30.0))
    grouped = torch.zeros(
        (ratios.shape[0], int(track_group_count)), dtype=ratios.dtype, device=ratios.device
    )
    grouped.scatter_add_(1, groups[None].expand(ratios.shape[0], -1), values)
    track_ratios = torch.log(grouped / counts[None])
    active = torch.zeros(
        (ratios.shape[0], int(track_group_count)), dtype=torch.int32, device=ratios.device
    )
    active.scatter_add_(
        1,
        groups[None].expand(ratios.shape[0], -1),
        valid.to(dtype=torch.int32),
    )
    return track_ratios, active > 0


def aggregate_observation_group_log_ratios(
    observation_log_ratios: torch.Tensor,
    observation_valid: torch.Tensor,
    *,
    observation_groups: torch.Tensor,
    group_count: int,
    group_observation_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average fixed same-view observation evidence without cross-view mixing.

    Unlike :func:`aggregate_view_log_ratios`, this operates on the log
    likelihood ratios directly.  A group is a fixed support image or fixed
    support-image spatial block, so all observations in it share one real
    support view.  Invalid projections remain neutral and never trigger a
    fallback geometric score.
    """

    ratios = torch.as_tensor(observation_log_ratios)
    valid = torch.as_tensor(observation_valid, dtype=torch.bool, device=ratios.device)
    groups = torch.as_tensor(observation_groups, dtype=torch.long, device=ratios.device)
    counts = torch.as_tensor(
        group_observation_counts, dtype=ratios.dtype, device=ratios.device
    )
    if (
        ratios.ndim != 2
        or valid.shape != ratios.shape
        or groups.shape != (ratios.shape[1],)
        or counts.shape != (int(group_count),)
        or int(group_count) <= 0
        or int(groups.min()) < 0
        or int(groups.max()) >= int(group_count)
        or torch.any(counts <= 0)
        or not bool(torch.isfinite(ratios).all())
    ):
        raise ValueError("same-view observation likelihood groups are invalid")
    grouped = torch.zeros(
        (ratios.shape[0], int(group_count)), dtype=ratios.dtype, device=ratios.device
    )
    grouped.scatter_add_(
        1,
        groups[None].expand(ratios.shape[0], -1),
        torch.where(valid, ratios, torch.zeros_like(ratios)),
    )
    grouped /= counts[None]
    active = torch.zeros(
        (ratios.shape[0], int(group_count)), dtype=torch.int32, device=ratios.device
    )
    active.scatter_add_(
        1,
        groups[None].expand(ratios.shape[0], -1),
        valid.to(dtype=torch.int32),
    )
    return grouped, active > 0


def summarize_track_log_ratios(
    track_log_ratios: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Stable robust summaries over a fixed set of physical support tracks."""

    values = torch.as_tensor(track_log_ratios)
    if values.ndim != 2 or values.shape[1] == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("track likelihood ratios are invalid")
    count = values.shape[1]
    ordered, _ = torch.sort(values, dim=1)
    trim = int(np.floor(float(count) * 0.10))
    trimmed = ordered[:, trim : count - trim] if 2 * trim < count else ordered
    upper_begin = int(np.floor(float(count) * 0.75))
    return {
        "mean": torch.mean(values, dim=1),
        "median": torch.median(values, dim=1).values,
        "trimmed_mean_10": torch.mean(trimmed, dim=1),
        "top_quartile_mean": torch.mean(ordered[:, upper_begin:], dim=1),
    }


def score_multiscale_position_likelihoods(
    *,
    query_grids: Mapping[str, torch.Tensor],
    support_descriptors: Mapping[str, torch.Tensor],
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperatures: Mapping[str, float],
    observation_track_groups: torch.Tensor,
    track_group_count: int,
    group_observation_counts: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Score each feature family and their equal-log-ratio multiscale mixture."""

    names = tuple(query_grids)
    if not names or set(names) != set(support_descriptors) or set(names) != set(temperatures):
        raise ValueError("multiscale evidence names differ")
    per_scale_tracks: dict[str, torch.Tensor] = {}
    active: torch.Tensor | None = None
    for name in names:
        ratios = normalized_position_log_ratios(
            query_grid=query_grids[name],
            support_descriptors=support_descriptors[name],
            projected_xy=projected_xy,
            projection_valid=projection_valid,
            image_width=int(image_width),
            image_height=int(image_height),
            temperature=float(temperatures[name]),
        )
        tracks, local_active = aggregate_view_log_ratios(
            ratios,
            projection_valid,
            observation_track_groups=observation_track_groups,
            track_group_count=int(track_group_count),
            group_observation_counts=group_observation_counts,
        )
        per_scale_tracks[name] = tracks
        active = local_active if active is None else (active | local_active)
    if active is None:
        raise RuntimeError("multiscale evidence produced no feature family")
    output = {name: summarize_track_log_ratios(values) for name, values in per_scale_tracks.items()}
    output["multiscale_equal"] = summarize_track_log_ratios(
        torch.stack([per_scale_tracks[name] for name in names], dim=0).mean(dim=0)
    )
    return output, active.sum(dim=1), projection_valid.sum(dim=1)


def score_multiscale_observation_group_position_likelihoods(
    *,
    query_grids: Mapping[str, torch.Tensor],
    support_descriptors: Mapping[str, torch.Tensor],
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    image_width: int,
    image_height: int,
    temperatures: Mapping[str, float],
    observation_groups: torch.Tensor,
    group_count: int,
    group_observation_counts: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    """Score fixed same-view observation groups across feature families.

    This is intentionally distinct from track-level view marginalization.  A
    caller supplies fixed groups such as ``(support image, support-image
    block)``; each group directly averages log likelihood ratios from one
    real support view.  Consequently a candidate pose must explain several
    observations together instead of allowing every physical track to choose
    a different support image independently.
    """

    names = tuple(query_grids)
    if not names or set(names) != set(support_descriptors) or set(names) != set(temperatures):
        raise ValueError("multiscale evidence names differ")
    per_scale_groups: dict[str, torch.Tensor] = {}
    active: torch.Tensor | None = None
    for name in names:
        ratios = normalized_position_log_ratios(
            query_grid=query_grids[name],
            support_descriptors=support_descriptors[name],
            projected_xy=projected_xy,
            projection_valid=projection_valid,
            image_width=int(image_width),
            image_height=int(image_height),
            temperature=float(temperatures[name]),
        )
        grouped, local_active = aggregate_observation_group_log_ratios(
            ratios,
            projection_valid,
            observation_groups=observation_groups,
            group_count=int(group_count),
            group_observation_counts=group_observation_counts,
        )
        per_scale_groups[name] = grouped
        active = local_active if active is None else (active | local_active)
    if active is None:
        raise RuntimeError("same-view multiscale evidence produced no feature family")
    output = {
        name: summarize_track_log_ratios(values)
        for name, values in per_scale_groups.items()
    }
    output["multiscale_equal"] = summarize_track_log_ratios(
        torch.stack([per_scale_groups[name] for name in names], dim=0).mean(dim=0)
    )
    return output, active.sum(dim=1)
