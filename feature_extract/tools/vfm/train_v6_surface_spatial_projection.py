"""Train RADIO-final features to distinguish exact cells within a maplet.

The identity retrieval branch and this spatial branch are deliberately
separate. Training positives are the same canonical clean-2DGS surface cell
seen from disjoint trajectories. Negatives are maplet-balanced, so different
cells of the same local surface cannot be solved by maplet identity alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.train_v6_metric_encoder import _visibility
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    SurfaceSpatialProjection,
    SurfaceSpatialProjectionConfig,
    save_surface_spatial_projection,
)


@dataclass(frozen=True)
class SpatialView:
    image_id: str
    trajectory_id: str
    token_path: Path
    visible_rows: np.ndarray
    radio_xy: np.ndarray
    radio_shape: tuple[int, int, int]


@dataclass(frozen=True)
class QuerySurfaceSupport:
    query_index: int
    surface_rows: np.ndarray
    reference_starts: np.ndarray
    reference_counts: np.ndarray
    reference_indices: np.ndarray


@dataclass(frozen=True)
class ValidationEpisode:
    query_index: int
    rows: np.ndarray
    reference_view_indices: np.ndarray
    reference_mode_mask: np.ndarray


class RadioFeatureCache:
    def __init__(self, maximum_items: int = 32) -> None:
        self.maximum_items = max(1, int(maximum_items))
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()

    def load(self, path: Path) -> np.ndarray:
        key = Path(path)
        cached = self._items.pop(key, None)
        if cached is None:
            with np.load(key, allow_pickle=False) as data:
                cached = np.asarray(data["radio_final"]).copy()
            if cached.ndim != 3:
                raise ValueError(f"{key} RADIO-final must have shape CxHxW")
        self._items[key] = cached
        while len(self._items) > self.maximum_items:
            self._items.popitem(last=False)
        return cached


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--contributor_dirs", nargs="+", required=True)
    parser.add_argument("--initial_maplet_bank", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--reference_trajectory_ids",
        nargs="+",
        default=["seq1", "seq2", "seq4", "seq6", "seq7", "seq8"],
    )
    parser.add_argument(
        "--train_query_trajectory_ids",
        nargs="+",
        default=["seq9", "seq10", "seq12", "seq14"],
    )
    parser.add_argument(
        "--validation_trajectory_ids", nargs="+", default=["seq11"]
    )
    parser.add_argument(
        "--strict_holdout_trajectory_ids",
        nargs="+",
        default=["seq3", "seq5", "seq13"],
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation_every", type=int, default=50)
    parser.add_argument("--validation_episodes", type=int, default=32)
    parser.add_argument("--maplets_per_batch", type=int, default=12)
    parser.add_argument("--cells_per_maplet", type=int, default=8)
    parser.add_argument("--minimum_common_rows", type=int, default=64)
    parser.add_argument("--minimum_supported_maplets", type=int, default=4)
    parser.add_argument("--minimum_reference_trajectories", type=int, default=2)
    parser.add_argument("--reference_modes", type=int, default=4)
    parser.add_argument(
        "--output_dim",
        type=int,
        default=0,
        help=(
            "Spatial projection width. Zero preserves the initialization "
            "bank width; larger values complete its RADIO subspace with "
            "orthogonal residual directions."
        ),
    )
    parser.add_argument(
        "--local_hard_fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of each maplet batch drawn from one metric-local "
            "surface neighbourhood instead of only farthest/easy cells."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--within_maplet_loss_weight", type=float, default=1.0)
    parser.add_argument("--global_loss_weight", type=float, default=0.25)
    parser.add_argument("--orthogonality_weight", type=float, default=0.01)
    parser.add_argument("--initialization_weight", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--cache_items", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _lineage_keys() -> tuple[str, ...]:
    return (
        "geometry_source_sha256",
        "clean_geometry_source_sha256",
        "clean_source_index_sha256",
    )


def _load_views(
    directories: Sequence[Path],
    atlas: MapletFeatureAtlasBank,
    requested_trajectories: set[str],
    strict_holdout_trajectories: set[str],
) -> list[SpatialView]:
    geometry_metadata = dict(atlas.metadata or {})
    paths = sorted(
        {
            path.resolve()
            for directory in directories
            for path in Path(directory).glob("*.npz")
        }
    )
    views: dict[str, SpatialView] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            image_id = str(metadata["image_id"])
            trajectory_id = str(
                metadata.get("trajectory_id", image_id.split("/", 1)[0])
            )
            if trajectory_id in strict_holdout_trajectories:
                raise ValueError(
                    f"strict holdout contributor reached training: {image_id}"
                )
            if trajectory_id not in requested_trajectories:
                continue
            for key in _lineage_keys():
                expected = str(geometry_metadata.get(key, ""))
                observed = str(metadata.get(key, ""))
                if expected and observed != expected:
                    raise ValueError(
                        f"{path.name} contributor {key} differs from geometry"
                    )
            camera = ColmapCamera(
                camera_id=0,
                model_id=int(data["camera_model_id"]),
                width=int(data["camera_width"]),
                height=int(data["camera_height"]),
                params=tuple(
                    np.asarray(data["camera_params"], dtype=np.float64)
                ),
            )
            visible_rows, image_xy = _visibility(
                atlas,
                np.asarray(data["pose_w2c"], dtype=np.float64),
                camera,
                np.asarray(data["topk_ids"], dtype=np.int64),
                np.asarray(data["topk_weights"], dtype=np.float32),
            )
            token_path = Path(str(metadata["token_path"])).resolve()
        with np.load(token_path, allow_pickle=False) as token_data:
            radio_shape = tuple(
                int(value) for value in token_data["radio_final"].shape
            )
        if len(radio_shape) != 3:
            raise ValueError(f"{token_path} RADIO-final must have shape CxHxW")
        _, radio_height, radio_width = radio_shape
        radio_xy = np.stack(
            [
                (image_xy[:, 0] + 0.5)
                * radio_width
                / camera.width
                - 0.5,
                (image_xy[:, 1] + 0.5)
                * radio_height
                / camera.height
                - 0.5,
            ],
            axis=1,
        ).astype(np.float32)
        inside = (
            np.isfinite(radio_xy).all(axis=1)
            & (radio_xy[:, 0] >= 0.0)
            & (radio_xy[:, 0] <= radio_width - 1)
            & (radio_xy[:, 1] >= 0.0)
            & (radio_xy[:, 1] <= radio_height - 1)
        )
        view = SpatialView(
            image_id=image_id,
            trajectory_id=trajectory_id,
            token_path=token_path,
            visible_rows=np.asarray(visible_rows[inside], dtype=np.int64),
            radio_xy=np.asarray(radio_xy[inside], dtype=np.float32),
            radio_shape=radio_shape,
        )
        previous = views.get(image_id)
        if previous is not None:
            if previous.token_path != view.token_path:
                raise ValueError(
                    f"duplicate image has different token path: {image_id}"
                )
            continue
        views[image_id] = view
    return sorted(views.values(), key=lambda view: view.image_id)


def _build_query_supports(
    references: Sequence[SpatialView],
    queries: Sequence[SpatialView],
    *,
    flat_cells: int,
    minimum_common_rows: int,
    minimum_supported_maplets: int,
    minimum_reference_trajectories: int,
) -> list[QuerySurfaceSupport]:
    if int(minimum_reference_trajectories) < 1:
        raise ValueError("minimum reference trajectories must be positive")
    trajectory_ids = sorted(
        {reference.trajectory_id for reference in references}
    )
    trajectory_code = {
        trajectory_id: index
        for index, trajectory_id in enumerate(trajectory_ids)
    }
    if len(trajectory_code) > 62:
        raise ValueError("too many reference trajectories for support audit")
    reference_bits = np.asarray(
        [
            np.uint64(1) << np.uint64(trajectory_code[view.trajectory_id])
            for view in references
        ],
        dtype=np.uint64,
    )
    supports = []
    for query_index, query in enumerate(queries):
        row_blocks = []
        reference_blocks = []
        for reference_index, reference in enumerate(references):
            common = np.intersect1d(
                reference.visible_rows,
                query.visible_rows,
                assume_unique=True,
            )
            if common.size:
                row_blocks.append(common)
                reference_blocks.append(
                    np.full(
                        common.shape,
                        reference_index,
                        dtype=np.int32,
                    )
                )
        if not row_blocks:
            continue
        observation_rows = np.concatenate(row_blocks)
        observation_references = np.concatenate(reference_blocks)
        order = np.argsort(observation_rows, kind="stable")
        observation_rows = observation_rows[order]
        observation_references = observation_references[order]
        surface_rows, starts, counts = np.unique(
            observation_rows, return_index=True, return_counts=True
        )
        trajectory_masks = np.bitwise_or.reduceat(
            reference_bits[observation_references], starts
        )
        trajectory_counts = np.asarray(
            [
                bin(int(value)).count("1")
                for value in trajectory_masks.tolist()
            ],
            dtype=np.int32,
        )
        eligible = (
            (counts >= int(minimum_reference_trajectories))
            & (
                trajectory_counts
                >= int(minimum_reference_trajectories)
            )
        )
        surface_rows = surface_rows[eligible]
        starts = starts[eligible]
        counts = counts[eligible]
        if surface_rows.size < int(minimum_common_rows):
            continue
        _, maplet_cell_counts = np.unique(
            surface_rows // int(flat_cells), return_counts=True
        )
        if (
            int(np.sum(maplet_cell_counts >= 2))
            < int(minimum_supported_maplets)
        ):
            continue
        supports.append(
            QuerySurfaceSupport(
                query_index=query_index,
                surface_rows=np.asarray(surface_rows, dtype=np.int64),
                reference_starts=np.asarray(starts, dtype=np.int64),
                reference_counts=np.asarray(counts, dtype=np.int32),
                reference_indices=np.asarray(
                    observation_references, dtype=np.int32
                ),
            )
        )
    return supports


def _support_sampling_weights(
    supports: Sequence[QuerySurfaceSupport],
    queries: Sequence[SpatialView],
) -> np.ndarray:
    counts: dict[str, int] = {}
    for support in supports:
        trajectory = queries[support.query_index].trajectory_id
        counts[trajectory] = counts.get(trajectory, 0) + 1
    weights = np.asarray(
        [
            1.0 / counts[queries[support.query_index].trajectory_id]
            for support in supports
        ],
        dtype=np.float64,
    )
    return weights / np.sum(weights)


def _select_reference_modes(
    support: QuerySurfaceSupport,
    rows: np.ndarray,
    references: Sequence[SpatialView],
    *,
    maximum_modes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(support.surface_rows, rows)
    if (
        np.any(positions >= support.surface_rows.size)
        or not np.array_equal(support.surface_rows[positions], rows)
    ):
        raise ValueError("sampled row has no multi-view map support")
    maximum_modes = max(1, int(maximum_modes))
    selected = np.full(
        (len(rows), maximum_modes), -1, dtype=np.int32
    )
    mask = np.zeros((len(rows), maximum_modes), dtype=bool)
    for output_row, position in enumerate(positions.tolist()):
        begin = int(support.reference_starts[position])
        end = begin + int(support.reference_counts[position])
        candidates = support.reference_indices[begin:end]
        candidates = candidates[rng.permutation(candidates.size)]
        chosen = []
        used_trajectories: set[str] = set()
        for candidate in candidates.tolist():
            trajectory = references[int(candidate)].trajectory_id
            if trajectory in used_trajectories:
                continue
            chosen.append(int(candidate))
            used_trajectories.add(trajectory)
            if len(chosen) >= maximum_modes:
                break
        if len(chosen) < maximum_modes:
            for candidate in candidates.tolist():
                if int(candidate) in chosen:
                    continue
                chosen.append(int(candidate))
                if len(chosen) >= maximum_modes:
                    break
        selected[output_row, : len(chosen)] = chosen
        mask[output_row, : len(chosen)] = True
    if not np.all(np.sum(mask, axis=1) >= 1):
        raise AssertionError("surface cell has no selected reference mode")
    return selected, mask


def _farthest_rows(
    candidates: np.ndarray,
    xyz: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    count = min(int(count), int(candidates.size))
    if count >= candidates.size:
        return candidates[rng.permutation(candidates.size)]
    points = np.asarray(xyz[candidates], dtype=np.float64)
    selected = [int(rng.integers(candidates.size))]
    distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(distance))
        selected.append(next_index)
        distance = np.minimum(
            distance,
            np.sum((points - points[next_index]) ** 2, axis=1),
        )
    return candidates[np.asarray(selected, dtype=np.int64)]


def _spatial_projection_initialization(
    source: np.ndarray,
    *,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Complete a fitted RADIO subspace without discarding its span."""

    initial = np.asarray(source, dtype=np.float64)
    if initial.ndim != 2 or not np.all(np.isfinite(initial)):
        raise ValueError("initial RADIO projection must be a finite matrix")
    dimension = int(output_dim) if int(output_dim) > 0 else initial.shape[0]
    if dimension <= 0 or dimension > initial.shape[1]:
        raise ValueError("spatial projection output dimension is invalid")
    # QR removes harmless row-scale/non-orthogonality while preserving the
    # fitted PCA span used by the existing map artifacts.
    base = np.linalg.qr(initial.T, mode="reduced")[0].T
    if dimension <= base.shape[0]:
        return base[:dimension].astype(np.float32)
    rng = np.random.default_rng(int(seed))
    extra_count = dimension - base.shape[0]
    random = rng.standard_normal(
        (initial.shape[1], extra_count), dtype=np.float64
    )
    random -= base.T @ (base @ random)
    extra = np.linalg.qr(random, mode="reduced")[0].T
    if extra.shape[0] < extra_count:
        raise ValueError("could not complete RADIO projection subspace")
    return np.concatenate([base, extra[:extra_count]], axis=0).astype(
        np.float32
    )


def _balanced_surface_rows(
    common_rows: np.ndarray,
    *,
    flat_cells: int,
    xyz: np.ndarray,
    maplets_per_batch: int,
    cells_per_maplet: int,
    rng: np.random.Generator,
    local_hard_fraction: float = 0.5,
) -> np.ndarray:
    rows = np.asarray(common_rows, dtype=np.int64)
    _, inverse, counts = np.unique(
        rows // int(flat_cells), return_inverse=True, return_counts=True
    )
    eligible = np.flatnonzero(counts >= 2)
    if eligible.size < 2:
        raise ValueError("pair has fewer than two multi-cell maplets")
    chosen = rng.choice(
        eligible,
        size=min(int(maplets_per_batch), int(eligible.size)),
        replace=False,
    )
    sampled = []
    for group in chosen.tolist():
        candidates = rows[inverse == int(group)]
        count = min(int(cells_per_maplet), int(candidates.size))
        local_count = min(
            count,
            max(
                2,
                int(round(count * float(local_hard_fraction))),
            ),
        )
        points = np.asarray(xyz[candidates], dtype=np.float64)
        seed_index = int(rng.integers(candidates.size))
        distance = np.linalg.norm(
            points - points[seed_index][None], axis=1
        )
        local = candidates[
            np.argsort(distance, kind="mergesort")[:local_count]
        ]
        remaining = np.setdiff1d(candidates, local, assume_unique=False)
        far_count = count - int(local.size)
        far = (
            _farthest_rows(remaining, xyz, far_count, rng)
            if far_count > 0
            else np.zeros((0,), dtype=np.int64)
        )
        sampled.append(np.concatenate([local, far]))
    result = np.concatenate(sampled)
    if np.unique(result).size != result.size:
        raise AssertionError("surface-row sampler emitted duplicate cells")
    return result


def _coordinates_for_rows(view: SpatialView, rows: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(view.visible_rows, rows)
    if (
        np.any(positions >= view.visible_rows.size)
        or not np.array_equal(view.visible_rows[positions], rows)
    ):
        raise ValueError("pair contains an invisible canonical surface row")
    return view.radio_xy[positions]


def _sample_radio(
    cache: RadioFeatureCache,
    view: SpatialView,
    rows: np.ndarray,
) -> np.ndarray:
    feature = cache.load(view.token_path)
    points = _coordinates_for_rows(view, rows).astype(np.float64)
    channels, height, width = feature.shape
    if (channels, height, width) != view.radio_shape:
        raise ValueError("cached RADIO shape changed")
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (points[:, 0] - x0).astype(np.float32)
    dy = (points[:, 1] - y0).astype(np.float32)
    flat = feature.reshape(channels, -1)
    sampled = (
        np.asarray(flat[:, y0 * width + x0].T, dtype=np.float32)
        * ((1.0 - dx) * (1.0 - dy))[:, None]
        + np.asarray(flat[:, y0 * width + x1].T, dtype=np.float32)
        * (dx * (1.0 - dy))[:, None]
        + np.asarray(flat[:, y1 * width + x0].T, dtype=np.float32)
        * ((1.0 - dx) * dy)[:, None]
        + np.asarray(flat[:, y1 * width + x1].T, dtype=np.float32)
        * (dx * dy)[:, None]
    )
    return sampled


def _sample_reference_modes(
    cache: RadioFeatureCache,
    references: Sequence[SpatialView],
    rows: np.ndarray,
    reference_view_indices: np.ndarray,
    reference_mode_mask: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(reference_view_indices, dtype=np.int32)
    mask = np.asarray(reference_mode_mask, dtype=bool)
    if indices.shape != mask.shape or indices.shape[0] != len(rows):
        raise ValueError("reference-mode arrays differ from surface rows")
    channels = int(references[0].radio_shape[0])
    output = np.zeros(
        (indices.shape[0], indices.shape[1], channels), dtype=np.float32
    )
    for reference_index in np.unique(indices[mask]).tolist():
        locations = np.argwhere(mask & (indices == int(reference_index)))
        selected_rows = np.asarray(
            [rows[int(row)] for row, _mode in locations], dtype=np.int64
        )
        descriptors = _sample_radio(
            cache, references[int(reference_index)], selected_rows
        )
        output[locations[:, 0], locations[:, 1]] = descriptors
    return output


def _mode_marginalized_logits(
    model: SurfaceSpatialProjection,
    reference_modes: torch.Tensor,
    reference_mode_mask: torch.Tensor,
    query: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        reference_modes.ndim != 3
        or reference_mode_mask.shape != reference_modes.shape[:2]
    ):
        raise ValueError("reference modes must have shape NxKxC and NxK")
    count, mode_count, channels = reference_modes.shape
    map_descriptor = model(
        reference_modes.reshape(count * mode_count, channels)
    ).reshape(count, mode_count, -1)
    query_descriptor = model(query)
    cosine = torch.einsum(
        "ikc,jc->ikj", map_descriptor, query_descriptor
    )
    temperature = max(float(temperature), 1e-4)
    mode_count_per_cell = reference_mode_mask.sum(dim=1).clamp_min(1)
    mode_logits = cosine / temperature
    mode_logits = torch.where(
        reference_mode_mask[:, :, None],
        mode_logits,
        torch.full_like(mode_logits, -1e4),
    )
    logits = torch.logsumexp(mode_logits, dim=1) - torch.log(
        mode_count_per_cell.to(mode_logits.dtype)
    )[:, None]
    # Temperature-scaled log-mean-exp preserves the ranking while retaining
    # a cosine-like diagnostic range.
    similarity = logits * temperature
    return logits, similarity


def _projection_loss(
    model: SurfaceSpatialProjection,
    reference_modes: torch.Tensor,
    reference_mode_mask: torch.Tensor,
    query: torch.Tensor,
    maplet_ids: torch.Tensor,
    initial_projection: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, similarity = _mode_marginalized_logits(
        model,
        reference_modes,
        reference_mode_mask,
        query,
        temperature=float(args.temperature),
    )
    targets = torch.arange(logits.shape[0], device=logits.device)
    global_loss = 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )
    same_maplet = maplet_ids[:, None] == maplet_ids[None, :]
    local_logits = torch.where(
        same_maplet,
        logits,
        torch.full_like(logits, -1e4),
    )
    within_loss = 0.5 * (
        F.cross_entropy(local_logits, targets)
        + F.cross_entropy(local_logits.T, targets)
    )
    normalized_projection = F.normalize(model.projection, dim=1)
    gram = normalized_projection @ normalized_projection.T
    identity = torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    orthogonality = torch.mean((gram - identity) ** 2)
    initialization = torch.mean(
        (model.projection - initial_projection) ** 2
    )
    total = (
        float(args.within_maplet_loss_weight) * within_loss
        + float(args.global_loss_weight) * global_loss
        + float(args.orthogonality_weight) * orthogonality
        + float(args.initialization_weight) * initialization
    )
    return total, {
        "total": total,
        "within_maplet": within_loss,
        "global": global_loss,
        "orthogonality": orthogonality,
        "initialization": initialization,
        "similarity": similarity,
    }


def _episode_metrics(
    similarity: torch.Tensor,
    rows: np.ndarray,
    maplet_ids: torch.Tensor,
    xyz: np.ndarray,
) -> dict[str, float]:
    count = int(similarity.shape[0])
    targets = torch.arange(count, device=similarity.device)
    same_maplet = maplet_ids[:, None] == maplet_ids[None, :]
    local = torch.where(
        same_maplet,
        similarity,
        torch.full_like(similarity, -1e4),
    )
    predictions = torch.cat(
        [torch.argmax(local, dim=1), torch.argmax(local.T, dim=1)]
    )
    doubled_targets = torch.cat([targets, targets])
    row_array = np.asarray(rows, dtype=np.int64)
    predicted_rows = row_array[predictions.detach().cpu().numpy()]
    target_rows = row_array[doubled_targets.detach().cpu().numpy()]
    error = np.linalg.norm(
        np.asarray(xyz[predicted_rows], dtype=np.float64)
        - np.asarray(xyz[target_rows], dtype=np.float64),
        axis=1,
    )
    global_predictions = torch.cat(
        [torch.argmax(similarity, dim=1), torch.argmax(similarity.T, dim=1)]
    )
    positive = torch.diagonal(similarity)
    negative_mask = same_maplet & ~torch.eye(
        count, dtype=torch.bool, device=similarity.device
    )
    negative = similarity[negative_mask]
    return {
        "within_maplet_recall": float(
            torch.mean((predictions == doubled_targets).float()).item()
        ),
        "global_recall": float(
            torch.mean(
                (global_predictions == doubled_targets).float()
            ).item()
        ),
        "surface_error_median": float(np.median(error)),
        "surface_within_10cm": float(np.mean(error <= 0.10)),
        "surface_within_30cm": float(np.mean(error <= 0.30)),
        "positive_similarity": float(torch.mean(positive).item()),
        "same_maplet_negative_similarity": (
            float(torch.mean(negative).item())
            if negative.numel()
            else float("nan")
        ),
    }


def _make_validation_episodes(
    supports: Sequence[QuerySurfaceSupport],
    references: Sequence[SpatialView],
    xyz: np.ndarray,
    *,
    flat_cells: int,
    episode_count: int,
    maplets_per_batch: int,
    cells_per_maplet: int,
    reference_modes: int,
    seed: int,
    local_hard_fraction: float = 0.5,
) -> list[ValidationEpisode]:
    rng = np.random.default_rng(int(seed))
    count = max(1, int(episode_count))
    order = rng.permutation(len(supports))
    episodes = []
    for episode_index in range(count):
        support = supports[int(order[episode_index % len(order)])]
        rows = _balanced_surface_rows(
            support.surface_rows,
            flat_cells=int(flat_cells),
            xyz=xyz,
            maplets_per_batch=int(maplets_per_batch),
            cells_per_maplet=int(cells_per_maplet),
            rng=rng,
            local_hard_fraction=float(local_hard_fraction),
        )
        reference_view_indices, reference_mode_mask = (
            _select_reference_modes(
                support,
                rows,
                references,
                maximum_modes=int(reference_modes),
                rng=rng,
            )
        )
        episodes.append(
            ValidationEpisode(
                query_index=support.query_index,
                rows=rows,
                reference_view_indices=reference_view_indices,
                reference_mode_mask=reference_mode_mask,
            )
        )
    return episodes


@torch.no_grad()
def _validate(
    model: SurfaceSpatialProjection,
    references: Sequence[SpatialView],
    queries: Sequence[SpatialView],
    episodes: Sequence[ValidationEpisode],
    cache: RadioFeatureCache,
    xyz: np.ndarray,
    flat_cells: int,
    device: str,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for episode in episodes:
        rows = episode.rows
        reference_modes = torch.as_tensor(
            _sample_reference_modes(
                cache,
                references,
                rows,
                episode.reference_view_indices,
                episode.reference_mode_mask,
            ),
            dtype=torch.float32,
            device=device,
        )
        reference_mode_mask = torch.as_tensor(
            episode.reference_mode_mask, dtype=torch.bool, device=device
        )
        query = torch.as_tensor(
            _sample_radio(cache, queries[episode.query_index], rows),
            dtype=torch.float32,
            device=device,
        )
        _logits, similarity = _mode_marginalized_logits(
            model,
            reference_modes,
            reference_mode_mask,
            query,
            temperature=float(temperature),
        )
        maplet_ids = torch.as_tensor(
            rows // int(flat_cells), dtype=torch.long, device=device
        )
        metrics.append(
            _episode_metrics(similarity, rows, maplet_ids, xyz)
        )
    names = tuple(metrics[0])
    return {
        name: float(np.nanmean([row[name] for row in metrics]))
        for name in names
    }


def _selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    return (
        float(metrics["surface_error_median"]),
        -float(metrics["surface_within_10cm"]),
        -float(metrics["within_maplet_recall"]),
        -float(metrics["global_recall"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 0.0 <= float(args.local_hard_fraction) <= 1.0:
        raise ValueError("local_hard_fraction must lie in [0,1]")
    output_checkpoint = Path(args.output_checkpoint)
    summary_json = Path(args.summary_json)
    if (
        (output_checkpoint.exists() or summary_json.exists())
        and not bool(args.force)
    ):
        raise FileExistsError("refusing to overwrite spatial projection output")
    reference_ids = {str(value) for value in args.reference_trajectory_ids}
    train_query_ids = {
        str(value) for value in args.train_query_trajectory_ids
    }
    validation_ids = {
        str(value) for value in args.validation_trajectory_ids
    }
    holdout_ids = {
        str(value) for value in args.strict_holdout_trajectory_ids
    }
    split_sets = (reference_ids, train_query_ids, validation_ids, holdout_ids)
    for first in range(len(split_sets)):
        for second in range(first + 1, len(split_sets)):
            if split_sets[first] & split_sets[second]:
                raise ValueError("trajectory roles must be mutually disjoint")
    atlas_path = Path(args.atlas_geometry)
    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    requested = reference_ids | train_query_ids | validation_ids
    all_views = _load_views(
        [Path(value) for value in args.contributor_dirs],
        atlas,
        requested,
        holdout_ids,
    )
    references = [
        view for view in all_views if view.trajectory_id in reference_ids
    ]
    train_queries = [
        view for view in all_views if view.trajectory_id in train_query_ids
    ]
    validation_queries = [
        view for view in all_views if view.trajectory_id in validation_ids
    ]
    if (
        {view.trajectory_id for view in references} != reference_ids
        or {view.trajectory_id for view in train_queries} != train_query_ids
        or {
            view.trajectory_id for view in validation_queries
        }
        != validation_ids
    ):
        raise ValueError("one or more declared trajectory roles has no views")
    flat_cells = int(atlas.height * atlas.width)
    xyz = np.asarray(atlas.xyz, dtype=np.float32).reshape(-1, 3)
    train_supports = _build_query_supports(
        references,
        train_queries,
        flat_cells=flat_cells,
        minimum_common_rows=int(args.minimum_common_rows),
        minimum_supported_maplets=int(args.minimum_supported_maplets),
        minimum_reference_trajectories=int(
            args.minimum_reference_trajectories
        ),
    )
    validation_supports = _build_query_supports(
        references,
        validation_queries,
        flat_cells=flat_cells,
        minimum_common_rows=int(args.minimum_common_rows),
        minimum_supported_maplets=int(args.minimum_supported_maplets),
        minimum_reference_trajectories=int(
            args.minimum_reference_trajectories
        ),
    )
    if not train_supports or not validation_supports:
        raise ValueError("train or validation multi-view surface support is empty")
    initial_bank_path = Path(args.initial_maplet_bank)
    initial_bank = SurfaceRetrievalMapletBank.load_npz(initial_bank_path)
    if initial_bank.query_projection is None:
        raise ValueError("initial maplet bank has no RADIO query projection")
    source_projection = np.asarray(
        initial_bank.query_projection, dtype=np.float32
    )
    initial_projection = _spatial_projection_initialization(
        source_projection,
        output_dim=int(args.output_dim),
        seed=int(args.seed),
    )
    model = SurfaceSpatialProjection(
        SurfaceSpatialProjectionConfig(
            input_dim=int(initial_projection.shape[1]),
            output_dim=int(initial_projection.shape[0]),
        ),
        initial_projection=initial_projection,
    ).to(str(args.device))
    initial_tensor = torch.as_tensor(
        initial_projection, dtype=torch.float32, device=str(args.device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    cache = RadioFeatureCache(int(args.cache_items))
    validation_episodes = _make_validation_episodes(
        validation_supports,
        references,
        xyz,
        flat_cells=flat_cells,
        episode_count=int(args.validation_episodes),
        maplets_per_batch=int(args.maplets_per_batch),
        cells_per_maplet=int(args.cells_per_maplet),
        reference_modes=int(args.reference_modes),
        seed=int(args.seed) + 1,
        local_hard_fraction=float(args.local_hard_fraction),
    )
    metadata_base: dict[str, object] = {
        "stage": "v6_exact_surface_spatial_projection",
        "atlas_geometry_sha256": _file_sha256(atlas_path),
        "initial_maplet_bank_sha256": _file_sha256(initial_bank_path),
        "initial_projection_sha256": _array_sha256(initial_projection),
        **{
            key: str(dict(atlas.metadata or {}).get(key, ""))
            for key in _lineage_keys()
        },
        "reference_trajectory_ids": sorted(reference_ids),
        "train_query_trajectory_ids": sorted(train_query_ids),
        "validation_trajectory_ids": sorted(validation_ids),
        "strict_holdout_trajectory_ids": sorted(holdout_ids),
        "trajectory_roles_disjoint": True,
        "strict_holdout_used_for_training_or_selection": False,
        "positive_identity": "exact_canonical_clean_2dgs_surface_cell",
        "negative_sampling": (
            "maplet_balanced_metric_local_hard_and_farthest_cells"
        ),
        "local_hard_fraction": float(args.local_hard_fraction),
        "map_side_training_target": (
            "multi_trajectory_reference_mode_marginalization"
        ),
        "minimum_reference_trajectories": int(
            args.minimum_reference_trajectories
        ),
        "reference_modes_per_surface_cell": int(args.reference_modes),
        "vfm_layer": "radio_final",
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_pairwise_image_matching": False,
    }
    baseline = _validate(
        model,
        references,
        validation_queries,
        validation_episodes,
        cache,
        xyz,
        flat_cells,
        str(args.device),
        float(args.temperature),
    )
    best_step = 0
    best_metrics = baseline
    best_key = _selection_key(baseline)
    save_surface_spatial_projection(
        output_checkpoint,
        model,
        {**metadata_base, "best_step": 0, "best_validation_metrics": baseline},
    )
    history: list[dict[str, object]] = [
        {"step": 0, "validation": baseline}
    ]
    print(json.dumps(history[-1], sort_keys=True), flush=True)
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    support_weights = _support_sampling_weights(
        train_supports, train_queries
    )
    for step in range(1, int(args.steps) + 1):
        support = train_supports[
            int(rng.choice(len(train_supports), p=support_weights))
        ]
        rows = _balanced_surface_rows(
            support.surface_rows,
            flat_cells=flat_cells,
            xyz=xyz,
            maplets_per_batch=int(args.maplets_per_batch),
            cells_per_maplet=int(args.cells_per_maplet),
            rng=rng,
            local_hard_fraction=float(args.local_hard_fraction),
        )
        reference_view_indices, reference_mode_mask_numpy = (
            _select_reference_modes(
                support,
                rows,
                references,
                maximum_modes=int(args.reference_modes),
                rng=rng,
            )
        )
        reference_modes = torch.as_tensor(
            _sample_reference_modes(
                cache,
                references,
                rows,
                reference_view_indices,
                reference_mode_mask_numpy,
            ),
            dtype=torch.float32,
            device=str(args.device),
        )
        reference_mode_mask = torch.as_tensor(
            reference_mode_mask_numpy,
            dtype=torch.bool,
            device=str(args.device),
        )
        query = torch.as_tensor(
            _sample_radio(
                cache, train_queries[support.query_index], rows
            ),
            dtype=torch.float32,
            device=str(args.device),
        )
        maplet_ids = torch.as_tensor(
            rows // flat_cells,
            dtype=torch.long,
            device=str(args.device),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = _projection_loss(
            model,
            reference_modes,
            reference_mode_mask,
            query,
            maplet_ids,
            initial_tensor,
            args,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if (
            step == 1
            or step % int(args.validation_every) == 0
            or step == int(args.steps)
        ):
            validation = _validate(
                model,
                references,
                validation_queries,
                validation_episodes,
                cache,
                xyz,
                flat_cells,
                str(args.device),
                float(args.temperature),
            )
            row: dict[str, object] = {
                "step": step,
                "train_total": float(components["total"].detach().item()),
                "train_within_maplet": float(
                    components["within_maplet"].detach().item()
                ),
                "train_global": float(
                    components["global"].detach().item()
                ),
                "validation": validation,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            selection_key = _selection_key(validation)
            if selection_key < best_key:
                best_key = selection_key
                best_step = step
                best_metrics = validation
                save_surface_spatial_projection(
                    output_checkpoint,
                    model,
                    {
                        **metadata_base,
                        "best_step": int(best_step),
                        "best_validation_metrics": best_metrics,
                    },
                )
    report = {
        **metadata_base,
        "output_checkpoint": str(output_checkpoint),
        "reference_view_count": len(references),
        "train_query_view_count": len(train_queries),
        "validation_query_view_count": len(validation_queries),
        "train_query_support_count": len(train_supports),
        "validation_query_support_count": len(validation_supports),
        "validation_episode_count": len(validation_episodes),
        "input_dim": int(initial_projection.shape[1]),
        "output_dim": int(initial_projection.shape[0]),
        "steps": int(args.steps),
        "best_step": int(best_step),
        "baseline_validation_metrics": baseline,
        "best_validation_metrics": best_metrics,
        "selection_order": [
            "minimize_surface_error_median",
            "maximize_surface_within_10cm",
            "maximize_within_maplet_recall",
            "maximize_global_recall",
        ],
        "history": history,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
