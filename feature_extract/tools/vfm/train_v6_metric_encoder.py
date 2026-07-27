"""Train the V6 phase-preserving metric encoder with exact atlas identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.atlas_renderer import (
    render_selected_maplet_atlases,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    V6MetricEncoder,
    V6MetricEncoderConfig,
    load_v6_metric_encoder,
    save_v6_metric_encoder,
)
from feature_extract.vfm.localization_v6.metric_training import (
    analytic_one_step_pose_loss,
    local_correlation_training_loss,
)
from feature_extract.vfm.localization_v6.se3_update import (
    projection_jacobian,
    se3_exp,
)


@dataclass(frozen=True)
class TrainingView:
    image_id: str
    trajectory_id: str
    rgb: torch.Tensor
    radio: torch.Tensor
    pose_w2c: np.ndarray
    camera: ColmapCamera
    visible_rows: np.ndarray
    image_xy: np.ndarray
    clean_surface_mask: np.ndarray


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--train_contributor_dir", required=True)
    parser.add_argument("--validation_contributor_dir", required=True)
    parser.add_argument("--train_trajectory_ids", nargs="*", default=[])
    parser.add_argument(
        "--validation_trajectory_ids", nargs="*", default=[]
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation_every", type=int, default=50)
    parser.add_argument("--validation_episodes", type=int, default=48)
    parser.add_argument("--samples_per_episode", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--initial_checkpoint", default="")
    parser.add_argument("--frozen_metric_atlas", default="")
    parser.add_argument("--frozen_metric_atlas_middle", default="")
    parser.add_argument("--frozen_metric_atlas_coarse", default="")
    parser.add_argument(
        "--legacy_point_pair_training",
        action="store_true",
        help="Disable frozen-atlas render/correlate training (diagnostic only).",
    )
    return parser.parse_args(argv)


def _load_cache_paths(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npz"))


def _visibility(
    atlas: MapletFeatureAtlasBank,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    topk_ids: np.ndarray,
    topk_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_rows = np.flatnonzero(atlas.valid_mask.reshape(-1))
    xyz = atlas.xyz.reshape(-1, 3)[valid_rows]
    primitive = atlas.primitive_ids.reshape(-1)[valid_rows]
    xy, depth = project_world_points(xyz, pose_w2c, camera)
    height, width = topk_ids.shape[:2]
    grid = np.stack(
        [
            (xy[:, 0] + 0.5) * width / camera.width - 0.5,
            (xy[:, 1] + 0.5) * height / camera.height - 0.5,
        ],
        axis=1,
    )
    inside = (
        np.isfinite(grid).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (grid[:, 0] >= 0.0)
        & (grid[:, 0] <= width - 1)
        & (grid[:, 1] >= 0.0)
        & (grid[:, 1] <= height - 1)
    )
    candidate = np.flatnonzero(inside)
    points = grid[candidate]
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    expected = primitive[candidate]
    weight = np.zeros((candidate.size,), dtype=np.float64)
    for px, py, footprint in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        identity = topk_ids[py, px] == expected[:, None]
        weight += footprint * np.sum(
            np.where(identity, topk_weights[py, px], 0.0), axis=1
        )
    accepted = candidate[weight >= 0.01]
    return valid_rows[accepted], xy[accepted].astype(np.float32)


def _load_views(
    directory: Path,
    atlas: MapletFeatureAtlasBank,
    image_root: Path,
    trajectory_ids: Sequence[str] = (),
) -> list[TrainingView]:
    views = []
    requested = {str(value) for value in trajectory_ids}
    geometry_metadata = dict(atlas.metadata or {})
    lineage_keys = (
        "geometry_source_sha256",
        "clean_geometry_source_sha256",
        "clean_source_index_sha256",
    )
    for path in _load_cache_paths(directory):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            for key in lineage_keys:
                expected = str(geometry_metadata.get(key, ""))
                observed = str(metadata.get(key, ""))
                if expected and observed != expected:
                    raise ValueError(
                        f"{path.name} contributor {key} differs from atlas"
                    )
            image_id = str(metadata["image_id"])
            trajectory_id = str(
                metadata.get("trajectory_id", image_id.split("/", 1)[0])
            )
            if requested and trajectory_id not in requested:
                continue
            token_path = Path(str(metadata["token_path"]))
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
            camera = ColmapCamera(
                camera_id=0,
                model_id=int(data["camera_model_id"]),
                width=int(data["camera_width"]),
                height=int(data["camera_height"]),
                params=tuple(np.asarray(data["camera_params"], dtype=np.float64)),
            )
            topk_ids = np.asarray(data["topk_ids"], dtype=np.int64)
            topk_weights = np.asarray(
                data["topk_weights"], dtype=np.float32
            )
            rows, xy = _visibility(
                atlas,
                pose,
                camera,
                topk_ids,
                topk_weights,
            )
            clean_surface_mask = np.any(
                (topk_ids >= 0) & (topk_weights > 1e-6), axis=2
            )
        image = Image.open(image_root / image_id).convert("RGB").resize(
            (camera.width, camera.height), Image.Resampling.BILINEAR
        )
        rgb_array = np.asarray(image, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgb_array.transpose(2, 0, 1).copy())
        with np.load(token_path, allow_pickle=False) as token_data:
            radio = torch.from_numpy(
                np.asarray(token_data["radio_final"], dtype=np.float32)
            )
        views.append(
            TrainingView(
                image_id=image_id,
                trajectory_id=trajectory_id,
                rgb=rgb,
                radio=radio,
                pose_w2c=pose,
                camera=camera,
                visible_rows=rows,
                image_xy=xy,
                clean_surface_mask=clean_surface_mask,
            )
        )
    return views


def _pairs(
    views: list[TrainingView], *, allow_cross_trajectory: bool = False
) -> list[tuple[int, int, np.ndarray]]:
    result = []
    for first in range(len(views)):
        for second in range(first + 1, len(views)):
            if (
                not allow_cross_trajectory
                and views[first].trajectory_id != views[second].trajectory_id
            ):
                continue
            common = np.intersect1d(
                views[first].visible_rows,
                views[second].visible_rows,
                assume_unique=True,
            )
            if common.size >= 32:
                result.append((first, second, common))
    return result


def _single_view_episodes(
    views: list[TrainingView], *, minimum_visible_rows: int = 32
) -> list[tuple[int, int, np.ndarray]]:
    """Build inference-matched episodes without an artificial second image.

    The deployed map descriptor comes from the frozen atlas, not another
    training image.  Intersecting two query views therefore discards valid
    surface support and biases which trajectories can appear as the query.
    """

    return [
        (index, index, np.asarray(view.visible_rows, dtype=np.int64))
        for index, view in enumerate(views)
        if int(view.visible_rows.size) >= int(minimum_visible_rows)
    ]


def _coordinates_for_rows(view: TrainingView, rows: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(view.visible_rows, rows)
    if not np.array_equal(view.visible_rows[positions], rows):
        raise ValueError("pair contains invisible atlas rows")
    return view.image_xy[positions]


def _grid_xy(image_xy: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(image_xy, dtype=np.float32) + 0.5) / float(scale) - 0.5


def _clean_surface_feature_mask(
    view: TrainingView, *, width: int, height: int
) -> np.ndarray:
    """Resample declared clean-2DGS support onto a feature grid."""

    yy, xx = np.meshgrid(
        np.arange(int(height), dtype=np.float64),
        np.arange(int(width), dtype=np.float64),
        indexing="ij",
    )
    mask_height, mask_width = view.clean_surface_mask.shape
    contributor_x = np.clip(
        np.rint((xx + 0.5) * mask_width / int(width) - 0.5),
        0,
        mask_width - 1,
    ).astype(np.int64)
    contributor_y = np.clip(
        np.rint((yy + 0.5) * mask_height / int(height) - 0.5),
        0,
        mask_height - 1,
    ).astype(np.int64)
    return np.asarray(
        view.clean_surface_mask[contributor_y, contributor_x], dtype=bool
    )


def _unmapped_query_grid(
    view: TrainingView, *, width: int, height: int
) -> np.ndarray:
    """Feature-grid cells with no declared clean-2DGS contributor."""

    unmapped = ~_clean_surface_feature_mask(
        view, width=int(width), height=int(height)
    )
    rows_y, rows_x = np.nonzero(unmapped)
    return np.stack([rows_x, rows_y], axis=1).astype(np.float32)


def _query_matchable_at_grid(
    view: TrainingView,
    grid_xy: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Label query locations independently from map/query pair validity."""

    coordinates = np.asarray(grid_xy, dtype=np.float64)
    inside = (
        np.isfinite(coordinates).all(axis=1)
        & (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= int(width) - 1)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= int(height) - 1)
    )
    output = np.zeros((coordinates.shape[0],), dtype=bool)
    if not np.any(inside):
        return output
    rows = np.flatnonzero(inside)
    x = np.clip(
        np.rint(coordinates[rows, 0]).astype(np.int64), 0, int(width) - 1
    )
    y = np.clip(
        np.rint(coordinates[rows, 1]).astype(np.int64), 0, int(height) - 1
    )
    clean = _clean_surface_feature_mask(
        view, width=int(width), height=int(height)
    )
    output[rows] = clean[y, x]
    return output


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if labels.size == 0 or not np.any(labels):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked]) / np.sum(ranked))


def _inference_matched_loss(
    *,
    query_output: dict[str, torch.Tensor],
    query_view: TrainingView,
    frozen_atlas: MapletFeatureAtlasBank,
    common_rows: np.ndarray,
    proposal_pose: np.ndarray,
    perturbation: np.ndarray,
    feature_key: str,
    scale: float,
    radius: int,
    samples: int,
    rng: np.random.Generator,
    device: str,
) -> dict[str, torch.Tensor]:
    """Train from the exact atlas renderer used by localization."""

    flat_cells = frozen_atlas.height * frozen_atlas.width
    maplet_rows, counts = np.unique(
        np.asarray(common_rows, dtype=np.int64) // flat_cells,
        return_counts=True,
    )
    valid_maplets = frozen_atlas.valid_mask[maplet_rows].any(axis=(1, 2))
    maplet_rows = maplet_rows[valid_maplets]
    counts = counts[valid_maplets]
    maplet_rows = maplet_rows[np.argsort(-counts)]
    center_xy, center_depth = project_world_points(
        frozen_atlas.centers, query_view.pose_w2c, query_view.camera
    )
    support = np.sum(frozen_atlas.valid_mask, axis=(1, 2))
    visible = (
        np.isfinite(center_xy).all(axis=1)
        & np.isfinite(center_depth)
        & (center_depth > 0.0)
        & (center_xy[:, 0] >= 0.0)
        & (center_xy[:, 0] < query_view.camera.width)
        & (center_xy[:, 1] >= 0.0)
        & (center_xy[:, 1] < query_view.camera.height)
        & (support > 0)
    )
    visible_rows = np.flatnonzero(visible)
    visible_rows = visible_rows[np.argsort(-support[visible_rows])]
    # Include visible atlas coverage even when exact contributor-row
    # intersection is sparse; the renderer and z-buffer decide final support.
    maplet_rows = np.asarray(
        list(dict.fromkeys(np.r_[maplet_rows, visible_rows].tolist()))[:18],
        dtype=np.int64,
    )
    if maplet_rows.size < 2:
        raise ValueError("frozen atlas has insufficient visible maplets")
    query_features = query_output[feature_key]
    height, width = query_features.shape[-2:]
    rendered = render_selected_maplet_atlases(
        frozen_atlas,
        frozen_atlas.maplet_ids[maplet_rows],
        proposal_pose,
        query_view.camera,
        width=width,
        height=height,
    )
    yy, xx = np.nonzero(rendered.mask)
    if yy.size < 6:
        raise ValueError("frozen atlas render has insufficient covered cells")
    if yy.size > int(samples):
        selected = np.linspace(0, yy.size - 1, int(samples), dtype=np.int64)
        yy, xx = yy[selected], xx[selected]
    xyz = rendered.xyz[yy, xx]
    descriptors = rendered.feature[:, yy, xx].T.astype(np.float32)
    map_uncertainty = rendered.uncertainty[yy, xx].astype(np.float32)
    mode_descriptors = (
        rendered.mode_feature[:, :, yy, xx].transpose(2, 0, 1).astype(np.float32)
        if rendered.mode_feature is not None
        else None
    )
    mode_log_prior = (
        rendered.mode_log_prior[:, yy, xx].T.astype(np.float32)
        if rendered.mode_log_prior is not None
        else None
    )
    proposal_grid = np.stack([xx, yy], axis=1).astype(np.float32)
    target_image_xy, target_depth = project_world_points(
        xyz, query_view.pose_w2c, query_view.camera
    )
    target_grid = _grid_xy(target_image_xy, scale)
    positive = (
        np.isfinite(target_grid).all(axis=1)
        & np.isfinite(target_depth)
        & (target_depth > 0.0)
        & (target_grid[:, 0] >= 0.0)
        & (target_grid[:, 0] <= width - 1)
        & (target_grid[:, 1] >= 0.0)
        & (target_grid[:, 1] <= height - 1)
    )
    # A point can be in front of the camera yet be hidden by an unselected
    # 2DGS surface.  Use the query contributor buffer only as a training label:
    # runtime rendering still receives no ground-truth depth or visibility.
    surface_ids = np.asarray(rendered.surface_id[yy, xx], dtype=np.int64)
    exactly_visible = np.isin(
        surface_ids,
        np.asarray(query_view.visible_rows, dtype=np.int64),
        assume_unique=False,
    )
    positive &= (surface_ids >= 0) & exactly_visible
    # Structured pair-null negatives over the actually rendered atlas.
    negative = rng.random(yy.size) < 0.20
    permutation = np.arange(yy.size, dtype=np.int64)
    point_maplets = rendered.maplet_id[yy, xx]
    for index in np.flatnonzero(negative):
        candidates = np.delete(np.arange(yy.size), index)
        draw = float(rng.random())
        if draw < 0.20:
            pass
        elif draw < 0.50:
            same = candidates[point_maplets[candidates] == point_maplets[index]]
            if same.size:
                distance = np.linalg.norm(
                    np.stack([xx[same] - xx[index], yy[same] - yy[index]], axis=1),
                    axis=1,
                )
                candidates = same[np.argsort(distance)[: min(8, same.size)]]
        else:
            other = candidates[point_maplets[candidates] != point_maplets[index]]
            if other.size:
                similarity = descriptors[other] @ descriptors[index]
                candidates = other[np.argsort(-similarity)[: min(8, other.size)]]
        permutation[index] = int(rng.choice(candidates))
    descriptors[negative] = descriptors[permutation[negative]]
    if mode_descriptors is not None:
        mode_descriptors[negative] = mode_descriptors[permutation[negative]]
        mode_log_prior[negative] = mode_log_prior[permutation[negative]]
    positive &= ~negative
    requested_unmatchable = rng.random(yy.size) < 0.10
    query_unmatchable = np.zeros((yy.size,), dtype=bool)
    visible_targets = target_grid[positive]
    if visible_targets.size == 0:
        visible_targets = target_grid[np.isfinite(target_grid).all(axis=1)]
    unmapped_grid = _unmapped_query_grid(
        query_view, width=width, height=height
    )
    if unmapped_grid.size:
        order = rng.permutation(unmapped_grid.shape[0])
        cursor = 0
        for index in np.flatnonzero(requested_unmatchable).tolist():
            while cursor < order.size:
                candidate = unmapped_grid[order[cursor]]
                cursor += 1
                if visible_targets.size == 0 or np.min(
                    np.linalg.norm(visible_targets - candidate[None], axis=1)
                ) > radius + 1:
                    proposal_grid[index] = candidate
                    target_grid[index] = candidate
                    query_unmatchable[index] = True
                    break
    positive &= ~query_unmatchable
    query_matchable = _query_matchable_at_grid(
        query_view,
        target_grid,
        width=width,
        height=height,
    )
    query_matchable[query_unmatchable] = False
    proposal_xy = torch.from_numpy(proposal_grid)[None].to(device)
    target_xy = torch.from_numpy(target_grid)[None].to(device)
    positive_tensor = torch.from_numpy(positive)[None].to(device)
    query_matchable_tensor = torch.from_numpy(query_matchable)[None].to(device)
    loss = local_correlation_training_loss(
        {},
        query_output,
        map_xy=proposal_xy,
        proposal_xy=proposal_xy,
        target_xy=target_xy,
        positive_mask=positive_tensor,
        query_matchable_mask=query_matchable_tensor,
        map_descriptor_override=(
            None
            if mode_descriptors is not None
            else torch.from_numpy(descriptors)[None].to(device)
        ),
        map_descriptor_modes_override=(
            torch.from_numpy(mode_descriptors)[None].to(device)
            if mode_descriptors is not None
            else None
        ),
        map_mode_log_prior=(
            torch.from_numpy(mode_log_prior)[None].to(device)
            if mode_log_prior is not None
            else None
        ),
        map_uncertainty_override=torch.from_numpy(map_uncertainty)[None].to(
            device
        ),
        feature_key=feature_key,
        radius=radius,
    )
    _projected, jacobian = projection_jacobian(
        xyz, proposal_pose, query_view.camera
    )
    pose_loss = analytic_one_step_pose_loss(
        loss["mean_displacement"],
        loss["displacement_covariance"],
        torch.from_numpy(jacobian.astype(np.float32))[None].to(device),
        loss["positive_mask"],
        torch.from_numpy((-perturbation).astype(np.float32))[None].to(device),
        displacement_scale=scale,
    )
    loss["one_step_pose"] = pose_loss
    loss["total"] = loss["total"] + 0.05 * pose_loss
    loss["pair_negative_fraction"] = torch.as_tensor(
        np.mean(negative), dtype=torch.float32, device=device
    )
    loss["query_unmatchable_fraction"] = torch.as_tensor(
        np.mean(~query_matchable), dtype=torch.float32, device=device
    )
    loss["flow_median_cells"] = torch.as_tensor(
        np.median(np.linalg.norm(target_grid - proposal_grid, axis=1)),
        dtype=torch.float32,
        device=device,
    )
    loss["feature_level"] = feature_key
    return loss


def _episode(
    model: V6MetricEncoder,
    views: list[TrainingView],
    pair: tuple[int, int, np.ndarray],
    atlas: MapletFeatureAtlasBank,
    rng: np.random.Generator,
    *,
    samples: int,
    device: str,
    frozen_atlases: Mapping[str, MapletFeatureAtlasBank] | None = None,
    feature_key: str | None = None,
    inference_matched: bool = False,
) -> dict[str, torch.Tensor]:
    first, second, common = pair
    feature_key = (
        str(rng.choice(["fine", "middle", "coarse"]))
        if feature_key is None
        else str(feature_key)
    )
    frozen_atlas = (
        frozen_atlases.get(feature_key) if frozen_atlases is not None else None
    )
    if common.size > int(samples):
        common = rng.choice(common, size=int(samples), replace=False)
        common.sort()
    if frozen_atlas is not None and not inference_matched:
        atlas_valid = frozen_atlas.valid_mask.reshape(-1)[common]
        common = common[atlas_valid]
        if common.size < 16:
            raise ValueError("frozen atlas has insufficient visible overlap")
    map_view, query_view = views[first], views[second]
    if frozen_atlas is not None and inference_matched:
        # The deployed map path is the frozen atlas.  Encoding a second RGB
        # image here wastes half the batch and silently reintroduces a training
        # path that is absent at inference.
        query_output = model(
            query_view.radio[None].to(device), query_view.rgb[None].to(device)
        )
    else:
        rgb = torch.stack([map_view.rgb, query_view.rgb]).to(device)
        radio = torch.stack([map_view.radio, query_view.radio]).to(device)
        output = model(radio, rgb)
        map_output = {key: value[0:1] for key, value in output.items()}
        query_output = {key: value[1:2] for key, value in output.items()}
    map_image_xy = _coordinates_for_rows(map_view, common)
    target_image_xy = _coordinates_for_rows(query_view, common)
    # Coupled 6DoF curriculum matches the three inference search levels.
    scale = {"fine": 4.0, "middle": 8.0, "coarse": 16.0}[feature_key]
    radius = {"fine": 4, "middle": 6, "coarse": 8}[feature_key]
    translation_range = {
        "fine": (0.005, 0.05),
        "middle": (0.03, 0.15),
        "coarse": (0.05, 0.30),
    }[feature_key]
    rotation_range = {
        "fine": (0.05, 0.5),
        "middle": (0.2, 1.0),
        "coarse": (0.5, 2.0),
    }[feature_key]
    # Select the perturbation by its actual correlation-grid displacement.
    # A metric-only curriculum silently converts many examples into nulls as
    # focal length and scene depth change.
    desired_flow = rng.uniform(0.25, radius * 0.9)
    best = None
    for _attempt in range(16):
        rotation = rng.normal(size=3)
        rotation /= max(np.linalg.norm(rotation), 1e-8)
        rotation *= np.deg2rad(rng.uniform(*rotation_range))
        translation = rng.normal(size=3)
        translation /= max(np.linalg.norm(translation), 1e-8)
        translation *= rng.uniform(*translation_range)
        candidate_perturbation = np.r_[rotation, translation]
        candidate_pose = se3_exp(candidate_perturbation) @ query_view.pose_w2c
        candidate_xy, candidate_depth = project_world_points(
            atlas.xyz.reshape(-1, 3)[common],
            candidate_pose,
            query_view.camera,
        )
        flow = np.linalg.norm(
            (target_image_xy - candidate_xy) / scale, axis=1
        )
        finite = np.isfinite(flow) & (candidate_depth > 0.0)
        median_flow = float(np.median(flow[finite])) if np.any(finite) else np.inf
        score = abs(median_flow - desired_flow)
        if best is None or score < best[0]:
            best = (
                score,
                candidate_perturbation,
                candidate_pose,
                candidate_xy,
                candidate_depth,
            )
    assert best is not None
    _score, perturbation, proposal_pose, proposal_image_xy, depth = best
    if frozen_atlas is not None and inference_matched:
        return _inference_matched_loss(
            query_output=query_output,
            query_view=query_view,
            frozen_atlas=frozen_atlas,
            common_rows=common,
            proposal_pose=proposal_pose,
            perturbation=perturbation,
            feature_key=feature_key,
            scale=scale,
            radius=radius,
            samples=samples,
            rng=rng,
            device=device,
        )
    map_xy = torch.from_numpy(_grid_xy(map_image_xy, scale))[None].to(device)
    target_xy = torch.from_numpy(_grid_xy(target_image_xy, scale))[None].to(device)
    proposal_xy = torch.from_numpy(
        _grid_xy(proposal_image_xy, scale)
    )[None].to(device)
    positive = torch.from_numpy(
        np.isfinite(proposal_image_xy).all(axis=1) & (depth > 0.0)
    )[None].to(device)
    # Explicit wrong-texel/repeated-structure negatives: preserve query
    # matchability, but require the pairwise correlation outcome to be null.
    negative = rng.random(common.size) < 0.20
    if common.size >= 2 and not np.any(negative):
        negative[int(rng.integers(common.size))] = True
    permutation = np.arange(common.size, dtype=np.int64)
    flat_cells = atlas.height * atlas.width
    maplet_rows = common // flat_cells
    cell_xy = np.stack(
        [common % atlas.width, (common % flat_cells) // atlas.width], axis=1
    )
    descriptor_bank = None
    if frozen_atlas is not None:
        descriptor_bank = frozen_atlas.features.transpose(0, 2, 3, 1).reshape(
            -1, frozen_atlas.feature_dim
        )[common]
    for index in np.flatnonzero(negative):
        kind = float(rng.random())
        candidates = np.delete(np.arange(common.size), index)
        if kind < 0.20:
            pass
        elif kind < 0.50:
            same = candidates[maplet_rows[candidates] == maplet_rows[index]]
            if same.size:
                distance = np.linalg.norm(
                    cell_xy[same] - cell_xy[index], axis=1
                )
                candidates = same[np.argsort(distance)[: min(8, same.size)]]
        else:
            other = candidates[maplet_rows[candidates] != maplet_rows[index]]
            if other.size:
                if descriptor_bank is not None:
                    similarity = descriptor_bank[other] @ descriptor_bank[index]
                    candidates = other[
                        np.argsort(-similarity)[: min(8, other.size)]
                    ]
                else:
                    candidates = other
        permutation[index] = int(rng.choice(candidates))
    negative_tensor = torch.from_numpy(negative)[None].to(device)
    shuffled_map_xy = map_xy.clone()
    shuffled_map_xy[:, negative_tensor[0]] = map_xy[
        :, torch.from_numpy(permutation[negative]).to(device)
    ]
    map_xy = shuffled_map_xy
    map_descriptor_override = None
    if frozen_atlas is not None:
        descriptor = frozen_atlas.features.transpose(0, 2, 3, 1).reshape(
            -1, frozen_atlas.feature_dim
        )[common].astype(np.float32)
        descriptor[negative] = descriptor[permutation[negative]]
        map_descriptor_override = torch.from_numpy(descriptor)[None].to(device)
    positive &= ~negative_tensor
    # Query-only negatives are sampled away from every known visible atlas
    # projection.  They supervise query matchability independently from the
    # pairwise null label (wrong texel/maplet).
    requested_unmatchable = rng.random(common.size) < 0.10
    query_unmatchable = np.zeros((common.size,), dtype=bool)
    if np.any(requested_unmatchable):
        feature_height = int(query_output[feature_key].shape[-2])
        feature_width = int(query_output[feature_key].shape[-1])
        visible_grid = _grid_xy(target_image_xy, scale)
        unmapped_grid = _unmapped_query_grid(
            query_view, width=feature_width, height=feature_height
        )
        order = rng.permutation(unmapped_grid.shape[0])
        cursor = 0
        for index in np.flatnonzero(requested_unmatchable).tolist():
            while cursor < order.size:
                candidate = unmapped_grid[order[cursor]]
                cursor += 1
                if np.min(
                    np.linalg.norm(visible_grid - candidate[None], axis=1)
                ) > radius + 1:
                    proposal_xy[0, index] = torch.from_numpy(candidate).to(
                        device
                    )
                    target_xy[0, index] = proposal_xy[0, index]
                    query_unmatchable[index] = True
                    break
    query_unmatchable_tensor = torch.from_numpy(query_unmatchable)[None].to(device)
    positive &= ~query_unmatchable_tensor
    query_matchable = _query_matchable_at_grid(
        query_view,
        target_xy[0].detach().cpu().numpy(),
        width=int(query_output[feature_key].shape[-1]),
        height=int(query_output[feature_key].shape[-2]),
    )
    query_matchable[query_unmatchable] = False
    query_matchable_tensor = torch.from_numpy(query_matchable)[None].to(device)
    loss = local_correlation_training_loss(
        map_output,
        query_output,
        map_xy=map_xy,
        proposal_xy=proposal_xy,
        target_xy=target_xy,
        positive_mask=positive,
        query_matchable_mask=query_matchable_tensor,
        map_descriptor_override=map_descriptor_override,
        feature_key=feature_key,
        radius=radius,
    )
    _projected, jacobian = projection_jacobian(
        atlas.xyz.reshape(-1, 3)[common], proposal_pose, query_view.camera
    )
    pose_loss = analytic_one_step_pose_loss(
        loss["mean_displacement"],
        loss["displacement_covariance"],
        torch.from_numpy(jacobian.astype(np.float32))[None].to(device),
        loss["positive_mask"],
        torch.from_numpy((-perturbation).astype(np.float32))[None].to(device),
        displacement_scale=scale,
    )
    loss["one_step_pose"] = pose_loss
    loss["total"] = loss["total"] + 0.05 * pose_loss
    loss["pair_negative_fraction"] = negative_tensor.float().mean()
    loss["query_unmatchable_fraction"] = (
        ~query_matchable_tensor
    ).float().mean()
    actual_flow = np.linalg.norm(
        (target_image_xy - proposal_image_xy) / scale, axis=1
    )
    loss["flow_median_cells"] = torch.as_tensor(
        np.median(actual_flow), dtype=torch.float32, device=device
    )
    loss["feature_level"] = feature_key
    return loss


def _validate(
    model: V6MetricEncoder,
    views: list[TrainingView],
    pairs: list[tuple[int, int, np.ndarray]],
    atlas: MapletFeatureAtlasBank,
    rng: np.random.Generator,
    args: argparse.Namespace,
    frozen_atlases: Mapping[str, MapletFeatureAtlasBank] | None = None,
    inference_matched: bool = False,
) -> dict[str, float]:
    model.eval()
    validation_rng = np.random.default_rng(99173)
    values: dict[str, list[float]] = {
        "total": [],
        "flow_epe": [],
        "mode_recall": [],
        "mode_recall_radius1": [],
        "direction_cosine": [],
        "in_window_fraction": [],
        "positive_fraction": [],
        "one_step_pose": [],
        "flow_median_cells": [],
    }
    level_values = {
        level: {key: [] for key in values}
        for level in ("fine", "middle", "coarse")
    }
    null_labels: list[np.ndarray] = []
    null_scores: list[np.ndarray] = []
    with torch.no_grad():
        episode_count = max(3, int(args.validation_episodes))
        levels = ("fine", "middle", "coarse")
        for episode_index in range(episode_count):
            # Evaluate a view/scale cross product.  The old indexing bound
            # each fixed validation view to only one pyramid level.
            pair_index = (episode_index // len(levels)) % len(pairs)
            level = levels[episode_index % len(levels)]
            loss = _episode(
                model,
                views,
                pairs[pair_index],
                atlas,
                validation_rng,
                samples=min(int(args.samples_per_episode), 128),
                device=str(args.device),
                frozen_atlases=frozen_atlases,
                feature_key=level,
                inference_matched=inference_matched,
            )
            for key in values:
                value = float(loss[key].item())
                values[key].append(value)
                level_values[level][key].append(value)
            null_labels.append(
                loss["null_target"].detach().cpu().numpy().reshape(-1)
            )
            null_scores.append(
                loss["null_probability_per_sample"]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
    model.train()
    result = {
        key: float(np.mean(items)) if items else float("nan")
        for key, items in values.items()
    }
    labels = np.concatenate(null_labels)
    scores = np.concatenate(null_scores)
    result["null_auprc"] = _average_precision(labels, scores)
    result["null_prevalence"] = float(np.mean(labels))
    result["null_brier"] = float(
        np.mean((scores - labels.astype(np.float64)) ** 2)
    )
    for level, metrics in level_values.items():
        for key, items in metrics.items():
            result[f"{level}_{key}"] = (
                float(np.mean(items)) if items else float("nan")
            )
    result["score"] = (
        result["total"]
        + 0.25 * result["flow_epe"]
        + 0.50 * result["one_step_pose"]
        - 0.5 * result["mode_recall"]
        - 0.1 * result["direction_cosine"]
        - 0.25 * result["null_auprc"]
    )
    result["episode_count"] = float(episode_count)
    return result


def _validation_selection_key(
    validation: Mapping[str, float],
) -> tuple[float, ...]:
    """Order checkpoints by the diagnostic priorities in the V6 protocol."""

    def finite(name: str, fallback: float) -> float:
        value = float(validation.get(name, fallback))
        return value if np.isfinite(value) else float(fallback)

    return (
        -finite("mode_recall", -1.0),
        -finite("direction_cosine", -1.0),
        finite("one_step_pose", np.inf),
        -finite("null_auprc", -1.0),
        finite("flow_epe", np.inf),
        finite("total", np.inf),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if (checkpoint.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite V6 metric encoder")
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    frozen_atlas = (
        MapletFeatureAtlasBank.load_npz(Path(args.frozen_metric_atlas))
        if str(args.frozen_metric_atlas)
        else None
    )
    frozen_atlases = (
        {
            "fine": frozen_atlas,
            "middle": (
                MapletFeatureAtlasBank.load_npz(
                    Path(args.frozen_metric_atlas_middle)
                )
                if str(args.frozen_metric_atlas_middle)
                else frozen_atlas
            ),
            "coarse": (
                MapletFeatureAtlasBank.load_npz(
                    Path(args.frozen_metric_atlas_coarse)
                )
                if str(args.frozen_metric_atlas_coarse)
                else frozen_atlas
            ),
        }
        if frozen_atlas is not None
        else None
    )
    train_views = _load_views(
        Path(args.train_contributor_dir),
        atlas,
        Path(args.image_root),
        args.train_trajectory_ids,
    )
    validation_views = _load_views(
        Path(args.validation_contributor_dir),
        atlas,
        Path(args.image_root),
        args.validation_trajectory_ids,
    )
    inference_matched = bool(
        frozen_atlas is not None and not args.legacy_point_pair_training
    )
    if inference_matched:
        # Every image is an independent query against the frozen map.  Pairing
        # query images would create a non-deployment intersection filter and
        # systematically under-sample early trajectory names.
        train_pairs = _single_view_episodes(train_views)
        validation_pairs = _single_view_episodes(validation_views)
    else:
        # Cross-trajectory positives remain useful for the legacy point-pair
        # diagnostic.  The held-out validation trajectory is still disjoint.
        train_pairs = _pairs(train_views, allow_cross_trajectory=True)
        validation_pairs = _pairs(validation_views)
    if frozen_atlas is not None and not inference_matched:
        flat_frozen_valid = frozen_atlas.valid_mask.reshape(-1)
        train_pairs = [
            pair
            for pair in train_pairs
            if int(np.sum(flat_frozen_valid[pair[2]])) >= 16
        ]
        validation_pairs = [
            pair
            for pair in validation_pairs
            if int(np.sum(flat_frozen_valid[pair[2]])) >= 16
        ]
    train_trajectories = sorted({view.trajectory_id for view in train_views})
    validation_trajectories = sorted(
        {view.trajectory_id for view in validation_views}
    )
    if set(train_trajectories) & set(validation_trajectories):
        raise ValueError("training and validation trajectories must be disjoint")
    if not train_pairs or not validation_pairs:
        raise ValueError("insufficient within-trajectory atlas overlap")
    initial_checkpoint_sha256 = ""
    initial_metadata: Mapping[str, object] = {}
    if str(args.initial_checkpoint):
        initial_path = Path(args.initial_checkpoint)
        initial_checkpoint_sha256 = hashlib.sha256(
            initial_path.read_bytes()
        ).hexdigest()
        model, _initial_metadata = load_v6_metric_encoder(
            initial_path, device=str(args.device)
        )
        initial_metadata = dict(_initial_metadata)
        # A legacy checkpoint retains average pooling for faithful evaluation,
        # but a new fine-tuning run must activate and learn the independent
        # stride-8/16 heads.
        model._legacy_average_pool_pyramid = False
    else:
        model = V6MetricEncoder(
            V6MetricEncoderConfig(
                hidden_dim=int(args.hidden_dim), output_dim=int(args.output_dim)
            )
        ).to(str(args.device))
    atlas_mapping_trajectories: list[str] = []
    compatible_map_encoder_sha256 = ""
    if frozen_atlases is not None:
        atlas_encoder_hashes = {
            str((level_atlas.metadata or {}).get("metric_encoder_sha256", ""))
            for level_atlas in frozen_atlases.values()
        }
        if "" in atlas_encoder_hashes or len(atlas_encoder_hashes) != 1:
            raise ValueError(
                "scale-specific frozen atlases require one declared map "
                "encoder lineage"
            )
        compatible_map_encoder_sha256 = next(iter(atlas_encoder_hashes))
        if not initial_checkpoint_sha256:
            raise ValueError(
                "inference-matched training must initialize from the frozen "
                "atlas map encoder"
            )
        if initial_checkpoint_sha256 != compatible_map_encoder_sha256:
            raise ValueError(
                "initial query encoder does not match frozen atlas lineage"
            )
        atlas_mapping_trajectories = sorted(
            {
                str(value)
                for level_atlas in frozen_atlases.values()
                for value in (level_atlas.metadata or {}).get(
                    "mapping_trajectory_ids", []
                )
            }
        )
        if not atlas_mapping_trajectories:
            raise ValueError(
                "inference-matched training requires atlas mapping lineage"
            )
        if set(atlas_mapping_trajectories) & set(train_trajectories):
            raise ValueError(
                "frozen atlas overlaps train-query trajectories"
            )
        if set(atlas_mapping_trajectories) & set(validation_trajectories):
            raise ValueError(
                "frozen atlas overlaps validation-query trajectories"
            )
        for level, level_atlas in frozen_atlases.items():
            if level_atlas.feature_dim != model.config.output_dim:
                raise ValueError("frozen atlas and metric encoder dimensions differ")
            declared_level = str(
                (level_atlas.metadata or {}).get("metric_feature_level", "")
            )
            if declared_level and declared_level != level:
                raise ValueError(f"{level} atlas uses {declared_level} features")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    best_validation = float("inf")
    best_validation_key: tuple[float, ...] | None = None
    best_step = 0
    history = []
    training_levels = ("fine", "middle", "coarse")
    for step in range(1, int(args.steps) + 1):
        # One deterministic view/scale cross product is an epoch.  Random
        # perturbations and negatives remain stochastic, while no trajectory
        # or pyramid level can be starved by random episode selection.
        level_index = (step - 1) % len(training_levels)
        pair_index = (
            (step - 1) // len(training_levels)
        ) % len(train_pairs)
        pair = train_pairs[pair_index]
        loss = _episode(
            model,
            train_views,
            pair,
            atlas,
            rng,
            samples=int(args.samples_per_episode),
            device=str(args.device),
            frozen_atlases=frozen_atlases,
            feature_key=training_levels[level_index],
            inference_matched=inference_matched,
        )
        optimizer.zero_grad(set_to_none=True)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % int(args.validation_every) == 0:
            validation = _validate(
                model,
                validation_views,
                validation_pairs,
                atlas,
                rng,
                args,
                frozen_atlases,
                inference_matched,
            )
            row = {
                "step": step,
                "train_total": float(loss["total"].item()),
                "train_flow_epe": float(loss["flow_epe"].item()),
                "validation_total": validation["total"],
                "validation_score": validation["score"],
                "validation_flow_epe": validation["flow_epe"],
                "validation_mode_recall": validation["mode_recall"],
                "validation_mode_recall_radius1": validation[
                    "mode_recall_radius1"
                ],
                "validation_direction_cosine": validation["direction_cosine"],
                "validation_one_step_pose": validation["one_step_pose"],
                "validation_null_auprc": validation["null_auprc"],
                "validation_null_prevalence": validation["null_prevalence"],
                "validation_null_brier": validation["null_brier"],
                "validation_in_window_fraction": validation[
                    "in_window_fraction"
                ],
                "validation_by_level": {
                    level: {
                        key: validation[f"{level}_{key}"]
                        for key in (
                            "flow_epe",
                            "mode_recall",
                            "mode_recall_radius1",
                            "direction_cosine",
                            "one_step_pose",
                            "in_window_fraction",
                        )
                    }
                    for level in ("fine", "middle", "coarse")
                },
                "validation_selection_key": list(
                    _validation_selection_key(validation)
                ),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            selection_key = _validation_selection_key(validation)
            if (
                best_validation_key is None
                or selection_key < best_validation_key
            ):
                best_validation = validation["score"]
                best_validation_key = selection_key
                best_step = step
                save_v6_metric_encoder(
                    checkpoint,
                    model,
                    {
                        "vfm_layer": "radio_final",
                        "output_stride": 4,
                        "best_step": step,
                        "best_validation_score": validation["score"],
                        "best_validation_selection_key": list(selection_key),
                        "best_validation_metrics": validation,
                        "training_trajectories": train_trajectories,
                        "validation_trajectories": validation_trajectories,
                        "atlas_mapping_trajectories": (
                            atlas_mapping_trajectories
                        ),
                        "atlas_query_trajectory_disjoint": True,
                        "trajectory_disjoint_validation": True,
                        "query_encoder_role": "student",
                        "map_encoder_role": "frozen_teacher",
                        "map_encoder_is_frozen_teacher": bool(
                            inference_matched
                        ),
                        "compatible_map_encoder_sha256": (
                            compatible_map_encoder_sha256
                        ),
                        "initial_query_encoder_sha256": (
                            initial_checkpoint_sha256
                        ),
                        "compatible_map_encoder_verified_at_training": bool(
                            inference_matched
                            and compatible_map_encoder_sha256
                            and initial_checkpoint_sha256
                            == compatible_map_encoder_sha256
                        ),
                        "initial_query_encoder_legacy_pyramid": bool(
                            initial_metadata.get(
                                "legacy_average_pool_pyramid", False
                            )
                        ),
                        "uses_exact_query_surface_visibility_labels": bool(
                            inference_matched
                        ),
                        "uses_ground_truth_depth_at_runtime": False,
                        "uses_joint_6dof_perturbations": True,
                        "maximum_translation_m": 0.30,
                        "maximum_rotation_deg": 2.0,
                        "training_episode_schedule": (
                            "deterministic_view_scale_cross_product"
                        ),
                        "training_episodes_per_epoch": int(
                            len(train_pairs) * len(training_levels)
                        ),
                        "training_objectives": [
                            "local_correlation_nll",
                            "balanced_positive_null_correlation_nll",
                            "explicit_null",
                            "matchability",
                            "candidate_offset_matchability",
                            "query_only_no_clean_surface_negatives",
                            "structured_wrong_mode_negatives",
                            "pixel_flow_bucketed_curriculum",
                            "heteroscedastic_displacement_nll",
                            "atlas_render_correlation_se3_replay"
                            if inference_matched
                            else "frozen_atlas_point_correlation"
                            if frozen_atlas is not None
                            else "cross_view_correlation",
                        ],
                        "uses_radio_intermediate": False,
                        "uses_sfm_points": False,
                        "uses_sfm_tracks": False,
                        "uses_alike_descriptors": False,
                        "uses_pairwise_image_matching": False,
                    },
                )
    report = {
        "stage": "v6_metric_encoder_training",
        "best_step": best_step,
        "best_validation_score": best_validation,
        "best_validation_selection_key": (
            list(best_validation_key)
            if best_validation_key is not None
            else None
        ),
        "validation_episode_count": int(args.validation_episodes),
        "validation_checkpoint_metrics": [
            "total",
            "flow_epe",
            "mode_recall",
            "mode_recall_radius1",
            "direction_cosine",
            "one_step_pose",
            "null_auprc",
            "null_prevalence",
            "null_brier",
            "in_window_fraction",
            "per_level_metrics",
        ],
        "validation_checkpoint_selection_order": [
            "maximize_mode_recall",
            "maximize_direction_cosine",
            "minimize_one_step_pose",
            "maximize_null_auprc",
            "minimize_flow_epe",
            "minimize_total",
        ],
        "train_view_count": len(train_views),
        "validation_view_count": len(validation_views),
        "train_pair_count": len(train_pairs),
        "training_episode_schedule": "deterministic_view_scale_cross_product",
        "training_episodes_per_epoch": int(
            len(train_pairs) * len(training_levels)
        ),
        "validation_pair_count": len(validation_pairs),
        "training_trajectories": train_trajectories,
        "validation_trajectories": validation_trajectories,
        "atlas_mapping_trajectories": atlas_mapping_trajectories,
        "query_encoder_role": (
            "student" if inference_matched else "symmetric"
        ),
        "map_encoder_role": (
            "frozen_teacher" if inference_matched else "shared"
        ),
        "compatible_map_encoder_sha256": compatible_map_encoder_sha256,
        "initial_query_encoder_sha256": initial_checkpoint_sha256,
        "compatible_map_encoder_verified_at_training": bool(
            inference_matched
            and compatible_map_encoder_sha256
            and initial_checkpoint_sha256 == compatible_map_encoder_sha256
        ),
        "initial_query_encoder_legacy_pyramid": bool(
            initial_metadata.get("legacy_average_pool_pyramid", False)
        ),
        "uses_exact_query_surface_visibility_labels": bool(inference_matched),
        "uses_ground_truth_depth_at_runtime": False,
        "atlas_query_trajectory_disjoint": bool(
            frozen_atlases is None
            or not (
                set(atlas_mapping_trajectories)
                & (set(train_trajectories) | set(validation_trajectories))
            )
        ),
        "trajectory_disjoint_validation": True,
        "uses_frozen_metric_atlas": frozen_atlas is not None,
        "uses_scale_specific_atlases": bool(
            str(args.frozen_metric_atlas_middle)
            and str(args.frozen_metric_atlas_coarse)
        ),
        "uses_inference_matched_atlas_rendering": inference_matched,
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
