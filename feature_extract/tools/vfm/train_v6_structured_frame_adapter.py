"""Train a RADIO-final adapter on complete chart-transform distributions.

Unlike the historical per-texel location classifier, every episode compares
the correct projection of several complete charts against translation,
rotation, scale, shear and global repeated-structure modes.  The canonical
map atlas stays frozen and image-free; only a query-side 1x1 metric adapter is
learned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.structured_frame_adapter import (
    StructuredFrameAdapter,
    StructuredFrameAdapterConfig,
    save_structured_frame_adapter,
)


@dataclass(frozen=True)
class PreparedView:
    image_id: str
    trajectory_id: str
    image_size_wh: tuple[int, int]
    query_feature: torch.Tensor
    chart_rows: tuple[np.ndarray, ...]
    chart_xy: tuple[np.ndarray, ...]
    pose_w2c: np.ndarray
    camera: object


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--surface_mapper_checkpoint", required=True)
    parser.add_argument("--train_contributor_dir", required=True)
    parser.add_argument("--validation_contributor_dir", required=True)
    parser.add_argument(
        "--train_trajectory_ids",
        nargs="*",
        default=("seq9", "seq10", "seq12", "seq14"),
    )
    parser.add_argument(
        "--validation_trajectory_ids", nargs="*", default=("seq11",)
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation_every", type=int, default=25)
    parser.add_argument("--charts_per_episode", type=int, default=4)
    parser.add_argument("--cells_per_chart", type=int, default=64)
    parser.add_argument("--candidate_transforms", type=int, default=16)
    parser.add_argument(
        "--validation_episodes_per_view", type=int, default=8
    )
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_views(
    views: Sequence[object],
    atlas: MapletFeatureAtlasBank,
    mapper: object,
) -> list[PreparedView]:
    flat_feature = atlas.features.transpose(0, 2, 3, 1).reshape(
        -1, atlas.feature_dim
    )
    flat_support = atlas.support_count.reshape(-1)
    cell_count = atlas.height * atlas.width
    prepared = []
    for view in views:
        rows = np.asarray(view.visible_rows, dtype=np.int64)
        xy = np.asarray(view.image_xy, dtype=np.float32)
        usable = (
            (flat_support[rows] > 0)
            & (np.linalg.norm(flat_feature[rows], axis=1) > 0.5)
            & np.isfinite(xy).all(axis=1)
        )
        rows = rows[usable]
        xy = xy[usable]
        chart_rows = []
        chart_xy = []
        for chart_row in np.unique(rows // cell_count).tolist():
            local = rows // cell_count == int(chart_row)
            if int(np.sum(local)) < 12:
                continue
            chart_rows.append(rows[local])
            chart_xy.append(xy[local])
        if len(chart_rows) < 2:
            continue
        mapped = mapper.project(
            view.radio.numpy()
        ).measurement_context.astype(np.float32)
        prepared.append(
            PreparedView(
                image_id=str(view.image_id),
                trajectory_id=str(view.trajectory_id),
                image_size_wh=(
                    int(view.camera.width),
                    int(view.camera.height),
                ),
                query_feature=torch.from_numpy(mapped),
                chart_rows=tuple(chart_rows),
                chart_xy=tuple(chart_xy),
                pose_w2c=np.asarray(view.pose_w2c, dtype=np.float64),
                camera=view.camera,
            )
        )
    return prepared


def _candidate_coordinates(
    target_xy: np.ndarray,
    *,
    candidate_count: int,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate complete affine chart modes around one exact projection."""

    target = np.asarray(target_xy, dtype=np.float32)
    center = np.mean(target, axis=0)
    result = [target]
    for candidate in range(1, int(candidate_count)):
        global_mode = candidate >= max(
            2 * int(candidate_count) // 3, 2
        )
        if global_mode:
            destination = np.asarray(
                [
                    rng.uniform(0.0, max(width - 1, 0)),
                    rng.uniform(0.0, max(height - 1, 0)),
                ],
                dtype=np.float32,
            )
            translation = destination - center
            angle = rng.uniform(-np.pi, np.pi)
            log_scale = rng.uniform(-0.55, 0.55)
            anisotropy = rng.uniform(-0.30, 0.30)
            shear = rng.uniform(-0.30, 0.30)
        else:
            radius = rng.uniform(1.0, 6.0)
            direction = rng.uniform(-np.pi, np.pi)
            translation = radius * np.asarray(
                [np.cos(direction), np.sin(direction)], dtype=np.float32
            )
            angle = rng.uniform(-np.deg2rad(35.0), np.deg2rad(35.0))
            log_scale = rng.uniform(-0.35, 0.35)
            anisotropy = rng.uniform(-0.20, 0.20)
            shear = rng.uniform(-0.20, 0.20)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float32
        )
        scale = np.asarray(
            [
                [np.exp(log_scale + anisotropy), shear],
                [0.0, np.exp(log_scale - anisotropy)],
            ],
            dtype=np.float32,
        )
        linear = rotation @ scale
        result.append(
            (target - center[None]) @ linear.T
            + center[None]
            + translation[None]
        )
    return np.stack(result).astype(np.float32)


def _sample_query(
    query_feature: torch.Tensor, candidate_xy: torch.Tensor
) -> torch.Tensor:
    """Bilinearly sample BxK complete chart modes with a fixed denominator."""

    if query_feature.ndim != 4 or query_feature.shape[0] != 1:
        raise ValueError("query feature must have shape (1,C,H,W)")
    batch, candidates, cells = candidate_xy.shape[:3]
    height, width = query_feature.shape[-2:]
    grid = candidate_xy.clone()
    grid[..., 0] = 2.0 * (grid[..., 0] + 0.5) / width - 1.0
    grid[..., 1] = 2.0 * (grid[..., 1] + 0.5) / height - 1.0
    sampled = F.grid_sample(
        query_feature,
        grid.reshape(1, batch * candidates, cells, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0].permute(1, 2, 0).reshape(
        batch, candidates, cells, query_feature.shape[1]
    )


def _episode(
    model: StructuredFrameAdapter,
    view: PreparedView,
    flat_map_feature: torch.Tensor,
    *,
    charts_per_episode: int,
    cells_per_chart: int,
    candidate_transforms: int,
    temperature: float,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    available = len(view.chart_rows)
    chart_count = min(int(charts_per_episode), available)
    selected = rng.choice(available, size=chart_count, replace=False)
    map_rows = []
    target_xy = []
    query_height, query_width = view.query_feature.shape[-2:]
    image_width, image_height = view.image_size_wh
    for chart_index in selected.tolist():
        rows = view.chart_rows[int(chart_index)]
        xy = view.chart_xy[int(chart_index)]
        choice = rng.choice(
            rows.size,
            size=int(cells_per_chart),
            replace=rows.size < int(cells_per_chart),
        )
        map_rows.append(rows[choice])
        points = xy[choice]
        target_xy.append(
            np.stack(
                [
                    (points[:, 0] + 0.5)
                    * query_width
                    / image_width
                    - 0.5,
                    (points[:, 1] + 0.5)
                    * query_height
                    / image_height
                    - 0.5,
                ],
                axis=1,
            ).astype(np.float32)
        )
    target_array = np.stack(target_xy)
    candidates = np.stack(
        [
            _candidate_coordinates(
                value,
                candidate_count=int(candidate_transforms),
                width=int(query_width),
                height=int(query_height),
                rng=rng,
            )
            for value in target_array
        ]
    )
    adapted = model(view.query_feature[None].to(device))
    sampled = _sample_query(
        adapted,
        torch.from_numpy(candidates).to(
            device=device, dtype=adapted.dtype
        ),
    )
    map_feature = flat_map_feature[
        torch.from_numpy(np.stack(map_rows)).to(device=device)
    ]
    map_feature = F.normalize(map_feature, dim=2, eps=1e-6)
    transform_score = torch.sum(
        sampled * map_feature[:, None], dim=3
    ).mean(dim=2)
    target = torch.zeros(
        (chart_count,), dtype=torch.long, device=adapted.device
    )
    transform_loss = F.cross_entropy(
        transform_score / float(temperature), target
    )
    gt_query = sampled[:, 0]
    identity_score = torch.einsum(
        "bnc,dnc->bd", map_feature, gt_query
    ) / float(cells_per_chart)
    identity_target = torch.arange(chart_count, device=adapted.device)
    identity_loss = F.cross_entropy(
        identity_score / float(temperature), identity_target
    )
    identity = torch.eye(
        model.config.feature_dim,
        device=adapted.device,
        dtype=adapted.dtype,
    )
    weight = model.projection.weight[:, :, 0, 0]
    regularization = torch.mean((weight - identity) ** 2)
    loss = transform_loss + 0.25 * identity_loss + 1e-3 * regularization
    order = torch.argsort(transform_score, dim=1, descending=True)
    gt_rank = (
        torch.argmax((order == 0).to(torch.int64), dim=1).float() + 1.0
    )
    best = order[:, 0]
    candidates_torch = torch.from_numpy(candidates).to(
        device=adapted.device, dtype=adapted.dtype
    )
    selected_xy = candidates_torch[
        torch.arange(chart_count, device=adapted.device), best
    ]
    target_torch = torch.from_numpy(target_array).to(
        device=adapted.device, dtype=adapted.dtype
    )
    control_error = torch.mean(
        torch.linalg.vector_norm(selected_xy - target_torch, dim=2),
        dim=1,
    )
    maximum_wrong = transform_score[:, 1:].max(dim=1).values
    metrics = {
        "loss": float(loss.detach().item()),
        "transform_recall_at_1": float(
            torch.mean((best == 0).float()).item()
        ),
        "transform_gt_rank_mean": float(torch.mean(gt_rank).item()),
        "transform_control_error_cells": float(
            torch.mean(control_error).item()
        ),
        "transform_positive_score": float(
            torch.mean(transform_score[:, 0]).item()
        ),
        "transform_margin": float(
            torch.mean(transform_score[:, 0] - maximum_wrong).item()
        ),
        "identity_recall_at_1": float(
            torch.mean(
                (
                    torch.argmax(identity_score, dim=1) == identity_target
                ).float()
            ).item()
        ),
    }
    return loss, metrics


@torch.no_grad()
def _validate(
    model: StructuredFrameAdapter,
    views: Sequence[PreparedView],
    flat_map_feature: torch.Tensor,
    *,
    charts_per_episode: int,
    cells_per_chart: int,
    candidate_transforms: int,
    episodes_per_view: int,
    temperature: float,
    seed: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for index, view in enumerate(views):
        for episode in range(max(int(episodes_per_view), 1)):
            _loss, metrics = _episode(
                model,
                view,
                flat_map_feature,
                charts_per_episode=int(charts_per_episode),
                cells_per_chart=int(cells_per_chart),
                candidate_transforms=int(candidate_transforms),
                temperature=float(temperature),
                rng=np.random.default_rng(
                    int(seed) + 1000 * index + episode
                ),
                device=str(device),
            )
            rows.append(metrics)
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    result["selection_score"] = float(
        result["transform_recall_at_1"]
        + 0.25 * result["identity_recall_at_1"]
        + 0.05 * result["transform_margin"]
        - 0.01 * result["transform_control_error_cells"]
    )
    return result


def _metadata(
    *,
    atlas_path: Path,
    mapper_path: Path,
    atlas: MapletFeatureAtlasBank,
    train_trajectories: Sequence[str],
    validation_trajectories: Sequence[str],
    best_step: int,
    best_metrics: Mapping[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "vfm_layer": "radio_final",
        "training_objective": (
            "complete_chart_transform_distribution_with_global_and_"
            "repeated_structure_modes"
        ),
        "compatible_radio_atlas_sha256": _sha256(atlas_path),
        "surface_mapper_checkpoint_sha256": _sha256(mapper_path),
        "atlas_mapping_trajectory_ids": sorted(
            str(value)
            for value in (atlas.metadata or {}).get(
                "mapping_trajectory_ids", []
            )
        ),
        "training_trajectory_ids": list(train_trajectories),
        "validation_trajectory_ids": list(validation_trajectories),
        "best_step": int(best_step),
        "best_validation_metrics": dict(best_metrics),
        "candidate_transforms": int(args.candidate_transforms),
        "cells_per_chart": int(args.cells_per_chart),
        "charts_per_episode": int(args.charts_per_episode),
        "validation_episodes_per_view": int(
            args.validation_episodes_per_view
        ),
        "stores_mapping_rgb": False,
        "stores_mapping_image_ids": False,
        "stores_mapping_image_paths": False,
        "uses_rgb_after_radio": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_pairwise_image_matching": False,
        "uses_point_correspondence_pnp": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if not bool(args.force) and (output.exists() or summary_path.exists()):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas_path = Path(args.radio_atlas)
    mapper_path = Path(args.surface_mapper_checkpoint)
    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    mapper, _mapper_metadata = load_surface_maplet_mapper(
        mapper_path, device=str(args.device)
    )
    if int(atlas.feature_dim) <= 0:
        raise ValueError("atlas has no metric features")
    train_views_raw = _load_views(
        Path(args.train_contributor_dir),
        atlas,
        Path(args.image_root),
        tuple(str(value) for value in args.train_trajectory_ids),
    )
    validation_views_raw = _load_views(
        Path(args.validation_contributor_dir),
        atlas,
        Path(args.image_root),
        tuple(str(value) for value in args.validation_trajectory_ids),
    )
    train_views = _prepare_views(train_views_raw, atlas, mapper)
    validation_views = _prepare_views(validation_views_raw, atlas, mapper)
    if not train_views or not validation_views:
        raise ValueError("structured-frame train/validation views are empty")
    train_trajectories = sorted(
        {value.trajectory_id for value in train_views}
    )
    validation_trajectories = sorted(
        {value.trajectory_id for value in validation_views}
    )
    mapping_trajectories = {
        str(value)
        for value in (atlas.metadata or {}).get(
            "mapping_trajectory_ids", []
        )
    }
    if (
        set(train_trajectories) & set(validation_trajectories)
        or set(train_trajectories) & mapping_trajectories
        or set(validation_trajectories) & mapping_trajectories
    ):
        raise ValueError("adapter train/validation/map trajectories overlap")
    model = StructuredFrameAdapter(
        StructuredFrameAdapterConfig(feature_dim=int(atlas.feature_dim))
    ).to(str(args.device))
    flat_map_feature = torch.from_numpy(
        atlas.features.transpose(0, 2, 3, 1)
        .reshape(-1, atlas.feature_dim)
        .copy()
    ).to(str(args.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    validation_seed = int(args.seed) + 100000
    baseline = _validate(
        model,
        validation_views,
        flat_map_feature,
        charts_per_episode=int(args.charts_per_episode),
        cells_per_chart=int(args.cells_per_chart),
        candidate_transforms=int(args.candidate_transforms),
        episodes_per_view=int(args.validation_episodes_per_view),
        temperature=float(args.temperature),
        seed=validation_seed,
        device=str(args.device),
    )
    best_step = 0
    best_metrics = dict(baseline)
    best_score = float(baseline["selection_score"])
    save_structured_frame_adapter(
        output,
        model,
        _metadata(
            atlas_path=atlas_path,
            mapper_path=mapper_path,
            atlas=atlas,
            train_trajectories=train_trajectories,
            validation_trajectories=validation_trajectories,
            best_step=best_step,
            best_metrics=best_metrics,
            args=args,
        ),
    )
    history = [{"step": 0, "validation": baseline}]
    rng = np.random.default_rng(int(args.seed))
    for step in range(1, int(args.steps) + 1):
        model.train()
        view = train_views[(step - 1) % len(train_views)]
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = _episode(
            model,
            view,
            flat_map_feature,
            charts_per_episode=int(args.charts_per_episode),
            cells_per_chart=int(args.cells_per_chart),
            candidate_transforms=int(args.candidate_transforms),
            temperature=float(args.temperature),
            rng=rng,
            device=str(args.device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if step == 1 or step % int(args.validation_every) == 0:
            validation = _validate(
                model,
                validation_views,
                flat_map_feature,
                charts_per_episode=int(args.charts_per_episode),
                cells_per_chart=int(args.cells_per_chart),
                candidate_transforms=int(args.candidate_transforms),
                episodes_per_view=int(
                    args.validation_episodes_per_view
                ),
                temperature=float(args.temperature),
                seed=validation_seed,
                device=str(args.device),
            )
            history.append(
                {
                    "step": int(step),
                    "train": train_metrics,
                    "validation": validation,
                }
            )
            if float(validation["selection_score"]) > best_score:
                best_step = int(step)
                best_metrics = dict(validation)
                best_score = float(validation["selection_score"])
                save_structured_frame_adapter(
                    output,
                    model,
                    _metadata(
                        atlas_path=atlas_path,
                        mapper_path=mapper_path,
                        atlas=atlas,
                        train_trajectories=train_trajectories,
                        validation_trajectories=validation_trajectories,
                        best_step=best_step,
                        best_metrics=best_metrics,
                        args=args,
                    ),
                )
            print(
                json.dumps(
                    {
                        "step": int(step),
                        "train": train_metrics,
                        "validation": validation,
                        "best_step": best_step,
                    }
                ),
                flush=True,
            )
    report = {
        "stage": "v6_structured_frame_adapter_training",
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "baseline_validation_metrics": baseline,
        "best_step": int(best_step),
        "best_validation_metrics": best_metrics,
        "best_validation_score": best_score,
        "training_trajectory_ids": train_trajectories,
        "validation_trajectory_ids": validation_trajectories,
        "atlas_mapping_trajectory_ids": sorted(mapping_trajectories),
        "trajectory_disjoint": True,
        "history": history,
        "map_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "stores_mapping_image_paths": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_loftr": False,
            "uses_radio_intermediate": False,
            "uses_pairwise_image_matching": False,
            "uses_point_correspondence_pnp": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
