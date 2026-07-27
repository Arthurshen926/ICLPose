"""Train the V6 phase-preserving metric encoder with exact atlas identities."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--train_contributor_dir", required=True)
    parser.add_argument("--validation_contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation_every", type=int, default=50)
    parser.add_argument("--samples_per_episode", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--initial_checkpoint", default="")
    parser.add_argument("--frozen_metric_atlas", default="")
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
    ix = np.rint(grid[candidate, 0]).astype(np.int64)
    iy = np.rint(grid[candidate, 1]).astype(np.int64)
    identity = topk_ids[iy, ix] == primitive[candidate, None]
    weight = np.max(np.where(identity, topk_weights[iy, ix], 0.0), axis=1)
    accepted = candidate[weight >= 0.01]
    return valid_rows[accepted], xy[accepted].astype(np.float32)


def _load_views(
    directory: Path,
    atlas: MapletFeatureAtlasBank,
    image_root: Path,
) -> list[TrainingView]:
    views = []
    for path in _load_cache_paths(directory):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            image_id = str(metadata["image_id"])
            token_path = Path(str(metadata["token_path"]))
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
            camera = ColmapCamera(
                camera_id=0,
                model_id=int(data["camera_model_id"]),
                width=int(data["camera_width"]),
                height=int(data["camera_height"]),
                params=tuple(np.asarray(data["camera_params"], dtype=np.float64)),
            )
            rows, xy = _visibility(
                atlas,
                pose,
                camera,
                np.asarray(data["topk_ids"], dtype=np.int64),
                np.asarray(data["topk_weights"], dtype=np.float32),
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
                trajectory_id=image_id.split("/", 1)[0],
                rgb=rgb,
                radio=radio,
                pose_w2c=pose,
                camera=camera,
                visible_rows=rows,
                image_xy=xy,
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


def _coordinates_for_rows(view: TrainingView, rows: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(view.visible_rows, rows)
    if not np.array_equal(view.visible_rows[positions], rows):
        raise ValueError("pair contains invisible atlas rows")
    return view.image_xy[positions]


def _episode(
    model: V6MetricEncoder,
    views: list[TrainingView],
    pair: tuple[int, int, np.ndarray],
    atlas: MapletFeatureAtlasBank,
    rng: np.random.Generator,
    *,
    samples: int,
    device: str,
    frozen_atlas: MapletFeatureAtlasBank | None = None,
) -> dict[str, torch.Tensor]:
    first, second, common = pair
    if common.size > int(samples):
        common = rng.choice(common, size=int(samples), replace=False)
        common.sort()
    if frozen_atlas is not None:
        atlas_valid = frozen_atlas.valid_mask.reshape(-1)[common]
        common = common[atlas_valid]
        if common.size < 16:
            raise ValueError("frozen atlas has insufficient visible overlap")
    map_view, query_view = views[first], views[second]
    rgb = torch.stack([map_view.rgb, query_view.rgb]).to(device)
    radio = torch.stack([map_view.radio, query_view.radio]).to(device)
    output = model(radio, rgb)
    map_output = {key: value[0:1] for key, value in output.items()}
    query_output = {key: value[1:2] for key, value in output.items()}
    map_image_xy = _coordinates_for_rows(map_view, common)
    target_image_xy = _coordinates_for_rows(query_view, common)
    # Coupled 6DoF curriculum matches the three inference search levels.
    feature_key = str(rng.choice(["fine", "middle", "coarse"]))
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
    rotation = rng.normal(size=3)
    rotation /= max(np.linalg.norm(rotation), 1e-8)
    rotation *= np.deg2rad(rng.uniform(*rotation_range))
    translation = rng.normal(size=3)
    translation /= max(np.linalg.norm(translation), 1e-8)
    translation *= rng.uniform(*translation_range)
    perturbation = np.r_[rotation, translation]
    proposal_pose = se3_exp(perturbation) @ query_view.pose_w2c
    proposal_image_xy, depth = project_world_points(
        atlas.xyz.reshape(-1, 3)[common],
        proposal_pose,
        query_view.camera,
    )
    map_xy = torch.from_numpy(map_image_xy / scale)[None].to(device)
    target_xy = torch.from_numpy(target_image_xy / scale)[None].to(device)
    proposal_xy = torch.from_numpy(
        proposal_image_xy.astype(np.float32) / scale
    )[None].to(device)
    positive = torch.from_numpy(
        np.isfinite(proposal_image_xy).all(axis=1) & (depth > 0.0)
    )[None].to(device)
    # Explicit wrong-texel/repeated-structure negatives: preserve query
    # matchability, but require the pairwise correlation outcome to be null.
    negative = rng.random(common.size) < 0.20
    if common.size >= 2 and not np.any(negative):
        negative[int(rng.integers(common.size))] = True
    permutation = rng.permutation(common.size)
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
    loss = local_correlation_training_loss(
        map_output,
        query_output,
        map_xy=map_xy,
        proposal_xy=proposal_xy,
        target_xy=target_xy,
        positive_mask=positive,
        query_matchable_mask=torch.ones_like(positive),
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
    return loss


def _validate(
    model: V6MetricEncoder,
    views: list[TrainingView],
    pairs: list[tuple[int, int, np.ndarray]],
    atlas: MapletFeatureAtlasBank,
    rng: np.random.Generator,
    args: argparse.Namespace,
    frozen_atlas: MapletFeatureAtlasBank | None = None,
) -> float:
    model.eval()
    validation_rng = np.random.default_rng(99173)
    values = []
    with torch.no_grad():
        for pair in pairs[: min(len(pairs), 4)]:
            loss = _episode(
                model,
                views,
                pair,
                atlas,
                validation_rng,
                samples=min(int(args.samples_per_episode), 128),
                device=str(args.device),
                frozen_atlas=frozen_atlas,
            )
            values.append(float(loss["total"].item()))
    model.train()
    return float(np.mean(values)) if values else float("inf")


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
    train_views = _load_views(
        Path(args.train_contributor_dir), atlas, Path(args.image_root)
    )
    validation_views = _load_views(
        Path(args.validation_contributor_dir), atlas, Path(args.image_root)
    )
    # Cross-trajectory positives are essential for viewpoint invariance. The
    # held-out validation trajectory remains completely absent from training.
    train_pairs = _pairs(train_views, allow_cross_trajectory=True)
    validation_pairs = _pairs(validation_views)
    train_trajectories = sorted({view.trajectory_id for view in train_views})
    validation_trajectories = sorted(
        {view.trajectory_id for view in validation_views}
    )
    if set(train_trajectories) & set(validation_trajectories):
        raise ValueError("training and validation trajectories must be disjoint")
    if not train_pairs or not validation_pairs:
        raise ValueError("insufficient within-trajectory atlas overlap")
    if str(args.initial_checkpoint):
        model, _initial_metadata = load_v6_metric_encoder(
            Path(args.initial_checkpoint), device=str(args.device)
        )
    else:
        model = V6MetricEncoder(
            V6MetricEncoderConfig(
                hidden_dim=int(args.hidden_dim), output_dim=int(args.output_dim)
            )
        ).to(str(args.device))
    if frozen_atlas is not None and frozen_atlas.feature_dim != model.config.output_dim:
        raise ValueError("frozen atlas and metric encoder dimensions differ")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    best_validation = float("inf")
    best_step = 0
    history = []
    for step in range(1, int(args.steps) + 1):
        pair = train_pairs[int(rng.integers(len(train_pairs)))]
        loss = _episode(
            model,
            train_views,
            pair,
            atlas,
            rng,
            samples=int(args.samples_per_episode),
            device=str(args.device),
            frozen_atlas=frozen_atlas,
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
                frozen_atlas,
            )
            row = {
                "step": step,
                "train_total": float(loss["total"].item()),
                "train_flow_epe": float(loss["flow_epe"].item()),
                "validation_total": validation,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation < best_validation:
                best_validation = validation
                best_step = step
                save_v6_metric_encoder(
                    checkpoint,
                    model,
                    {
                        "vfm_layer": "radio_final",
                        "output_stride": 4,
                        "best_step": step,
                        "best_validation_loss": validation,
                        "training_trajectories": train_trajectories,
                        "validation_trajectories": validation_trajectories,
                        "trajectory_disjoint_validation": True,
                        "uses_joint_6dof_perturbations": True,
                        "maximum_translation_m": 0.30,
                        "maximum_rotation_deg": 2.0,
                        "training_objectives": [
                            "local_correlation_nll",
                            "explicit_null",
                            "matchability",
                            "heteroscedastic_displacement_nll",
                            "frozen_atlas_correlation"
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
        "best_validation_loss": best_validation,
        "train_view_count": len(train_views),
        "validation_view_count": len(validation_views),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "training_trajectories": train_trajectories,
        "validation_trajectories": validation_trajectories,
        "trajectory_disjoint_validation": True,
        "uses_frozen_metric_atlas": frozen_atlas is not None,
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
