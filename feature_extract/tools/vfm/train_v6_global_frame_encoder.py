"""Fine-tune the V6 query student with runtime-global chart correlation.

The frozen map target is the feature-only 2DGS metric atlas.  Each training
example asks visible canonical chart cells to identify their projected
location among every query cell at the same pyramid level, so repeated facade
positions are real hard negatives.  Mapping RGB/reference images are never
stored in the runtime map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    load_v6_metric_encoder,
    save_v6_metric_encoder,
)


LEVELS = ("coarse", "middle", "fine")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_coarse", required=True)
    parser.add_argument("--atlas_middle", required=True)
    parser.add_argument("--atlas_fine", required=True)
    parser.add_argument("--initial_query_encoder", required=True)
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
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--validation_every", type=int, default=10)
    parser.add_argument("--samples_per_level", type=int, default=192)
    parser.add_argument("--validation_samples_per_level", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2207)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_atlases(
    atlases: Mapping[str, MapletFeatureAtlasBank],
    encoder_metadata: Mapping[str, object],
) -> None:
    coarse = atlases["coarse"]
    for level, atlas in atlases.items():
        if not np.array_equal(atlas.maplet_ids, coarse.maplet_ids):
            raise ValueError(f"{level} atlas identities differ")
        if not np.allclose(atlas.centers, coarse.centers, atol=1e-6):
            raise ValueError(f"{level} atlas geometry differs")
        if str((atlas.metadata or {}).get("metric_feature_level")) != level:
            raise ValueError(f"{level} atlas feature-level metadata differs")
    expected = str(
        encoder_metadata.get("compatible_map_encoder_sha256", "")
    )
    observed = str(
        (coarse.metadata or {}).get("metric_encoder_sha256", "")
    )
    if not expected or expected != observed:
        raise ValueError("query student is incompatible with frozen map atlas")


def _sample_rows(
    view: object,
    atlas: MapletFeatureAtlasBank,
    sample_count: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    flat_rows = np.asarray(view.visible_rows, dtype=np.int64)
    xy = np.asarray(view.image_xy, dtype=np.float32)
    flat_features = atlas.features.transpose(0, 2, 3, 1).reshape(
        -1, atlas.feature_dim
    )
    support = atlas.support_count.reshape(-1)
    usable = (
        (support[flat_rows] > 0)
        & (np.linalg.norm(flat_features[flat_rows], axis=1) > 0.5)
    )
    flat_rows = flat_rows[usable]
    xy = xy[usable]
    if flat_rows.size == 0:
        return flat_rows, xy
    count = min(int(sample_count), int(flat_rows.size))
    selected = generator.choice(flat_rows.size, size=count, replace=False)
    return flat_rows[selected], xy[selected]


def _soft_projection_nll(
    query_feature: torch.Tensor,
    map_feature: torch.Tensor,
    image_xy: np.ndarray,
    image_size_wh: tuple[int, int],
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if query_feature.ndim != 3:
        raise ValueError("query feature must have shape (C,H,W)")
    height, width = query_feature.shape[-2:]
    query_flat = query_feature.reshape(query_feature.shape[0], -1).T
    logits = map_feature @ query_flat.T / float(temperature)
    log_probability = F.log_softmax(logits, dim=1)
    xy = torch.from_numpy(np.asarray(image_xy, dtype=np.float32)).to(
        device=query_feature.device, dtype=query_feature.dtype
    )
    grid_x = (xy[:, 0] + 0.5) * width / image_size_wh[0] - 0.5
    grid_y = (xy[:, 1] + 0.5) * height / image_size_wh[1] - 0.5
    x0 = torch.clamp(torch.floor(grid_x).long(), 0, width - 1)
    y0 = torch.clamp(torch.floor(grid_y).long(), 0, height - 1)
    x1 = torch.clamp(x0 + 1, 0, width - 1)
    y1 = torch.clamp(y0 + 1, 0, height - 1)
    dx = torch.clamp(grid_x - x0, 0.0, 1.0)
    dy = torch.clamp(grid_y - y0, 0.0, 1.0)
    loss = torch.zeros((), device=query_feature.device)
    for px, py, weight in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        index = py * width + px
        loss = loss - torch.mean(
            weight * log_probability[
                torch.arange(index.numel(), device=index.device), index
            ]
        )
    target_x = torch.clamp(torch.round(grid_x).long(), 0, width - 1)
    target_y = torch.clamp(torch.round(grid_y).long(), 0, height - 1)
    target = target_y * width + target_x
    top_count = min(5, int(logits.shape[1]))
    top = torch.topk(logits, k=top_count, dim=1).indices
    top_xy = torch.stack(
        [top % width, torch.div(top, width, rounding_mode="floor")], dim=2
    )
    target_xy = torch.stack([target_x, target_y], dim=1)
    distance = torch.linalg.vector_norm(
        top_xy.float() - target_xy[:, None].float(), dim=2
    )
    positive = logits[
        torch.arange(target.numel(), device=target.device), target
    ]
    maximum_wrong = torch.where(
        torch.arange(logits.shape[1], device=logits.device)[None]
        == target[:, None],
        torch.full_like(logits, -torch.inf),
        logits,
    ).max(dim=1).values
    return loss, {
        "recall_at_1_radius1": float(
            torch.mean((distance[:, 0] <= 1.0).float()).item()
        ),
        "recall_at_5_radius1": float(
            torch.mean(torch.any(distance <= 1.0, dim=1).float()).item()
        ),
        "positive_cosine": float(
            torch.mean(positive * float(temperature)).item()
        ),
        "positive_wrong_margin": float(
            torch.mean((positive - maximum_wrong) * float(temperature)).item()
        ),
    }


def _view_loss(
    model: torch.nn.Module,
    view: object,
    atlases: Mapping[str, MapletFeatureAtlasBank],
    *,
    sample_count: int,
    temperature: float,
    generator: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(
        view.radio[None].to(device),
        view.rgb[None].to(device),
    )
    losses = []
    metrics: dict[str, float] = {}
    level_weight = {"coarse": 0.40, "middle": 0.35, "fine": 0.25}
    for level in LEVELS:
        atlas = atlases[level]
        rows, xy = _sample_rows(view, atlas, sample_count, generator)
        if rows.size == 0:
            continue
        flat = atlas.features.transpose(0, 2, 3, 1).reshape(
            -1, atlas.feature_dim
        )
        map_feature = torch.from_numpy(flat[rows]).to(
            device=device, dtype=output[level].dtype
        )
        loss, local_metrics = _soft_projection_nll(
            output[level][0],
            map_feature,
            xy,
            (int(view.camera.width), int(view.camera.height)),
            float(temperature),
        )
        losses.append(float(level_weight[level]) * loss)
        metrics[f"{level}_loss"] = float(loss.detach().item())
        metrics.update(
            {
                f"{level}_{name}": value
                for name, value in local_metrics.items()
            }
        )
    clean = torch.from_numpy(
        np.asarray(view.clean_surface_mask, dtype=np.float32)
    ).to(device=device)
    target_matchability = F.interpolate(
        clean[None, None],
        size=output["matchability_logits"].shape[-2:],
        mode="nearest",
    )
    matchability_loss = F.binary_cross_entropy_with_logits(
        output["matchability_logits"], target_matchability
    )
    metrics["matchability_loss"] = float(matchability_loss.detach().item())
    if not losses:
        raise ValueError(f"{view.image_id} has no global-frame targets")
    total = torch.stack(losses).sum() + 0.05 * matchability_loss
    metrics["total"] = float(total.detach().item())
    return total, metrics


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    views: Sequence[object],
    atlases: Mapping[str, MapletFeatureAtlasBank],
    *,
    sample_count: int,
    temperature: float,
    seed: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for index, view in enumerate(views):
        _loss, metrics = _view_loss(
            model,
            view,
            atlases,
            sample_count=int(sample_count),
            temperature=float(temperature),
            generator=np.random.default_rng(int(seed) + index),
            device=device,
        )
        rows.append(metrics)
    keys = sorted(set.intersection(*(set(value) for value in rows)))
    result = {
        key: float(np.mean([value[key] for value in rows])) for key in keys
    }
    result["selection_score"] = float(
        0.45 * result["coarse_recall_at_5_radius1"]
        + 0.35 * result["middle_recall_at_5_radius1"]
        + 0.20 * result["fine_recall_at_5_radius1"]
        + 0.05 * result["coarse_positive_wrong_margin"]
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if not bool(args.force) and (output.exists() or summary_path.exists()):
        raise FileExistsError("output exists; pass --force to replace it")
    atlas_paths = {
        level: Path(getattr(args, f"atlas_{level}")) for level in LEVELS
    }
    atlases = {
        level: MapletFeatureAtlasBank.load_npz(path)
        for level, path in atlas_paths.items()
    }
    initial_path = Path(args.initial_query_encoder)
    model, initial_metadata = load_v6_metric_encoder(
        initial_path, device=str(args.device)
    )
    _validate_atlases(atlases, initial_metadata)
    train_views = _load_views(
        Path(args.train_contributor_dir),
        atlases["coarse"],
        Path(args.image_root),
        tuple(str(value) for value in args.train_trajectory_ids),
    )
    validation_views = _load_views(
        Path(args.validation_contributor_dir),
        atlases["coarse"],
        Path(args.image_root),
        tuple(str(value) for value in args.validation_trajectory_ids),
    )
    train_trajectories = sorted(
        {str(view.trajectory_id) for view in train_views}
    )
    validation_trajectories = sorted(
        {str(view.trajectory_id) for view in validation_views}
    )
    mapping_trajectories = sorted(
        str(value)
        for value in (atlases["coarse"].metadata or {}).get(
            "mapping_trajectory_ids", []
        )
    )
    if (
        set(train_trajectories) & set(validation_trajectories)
        or set(train_trajectories) & set(mapping_trajectories)
        or set(validation_trajectories) & set(mapping_trajectories)
    ):
        raise ValueError("global-frame train/validation/map trajectories overlap")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    best_score = -float("inf")
    best_step = 0
    best_metrics: dict[str, float] = {}
    history = []
    generator = np.random.default_rng(int(args.seed))
    for step in range(1, int(args.steps) + 1):
        model.train()
        view = train_views[(step - 1) % len(train_views)]
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = _view_loss(
            model,
            view,
            atlases,
            sample_count=int(args.samples_per_level),
            temperature=float(args.temperature),
            generator=generator,
            device=str(args.device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if step == 1 or step % int(args.validation_every) == 0:
            validation = _validate(
                model,
                validation_views,
                atlases,
                sample_count=int(args.validation_samples_per_level),
                temperature=float(args.temperature),
                seed=int(args.seed) + 100000,
                device=str(args.device),
            )
            history.append(
                {
                    "step": step,
                    "train": train_metrics,
                    "validation": validation,
                }
            )
            if validation["selection_score"] > best_score:
                best_score = float(validation["selection_score"])
                best_step = int(step)
                best_metrics = dict(validation)
                metadata = dict(initial_metadata)
                metadata.update(
                    global_frame_training=True,
                    global_frame_objective=(
                        "full_query_soft_projection_nll_with_runtime_hard_negatives"
                    ),
                    global_frame_best_step=best_step,
                    global_frame_best_validation_metrics=best_metrics,
                    global_frame_temperature=float(args.temperature),
                    global_frame_training_trajectories=train_trajectories,
                    global_frame_validation_trajectories=validation_trajectories,
                    atlas_mapping_trajectories=mapping_trajectories,
                    initial_query_encoder_sha256=_sha256(initial_path),
                    compatible_map_encoder_sha256=str(
                        (atlases["coarse"].metadata or {}).get(
                            "metric_encoder_sha256", ""
                        )
                    ),
                    uses_runtime_global_hard_negatives=True,
                    uses_exact_query_surface_visibility_labels=True,
                    uses_ground_truth_depth_at_runtime=False,
                    uses_radio_intermediate=False,
                    uses_sfm_points=False,
                    uses_sfm_tracks=False,
                    uses_alike_descriptors=False,
                    uses_pairwise_image_matching=False,
                )
                save_v6_metric_encoder(output, model, metadata)
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_total": train_metrics["total"],
                        "validation": validation,
                        "best_step": best_step,
                    }
                ),
                flush=True,
            )
    payload = {
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "best_step": best_step,
        "best_validation_score": best_score,
        "best_validation_metrics": best_metrics,
        "steps": int(args.steps),
        "training_trajectories": train_trajectories,
        "validation_trajectories": validation_trajectories,
        "atlas_mapping_trajectories": mapping_trajectories,
        "trajectory_disjoint": True,
        "atlas_sha256": {
            level: _sha256(path) for level, path in atlas_paths.items()
        },
        "history": history,
        "map_contract": {
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
            "uses_alike_descriptors": False,
            "uses_pairwise_image_matching": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
