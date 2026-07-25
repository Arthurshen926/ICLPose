"""Pretrain candidate-specific RGB spatial likelihood on train SfM observations.

This command is deliberately upstream of P1 pose fitting.  It trains a
target-free scorer from many registered train-query observation pairs:

* candidate zero is a real same-track mapping observation at local offset 0;
* the remaining fixed candidates are different-track RADIO-PCA ANN negatives;
* the encoder receives only image IDs, image coordinates, RGB, and frozen
  RADIO/ALIKE grids, never a pose, track ID, coarse score, or target.

An inner query split is held out before training.  A checkpoint can seed P1
fine-tuning only after its own target-free appearance gate passes; it is never
promoted directly to pose ranking or PnP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_support_appearance,
    resolve_candidate_pose_rgb_spatial_context_windows,
    selected_candidate_view_log_likelihood_ratio_at_offsets,
    spatial_density_nll,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ContextAttentionSource,
    load_context_attention_source_headers,
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _crop_cached_rgb_windows_grouped,
    _load_query_rgb,
    _load_query_rgb_uint8,
    resolve_rgb_image_cache_storage_dtype,
)


OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT = (
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
)


@dataclass(frozen=True)
class _DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-radius-px", type=float, default=8.0)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument(
        "--rgb-cost-volume-only",
        action="store_true",
        help=(
            "Pretrain only the target-free real-RGB FPN/template cost volume; "
            "do not permit RADIO/ALIKE context or learned residual/dustbin paths."
        ),
    )
    parser.add_argument("--radio-final-context-window", type=int, default=9)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=9)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.5)
    parser.add_argument("--dustbin-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument(
        "--rgb-cache-dtype",
        choices=("float16", "uint8"),
        default="float16",
        help="Full-image RGB cache storage; uint8 avoids CPU float conversion on cache misses.",
    )
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _initialize_distributed(device_name: str) -> _DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP observation pretraining requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("observation pretraining requested an unavailable CUDA device")
    return _DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        enabled=enabled,
    )


def _finalize_distributed(state: _DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def _reduce_sum(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def _source_table(
    sources: Sequence[ContextAttentionSource],
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    by_name = {str(source.name): source for source in sources}
    if set(by_name) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("observation pretraining context sources are incomplete")
    reference = by_name["radio_final"]
    image_ids = np.asarray(reference.image_ids).astype(str)
    image_sizes = np.asarray(reference.image_sizes, dtype=np.int64)
    if len(image_ids) == 0 or image_sizes.shape != (len(image_ids), 2):
        raise ValueError("observation pretraining image source table is invalid")
    for source in by_name.values():
        if not np.array_equal(np.asarray(source.image_ids).astype(str), image_ids) or not np.array_equal(
            np.asarray(source.image_sizes, dtype=np.int64), image_sizes
        ):
            raise ValueError("observation pretraining context source ownership differs")
    return image_ids, image_sizes, {
        name: torch.from_numpy(np.asarray(source.grid, dtype=np.float32))
        for name, source in by_name.items()
    }


def _discover_rgb_image_size(*, image_root: Path, image_id: str) -> tuple[int, int]:
    from PIL import Image

    path = Path(image_root) / str(image_id)
    if not path.is_file():
        raise FileNotFoundError(f"RGB image is absent: {path}")
    with Image.open(path) as image:
        width, height = image.size
    if min(int(width), int(height)) <= 1:
        raise ValueError("RGB root has an invalid image size")
    return int(width), int(height)


def _rgb_coordinate_scale(
    *, coordinate_image_size: tuple[int, int], rgb_image_size: tuple[int, int]
) -> float:
    coordinate_width, coordinate_height = (int(value) for value in coordinate_image_size)
    rgb_width, rgb_height = (int(value) for value in rgb_image_size)
    if min(coordinate_width, coordinate_height, rgb_width, rgb_height) <= 1:
        raise ValueError("RGB coordinate bridge image sizes are invalid")
    scale_x = float(rgb_width) / float(coordinate_width)
    scale_y = float(rgb_height) / float(coordinate_height)
    if not math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("RGB and context coordinates are not isotropically aligned")
    return scale_x


def _validate_rgb_coordinate_bridge(
    *,
    source_metadata: Mapping[str, object],
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
) -> dict[str, object]:
    scale = _rgb_coordinate_scale(
        coordinate_image_size=coordinate_image_size, rgb_image_size=rgb_image_size
    )
    bridge = source_metadata.get("coordinate_bridge")
    if bridge is None:
        if coordinate_image_size != rgb_image_size:
            raise ValueError("context cache lacks a required RGB coordinate bridge")
        return {
            "format": "identity_rgb_coordinate_bridge_v1",
            "aligned_coordinate_size": list(coordinate_image_size),
            "raw_rgb_size": list(rgb_image_size),
            "raw_pixels_per_aligned_pixel": 1.0,
        }
    if not isinstance(bridge, Mapping):
        raise ValueError("context-cache RGB coordinate bridge is invalid")
    if (
        tuple(int(value) for value in bridge.get("aligned_coordinate_size", ()))
        != tuple(coordinate_image_size)
        or tuple(int(value) for value in bridge.get("raw_rgb_size", ()))
        != tuple(rgb_image_size)
        or not math.isclose(
            float(bridge.get("raw_pixels_per_aligned_pixel", float("nan"))),
            scale,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("context-cache RGB coordinate bridge differs from image root")
    return dict(bridge)


def _runtime_from_pair_rows(
    *, pairs: CandidatePoseRGBSpatialObservationPairs, rows: np.ndarray, image_index_by_id: Mapping[str, int]
) -> CandidatePoseRGBSpatialRuntime:
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(selected) == 0 or np.any(selected < 0) or np.any(selected >= pairs.row_count):
        raise ValueError("observation-pair batch rows are invalid")
    query_ids = pairs.query_image_ids[selected]
    support_ids = np.concatenate(
        (
            pairs.positive_support_image_ids[selected, None],
            pairs.negative_support_image_ids[selected],
        ),
        axis=1,
    )
    support_xy = np.concatenate(
        (
            pairs.positive_support_xy[selected, None, :],
            pairs.negative_support_xy[selected],
        ),
        axis=1,
    )
    try:
        query_indices = np.asarray(
            [image_index_by_id[str(value)] for value in query_ids.tolist()], dtype=np.int64
        )
        support_indices = np.asarray(
            [[image_index_by_id[str(value)] for value in row] for row in support_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("observation-pair image is absent from context sources") from error
    candidate_count = int(support_indices.shape[1])
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.from_numpy(query_indices),
        query_xy=torch.from_numpy(np.asarray(pairs.query_xy[selected], dtype=np.float32)),
        support_image_indices=torch.from_numpy(support_indices[:, :, None]),
        support_xy=torch.from_numpy(support_xy[:, :, None, :].astype(np.float32, copy=False)),
        support_view_valid=torch.ones((len(selected), candidate_count, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones(
            (len(selected), candidate_count, 1), dtype=torch.float32
        ),
        candidate_probabilities=torch.full(
            (len(selected), candidate_count), 1.0 / float(candidate_count), dtype=torch.float32
        ),
        null_probabilities=torch.zeros((len(selected),), dtype=torch.float32),
    )


def _crop_pair_rgb_patches(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = runtime.to("cpu")
    ids = np.asarray(image_ids).astype(str)
    scale = _rgb_coordinate_scale(
        coordinate_image_size=coordinate_image_size, rgb_image_size=rgb_image_size
    )
    width, height = (int(rgb_image_size[0]), int(rgb_image_size[1]))
    loaders: dict[str, Any] = {}

    def specification(index: int) -> tuple[str, Any]:
        image_id = str(ids[int(index)])
        loader = loaders.get(image_id)
        if loader is None:
            path = Path(image_root) / image_id
            if not path.is_file():
                raise FileNotFoundError(f"RGB image is absent: {path}")

            def load(path: Path = path) -> torch.Tensor:
                image = (
                    _load_query_rgb_uint8(path)
                    if cache.storage_dtype == torch.uint8
                    else _load_query_rgb(path)
                )
                if image.ndim != 3 or image.shape[0] != 3 or (
                    int(image.shape[2]), int(image.shape[1])
                ) != (width, height):
                    raise ValueError(f"RGB image size differs from context source: {path}")
                return image

            loader = load
            loaders[image_id] = loader
        return image_id, loader

    query_indices = active.query_image_indices.detach().cpu().numpy().astype(np.int64)
    query_specs = [specification(int(index)) for index in query_indices.tolist()]
    query_centers = (active.query_xy.detach().cpu().numpy() * scale).tolist()
    query_patches = _crop_cached_rgb_windows_grouped(
        query_specs,
        query_centers,
        cache=cache,
        cache_device=device,
        crop_device=device,
        radius_px=float(radius_px) * scale,
        step_px=float(step_px) * scale,
        image_width=width,
        image_height=height,
    )
    point_count, candidate_count, view_count = active.support_image_indices.shape
    if view_count != 1:
        raise ValueError("observation-pair pretraining expects one support view per candidate")
    support_indices = active.support_image_indices.reshape(-1).detach().cpu().numpy().astype(np.int64)
    support_centers = active.support_xy.reshape(-1, 2).detach().cpu().numpy()
    support_patches = _crop_cached_rgb_windows_grouped(
        [specification(int(index)) for index in support_indices.tolist()],
        (support_centers * scale).tolist(),
        cache=cache,
        cache_device=device,
        crop_device=device,
        radius_px=float(radius_px) * scale,
        step_px=float(step_px) * scale,
        image_width=width,
        image_height=height,
    )
    return query_patches, support_patches.reshape(
        point_count, candidate_count, view_count, *support_patches.shape[1:]
    )


def _center_candidate_scores(
    *, runtime: CandidatePoseRGBSpatialRuntime, prediction: CandidatePoseRGBSpatialEdgePrediction
) -> tuple[torch.Tensor, torch.Tensor]:
    point_count = runtime.point_count
    candidate_count = runtime.candidate_count
    device = prediction.joint_log_probabilities.device
    points = torch.arange(point_count, device=device).repeat_interleave(candidate_count)
    candidates = torch.arange(candidate_count, device=device).repeat(point_count)
    scores, usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=points,
        candidate_indices=candidates,
        offsets_xy=torch.zeros((len(points), 2), device=device),
        projection_valid=torch.ones((len(points),), dtype=torch.bool, device=device),
    )
    return scores.reshape(point_count, candidate_count), usable.reshape(point_count, candidate_count)


def observation_pair_margin_loss(
    *, scores: torch.Tensor, usable: torch.Tensor, margin: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require candidate zero to beat the hardest different-track candidate."""

    values = torch.as_tensor(scores, dtype=torch.float32)
    valid = torch.as_tensor(usable, dtype=torch.bool, device=values.device)
    if values.ndim != 2 or values.shape[1] < 2 or valid.shape != values.shape:
        raise ValueError("observation-pair margin inputs are invalid")
    required_margin = float(margin)
    if not math.isfinite(required_margin) or required_margin < 0.0:
        raise ValueError("observation-pair margin is invalid")
    negative_scores = values[:, 1:].masked_fill(~valid[:, 1:], -torch.inf)
    hardest_negative = negative_scores.max(dim=1).values
    active = valid[:, 0] & torch.isfinite(hardest_negative)
    if not bool(active.any()):
        loss = values.sum() * 0.0
        return loss, {
            "active_rows": 0.0,
            "mean_positive_minus_hardest_negative": 0.0,
            "correct_win_fraction": 0.0,
            "margin_loss": 0.0,
        }
    gaps = values[active, 0] - hardest_negative[active]
    loss = F.relu(required_margin - gaps).mean()
    return loss, {
        "active_rows": float(active.sum().item()),
        "mean_positive_minus_hardest_negative": float(gaps.detach().mean().item()),
        "correct_win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
        "margin_loss": float(loss.detach().item()),
    }


def _density_targets(
    *, point_count: int, candidate_count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.zeros((point_count, candidate_count, 2), dtype=torch.float32, device=device)
    dustbin = torch.ones((point_count, candidate_count), dtype=torch.bool, device=device)
    dustbin[:, 0] = False
    return offsets, dustbin, torch.ones_like(dustbin)


def _configure_rgb_cost_volume_only_trainable_parameters(
    model: CandidatePoseRGBSpatialLikelihood,
) -> tuple[str, ...]:
    """Freeze non-RGB parameters before DDP captures the trainable set."""

    selected: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("texture_encoder.")
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    if not selected:
        raise RuntimeError("RGB-only observation pretrain found no texture parameters")
    return tuple(selected)


def _evaluate_inner_validation(
    *,
    model: CandidatePoseRGBSpatialLikelihood,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    radius_px: float,
    step_px: float,
    batch_size: int,
    margin: float,
    dustbin_loss_weight: float,
    cache: TensorImageLRUCache,
    device: torch.device,
    amp_enabled: bool,
    rgb_cost_volume_only: bool = False,
) -> dict[str, float]:
    model.eval()
    totals = np.zeros((8,), dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(rows), int(batch_size)):
            batch_rows = rows[begin : begin + int(batch_size)]
            runtime = _runtime_from_pair_rows(
                pairs=pairs, rows=batch_rows, image_index_by_id=image_index_by_id
            )
            query_patches, support_patches = _crop_pair_rgb_patches(
                runtime=runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=radius_px,
                step_px=step_px,
                cache=cache,
                device=device,
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(
                    runtime=runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=support_patches,
                    rgb_cost_volume_only=bool(rgb_cost_volume_only),
                )
                offsets, dustbin, supervised = _density_targets(
                    point_count=runtime.point_count,
                    candidate_count=runtime.candidate_count,
                    device=device,
                )
                if bool(rgb_cost_volume_only):
                    supervised = ~dustbin
                    dustbin = torch.zeros_like(dustbin)
                density_loss, _density_metrics = spatial_density_nll(
                    prediction=prediction,
                    target_offsets_xy=offsets,
                    target_dustbin=dustbin,
                    target_supervised=supervised,
                    dustbin_weight=float(dustbin_loss_weight),
                    balance_observed_and_dustbin=True,
                )
                scores, usable = _center_candidate_scores(
                    runtime=runtime, prediction=prediction
                )
                margin_loss, margin_metrics = observation_pair_margin_loss(
                    scores=scores, usable=usable, margin=float(margin)
                )
                permuted_runtime = permute_runtime_support_appearance(runtime, shift=1)
                permuted_prediction = model(
                    runtime=permuted_runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=torch.roll(support_patches, shifts=1, dims=1),
                    rgb_cost_volume_only=bool(rgb_cost_volume_only),
                )
                permuted_scores, permuted_usable = _center_candidate_scores(
                    runtime=permuted_runtime, prediction=permuted_prediction
                )
                _permuted_loss, permuted_metrics = observation_pair_margin_loss(
                    scores=permuted_scores,
                    usable=permuted_usable,
                    margin=float(margin),
                )
            active = float(margin_metrics["active_rows"])
            totals += np.asarray(
                [
                    active,
                    active * float(margin_metrics["margin_loss"]),
                    active * float(margin_metrics["mean_positive_minus_hardest_negative"]),
                    active * float(margin_metrics["correct_win_fraction"]),
                    active * float(permuted_metrics["mean_positive_minus_hardest_negative"]),
                    active * float(permuted_metrics["correct_win_fraction"]),
                    active * float(density_loss.detach().item()),
                    1.0,
                ],
                dtype=np.float64,
            )
    if totals[0] <= 0.0:
        raise RuntimeError("observation-pair inner validation has no usable positive/negative rows")
    return {
        "normal_margin_loss": float(totals[1] / totals[0]),
        "normal_mean_gap": float(totals[2] / totals[0]),
        "normal_correct_win_fraction": float(totals[3] / totals[0]),
        "permuted_mean_gap": float(totals[4] / totals[0]),
        "permuted_correct_win_fraction": float(totals[5] / totals[0]),
        "visual_gap_delta": float((totals[2] - totals[4]) / totals[0]),
        "spatial_density_nll": float(totals[6] / totals[0]),
        "batch_count": float(totals[7]),
        "active_rows": float(totals[0]),
    }


def _is_better_inner_validation_epoch(
    *, candidate: Mapping[str, float], incumbent: Mapping[str, float] | None
) -> bool:
    if incumbent is None:
        return True
    loss = float(candidate["normal_margin_loss"])
    previous = float(incumbent["normal_margin_loss"])
    if loss < previous - 1e-12:
        return True
    return abs(loss - previous) <= 1e-12 and float(
        candidate["normal_correct_win_fraction"]
    ) > float(incumbent["normal_correct_win_fraction"])


def observation_pretrain_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, object]:
    required = {
        "normal_correct_win_fraction": float(metrics["normal_correct_win_fraction"]),
        "normal_mean_gap": float(metrics["normal_mean_gap"]),
        "visual_gap_delta": float(metrics["visual_gap_delta"]),
    }
    if not all(math.isfinite(value) for value in required.values()):
        raise ValueError("observation pretrain gate metrics are invalid")
    checks = {
        "correct_win_fraction": required["normal_correct_win_fraction"]
        >= float(minimum_win_fraction),
        "normal_gap": required["normal_mean_gap"] >= float(minimum_normal_gap),
        "visual_gap_delta": required["visual_gap_delta"]
        >= float(minimum_visual_gap_delta),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_win_fraction": float(minimum_win_fraction),
            "minimum_normal_gap": float(minimum_normal_gap),
            "minimum_visual_gap_delta": float(minimum_visual_gap_delta),
        },
    }


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.search_radius_px),
        float(args.context_radius_px),
        float(args.step_px),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.margin),
        float(args.margin_loss_weight),
        float(args.density_loss_weight),
        float(args.dustbin_loss_weight),
        float(args.gradient_clip_norm),
        float(args.rgb_cache_gb),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_visual_gap_delta),
    )
    if (
        int(args.batch_size) <= 1
        or int(args.epochs) <= 0
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.search_radius_px) < float(args.step_px)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.margin) < 0.0
        or float(args.margin_loss_weight) <= 0.0
        or float(args.density_loss_weight) <= 0.0
        or float(args.dustbin_loss_weight) < 0.0
        or float(args.gradient_clip_norm) <= 0.0
        or float(args.rgb_cache_gb) <= 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or float(args.minimum_visual_gap_delta) < 0.0
    ):
        raise ValueError("observation pretraining arguments are invalid")
    if bool(args.rgb_cost_volume_only) and float(args.dustbin_loss_weight) != 0.0:
        raise ValueError("RGB-only observation pretraining requires zero dustbin loss weight")
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _rank_batch_rows(
    *, rows: np.ndarray, batch_size: int, rank: int, world_size: int, seed: int, epoch: int
) -> list[np.ndarray]:
    order = np.random.default_rng(int(seed) + int(epoch) * 1000003).permutation(rows)
    per_rank = int(math.ceil(len(order) / float(world_size)))
    padded = np.resize(order, per_rank * int(world_size))
    rank_rows = padded[int(rank) * per_rank : (int(rank) + 1) * per_rank]
    step_count = int(math.ceil(len(rank_rows) / float(batch_size)))
    return [
        np.resize(rank_rows[begin : begin + int(batch_size)], int(batch_size))
        for begin in range(0, step_count * int(batch_size), int(batch_size))
    ]


def pretrain_candidate_pose_rgb_spatial_likelihood(args: argparse.Namespace) -> dict[str, object]:
    """Run DDP pretraining and save only a target-free model state dict."""

    context_windows = _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_rgb_spatial_observation_pretrain.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if state.rank == 0 and (
            checkpoint_path.exists() or history_path.exists() or summary_path.exists()
        ) and not bool(args.force):
            raise FileExistsError("refusing to overwrite observation pretraining output")
        if state.enabled:
            distributed.barrier()
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - older torch releases.
            pass

        pair_path = Path(args.observation_pairs)
        pairs = load_candidate_pose_rgb_spatial_observation_pairs(pair_path)
        train_rows = np.flatnonzero(pairs.split_names == "inner_train").astype(np.int64)
        validation_rows = np.flatnonzero(
            pairs.split_names == "inner_validation"
        ).astype(np.int64)
        if len(train_rows) == 0 or len(validation_rows) == 0:
            raise ValueError("observation pairs lack an inner train or validation partition")
        rgb_cost_volume_only = bool(args.rgb_cost_volume_only)
        if rgb_cost_volume_only:
            source_headers = load_context_attention_source_headers(
                radio_final_context_cache=Path(args.radio_final_context_cache),
                radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
                alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
                expected_radio_checkpoint="",
            )
            image_ids = source_headers.image_ids
            image_sizes = source_headers.image_sizes
            source_tensors = None
            context_source_dimensions = source_headers.descriptor_dimensions
            source_metadata = source_headers.metadata_by_name["radio_final"]
        else:
            sources = load_context_attention_sources(
                radio_final_context_cache=Path(args.radio_final_context_cache),
                radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
                alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
                expected_radio_checkpoint="",
                require_equal_descriptor_dimensions=False,
            )
            image_ids, image_sizes, source_tensors = _source_table(sources)
            context_source_dimensions = None
            source_metadata = sources[0].metadata
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("observation pretraining currently requires common image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = _validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        image_index_by_id = {
            str(image_id): index for index, image_id in enumerate(image_ids.tolist())
        }
        expected_pair_ids = set(pairs.query_image_ids.tolist())
        expected_pair_ids.update(pairs.positive_support_image_ids.tolist())
        expected_pair_ids.update(pairs.negative_support_image_ids.reshape(-1).tolist())
        if not expected_pair_ids.issubset(image_index_by_id):
            raise ValueError("observation pairs reference an image absent from context sources")

        model = CandidatePoseRGBSpatialLikelihood(
            sources=source_tensors,
            context_source_dimensions=context_source_dimensions,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            search_radius_px=float(args.search_radius_px),
            context_radius_px=float(args.context_radius_px),
            step_px=float(args.step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=True,
            context_windows=context_windows,
        )
        if rgb_cost_volume_only:
            trainable_parameter_names = _configure_rgb_cost_volume_only_trainable_parameters(
                model
            )
        else:
            trainable_parameter_names = tuple(
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            )
        model = model.to(state.device)
        model_for_train: torch.nn.Module
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        core_model = model_for_train.module if state.enabled else model_for_train
        assert isinstance(core_model, CandidatePoseRGBSpatialLikelihood)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model_for_train.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        best_metrics: dict[str, float] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        history: list[dict[str, object]] = []
        start_time = time.time()
        patch_radius = float(args.search_radius_px) + float(args.context_radius_px)
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            batches = _rank_batch_rows(
                rows=train_rows,
                batch_size=int(args.batch_size),
                rank=state.rank,
                world_size=state.world_size,
                seed=int(args.seed),
                epoch=int(epoch),
            )
            totals = torch.zeros((7,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for batch_rows in batches:
                runtime = _runtime_from_pair_rows(
                    pairs=pairs, rows=batch_rows, image_index_by_id=image_index_by_id
                )
                query_patches, support_patches = _crop_pair_rgb_patches(
                    runtime=runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=patch_radius,
                    step_px=float(args.step_px),
                    cache=cache,
                    device=state.device,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(
                        runtime=runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        rgb_cost_volume_only=rgb_cost_volume_only,
                    )
                    offsets, dustbin, supervised = _density_targets(
                        point_count=runtime.point_count,
                        candidate_count=runtime.candidate_count,
                        device=state.device,
                    )
                    if rgb_cost_volume_only:
                        supervised = ~dustbin
                        dustbin = torch.zeros_like(dustbin)
                    density_loss, density_metrics = spatial_density_nll(
                        prediction=prediction,
                        target_offsets_xy=offsets,
                        target_dustbin=dustbin,
                        target_supervised=supervised,
                        dustbin_weight=float(args.dustbin_loss_weight),
                        balance_observed_and_dustbin=True,
                    )
                    scores, usable = _center_candidate_scores(
                        runtime=runtime, prediction=prediction
                    )
                    margin_loss, margin_metrics = observation_pair_margin_loss(
                        scores=scores, usable=usable, margin=float(args.margin)
                    )
                    loss = (
                        float(args.margin_loss_weight) * margin_loss
                        + float(args.density_loss_weight) * density_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model_for_train.parameters(), float(args.gradient_clip_norm)
                )
                scaler.step(optimizer)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(loss.detach().item()),
                        float(margin_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(margin_metrics["mean_positive_minus_hardest_negative"]),
                        float(margin_metrics["correct_win_fraction"]),
                        float(margin_metrics["active_rows"]),
                        float(density_metrics["spatial_density_active_edges"]),
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce_sum(state, totals)
            global_steps = int(len(batches) * state.world_size)
            if state.rank == 0:
                inner = _evaluate_inner_validation(
                    model=core_model,
                    pairs=pairs,
                    rows=validation_rows,
                    image_index_by_id=image_index_by_id,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=patch_radius,
                    step_px=float(args.step_px),
                    batch_size=int(args.batch_size),
                    margin=float(args.margin),
                    dustbin_loss_weight=float(args.dustbin_loss_weight),
                    cache=cache,
                    device=state.device,
                    amp_enabled=amp_enabled,
                    rgb_cost_volume_only=rgb_cost_volume_only,
                )
                if _is_better_inner_validation_epoch(
                    candidate=inner, incumbent=best_metrics
                ):
                    best_metrics = dict(inner)
                    best_epoch = int(epoch + 1)
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in core_model.state_dict().items()
                    }
                record: dict[str, object] = {
                    "epoch": int(epoch + 1),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_margin_loss": float((totals[1] / global_steps).item()),
                    "train_density_loss": float((totals[2] / global_steps).item()),
                    "train_mean_positive_minus_hardest_negative": float(
                        (totals[3] / global_steps).item()
                    ),
                    "train_correct_win_fraction": float((totals[4] / global_steps).item()),
                    "train_margin_active_rows_per_step": float(
                        (totals[5] / global_steps).item()
                    ),
                    "train_spatial_active_edges_per_step": float(
                        (totals[6] / global_steps).item()
                    ),
                    "epoch_seconds": float(time.time() - epoch_start),
                    **{f"inner_{key}": value for key, value in inner.items()},
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank == 0:
            if best_state_dict is None or best_metrics is None or best_epoch < 1:
                raise RuntimeError("observation pretraining did not select an inner checkpoint")
            gate = observation_pretrain_gate(
                best_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata: dict[str, object] = {
                "format": OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "architecture": (
                    "real_rgb_fpn_candidate_specific_high_resolution_cost_volume_only_observation_pretrain_v1"
                    if rgb_cost_volume_only
                    else "local_2d_radio_final_intermediate_and_alike_context_plus_high_resolution_real_rgb_cost_volume_observation_identity_pretrain_v1"
                ),
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "p1_finetune_allowed": bool(gate["passed"]),
                "raw_scores_must_not_feed_pnp": True,
                "encoder_inputs": (
                    [
                        "query_image_xy",
                        "fixed_support_image_xy",
                        "real_rgb_query_and_support_patches",
                    ]
                    if rgb_cost_volume_only
                    else [
                        "query_image_xy",
                        "fixed_support_image_xy",
                        "full_2d_radio_final_context_crop",
                        "full_2d_radio_intermediate_context_crop",
                        "full_2d_alike_context_crop",
                        "real_rgb_query_and_support_patches",
                    ]
                ),
                "encoder_excludes": [
                    "pose_matrix",
                    "projection_offset",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "candidate_rank",
                    "coarse_score",
                ],
                "config": {
                    "search_radius_px": float(args.search_radius_px),
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_context_log_ratio": float(args.max_abs_context_log_ratio),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "rgb_cache_dtype": str(args.rgb_cache_dtype),
                    "rgb_cost_volume_only": rgb_cost_volume_only,
                    "trainable_parameter_scope": (
                        "texture_encoder_only" if rgb_cost_volume_only else "all_candidate_likelihood_parameters"
                    ),
                    "trainable_parameter_count": int(len(trainable_parameter_names)),
                    "context_windows": dict(context_windows),
                },
                "training": {
                    "objective": (
                        "same_track_center_observed_rgb_density_plus_distinct_track_radio_pca_ann_hardest_margin_rgb_cost_volume_only_v1"
                        if rgb_cost_volume_only
                        else "same_track_center_offset_density_plus_distinct_track_radio_pca_ann_hardest_margin_v1"
                    ),
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "margin": float(args.margin),
                    "margin_loss_weight": float(args.margin_loss_weight),
                    "density_loss_weight": float(args.density_loss_weight),
                    "dustbin_loss_weight": float(args.dustbin_loss_weight),
                    "dustbin_targets_ignored_by_rgb_cost_volume_only": rgb_cost_volume_only,
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only",
                        "selected_epoch": int(best_epoch),
                        "selected_metrics": best_metrics,
                        "support_permutation_control": "candidate_slot_cyclic_shift_not_used_by_training_loss_v1",
                        "gate": gate,
                    },
                },
                "lineage": {
                    "observation_pairs": str(pair_path.resolve()),
                    "observation_pairs_sha256": file_sha256_short(pair_path),
                    "radio_final_context_cache_sha256": file_sha256_short(
                        Path(args.radio_final_context_cache)
                    ),
                    "radio_intermediate_context_cache_sha256": file_sha256_short(
                        Path(args.radio_intermediate_context_cache)
                    ),
                    "alike_spatial_context_cache_sha256": file_sha256_short(
                        Path(args.alike_spatial_context_cache)
                    ),
                    "source_image_manifest_sha256": str(
                        source_metadata.get("source_image_manifest_sha256", "")
                    ),
                    "rgb_coordinate_bridge": rgb_bridge,
                },
            }
            torch.save(
                {
                    "format": OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
                    "state_dict": best_state_dict,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            summary = {
                "stage": "pretrain_candidate_pose_rgb_spatial_likelihood",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "checkpoint_selection": {
                    "selected_epoch": int(best_epoch),
                    "inner_validation": best_metrics,
                    "gate": gate,
                },
                "rgb_cache_rank0": cache.summary(),
                "protocol": {
                    "train_only_observation_targets": True,
                    "runtime_checkpoint_target_free": True,
                    "validation_or_test_labels_used_by_fit": False,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "pnp_or_heldout_pose_not_run": True,
                },
            }
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {"checkpoint": str(checkpoint_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = pretrain_candidate_pose_rgb_spatial_likelihood(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
