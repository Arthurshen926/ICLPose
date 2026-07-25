"""Fine-tune the mapper directly on frozen P1 candidate groups.

This trainer addresses a specific supervision mismatch in the ordinary
image-to-image mapper path: coherent wrong-pose identities are defined for
fixed P1 query tokens, while sparse SfM observations can be tens of pixels
away.  It therefore maps each real RADIO feature map, samples the exact P1
coordinates, and trains against the immutable top-L projected-observation
bank.

The P1 runtime proposal is target-free.  Registered-positive and
coherent-wrong memberships are loaded only from a separate train-only artifact
after all target-free proposals and PnP hypotheses have already been frozen.
The resulting mapper must be followed by a projected-observation bank rebuild
before any retrieval or pose evaluation.

Typical two-GPU run::

    torchrun --standalone --nproc_per_node=2 \
      feature_extract/tools/vfm/train_current_p1_landmark_mapper.py \
      --p1-proposals .../current_p1_train_proposals_inference_only_v1.npz \
      --p1-targets .../pose_conditioned_system_hard_modes_v2.npz \
      --projected-landmark-bank .../projected_observations_mean.npz \
      --token-manifest output/vfm_tokens_radio/OldHospital/train_manifest.json \
      --image-root /hy-tmp/Cambridge_stdloc/OldHospital/processed \
      --init-checkpoint .../joint.pt \
      --output-dir .../direct_p1_mapper_v1
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.current_p1_mapper_direct import (
    CurrentP1MapperDirectTargets,
    CurrentP1MapperRuntime,
    current_p1_mapper_direct_loss,
    dense_descriptor_anchor_distillation_loss,
    group_current_p1_runtime_rows_by_query,
    load_current_p1_mapper_direct_supervision,
    sample_mapper_descriptors_at_p1_xy,
)
from feature_extract.vfm.matcha_joint_training import (
    MatchaJointTrainingRun,
    load_matcha_joint_model,
    save_matcha_joint_model,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


DIRECT_P1_MAPPER_TRAINER_FORMAT = "current_p1_direct_mapper_trainer_v1"
_EVALUATION_METRIC_KEYS = (
    "direct_positive_row_count",
    "direct_positive_edge_count",
    "direct_positive_nll_loss",
    "direct_candidate_top1_positive_rate",
    "direct_positive_similarity_mean",
    "direct_coherent_mode_count",
    "direct_coherent_mode_row_count",
    "direct_coherent_margin_loss",
    "direct_coherent_mean_log_posterior_gap",
    "direct_coherent_violation_fraction",
    "direct_total_loss",
    "anchor_distillation_loss",
    "anchor_descriptor_cosine_mean",
    "anchor_token_count",
    "anchor_excluded_token_count",
    "mapper_total_loss",
)


@dataclass(frozen=True)
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool

    @property
    def is_primary(self) -> bool:
        return int(self.rank) == 0


@dataclass(frozen=True)
class QueryGroup:
    query_id: str
    rows: np.ndarray
    token_path: Path
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64).reshape(-1)
        width, height = int(self.image_size[0]), int(self.image_size[1])
        if (
            not str(self.query_id)
            or len(rows) == 0
            or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0)
            or not Path(self.token_path).exists()
            or width <= 1
            or height <= 1
        ):
            raise ValueError("direct P1 mapper query group is invalid")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "token_path", Path(self.token_path))
        object.__setattr__(self, "image_size", (width, height))


@dataclass(frozen=True)
class _TrainingStepLoss:
    """Direct-P1 objective plus optional train-only dense mapper anchoring."""

    total_loss: torch.Tensor
    direct_loss: torch.Tensor
    anchor_loss: torch.Tensor
    metrics: dict[str, float]


class _FeatureMapCache:
    """Per-rank bounded CPU cache for lazily read RADIO feature maps."""

    def __init__(self, *, feature_key: str, max_images: int) -> None:
        if not str(feature_key) or int(max_images) < 0:
            raise ValueError("direct P1 feature cache configuration is invalid")
        self.feature_key = str(feature_key)
        self.max_images = int(max_images)
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def load(self, path: Path) -> np.ndarray:
        value = Path(path)
        cached = self._items.get(value)
        if cached is not None:
            self.hits += 1
            self._items.move_to_end(value)
            return cached
        self.misses += 1
        with np.load(value, allow_pickle=False) as payload:
            if self.feature_key not in payload:
                raise KeyError(f"{value} does not contain feature key {self.feature_key!r}")
            array = np.asarray(payload[self.feature_key], dtype=np.float32)
        if array.ndim != 3 or array.shape[0] <= 0 or not np.isfinite(array).all():
            raise ValueError(f"direct P1 feature map is invalid: {value}")
        if self.max_images > 0:
            self._items[value] = array
            self._items.move_to_end(value)
            while len(self._items) > self.max_images:
                self._items.popitem(last=False)
        return array

    def summary(self) -> dict[str, int]:
        return {
            "capacity": int(self.max_images),
            "entries": int(len(self._items)),
            "hits": int(self.hits),
            "misses": int(self.misses),
        }


class _FullMapDescriptorForward(nn.Module):
    """Expose only the mapper descriptor branch through a DDP-compatible forward."""

    def __init__(self, mapper: nn.Module) -> None:
        super().__init__()
        self.mapper = mapper

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        descriptors, _heatmap, _offsets = self.mapper.forward_feature_map(feature_maps)
        return descriptors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-proposals", required=True)
    parser.add_argument("--p1-targets", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-key", default="radio_final")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-query-images", type=int, default=8)
    parser.add_argument(
        "--coherent-queries-per-batch",
        type=int,
        default=1,
        help="Exact-P1 coherent query groups guaranteed in every per-rank batch.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--coherent-margin", type=float, default=0.05)
    parser.add_argument("--coherent-margin-weight", type=float, default=0.25)
    parser.add_argument("--coherent-min-mode-rows", type=int, default=4)
    parser.add_argument(
        "--anchor-distillation-weight",
        type=float,
        default=0.0,
        help="Training-only dense frozen-mapper anchor weight; zero preserves the legacy direct-P1 objective.",
    )
    parser.add_argument(
        "--anchor-exclusion-radius-tokens",
        type=int,
        default=1,
        help="Chebyshev radius around exact P1 cells excluded from dense descriptor anchoring.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-cache-images", type=int, default=16)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--evaluation-query-limit",
        type=int,
        default=0,
        help="Diagnostic limit only; zero evaluates every train-only inner-validation query.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=0,
        help="Run train-only query-grouped inner validation every N steps; zero means final only.",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    finite_positive = (float(args.lr), float(args.temperature), float(args.gradient_clip_norm))
    finite_nonnegative = (
        float(args.weight_decay),
        float(args.coherent_margin),
        float(args.coherent_margin_weight),
        float(args.anchor_distillation_weight),
        float(args.inner_validation_fraction),
    )
    if (
        not str(args.feature_key)
        or int(args.steps) <= 0
        or int(args.batch_query_images) <= 0
        or int(args.coherent_queries_per_batch) <= 0
        or int(args.coherent_queries_per_batch) > int(args.batch_query_images)
        or int(args.coherent_min_mode_rows) < 2
        or int(args.anchor_exclusion_radius_tokens) < 0
        or int(args.feature_cache_images) < 0
        or int(args.evaluation_query_limit) < 0
        or int(args.log_interval) <= 0
        or int(args.validation_interval) < 0
        or int(args.seed) < 0
        or not all(math.isfinite(value) and value > 0.0 for value in finite_positive)
        or not all(math.isfinite(value) and value >= 0.0 for value in finite_nonnegative)
        or float(args.inner_validation_fraction) >= 0.5
    ):
        raise ValueError("direct P1 mapper training arguments are invalid")


def _init_distributed() -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process direct P1 mapper training requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if enabled else "cuda:0")
    else:
        device = torch.device("cpu")
    return DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        enabled=enabled,
    )


def _finish_distributed(state: DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def _seed_everything(seed: int, state: DistributedState) -> None:
    value = int(seed) + int(state.rank) * 1_000_003
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _stable_query_order(query_ids: Iterable[str], *, seed: int) -> list[str]:
    def key(query_id: str) -> bytes:
        return hashlib.sha256(f"{int(seed)}\0{query_id}".encode("utf-8")).digest()

    return sorted((str(query_id) for query_id in query_ids), key=key)


def split_train_only_inner_validation(
    query_ids: Iterable[str], *, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Create a deterministic query-disjoint inner validation split from train only."""

    ordered = _stable_query_order(query_ids, seed=int(seed))
    if len(ordered) < 2:
        raise ValueError("direct P1 mapper needs at least two train query groups")
    value = float(fraction)
    if value <= 0.0:
        return ordered, []
    validation_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * value))))
    validation = sorted(ordered[:validation_count])
    training = sorted(ordered[validation_count:])
    if not training or set(training).intersection(validation):
        raise RuntimeError("direct P1 mapper inner split is invalid")
    return training, validation


def validate_distributed_evaluation_cardinality(
    *,
    query_count: int,
    state: DistributedState,
) -> None:
    """Reject diagnostic evaluation shards that cannot occupy every DDP rank."""

    count = int(query_count)
    if count < 0:
        raise ValueError("direct P1 mapper evaluation query count is invalid")
    if state.enabled and 0 < count < int(state.world_size):
        raise ValueError(
            "direct P1 mapper DDP evaluation requires at least one query per rank; "
            f"query_count={count}, world_size={state.world_size}. "
            "Use an unbounded evaluation split or run the small diagnostic on one GPU."
        )


def _load_query_groups(
    *,
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    token_manifest: Path,
    image_root: Path,
) -> dict[str, QueryGroup]:
    manifest = TokenBankManifest.from_json(Path(token_manifest))
    records = {str(record.image_id): record for record in manifest.records}
    if len(records) != len(manifest.records):
        raise ValueError("direct P1 token manifest contains duplicate image IDs")
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(root)
    groups = group_current_p1_runtime_rows_by_query(runtime, targets)
    output: dict[str, QueryGroup] = {}
    for query_id, rows in groups.items():
        record = records.get(str(query_id))
        if record is None:
            raise ValueError(f"direct P1 query is absent from token manifest: {query_id!r}")
        image_path = root / str(query_id)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            size = (int(image.width), int(image.height))
        output[str(query_id)] = QueryGroup(
            query_id=str(query_id),
            rows=rows,
            token_path=Path(record.token_path),
            image_size=size,
        )
    return output


def direct_target_coverage_audit(
    *,
    targets: CurrentP1MapperDirectTargets,
    minimum_mode_rows: int,
) -> dict[str, object]:
    """Audit how much coherent supervision reaches exact P1 rows before training."""

    minimum = int(minimum_mode_rows)
    if minimum < 2:
        raise ValueError("direct P1 minimum coherent mode rows must be at least two")
    eligible_rows_by_mode: dict[int, int] = {}
    all_rows_by_mode: dict[int, int] = {}
    for row in range(targets.row_count):
        has_positive = bool(np.any(targets.positive_mask[row]))
        for slot, mode_id in enumerate(targets.hard_mode_ids[row].tolist()):
            if int(mode_id) < 0:
                continue
            hard_members = (
                targets.hard_mode_candidate_mask[row, slot]
                & targets.hard_negative_mask[row]
            )
            if not bool(np.any(hard_members)):
                continue
            all_rows_by_mode[int(mode_id)] = all_rows_by_mode.get(int(mode_id), 0) + 1
            if has_positive:
                eligible_rows_by_mode[int(mode_id)] = eligible_rows_by_mode.get(int(mode_id), 0) + 1
    active = {
        mode: count
        for mode, count in eligible_rows_by_mode.items()
        if int(count) >= minimum
    }
    return {
        "hard_group_count": int(targets.group_hard_mask.sum()),
        "positive_row_count": int(np.any(targets.positive_mask, axis=1).sum()),
        "coherent_mode_count": int(len(all_rows_by_mode)),
        "coherent_mode_with_positive_count": int(len(eligible_rows_by_mode)),
        "coherent_mode_active_count": int(len(active)),
        "coherent_mode_active_fraction": float(len(active) / max(1, len(all_rows_by_mode))),
        "coherent_mode_rows_total": int(sum(all_rows_by_mode.values())),
        "coherent_mode_rows_with_positive": int(sum(eligible_rows_by_mode.values())),
        "coherent_min_mode_rows": int(minimum),
        "coherent_min_rows_observed": int(min(eligible_rows_by_mode.values(), default=0)),
        "coherent_max_rows_observed": int(max(eligible_rows_by_mode.values(), default=0)),
    }


def active_coherent_query_ids(
    *,
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    groups_by_id: dict[str, QueryGroup],
    minimum_mode_rows: int,
) -> list[str]:
    """Return queries with at least one trainable exact-P1 coherent mode."""

    active: list[str] = []
    for query_id in sorted(groups_by_id):
        rows = groups_by_id[query_id].rows
        audit = direct_target_coverage_audit(
            targets=targets.subset(rows), minimum_mode_rows=int(minimum_mode_rows)
        )
        if int(audit["coherent_mode_active_count"]) > 0:
            active.append(str(query_id))
    return active


def _freeze_non_descriptor_mapper_parameters(model: nn.Module) -> dict[str, int]:
    """Fine-tune only feature fusion and the descriptor adapter path.

    Selector, dustbin, fine, and RGB measurement heads are not direct P1
    identity evidence.  Leaving them frozen keeps this experiment scoped and
    prevents the checkpoint from silently changing unrelated inference heads.
    """

    trainable_prefixes = ("feature_fusion.", "adapter.")
    counts = {"trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        enabled = str(name).startswith(trainable_prefixes)
        parameter.requires_grad_(enabled)
        counts["trainable" if enabled else "frozen"] += int(parameter.numel())
    if counts["trainable"] <= 0:
        raise RuntimeError("direct P1 mapper has no trainable descriptor parameters")
    return counts


def _load_feature_batch(
    groups: Sequence[QueryGroup],
    *,
    cache: _FeatureMapCache,
    device: torch.device,
) -> torch.Tensor:
    arrays = [cache.load(group.token_path) for group in groups]
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        raise ValueError("direct P1 mapper batches require equal raw feature map shapes")
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=True)


def _build_batch_runtime(
    *,
    groups: Sequence[QueryGroup],
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    device: torch.device,
) -> tuple[CurrentP1MapperRuntime, CurrentP1MapperDirectTargets, torch.Tensor, torch.Tensor]:
    if not groups:
        raise ValueError("direct P1 mapper batch has no query groups")
    point_counts = {int(len(group.rows)) for group in groups}
    if len(point_counts) != 1:
        raise ValueError("direct P1 mapper batches require equal P1 point counts per query")
    rows = np.concatenate([group.rows for group in groups], axis=0).astype(np.int64, copy=False)
    batch_runtime = runtime.subset(rows)
    batch_targets = targets.subset(rows)
    points_per_query = next(iter(point_counts))
    xy = torch.from_numpy(batch_runtime.xy.reshape(len(groups), points_per_query, 2)).to(
        device=device, non_blocking=True
    )
    image_sizes = torch.tensor(
        [group.image_size for group in groups], dtype=torch.float32, device=device
    )
    return batch_runtime, batch_targets, xy, image_sizes


def _candidate_descriptors_for_runtime(
    *,
    runtime: CurrentP1MapperRuntime,
    frozen_bank: torch.Tensor,
) -> torch.Tensor:
    rows = torch.from_numpy(runtime.candidate_bank_rows).to(
        device=frozen_bank.device, dtype=torch.long, non_blocking=True
    )
    if torch.any(rows < 0) or torch.any(rows >= frozen_bank.shape[0]):
        raise ValueError("direct P1 runtime contains an invalid frozen bank row")
    return frozen_bank[rows]


def _forward_direct_loss(
    *,
    model: nn.Module,
    teacher_mapper: nn.Module | None,
    groups: Sequence[QueryGroup],
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    frozen_bank: torch.Tensor,
    cache: _FeatureMapCache,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> _TrainingStepLoss:
    batch_runtime, batch_targets, xy, image_sizes = _build_batch_runtime(
        groups=groups,
        runtime=runtime,
        targets=targets,
        device=device,
    )
    feature_maps = _load_feature_batch(groups, cache=cache, device=device)
    context = (
        torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
        if amp_enabled
        else nullcontext()
    )
    with context:
        descriptor_maps = model(feature_maps)
        sampled = sample_mapper_descriptors_at_p1_xy(descriptor_maps, xy, image_sizes)
    descriptors = sampled.reshape(-1, sampled.shape[-1])
    candidate_descriptors = _candidate_descriptors_for_runtime(
        runtime=batch_runtime,
        frozen_bank=frozen_bank,
    )
    direct = current_p1_mapper_direct_loss(
        query_descriptors=descriptors,
        candidate_descriptors=candidate_descriptors,
        runtime=batch_runtime,
        targets=batch_targets,
        temperature=float(args.temperature),
        coherent_margin=float(args.coherent_margin),
        coherent_margin_weight=float(args.coherent_margin_weight),
        coherent_min_mode_rows=int(args.coherent_min_mode_rows),
    )
    anchor_weight = float(args.anchor_distillation_weight)
    if anchor_weight > 0.0:
        if teacher_mapper is None:
            raise RuntimeError("anchor distillation requires a frozen mapper teacher")
        teacher_context = (
            torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
            if amp_enabled
            else nullcontext()
        )
        with torch.no_grad():
            with teacher_context:
                teacher_descriptors, _teacher_heatmap, _teacher_offsets = teacher_mapper.forward_feature_map(
                    feature_maps
                )
        anchor_loss, anchor_metrics = dense_descriptor_anchor_distillation_loss(
            student_descriptor_maps=descriptor_maps,
            teacher_descriptor_maps=teacher_descriptors,
            p1_xy=xy,
            image_sizes=image_sizes,
            exclusion_radius_tokens=int(args.anchor_exclusion_radius_tokens),
        )
    else:
        anchor_loss = direct.total_loss.new_zeros(())
        anchor_metrics = {
            "anchor_distillation_loss": 0.0,
            "anchor_descriptor_cosine_mean": 1.0,
            "anchor_token_count": 0.0,
            "anchor_excluded_token_count": 0.0,
        }
    total = direct.total_loss + anchor_weight * anchor_loss
    metrics = dict(direct.metrics)
    metrics.update(anchor_metrics)
    metrics["mapper_total_loss"] = float(total.detach().cpu().item())
    return _TrainingStepLoss(
        total_loss=total,
        direct_loss=direct.total_loss,
        anchor_loss=anchor_loss,
        metrics=metrics,
    )


def _reduce_scalar(value: float, state: DistributedState) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=state.device)
    if state.enabled:
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        tensor /= float(state.world_size)
    return float(tensor.item())


def _evaluate_groups(
    *,
    model: nn.Module,
    teacher_mapper: nn.Module | None,
    groups: Sequence[QueryGroup],
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    frozen_bank: torch.Tensor,
    cache: _FeatureMapCache,
    state: DistributedState,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, float]:
    """Evaluate query-disjoint train-only groups without optimizer updates."""

    if not groups:
        return {"query_count": 0.0}
    was_training = model.training
    model.eval()
    # Evaluation is no-grad and its metrics are reduced explicitly below.  Do
    # not call a DDP wrapper when a query-grouped shard is empty on one rank:
    # DDP forward collectives then become asymmetric even though no backward is
    # required.  The underlying mapper remains shared with the training model.
    evaluation_model = _FullMapDescriptorForward(_mapper_from_wrapper(model)).eval()
    local_groups = list(groups)[int(state.rank) :: int(state.world_size)]
    sums: dict[str, float] = {key: 0.0 for key in _EVALUATION_METRIC_KEYS}
    count = 0
    with torch.no_grad():
        for start in range(0, len(local_groups), int(args.batch_query_images)):
            batch = local_groups[start : start + int(args.batch_query_images)]
            result = _forward_direct_loss(
                model=evaluation_model,
                teacher_mapper=teacher_mapper,
                groups=batch,
                runtime=runtime,
                targets=targets,
                frozen_bank=frozen_bank,
                cache=cache,
                device=state.device,
                args=args,
                amp_enabled=amp_enabled,
            )
            weight = float(len(batch))
            for key in _EVALUATION_METRIC_KEYS:
                sums[key] += float(result.metrics[key]) * weight
            count += len(batch)
    keys = _EVALUATION_METRIC_KEYS
    packed = torch.tensor(
        [float(count)] + [sums[key] for key in keys], dtype=torch.float64, device=state.device
    )
    if state.enabled:
        distributed.all_reduce(packed, op=distributed.ReduceOp.SUM)
    total_count = int(round(float(packed[0].item())))
    result = {"query_count": float(total_count)}
    if total_count:
        result.update(
            {
                key: float(packed[index + 1].item() / float(total_count))
                for index, key in enumerate(keys)
            }
        )
    if was_training:
        model.train()
    return result


def _rank_owned_groups(
    groups: Sequence[QueryGroup], *, state: DistributedState
) -> list[QueryGroup]:
    """Return a stable query-disjoint shard that can remain in one rank's LRU."""

    if not groups:
        raise ValueError("direct P1 mapper has no training query groups")
    # Keep an image on one rank for the entire run.  Repartitioning after each
    # epoch makes the per-rank feature LRU repeatedly reload the same 63 train
    # images and turns otherwise inexpensive mapper training into storage I/O.
    # The initial deterministic shard remains query-disjoint across ranks; only
    # its local order changes from epoch to epoch.
    ordered = sorted(list(groups), key=lambda item: str(item.query_id))
    local_groups = ordered[int(state.rank) :: int(state.world_size)]
    if not local_groups:
        raise RuntimeError("direct P1 mapper rank received no train query groups")
    return local_groups


def _local_group_stream(groups: Sequence[QueryGroup], *, seed: int) -> Iterable[QueryGroup]:
    """Yield a pre-sharded query set in a new deterministic local order each epoch."""

    if not groups:
        raise ValueError("direct P1 mapper local group stream is empty")
    epoch = 0
    while True:
        generator = random.Random(int(seed) + epoch * 1_000_003)
        shuffled = list(groups)
        generator.shuffle(shuffled)
        for value in shuffled:
            yield value
        epoch += 1


def _batched_training_groups(
    groups: Sequence[QueryGroup],
    coherent_groups: Sequence[QueryGroup],
    *,
    state: DistributedState,
    seed: int,
    batch_size: int,
    coherent_per_batch: int,
) -> Iterable[list[QueryGroup]]:
    """Cycle per-rank batches while guaranteeing exact coherent-mode supervision."""

    if not coherent_groups:
        raise ValueError("direct P1 mapper has no coherent query group for batch sampling")
    local_generic_groups = _rank_owned_groups(groups, state=state)
    coherent_ids = {str(group.query_id) for group in coherent_groups}
    local_coherent_groups = [
        group for group in local_generic_groups if str(group.query_id) in coherent_ids
    ]
    if not local_coherent_groups:
        raise RuntimeError("direct P1 mapper rank has no coherent query in its stable shard")
    generic_source = _local_group_stream(local_generic_groups, seed=seed)
    coherent_source = _local_group_stream(local_coherent_groups, seed=int(seed) + 31_337)
    while True:
        batch: list[QueryGroup] = []
        seen_ids: set[str] = set()
        coherent_attempts = 0
        while len(batch) < int(coherent_per_batch):
            candidate = next(coherent_source)
            coherent_attempts += 1
            if candidate.query_id in seen_ids and coherent_attempts <= max(8, len(coherent_groups) * 2):
                continue
            batch.append(candidate)
            seen_ids.add(candidate.query_id)
        attempts = 0
        while len(batch) < int(batch_size):
            candidate = next(generic_source)
            attempts += 1
            if candidate.query_id in seen_ids and attempts <= max(8, len(groups) * 2):
                continue
            batch.append(candidate)
            seen_ids.add(candidate.query_id)
        if len(batch) != int(batch_size):
            raise RuntimeError("direct P1 mapper batch construction is incomplete")
        yield batch


def _save_summary(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _mapper_from_wrapper(wrapper: nn.Module) -> nn.Module:
    if isinstance(wrapper, DistributedDataParallel):
        return wrapper.module.mapper
    return wrapper.mapper


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(name): value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _selection_value(metrics: Mapping[str, float]) -> float:
    value = float(metrics.get("mapper_total_loss", float("inf")))
    return value if math.isfinite(value) else float("inf")


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    state = _init_distributed()
    try:
        _seed_everything(int(args.seed), state)
        if bool(args.allow_tf32) and state.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        amp_enabled = bool(state.device.type == "cuda" and not bool(args.no_amp))
        runtime, targets, bank_features, supervision_summary = load_current_p1_mapper_direct_supervision(
            proposals_path=Path(args.p1_proposals),
            targets_path=Path(args.p1_targets),
            projected_landmark_bank_path=Path(args.projected_landmark_bank),
        )
        init_checkpoint = Path(args.init_checkpoint)
        init_hash = file_sha256_short(init_checkpoint)
        expected_hash = str(supervision_summary.get("bank_mapper_checkpoint_sha256", ""))
        if expected_hash and expected_hash != init_hash:
            raise ValueError(
                "direct P1 mapper initializer must exactly match the frozen projected-bank mapper: "
                f"expected={expected_hash!r}, actual={init_hash!r}"
            )
        groups_by_id = _load_query_groups(
            runtime=runtime,
            targets=targets,
            token_manifest=Path(args.token_manifest),
            image_root=Path(args.image_root),
        )
        train_ids, validation_ids = split_train_only_inner_validation(
            groups_by_id,
            fraction=float(args.inner_validation_fraction),
            seed=int(args.seed),
        )
        train_groups = [groups_by_id[query_id] for query_id in train_ids]
        validation_groups = [groups_by_id[query_id] for query_id in validation_ids]
        coherent_query_ids = active_coherent_query_ids(
            runtime=runtime,
            targets=targets,
            groups_by_id=groups_by_id,
            minimum_mode_rows=int(args.coherent_min_mode_rows),
        )
        coherent_train_groups = [
            groups_by_id[query_id]
            for query_id in train_ids
            if query_id in set(coherent_query_ids)
        ]
        if not coherent_train_groups:
            raise RuntimeError("train-only direct P1 split has no active coherent query group")
        limit = int(args.evaluation_query_limit)
        evaluation_groups = validation_groups if limit <= 0 else validation_groups[:limit]
        validate_distributed_evaluation_cardinality(
            query_count=len(evaluation_groups),
            state=state,
        )
        coverage = direct_target_coverage_audit(
            targets=targets,
            minimum_mode_rows=int(args.coherent_min_mode_rows),
        )
        if int(coverage["coherent_mode_active_count"]) <= 0:
            raise RuntimeError("direct P1 mapper has no active coherent mode after exact-token alignment")

        loaded = load_matcha_joint_model(init_checkpoint, device="cpu")
        inherited_descriptor_source = loaded.summary.get("descriptor_source_config")
        inherited_prototype_builder = loaded.summary.get("track_prototype_builder")
        if not isinstance(inherited_descriptor_source, dict):
            raise ValueError(
                "direct P1 mapper initializer lacks descriptor_source_config; "
                "refusing to produce a bank-ambiguous checkpoint"
            )
        if not isinstance(inherited_prototype_builder, dict):
            raise ValueError(
                "direct P1 mapper initializer lacks track_prototype_builder; "
                "refusing to produce a bank-ambiguous checkpoint"
            )
        mapper = loaded.model
        teacher_mapper: nn.Module | None = None
        if float(args.anchor_distillation_weight) > 0.0:
            teacher_mapper = copy.deepcopy(mapper).to(state.device).eval()
            for parameter in teacher_mapper.parameters():
                parameter.requires_grad_(False)
        trainable_counts = _freeze_non_descriptor_mapper_parameters(mapper)
        mapper = mapper.to(state.device)
        wrapper: nn.Module = _FullMapDescriptorForward(mapper)
        if state.enabled:
            wrapper = DistributedDataParallel(
                wrapper,
                device_ids=[int(state.local_rank)],
                output_device=int(state.local_rank),
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        trainable_parameters = [parameter for parameter in wrapper.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        frozen_bank = torch.from_numpy(np.asarray(bank_features, dtype=np.float32)).to(
            device=state.device, non_blocking=True
        )
        frozen_bank = torch.nn.functional.normalize(frozen_bank, p=2, dim=1)
        cache = _FeatureMapCache(
            feature_key=str(args.feature_key), max_images=int(args.feature_cache_images)
        )

        if state.is_primary:
            print(
                json.dumps(
                    {
                        "stage": "direct_p1_mapper_setup",
                        "world_size": int(state.world_size),
                        "device": str(state.device),
                        "train_query_count": len(train_groups),
                        "coherent_train_query_count": len(coherent_train_groups),
                        "inner_validation_query_count": len(validation_groups),
                        "evaluation_query_count": len(evaluation_groups),
                        "direct_target_coverage": coverage,
                        "trainable_parameter_count": trainable_counts["trainable"],
                        "anchor_distillation_enabled": teacher_mapper is not None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        baseline_validation = _evaluate_groups(
            model=wrapper,
            teacher_mapper=teacher_mapper,
            groups=evaluation_groups,
            runtime=runtime,
            targets=targets,
            frozen_bank=frozen_bank,
            cache=cache,
            state=state,
            args=args,
            amp_enabled=amp_enabled,
        )
        base_mapper = _mapper_from_wrapper(wrapper)
        best_validation = dict(baseline_validation)
        best_validation_step = 0
        best_selection_value = _selection_value(best_validation)
        best_state = _cpu_state_dict(base_mapper) if evaluation_groups else None
        validation_history: list[dict[str, float]] = [
            {"step": 0.0, **{key: float(value) for key, value in baseline_validation.items()}}
        ]

        started = time.monotonic()
        history: list[dict[str, float]] = []
        batch_source = _batched_training_groups(
            train_groups,
            coherent_train_groups,
            state=state,
            seed=int(args.seed),
            batch_size=int(args.batch_query_images),
            coherent_per_batch=int(args.coherent_queries_per_batch),
        )
        wrapper.train()
        for step in range(1, int(args.steps) + 1):
            groups = next(batch_source)
            optimizer.zero_grad(set_to_none=True)
            result = _forward_direct_loss(
                model=wrapper,
                teacher_mapper=teacher_mapper,
                groups=groups,
                runtime=runtime,
                targets=targets,
                frozen_bank=frozen_bank,
                cache=cache,
                device=state.device,
                args=args,
                amp_enabled=amp_enabled,
            )
            scaler.scale(result.total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(args.gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
                values = {
                    "step": float(step),
                    **{
                        key: _reduce_scalar(value, state)
                        for key, value in result.metrics.items()
                        if key != "format"
                    },
                }
                history.append(values)
                if state.is_primary:
                    print(json.dumps({"stage": "direct_p1_mapper_train", **values}, sort_keys=True), flush=True)
            if (
                evaluation_groups
                and int(args.validation_interval) > 0
                and step % int(args.validation_interval) == 0
                and step < int(args.steps)
            ):
                validation = _evaluate_groups(
                    model=wrapper,
                    teacher_mapper=teacher_mapper,
                    groups=evaluation_groups,
                    runtime=runtime,
                    targets=targets,
                    frozen_bank=frozen_bank,
                    cache=cache,
                    state=state,
                    args=args,
                    amp_enabled=amp_enabled,
                )
                validation_record = {
                    "step": float(step),
                    **{key: float(value) for key, value in validation.items()},
                }
                validation_history.append(validation_record)
                selection = _selection_value(validation)
                if selection < best_selection_value:
                    best_selection_value = selection
                    best_validation = dict(validation)
                    best_validation_step = int(step)
                    best_state = _cpu_state_dict(_mapper_from_wrapper(wrapper))
                if state.is_primary:
                    print(
                        json.dumps(
                            {
                                "stage": "direct_p1_mapper_validation",
                                "selection_value": selection,
                                "best_step": int(best_validation_step),
                                **validation_record,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        last_validation = _evaluate_groups(
            model=wrapper,
            teacher_mapper=teacher_mapper,
            groups=evaluation_groups,
            runtime=runtime,
            targets=targets,
            frozen_bank=frozen_bank,
            cache=cache,
            state=state,
            args=args,
            amp_enabled=amp_enabled,
        )
        validation_history.append(
            {"step": float(args.steps), **{key: float(value) for key, value in last_validation.items()}}
        )
        last_selection_value = _selection_value(last_validation)
        if last_selection_value < best_selection_value:
            best_selection_value = last_selection_value
            best_validation = dict(last_validation)
            best_validation_step = int(args.steps)
            best_state = _cpu_state_dict(_mapper_from_wrapper(wrapper))
        if best_state is not None:
            _mapper_from_wrapper(wrapper).load_state_dict(best_state, strict=True)
        selected_validation = dict(best_validation)
        elapsed = float(time.monotonic() - started)
        summary: dict[str, object] = {
            "format": DIRECT_P1_MAPPER_TRAINER_FORMAT,
            "config": vars(args),
            "distributed_runtime": {
                "enabled": bool(state.enabled),
                "rank": int(state.rank),
                "world_size": int(state.world_size),
                "local_rank": int(state.local_rank),
                "device": str(state.device),
                "amp_enabled": bool(amp_enabled),
            },
            "supervision": supervision_summary,
            "direct_target_coverage": coverage,
            "initializer": {
                "path": str(init_checkpoint),
                "sha256": init_hash,
                "bank_checkpoint_match": True,
                "missing_state_keys": list(loaded.summary.get("missing_state_keys", [])),
                "unexpected_state_keys": list(loaded.summary.get("unexpected_state_keys", [])),
            },
            # These fields are consumed by projected-observation bank building.
            # Direct P1 only changes the mapper descriptor path; it must not
            # silently erase the source/preprocessing or aggregation contract.
            "descriptor_source_config": dict(inherited_descriptor_source),
            "track_prototype_builder": dict(inherited_prototype_builder),
            "inherited_descriptor_contract": {
                "source_checkpoint": str(init_checkpoint),
                "source_checkpoint_sha256": init_hash,
                "descriptor_source_config": dict(inherited_descriptor_source),
                "track_prototype_builder": dict(inherited_prototype_builder),
            },
            "query_split": {
                "source": "train_only_deterministic_query_grouped_inner_validation_v1",
                "train_query_count": len(train_groups),
                "coherent_train_query_count": len(coherent_train_groups),
                "inner_validation_query_count": len(validation_groups),
                "evaluation_query_count": len(evaluation_groups),
                "evaluation_limited": bool(limit > 0),
            },
            "trainable_parameters": trainable_counts,
            "anchor_distillation": {
                "enabled": teacher_mapper is not None,
                "weight": float(args.anchor_distillation_weight),
                "exclusion_radius_tokens": int(args.anchor_exclusion_radius_tokens),
                "teacher_checkpoint_sha256": init_hash if teacher_mapper is not None else "",
                "teacher_runtime_available": False,
                "loss": "dense_non_p1_descriptor_cosine_v1",
            },
            "baseline_inner_validation": baseline_validation,
            "last_inner_validation": last_validation,
            "best_inner_validation": selected_validation,
            "final_inner_validation": selected_validation,
            "best_inner_validation_step": int(best_validation_step),
            "best_inner_validation_selection_value": float(best_selection_value),
            "training_history": history,
            "validation_history": validation_history,
            "elapsed_sec": elapsed,
            "feature_cache": cache.summary(),
            "runtime_ready": False,
            "required_next_step": "rebuild_projected_observation_bank_with_output_checkpoint_before_retrieval_or_pose_evaluation",
        }
        if state.enabled:
            distributed.barrier()
        output_dir = Path(args.output_dir)
        if state.is_primary:
            output_dir.mkdir(parents=True, exist_ok=True)
            run = MatchaJointTrainingRun(model=_mapper_from_wrapper(wrapper), summary=dict(summary))
            checkpoint_path = output_dir / "joint.pt"
            save_matcha_joint_model(run, checkpoint_path)
            summary["output_checkpoint"] = str(checkpoint_path)
            summary["output_checkpoint_sha256"] = file_sha256_short(checkpoint_path)
            _save_summary(output_dir / "summary.json", summary)
            print(
                json.dumps(
                    {
                        "stage": "direct_p1_mapper_complete",
                        "output_checkpoint": str(checkpoint_path),
                        "required_next_step": summary["required_next_step"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if state.enabled:
            distributed.barrier()
        return summary
    finally:
        _finish_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
