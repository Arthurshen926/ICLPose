"""Supervised linear selector training for patch-to-3D VFM matching."""

from __future__ import annotations

import random
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.feature_compression import FeatureCompressionTransform
from feature_extract.vfm.patch_to_3d_matching import PatchPositiveSets
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


_EPS = 1e-8
_SAMPLE_CACHE_FORMAT = "vfm_patch_selector_training_set_v1"


@dataclass(frozen=True)
class PatchSelectorSampleConfig:
    max_tokens_per_query: int = 256
    max_positives_per_token: int = 4
    hard_negatives_per_token: int = 32
    hard_negative_pool: int = 256
    query_token_step: int = 1
    min_positive_count: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens_per_query <= 0:
            raise ValueError("max_tokens_per_query must be positive")
        if self.max_positives_per_token <= 0:
            raise ValueError("max_positives_per_token must be positive")
        if self.hard_negatives_per_token <= 0:
            raise ValueError("hard_negatives_per_token must be positive")
        if self.hard_negative_pool <= 0:
            raise ValueError("hard_negative_pool must be positive")
        if self.query_token_step <= 0:
            raise ValueError("query_token_step must be positive")
        if self.min_positive_count <= 0:
            raise ValueError("min_positive_count must be positive")


@dataclass(frozen=True)
class PatchSelectorTrainingSet:
    query_features: np.ndarray
    positive_features: np.ndarray
    positive_mask: np.ndarray
    negative_features: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        query = np.asarray(self.query_features, dtype=np.float32)
        positives = np.asarray(self.positive_features, dtype=np.float32)
        mask = np.asarray(self.positive_mask, dtype=bool)
        negatives = np.asarray(self.negative_features, dtype=np.float32)
        if query.ndim != 2:
            raise ValueError("query_features must have shape (N, C)")
        if positives.ndim != 3:
            raise ValueError("positive_features must have shape (N, P, C)")
        if negatives.ndim != 3:
            raise ValueError("negative_features must have shape (N, K, C)")
        if positives.shape[0] != query.shape[0] or negatives.shape[0] != query.shape[0]:
            raise ValueError("query, positive and negative sample counts must match")
        if positives.shape[2] != query.shape[1] or negatives.shape[2] != query.shape[1]:
            raise ValueError("query, positive and negative feature dimensions must match")
        if mask.shape != positives.shape[:2]:
            raise ValueError("positive_mask must have shape (N, P)")
        if query.shape[0] and not np.all(mask.any(axis=1)):
            raise ValueError("each sample must have at least one positive")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "positive_features", positives)
        object.__setattr__(self, "positive_mask", mask)
        object.__setattr__(self, "negative_features", negatives)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def sample_count(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.query_features.shape[1]) if self.query_features.ndim == 2 else 0


@dataclass(frozen=True)
class PatchSelectorTrainingConfig:
    output_dim: int = 128
    steps: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    temperature: float = 0.07
    seed: int = 0
    device: str = "cpu"
    eval_split_fraction: float = 0.1
    center_inputs: bool = False
    group_size: int = 0
    group_lasso_weight: float = 0.0
    hard_gate_keep_fraction: float = 1.0
    hard_gate_min_groups: int = 1

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")
        if int(self.group_size) < 0:
            raise ValueError("group_size must be non-negative")
        if float(self.group_lasso_weight) < 0.0:
            raise ValueError("group_lasso_weight must be non-negative")
        if not 0.0 < float(self.hard_gate_keep_fraction) <= 1.0:
            raise ValueError("hard_gate_keep_fraction must be in (0, 1]")
        if int(self.hard_gate_min_groups) <= 0:
            raise ValueError("hard_gate_min_groups must be positive")


@dataclass(frozen=True)
class PatchSelectorTrainingSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    input_dim: int
    output_dim: int
    steps: int
    batch_size: int
    group_size: int
    group_lasso_weight: float
    hard_gate_keep_fraction: float
    group_count: int
    active_group_count: int
    active_channel_count: int
    top_group_energy_fraction: float
    gated_train_top1_acc: float
    gated_eval_top1_acc: float


@dataclass(frozen=True)
class PatchSelectorTrainingRun:
    transform: FeatureCompressionTransform
    summary: PatchSelectorTrainingSummary


class LinearPatchSelector(nn.Module):
    """Bias-free linear projection followed by L2 normalization."""

    def __init__(self, input_dim: int, output_dim: int, input_mean: np.ndarray | None = None) -> None:
        super().__init__()
        if output_dim > input_dim:
            raise ValueError("output_dim must be <= input_dim")
        self.projection = nn.Linear(int(input_dim), int(output_dim), bias=False)
        mean = np.zeros((input_dim,), dtype=np.float32) if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self.register_buffer("input_mean", torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32))
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        centered = features - self.input_mean.to(features.device)
        return F.normalize(self.projection(centered), dim=-1, eps=1e-8)


@dataclass(frozen=True)
class SafePatchSelectorTrainingConfig:
    output_dim: int = 128
    residual_hidden_dim: int = 256
    steps: int = 800
    batch_size: int = 512
    lr: float = 5e-4
    temperature: float = 0.07
    inlier_loss_weight: float = 0.2
    group_lasso_weight: float = 0.0
    seed: int = 0
    device: str = "cpu"
    eval_split_fraction: float = 0.1
    center_inputs: bool = False
    group_size: int = 64
    hard_gate_keep_fraction: float = 1.0
    hard_gate_min_groups: int = 1

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if float(self.inlier_loss_weight) < 0.0:
            raise ValueError("inlier_loss_weight must be non-negative")
        if float(self.group_lasso_weight) < 0.0:
            raise ValueError("group_lasso_weight must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")
        if int(self.group_size) < 0:
            raise ValueError("group_size must be non-negative")
        if not 0.0 < float(self.hard_gate_keep_fraction) <= 1.0:
            raise ValueError("hard_gate_keep_fraction must be in (0, 1]")
        if int(self.hard_gate_min_groups) <= 0:
            raise ValueError("hard_gate_min_groups must be positive")


@dataclass(frozen=True)
class SafePatchSelectorTrainingSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    inlier_train_accuracy: float
    inlier_eval_accuracy: float
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    input_dim: int
    output_dim: int
    residual_hidden_dim: int
    steps: int
    batch_size: int
    group_size: int
    group_count: int
    active_group_count: int
    active_channel_count: int
    active_group_fraction: float
    gate_mean: float
    gate_min: float
    gate_max: float
    parameter_count: int
    inlier_loss_weight: float
    group_lasso_weight: float
    hard_gate_keep_fraction: float


@dataclass(frozen=True)
class SafePatchSelectorTrainingRun:
    model: "ResidualGatedPatchSelector"
    summary: SafePatchSelectorTrainingSummary
    active_group_mask: np.ndarray

    def encode_rows(self, rows: np.ndarray, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
        return encode_rows_with_safe_selector(
            self.model,
            rows,
            device=device,
            batch_size=batch_size,
            active_group_mask=self.active_group_mask,
        )


class ResidualGatedPatchSelector(nn.Module):
    """LayerNorm + group gate + linear projection + residual MLP descriptor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        residual_hidden_dim: int = 256,
        group_size: int = 64,
        input_mean: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if int(output_dim) > int(input_dim):
            raise ValueError("output_dim must be <= input_dim")
        if int(residual_hidden_dim) <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.group_size = int(group_size)
        group_count = len(_group_slices(self.input_dim, self.group_size))
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.residual_hidden_dim),
            nn.GELU(),
            nn.Linear(self.residual_hidden_dim, self.output_dim),
        )
        self.query_matchability = nn.Linear(self.output_dim, 1)
        self.landmark_reliability = nn.Linear(self.output_dim, 1)
        self.pairwise_inlier = nn.Sequential(
            nn.Linear(self.output_dim * 4, self.residual_hidden_dim),
            nn.GELU(),
            nn.Linear(self.residual_hidden_dim, 1),
        )
        self.group_logits = nn.Parameter(torch.full((group_count,), 2.0, dtype=torch.float32))
        mean = np.zeros((self.input_dim,), dtype=np.float32) if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self.register_buffer("input_mean", torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32))
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def group_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.group_logits)

    def _channel_gates(self, active_group_mask: torch.Tensor | None = None) -> torch.Tensor:
        gates = self.group_gates()
        if active_group_mask is not None:
            gates = gates * active_group_mask.to(gates.device, dtype=gates.dtype)
        chunks = []
        for group_idx, group in enumerate(_group_slices(self.input_dim, self.group_size)):
            chunks.append(gates[group_idx].expand(group.stop - group.start))
        return torch.cat(chunks, dim=0).reshape(1, -1)

    def forward(self, features: torch.Tensor, active_group_mask: torch.Tensor | None = None) -> torch.Tensor:
        centered = features - self.input_mean.to(features.device)
        normalized = self.input_norm(centered)
        gated = normalized * self._channel_gates(active_group_mask).to(features.device)
        projected = self.projection(gated)
        residual = self.residual(projected)
        return F.normalize(projected + residual, dim=-1, eps=1e-8)

    def pairwise_inlier_logit(self, query_z: torch.Tensor, landmark_z: torch.Tensor) -> torch.Tensor:
        if query_z.shape != landmark_z.shape:
            raise ValueError("query_z and landmark_z must have the same shape")
        pair = torch.cat([query_z, landmark_z, torch.abs(query_z - landmark_z), query_z * landmark_z], dim=-1)
        logits = self.pairwise_inlier(pair)
        logits = logits + self.query_matchability(query_z) + self.landmark_reliability(landmark_z)
        return logits.squeeze(-1)


def _token_rows(feature_map: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("query feature map must have shape (C, H, W)")
    channels, height, width = values.shape
    features = []
    token_indices = []
    for y_idx in range(0, height, int(step)):
        for x_idx in range(0, width, int(step)):
            features.append(values[:, y_idx, x_idx])
            token_indices.append(y_idx * width + x_idx)
    if not features:
        return np.zeros((0, channels), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(features, axis=0).astype(np.float32), np.asarray(token_indices, dtype=np.int64)


def _empty_training_set(input_dim: int, config: PatchSelectorSampleConfig, metadata: Mapping[str, object]) -> PatchSelectorTrainingSet:
    return PatchSelectorTrainingSet(
        query_features=np.zeros((0, input_dim), dtype=np.float32),
        positive_features=np.zeros((0, config.max_positives_per_token, input_dim), dtype=np.float32),
        positive_mask=np.zeros((0, config.max_positives_per_token), dtype=bool),
        negative_features=np.zeros((0, config.hard_negatives_per_token, input_dim), dtype=np.float32),
        metadata=metadata,
    )


def _sample_positive_indices(
    positive_indices: list[int],
    max_count: int,
    rng: np.random.Generator,
) -> list[int]:
    ordered = sorted(int(idx) for idx in positive_indices)
    if len(ordered) <= max_count:
        return ordered
    selected = rng.choice(np.asarray(ordered, dtype=np.int64), size=int(max_count), replace=False)
    return sorted(int(idx) for idx in selected.tolist())


def _sample_negative_indices(
    query_feature: np.ndarray,
    landmark_features: np.ndarray,
    positive_row_indices: set[int],
    count: int,
    pool: int,
) -> list[int]:
    query_norm, query_valid = normalize_rows(np.asarray(query_feature, dtype=np.float32).reshape(1, -1))
    landmark_norm, valid_landmarks = normalize_rows(landmark_features)
    if not bool(query_valid[0]):
        candidates = np.flatnonzero(valid_landmarks)
    else:
        scores = (landmark_norm @ query_norm[0]).astype(np.float32)
        scores[~valid_landmarks] = -np.inf
        if positive_row_indices:
            scores[np.asarray(sorted(positive_row_indices), dtype=np.int64)] = -np.inf
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            return []
        take = min(int(pool), finite.size)
        if take == finite.size:
            order = finite[np.argsort(-scores[finite])]
        else:
            partial = np.argpartition(-scores[finite], kth=take - 1)[:take]
            order = finite[partial[np.argsort(-scores[finite][partial])]]
        candidates = order
    candidates = [int(idx) for idx in candidates.tolist() if int(idx) not in positive_row_indices]
    return candidates[: int(count)]


def _sample_negative_indices_batch(
    query_features: np.ndarray,
    landmark_features: np.ndarray,
    positive_row_indices: list[set[int]],
    count: int,
    pool: int,
) -> list[list[int]]:
    queries = np.asarray(query_features, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError("query_features must have shape (N, C)")
    if queries.shape[0] != len(positive_row_indices):
        raise ValueError("positive_row_indices must contain one set per query")
    query_norm, valid_query = normalize_rows(queries)
    landmark_norm, valid_landmarks = normalize_rows(landmark_features)
    scores = (query_norm @ landmark_norm.T).astype(np.float32, copy=False)
    scores[:, ~valid_landmarks] = -np.inf
    output: list[list[int]] = []
    for row_idx in range(scores.shape[0]):
        if not bool(valid_query[row_idx]):
            output.append([])
            continue
        row_scores = scores[row_idx].copy()
        positives = positive_row_indices[row_idx]
        if positives:
            row_scores[np.asarray(sorted(positives), dtype=np.int64)] = -np.inf
        finite = np.flatnonzero(np.isfinite(row_scores))
        if finite.size == 0:
            output.append([])
            continue
        take = min(int(pool), finite.size)
        if take == finite.size:
            order = finite[np.argsort(-row_scores[finite])]
        else:
            partial = np.argpartition(-row_scores[finite], kth=take - 1)[:take]
            order = finite[partial[np.argsort(-row_scores[finite][partial])]]
        output.append([int(idx) for idx in order[: int(count)].tolist()])
    return output


def build_patch_selector_samples_for_query(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    positives: PatchPositiveSets,
    config: PatchSelectorSampleConfig | None = None,
) -> PatchSelectorTrainingSet:
    config = config or PatchSelectorSampleConfig()
    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")
    if landmark_index.feature_dim != int(feature_map.shape[0]):
        raise ValueError("landmark feature dimension must match query feature dimension")
    rng = np.random.default_rng(int(config.seed))
    query_features, token_indices = _token_rows(feature_map, config.query_token_step)
    if query_features.size == 0 or len(landmark_index) == 0:
        return _empty_training_set(int(feature_map.shape[0]), config, {"sample_count": 0})

    row_by_track = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    token_candidates: list[tuple[int, np.ndarray, list[int], list[int]]] = []
    for row_idx, token_index in enumerate(token_indices.tolist()):
        if float(np.linalg.norm(query_features[row_idx])) <= _EPS:
            continue
        positive = positives.by_token.get(int(token_index))
        if positive is None or positive.count < int(config.min_positive_count):
            continue
        all_positive_indices = [
            row_by_track[int(track_id)]
            for track_id in positive.track_ids
            if int(track_id) in row_by_track
        ]
        if len(all_positive_indices) < int(config.min_positive_count):
            continue
        positive_indices = _sample_positive_indices(all_positive_indices, config.max_positives_per_token, rng)
        token_candidates.append((int(token_index), query_features[row_idx], all_positive_indices, positive_indices))

    if len(token_candidates) > int(config.max_tokens_per_query):
        selected = rng.choice(len(token_candidates), size=int(config.max_tokens_per_query), replace=False)
        token_candidates = [token_candidates[int(idx)] for idx in sorted(selected.tolist())]

    if not token_candidates:
        return _empty_training_set(
            int(feature_map.shape[0]),
            config,
            {
                "sample_count": 0,
                "candidate_token_count": 0,
                "raw_false_nearest_negative_count": 0,
            },
        )

    selected_query_features = np.stack([item[1] for item in token_candidates], axis=0).astype(np.float32)
    all_positive_sets = [set(item[2]) for item in token_candidates]
    negative_lists = _sample_negative_indices_batch(
        selected_query_features,
        landmark_index.features,
        positive_row_indices=all_positive_sets,
        count=config.hard_negatives_per_token,
        pool=config.hard_negative_pool,
    )

    candidate_rows: list[tuple[int, np.ndarray, list[int], list[int]]] = []
    for (token_index, query_feature, _all_positive_indices, positive_indices), negative_indices in zip(
        token_candidates,
        negative_lists,
    ):
        if len(negative_indices) < int(config.hard_negatives_per_token):
            continue
        candidate_rows.append((int(token_index), query_feature, positive_indices, negative_indices))

    if not candidate_rows:
        return _empty_training_set(
            int(feature_map.shape[0]),
            config,
            {
                "sample_count": 0,
                "candidate_token_count": len(token_candidates),
                "raw_false_nearest_negative_count": 0,
            },
        )

    queries = []
    pos = []
    pos_mask = []
    neg = []
    for _token_index, query_feature, positive_indices, negative_indices in candidate_rows:
        positive_array = np.zeros((config.max_positives_per_token, landmark_index.feature_dim), dtype=np.float32)
        mask_array = np.zeros((config.max_positives_per_token,), dtype=bool)
        for dst, landmark_idx in enumerate(positive_indices[: config.max_positives_per_token]):
            positive_array[dst] = landmark_index.features[int(landmark_idx)]
            mask_array[dst] = True
        negative_array = landmark_index.features[np.asarray(negative_indices[: config.hard_negatives_per_token], dtype=np.int64)]
        queries.append(query_feature.astype(np.float32, copy=True))
        pos.append(positive_array)
        pos_mask.append(mask_array)
        neg.append(negative_array.astype(np.float32, copy=True))

    return PatchSelectorTrainingSet(
        query_features=np.stack(queries, axis=0),
        positive_features=np.stack(pos, axis=0),
        positive_mask=np.stack(pos_mask, axis=0),
        negative_features=np.stack(neg, axis=0),
        metadata={
            "sample_count": len(queries),
            "candidate_token_count": len(candidate_rows),
            "raw_false_nearest_negative_count": len(queries) * int(config.hard_negatives_per_token),
            "max_positives_per_token": int(config.max_positives_per_token),
            "hard_negatives_per_token": int(config.hard_negatives_per_token),
        },
    )


def merge_patch_selector_training_sets(
    sample_sets: list[PatchSelectorTrainingSet],
    max_samples: int = 0,
    seed: int = 0,
) -> PatchSelectorTrainingSet:
    present = [samples for samples in sample_sets if samples.sample_count > 0]
    if not present:
        input_dim = sample_sets[0].input_dim if sample_sets else 0
        return PatchSelectorTrainingSet(
            query_features=np.zeros((0, input_dim), dtype=np.float32),
            positive_features=np.zeros((0, 0, input_dim), dtype=np.float32),
            positive_mask=np.zeros((0, 0), dtype=bool),
            negative_features=np.zeros((0, 0, input_dim), dtype=np.float32),
            metadata={"sample_count": 0},
        )
    query = np.concatenate([samples.query_features for samples in present], axis=0)
    positive = np.concatenate([samples.positive_features for samples in present], axis=0)
    positive_mask = np.concatenate([samples.positive_mask for samples in present], axis=0)
    negative = np.concatenate([samples.negative_features for samples in present], axis=0)
    if max_samples > 0 and query.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(query.shape[0], size=int(max_samples), replace=False))
        query = query[indices]
        positive = positive[indices]
        positive_mask = positive_mask[indices]
        negative = negative[indices]
    return PatchSelectorTrainingSet(
        query_features=query,
        positive_features=positive,
        positive_mask=positive_mask,
        negative_features=negative,
        metadata={"sample_count": int(query.shape[0]), "source_set_count": len(present)},
    )


def save_patch_selector_training_set_npz(samples: PatchSelectorTrainingSet, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _SAMPLE_CACHE_FORMAT,
        "sample_count": int(samples.sample_count),
        "input_dim": int(samples.input_dim),
        "positive_count": int(samples.positive_features.shape[1]),
        "negative_count": int(samples.negative_features.shape[1]),
        "metadata": dict(samples.metadata or {}),
    }
    np.savez_compressed(
        output,
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
        query_features=samples.query_features.astype(np.float32, copy=False),
        positive_features=samples.positive_features.astype(np.float32, copy=False),
        positive_mask=samples.positive_mask.astype(bool, copy=False),
        negative_features=samples.negative_features.astype(np.float32, copy=False),
    )


def load_patch_selector_training_set_npz(path: Path) -> tuple[PatchSelectorTrainingSet, dict[str, object]]:
    cache_path = Path(path)
    with np.load(cache_path) as data:
        if "metadata" not in data:
            raise ValueError(f"sample cache {cache_path} is missing metadata")
        payload = json.loads(str(data["metadata"].item()))
        if payload.get("format") != _SAMPLE_CACHE_FORMAT:
            raise ValueError(f"unsupported patch selector sample cache format in {cache_path}")
        samples = PatchSelectorTrainingSet(
            query_features=np.asarray(data["query_features"], dtype=np.float32),
            positive_features=np.asarray(data["positive_features"], dtype=np.float32),
            positive_mask=np.asarray(data["positive_mask"], dtype=bool),
            negative_features=np.asarray(data["negative_features"], dtype=np.float32),
            metadata=dict(payload.get("metadata", {})),
        )
    return samples, dict(payload.get("metadata", {}))


def _split_indices(count: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    eval_count = int(round(count * float(eval_fraction)))
    if eval_fraction > 0.0 and count > 1:
        eval_count = max(1, eval_count)
    eval_count = min(eval_count, max(count - 1, 0))
    return indices[eval_count:], indices[:eval_count]


def _subset_samples(samples: PatchSelectorTrainingSet, indices: np.ndarray) -> PatchSelectorTrainingSet:
    return PatchSelectorTrainingSet(
        query_features=samples.query_features[indices],
        positive_features=samples.positive_features[indices],
        positive_mask=samples.positive_mask[indices],
        negative_features=samples.negative_features[indices],
        metadata=samples.metadata,
    )


def _selector_loss(
    selector: LinearPatchSelector,
    query: torch.Tensor,
    positives: torch.Tensor,
    positive_mask: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_z = selector(query)
    positive_z = selector(positives.reshape(-1, positives.shape[-1])).reshape(positives.shape[0], positives.shape[1], -1)
    negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
    pos_logits = torch.einsum("bd,bpd->bp", query_z, positive_z) / float(temperature)
    neg_logits = torch.einsum("bd,bkd->bk", query_z, negative_z) / float(temperature)
    pos_logits = pos_logits.masked_fill(~positive_mask, -1.0e9)
    numerator = torch.logsumexp(pos_logits, dim=1)
    denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
    return torch.mean(denominator - numerator)


def _group_slices(input_dim: int, group_size: int) -> list[slice]:
    if int(group_size) <= 0:
        return [slice(0, int(input_dim))]
    return [slice(start, min(start + int(group_size), int(input_dim))) for start in range(0, int(input_dim), int(group_size))]


def _projection_group_energy(matrix: np.ndarray, group_size: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("matrix must have shape (input_dim, output_dim)")
    return np.asarray([float(np.sum(np.square(values[group, :]))) for group in _group_slices(values.shape[0], int(group_size))], dtype=np.float32)


def _group_lasso_penalty(selector: LinearPatchSelector, group_size: int) -> torch.Tensor:
    if int(group_size) <= 0:
        return torch.zeros((), device=selector.projection.weight.device)
    weights = selector.projection.weight.T
    penalties = []
    for group in _group_slices(int(weights.shape[0]), int(group_size)):
        penalties.append(torch.sqrt(torch.sum(weights[group, :] ** 2) + 1e-8))
    if not penalties:
        return torch.zeros((), device=weights.device)
    return torch.stack(penalties).mean()


def _apply_hard_group_gate(
    matrix: np.ndarray,
    group_size: int,
    keep_fraction: float,
    min_groups: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    values = np.asarray(matrix, dtype=np.float32).copy()
    energy = _projection_group_energy(values, int(group_size))
    group_count = int(energy.shape[0])
    if group_count == 0:
        return values, np.zeros((values.shape[0],), dtype=np.float32), {
            "group_count": 0,
            "active_group_count": 0,
            "active_channel_count": 0,
            "top_group_energy_fraction": 0.0,
        }
    keep_count = int(np.ceil(float(keep_fraction) * group_count))
    keep_count = max(int(min_groups), keep_count)
    keep_count = min(keep_count, group_count)
    order = np.lexsort((np.arange(group_count), -energy))
    keep_groups = set(int(idx) for idx in order[:keep_count].tolist())
    active_channels = 0
    for group_idx, group in enumerate(_group_slices(values.shape[0], int(group_size))):
        if group_idx in keep_groups:
            active_channels += int(group.stop - group.start)
        else:
            values[group, :] = 0.0
    gated_energy = _projection_group_energy(values, int(group_size))
    total_energy = float(np.sum(energy))
    channel_scores = np.zeros((values.shape[0],), dtype=np.float32)
    for group_idx, group in enumerate(_group_slices(values.shape[0], int(group_size))):
        channel_scores[group] = float(gated_energy[group_idx])
    return values, channel_scores, {
        "group_count": int(group_count),
        "active_group_count": int(keep_count),
        "active_channel_count": int(active_channels),
        "top_group_energy_fraction": 0.0 if total_energy <= 1e-12 else float(np.sum(energy[list(keep_groups)]) / total_energy),
    }


def _top1_acc(
    selector: LinearPatchSelector | None,
    samples: PatchSelectorTrainingSet,
    device: torch.device,
) -> float:
    if samples.sample_count == 0:
        return 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, 1024):
            end = min(start + 1024, samples.sample_count)
            query = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            positives = torch.as_tensor(samples.positive_features[start:end], dtype=torch.float32, device=device)
            mask = torch.as_tensor(samples.positive_mask[start:end], dtype=torch.bool, device=device)
            negatives = torch.as_tensor(samples.negative_features[start:end], dtype=torch.float32, device=device)
            if selector is None:
                query_z = F.normalize(query, dim=-1, eps=1e-8)
                positive_z = F.normalize(positives, dim=-1, eps=1e-8)
                negative_z = F.normalize(negatives, dim=-1, eps=1e-8)
            else:
                query_z = selector(query)
                positive_z = selector(positives.reshape(-1, positives.shape[-1])).reshape(positives.shape[0], positives.shape[1], -1)
                negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
            pos_scores = torch.einsum("bd,bpd->bp", query_z, positive_z).masked_fill(~mask, -1.0e9)
            neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
            correct += int((torch.max(pos_scores, dim=1).values > torch.max(neg_scores, dim=1).values).sum().item())
    return float(correct / max(samples.sample_count, 1))


def _transform_top1_acc(transform: FeatureCompressionTransform, samples: PatchSelectorTrainingSet) -> float:
    if samples.sample_count == 0:
        return 0.0
    query_z = transform.apply_rows(samples.query_features)
    positive_z = transform.apply_rows(samples.positive_features.reshape(-1, samples.input_dim)).reshape(
        samples.sample_count,
        samples.positive_features.shape[1],
        -1,
    )
    negative_z = transform.apply_rows(samples.negative_features.reshape(-1, samples.input_dim)).reshape(
        samples.sample_count,
        samples.negative_features.shape[1],
        -1,
    )
    pos_scores = np.einsum("bd,bpd->bp", query_z, positive_z)
    pos_scores = np.where(samples.positive_mask, pos_scores, -1.0e9)
    neg_scores = np.einsum("bd,bkd->bk", query_z, negative_z)
    return float(np.mean(np.max(pos_scores, axis=1) > np.max(neg_scores, axis=1)))


def train_linear_patch_selector(
    samples: PatchSelectorTrainingSet,
    config: PatchSelectorTrainingConfig | None = None,
) -> PatchSelectorTrainingRun:
    config = config or PatchSelectorTrainingConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    if config.output_dim > samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, config.eval_split_fraction, config.seed)
    train_samples = _subset_samples(samples, train_idx)
    eval_samples = _subset_samples(samples, eval_idx) if eval_idx.size else _subset_samples(samples, train_idx[:0])
    input_mean = train_samples.query_features.mean(axis=0) if config.center_inputs else np.zeros((samples.input_dim,), dtype=np.float32)
    selector = LinearPatchSelector(samples.input_dim, config.output_dim, input_mean=input_mean).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))

    def loss_for_subset(subset: PatchSelectorTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        with torch.no_grad():
            query = torch.as_tensor(subset.query_features, dtype=torch.float32, device=device)
            positives = torch.as_tensor(subset.positive_features, dtype=torch.float32, device=device)
            mask = torch.as_tensor(subset.positive_mask, dtype=torch.bool, device=device)
            negatives = torch.as_tensor(subset.negative_features, dtype=torch.float32, device=device)
            return float(_selector_loss(selector, query, positives, mask, negatives, config.temperature).detach().cpu())

    initial_loss = loss_for_subset(train_samples)
    for _step in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_samples.sample_count)
        batch_idx = rng.choice(train_samples.sample_count, size=batch_count, replace=False)
        query = torch.as_tensor(train_samples.query_features[batch_idx], dtype=torch.float32, device=device)
        positives = torch.as_tensor(train_samples.positive_features[batch_idx], dtype=torch.float32, device=device)
        mask = torch.as_tensor(train_samples.positive_mask[batch_idx], dtype=torch.bool, device=device)
        negatives = torch.as_tensor(train_samples.negative_features[batch_idx], dtype=torch.float32, device=device)
        loss = _selector_loss(selector, query, positives, mask, negatives, config.temperature)
        if config.group_lasso_weight > 0.0 and config.group_size > 0:
            loss = loss + float(config.group_lasso_weight) * _group_lasso_penalty(selector, int(config.group_size))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_loss = loss_for_subset(train_samples)
    train_top1 = _top1_acc(selector, train_samples, device)
    eval_top1 = _top1_acc(selector, eval_samples, device)
    raw_train_top1 = _top1_acc(None, train_samples, device)
    raw_eval_top1 = _top1_acc(None, eval_samples, device)
    matrix = selector.projection.weight.detach().cpu().numpy().astype(np.float32).T
    if config.group_size > 0:
        matrix, channel_scores, group_summary = _apply_hard_group_gate(
            matrix,
            group_size=int(config.group_size),
            keep_fraction=float(config.hard_gate_keep_fraction),
            min_groups=int(config.hard_gate_min_groups),
        )
    else:
        channel_scores = _projection_group_energy(matrix, matrix.shape[0]).repeat(matrix.shape[0]).astype(np.float32)
        group_summary = {
            "group_count": 1,
            "active_group_count": 1,
            "active_channel_count": int(matrix.shape[0]),
            "top_group_energy_fraction": 1.0,
        }
    transform = FeatureCompressionTransform(
        method="learned_linear_patch",
        input_dim=samples.input_dim,
        output_dim=config.output_dim,
        mean=input_mean.astype(np.float32),
        matrix=matrix,
        channel_scores=channel_scores,
        l2_normalize=True,
    )
    gated_train_top1 = _transform_top1_acc(transform, train_samples)
    gated_eval_top1 = _transform_top1_acc(transform, eval_samples)
    return PatchSelectorTrainingRun(
        transform=transform,
        summary=PatchSelectorTrainingSummary(
            initial_loss=initial_loss,
            final_loss=final_loss,
            raw_train_top1_acc=raw_train_top1,
            raw_eval_top1_acc=raw_eval_top1,
            train_top1_acc=train_top1,
            eval_top1_acc=eval_top1,
            sample_count=samples.sample_count,
            train_sample_count=train_samples.sample_count,
            eval_sample_count=eval_samples.sample_count,
            input_dim=samples.input_dim,
            output_dim=config.output_dim,
            steps=int(config.steps),
            batch_size=int(config.batch_size),
            group_size=int(config.group_size),
            group_lasso_weight=float(config.group_lasso_weight),
            hard_gate_keep_fraction=float(config.hard_gate_keep_fraction),
            group_count=int(group_summary["group_count"]),
            active_group_count=int(group_summary["active_group_count"]),
            active_channel_count=int(group_summary["active_channel_count"]),
            top_group_energy_fraction=float(group_summary["top_group_energy_fraction"]),
            gated_train_top1_acc=gated_train_top1,
            gated_eval_top1_acc=gated_eval_top1,
        ),
    )


def _safe_selector_loss(
    selector: ResidualGatedPatchSelector,
    query: torch.Tensor,
    positives: torch.Tensor,
    positive_mask: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
    inlier_loss_weight: float,
) -> torch.Tensor:
    query_z = selector(query)
    positive_z = selector(positives.reshape(-1, positives.shape[-1])).reshape(positives.shape[0], positives.shape[1], -1)
    negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
    pos_logits = torch.einsum("bd,bpd->bp", query_z, positive_z) / float(temperature)
    neg_logits = torch.einsum("bd,bkd->bk", query_z, negative_z) / float(temperature)
    pos_logits = pos_logits.masked_fill(~positive_mask, -1.0e9)
    numerator = torch.logsumexp(pos_logits, dim=1)
    denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
    loss = torch.mean(denominator - numerator)
    if float(inlier_loss_weight) <= 0.0:
        return loss

    query_pos = query_z[:, None, :].expand(-1, positive_z.shape[1], -1).reshape(-1, query_z.shape[-1])
    pos_pair_logits = selector.pairwise_inlier_logit(query_pos, positive_z.reshape(-1, query_z.shape[-1])).reshape(
        positive_z.shape[0],
        positive_z.shape[1],
    )
    query_neg = query_z[:, None, :].expand(-1, negative_z.shape[1], -1).reshape(-1, query_z.shape[-1])
    neg_pair_logits = selector.pairwise_inlier_logit(query_neg, negative_z.reshape(-1, query_z.shape[-1]))
    valid_pos_logits = pos_pair_logits[positive_mask]
    bce_terms = []
    if valid_pos_logits.numel() > 0:
        bce_terms.append(F.binary_cross_entropy_with_logits(valid_pos_logits, torch.ones_like(valid_pos_logits)))
    if neg_pair_logits.numel() > 0:
        bce_terms.append(F.binary_cross_entropy_with_logits(neg_pair_logits, torch.zeros_like(neg_pair_logits)))
    if bce_terms:
        loss = loss + float(inlier_loss_weight) * torch.stack(bce_terms).mean()
    return loss


def _safe_group_gate_penalty(selector: ResidualGatedPatchSelector) -> torch.Tensor:
    gates = selector.group_gates()
    return gates.mean()


def _safe_active_group_mask(
    selector: ResidualGatedPatchSelector,
    keep_fraction: float,
    min_groups: int,
) -> np.ndarray:
    gates = selector.group_gates().detach().cpu().numpy().astype(np.float32)
    weights = selector.projection.weight.detach().cpu().numpy().astype(np.float32).T
    energy = _projection_group_energy(weights, selector.group_size)
    scores = gates * energy
    group_count = int(scores.shape[0])
    if group_count == 0:
        return np.zeros((0,), dtype=np.float32)
    keep_count = int(np.ceil(float(keep_fraction) * group_count))
    keep_count = max(int(min_groups), keep_count)
    keep_count = min(keep_count, group_count)
    order = np.lexsort((np.arange(group_count), -scores))
    mask = np.zeros((group_count,), dtype=np.float32)
    mask[order[:keep_count]] = 1.0
    return mask


def _safe_top1_acc(
    selector: ResidualGatedPatchSelector | None,
    samples: PatchSelectorTrainingSet,
    device: torch.device,
    active_group_mask: np.ndarray | None = None,
) -> float:
    if samples.sample_count == 0:
        return 0.0
    correct = 0
    active_mask_tensor = None
    if active_group_mask is not None and selector is not None:
        active_mask_tensor = torch.as_tensor(active_group_mask, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, samples.sample_count, 1024):
            end = min(start + 1024, samples.sample_count)
            query = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            positives = torch.as_tensor(samples.positive_features[start:end], dtype=torch.float32, device=device)
            mask = torch.as_tensor(samples.positive_mask[start:end], dtype=torch.bool, device=device)
            negatives = torch.as_tensor(samples.negative_features[start:end], dtype=torch.float32, device=device)
            if selector is None:
                query_z = F.normalize(query, dim=-1, eps=1e-8)
                positive_z = F.normalize(positives, dim=-1, eps=1e-8)
                negative_z = F.normalize(negatives, dim=-1, eps=1e-8)
            else:
                query_z = selector(query, active_group_mask=active_mask_tensor)
                positive_z = selector(positives.reshape(-1, positives.shape[-1]), active_group_mask=active_mask_tensor).reshape(
                    positives.shape[0],
                    positives.shape[1],
                    -1,
                )
                negative_z = selector(negatives.reshape(-1, negatives.shape[-1]), active_group_mask=active_mask_tensor).reshape(
                    negatives.shape[0],
                    negatives.shape[1],
                    -1,
                )
            pos_scores = torch.einsum("bd,bpd->bp", query_z, positive_z).masked_fill(~mask, -1.0e9)
            neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
            correct += int((torch.max(pos_scores, dim=1).values > torch.max(neg_scores, dim=1).values).sum().item())
    return float(correct / max(samples.sample_count, 1))


def _safe_inlier_accuracy(
    selector: ResidualGatedPatchSelector,
    samples: PatchSelectorTrainingSet,
    device: torch.device,
) -> float:
    if samples.sample_count == 0:
        return 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, 1024):
            end = min(start + 1024, samples.sample_count)
            query = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            positives = torch.as_tensor(samples.positive_features[start:end], dtype=torch.float32, device=device)
            mask = torch.as_tensor(samples.positive_mask[start:end], dtype=torch.bool, device=device)
            negatives = torch.as_tensor(samples.negative_features[start:end], dtype=torch.float32, device=device)
            query_z = selector(query)
            positive_z = selector(positives.reshape(-1, positives.shape[-1])).reshape(positives.shape[0], positives.shape[1], -1)
            negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
            query_pos = query_z[:, None, :].expand(-1, positive_z.shape[1], -1).reshape(-1, query_z.shape[-1])
            pos_logits = selector.pairwise_inlier_logit(query_pos, positive_z.reshape(-1, query_z.shape[-1])).reshape(
                positive_z.shape[0],
                positive_z.shape[1],
            )
            query_neg = query_z[:, None, :].expand(-1, negative_z.shape[1], -1).reshape(-1, query_z.shape[-1])
            neg_logits = selector.pairwise_inlier_logit(query_neg, negative_z.reshape(-1, query_z.shape[-1]))
            valid_pos = pos_logits[mask]
            if valid_pos.numel() > 0:
                correct += int((valid_pos > 0.0).sum().item())
                total += int(valid_pos.numel())
            correct += int((neg_logits <= 0.0).sum().item())
            total += int(neg_logits.numel())
    return float(correct / max(total, 1))


def encode_rows_with_safe_selector(
    selector: ResidualGatedPatchSelector,
    rows: np.ndarray,
    device: str = "cpu",
    batch_size: int = 65536,
    active_group_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != int(selector.input_dim):
        raise ValueError("rows must have shape (N, input_dim)")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    torch_device = torch.device(device)
    was_training = selector.training
    selector = selector.to(torch_device)
    selector.eval()
    active_mask_tensor = None
    if active_group_mask is not None:
        active_mask_tensor = torch.as_tensor(active_group_mask, dtype=torch.float32, device=torch_device)
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            batch = torch.as_tensor(values[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            encoded = selector(batch, active_group_mask=active_mask_tensor)
            chunks.append(encoded.detach().cpu().numpy().astype(np.float32))
    if was_training:
        selector.train()
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, selector.output_dim), dtype=np.float32)


def train_safe_patch_selector(
    samples: PatchSelectorTrainingSet,
    config: SafePatchSelectorTrainingConfig | None = None,
) -> SafePatchSelectorTrainingRun:
    config = config or SafePatchSelectorTrainingConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    if config.output_dim > samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, config.eval_split_fraction, config.seed)
    train_samples = _subset_samples(samples, train_idx)
    eval_samples = _subset_samples(samples, eval_idx) if eval_idx.size else _subset_samples(samples, train_idx[:0])
    input_mean = train_samples.query_features.mean(axis=0) if config.center_inputs else np.zeros((samples.input_dim,), dtype=np.float32)
    selector = ResidualGatedPatchSelector(
        samples.input_dim,
        config.output_dim,
        residual_hidden_dim=int(config.residual_hidden_dim),
        group_size=int(config.group_size),
        input_mean=input_mean,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))

    def loss_for_subset(subset: PatchSelectorTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        with torch.no_grad():
            query = torch.as_tensor(subset.query_features, dtype=torch.float32, device=device)
            positives = torch.as_tensor(subset.positive_features, dtype=torch.float32, device=device)
            mask = torch.as_tensor(subset.positive_mask, dtype=torch.bool, device=device)
            negatives = torch.as_tensor(subset.negative_features, dtype=torch.float32, device=device)
            return float(
                _safe_selector_loss(
                    selector,
                    query,
                    positives,
                    mask,
                    negatives,
                    config.temperature,
                    config.inlier_loss_weight,
                )
                .detach()
                .cpu()
            )

    initial_loss = loss_for_subset(train_samples)
    for _step in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_samples.sample_count)
        batch_idx = rng.choice(train_samples.sample_count, size=batch_count, replace=False)
        query = torch.as_tensor(train_samples.query_features[batch_idx], dtype=torch.float32, device=device)
        positives = torch.as_tensor(train_samples.positive_features[batch_idx], dtype=torch.float32, device=device)
        mask = torch.as_tensor(train_samples.positive_mask[batch_idx], dtype=torch.bool, device=device)
        negatives = torch.as_tensor(train_samples.negative_features[batch_idx], dtype=torch.float32, device=device)
        loss = _safe_selector_loss(
            selector,
            query,
            positives,
            mask,
            negatives,
            config.temperature,
            config.inlier_loss_weight,
        )
        if config.group_lasso_weight > 0.0 and config.group_size > 0:
            loss = loss + float(config.group_lasso_weight) * _safe_group_gate_penalty(selector)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_loss = loss_for_subset(train_samples)
    active_group_mask = _safe_active_group_mask(
        selector,
        keep_fraction=float(config.hard_gate_keep_fraction),
        min_groups=int(config.hard_gate_min_groups),
    )
    gates = selector.group_gates().detach().cpu().numpy().astype(np.float32)
    active_channels = 0
    for group_idx, group in enumerate(_group_slices(samples.input_dim, int(config.group_size))):
        if group_idx < active_group_mask.shape[0] and active_group_mask[group_idx] > 0.0:
            active_channels += int(group.stop - group.start)
    train_top1 = _safe_top1_acc(selector, train_samples, device, active_group_mask=active_group_mask)
    eval_top1 = _safe_top1_acc(selector, eval_samples, device, active_group_mask=active_group_mask)
    raw_train_top1 = _safe_top1_acc(None, train_samples, device)
    raw_eval_top1 = _safe_top1_acc(None, eval_samples, device)
    parameter_count = int(sum(param.numel() for param in selector.parameters()))
    return SafePatchSelectorTrainingRun(
        model=selector.cpu(),
        summary=SafePatchSelectorTrainingSummary(
            initial_loss=initial_loss,
            final_loss=final_loss,
            raw_train_top1_acc=raw_train_top1,
            raw_eval_top1_acc=raw_eval_top1,
            train_top1_acc=train_top1,
            eval_top1_acc=eval_top1,
            inlier_train_accuracy=_safe_inlier_accuracy(selector.to(device), train_samples, device),
            inlier_eval_accuracy=_safe_inlier_accuracy(selector.to(device), eval_samples, device),
            sample_count=samples.sample_count,
            train_sample_count=train_samples.sample_count,
            eval_sample_count=eval_samples.sample_count,
            input_dim=samples.input_dim,
            output_dim=int(config.output_dim),
            residual_hidden_dim=int(config.residual_hidden_dim),
            steps=int(config.steps),
            batch_size=int(config.batch_size),
            group_size=int(config.group_size),
            group_count=int(active_group_mask.shape[0]),
            active_group_count=int(np.sum(active_group_mask > 0.0)),
            active_channel_count=int(active_channels),
            active_group_fraction=float(np.mean(active_group_mask > 0.0)) if active_group_mask.size else 0.0,
            gate_mean=float(np.mean(gates)) if gates.size else 0.0,
            gate_min=float(np.min(gates)) if gates.size else 0.0,
            gate_max=float(np.max(gates)) if gates.size else 0.0,
            parameter_count=parameter_count,
            inlier_loss_weight=float(config.inlier_loss_weight),
            group_lasso_weight=float(config.group_lasso_weight),
            hard_gate_keep_fraction=float(config.hard_gate_keep_fraction),
        ),
        active_group_mask=active_group_mask.astype(np.float32, copy=False),
    )


def save_safe_patch_selector_checkpoint(run: SafePatchSelectorTrainingRun, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = run.model.cpu()
    payload = {
        "format": "vfm_stage_c2_safe_patch_selector_v1",
        "model_config": {
            "input_dim": int(model.input_dim),
            "output_dim": int(model.output_dim),
            "residual_hidden_dim": int(model.residual_hidden_dim),
            "group_size": int(model.group_size),
            "input_mean": model.input_mean.detach().cpu().numpy().reshape(-1),
        },
        "state_dict": model.state_dict(),
        "summary": asdict(run.summary),
        "active_group_mask": np.asarray(run.active_group_mask, dtype=np.float32),
    }
    torch.save(payload, output)


def load_safe_patch_selector_checkpoint(path: Path, device: str = "cpu") -> SafePatchSelectorTrainingRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "vfm_stage_c2_safe_patch_selector_v1":
        raise ValueError(f"unsupported safe patch selector checkpoint format in {path}")
    config = dict(payload["model_config"])
    model = ResidualGatedPatchSelector(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        group_size=int(config["group_size"]),
        input_mean=np.asarray(config["input_mean"], dtype=np.float32),
    )
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.device(device)).eval()
    summary = SafePatchSelectorTrainingSummary(**dict(payload["summary"]))
    active_group_mask = np.asarray(payload["active_group_mask"], dtype=np.float32)
    return SafePatchSelectorTrainingRun(model=model.cpu(), summary=summary, active_group_mask=active_group_mask)


@dataclass
class SafePairwiseInlierScorer:
    model: ResidualGatedPatchSelector
    device: str = "cpu"
    batch_size: int = 65536

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: str = "cpu",
        batch_size: int = 65536,
    ) -> "SafePairwiseInlierScorer":
        run = load_safe_patch_selector_checkpoint(Path(path), device=device)
        return cls(model=run.model, device=device, batch_size=int(batch_size))

    def score_pairs(self, query_descriptors: np.ndarray, landmark_descriptors: np.ndarray) -> np.ndarray:
        query = np.asarray(query_descriptors, dtype=np.float32)
        landmarks = np.asarray(landmark_descriptors, dtype=np.float32)
        if query.ndim != 2 or landmarks.ndim != 2:
            raise ValueError("query_descriptors and landmark_descriptors must have shape (N, D)")
        if query.shape != landmarks.shape:
            raise ValueError("query_descriptors and landmark_descriptors must have the same shape")
        if query.shape[1] != int(self.model.output_dim):
            raise ValueError("descriptor dimension must match the safe selector output_dim")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        torch_device = torch.device(self.device)
        was_training = self.model.training
        model = self.model.to(torch_device).eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, query.shape[0], int(self.batch_size)):
                end = min(start + int(self.batch_size), query.shape[0])
                q = torch.as_tensor(query[start:end], dtype=torch.float32, device=torch_device)
                x = torch.as_tensor(landmarks[start:end], dtype=torch.float32, device=torch_device)
                chunks.append(model.pairwise_inlier_logit(q, x).detach().cpu().numpy().astype(np.float32))
        if was_training:
            self.model.train()
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
