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
    negative_curriculum: str = "hard"
    medium_min_stride: float = 4.0
    semi_hard_min_stride: float = 2.0
    semi_hard_max_stride: float = 4.0
    mixed_hard_fraction: float = 0.1

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
        if self.negative_curriculum not in {"hard", "medium", "semi_hard", "mixed"}:
            raise ValueError("negative_curriculum must be one of: hard, medium, semi_hard, mixed")
        if self.medium_min_stride < 0.0:
            raise ValueError("medium_min_stride must be non-negative")
        if self.semi_hard_min_stride < 0.0 or self.semi_hard_max_stride < self.semi_hard_min_stride:
            raise ValueError("semi-hard stride bounds are invalid")
        if not 0.0 <= float(self.mixed_hard_fraction) <= 1.0:
            raise ValueError("mixed_hard_fraction must be in [0, 1]")


@dataclass(frozen=True)
class PatchSelectorTrainingSet:
    query_features: np.ndarray
    positive_features: np.ndarray
    positive_mask: np.ndarray
    negative_features: np.ndarray
    positive_reprojection_distances: np.ndarray | None = None
    negative_reprojection_distances: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        query = np.asarray(self.query_features, dtype=np.float32)
        positives = np.asarray(self.positive_features, dtype=np.float32)
        mask = np.asarray(self.positive_mask, dtype=bool)
        negatives = np.asarray(self.negative_features, dtype=np.float32)
        positive_distances = (
            None if self.positive_reprojection_distances is None else np.asarray(self.positive_reprojection_distances, dtype=np.float32)
        )
        negative_distances = (
            None if self.negative_reprojection_distances is None else np.asarray(self.negative_reprojection_distances, dtype=np.float32)
        )
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
        if positive_distances is not None and positive_distances.shape != positives.shape[:2]:
            raise ValueError("positive_reprojection_distances must have shape (N, P)")
        if negative_distances is not None and negative_distances.shape != negatives.shape[:2]:
            raise ValueError("negative_reprojection_distances must have shape (N, K)")
        if (positive_distances is None) != (negative_distances is None):
            raise ValueError("positive and negative reprojection distances must be provided together")
        if query.shape[0] and not np.all(mask.any(axis=1)):
            raise ValueError("each sample must have at least one positive")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "positive_features", positives)
        object.__setattr__(self, "positive_mask", mask)
        object.__setattr__(self, "negative_features", negatives)
        object.__setattr__(self, "positive_reprojection_distances", positive_distances)
        object.__setattr__(self, "negative_reprojection_distances", negative_distances)
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
    selector_arch: str = "residual_gated"
    output_dim: int = 128
    residual_hidden_dim: int = 256
    transformer_dim: int = 128
    transformer_layers: int = 1
    transformer_heads: int = 4
    transformer_ff_dim: int = 256
    transformer_dropout: float = 0.0
    steps: int = 800
    batch_size: int = 512
    lr: float = 5e-4
    projection_lr: float | None = None
    residual_lr: float | None = None
    pairwise_lr: float | None = None
    gate_lr: float | None = None
    temperature: float = 0.07
    inlier_loss_weight: float = 0.2
    anchor_loss_weight: float = 0.0
    reprojection_margin_loss_weight: float = 0.0
    reprojection_margin_max: float = 2.0
    group_lasso_weight: float = 0.0
    seed: int = 0
    device: str = "cpu"
    eval_split_fraction: float = 0.1
    center_inputs: bool = False
    group_size: int = 64
    input_norm_mode: str = "layernorm"
    gate_mode: str = "sigmoid"
    residual_gate_scale: float = 0.1
    hard_gate_keep_fraction: float = 1.0
    hard_gate_min_groups: int = 1

    def __post_init__(self) -> None:
        if str(self.selector_arch) not in {"residual_gated", "transformer_group_token"}:
            raise ValueError("selector_arch must be 'residual_gated' or 'transformer_group_token'")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if self.transformer_dim <= 0:
            raise ValueError("transformer_dim must be positive")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if self.transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive")
        if self.transformer_dim % self.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if self.transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if not 0.0 <= float(self.transformer_dropout) < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        for name in ("projection_lr", "residual_lr", "pairwise_lr", "gate_lr"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if float(self.inlier_loss_weight) < 0.0:
            raise ValueError("inlier_loss_weight must be non-negative")
        if float(self.anchor_loss_weight) < 0.0:
            raise ValueError("anchor_loss_weight must be non-negative")
        if float(self.reprojection_margin_loss_weight) < 0.0:
            raise ValueError("reprojection_margin_loss_weight must be non-negative")
        if float(self.reprojection_margin_max) <= 0.0:
            raise ValueError("reprojection_margin_max must be positive")
        if float(self.group_lasso_weight) < 0.0:
            raise ValueError("group_lasso_weight must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")
        if int(self.group_size) < 0:
            raise ValueError("group_size must be non-negative")
        if str(self.input_norm_mode) not in {"layernorm", "identity"}:
            raise ValueError("input_norm_mode must be 'layernorm' or 'identity'")
        if str(self.gate_mode) not in {"sigmoid", "residual"}:
            raise ValueError("gate_mode must be 'sigmoid' or 'residual'")
        if not 0.0 <= float(self.residual_gate_scale) < 1.0:
            raise ValueError("residual_gate_scale must be in [0, 1)")
        if not 0.0 < float(self.hard_gate_keep_fraction) <= 1.0:
            raise ValueError("hard_gate_keep_fraction must be in (0, 1]")
        if int(self.hard_gate_min_groups) <= 0:
            raise ValueError("hard_gate_min_groups must be positive")


@dataclass(frozen=True)
class SafePatchSelectorTrainingSummary:
    selector_arch: str
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
    transformer_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    transformer_dropout: float
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
    projection_lr: float = 5e-4
    residual_lr: float = 5e-4
    pairwise_lr: float = 5e-4
    gate_lr: float = 5e-4
    anchor_loss_weight: float = 0.0
    reprojection_margin_loss_weight: float = 0.0
    reprojection_margin_max: float = 2.0
    input_norm_mode: str = "layernorm"
    gate_mode: str = "sigmoid"
    residual_gate_scale: float = 0.1
    initialized_from_safe_checkpoint: bool = False


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
        input_norm_mode: str = "layernorm",
        gate_mode: str = "sigmoid",
        residual_gate_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if int(output_dim) > int(input_dim):
            raise ValueError("output_dim must be <= input_dim")
        if int(residual_hidden_dim) <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if str(input_norm_mode) not in {"layernorm", "identity"}:
            raise ValueError("input_norm_mode must be 'layernorm' or 'identity'")
        if str(gate_mode) not in {"sigmoid", "residual"}:
            raise ValueError("gate_mode must be 'sigmoid' or 'residual'")
        if not 0.0 <= float(residual_gate_scale) < 1.0:
            raise ValueError("residual_gate_scale must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.group_size = int(group_size)
        self.input_norm_mode = str(input_norm_mode)
        self.gate_mode = str(gate_mode)
        self.residual_gate_scale = float(residual_gate_scale)
        group_count = len(_group_slices(self.input_dim, self.group_size))
        self.input_norm = nn.LayerNorm(self.input_dim) if self.input_norm_mode == "layernorm" else nn.Identity()
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
        gate_init = 2.0 if self.gate_mode == "sigmoid" else 0.0
        self.group_logits = nn.Parameter(torch.full((group_count,), gate_init, dtype=torch.float32))
        mean = np.zeros((self.input_dim,), dtype=np.float32) if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self.register_buffer("input_mean", torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32))
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def group_gates(self) -> torch.Tensor:
        if self.gate_mode == "sigmoid":
            return torch.sigmoid(self.group_logits)
        return 1.0 + float(self.residual_gate_scale) * torch.tanh(self.group_logits)

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


class TransformerGroupPatchSelector(nn.Module):
    """Candidate-independent group-token Transformer descriptor selector."""

    selector_arch = "transformer_group_token"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        group_size: int = 64,
        transformer_dim: int = 128,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_dropout: float = 0.0,
        pairwise_hidden_dim: int = 256,
        input_mean: np.ndarray | None = None,
        input_norm_mode: str = "layernorm",
        gate_mode: str = "residual",
        residual_gate_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if int(output_dim) > int(input_dim):
            raise ValueError("output_dim must be <= input_dim")
        if int(group_size) <= 0:
            raise ValueError("group_size must be positive for transformer selector")
        if int(transformer_dim) <= 0:
            raise ValueError("transformer_dim must be positive")
        if int(transformer_layers) <= 0:
            raise ValueError("transformer_layers must be positive")
        if int(transformer_heads) <= 0 or int(transformer_dim) % int(transformer_heads) != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if int(transformer_ff_dim) <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if int(pairwise_hidden_dim) <= 0:
            raise ValueError("pairwise_hidden_dim must be positive")
        if str(input_norm_mode) not in {"layernorm", "identity"}:
            raise ValueError("input_norm_mode must be 'layernorm' or 'identity'")
        if str(gate_mode) not in {"sigmoid", "residual"}:
            raise ValueError("gate_mode must be 'sigmoid' or 'residual'")
        if not 0.0 <= float(residual_gate_scale) < 1.0:
            raise ValueError("residual_gate_scale must be in [0, 1)")
        if not 0.0 <= float(transformer_dropout) < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_hidden_dim = int(pairwise_hidden_dim)
        self.group_size = int(group_size)
        self.transformer_dim = int(transformer_dim)
        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.transformer_dropout = float(transformer_dropout)
        self.input_norm_mode = str(input_norm_mode)
        self.gate_mode = str(gate_mode)
        self.residual_gate_scale = float(residual_gate_scale)
        group_count = len(_group_slices(self.input_dim, self.group_size))
        padded_dim = group_count * self.group_size
        self.group_count = int(group_count)
        self.padded_dim = int(padded_dim)
        self.input_norm = nn.LayerNorm(self.input_dim) if self.input_norm_mode == "layernorm" else nn.Identity()
        self.group_projection = nn.Linear(self.group_size, self.transformer_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.group_count + 1, self.transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=self.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.output_norm = nn.LayerNorm(self.transformer_dim)
        self.projection = nn.Linear(self.transformer_dim, self.output_dim, bias=False)
        self.query_matchability = nn.Linear(self.output_dim, 1)
        self.landmark_reliability = nn.Linear(self.output_dim, 1)
        self.pairwise_inlier = nn.Sequential(
            nn.Linear(self.output_dim * 4, self.residual_hidden_dim),
            nn.GELU(),
            nn.Linear(self.residual_hidden_dim, 1),
        )
        gate_init = 2.0 if self.gate_mode == "sigmoid" else 0.0
        self.group_logits = nn.Parameter(torch.full((self.group_count,), gate_init, dtype=torch.float32))
        mean = np.zeros((self.input_dim,), dtype=np.float32) if input_mean is None else np.asarray(input_mean, dtype=np.float32)
        self.register_buffer("input_mean", torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.group_projection.weight)
        nn.init.zeros_(self.group_projection.bias)
        nn.init.xavier_uniform_(self.projection.weight)

    def group_gates(self) -> torch.Tensor:
        if self.gate_mode == "sigmoid":
            return torch.sigmoid(self.group_logits)
        return 1.0 + float(self.residual_gate_scale) * torch.tanh(self.group_logits)

    def _channel_gates(self, active_group_mask: torch.Tensor | None = None) -> torch.Tensor:
        gates = self.group_gates()
        if active_group_mask is not None:
            gates = gates * active_group_mask.to(gates.device, dtype=gates.dtype)
        chunks = []
        for group_idx, group in enumerate(_group_slices(self.input_dim, self.group_size)):
            chunks.append(gates[group_idx].expand(group.stop - group.start))
        return torch.cat(chunks, dim=0).reshape(1, -1)

    def group_score_estimates(self) -> np.ndarray:
        return self.group_gates().detach().cpu().numpy().astype(np.float32)

    def forward(self, features: torch.Tensor, active_group_mask: torch.Tensor | None = None) -> torch.Tensor:
        centered = features - self.input_mean.to(features.device)
        normalized = self.input_norm(centered)
        gated = normalized * self._channel_gates(active_group_mask).to(features.device)
        if self.padded_dim > self.input_dim:
            gated = F.pad(gated, (0, self.padded_dim - self.input_dim))
        tokens = gated.reshape(gated.shape[0], self.group_count, self.group_size)
        tokens = self.group_projection(tokens)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = torch.cat([cls, tokens], dim=1) + self.position_embedding
        encoded = self.encoder(encoded)
        descriptor = self.projection(self.output_norm(encoded[:, 0, :]))
        return F.normalize(descriptor, dim=-1, eps=1e-8)

    def pairwise_inlier_logit(self, query_z: torch.Tensor, landmark_z: torch.Tensor) -> torch.Tensor:
        if query_z.shape != landmark_z.shape:
            raise ValueError("query_z and landmark_z must have the same shape")
        pair = torch.cat([query_z, landmark_z, torch.abs(query_z - landmark_z), query_z * landmark_z], dim=-1)
        logits = self.pairwise_inlier(pair)
        logits = logits + self.query_matchability(query_z) + self.landmark_reliability(landmark_z)
        return logits.squeeze(-1)


def initialize_safe_selector_from_linear_transform(
    selector: ResidualGatedPatchSelector,
    transform: FeatureCompressionTransform,
) -> None:
    """Initialize the C2 selector so its descriptor starts as a C1 linear transform."""

    if transform.matrix is None:
        raise ValueError("init transform must contain a projection matrix")
    if int(transform.input_dim) != int(selector.input_dim):
        raise ValueError("init transform input_dim must match selector input_dim")
    if int(transform.output_dim) != int(selector.output_dim):
        raise ValueError("init transform output_dim must match selector output_dim")
    if not bool(transform.l2_normalize):
        raise ValueError("init transform must use L2 normalization")
    with torch.no_grad():
        selector.input_mean.copy_(torch.as_tensor(transform.mean.reshape(1, -1), dtype=torch.float32, device=selector.input_mean.device))
        selector.projection.weight.copy_(
            torch.as_tensor(transform.matrix.T, dtype=selector.projection.weight.dtype, device=selector.projection.weight.device)
        )
        nn.init.zeros_(selector.residual[-1].weight)
        nn.init.zeros_(selector.residual[-1].bias)
        if selector.gate_mode == "residual":
            selector.group_logits.zero_()
        else:
            selector.group_logits.fill_(8.0)


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


def _distance_to_patch_center_strides(
    token_index: int,
    track_index: int,
    track_ids: np.ndarray,
    positives: PatchPositiveSets,
) -> float:
    positive = positives.by_token.get(int(token_index))
    if positive is None:
        return float("inf")
    projected = None if positives.projected_xy_by_track is None else positives.projected_xy_by_track.get(int(track_ids[int(track_index)]))
    if projected is None:
        return float("inf")
    stride = max(float(positives.stride_x_px), float(positives.stride_y_px), 1.0)
    return float(np.linalg.norm(np.asarray(projected, dtype=np.float64).reshape(2) - positive.patch_box.center) / stride)


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


def _select_negative_indices_by_curriculum(
    token_index: int,
    ordered_candidates: list[int],
    landmark_index: LandmarkMapIndex,
    positives: PatchPositiveSets,
    config: PatchSelectorSampleConfig,
) -> list[int]:
    count = int(config.hard_negatives_per_token)
    if config.negative_curriculum == "hard":
        return ordered_candidates[:count]

    by_distance = [
        (
            int(idx),
            _distance_to_patch_center_strides(
                int(token_index),
                int(idx),
                landmark_index.track_ids,
                positives,
            ),
        )
        for idx in ordered_candidates
    ]

    def medium() -> list[int]:
        return [
            idx
            for idx, distance in by_distance
            if (not np.isfinite(float(distance))) or float(distance) >= float(config.medium_min_stride)
        ]

    def semi_hard() -> list[int]:
        return [
            idx
            for idx, distance in by_distance
            if np.isfinite(float(distance))
            and float(config.semi_hard_min_stride) <= float(distance) <= float(config.semi_hard_max_stride)
        ]

    if config.negative_curriculum == "medium":
        preferred = medium()
    elif config.negative_curriculum == "semi_hard":
        preferred = semi_hard()
    else:
        hard_count = int(round(count * float(config.mixed_hard_fraction)))
        remaining = max(count - hard_count, 0)
        preferred = semi_hard()[:remaining] + medium()
        preferred.extend(ordered_candidates[:hard_count])

    selected: list[int] = []
    seen: set[int] = set()
    for idx in preferred + ordered_candidates:
        if int(idx) in seen:
            continue
        selected.append(int(idx))
        seen.add(int(idx))
        if len(selected) >= count:
            break
    return selected


def build_patch_selector_samples_for_query(
    query_feature_map: np.ndarray,
    landmark_index: LandmarkMapIndex,
    positives: PatchPositiveSets,
    config: PatchSelectorSampleConfig | None = None,
    negative_mining_query_feature_map: np.ndarray | None = None,
    negative_mining_landmark_features: np.ndarray | None = None,
) -> PatchSelectorTrainingSet:
    config = config or PatchSelectorSampleConfig()
    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")
    if landmark_index.feature_dim != int(feature_map.shape[0]):
        raise ValueError("landmark feature dimension must match query feature dimension")
    rng = np.random.default_rng(int(config.seed))
    query_features, token_indices = _token_rows(feature_map, config.query_token_step)
    mining_query_features = query_features
    if negative_mining_query_feature_map is not None:
        mining_map = np.asarray(negative_mining_query_feature_map, dtype=np.float32)
        if mining_map.ndim != 3 or mining_map.shape[1:] != feature_map.shape[1:]:
            raise ValueError("negative_mining_query_feature_map must have shape (C2, H, W)")
        mining_query_features, mining_token_indices = _token_rows(mining_map, config.query_token_step)
        if not np.array_equal(mining_token_indices, token_indices):
            raise ValueError("negative mining token grid must match query token grid")
    mining_landmark_features = landmark_index.features
    negative_mining_descriptor = "raw"
    if negative_mining_landmark_features is not None:
        mining_landmark_features = np.asarray(negative_mining_landmark_features, dtype=np.float32)
        if mining_landmark_features.ndim != 2 or mining_landmark_features.shape[0] != len(landmark_index):
            raise ValueError("negative_mining_landmark_features must have shape (landmark_count, C2)")
        if mining_landmark_features.shape[1] != mining_query_features.shape[1]:
            raise ValueError("negative mining query and landmark feature dimensions must match")
        negative_mining_descriptor = "alternate"
    if query_features.size == 0 or len(landmark_index) == 0:
        return _empty_training_set(int(feature_map.shape[0]), config, {"sample_count": 0})

    row_by_track = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    token_candidates: list[tuple[int, np.ndarray, np.ndarray, list[int], list[int]]] = []
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
        token_candidates.append(
            (int(token_index), query_features[row_idx], mining_query_features[row_idx], all_positive_indices, positive_indices)
        )

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

    selected_mining_query_features = np.stack([item[2] for item in token_candidates], axis=0).astype(np.float32)
    all_positive_sets = [set(item[3]) for item in token_candidates]
    negative_search_count = (
        int(config.hard_negatives_per_token)
        if config.negative_curriculum == "hard"
        else int(config.hard_negative_pool)
    )
    negative_lists = _sample_negative_indices_batch(
        selected_mining_query_features,
        mining_landmark_features,
        positive_row_indices=all_positive_sets,
        count=negative_search_count,
        pool=config.hard_negative_pool,
    )

    candidate_rows: list[tuple[int, np.ndarray, list[int], list[int]]] = []
    for (token_index, query_feature, _mining_query_feature, _all_positive_indices, positive_indices), negative_indices in zip(
        token_candidates,
        negative_lists,
    ):
        negative_indices = _select_negative_indices_by_curriculum(
            int(token_index),
            negative_indices,
            landmark_index,
            positives,
            config,
        )
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
    pos_distances = []
    neg_distances = []
    for _token_index, query_feature, positive_indices, negative_indices in candidate_rows:
        positive_array = np.zeros((config.max_positives_per_token, landmark_index.feature_dim), dtype=np.float32)
        mask_array = np.zeros((config.max_positives_per_token,), dtype=bool)
        positive_distance_array = np.zeros((config.max_positives_per_token,), dtype=np.float32)
        for dst, landmark_idx in enumerate(positive_indices[: config.max_positives_per_token]):
            positive_array[dst] = landmark_index.features[int(landmark_idx)]
            mask_array[dst] = True
            positive_distance_array[dst] = _distance_to_patch_center_strides(_token_index, int(landmark_idx), landmark_index.track_ids, positives)
        negative_array = landmark_index.features[np.asarray(negative_indices[: config.hard_negatives_per_token], dtype=np.int64)]
        negative_distance_array = np.asarray(
            [
                _distance_to_patch_center_strides(_token_index, int(landmark_idx), landmark_index.track_ids, positives)
                for landmark_idx in negative_indices[: config.hard_negatives_per_token]
            ],
            dtype=np.float32,
        )
        queries.append(query_feature.astype(np.float32, copy=True))
        pos.append(positive_array)
        pos_mask.append(mask_array)
        neg.append(negative_array.astype(np.float32, copy=True))
        pos_distances.append(positive_distance_array)
        neg_distances.append(negative_distance_array)

    return PatchSelectorTrainingSet(
        query_features=np.stack(queries, axis=0),
        positive_features=np.stack(pos, axis=0),
        positive_mask=np.stack(pos_mask, axis=0),
        negative_features=np.stack(neg, axis=0),
        positive_reprojection_distances=np.stack(pos_distances, axis=0),
        negative_reprojection_distances=np.stack(neg_distances, axis=0),
        metadata={
            "sample_count": len(queries),
            "candidate_token_count": len(candidate_rows),
            "raw_false_nearest_negative_count": len(queries) * int(config.hard_negatives_per_token),
            "max_positives_per_token": int(config.max_positives_per_token),
            "hard_negatives_per_token": int(config.hard_negatives_per_token),
            "has_reprojection_distances": True,
            "negative_mining_descriptor": negative_mining_descriptor,
            "negative_curriculum": config.negative_curriculum,
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
    has_distances = all(samples.positive_reprojection_distances is not None for samples in present)
    positive_distances = (
        np.concatenate([samples.positive_reprojection_distances for samples in present if samples.positive_reprojection_distances is not None], axis=0)
        if has_distances
        else None
    )
    negative_distances = (
        np.concatenate([samples.negative_reprojection_distances for samples in present if samples.negative_reprojection_distances is not None], axis=0)
        if has_distances
        else None
    )
    if max_samples > 0 and query.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(query.shape[0], size=int(max_samples), replace=False))
        query = query[indices]
        positive = positive[indices]
        positive_mask = positive_mask[indices]
        negative = negative[indices]
        if positive_distances is not None:
            positive_distances = positive_distances[indices]
        if negative_distances is not None:
            negative_distances = negative_distances[indices]
    return PatchSelectorTrainingSet(
        query_features=query,
        positive_features=positive,
        positive_mask=positive_mask,
        negative_features=negative,
        positive_reprojection_distances=positive_distances,
        negative_reprojection_distances=negative_distances,
        metadata={"sample_count": int(query.shape[0]), "source_set_count": len(present), "has_reprojection_distances": bool(has_distances)},
    )


def _with_training_set_metadata(samples: PatchSelectorTrainingSet, metadata: Mapping[str, object]) -> PatchSelectorTrainingSet:
    return PatchSelectorTrainingSet(
        query_features=samples.query_features,
        positive_features=samples.positive_features,
        positive_mask=samples.positive_mask,
        negative_features=samples.negative_features,
        positive_reprojection_distances=samples.positive_reprojection_distances,
        negative_reprojection_distances=samples.negative_reprojection_distances,
        metadata=metadata,
    )


def append_patch_selector_training_set_capped(
    existing: PatchSelectorTrainingSet | None,
    incoming: PatchSelectorTrainingSet,
    max_samples: int = 0,
    seed: int = 0,
) -> PatchSelectorTrainingSet:
    """Append one query's samples while keeping the merged cache bounded."""

    if incoming.sample_count <= 0:
        if existing is None:
            return incoming
        return existing
    existing_source_count = 0
    if existing is not None:
        existing_source_count = int(existing.metadata.get("source_set_count", 1 if existing.sample_count > 0 else 0))
    incoming_source_count = 1
    merged = merge_patch_selector_training_sets(
        [incoming] if existing is None else [existing, incoming],
        max_samples=int(max_samples),
        seed=int(seed),
    )
    metadata = {
        **dict(merged.metadata or {}),
        "sample_count": int(merged.sample_count),
        "source_set_count": int(existing_source_count + incoming_source_count),
        "has_reprojection_distances": bool(merged.positive_reprojection_distances is not None),
    }
    return _with_training_set_metadata(merged, metadata)


def save_patch_selector_training_set_npz(samples: PatchSelectorTrainingSet, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _SAMPLE_CACHE_FORMAT,
        "sample_count": int(samples.sample_count),
        "input_dim": int(samples.input_dim),
        "positive_count": int(samples.positive_features.shape[1]),
        "negative_count": int(samples.negative_features.shape[1]),
        "metadata": {
            **dict(samples.metadata or {}),
            "has_reprojection_distances": bool(samples.positive_reprojection_distances is not None),
        },
    }
    arrays = dict(
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
        query_features=samples.query_features.astype(np.float32, copy=False),
        positive_features=samples.positive_features.astype(np.float32, copy=False),
        positive_mask=samples.positive_mask.astype(bool, copy=False),
        negative_features=samples.negative_features.astype(np.float32, copy=False),
    )
    if samples.positive_reprojection_distances is not None:
        arrays["positive_reprojection_distances"] = samples.positive_reprojection_distances.astype(np.float32, copy=False)
        arrays["negative_reprojection_distances"] = samples.negative_reprojection_distances.astype(np.float32, copy=False)
    np.savez_compressed(output, **arrays)


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
            positive_reprojection_distances=np.asarray(data["positive_reprojection_distances"], dtype=np.float32)
            if "positive_reprojection_distances" in data
            else None,
            negative_reprojection_distances=np.asarray(data["negative_reprojection_distances"], dtype=np.float32)
            if "negative_reprojection_distances" in data
            else None,
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
        positive_reprojection_distances=None
        if samples.positive_reprojection_distances is None
        else samples.positive_reprojection_distances[indices],
        negative_reprojection_distances=None
        if samples.negative_reprojection_distances is None
        else samples.negative_reprojection_distances[indices],
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


def _anchor_tensors(
    transform: FeatureCompressionTransform | None,
    device: torch.device,
    input_dim: int,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if transform is None:
        return None
    if transform.matrix is None:
        raise ValueError("anchor transform must contain a projection matrix")
    if int(transform.input_dim) != int(input_dim):
        raise ValueError("anchor transform input_dim must match sample input_dim")
    if int(transform.output_dim) != int(output_dim):
        raise ValueError("anchor transform output_dim must match selector output_dim")
    if not bool(transform.l2_normalize):
        raise ValueError("anchor transform must use L2 normalization")
    mean = torch.as_tensor(transform.mean.reshape(1, -1), dtype=torch.float32, device=device)
    matrix = torch.as_tensor(transform.matrix, dtype=torch.float32, device=device)
    return mean, matrix


def _apply_anchor_transform(features: torch.Tensor, anchor: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, matrix = anchor
    return F.normalize((features - mean.to(features.device)) @ matrix.to(features.device), dim=-1, eps=1e-8)


def _effective_lr(value: float | None, fallback: float) -> float:
    return float(fallback if value is None else value)


def _safe_selector_optimizer(selector: ResidualGatedPatchSelector | TransformerGroupPatchSelector, config: SafePatchSelectorTrainingConfig) -> torch.optim.Optimizer:
    projection_lr = _effective_lr(config.projection_lr, config.lr)
    residual_lr = _effective_lr(config.residual_lr, config.lr)
    pairwise_lr = _effective_lr(config.pairwise_lr, config.lr)
    gate_lr = _effective_lr(config.gate_lr, config.lr)
    param_groups = []
    if isinstance(selector, TransformerGroupPatchSelector):
        projection_params = (
            list(selector.input_norm.parameters())
            + list(selector.group_projection.parameters())
            + [selector.cls_token, selector.position_embedding]
            + list(selector.encoder.parameters())
            + list(selector.output_norm.parameters())
            + list(selector.projection.parameters())
        )
        residual_params = []
    else:
        projection_params = list(selector.input_norm.parameters()) + list(selector.projection.parameters())
        residual_params = list(selector.residual.parameters())
    if projection_params:
        param_groups.append({"params": projection_params, "lr": projection_lr, "weight_decay": 0.0 if projection_lr == 0.0 else 1e-4})
    if residual_params:
        param_groups.append({"params": residual_params, "lr": residual_lr, "weight_decay": 0.0 if residual_lr == 0.0 else 1e-4})
    pairwise_params = (
        list(selector.query_matchability.parameters())
        + list(selector.landmark_reliability.parameters())
        + list(selector.pairwise_inlier.parameters())
    )
    if pairwise_params:
        param_groups.append({"params": pairwise_params, "lr": pairwise_lr, "weight_decay": 0.0 if pairwise_lr == 0.0 else 1e-4})
    param_groups.append({"params": [selector.group_logits], "lr": gate_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups)


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
    anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
    anchor_loss_weight: float = 0.0,
    positive_reprojection_distances: torch.Tensor | None = None,
    negative_reprojection_distances: torch.Tensor | None = None,
    reprojection_margin_loss_weight: float = 0.0,
    reprojection_margin_max: float = 2.0,
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
    if float(reprojection_margin_loss_weight) > 0.0:
        if positive_reprojection_distances is None or negative_reprojection_distances is None:
            raise ValueError("reprojection margin loss requires positive and negative reprojection distances")
        pos_scores = torch.einsum("bd,bpd->bp", query_z, positive_z)
        neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
        pos_dist = torch.nan_to_num(positive_reprojection_distances.to(pos_scores.device), nan=0.0, posinf=float(reprojection_margin_max))
        neg_dist = torch.nan_to_num(negative_reprojection_distances.to(neg_scores.device), nan=float(reprojection_margin_max), posinf=float(reprojection_margin_max))
        margins = torch.clamp(neg_dist[:, None, :] - pos_dist[:, :, None], min=0.0, max=float(reprojection_margin_max))
        pair_losses = F.relu(neg_scores[:, None, :] - pos_scores[:, :, None] + margins)
        valid = positive_mask[:, :, None].expand_as(pair_losses)
        if bool(valid.any().item()):
            loss = loss + float(reprojection_margin_loss_weight) * pair_losses[valid].mean()
    if anchor is not None and float(anchor_loss_weight) > 0.0:
        anchor_query = _apply_anchor_transform(query, anchor)
        anchor_positive = _apply_anchor_transform(positives.reshape(-1, positives.shape[-1]), anchor).reshape(
            positives.shape[0],
            positives.shape[1],
            -1,
        )
        anchor_negative = _apply_anchor_transform(negatives.reshape(-1, negatives.shape[-1]), anchor).reshape(
            negatives.shape[0],
            negatives.shape[1],
            -1,
        )
        anchor_terms = [1.0 - torch.sum(query_z * anchor_query, dim=-1)]
        valid_positive = positive_mask.reshape(-1)
        if bool(valid_positive.any().item()):
            anchor_terms.append(1.0 - torch.sum(positive_z.reshape(-1, query_z.shape[-1])[valid_positive] * anchor_positive.reshape(-1, query_z.shape[-1])[valid_positive], dim=-1))
        anchor_terms.append(1.0 - torch.sum(negative_z.reshape(-1, query_z.shape[-1]) * anchor_negative.reshape(-1, query_z.shape[-1]), dim=-1))
        loss = loss + float(anchor_loss_weight) * torch.cat([term.reshape(-1) for term in anchor_terms], dim=0).mean()
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
    selector: ResidualGatedPatchSelector | TransformerGroupPatchSelector,
    keep_fraction: float,
    min_groups: int,
) -> np.ndarray:
    gates = selector.group_gates().detach().cpu().numpy().astype(np.float32)
    if hasattr(selector, "group_score_estimates"):
        scores = np.asarray(selector.group_score_estimates(), dtype=np.float32)
    else:
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
    *,
    init_transform: FeatureCompressionTransform | None = None,
    anchor_transform: FeatureCompressionTransform | None = None,
    init_run: SafePatchSelectorTrainingRun | None = None,
) -> SafePatchSelectorTrainingRun:
    config = config or SafePatchSelectorTrainingConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    if config.output_dim > samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    if init_transform is not None:
        if str(config.selector_arch) != "residual_gated":
            raise ValueError("init_transform is only supported by residual_gated selector")
        if init_transform.matrix is None:
            raise ValueError("init transform must contain a projection matrix")
        if int(init_transform.input_dim) != samples.input_dim:
            raise ValueError("init transform input_dim must match sample input_dim")
        if int(init_transform.output_dim) != int(config.output_dim):
            raise ValueError("init transform output_dim must match config output_dim")
    if init_run is not None:
        if int(init_run.summary.input_dim) != samples.input_dim:
            raise ValueError("init_run input_dim must match sample input_dim")
        if int(init_run.summary.output_dim) != int(config.output_dim):
            raise ValueError("init_run output_dim must match config output_dim")
        if str(init_run.summary.selector_arch) != str(config.selector_arch):
            raise ValueError("init_run selector_arch must match config selector_arch")
        if int(init_run.summary.residual_hidden_dim) != int(config.residual_hidden_dim):
            raise ValueError("init_run residual_hidden_dim must match config residual_hidden_dim")
        if str(config.selector_arch) == "transformer_group_token":
            if int(init_run.summary.transformer_dim) != int(config.transformer_dim):
                raise ValueError("init_run transformer_dim must match config transformer_dim")
            if int(init_run.summary.transformer_layers) != int(config.transformer_layers):
                raise ValueError("init_run transformer_layers must match config transformer_layers")
            if int(init_run.summary.transformer_heads) != int(config.transformer_heads):
                raise ValueError("init_run transformer_heads must match config transformer_heads")
            if int(init_run.summary.transformer_ff_dim) != int(config.transformer_ff_dim):
                raise ValueError("init_run transformer_ff_dim must match config transformer_ff_dim")
    if anchor_transform is None and float(config.anchor_loss_weight) > 0.0:
        anchor_transform = init_transform
    if float(config.reprojection_margin_loss_weight) > 0.0 and (
        samples.positive_reprojection_distances is None or samples.negative_reprojection_distances is None
    ):
        raise ValueError("reprojection_margin_loss_weight requires sample reprojection distances")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, config.eval_split_fraction, config.seed)
    train_samples = _subset_samples(samples, train_idx)
    eval_samples = _subset_samples(samples, eval_idx) if eval_idx.size else _subset_samples(samples, train_idx[:0])
    if init_transform is not None:
        input_mean = init_transform.mean.astype(np.float32, copy=False)
    else:
        input_mean = train_samples.query_features.mean(axis=0) if config.center_inputs else np.zeros((samples.input_dim,), dtype=np.float32)
    if str(config.selector_arch) == "transformer_group_token":
        selector = TransformerGroupPatchSelector(
            samples.input_dim,
            config.output_dim,
            group_size=int(config.group_size),
            transformer_dim=int(config.transformer_dim),
            transformer_layers=int(config.transformer_layers),
            transformer_heads=int(config.transformer_heads),
            transformer_ff_dim=int(config.transformer_ff_dim),
            transformer_dropout=float(config.transformer_dropout),
            pairwise_hidden_dim=int(config.residual_hidden_dim),
            input_mean=input_mean,
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
        ).to(device)
    else:
        selector = ResidualGatedPatchSelector(
            samples.input_dim,
            config.output_dim,
            residual_hidden_dim=int(config.residual_hidden_dim),
            group_size=int(config.group_size),
            input_mean=input_mean,
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
        ).to(device)
    if init_run is not None:
        selector.load_state_dict(init_run.model.state_dict())
    elif init_transform is not None:
        initialize_safe_selector_from_linear_transform(selector, init_transform)
    optimizer = _safe_selector_optimizer(selector, config)
    anchor = _anchor_tensors(anchor_transform, device, samples.input_dim, config.output_dim)
    rng = np.random.default_rng(int(config.seed))

    def loss_for_subset(subset: PatchSelectorTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        total_loss = 0.0
        total_count = 0
        eval_batch_size = min(max(int(config.batch_size), 1), 1024)
        with torch.no_grad():
            for start in range(0, subset.sample_count, eval_batch_size):
                end = min(start + eval_batch_size, subset.sample_count)
                query = torch.as_tensor(subset.query_features[start:end], dtype=torch.float32, device=device)
                positives = torch.as_tensor(subset.positive_features[start:end], dtype=torch.float32, device=device)
                mask = torch.as_tensor(subset.positive_mask[start:end], dtype=torch.bool, device=device)
                negatives = torch.as_tensor(subset.negative_features[start:end], dtype=torch.float32, device=device)
                pos_distances = (
                    None
                    if subset.positive_reprojection_distances is None
                    else torch.as_tensor(subset.positive_reprojection_distances[start:end], dtype=torch.float32, device=device)
                )
                neg_distances = (
                    None
                    if subset.negative_reprojection_distances is None
                    else torch.as_tensor(subset.negative_reprojection_distances[start:end], dtype=torch.float32, device=device)
                )
                batch_loss = _safe_selector_loss(
                    selector,
                    query,
                    positives,
                    mask,
                    negatives,
                    config.temperature,
                    config.inlier_loss_weight,
                    anchor=anchor,
                    anchor_loss_weight=config.anchor_loss_weight,
                    positive_reprojection_distances=pos_distances,
                    negative_reprojection_distances=neg_distances,
                    reprojection_margin_loss_weight=config.reprojection_margin_loss_weight,
                    reprojection_margin_max=config.reprojection_margin_max,
                )
                batch_count = int(end - start)
                total_loss += float(batch_loss.detach().cpu()) * batch_count
                total_count += batch_count
        return float(total_loss / max(total_count, 1))

    initial_loss = loss_for_subset(train_samples)
    for _step in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_samples.sample_count)
        batch_idx = rng.choice(train_samples.sample_count, size=batch_count, replace=False)
        query = torch.as_tensor(train_samples.query_features[batch_idx], dtype=torch.float32, device=device)
        positives = torch.as_tensor(train_samples.positive_features[batch_idx], dtype=torch.float32, device=device)
        mask = torch.as_tensor(train_samples.positive_mask[batch_idx], dtype=torch.bool, device=device)
        negatives = torch.as_tensor(train_samples.negative_features[batch_idx], dtype=torch.float32, device=device)
        pos_distances = (
            None
            if train_samples.positive_reprojection_distances is None
            else torch.as_tensor(train_samples.positive_reprojection_distances[batch_idx], dtype=torch.float32, device=device)
        )
        neg_distances = (
            None
            if train_samples.negative_reprojection_distances is None
            else torch.as_tensor(train_samples.negative_reprojection_distances[batch_idx], dtype=torch.float32, device=device)
        )
        loss = _safe_selector_loss(
            selector,
            query,
            positives,
            mask,
            negatives,
            config.temperature,
            config.inlier_loss_weight,
            anchor=anchor,
            anchor_loss_weight=config.anchor_loss_weight,
            positive_reprojection_distances=pos_distances,
            negative_reprojection_distances=neg_distances,
            reprojection_margin_loss_weight=config.reprojection_margin_loss_weight,
            reprojection_margin_max=config.reprojection_margin_max,
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
            selector_arch=str(config.selector_arch),
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
            transformer_dim=int(config.transformer_dim) if str(config.selector_arch) == "transformer_group_token" else 0,
            transformer_layers=int(config.transformer_layers) if str(config.selector_arch) == "transformer_group_token" else 0,
            transformer_heads=int(config.transformer_heads) if str(config.selector_arch) == "transformer_group_token" else 0,
            transformer_ff_dim=int(config.transformer_ff_dim) if str(config.selector_arch) == "transformer_group_token" else 0,
            transformer_dropout=float(config.transformer_dropout) if str(config.selector_arch) == "transformer_group_token" else 0.0,
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
            projection_lr=_effective_lr(config.projection_lr, config.lr),
            residual_lr=_effective_lr(config.residual_lr, config.lr),
            pairwise_lr=_effective_lr(config.pairwise_lr, config.lr),
            gate_lr=_effective_lr(config.gate_lr, config.lr),
            anchor_loss_weight=float(config.anchor_loss_weight),
            reprojection_margin_loss_weight=float(config.reprojection_margin_loss_weight),
            reprojection_margin_max=float(config.reprojection_margin_max),
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
            initialized_from_safe_checkpoint=bool(init_run is not None),
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
            "selector_arch": str(getattr(model, "selector_arch", "residual_gated")),
            "input_dim": int(model.input_dim),
            "output_dim": int(model.output_dim),
            "residual_hidden_dim": int(model.residual_hidden_dim),
            "group_size": int(model.group_size),
            "input_mean": model.input_mean.detach().cpu().numpy().reshape(-1),
            "input_norm_mode": str(model.input_norm_mode),
            "gate_mode": str(model.gate_mode),
            "residual_gate_scale": float(model.residual_gate_scale),
        },
        "state_dict": model.state_dict(),
        "summary": asdict(run.summary),
        "active_group_mask": np.asarray(run.active_group_mask, dtype=np.float32),
    }
    if isinstance(model, TransformerGroupPatchSelector):
        payload["model_config"].update(
            {
                "transformer_dim": int(model.transformer_dim),
                "transformer_layers": int(model.transformer_layers),
                "transformer_heads": int(model.transformer_heads),
                "transformer_ff_dim": int(model.transformer_ff_dim),
                "transformer_dropout": float(model.transformer_dropout),
            }
        )
    torch.save(payload, output)


def load_safe_patch_selector_checkpoint(path: Path, device: str = "cpu") -> SafePatchSelectorTrainingRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "vfm_stage_c2_safe_patch_selector_v1":
        raise ValueError(f"unsupported safe patch selector checkpoint format in {path}")
    config = dict(payload["model_config"])
    selector_arch = str(config.get("selector_arch", "residual_gated"))
    if selector_arch == "transformer_group_token":
        model = TransformerGroupPatchSelector(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            group_size=int(config["group_size"]),
            transformer_dim=int(config["transformer_dim"]),
            transformer_layers=int(config["transformer_layers"]),
            transformer_heads=int(config["transformer_heads"]),
            transformer_ff_dim=int(config["transformer_ff_dim"]),
            transformer_dropout=float(config.get("transformer_dropout", 0.0)),
            pairwise_hidden_dim=int(config["residual_hidden_dim"]),
            input_mean=np.asarray(config["input_mean"], dtype=np.float32),
            input_norm_mode=str(config.get("input_norm_mode", "layernorm")),
            gate_mode=str(config.get("gate_mode", "sigmoid")),
            residual_gate_scale=float(config.get("residual_gate_scale", 0.1)),
        )
    else:
        model = ResidualGatedPatchSelector(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            residual_hidden_dim=int(config["residual_hidden_dim"]),
            group_size=int(config["group_size"]),
            input_mean=np.asarray(config["input_mean"], dtype=np.float32),
            input_norm_mode=str(config.get("input_norm_mode", "layernorm")),
            gate_mode=str(config.get("gate_mode", "sigmoid")),
            residual_gate_scale=float(config.get("residual_gate_scale", 0.1)),
        )
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.device(device)).eval()
    summary_payload = dict(payload["summary"])
    summary_payload.setdefault("selector_arch", selector_arch)
    summary_payload.setdefault("projection_lr", 5e-4)
    summary_payload.setdefault("residual_lr", 5e-4)
    summary_payload.setdefault("pairwise_lr", 5e-4)
    summary_payload.setdefault("gate_lr", 5e-4)
    summary_payload.setdefault("anchor_loss_weight", 0.0)
    summary_payload.setdefault("reprojection_margin_loss_weight", 0.0)
    summary_payload.setdefault("reprojection_margin_max", 2.0)
    summary_payload.setdefault("input_norm_mode", str(config.get("input_norm_mode", "layernorm")))
    summary_payload.setdefault("gate_mode", str(config.get("gate_mode", "sigmoid")))
    summary_payload.setdefault("residual_gate_scale", float(config.get("residual_gate_scale", 0.1)))
    summary_payload.setdefault("initialized_from_safe_checkpoint", False)
    summary_payload.setdefault("transformer_dim", int(config.get("transformer_dim", 0)))
    summary_payload.setdefault("transformer_layers", int(config.get("transformer_layers", 0)))
    summary_payload.setdefault("transformer_heads", int(config.get("transformer_heads", 0)))
    summary_payload.setdefault("transformer_ff_dim", int(config.get("transformer_ff_dim", 0)))
    summary_payload.setdefault("transformer_dropout", float(config.get("transformer_dropout", 0.0)))
    summary = SafePatchSelectorTrainingSummary(**summary_payload)
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
