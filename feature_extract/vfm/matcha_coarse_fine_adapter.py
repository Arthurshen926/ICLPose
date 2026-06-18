"""Lightweight MATCHA-style coarse-to-fine descriptor and offset adapter."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.matcha_coarse_supervision import MatchaCoarseSupervision
from feature_extract.vfm.patch_selector_training import ResidualGatedPatchSelector
from feature_extract.vfm.query_to_3d_matching import normalize_rows


_FORMAT = "vfm_matcha_coarse_fine_adapter_v1"


@dataclass(frozen=True)
class MatchaCoarseFineTrainingSet:
    query_features: np.ndarray
    render_features: np.ndarray
    query_offset_labels: np.ndarray
    render_offset_labels: np.ndarray
    negative_render_features: np.ndarray
    roundtrip_errors_px: np.ndarray | None = None
    query_keypoint_features: np.ndarray | None = None
    query_keypoint_labels: np.ndarray | None = None
    render_keypoint_features: np.ndarray | None = None
    render_keypoint_labels: np.ndarray | None = None
    query_offset_soft_labels: np.ndarray | None = None
    render_offset_soft_labels: np.ndarray | None = None
    sample_confidence_targets: np.ndarray | None = None
    sample_uncertainty_px: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        query = np.asarray(self.query_features, dtype=np.float32)
        render = np.asarray(self.render_features, dtype=np.float32)
        qlabels = np.asarray(self.query_offset_labels, dtype=np.int64).reshape(-1)
        rlabels = np.asarray(self.render_offset_labels, dtype=np.int64).reshape(-1)
        negatives = np.asarray(self.negative_render_features, dtype=np.float32)
        roundtrip = (
            np.zeros((query.shape[0],), dtype=np.float32)
            if self.roundtrip_errors_px is None
            else np.asarray(self.roundtrip_errors_px, dtype=np.float32).reshape(-1)
        )
        qkp_features = (
            np.zeros((0, query.shape[1]), dtype=np.float32)
            if self.query_keypoint_features is None
            else np.asarray(self.query_keypoint_features, dtype=np.float32)
        )
        rkp_features = (
            np.zeros((0, query.shape[1]), dtype=np.float32)
            if self.render_keypoint_features is None
            else np.asarray(self.render_keypoint_features, dtype=np.float32)
        )
        qkp_labels = (
            np.zeros((qkp_features.shape[0],), dtype=np.int64)
            if self.query_keypoint_labels is None
            else np.asarray(self.query_keypoint_labels, dtype=np.int64).reshape(-1)
        )
        rkp_labels = (
            np.zeros((rkp_features.shape[0],), dtype=np.int64)
            if self.render_keypoint_labels is None
            else np.asarray(self.render_keypoint_labels, dtype=np.int64).reshape(-1)
        )
        qsoft = None if self.query_offset_soft_labels is None else np.asarray(self.query_offset_soft_labels, dtype=np.float32)
        rsoft = None if self.render_offset_soft_labels is None else np.asarray(self.render_offset_soft_labels, dtype=np.float32)
        confidence = None if self.sample_confidence_targets is None else np.asarray(self.sample_confidence_targets, dtype=np.float32).reshape(-1)
        uncertainty = None if self.sample_uncertainty_px is None else np.asarray(self.sample_uncertainty_px, dtype=np.float32).reshape(-1)
        if query.ndim != 2 or render.ndim != 2:
            raise ValueError("query_features and render_features must have shape (N, C)")
        if query.shape != render.shape:
            raise ValueError("query_features and render_features must have the same shape")
        if negatives.ndim != 3 or negatives.shape[0] != query.shape[0] or negatives.shape[2] != query.shape[1]:
            raise ValueError("negative_render_features must have shape (N, K, C)")
        if qlabels.shape[0] != query.shape[0] or rlabels.shape[0] != query.shape[0]:
            raise ValueError("offset labels must have one value per sample")
        if roundtrip.shape[0] != query.shape[0]:
            raise ValueError("roundtrip_errors_px must have one value per sample")
        if np.any(qlabels < 0) or np.any(qlabels > 64) or np.any(rlabels < 0) or np.any(rlabels > 64):
            raise ValueError("offset labels must be in [0, 64]")
        for name, value in (("query_offset_soft_labels", qsoft), ("render_offset_soft_labels", rsoft)):
            if value is None:
                continue
            if value.shape != (query.shape[0], 65):
                raise ValueError(f"{name} must have shape (N, 65)")
        for name, value in (("sample_confidence_targets", confidence), ("sample_uncertainty_px", uncertainty)):
            if value is None:
                continue
            if value.shape[0] != query.shape[0]:
                raise ValueError(f"{name} must contain one value per sample")
        for name, features, labels in (
            ("query", qkp_features, qkp_labels),
            ("render", rkp_features, rkp_labels),
        ):
            if features.ndim != 2 or features.shape[1] != query.shape[1]:
                raise ValueError(f"{name}_keypoint_features must have shape (M, C)")
            if labels.shape[0] != features.shape[0]:
                raise ValueError(f"{name}_keypoint_labels must have one value per feature")
            if labels.size and (np.any(labels < 0) or np.any(labels > 64)):
                raise ValueError(f"{name}_keypoint_labels must be in [0, 64]")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "render_features", render)
        object.__setattr__(self, "query_offset_labels", qlabels)
        object.__setattr__(self, "render_offset_labels", rlabels)
        object.__setattr__(self, "negative_render_features", negatives)
        object.__setattr__(self, "roundtrip_errors_px", roundtrip)
        object.__setattr__(self, "query_keypoint_features", qkp_features)
        object.__setattr__(self, "query_keypoint_labels", qkp_labels)
        object.__setattr__(self, "render_keypoint_features", rkp_features)
        object.__setattr__(self, "render_keypoint_labels", rkp_labels)
        object.__setattr__(self, "query_offset_soft_labels", qsoft)
        object.__setattr__(self, "render_offset_soft_labels", rsoft)
        object.__setattr__(self, "sample_confidence_targets", None if confidence is None else np.clip(confidence, 0.0, 1.0))
        object.__setattr__(self, "sample_uncertainty_px", uncertainty)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def sample_count(self) -> int:
        return int(self.query_features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.query_features.shape[1]) if self.query_features.ndim == 2 else 0


@dataclass(frozen=True)
class MatchaCoarseFineTrainingConfig:
    output_dim: int = 128
    residual_hidden_dim: int = 256
    steps: int = 300
    batch_size: int = 512
    lr: float = 5e-5
    temperature: float = 0.07
    dual_softmax_weight: float = 1.0
    offset_loss_weight: float = 0.25
    pair_fine_loss_weight: float = 0.25
    confidence_loss_weight: float = 0.1
    detector_loss_weight: float = 0.05
    detector_target_mode: str = "matcha_confidence"
    keypoint_loss_weight: float = 0.0
    hard_negative_weight: float = 0.2
    hard_negative_margin: float = 0.2
    eval_split_fraction: float = 0.1
    group_size: int = 64
    input_norm_mode: str = "identity"
    gate_mode: str = "residual"
    residual_gate_scale: float = 0.1
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if int(self.residual_hidden_dim) <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("batch_size must be greater than one")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        for name in (
            "dual_softmax_weight",
            "offset_loss_weight",
            "pair_fine_loss_weight",
            "confidence_loss_weight",
            "detector_loss_weight",
            "keypoint_loss_weight",
            "hard_negative_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.hard_negative_margin) < 0.0:
            raise ValueError("hard_negative_margin must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")
        if str(self.detector_target_mode) not in {"matcha_confidence", "hard_negative_bce"}:
            raise ValueError("detector_target_mode must be 'matcha_confidence' or 'hard_negative_bce'")


@dataclass
class MatchaCoarseFineTrainingRun:
    model: "MatchaCoarseFineAdapter"
    summary: dict[str, object]


class _OriginalMatchaFineMatcher(nn.Module):
    """MATCHA-style 64-bin coordinate classifier for a matched descriptor pair."""

    def __init__(self, descriptor_dim: int, hidden_dim: int, output_bins: int = 64) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(descriptor_dim) * 2, hidden),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(output_bins)),
        )

    def forward(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        if query_descriptors.shape != render_descriptors.shape:
            raise ValueError("query_descriptors and render_descriptors must have matching shape")
        return self.net(torch.cat([query_descriptors, render_descriptors], dim=-1))


class MatchaCoarseFineAdapter(nn.Module):
    """Residual gated descriptor selector plus MATCHA-style 65-way offset head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        residual_hidden_dim: int = 256,
        group_size: int = 64,
        input_mean: np.ndarray | None = None,
        input_norm_mode: str = "identity",
        gate_mode: str = "residual",
        residual_gate_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.group_size = int(group_size)
        self.input_norm_mode = str(input_norm_mode)
        self.gate_mode = str(gate_mode)
        self.residual_gate_scale = float(residual_gate_scale)
        self.selector = ResidualGatedPatchSelector(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            residual_hidden_dim=int(residual_hidden_dim),
            group_size=int(group_size),
            input_mean=input_mean,
            input_norm_mode=str(input_norm_mode),
            gate_mode=str(gate_mode),
            residual_gate_scale=float(residual_gate_scale),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 65),
        )
        self.detector_head = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.keypoint_head = nn.Sequential(
            nn.LayerNorm(int(output_dim)),
            nn.Linear(int(output_dim), int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 65),
        )
        pair_dim = int(output_dim) * 4
        self.pair_confidence_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.pair_fine_uncertainty_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.pair_fine_head = _OriginalMatchaFineMatcher(
            descriptor_dim=int(output_dim),
            hidden_dim=int(residual_hidden_dim),
            output_bins=64,
        )

    @property
    def input_mean(self) -> torch.Tensor:
        return self.selector.input_mean

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.selector(features)

    def offset_logits_from_descriptor(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.offset_head(descriptors)

    def detector_logits_from_descriptor(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.detector_head(descriptors).squeeze(-1)

    def keypoint_logits_from_descriptor(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.keypoint_head(descriptors)

    def keypoint_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.keypoint_logits_from_descriptor(self.encode(features))

    @staticmethod
    def pair_features(query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        if query_descriptors.shape != render_descriptors.shape:
            raise ValueError("query_descriptors and render_descriptors must have matching shape")
        return torch.cat(
            [
                query_descriptors,
                render_descriptors,
                torch.abs(query_descriptors - render_descriptors),
                query_descriptors * render_descriptors,
            ],
            dim=-1,
        )

    def pair_confidence_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.pair_confidence_head(self.pair_features(query_descriptors, render_descriptors)).squeeze(-1)

    def pair_fine_uncertainty_log_sigma(
        self,
        query_descriptors: torch.Tensor,
        render_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return self.pair_fine_uncertainty_head(self.pair_features(query_descriptors, render_descriptors)).squeeze(-1)

    def pair_fine_logits(self, query_descriptors: torch.Tensor, render_descriptors: torch.Tensor) -> torch.Tensor:
        return self.pair_fine_head(query_descriptors, render_descriptors)

    def offset_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.offset_logits_from_descriptor(self.encode(features))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptors = self.encode(features)
        return descriptors, self.offset_logits_from_descriptor(descriptors)

    def forward_full(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptors = self.encode(features)
        return (
            descriptors,
            self.offset_logits_from_descriptor(descriptors),
            self.detector_logits_from_descriptor(descriptors),
        )


def _flatten_feature_map(feature_map: np.ndarray) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    return fmap.reshape(channels, height * width).T.astype(np.float32, copy=False)


def _mine_negative_render_features(
    query_features: np.ndarray,
    render_features: np.ndarray,
    positive_render_indices: np.ndarray,
    *,
    count: int,
    excluded_render_indices_by_query: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    qnorm, qvalid = normalize_rows(query_features)
    rnorm, rvalid = normalize_rows(render_features)
    positives = np.asarray(positive_render_indices, dtype=np.int64).reshape(-1)
    if positives.shape[0] != query_features.shape[0]:
        raise ValueError("positive_render_indices must contain one index per query feature")
    if excluded_render_indices_by_query is not None and len(excluded_render_indices_by_query) != query_features.shape[0]:
        raise ValueError("excluded_render_indices_by_query must contain one exclusion set per query feature")
    scores = qnorm @ rnorm.T
    scores[~qvalid, :] = -np.inf
    scores[:, ~rvalid] = -np.inf
    output = np.zeros((query_features.shape[0], int(count), render_features.shape[1]), dtype=np.float32)
    for row in range(query_features.shape[0]):
        row_scores = scores[row].copy()
        row_scores[int(positives[row])] = -np.inf
        if excluded_render_indices_by_query is not None:
            excluded = np.asarray(excluded_render_indices_by_query[row], dtype=np.int64).reshape(-1)
            excluded = excluded[(excluded >= 0) & (excluded < row_scores.shape[0])]
            row_scores[excluded] = -np.inf
        order = np.argsort(-row_scores)
        order = order[np.isfinite(row_scores[order])]
        if order.size == 0:
            order = np.asarray([int(positives[row])], dtype=np.int64)
        if order.size < int(count):
            order = np.resize(order, int(count))
        output[row] = render_features[order[: int(count)]]
    return output


def _same_query_render_exclusion_sets(query_indices: np.ndarray, render_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    qidx = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    ridx = np.asarray(render_indices, dtype=np.int64).reshape(-1)
    if qidx.shape[0] != ridx.shape[0]:
        raise ValueError("query_indices and render_indices must have the same length")
    by_query: dict[int, list[int]] = {}
    for query_id, render_id in zip(qidx.tolist(), ridx.tolist()):
        if int(query_id) < 0 or int(render_id) < 0:
            continue
        by_query.setdefault(int(query_id), []).append(int(render_id))
    return tuple(np.unique(np.asarray(by_query.get(int(query_id), [int(render_id)]), dtype=np.int64)) for query_id, render_id in zip(qidx.tolist(), ridx.tolist()))


def _keypoint_training_rows(
    feature_rows: np.ndarray,
    label_map: np.ndarray | None,
    *,
    max_rows: int = 0,
    non_keypoint_divisor: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if label_map is None:
        return (
            np.zeros((0, feature_rows.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    labels = np.asarray(label_map, dtype=np.int64).reshape(-1)
    if labels.shape[0] != feature_rows.shape[0]:
        raise ValueError("keypoint label map must contain one label per feature cell")
    if labels.size and (np.any(labels < 0) or np.any(labels > 64)):
        raise ValueError("keypoint labels must be in [0, 64]")
    positive = np.flatnonzero(labels < 64)
    negative = np.flatnonzero(labels == 64)
    rng = np.random.default_rng(int(seed))
    negative_count = min(negative.shape[0], max(1, positive.shape[0] // max(int(non_keypoint_divisor), 1))) if positive.size else min(negative.shape[0], int(max_rows) if int(max_rows) > 0 else negative.shape[0])
    if negative_count > 0:
        negative = np.sort(rng.choice(negative, size=int(negative_count), replace=False))
    else:
        negative = np.zeros((0,), dtype=np.int64)
    keep = np.concatenate([positive, negative]).astype(np.int64, copy=False)
    if keep.size == 0:
        return (
            np.zeros((0, feature_rows.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    if int(max_rows) > 0 and keep.size > int(max_rows):
        keep = np.sort(rng.choice(keep, size=int(max_rows), replace=False))
    return feature_rows[keep].astype(np.float32, copy=False), labels[keep].astype(np.int64, copy=False)


def build_matcha_coarse_fine_training_set(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    supervision: MatchaCoarseSupervision,
    *,
    hard_negatives_per_match: int = 16,
    query_keypoint_label_map: np.ndarray | None = None,
    render_keypoint_label_map: np.ndarray | None = None,
    max_keypoint_rows: int = 4096,
    keypoint_nonkeypoint_divisor: int = 32,
    seed: int = 0,
) -> MatchaCoarseFineTrainingSet:
    """Convert coarse supervision into descriptor/offset training rows."""

    query_rows = _flatten_feature_map(query_feature_map)
    render_rows = _flatten_feature_map(render_feature_map)
    if query_rows.shape[1] != render_rows.shape[1]:
        raise ValueError("query and render feature dimensions must match")
    if supervision.count == 0:
        qkp_features, qkp_labels = _keypoint_training_rows(
            query_rows,
            query_keypoint_label_map,
            max_rows=int(max_keypoint_rows),
            non_keypoint_divisor=int(keypoint_nonkeypoint_divisor),
            seed=int(seed),
        )
        rkp_features, rkp_labels = _keypoint_training_rows(
            render_rows,
            render_keypoint_label_map,
            max_rows=int(max_keypoint_rows),
            non_keypoint_divisor=int(keypoint_nonkeypoint_divisor),
            seed=int(seed) + 1,
        )
        return MatchaCoarseFineTrainingSet(
            query_features=np.zeros((0, query_rows.shape[1]), dtype=np.float32),
            render_features=np.zeros((0, query_rows.shape[1]), dtype=np.float32),
            query_offset_labels=np.zeros((0,), dtype=np.int64),
            render_offset_labels=np.zeros((0,), dtype=np.int64),
            negative_render_features=np.zeros((0, int(hard_negatives_per_match), query_rows.shape[1]), dtype=np.float32),
            roundtrip_errors_px=np.zeros((0,), dtype=np.float32),
            query_keypoint_features=qkp_features,
            query_keypoint_labels=qkp_labels,
            render_keypoint_features=rkp_features,
            render_keypoint_labels=rkp_labels,
            metadata={"sample_count": 0, "supervision_source": str(getattr(supervision, "source", "geometry_depth_pose"))},
        )
    if np.max(supervision.query_indices, initial=-1) >= query_rows.shape[0]:
        raise ValueError("supervision query index exceeds query feature map size")
    if np.max(supervision.render_indices, initial=-1) >= render_rows.shape[0]:
        raise ValueError("supervision render index exceeds render feature map size")
    query = query_rows[supervision.query_indices]
    render = render_rows[supervision.render_indices]
    negatives = _mine_negative_render_features(
        query,
        render_rows,
        supervision.render_indices,
        count=int(hard_negatives_per_match),
        excluded_render_indices_by_query=_same_query_render_exclusion_sets(
            supervision.query_indices,
            supervision.render_indices,
        ),
    )
    qkp_features, qkp_labels = _keypoint_training_rows(
        query_rows,
        query_keypoint_label_map,
        max_rows=int(max_keypoint_rows),
        non_keypoint_divisor=int(keypoint_nonkeypoint_divisor),
        seed=int(seed),
    )
    rkp_features, rkp_labels = _keypoint_training_rows(
        render_rows,
        render_keypoint_label_map,
        max_rows=int(max_keypoint_rows),
        non_keypoint_divisor=int(keypoint_nonkeypoint_divisor),
        seed=int(seed) + 1,
    )
    return MatchaCoarseFineTrainingSet(
        query_features=query,
        render_features=render,
        query_offset_labels=supervision.query_offset_labels,
        render_offset_labels=supervision.render_offset_labels,
        negative_render_features=negatives,
        roundtrip_errors_px=supervision.roundtrip_errors_px,
        query_keypoint_features=qkp_features,
        query_keypoint_labels=qkp_labels,
        render_keypoint_features=rkp_features,
        render_keypoint_labels=rkp_labels,
        metadata={
            "source": "matcha_coarse_supervision",
            "supervision_source": str(getattr(supervision, "source", "geometry_depth_pose")),
            "sample_count": int(supervision.count),
            "hard_negatives_per_match": int(hard_negatives_per_match),
            "query_keypoint_sample_count": int(qkp_labels.shape[0]),
            "render_keypoint_sample_count": int(rkp_labels.shape[0]),
        },
    )


def _split_indices(count: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(count), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    eval_count = int(round(float(eval_fraction) * int(count)))
    eval_count = min(max(eval_count, 0), max(int(count) - 1, 0))
    return indices[eval_count:], indices[:eval_count]


def _subset(samples: MatchaCoarseFineTrainingSet, indices: np.ndarray) -> MatchaCoarseFineTrainingSet:
    idx = np.asarray(indices, dtype=np.int64)
    return MatchaCoarseFineTrainingSet(
        query_features=samples.query_features[idx],
        render_features=samples.render_features[idx],
        query_offset_labels=samples.query_offset_labels[idx],
        render_offset_labels=samples.render_offset_labels[idx],
        negative_render_features=samples.negative_render_features[idx],
        roundtrip_errors_px=samples.roundtrip_errors_px[idx],
        query_keypoint_features=samples.query_keypoint_features,
        query_keypoint_labels=samples.query_keypoint_labels,
        render_keypoint_features=samples.render_keypoint_features,
        render_keypoint_labels=samples.render_keypoint_labels,
        metadata=samples.metadata,
    )


def _subset_keypoint_rows(samples: MatchaCoarseFineTrainingSet, seed: int, max_rows: int = 0) -> MatchaCoarseFineTrainingSet:
    if int(max_rows) <= 0:
        return samples
    rng = np.random.default_rng(int(seed))

    def choose(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if features.shape[0] <= int(max_rows):
            return features, labels
        keep = np.sort(rng.choice(features.shape[0], size=int(max_rows), replace=False))
        return features[keep], labels[keep]

    qf, ql = choose(samples.query_keypoint_features, samples.query_keypoint_labels)
    rf, rl = choose(samples.render_keypoint_features, samples.render_keypoint_labels)
    return MatchaCoarseFineTrainingSet(
        query_features=samples.query_features,
        render_features=samples.render_features,
        query_offset_labels=samples.query_offset_labels,
        render_offset_labels=samples.render_offset_labels,
        negative_render_features=samples.negative_render_features,
        roundtrip_errors_px=samples.roundtrip_errors_px,
        query_keypoint_features=qf,
        query_keypoint_labels=ql,
        render_keypoint_features=rf,
        render_keypoint_labels=rl,
        metadata=samples.metadata,
    )


def _dual_softmax_descriptor_loss_and_confidence(
    query_z: torch.Tensor,
    render_z: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MATCHA-style dual-softmax descriptor loss and per-row confidence.

    The confidence mirrors MATCHA's heatmap supervision signal: it is the
    product of the best row-wise and column-wise matching probabilities. It is
    detached before being used as a reliability target, so the heatmap/reliability
    head follows the descriptor matcher instead of becoming a separate
    hard-negative classifier.
    """

    logits = query_z @ render_z.T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    log_prob_qr = F.log_softmax(logits, dim=1)
    log_prob_rq = F.log_softmax(logits.T, dim=1)
    loss = 0.5 * (F.nll_loss(log_prob_qr, labels) + F.nll_loss(log_prob_rq, labels))
    with torch.no_grad():
        confidence_q = torch.exp(log_prob_qr).max(dim=1)[0]
        confidence_r = torch.exp(log_prob_rq).max(dim=1)[0]
        confidence = (confidence_q * confidence_r).clamp(0.0, 1.0)
    return loss, confidence


def _dual_softmax_descriptor_loss(query_z: torch.Tensor, render_z: torch.Tensor, temperature: float) -> torch.Tensor:
    return _dual_softmax_descriptor_loss_and_confidence(query_z, render_z, float(temperature))[0]


def _fine_coordinate_loss_and_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    soft_targets: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    confidence_acc_threshold: float = 0.1,
    continuous_loss_weight: float = 0.0,
    uncertainty_log_sigma: torch.Tensor | None = None,
    uncertainty_loss_weight: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """64-bin coordinate classification loss for matched feature pairs.

    Labels use the existing offset convention: 0-63 are spatial bins and 64 is
    the dustbin/non-match value. Fine matcher supervision follows MATCHA by
    optimizing only the 64 spatial bins and, when supplied, weighting rows by
    detached dual-softmax match confidence.
    """

    if logits.ndim != 2:
        raise ValueError("fine matcher logits must have shape (N, C)")
    if int(logits.shape[1]) < 64:
        raise ValueError("fine matcher logits must contain at least 64 coordinate bins")
    target = labels.long().reshape(-1)
    if int(target.shape[0]) != int(logits.shape[0]):
        raise ValueError("fine matcher labels must contain one value per logit row")
    if soft_targets is not None:
        soft_targets = soft_targets.float()
        if soft_targets.shape != (int(logits.shape[0]), 65):
            raise ValueError("fine matcher soft_targets must have shape (N, 65)")
    if confidence is not None:
        confidence = confidence.detach().float().reshape(-1)
        if int(confidence.shape[0]) != int(logits.shape[0]):
            raise ValueError("fine matcher confidence must contain one value per logit row")
    if uncertainty_log_sigma is not None:
        uncertainty_log_sigma = uncertainty_log_sigma.float().reshape(-1)
        if int(uncertainty_log_sigma.shape[0]) != int(logits.shape[0]):
            raise ValueError("fine matcher uncertainty_log_sigma must contain one value per logit row")
    valid = (target >= 0) & (target < 64)
    valid_count = int(torch.count_nonzero(valid).detach().cpu().item())
    if valid_count == 0:
        return None, {"valid_count": 0.0, "acc": 0.0}
    spatial_logits = logits[:, :64]
    if soft_targets is None:
        per_row = F.cross_entropy(spatial_logits[valid], target[valid], reduction="none")
    else:
        spatial_target = soft_targets[valid, :64]
        spatial_target_sum = torch.sum(spatial_target, dim=1, keepdim=True).clamp_min(1e-8)
        spatial_target = spatial_target / spatial_target_sum
        per_row = -torch.sum(spatial_target * F.log_softmax(spatial_logits[valid], dim=1), dim=1)
    if confidence is None:
        loss = per_row.mean()
        acc_mask = valid
    else:
        valid_confidence = torch.clamp(confidence[valid], min=0.0)
        confidence_sum = torch.sum(valid_confidence)
        if float(confidence_sum.detach().cpu().item()) > 0.0:
            loss = torch.sum(per_row * (valid_confidence / confidence_sum))
        else:
            loss = per_row.mean()
        acc_mask = valid & (confidence > float(confidence_acc_threshold))
        if not torch.any(acc_mask):
            acc_mask = valid
    probs_for_loss = F.softmax(spatial_logits[valid], dim=1)
    bins = 8
    coords_for_loss = torch.arange(64, dtype=probs_for_loss.dtype, device=probs_for_loss.device)
    bin_x_for_loss = torch.remainder(coords_for_loss, bins) + 0.5
    bin_y_for_loss = torch.floor(coords_for_loss / bins) + 0.5
    expected_x_for_loss = torch.sum(probs_for_loss * bin_x_for_loss[None], dim=1)
    expected_y_for_loss = torch.sum(probs_for_loss * bin_y_for_loss[None], dim=1)
    target_valid_for_loss = target[valid].to(probs_for_loss.device)
    target_x_for_loss = torch.remainder(target_valid_for_loss, bins).to(probs_for_loss.dtype) + 0.5
    target_y_for_loss = torch.floor(target_valid_for_loss.to(probs_for_loss.dtype) / float(bins)) + 0.5
    continuous_epe = torch.sqrt(
        torch.clamp(
            (expected_x_for_loss - target_x_for_loss) ** 2 + (expected_y_for_loss - target_y_for_loss) ** 2,
            min=1e-12,
        )
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        if confidence is None:
            return torch.mean(values)
        valid_weights = torch.clamp(confidence[valid].to(values.device), min=0.0)
        weight_sum = torch.sum(valid_weights)
        if float(weight_sum.detach().cpu().item()) <= 0.0:
            return torch.mean(values)
        return torch.sum(values * (valid_weights / weight_sum))

    continuous_loss = weighted_mean(continuous_epe)
    if float(continuous_loss_weight) > 0.0:
        loss = loss + float(continuous_loss_weight) * continuous_loss
    uncertainty_nll = None
    learned_uncertainty = None
    if uncertainty_log_sigma is not None and float(uncertainty_loss_weight) > 0.0:
        log_sigma = torch.clamp(uncertainty_log_sigma[valid].to(continuous_epe.device), min=-5.0, max=5.0)
        sigma = torch.exp(log_sigma).clamp_min(1e-6)
        uncertainty_nll = weighted_mean(continuous_epe / sigma + log_sigma)
        learned_uncertainty = weighted_mean(sigma)
        loss = loss + float(uncertainty_loss_weight) * uncertainty_nll
    with torch.no_grad():
        predictions = torch.argmax(spatial_logits[acc_mask], dim=1)
        acc = float(torch.mean((predictions == target[acc_mask]).float()).detach().cpu().item())
        probs = F.softmax(spatial_logits[valid], dim=1)
        bins = 8
        coords = torch.arange(64, dtype=probs.dtype, device=probs.device)
        bin_x = torch.remainder(coords, bins) + 0.5
        bin_y = torch.floor(coords / bins) + 0.5
        expected_x = torch.sum(probs * bin_x[None], dim=1)
        expected_y = torch.sum(probs * bin_y[None], dim=1)
        target_valid = target[valid].to(probs.device)
        target_x = torch.remainder(target_valid, bins).to(probs.dtype) + 0.5
        target_y = torch.floor(target_valid.to(probs.dtype) / float(bins)) + 0.5
        epe = torch.sqrt((expected_x - target_x) ** 2 + (expected_y - target_y) ** 2)
        variance = torch.sum(probs * ((bin_x[None] - expected_x[:, None]) ** 2 + (bin_y[None] - expected_y[:, None]) ** 2), dim=1)
        uncertainty = torch.sqrt(torch.clamp(variance, min=0.0))
        if confidence is not None:
            valid_weights = torch.clamp(confidence[valid].to(probs.device), min=0.0)
            weight_sum = torch.sum(valid_weights)
            if float(weight_sum.detach().cpu().item()) > 0.0:
                epe_value = float(torch.sum(epe * (valid_weights / weight_sum)).detach().cpu().item())
                uncertainty_value = float(torch.sum(uncertainty * (valid_weights / weight_sum)).detach().cpu().item())
            else:
                epe_value = float(torch.mean(epe).detach().cpu().item())
                uncertainty_value = float(torch.mean(uncertainty).detach().cpu().item())
        else:
            epe_value = float(torch.mean(epe).detach().cpu().item())
            uncertainty_value = float(torch.mean(uncertainty).detach().cpu().item())
    metrics = {
        "valid_count": float(valid_count),
        "acc": acc,
        "epe_bins": epe_value,
        "uncertainty_bins": uncertainty_value,
        "continuous_epe_bins": float(continuous_loss.detach().cpu().item()),
    }
    if uncertainty_nll is not None:
        metrics["uncertainty_nll"] = float(uncertainty_nll.detach().cpu().item())
    if learned_uncertainty is not None:
        metrics["learned_uncertainty_bins"] = float(learned_uncertainty.detach().cpu().item())
    return loss, metrics


def _loss(
    model: MatchaCoarseFineAdapter,
    query: torch.Tensor,
    render: torch.Tensor,
    qlabels: torch.Tensor,
    rlabels: torch.Tensor,
    negatives: torch.Tensor,
    config: MatchaCoarseFineTrainingConfig,
    query_keypoint: torch.Tensor | None = None,
    query_keypoint_labels: torch.Tensor | None = None,
    render_keypoint: torch.Tensor | None = None,
    render_keypoint_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    query_z, query_offsets = model(query)
    render_z, render_offsets = model(render)
    loss = query.new_tensor(0.0)
    match_confidence = None
    if float(config.dual_softmax_weight) > 0.0:
        descriptor_loss, match_confidence = _dual_softmax_descriptor_loss_and_confidence(
            query_z,
            render_z,
            float(config.temperature),
        )
        loss = loss + float(config.dual_softmax_weight) * descriptor_loss
    if float(config.offset_loss_weight) > 0.0:
        offset_loss = 0.5 * (F.cross_entropy(query_offsets, qlabels) + F.cross_entropy(render_offsets, rlabels))
        loss = loss + float(config.offset_loss_weight) * offset_loss
    if float(config.pair_fine_loss_weight) > 0.0:
        if match_confidence is None:
            _descriptor_loss, match_confidence = _dual_softmax_descriptor_loss_and_confidence(
                query_z,
                render_z,
                float(config.temperature),
            )
        pair_fine = model.pair_fine_logits(query_z, render_z)
        pair_fine_loss, _metrics = _fine_coordinate_loss_and_metrics(
            pair_fine,
            rlabels,
            confidence=match_confidence,
        )
        if pair_fine_loss is not None:
            loss = loss + float(config.pair_fine_loss_weight) * pair_fine_loss
    if float(config.detector_loss_weight) > 0.0:
        qdet = model.detector_logits_from_descriptor(query_z)
        rdet = model.detector_logits_from_descriptor(render_z)
        if str(config.detector_target_mode) == "matcha_confidence":
            if match_confidence is None:
                _, match_confidence = _dual_softmax_descriptor_loss_and_confidence(
                    query_z,
                    render_z,
                    float(config.temperature),
                )
            target = match_confidence.detach()
            detector_loss = 0.5 * (
                F.l1_loss(torch.sigmoid(qdet), target)
                + F.l1_loss(torch.sigmoid(rdet), target)
            )
        else:
            positive_labels = torch.ones_like(qdet)
            detector_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(qdet, positive_labels)
                + F.binary_cross_entropy_with_logits(rdet, positive_labels)
            )
            if negatives.numel() > 0:
                negative_z_for_detector = model.encode(negatives.reshape(-1, negatives.shape[-1]))
                ndet = model.detector_logits_from_descriptor(negative_z_for_detector)
                detector_loss = detector_loss + F.binary_cross_entropy_with_logits(ndet, torch.zeros_like(ndet))
        loss = loss + float(config.detector_loss_weight) * detector_loss
    if float(config.hard_negative_weight) > 0.0 and negatives.numel() > 0:
        negative_z = model.encode(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
        pos_scores = torch.sum(query_z * render_z, dim=1, keepdim=True)
        neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
        hard_loss = torch.relu(neg_scores - pos_scores + float(config.hard_negative_margin)).mean()
        loss = loss + float(config.hard_negative_weight) * hard_loss
    if float(config.confidence_loss_weight) > 0.0 and negatives.numel() > 0:
        if "negative_z" not in locals():
            negative_z = model.encode(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
        pos_logits = model.pair_confidence_logits(query_z, render_z)
        neg_logits = model.pair_confidence_logits(
            query_z[:, None, :].expand_as(negative_z).reshape(-1, query_z.shape[-1]),
            negative_z.reshape(-1, query_z.shape[-1]),
        )
        confidence_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        confidence_loss = confidence_loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = loss + float(config.confidence_loss_weight) * confidence_loss
    if float(config.keypoint_loss_weight) > 0.0:
        keypoint_losses = []
        if query_keypoint is not None and query_keypoint_labels is not None and query_keypoint.numel() > 0:
            keypoint_losses.append(F.cross_entropy(model.keypoint_logits(query_keypoint), query_keypoint_labels))
        if render_keypoint is not None and render_keypoint_labels is not None and render_keypoint.numel() > 0:
            keypoint_losses.append(F.cross_entropy(model.keypoint_logits(render_keypoint), render_keypoint_labels))
        if keypoint_losses:
            loss = loss + float(config.keypoint_loss_weight) * torch.stack(keypoint_losses).mean()
    return loss


def _evaluate(model: MatchaCoarseFineAdapter, samples: MatchaCoarseFineTrainingSet, device: torch.device, batch_size: int) -> dict[str, float]:
    def empty_keypoint_metrics(prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_keypoint_acc": 0.0,
            f"{prefix}_keypoint_positive_acc": 0.0,
            f"{prefix}_keypoint_non_keypoint_acc": 0.0,
            f"{prefix}_keypoint_confidence_mean": 0.0,
            f"{prefix}_keypoint_positive_confidence_mean": 0.0,
            f"{prefix}_keypoint_non_keypoint_confidence_mean": 0.0,
            f"{prefix}_keypoint_positive_count": 0.0,
            f"{prefix}_keypoint_non_keypoint_count": 0.0,
        }

    if samples.sample_count == 0:
        metrics = {
            "loss": 0.0,
            "top1_acc": 0.0,
            "query_offset_acc": 0.0,
            "render_offset_acc": 0.0,
            "render_pair_fine_acc": 0.0,
            "render_pair_fine_valid_count": 0.0,
            "query_detector_mean": 0.0,
            "render_detector_mean": 0.0,
            "detector_mean_gap": 0.0,
        }
        metrics.update(empty_keypoint_metrics("query"))
        metrics.update(empty_keypoint_metrics("render"))
        return metrics
    model.eval()
    query_all = []
    render_all = []
    qoffset_ok = []
    roffset_ok = []
    qdetectors = []
    rdetectors = []
    render_pair_fine_ok = []
    render_pair_fine_valid_count = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, int(batch_size)):
            end = min(start + int(batch_size), samples.sample_count)
            query = torch.as_tensor(samples.query_features[start:end], dtype=torch.float32, device=device)
            render = torch.as_tensor(samples.render_features[start:end], dtype=torch.float32, device=device)
            qz, qlogits = model(query)
            rz, rlogits = model(render)
            query_all.append(qz.cpu())
            render_all.append(rz.cpu())
            qdetectors.append(torch.sigmoid(model.detector_logits_from_descriptor(qz)).detach().cpu())
            rdetectors.append(torch.sigmoid(model.detector_logits_from_descriptor(rz)).detach().cpu())
            qlabels = torch.as_tensor(samples.query_offset_labels[start:end], dtype=torch.long, device=device)
            rlabels = torch.as_tensor(samples.render_offset_labels[start:end], dtype=torch.long, device=device)
            qoffset_ok.append((torch.argmax(qlogits, dim=1) == qlabels).detach().cpu())
            roffset_ok.append((torch.argmax(rlogits, dim=1) == rlabels).detach().cpu())
            pair_logits = model.pair_fine_logits(qz, rz)
            valid_pair_fine = rlabels < 64
            render_pair_fine_valid_count += int(torch.count_nonzero(valid_pair_fine).item())
            if torch.any(valid_pair_fine):
                render_pair_fine_ok.append(
                    (torch.argmax(pair_logits[valid_pair_fine, :64], dim=1) == rlabels[valid_pair_fine]).detach().cpu()
                )
    query_z = torch.cat(query_all, dim=0)
    render_z = torch.cat(render_all, dim=0)
    scores = query_z @ render_z.T
    top1 = torch.argmax(scores, dim=1)
    labels = torch.arange(scores.shape[0])
    query_detector = torch.cat(qdetectors, dim=0)
    render_detector = torch.cat(rdetectors, dim=0)
    def keypoint_metrics(prefix: str, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        if features.shape[0] == 0:
            return empty_keypoint_metrics(prefix)
        correct = []
        positive_correct = []
        non_keypoint_correct = []
        confidences = []
        positive_confidences = []
        non_keypoint_confidences = []
        positive_count = 0
        non_keypoint_count = 0
        with torch.no_grad():
            for start in range(0, features.shape[0], int(batch_size)):
                end = min(start + int(batch_size), features.shape[0])
                x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
                y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
                logits = model.keypoint_logits(x)
                pred = torch.argmax(logits, dim=1)
                confidence = 1.0 - torch.softmax(logits, dim=1)[:, 64]
                positive = y < 64
                non_keypoint = y == 64
                correct.append((pred == y).detach().cpu())
                confidences.append(confidence.detach().cpu())
                positive_count += int(torch.count_nonzero(positive).item())
                non_keypoint_count += int(torch.count_nonzero(non_keypoint).item())
                if torch.any(positive):
                    positive_correct.append((pred[positive] == y[positive]).detach().cpu())
                    positive_confidences.append(confidence[positive].detach().cpu())
                if torch.any(non_keypoint):
                    non_keypoint_correct.append((pred[non_keypoint] == 64).detach().cpu())
                    non_keypoint_confidences.append(confidence[non_keypoint].detach().cpu())

        def mean_float(values: list[torch.Tensor]) -> float:
            if not values:
                return 0.0
            return float(torch.mean(torch.cat(values).float()).item())

        return {
            f"{prefix}_keypoint_acc": mean_float(correct),
            f"{prefix}_keypoint_positive_acc": mean_float(positive_correct),
            f"{prefix}_keypoint_non_keypoint_acc": mean_float(non_keypoint_correct),
            f"{prefix}_keypoint_confidence_mean": mean_float(confidences),
            f"{prefix}_keypoint_positive_confidence_mean": mean_float(positive_confidences),
            f"{prefix}_keypoint_non_keypoint_confidence_mean": mean_float(non_keypoint_confidences),
            f"{prefix}_keypoint_positive_count": float(positive_count),
            f"{prefix}_keypoint_non_keypoint_count": float(non_keypoint_count),
        }

    metrics = {
        "top1_acc": float(torch.mean((top1 == labels).float()).item()),
        "query_offset_acc": float(torch.mean(torch.cat(qoffset_ok).float()).item()),
        "render_offset_acc": float(torch.mean(torch.cat(roffset_ok).float()).item()),
        "render_pair_fine_acc": float(torch.mean(torch.cat(render_pair_fine_ok).float()).item()) if render_pair_fine_ok else 0.0,
        "render_pair_fine_valid_count": float(render_pair_fine_valid_count),
        "query_detector_mean": float(torch.mean(query_detector).item()),
        "render_detector_mean": float(torch.mean(render_detector).item()),
        "detector_mean_gap": float(torch.abs(torch.mean(query_detector) - torch.mean(render_detector)).item()),
    }
    metrics.update(keypoint_metrics("query", samples.query_keypoint_features, samples.query_keypoint_labels))
    metrics.update(keypoint_metrics("render", samples.render_keypoint_features, samples.render_keypoint_labels))
    return metrics


def train_matcha_coarse_fine_adapter(
    samples: MatchaCoarseFineTrainingSet,
    config: MatchaCoarseFineTrainingConfig | None = None,
) -> MatchaCoarseFineTrainingRun:
    config = config or MatchaCoarseFineTrainingConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one coarse-fine sample is required")
    if int(config.output_dim) > samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = MatchaCoarseFineAdapter(
        input_dim=samples.input_dim,
        output_dim=int(config.output_dim),
        residual_hidden_dim=int(config.residual_hidden_dim),
        group_size=int(config.group_size),
        input_norm_mode=str(config.input_norm_mode),
        gate_mode=str(config.gate_mode),
        residual_gate_scale=float(config.residual_gate_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))

    def keypoint_batch(subset: MatchaCoarseFineTrainingSet, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def sample(features: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            if features.shape[0] == 0:
                return (
                    torch.zeros((0, samples.input_dim), dtype=torch.float32, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                )
            take = min(int(count), int(features.shape[0]))
            idx = rng.choice(features.shape[0], size=take, replace=False)
            return (
                torch.as_tensor(features[idx], dtype=torch.float32, device=device),
                torch.as_tensor(labels[idx], dtype=torch.long, device=device),
            )

        qf, ql = sample(subset.query_keypoint_features, subset.query_keypoint_labels)
        rf, rl = sample(subset.render_keypoint_features, subset.render_keypoint_labels)
        return qf, ql, rf, rl

    def subset_loss(subset: MatchaCoarseFineTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        model.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, subset.sample_count, min(int(config.batch_size), 1024)):
                end = min(start + min(int(config.batch_size), 1024), subset.sample_count)
                query = torch.as_tensor(subset.query_features[start:end], dtype=torch.float32, device=device)
                render = torch.as_tensor(subset.render_features[start:end], dtype=torch.float32, device=device)
                qlabels = torch.as_tensor(subset.query_offset_labels[start:end], dtype=torch.long, device=device)
                rlabels = torch.as_tensor(subset.render_offset_labels[start:end], dtype=torch.long, device=device)
                negatives = torch.as_tensor(subset.negative_render_features[start:end], dtype=torch.float32, device=device)
                qkf, qkl, rkf, rkl = keypoint_batch(subset, end - start)
                losses.append(float(_loss(model, query, render, qlabels, rlabels, negatives, config, qkf, qkl, rkf, rkl).detach().cpu()))
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    initial_loss = subset_loss(train_samples)
    model.train()
    for _ in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_samples.sample_count)
        batch_idx = rng.choice(train_samples.sample_count, size=batch_count, replace=False)
        query = torch.as_tensor(train_samples.query_features[batch_idx], dtype=torch.float32, device=device)
        render = torch.as_tensor(train_samples.render_features[batch_idx], dtype=torch.float32, device=device)
        qlabels = torch.as_tensor(train_samples.query_offset_labels[batch_idx], dtype=torch.long, device=device)
        rlabels = torch.as_tensor(train_samples.render_offset_labels[batch_idx], dtype=torch.long, device=device)
        negatives = torch.as_tensor(train_samples.negative_render_features[batch_idx], dtype=torch.float32, device=device)
        qkf, qkl, rkf, rkl = keypoint_batch(train_samples, batch_count)
        loss = _loss(model, query, render, qlabels, rlabels, negatives, config, qkf, qkl, rkf, rkl)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = subset_loss(train_samples)
    train_eval = _evaluate(model, train_samples, device, min(int(config.batch_size), 2048))
    eval_eval = _evaluate(model, eval_samples, device, min(int(config.batch_size), 2048))
    summary = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "sample_count": int(samples.sample_count),
        "train_sample_count": int(train_samples.sample_count),
        "eval_sample_count": int(eval_samples.sample_count),
        "input_dim": int(samples.input_dim),
        "output_dim": int(config.output_dim),
        "steps": int(config.steps),
        "batch_size": int(config.batch_size),
        "train_top1_acc": float(train_eval["top1_acc"]),
        "eval_top1_acc": float(eval_eval["top1_acc"]),
        "query_offset_acc": float(train_eval["query_offset_acc"]),
        "render_offset_acc": float(train_eval["render_offset_acc"]),
        "render_pair_fine_acc": float(train_eval["render_pair_fine_acc"]),
        "render_pair_fine_valid_count": int(train_eval["render_pair_fine_valid_count"]),
        "eval_query_offset_acc": float(eval_eval["query_offset_acc"]),
        "eval_render_offset_acc": float(eval_eval["render_offset_acc"]),
        "eval_render_pair_fine_acc": float(eval_eval["render_pair_fine_acc"]),
        "eval_render_pair_fine_valid_count": int(eval_eval["render_pair_fine_valid_count"]),
        "query_detector_mean": float(train_eval["query_detector_mean"]),
        "render_detector_mean": float(train_eval["render_detector_mean"]),
        "detector_mean_gap": float(train_eval["detector_mean_gap"]),
        "eval_query_detector_mean": float(eval_eval["query_detector_mean"]),
        "eval_render_detector_mean": float(eval_eval["render_detector_mean"]),
        "eval_detector_mean_gap": float(eval_eval["detector_mean_gap"]),
        "query_keypoint_acc": float(train_eval["query_keypoint_acc"]),
        "render_keypoint_acc": float(train_eval["render_keypoint_acc"]),
        "query_keypoint_positive_acc": float(train_eval["query_keypoint_positive_acc"]),
        "render_keypoint_positive_acc": float(train_eval["render_keypoint_positive_acc"]),
        "query_keypoint_non_keypoint_acc": float(train_eval["query_keypoint_non_keypoint_acc"]),
        "render_keypoint_non_keypoint_acc": float(train_eval["render_keypoint_non_keypoint_acc"]),
        "query_keypoint_confidence_mean": float(train_eval["query_keypoint_confidence_mean"]),
        "render_keypoint_confidence_mean": float(train_eval["render_keypoint_confidence_mean"]),
        "query_keypoint_positive_confidence_mean": float(train_eval["query_keypoint_positive_confidence_mean"]),
        "render_keypoint_positive_confidence_mean": float(train_eval["render_keypoint_positive_confidence_mean"]),
        "query_keypoint_non_keypoint_confidence_mean": float(train_eval["query_keypoint_non_keypoint_confidence_mean"]),
        "render_keypoint_non_keypoint_confidence_mean": float(train_eval["render_keypoint_non_keypoint_confidence_mean"]),
        "query_keypoint_positive_count": int(train_eval["query_keypoint_positive_count"]),
        "render_keypoint_positive_count": int(train_eval["render_keypoint_positive_count"]),
        "query_keypoint_non_keypoint_count": int(train_eval["query_keypoint_non_keypoint_count"]),
        "render_keypoint_non_keypoint_count": int(train_eval["render_keypoint_non_keypoint_count"]),
        "eval_query_keypoint_acc": float(eval_eval["query_keypoint_acc"]),
        "eval_render_keypoint_acc": float(eval_eval["render_keypoint_acc"]),
        "eval_query_keypoint_positive_acc": float(eval_eval["query_keypoint_positive_acc"]),
        "eval_render_keypoint_positive_acc": float(eval_eval["render_keypoint_positive_acc"]),
        "eval_query_keypoint_non_keypoint_acc": float(eval_eval["query_keypoint_non_keypoint_acc"]),
        "eval_render_keypoint_non_keypoint_acc": float(eval_eval["render_keypoint_non_keypoint_acc"]),
        "eval_query_keypoint_confidence_mean": float(eval_eval["query_keypoint_confidence_mean"]),
        "eval_render_keypoint_confidence_mean": float(eval_eval["render_keypoint_confidence_mean"]),
        "eval_query_keypoint_positive_confidence_mean": float(eval_eval["query_keypoint_positive_confidence_mean"]),
        "eval_render_keypoint_positive_confidence_mean": float(eval_eval["render_keypoint_positive_confidence_mean"]),
        "eval_query_keypoint_non_keypoint_confidence_mean": float(eval_eval["query_keypoint_non_keypoint_confidence_mean"]),
        "eval_render_keypoint_non_keypoint_confidence_mean": float(eval_eval["render_keypoint_non_keypoint_confidence_mean"]),
        "eval_query_keypoint_positive_count": int(eval_eval["query_keypoint_positive_count"]),
        "eval_render_keypoint_positive_count": int(eval_eval["render_keypoint_positive_count"]),
        "eval_query_keypoint_non_keypoint_count": int(eval_eval["query_keypoint_non_keypoint_count"]),
        "eval_render_keypoint_non_keypoint_count": int(eval_eval["render_keypoint_non_keypoint_count"]),
    }
    return MatchaCoarseFineTrainingRun(model=model.cpu().eval(), summary=summary)


def project_feature_map_with_matcha_adapter(
    model: MatchaCoarseFineAdapter,
    feature_map: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    rows = fmap.reshape(channels, height * width).T
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    descriptors = []
    offsets = []
    with torch.no_grad():
        for start in range(0, rows.shape[0], int(batch_size)):
            tensor = torch.as_tensor(rows[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            z, logits = model(tensor)
            descriptors.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            offsets.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    desc = np.concatenate(descriptors, axis=0).T.reshape(model.output_dim, height, width)
    logits = np.concatenate(offsets, axis=0).T.reshape(65, height, width)
    return desc.astype(np.float32, copy=False), logits.astype(np.float32, copy=False)


def project_feature_map_with_matcha_adapter_full(
    model: MatchaCoarseFineAdapter,
    feature_map: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    rows = fmap.reshape(channels, height * width).T
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    descriptors = []
    offsets = []
    detectors = []
    with torch.no_grad():
        for start in range(0, rows.shape[0], int(batch_size)):
            tensor = torch.as_tensor(rows[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            z, logits, detector_logits = model.forward_full(tensor)
            descriptors.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            offsets.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            detectors.append(detector_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    desc = np.concatenate(descriptors, axis=0).T.reshape(model.output_dim, height, width)
    offset_logits = np.concatenate(offsets, axis=0).T.reshape(65, height, width)
    detector_logits = np.concatenate(detectors, axis=0).reshape(height, width)
    return (
        desc.astype(np.float32, copy=False),
        offset_logits.astype(np.float32, copy=False),
        detector_logits.astype(np.float32, copy=False),
    )


def project_feature_map_with_matcha_keypoint_logits(
    model: MatchaCoarseFineAdapter,
    feature_map: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    rows = fmap.reshape(channels, height * width).T
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    logits = []
    with torch.no_grad():
        for start in range(0, rows.shape[0], int(batch_size)):
            tensor = torch.as_tensor(rows[start : start + int(batch_size)], dtype=torch.float32, device=torch_device)
            logits.append(model.keypoint_logits(tensor).detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    return np.concatenate(logits, axis=0).T.reshape(65, height, width).astype(np.float32, copy=False)


def _descriptor_rows_from_feature_map(feature_map: np.ndarray) -> np.ndarray:
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    channels, height, width = fmap.shape
    return fmap.reshape(channels, height * width).T.astype(np.float32, copy=False)


def predict_matcha_pair_heads_for_matches(
    model: MatchaCoarseFineAdapter,
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    matches,
    *,
    device: str = "cpu",
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    values = list(matches)
    if not values:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 64), dtype=np.float32)
    query_rows = _descriptor_rows_from_feature_map(query_feature_map)
    render_rows = _descriptor_rows_from_feature_map(render_feature_map)
    query_indices = np.asarray([int(match.query_index) for match in values], dtype=np.int64)
    render_indices = np.asarray([int(match.render_index) for match in values], dtype=np.int64)
    if np.any(query_indices < 0) or np.any(query_indices >= query_rows.shape[0]):
        raise ValueError("match query_index exceeds query feature map size")
    if np.any(render_indices < 0) or np.any(render_indices >= render_rows.shape[0]):
        raise ValueError("match render_index exceeds render feature map size")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    was_training = model.training
    model = model.to(torch_device).eval()
    confidences = []
    fine_logits = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            end = min(start + int(batch_size), len(values))
            query = torch.as_tensor(query_rows[query_indices[start:end]], dtype=torch.float32, device=torch_device)
            render = torch.as_tensor(render_rows[render_indices[start:end]], dtype=torch.float32, device=torch_device)
            if int(query.shape[1]) == int(model.output_dim) and int(render.shape[1]) == int(model.output_dim):
                query_z = torch.nn.functional.normalize(query, dim=1)
                render_z = torch.nn.functional.normalize(render, dim=1)
            elif int(query.shape[1]) == int(model.input_dim) and int(render.shape[1]) == int(model.input_dim):
                query_z = model.encode(query)
                render_z = model.encode(render)
            else:
                raise ValueError(
                    "feature maps must contain either raw adapter input descriptors or projected adapter descriptors"
                )
            logits = model.pair_confidence_logits(query_z, render_z)
            fine = model.pair_fine_logits(query_z, render_z)
            confidences.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False))
            fine_logits.append(fine.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    return np.concatenate(confidences, axis=0), np.concatenate(fine_logits, axis=0)


def save_matcha_coarse_fine_adapter(run: MatchaCoarseFineTrainingRun, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = run.model.cpu().eval()
    torch.save(
        {
            "format": _FORMAT,
            "model_config": {
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
            "summary": dict(run.summary),
        },
        output,
    )


def load_matcha_coarse_fine_adapter(path: Path, device: str = "cpu") -> MatchaCoarseFineTrainingRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != _FORMAT:
        raise ValueError(f"unsupported MATCHA coarse-fine adapter checkpoint format in {path}")
    cfg = dict(payload["model_config"])
    model = MatchaCoarseFineAdapter(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["output_dim"]),
        residual_hidden_dim=int(cfg["residual_hidden_dim"]),
        group_size=int(cfg["group_size"]),
        input_mean=np.asarray(cfg["input_mean"], dtype=np.float32),
        input_norm_mode=str(cfg.get("input_norm_mode", "identity")),
        gate_mode=str(cfg.get("gate_mode", "residual")),
        residual_gate_scale=float(cfg.get("residual_gate_scale", 0.1)),
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    summary = dict(payload.get("summary", {}))
    missing = [str(item) for item in getattr(incompatible, "missing_keys", [])]
    unexpected = [str(item) for item in getattr(incompatible, "unexpected_keys", [])]
    if missing:
        summary["missing_state_keys"] = missing
    if unexpected:
        summary["unexpected_state_keys"] = unexpected
    return MatchaCoarseFineTrainingRun(model=model.to(torch.device(device)).eval(), summary=summary)


def save_matcha_coarse_fine_training_set_npz(samples: MatchaCoarseFineTrainingSet, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray([_FORMAT], dtype=object),
        query_features=samples.query_features,
        render_features=samples.render_features,
        query_offset_labels=samples.query_offset_labels,
        render_offset_labels=samples.render_offset_labels,
        negative_render_features=samples.negative_render_features,
        roundtrip_errors_px=samples.roundtrip_errors_px,
        query_keypoint_features=samples.query_keypoint_features,
        query_keypoint_labels=samples.query_keypoint_labels,
        render_keypoint_features=samples.render_keypoint_features,
        render_keypoint_labels=samples.render_keypoint_labels,
        metadata=np.asarray([dict(samples.metadata or {})], dtype=object),
    )


def load_matcha_coarse_fine_training_set_npz(path: Path) -> tuple[MatchaCoarseFineTrainingSet, dict[str, object]]:
    with np.load(Path(path), allow_pickle=True) as data:
        fmt = str(data["format"][0]) if "format" in data else ""
        if fmt != _FORMAT:
            raise ValueError(f"unsupported MATCHA coarse-fine sample format in {path}")
        metadata = dict(data["metadata"][0]) if "metadata" in data else {}
        samples = MatchaCoarseFineTrainingSet(
            query_features=np.asarray(data["query_features"], dtype=np.float32),
            render_features=np.asarray(data["render_features"], dtype=np.float32),
            query_offset_labels=np.asarray(data["query_offset_labels"], dtype=np.int64),
            render_offset_labels=np.asarray(data["render_offset_labels"], dtype=np.int64),
            negative_render_features=np.asarray(data["negative_render_features"], dtype=np.float32),
            roundtrip_errors_px=np.asarray(data["roundtrip_errors_px"], dtype=np.float32),
            query_keypoint_features=np.asarray(data["query_keypoint_features"], dtype=np.float32)
            if "query_keypoint_features" in data
            else None,
            query_keypoint_labels=np.asarray(data["query_keypoint_labels"], dtype=np.int64)
            if "query_keypoint_labels" in data
            else None,
            render_keypoint_features=np.asarray(data["render_keypoint_features"], dtype=np.float32)
            if "render_keypoint_features" in data
            else None,
            render_keypoint_labels=np.asarray(data["render_keypoint_labels"], dtype=np.int64)
            if "render_keypoint_labels" in data
            else None,
            metadata=metadata,
        )
    return samples, metadata


def append_matcha_coarse_fine_training_set_capped(
    existing: MatchaCoarseFineTrainingSet | None,
    new: MatchaCoarseFineTrainingSet,
    *,
    max_samples: int = 0,
    seed: int = 0,
) -> MatchaCoarseFineTrainingSet:
    if existing is None:
        merged = new
    else:
        if existing.input_dim != new.input_dim:
            raise ValueError("cannot merge MATCHA coarse-fine samples with different dimensions")
        if existing.negative_render_features.shape[1] != new.negative_render_features.shape[1]:
            raise ValueError("cannot merge samples with different negative counts")
        merged = MatchaCoarseFineTrainingSet(
            query_features=np.concatenate([existing.query_features, new.query_features], axis=0),
            render_features=np.concatenate([existing.render_features, new.render_features], axis=0),
            query_offset_labels=np.concatenate([existing.query_offset_labels, new.query_offset_labels], axis=0),
            render_offset_labels=np.concatenate([existing.render_offset_labels, new.render_offset_labels], axis=0),
            negative_render_features=np.concatenate([existing.negative_render_features, new.negative_render_features], axis=0),
            roundtrip_errors_px=np.concatenate([existing.roundtrip_errors_px, new.roundtrip_errors_px], axis=0),
            query_keypoint_features=np.concatenate([existing.query_keypoint_features, new.query_keypoint_features], axis=0),
            query_keypoint_labels=np.concatenate([existing.query_keypoint_labels, new.query_keypoint_labels], axis=0),
            render_keypoint_features=np.concatenate([existing.render_keypoint_features, new.render_keypoint_features], axis=0),
            render_keypoint_labels=np.concatenate([existing.render_keypoint_labels, new.render_keypoint_labels], axis=0),
            metadata={**dict(existing.metadata or {}), **dict(new.metadata or {}), "merged": True},
        )
    if int(max_samples) <= 0 or merged.sample_count <= int(max_samples):
        return merged
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(merged.sample_count, size=int(max_samples), replace=False))
    return _subset(merged, keep)


def merge_matcha_coarse_fine_training_sets(
    samples: list[MatchaCoarseFineTrainingSet] | tuple[MatchaCoarseFineTrainingSet, ...],
    *,
    max_samples: int = 0,
    seed: int = 0,
) -> MatchaCoarseFineTrainingSet:
    """Merge multiple MATCHA coarse-fine training sets with optional cap."""

    values = list(samples)
    if not values:
        raise ValueError("at least one training set is required")
    input_dim = values[0].input_dim
    negative_count = int(values[0].negative_render_features.shape[1])
    for item in values:
        if int(item.input_dim) != int(input_dim):
            raise ValueError("cannot merge training sets with different input dimensions")
        if int(item.negative_render_features.shape[1]) != int(negative_count):
            raise ValueError("cannot merge training sets with different negative counts")
    merged = MatchaCoarseFineTrainingSet(
        query_features=np.concatenate([item.query_features for item in values], axis=0),
        render_features=np.concatenate([item.render_features for item in values], axis=0),
        query_offset_labels=np.concatenate([item.query_offset_labels for item in values], axis=0),
        render_offset_labels=np.concatenate([item.render_offset_labels for item in values], axis=0),
        negative_render_features=np.concatenate([item.negative_render_features for item in values], axis=0),
        roundtrip_errors_px=np.concatenate([item.roundtrip_errors_px for item in values], axis=0),
        query_keypoint_features=np.concatenate([item.query_keypoint_features for item in values], axis=0),
        query_keypoint_labels=np.concatenate([item.query_keypoint_labels for item in values], axis=0),
        render_keypoint_features=np.concatenate([item.render_keypoint_features for item in values], axis=0),
        render_keypoint_labels=np.concatenate([item.render_keypoint_labels for item in values], axis=0),
        metadata={
            "merged": True,
            "source_count": int(len(values)),
            "source_sample_counts": [int(item.sample_count) for item in values],
        },
    )
    if int(max_samples) <= 0 or merged.sample_count <= int(max_samples):
        return merged
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(merged.sample_count, size=int(max_samples), replace=False))
    return _subset(merged, keep)
