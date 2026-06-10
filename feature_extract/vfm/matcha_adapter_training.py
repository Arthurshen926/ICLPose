"""MATCHA-style scene-specific descriptor adapter training."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.patch_selector_training import (
    PatchSelectorTrainingSet,
    ResidualGatedPatchSelector,
    SafePatchSelectorTrainingConfig,
    SafePatchSelectorTrainingRun,
    SafePatchSelectorTrainingSummary,
    _safe_active_group_mask,
    _safe_inlier_accuracy,
    _safe_selector_optimizer,
    _safe_top1_acc,
    _split_indices,
    _subset_samples,
)


@dataclass(frozen=True)
class MatchaAdapterTrainingConfig:
    output_dim: int = 128
    residual_hidden_dim: int = 256
    steps: int = 300
    batch_size: int = 512
    lr: float = 5e-5
    projection_lr: float | None = None
    residual_lr: float | None = None
    pairwise_lr: float | None = None
    gate_lr: float | None = None
    temperature: float = 0.07
    dual_softmax_weight: float = 1.0
    hard_negative_weight: float = 0.2
    hard_negative_margin: float = 0.2
    inlier_loss_weight: float = 0.05
    anchor_loss_weight: float = 0.2
    eval_split_fraction: float = 0.1
    group_size: int = 64
    input_norm_mode: str = "identity"
    gate_mode: str = "residual"
    residual_gate_scale: float = 0.1
    hard_gate_keep_fraction: float = 1.0
    hard_gate_min_groups: int = 1
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than one for dual-softmax")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        for name in ("dual_softmax_weight", "hard_negative_weight", "inlier_loss_weight", "anchor_loss_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.hard_negative_margin < 0.0:
            raise ValueError("hard_negative_margin must be non-negative")


def _effective_lr(value: float | None, fallback: float) -> float:
    return float(fallback if value is None else value)


def dual_softmax_descriptor_loss(query_z: torch.Tensor, positive_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric batch matching loss, matching MATCHA's dual-softmax spirit."""

    if query_z.ndim != 2 or positive_z.ndim != 2 or query_z.shape != positive_z.shape:
        raise ValueError("query_z and positive_z must have the same shape (B, D)")
    logits = query_z @ positive_z.T / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def _first_positive_features(samples: PatchSelectorTrainingSet, indices: np.ndarray) -> np.ndarray:
    positives = samples.positive_features[indices]
    masks = samples.positive_mask[indices]
    output = np.zeros((indices.shape[0], samples.input_dim), dtype=np.float32)
    for row in range(indices.shape[0]):
        present = np.flatnonzero(masks[row])
        if present.size == 0:
            raise ValueError("each sample must contain at least one positive")
        output[row] = positives[row, int(present[0])]
    return output


def _selector_anchor_loss(
    selector: ResidualGatedPatchSelector,
    anchor: ResidualGatedPatchSelector | None,
    query: torch.Tensor,
    positive: torch.Tensor,
) -> torch.Tensor:
    if anchor is None:
        return query.new_tensor(0.0)
    with torch.no_grad():
        query_anchor = anchor(query)
        positive_anchor = anchor(positive)
    query_z = selector(query)
    positive_z = selector(positive)
    return 0.5 * ((1.0 - (query_z * query_anchor).sum(dim=1)).mean() + (1.0 - (positive_z * positive_anchor).sum(dim=1)).mean())


def _matcha_loss(
    selector: ResidualGatedPatchSelector,
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    config: MatchaAdapterTrainingConfig,
    anchor: ResidualGatedPatchSelector | None = None,
) -> torch.Tensor:
    query_z = selector(query)
    positive_z = selector(positive)
    loss = query.new_tensor(0.0)
    if float(config.dual_softmax_weight) > 0.0:
        loss = loss + float(config.dual_softmax_weight) * dual_softmax_descriptor_loss(query_z, positive_z, config.temperature)
    if float(config.hard_negative_weight) > 0.0 and negatives.numel() > 0:
        negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
        pos_scores = torch.sum(query_z * positive_z, dim=1, keepdim=True)
        neg_scores = torch.einsum("bd,bkd->bk", query_z, negative_z)
        hard_loss = torch.relu(neg_scores - pos_scores + float(config.hard_negative_margin)).mean()
        loss = loss + float(config.hard_negative_weight) * hard_loss
    if float(config.inlier_loss_weight) > 0.0 and negatives.numel() > 0:
        negative_z = selector(negatives.reshape(-1, negatives.shape[-1])).reshape(negatives.shape[0], negatives.shape[1], -1)
        pos_logits = selector.pairwise_inlier_logit(query_z, positive_z)
        neg_logits = selector.pairwise_inlier_logit(
            query_z[:, None, :].expand_as(negative_z).reshape(-1, query_z.shape[-1]),
            negative_z.reshape(-1, query_z.shape[-1]),
        )
        inlier_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        inlier_loss = inlier_loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = loss + float(config.inlier_loss_weight) * inlier_loss
    if float(config.anchor_loss_weight) > 0.0:
        loss = loss + float(config.anchor_loss_weight) * _selector_anchor_loss(selector, anchor, query, positive)
    return loss


def train_matcha_dual_softmax_selector(
    samples: PatchSelectorTrainingSet,
    config: MatchaAdapterTrainingConfig | None = None,
    *,
    init_run: SafePatchSelectorTrainingRun | None = None,
) -> SafePatchSelectorTrainingRun:
    config = config or MatchaAdapterTrainingConfig()
    if samples.sample_count == 0:
        raise ValueError("at least one training sample is required")
    if int(config.output_dim) > samples.input_dim:
        raise ValueError("output_dim must be <= sample input_dim")
    if init_run is not None:
        if int(init_run.summary.input_dim) != samples.input_dim or int(init_run.summary.output_dim) != int(config.output_dim):
            raise ValueError("init_run dimensions must match samples/config")
        if str(init_run.summary.selector_arch) != "residual_gated":
            raise ValueError("M2 trainer currently expects residual_gated init_run")
    torch.manual_seed(int(config.seed))
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, config.eval_split_fraction, config.seed)
    train_samples = _subset_samples(samples, train_idx)
    eval_samples = _subset_samples(samples, eval_idx) if eval_idx.size else _subset_samples(samples, train_idx[:0])
    input_mean = (
        init_run.model.input_mean.detach().cpu().numpy().reshape(-1)
        if init_run is not None
        else np.zeros((samples.input_dim,), dtype=np.float32)
    )
    selector = ResidualGatedPatchSelector(
        samples.input_dim,
        int(config.output_dim),
        residual_hidden_dim=int(config.residual_hidden_dim),
        group_size=int(config.group_size),
        input_mean=input_mean,
        input_norm_mode=str(config.input_norm_mode),
        gate_mode=str(config.gate_mode),
        residual_gate_scale=float(config.residual_gate_scale),
    ).to(device)
    anchor = None
    if init_run is not None:
        selector.load_state_dict(init_run.model.state_dict())
        anchor = ResidualGatedPatchSelector(
            samples.input_dim,
            int(config.output_dim),
            residual_hidden_dim=int(config.residual_hidden_dim),
            group_size=int(config.group_size),
            input_mean=input_mean,
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
        ).to(device)
        anchor.load_state_dict(init_run.model.state_dict())
        anchor.eval()
        for param in anchor.parameters():
            param.requires_grad_(False)
    optimizer_config = SafePatchSelectorTrainingConfig(
        output_dim=int(config.output_dim),
        residual_hidden_dim=int(config.residual_hidden_dim),
        lr=float(config.lr),
        projection_lr=None if config.projection_lr is None else float(config.projection_lr),
        residual_lr=None if config.residual_lr is None else float(config.residual_lr),
        pairwise_lr=None if config.pairwise_lr is None else float(config.pairwise_lr),
        gate_lr=None if config.gate_lr is None else float(config.gate_lr),
        group_size=int(config.group_size),
    )
    optimizer = _safe_selector_optimizer(selector, optimizer_config)
    rng = np.random.default_rng(int(config.seed))

    def subset_loss(subset: PatchSelectorTrainingSet) -> float:
        if subset.sample_count == 0:
            return 0.0
        selector.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, subset.sample_count, min(int(config.batch_size), 1024)):
                end = min(start + min(int(config.batch_size), 1024), subset.sample_count)
                idx = np.arange(start, end, dtype=np.int64)
                query = torch.as_tensor(subset.query_features[idx], dtype=torch.float32, device=device)
                positive = torch.as_tensor(_first_positive_features(subset, idx), dtype=torch.float32, device=device)
                negatives = torch.as_tensor(subset.negative_features[idx], dtype=torch.float32, device=device)
                losses.append(float(_matcha_loss(selector, query, positive, negatives, config, anchor=anchor).detach().cpu()))
        selector.train()
        return float(np.mean(losses)) if losses else 0.0

    initial_loss = subset_loss(train_samples)
    selector.train()
    for _step in range(int(config.steps)):
        batch_count = min(int(config.batch_size), train_samples.sample_count)
        batch_idx = rng.choice(train_samples.sample_count, size=batch_count, replace=False)
        query = torch.as_tensor(train_samples.query_features[batch_idx], dtype=torch.float32, device=device)
        positive = torch.as_tensor(_first_positive_features(train_samples, batch_idx), dtype=torch.float32, device=device)
        negatives = torch.as_tensor(train_samples.negative_features[batch_idx], dtype=torch.float32, device=device)
        loss = _matcha_loss(selector, query, positive, negatives, config, anchor=anchor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = subset_loss(train_samples)
    active_group_mask = _safe_active_group_mask(selector, keep_fraction=float(config.hard_gate_keep_fraction), min_groups=int(config.hard_gate_min_groups))
    gates = selector.group_gates().detach().cpu().numpy().astype(np.float32)
    active_channel_count = int(samples.input_dim)
    train_top1 = _safe_top1_acc(selector, train_samples, device, active_group_mask=active_group_mask)
    eval_top1 = _safe_top1_acc(selector, eval_samples, device, active_group_mask=active_group_mask)
    raw_train_top1 = _safe_top1_acc(None, train_samples, device)
    raw_eval_top1 = _safe_top1_acc(None, eval_samples, device)
    parameter_count = int(sum(param.numel() for param in selector.parameters()))
    return SafePatchSelectorTrainingRun(
        model=selector.cpu(),
        active_group_mask=active_group_mask.astype(np.float32, copy=False),
        summary=SafePatchSelectorTrainingSummary(
            selector_arch="residual_gated",
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
            transformer_dim=0,
            transformer_layers=0,
            transformer_heads=0,
            transformer_ff_dim=0,
            transformer_dropout=0.0,
            steps=int(config.steps),
            batch_size=int(config.batch_size),
            group_size=int(config.group_size),
            group_count=int(active_group_mask.shape[0]),
            active_group_count=int(np.sum(active_group_mask > 0.0)),
            active_channel_count=active_channel_count,
            active_group_fraction=float(np.mean(active_group_mask > 0.0)) if active_group_mask.size else 0.0,
            gate_mean=float(np.mean(gates)) if gates.size else 0.0,
            gate_min=float(np.min(gates)) if gates.size else 0.0,
            gate_max=float(np.max(gates)) if gates.size else 0.0,
            parameter_count=parameter_count,
            inlier_loss_weight=float(config.inlier_loss_weight),
            group_lasso_weight=0.0,
            hard_gate_keep_fraction=float(config.hard_gate_keep_fraction),
            projection_lr=_effective_lr(config.projection_lr, config.lr),
            residual_lr=_effective_lr(config.residual_lr, config.lr),
            pairwise_lr=_effective_lr(config.pairwise_lr, config.lr),
            gate_lr=_effective_lr(config.gate_lr, config.lr),
            anchor_loss_weight=float(config.anchor_loss_weight),
            reprojection_margin_loss_weight=float(config.hard_negative_weight),
            reprojection_margin_max=float(config.hard_negative_margin),
            input_norm_mode=str(config.input_norm_mode),
            gate_mode=str(config.gate_mode),
            residual_gate_scale=float(config.residual_gate_scale),
            initialized_from_safe_checkpoint=bool(init_run is not None),
        ),
    )
