"""Small training utilities for VFM selector smoke experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from feature_extract.vfm.losses import group_sparsity_loss, listwise_pose_rank_loss
from feature_extract.vfm.selector import LocalizableFeatureSelector


@dataclass(frozen=True)
class SyntheticSelectorTrainingConfig:
    steps: int = 100
    batch_size: int = 32
    input_dim: int = 16
    output_dim: int = 8
    group_size: int = 4
    candidates_per_query: int = 4
    seed: int = 0
    device: str = "cpu"
    lr: float = 1e-2


@dataclass(frozen=True)
class SyntheticSelectorTrainingResult:
    initial_loss: float
    final_loss: float
    final_top1_acc: float
    signal_group_gate: float
    noise_group_gate_mean: float


def _make_batch(config: SyntheticSelectorTrainingConfig, generator: torch.Generator):
    batch = config.batch_size
    candidates = config.candidates_per_query
    channels = config.input_dim
    device = torch.device(config.device)

    signal = torch.randn(batch, config.group_size, 1, 1, generator=generator, device=device)
    query = torch.randn(batch, channels, 1, 1, generator=generator, device=device) * 0.4
    query[:, : config.group_size] = signal

    candidate_tokens = torch.randn(
        batch, candidates, channels, 1, 1, generator=generator, device=device
    ) * 0.4
    candidate_tokens[:, 0, : config.group_size] = signal + 0.03 * torch.randn(
        batch, config.group_size, 1, 1, generator=generator, device=device
    )
    costs = torch.full((batch, candidates), 0.7, device=device)
    costs[:, 0] = 0.05
    return query, candidate_tokens, costs


def _score_candidates(selector: LocalizableFeatureSelector, query, candidate_tokens):
    batch, candidates, channels, height, width = candidate_tokens.shape
    query_selected = selector(query).selected.flatten(1)
    flat_candidates = candidate_tokens.reshape(batch * candidates, channels, height, width)
    candidate_selected = selector(flat_candidates).selected.flatten(1).reshape(batch, candidates, -1)
    query_selected = F.normalize(query_selected, dim=-1, eps=1e-6)
    candidate_selected = F.normalize(candidate_selected, dim=-1, eps=1e-6)
    return torch.einsum("bd,bkd->bk", query_selected, candidate_selected)


def run_synthetic_selector_training(
    config: SyntheticSelectorTrainingConfig,
) -> SyntheticSelectorTrainingResult:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    selector = LocalizableFeatureSelector(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        group_size=config.group_size,
    ).to(config.device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.lr)

    initial_loss = None
    final_scores = None
    final_costs = None
    for step in range(config.steps):
        query, candidate_tokens, costs = _make_batch(config, generator)
        scores = _score_candidates(selector, query, candidate_tokens)
        loss = listwise_pose_rank_loss(scores, costs, temperature=0.1)
        loss = loss + 0.005 * group_sparsity_loss(torch.sigmoid(selector.group_logits))
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_scores = scores.detach()
        final_costs = costs.detach()

    assert initial_loss is not None and final_scores is not None and final_costs is not None
    final_loss = float(loss.detach().cpu())
    final_top1_acc = float((final_scores.argmax(dim=1) == final_costs.argmin(dim=1)).float().mean().cpu())
    gates = torch.sigmoid(selector.group_logits).detach().cpu()
    return SyntheticSelectorTrainingResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_top1_acc=final_top1_acc,
        signal_group_gate=float(gates[0]),
        noise_group_gate_mean=float(gates[1:].mean()),
    )
