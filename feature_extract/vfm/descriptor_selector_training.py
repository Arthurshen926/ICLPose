"""Descriptor-level selector training for fixed VFM candidate banks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.losses import group_sparsity_loss, listwise_pose_rank_loss
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


@dataclass(frozen=True)
class DescriptorSelectorTrainingConfig:
    steps: int = 100
    batch_size: int = 16
    output_dim: int = 64
    group_size: int = 16
    seed: int = 0
    device: str = "cpu"
    lr: float = 1e-3
    eval_split_fraction: float = 0.2
    rank_temperature: float = 0.1
    sparsity_weight: float = 0.005


@dataclass(frozen=True)
class DescriptorSelectorTrainingSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    query_count: int
    train_query_count: int
    eval_query_count: int


@dataclass(frozen=True)
class DescriptorSelectorTrainingRun:
    selector: LocalizableFeatureSelector
    summary: DescriptorSelectorTrainingSummary


@dataclass(frozen=True)
class _QueryGroup:
    query_id: str
    query_descriptor: np.ndarray
    candidate_descriptors: np.ndarray
    costs: np.ndarray


def _candidate_cost(candidate: CandidateHypothesis, max_translation_m: float, max_rotation_deg: float) -> float:
    if candidate.pose_error is None:
        raise ValueError("pose_error is required for descriptor selector training")
    return float(candidate.pose_error.combined(max_translation_m, max_rotation_deg))


def _build_query_groups(
    bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
) -> List[_QueryGroup]:
    if query_descriptors.descriptors.shape[1] != map_descriptors.descriptors.shape[1]:
        raise ValueError("query and map descriptor dimensions must match")

    query_index = query_descriptors.index()
    map_index = map_descriptors.index()
    by_query = {}
    translation_values = []
    rotation_values = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_index:
            raise ValueError(f"query descriptor not found: {candidate.query_id}")
        if candidate.reference_image not in map_index:
            raise ValueError(f"reference descriptor not found: {candidate.reference_image}")
        by_query.setdefault(candidate.query_id, []).append(candidate)
        translation_values.append(float(candidate.pose_error.translation_m))
        rotation_values.append(float(candidate.pose_error.rotation_deg))

    if not by_query:
        raise ValueError("candidate bank has no candidates")

    max_translation = max(max(translation_values), 1e-6)
    max_rotation = max(max(rotation_values), 1e-6)
    groups = []
    for query_id in sorted(by_query):
        candidates = by_query[query_id]
        if len(candidates) < 2:
            raise ValueError(f"query {query_id} needs at least 2 candidates")
        query_descriptor = query_descriptors.descriptors[query_index[query_id]]
        candidate_rows = []
        costs = []
        for candidate in candidates:
            assert candidate.reference_image is not None
            candidate_rows.append(map_descriptors.descriptors[map_index[candidate.reference_image]])
            costs.append(_candidate_cost(candidate, max_translation, max_rotation))
        groups.append(
            _QueryGroup(
                query_id=query_id,
                query_descriptor=np.asarray(query_descriptor, dtype=np.float32),
                candidate_descriptors=np.asarray(candidate_rows, dtype=np.float32),
                costs=np.asarray(costs, dtype=np.float32),
            )
        )
    return groups


def _split_groups(
    groups: Sequence[_QueryGroup],
    eval_split_fraction: float,
    seed: int,
) -> Tuple[List[_QueryGroup], List[_QueryGroup]]:
    if not 0.0 <= eval_split_fraction < 1.0:
        raise ValueError("eval_split_fraction must be in [0, 1)")
    order = list(groups)
    random.Random(seed).shuffle(order)
    eval_count = int(round(len(order) * eval_split_fraction))
    if eval_split_fraction > 0.0 and len(order) > 1:
        eval_count = max(1, eval_count)
    eval_count = min(eval_count, len(order) - 1)
    return order[eval_count:], order[:eval_count]


def _as_descriptor_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    return tensor.reshape(tensor.shape[0], tensor.shape[1], 1, 1)


def _score_group(
    selector: LocalizableFeatureSelector,
    group: _QueryGroup,
    device: torch.device,
) -> torch.Tensor:
    query = torch.as_tensor(group.query_descriptor, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    candidates = _as_descriptor_tensor(group.candidate_descriptors, device)
    query_selected = selector(query).selected.flatten(1)
    candidate_selected = selector(candidates).selected.flatten(1)
    query_selected = F.normalize(query_selected, dim=-1, eps=1e-6)
    candidate_selected = F.normalize(candidate_selected, dim=-1, eps=1e-6)
    return torch.matmul(query_selected, candidate_selected.T)


def _loss_for_groups(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_QueryGroup],
    device: torch.device,
    rank_temperature: float,
    sparsity_weight: float,
) -> torch.Tensor:
    losses = []
    for group in groups:
        scores = _score_group(selector, group, device)
        costs = torch.as_tensor(group.costs, dtype=torch.float32, device=device).view(1, -1)
        losses.append(listwise_pose_rank_loss(scores, costs, temperature=rank_temperature))
    data_loss = torch.stack(losses).mean()
    gates = torch.sigmoid(selector.group_logits)
    return data_loss + sparsity_weight * group_sparsity_loss(gates)


def _top1_acc(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_QueryGroup],
    device: torch.device,
) -> float:
    if not groups:
        return 0.0
    correct = 0
    with torch.no_grad():
        for group in groups:
            scores = _score_group(selector, group, device)
            if int(scores.argmax(dim=1).item()) == int(np.argmin(group.costs)):
                correct += 1
    return float(correct / len(groups))


def _raw_top1_acc(groups: Sequence[_QueryGroup]) -> float:
    if not groups:
        return 0.0
    correct = 0
    for group in groups:
        query = np.asarray(group.query_descriptor, dtype=np.float32)
        candidates = np.asarray(group.candidate_descriptors, dtype=np.float32)
        query_norm = query / max(float(np.linalg.norm(query)), 1e-6)
        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        candidate_norms = np.maximum(candidate_norms, 1e-6)
        scores = candidates / candidate_norms @ query_norm
        if int(np.argmax(scores)) == int(np.argmin(group.costs)):
            correct += 1
    return float(correct / len(groups))


def run_descriptor_selector_training(
    bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
    config: DescriptorSelectorTrainingConfig,
) -> DescriptorSelectorTrainingRun:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device)

    groups = _build_query_groups(bank, query_descriptors, map_descriptors)
    train_groups, eval_groups = _split_groups(groups, config.eval_split_fraction, config.seed)
    input_dim = int(query_descriptors.descriptors.shape[1])
    selector = LocalizableFeatureSelector(
        input_dim=input_dim,
        output_dim=config.output_dim,
        group_size=config.group_size,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.lr)

    with torch.no_grad():
        initial_loss = float(
            _loss_for_groups(
                selector,
                train_groups,
                device,
                config.rank_temperature,
                config.sparsity_weight,
            ).detach().cpu()
        )

    for _ in range(config.steps):
        batch_size = min(config.batch_size, len(train_groups))
        batch = rng.sample(train_groups, batch_size)
        loss = _loss_for_groups(
            selector,
            batch,
            device,
            config.rank_temperature,
            config.sparsity_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(
            _loss_for_groups(
                selector,
                train_groups,
                device,
                config.rank_temperature,
                config.sparsity_weight,
            ).detach().cpu()
        )
        raw_train_top1_acc = _raw_top1_acc(train_groups)
        raw_eval_top1_acc = _raw_top1_acc(eval_groups) if eval_groups else raw_train_top1_acc
        train_top1_acc = _top1_acc(selector, train_groups, device)
        eval_top1_acc = _top1_acc(selector, eval_groups, device) if eval_groups else train_top1_acc

    summary = DescriptorSelectorTrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        raw_train_top1_acc=raw_train_top1_acc,
        raw_eval_top1_acc=raw_eval_top1_acc,
        train_top1_acc=train_top1_acc,
        eval_top1_acc=eval_top1_acc,
        query_count=len(groups),
        train_query_count=len(train_groups),
        eval_query_count=len(eval_groups),
    )
    return DescriptorSelectorTrainingRun(selector=selector, summary=summary)


def train_descriptor_selector(
    bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
    config: DescriptorSelectorTrainingConfig,
) -> DescriptorSelectorTrainingSummary:
    """Train a descriptor selector and return a compact smoke-test summary."""

    return run_descriptor_selector_training(bank, query_descriptors, map_descriptors, config).summary
