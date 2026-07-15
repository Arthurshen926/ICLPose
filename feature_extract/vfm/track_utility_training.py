"""Track-supervised selector pretraining for mapable localizable features."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint


@dataclass(frozen=True)
class TrackUtilityTrainingConfig:
    steps: int = 100
    batch_size: int = 64
    output_dim: int = 64
    group_size: int = 16
    seed: int = 0
    device: str = "cpu"
    lr: float = 1e-3
    min_observations: int = 2
    temperature: float = 0.1
    consistency_weight: float = 1.0
    contrastive_weight: float = 1.0
    utility_weight: float = 0.5
    sparsity_weight: float = 0.005
    init_checkpoint: str = ""


@dataclass(frozen=True)
class TrackUtilityTrainingSummary:
    initial_loss: float
    final_loss: float
    initial_positive_similarity: float
    initial_negative_similarity: float
    final_positive_similarity: float
    final_negative_similarity: float
    initial_utility_target_correlation: float
    final_utility_target_correlation: float
    track_count: int
    observation_count: int
    feature_dim: int
    selector_output_dim: int
    selector_group_size: int


@dataclass(frozen=True)
class TrackUtilityTrainingRun:
    selector: LocalizableFeatureSelector
    summary: TrackUtilityTrainingSummary


@dataclass(frozen=True)
class _TrackSupervisionGroup:
    track_id: int
    features: np.ndarray
    utility_targets: np.ndarray


def _normalize_targets(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < 1e-6:
        return np.full_like(array, 0.5, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32, copy=False)


def _build_track_supervision_groups(
    observations: Sequence[TrackObservation],
    min_observations: int = 2,
) -> list[_TrackSupervisionGroup]:
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    grouped: dict[int, list[TrackObservation]] = {}
    feature_dim: int | None = None
    for obs in observations:
        if not obs.visible or not obs.geometry_valid:
            continue
        feature = np.asarray(obs.feature, dtype=np.float32).reshape(-1)
        if feature_dim is None:
            feature_dim = int(feature.size)
        elif feature.size != feature_dim:
            raise ValueError("all track observation features must have the same dimension")
        grouped.setdefault(int(obs.track_id), []).append(
            TrackObservation(
                track_id=int(obs.track_id),
                image_id=obs.image_id,
                feature=feature,
                visible=True,
                geometry_valid=True,
                utility=float(obs.utility),
                camera_center=(
                    None
                    if obs.camera_center is None
                    else np.asarray(obs.camera_center, dtype=np.float64).reshape(3)
                ),
                viewing_ray=(
                    None
                    if obs.viewing_ray is None
                    else np.asarray(obs.viewing_ray, dtype=np.float64).reshape(3)
                ),
            )
        )

    raw_groups: list[tuple[int, list[TrackObservation]]] = []
    all_utilities: list[float] = []
    for track_id in sorted(grouped):
        track_obs = grouped[track_id]
        if len(track_obs) < min_observations:
            continue
        raw_groups.append((track_id, track_obs))
        all_utilities.extend(float(obs.utility) for obs in track_obs)

    if not raw_groups:
        raise ValueError("track supervision requires at least one track with at least two observations")
    if len(raw_groups) < 2:
        raise ValueError("track supervision requires at least two tracks for contrastive negatives")

    normalized_utilities = _normalize_targets(np.asarray(all_utilities, dtype=np.float32))
    groups: list[_TrackSupervisionGroup] = []
    offset = 0
    for track_id, track_obs in raw_groups:
        count = len(track_obs)
        utilities = normalized_utilities[offset : offset + count]
        offset += count
        groups.append(
            _TrackSupervisionGroup(
                track_id=track_id,
                features=np.stack([obs.feature for obs in track_obs], axis=0).astype(np.float32),
                utility_targets=utilities,
            )
        )
    return groups


def _track_descriptors(
    selector: LocalizableFeatureSelector,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = selector(features.view(features.shape[0], features.shape[1], 1, 1))
    selected = F.normalize(output.selected.flatten(1), p=2, dim=1, eps=1e-6)
    utility_logits = output.utility.flatten(1).mean(dim=1)
    return selected, utility_logits


def _loss_for_track_groups(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_TrackSupervisionGroup],
    config: TrackUtilityTrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    features = torch.cat(
        [torch.as_tensor(group.features, dtype=torch.float32, device=device) for group in groups],
        dim=0,
    )
    track_ids = torch.cat(
        [
            torch.full((group.features.shape[0],), idx, dtype=torch.long, device=device)
            for idx, group in enumerate(groups)
        ],
        dim=0,
    )
    utility_targets = torch.cat(
        [torch.as_tensor(group.utility_targets, dtype=torch.float32, device=device) for group in groups],
        dim=0,
    )
    descriptors, utility_scores = _track_descriptors(selector, features)

    consistency_losses = []
    track_means = []
    for idx in range(len(groups)):
        mask = track_ids == idx
        track_features = descriptors[mask]
        mean = F.normalize(track_features.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-6)
        track_means.append(mean[0])
        consistency_losses.append(torch.mean((track_features - mean) ** 2))
    consistency = torch.stack(consistency_losses).mean()

    means = F.normalize(torch.stack(track_means, dim=0), p=2, dim=1, eps=1e-6)
    logits = torch.matmul(descriptors, means.T) / max(float(config.temperature), 1e-6)
    contrastive = F.cross_entropy(logits, track_ids)
    utility = F.mse_loss(utility_scores, utility_targets)
    sparsity = torch.sigmoid(selector.group_logits).abs().mean()
    return (
        config.consistency_weight * consistency
        + config.contrastive_weight * contrastive
        + config.utility_weight * utility
        + config.sparsity_weight * sparsity
    )


def _pairwise_track_separation(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_TrackSupervisionGroup],
    device: torch.device,
) -> tuple[float, float]:
    with torch.no_grad():
        positives = []
        means = []
        for group in groups:
            features = torch.as_tensor(group.features, dtype=torch.float32, device=device)
            descriptors, _utility = _track_descriptors(selector, features)
            mean = F.normalize(descriptors.mean(dim=0, keepdim=True), p=2, dim=1, eps=1e-6)
            means.append(mean[0])
            if descriptors.shape[0] >= 2:
                sim = torch.matmul(descriptors, descriptors.T)
                mask = ~torch.eye(descriptors.shape[0], dtype=torch.bool, device=device)
                positives.append(sim[mask].mean())
        mean_matrix = F.normalize(torch.stack(means, dim=0), p=2, dim=1, eps=1e-6)
        sim_matrix = torch.matmul(mean_matrix, mean_matrix.T)
        negative_mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=device)
        positive = torch.stack(positives).mean() if positives else torch.zeros((), device=device)
        negative = sim_matrix[negative_mask].mean()
    return float(positive.detach().cpu()), float(negative.detach().cpu())


def _utility_target_correlation(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_TrackSupervisionGroup],
    device: torch.device,
) -> float:
    with torch.no_grad():
        features = torch.cat(
            [torch.as_tensor(group.features, dtype=torch.float32, device=device) for group in groups],
            dim=0,
        )
        targets = torch.cat(
            [torch.as_tensor(group.utility_targets, dtype=torch.float32, device=device) for group in groups],
            dim=0,
        )
        _descriptors, utility_scores = _track_descriptors(selector, features)
        centered_scores = utility_scores - utility_scores.mean()
        centered_targets = targets - targets.mean()
        denom = torch.linalg.norm(centered_scores) * torch.linalg.norm(centered_targets)
        if float(denom.detach().cpu()) < 1e-6:
            return 0.0
        return float((torch.sum(centered_scores * centered_targets) / denom).detach().cpu())


def run_track_utility_training(
    observations: Sequence[TrackObservation],
    config: TrackUtilityTrainingConfig,
) -> TrackUtilityTrainingRun:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device)

    groups = _build_track_supervision_groups(observations, min_observations=config.min_observations)
    feature_dim = int(groups[0].features.shape[1])
    if config.init_checkpoint:
        selector = load_selector_from_checkpoint(Path(config.init_checkpoint), input_dim=feature_dim, device=config.device)
    else:
        selector = LocalizableFeatureSelector(
            input_dim=feature_dim,
            output_dim=config.output_dim,
            group_size=config.group_size,
        ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.lr)

    with torch.no_grad():
        initial_loss = float(_loss_for_track_groups(selector, groups, config, device).detach().cpu())
        initial_positive, initial_negative = _pairwise_track_separation(selector, groups, device)
        initial_corr = _utility_target_correlation(selector, groups, device)

    for _step in range(config.steps):
        batch = rng.sample(groups, min(config.batch_size, len(groups)))
        loss = _loss_for_track_groups(selector, batch, config, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(_loss_for_track_groups(selector, groups, config, device).detach().cpu())
        final_positive, final_negative = _pairwise_track_separation(selector, groups, device)
        final_corr = _utility_target_correlation(selector, groups, device)

    summary = TrackUtilityTrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_positive_similarity=initial_positive,
        initial_negative_similarity=initial_negative,
        final_positive_similarity=final_positive,
        final_negative_similarity=final_negative,
        initial_utility_target_correlation=initial_corr,
        final_utility_target_correlation=final_corr,
        track_count=len(groups),
        observation_count=int(sum(group.features.shape[0] for group in groups)),
        feature_dim=feature_dim,
        selector_output_dim=int(selector.output_dim),
        selector_group_size=int(selector.group_size),
    )
    return TrackUtilityTrainingRun(selector=selector, summary=summary)
