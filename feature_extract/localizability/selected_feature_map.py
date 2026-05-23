"""Selected-feature map utilities for localization-driven POFD-FS training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .losses import (
    basin_bce_loss,
    channel_sparsity_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
    spatial_utility_entropy_loss,
)


@dataclass(frozen=True)
class SelectedTrackFeatureBank:
    """Aggregated selected 3D map features keyed by explicit track ids."""

    track_ids: torch.Tensor
    features: torch.Tensor
    visibility_count: torch.Tensor
    feature_variance: torch.Tensor
    utility_mean: torch.Tensor
    xyz: torch.Tensor | None = None


def save_selected_track_feature_bank(
    bank: SelectedTrackFeatureBank,
    save_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist a selected 3D map feature bank as a compact NPZ table."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track_ids": bank.track_ids.detach().cpu().numpy().astype(np.int64),
        "features": bank.features.detach().cpu().numpy().astype(np.float32),
        "visibility_count": bank.visibility_count.detach().cpu().numpy().astype(np.int64),
        "feature_variance": bank.feature_variance.detach().cpu().numpy().astype(np.float32),
        "utility_mean": bank.utility_mean.detach().cpu().numpy().astype(np.float32),
        "metadata": np.asarray([dict(metadata or {})], dtype=object),
    }
    if bank.xyz is not None:
        payload["xyz"] = bank.xyz.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(save_path, **payload)


def load_selected_track_feature_bank(load_path: str | Path) -> tuple[SelectedTrackFeatureBank, dict]:
    """Load a selected 3D map feature bank saved by ``save_selected_track_feature_bank``."""
    data = np.load(str(load_path), allow_pickle=True)
    metadata = {}
    if "metadata" in data.files:
        metadata_obj = data["metadata"][0]
        metadata = metadata_obj.item() if hasattr(metadata_obj, "item") else dict(metadata_obj)
    bank = SelectedTrackFeatureBank(
        track_ids=torch.as_tensor(data["track_ids"], dtype=torch.long),
        features=torch.as_tensor(data["features"], dtype=torch.float32),
        visibility_count=torch.as_tensor(data["visibility_count"], dtype=torch.long),
        feature_variance=torch.as_tensor(data["feature_variance"], dtype=torch.float32),
        utility_mean=torch.as_tensor(data["utility_mean"], dtype=torch.float32),
        xyz=torch.as_tensor(data["xyz"], dtype=torch.float32) if "xyz" in data.files else None,
    )
    return bank, dict(metadata)


def localization_feature_utility(
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    temperature_m: float = 0.05,
) -> torch.Tensor:
    """Convert localization pose costs into normalized feature-utility targets."""
    if pose_cost_m.ndim != 2:
        raise ValueError("pose_cost_m must have shape (B,K)")
    valid = valid_mask.to(device=pose_cost_m.device).bool() if valid_mask is not None else torch.ones_like(pose_cost_m, dtype=torch.bool)
    if basin_label is not None:
        if basin_label.shape != pose_cost_m.shape:
            raise ValueError("basin_label must have the same shape as pose_cost_m")
        basin_valid = basin_label.to(device=pose_cost_m.device).bool() & valid
        valid = torch.where(basin_valid.any(dim=1, keepdim=True), basin_valid, valid)
    logits = (-pose_cost_m.float() / max(float(temperature_m), 1.0e-6)).masked_fill(~valid, -1.0e9)
    utility = torch.softmax(logits, dim=1) * valid.to(dtype=pose_cost_m.dtype)
    return utility / utility.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def hard_hypothesis_selection_targets(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    cost_gap_m: float = 0.12,
) -> dict[str, torch.Tensor]:
    """Return oracle positives and score-hard false positives for each query."""
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    valid = valid_mask.to(device=scores.device).bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    positive_indices = []
    hard_negative_indices = []
    active_flags = []
    positive_mask = torch.zeros_like(valid)
    hard_negative_mask = torch.zeros_like(valid)
    for row_idx, (row_scores, row_costs, row_valid) in enumerate(zip(scores.float(), pose_cost_m.float(), valid)):
        if not row_valid.any():
            positive_indices.append(-1)
            hard_negative_indices.append(-1)
            active_flags.append(False)
            continue
        valid_costs = row_costs.masked_fill(~row_valid, float("inf"))
        positive_idx = int(valid_costs.argmin())
        hard_valid = row_valid & (row_costs > row_costs[positive_idx] + float(cost_gap_m))
        positive_mask[row_idx, positive_idx] = True
        positive_indices.append(positive_idx)
        if not hard_valid.any():
            hard_negative_indices.append(-1)
            active_flags.append(False)
            continue
        hard_scores = row_scores.masked_fill(~hard_valid, torch.finfo(row_scores.dtype).min / 4.0)
        negative_idx = int(hard_scores.argmax())
        hard_negative_mask[row_idx, negative_idx] = True
        hard_negative_indices.append(negative_idx)
        active_flags.append(True)
    return {
        "positive_index": torch.as_tensor(positive_indices, dtype=torch.long, device=scores.device),
        "hard_negative_index": torch.as_tensor(hard_negative_indices, dtype=torch.long, device=scores.device),
        "active_mask": torch.as_tensor(active_flags, dtype=torch.bool, device=scores.device),
        "positive_mask": positive_mask,
        "hard_negative_mask": hard_negative_mask,
    }


def aggregate_selected_track_features(
    features: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    utility: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    geometry_valid: torch.Tensor | None = None,
    xyz: torch.Tensor | None = None,
    min_observations: int = 2,
    l2_normalize: bool = True,
) -> SelectedTrackFeatureBank:
    """Aggregate selected observation descriptors into explicit 3D track features."""
    if features.ndim != 2:
        raise ValueError("features must have shape (N,C)")
    if track_ids.numel() != features.shape[0]:
        raise ValueError("track_ids length must match features")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    device = features.device
    track_ids = track_ids.to(device=device).long().view(-1)
    valid = track_ids >= 0
    if geometry_valid is not None:
        valid = valid & geometry_valid.to(device=device).bool().view(-1)
    if visibility is not None:
        visibility_weight = visibility.to(device=device, dtype=features.dtype).view(-1).clamp_min(0.0)
        valid = valid & (visibility_weight > 0.0)
    else:
        visibility_weight = torch.ones((features.shape[0],), dtype=features.dtype, device=device)
    if utility is not None:
        utility_weight = utility.to(device=device, dtype=features.dtype).view(-1).clamp_min(0.0)
    else:
        utility_weight = torch.ones((features.shape[0],), dtype=features.dtype, device=device)
    if xyz is not None and (xyz.ndim != 2 or xyz.shape[0] != features.shape[0] or xyz.shape[1] != 3):
        raise ValueError("xyz must have shape (N,3)")
    track_list = []
    centroid_list = []
    visibility_counts = []
    variance_list = []
    utility_means = []
    xyz_list = []
    for track_id in torch.unique(track_ids[valid]):
        mask = valid & (track_ids == track_id)
        obs_count = int(mask.sum().item())
        if obs_count < int(min_observations):
            continue
        vals = features.float()[mask]
        weights = (utility_weight[mask] * visibility_weight[mask]).float()
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        norm_weights = weights / weights.sum().clamp_min(1.0e-12)
        centroid = (vals * norm_weights[:, None]).sum(dim=0)
        if l2_normalize:
            centroid = F.normalize(centroid, dim=0, eps=1.0e-6)
        variance = ((vals - centroid[None]).pow(2).mean(dim=1) * norm_weights).sum()
        track_list.append(track_id)
        centroid_list.append(centroid)
        visibility_counts.append(features.new_tensor(float(obs_count)))
        variance_list.append(variance)
        utility_means.append(utility_weight[mask].float().mean())
        if xyz is not None:
            xyz_list.append((xyz.to(device=device, dtype=features.dtype)[mask].float() * norm_weights[:, None]).sum(dim=0))

    if centroid_list:
        bank_features = torch.stack(centroid_list, dim=0).to(dtype=features.dtype)
        bank_track_ids = torch.stack(track_list).long()
        bank_visibility = torch.stack(visibility_counts).long()
        bank_variance = torch.stack(variance_list).to(dtype=features.dtype)
        bank_utility = torch.stack(utility_means).to(dtype=features.dtype)
        bank_xyz = torch.stack(xyz_list, dim=0).to(dtype=features.dtype) if xyz is not None else None
    else:
        bank_features = features.new_zeros((0, features.shape[1]))
        bank_track_ids = torch.empty((0,), dtype=torch.long, device=device)
        bank_visibility = torch.empty((0,), dtype=torch.long, device=device)
        bank_variance = features.new_zeros((0,))
        bank_utility = features.new_zeros((0,))
        bank_xyz = features.new_zeros((0, 3)) if xyz is not None else None
    return SelectedTrackFeatureBank(
        track_ids=bank_track_ids,
        features=bank_features,
        visibility_count=bank_visibility,
        feature_variance=bank_variance,
        utility_mean=bank_utility,
        xyz=bank_xyz,
    )


def _reshape_render_selected(outputs: Mapping[str, torch.Tensor], batch_size: int, num_candidates: int) -> dict[str, torch.Tensor]:
    reshaped: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value) or value.shape[0] != batch_size * num_candidates:
            continue
        reshaped[key] = value.reshape(batch_size, num_candidates, *value.shape[1:])
    return reshaped


def _apply_map_adapter(
    map_adapter: torch.nn.Module | None,
    feature: torch.Tensor,
    *,
    domain: str,
    rgb: torch.Tensor | None = None,
) -> torch.Tensor:
    if map_adapter is None:
        return feature
    try:
        return map_adapter(feature, domain=domain, rgb=rgb)
    except TypeError:
        if domain == "query" and hasattr(map_adapter, "project_query"):
            return map_adapter.project_query(feature, rgb=rgb)  # type: ignore[attr-defined]
        if domain in {"render", "map"} and hasattr(map_adapter, "project_render"):
            return map_adapter.project_render(feature, rgb=rgb)  # type: ignore[attr-defined]
        return map_adapter(feature)


def score_query_with_selected_map_features(
    query_feature: torch.Tensor,
    rendered_map_feature: torch.Tensor,
    *,
    selector: torch.nn.Module,
    scorer: torch.nn.Module,
    map_adapter: torch.nn.Module | None = None,
    query_rgb: torch.Tensor | None = None,
    render_rgb: torch.Tensor | None = None,
    render_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply one selector to query/rendered map features and score hypotheses."""
    if query_feature.ndim != 4 or rendered_map_feature.ndim != 5:
        raise ValueError("query_feature must be (B,C,H,W), rendered_map_feature must be (B,K,C,H,W)")
    batch_size, num_candidates = rendered_map_feature.shape[:2]
    if query_feature.shape[0] != batch_size or query_feature.shape[1] != rendered_map_feature.shape[2]:
        raise ValueError("query and rendered map features have incompatible batch/channel dimensions")
    query_adapted = _apply_map_adapter(map_adapter, query_feature, domain="query", rgb=query_rgb)
    render_flat = rendered_map_feature.reshape(batch_size * num_candidates, *rendered_map_feature.shape[2:])
    render_rgb_flat = None
    if render_rgb is not None:
        render_rgb_flat = render_rgb.reshape(batch_size * num_candidates, *render_rgb.shape[2:])
    render_adapted = _apply_map_adapter(map_adapter, render_flat, domain="render", rgb=render_rgb_flat)
    query_selected = selector(query_adapted)
    render_selected_flat = selector(render_adapted)
    render_selected = _reshape_render_selected(render_selected_flat, batch_size, num_candidates)
    scores, scorer_aux = scorer(
        query_selected["z"],
        render_selected["z"],
        query_utility=query_selected.get("utility"),
        render_valid_mask=render_valid_mask,
    )
    return scores, {
        "query_selected": query_selected,
        "render_selected": render_selected,
        "scorer": scorer_aux,
    }


def _optional_tensor(mapping: Mapping[str, object] | None, key: str) -> torch.Tensor | None:
    if mapping is None:
        return None
    value = mapping.get(key)
    return value if torch.is_tensor(value) else None


def weak_joint_selected_feature_loss(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    selector_outputs: Mapping[str, object] | None = None,
    map_bank_aux: Mapping[str, torch.Tensor] | None = None,
    rank_weight: float = 1.0,
    hard_weight: float = 1.0,
    basin_weight: float = 0.0,
    sparsity_weight: float = 0.0,
    utility_entropy_weight: float = 0.0,
    map_variance_weight: float = 0.0,
    temperature_m: float = 0.05,
    cost_gap_m: float = 0.12,
    margin: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weakly combine selector, map adapter, and scorer supervision terms."""
    rank_loss, rank_metrics = pose_distance_soft_rank_loss(
        scores,
        pose_cost_m,
        valid_mask=valid_mask,
        temperature_m=temperature_m,
    )
    hard_loss, hard_metrics = online_score_hard_negative_loss(
        scores,
        pose_cost_m,
        valid_mask=valid_mask,
        cost_gap_m=cost_gap_m,
        margin=margin,
    )
    total = scores.sum() * 0.0 + rank_loss * float(rank_weight) + hard_loss * float(hard_weight)
    metrics: dict[str, torch.Tensor] = {
        "rank_loss": rank_loss.detach(),
        "hard_loss": hard_loss.detach(),
        **{key: value.detach() for key, value in rank_metrics.items()},
        **{key: value.detach() for key, value in hard_metrics.items()},
    }
    if basin_label is not None and float(basin_weight) != 0.0:
        basin_loss = basin_bce_loss(scores, basin_label, valid_mask=valid_mask)
        total = total + basin_loss * float(basin_weight)
        metrics["basin_loss"] = basin_loss.detach()
    channel_gate = _optional_tensor(selector_outputs, "channel_gate")
    utility = _optional_tensor(selector_outputs, "utility")
    if channel_gate is None and selector_outputs is not None and isinstance(selector_outputs.get("query_selected"), Mapping):
        channel_gate = _optional_tensor(selector_outputs["query_selected"], "channel_gate")  # type: ignore[index]
    if utility is None and selector_outputs is not None and isinstance(selector_outputs.get("query_selected"), Mapping):
        utility = _optional_tensor(selector_outputs["query_selected"], "utility")  # type: ignore[index]
    if channel_gate is not None and float(sparsity_weight) != 0.0:
        sparsity = channel_sparsity_loss(channel_gate)
        total = total + sparsity * float(sparsity_weight)
        metrics["channel_sparsity_loss"] = sparsity.detach()
    if utility is not None and float(utility_entropy_weight) != 0.0:
        utility_entropy = spatial_utility_entropy_loss(utility)
        total = total + utility_entropy * float(utility_entropy_weight)
        metrics["utility_entropy_loss"] = utility_entropy.detach()
    map_variance = _optional_tensor(map_bank_aux, "map_feature_variance")
    if map_variance is not None and float(map_variance_weight) != 0.0:
        variance_loss = map_variance.float().mean()
        total = total + variance_loss * float(map_variance_weight)
        metrics["map_variance_loss"] = variance_loss.detach()
    metrics["total_loss"] = total.detach()
    return total, metrics
