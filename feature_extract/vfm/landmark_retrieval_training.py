"""Full-map query-to-landmark retrieval supervision for joint localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LandmarkRetrievalLossConfig:
    temperature: float = 0.07
    prototype_history_mix: float = 0.5
    memory_candidate_pool_size: int = 4096
    semantic_hard_negatives_per_query: int = 16
    geometry_hard_negatives_per_track: int = 8
    random_negatives: int = 128
    max_memory_negatives: int = 2048
    dustbin_logit: float | None = 0.0

    def __post_init__(self) -> None:
        if float(self.temperature) <= 0.0:
            raise ValueError("landmark retrieval temperature must be positive")
        if not 0.0 <= float(self.prototype_history_mix) <= 1.0:
            raise ValueError("prototype_history_mix must be in [0, 1]")
        for name in (
            "memory_candidate_pool_size",
            "semantic_hard_negatives_per_query",
            "geometry_hard_negatives_per_track",
            "random_negatives",
            "max_memory_negatives",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


class LandmarkPrototypeMemoryBank:
    """Bounded EMA bank of detached, multi-observation track prototypes."""

    def __init__(
        self,
        *,
        capacity: int,
        descriptor_dim: int,
        device: torch.device | str,
        momentum: float = 0.9,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("landmark memory capacity must be positive")
        if int(descriptor_dim) <= 0:
            raise ValueError("descriptor_dim must be positive")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("landmark memory momentum must be in [0, 1)")
        self.capacity = int(capacity)
        self.descriptor_dim = int(descriptor_dim)
        self.device = torch.device(device)
        self.momentum = float(momentum)
        self.descriptors = torch.zeros(
            (self.capacity, self.descriptor_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.xyz = torch.full((self.capacity, 3), float("nan"), dtype=torch.float32, device=self.device)
        self.observation_counts = np.zeros((self.capacity,), dtype=np.int64)
        self.track_ids = np.full((self.capacity,), -1, dtype=np.int64)
        self._slot_by_track: dict[int, int] = {}
        self._size = 0
        self._next_evict = 0

    def __len__(self) -> int:
        return int(self._size)

    def _allocate_slot(self, track_id: int) -> int:
        if self._size < self.capacity:
            slot = int(self._size)
            self._size += 1
        else:
            slot = int(self._next_evict)
            old_track_id = int(self.track_ids[slot])
            self._slot_by_track.pop(old_track_id, None)
            self._next_evict = (self._next_evict + 1) % self.capacity
        self.track_ids[slot] = int(track_id)
        self._slot_by_track[int(track_id)] = int(slot)
        return int(slot)

    def lookup(
        self,
        track_ids: Sequence[int] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        descriptors = torch.zeros((ids.size, self.descriptor_dim), dtype=torch.float32, device=self.device)
        xyz = torch.full((ids.size, 3), float("nan"), dtype=torch.float32, device=self.device)
        found = torch.zeros((ids.size,), dtype=torch.bool, device=self.device)
        counts = np.zeros((ids.size,), dtype=np.int64)
        for row, track_id in enumerate(ids.tolist()):
            slot = self._slot_by_track.get(int(track_id))
            if slot is None:
                continue
            descriptors[row] = self.descriptors[int(slot)].detach().clone()
            xyz[row] = self.xyz[int(slot)].detach().clone()
            found[row] = True
            counts[row] = int(self.observation_counts[int(slot)])
        return descriptors, found, counts, xyz

    def candidates(
        self,
        *,
        exclude_track_ids: Sequence[int] | np.ndarray,
        max_count: int,
        seed: int,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        excluded = set(np.asarray(exclude_track_ids, dtype=np.int64).reshape(-1).tolist())
        slots = [slot for slot in range(self._size) if int(self.track_ids[slot]) not in excluded]
        limit = int(max_count)
        if limit > 0 and len(slots) > limit:
            rng = np.random.default_rng(int(seed))
            slots = np.sort(rng.choice(np.asarray(slots, dtype=np.int64), size=limit, replace=False)).tolist()
        if not slots:
            return (
                torch.zeros((0, self.descriptor_dim), dtype=torch.float32, device=self.device),
                np.zeros((0,), dtype=np.int64),
                torch.zeros((0, 3), dtype=torch.float32, device=self.device),
            )
        slot_array = np.asarray(slots, dtype=np.int64)
        slot_tensor = torch.as_tensor(slot_array, dtype=torch.long, device=self.device)
        return (
            self.descriptors[slot_tensor].detach().clone(),
            self.track_ids[slot_array].copy(),
            self.xyz[slot_tensor].detach().clone(),
        )

    @torch.no_grad()
    def update(
        self,
        *,
        track_ids: Sequence[int] | np.ndarray,
        descriptors: torch.Tensor,
        observation_counts: Sequence[int] | np.ndarray,
        xyz: torch.Tensor | None = None,
    ) -> None:
        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        counts = np.asarray(observation_counts, dtype=np.int64).reshape(-1)
        values = F.normalize(descriptors.detach().to(device=self.device, dtype=torch.float32), dim=1)
        if values.shape != (ids.size, self.descriptor_dim):
            raise ValueError("memory update descriptors must match track_ids and descriptor_dim")
        if counts.shape[0] != ids.shape[0]:
            raise ValueError("observation_counts must contain one value per track")
        xyz_values = None
        if xyz is not None:
            xyz_values = xyz.detach().to(device=self.device, dtype=torch.float32).reshape(-1, 3)
            if xyz_values.shape[0] != ids.shape[0]:
                raise ValueError("xyz must contain one 3D point per track")
        for row, track_id in enumerate(ids.tolist()):
            if int(track_id) < 0:
                continue
            slot = self._slot_by_track.get(int(track_id))
            if slot is None:
                slot = self._allocate_slot(int(track_id))
                self.descriptors[slot] = values[row]
                self.observation_counts[slot] = max(1, int(counts[row]))
            else:
                mixed = self.momentum * self.descriptors[slot] + (1.0 - self.momentum) * values[row]
                self.descriptors[slot] = F.normalize(mixed, dim=0)
                self.observation_counts[slot] += max(1, int(counts[row]))
            if xyz_values is not None and bool(torch.isfinite(xyz_values[row]).all()):
                self.xyz[slot] = xyz_values[row]


def _aggregate_track_rows(
    descriptors: torch.Tensor,
    inverse: torch.Tensor,
    track_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(descriptors, dim=1)
    sums = torch.zeros((int(track_count), int(normalized.shape[1])), dtype=normalized.dtype, device=normalized.device)
    sums.index_add_(0, inverse, normalized)
    counts = torch.bincount(inverse, minlength=int(track_count)).to(dtype=normalized.dtype)
    prototypes = F.normalize(sums / counts.clamp_min(1.0)[:, None], dim=1)
    return prototypes, counts


def _aggregate_track_xyz(xyz: torch.Tensor | None, inverse: torch.Tensor, track_count: int) -> torch.Tensor:
    if xyz is None:
        return torch.full((int(track_count), 3), float("nan"), dtype=torch.float32, device=inverse.device)
    values = xyz.to(device=inverse.device, dtype=torch.float32).reshape(-1, 3)
    finite = torch.isfinite(values).all(dim=1)
    sums = torch.zeros((int(track_count), 3), dtype=torch.float32, device=inverse.device)
    counts = torch.zeros((int(track_count),), dtype=torch.float32, device=inverse.device)
    if torch.any(finite):
        sums.index_add_(0, inverse[finite], values[finite])
        counts.index_add_(0, inverse[finite], torch.ones_like(inverse[finite], dtype=torch.float32))
    output = sums / counts.clamp_min(1.0)[:, None]
    output[counts == 0] = float("nan")
    return output


def _ordered_unique(indices: Sequence[int], *, limit: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in indices:
        index = int(value)
        if index in seen:
            continue
        seen.add(index)
        output.append(index)
        if int(limit) > 0 and len(output) >= int(limit):
            break
    return output


def landmark_retrieval_loss(
    query_descriptors: torch.Tensor,
    support_descriptors: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    track_xyz: torch.Tensor | None = None,
    query_group_ids: torch.Tensor | None = None,
    memory_bank: LandmarkPrototypeMemoryBank | None = None,
    config: LandmarkRetrievalLossConfig | None = None,
    update_memory: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Retrieve aggregated track prototypes from full-map query descriptors."""

    cfg = config or LandmarkRetrievalLossConfig()
    if query_descriptors.ndim != 2 or support_descriptors.shape != query_descriptors.shape:
        raise ValueError("query/support descriptors must have the same shape (N, C)")
    ids = track_ids.to(device=query_descriptors.device, dtype=torch.long).reshape(-1)
    if ids.shape[0] != query_descriptors.shape[0]:
        raise ValueError("track_ids must contain one value per descriptor pair")
    groups = None
    if query_group_ids is not None:
        groups = query_group_ids.to(device=query_descriptors.device, dtype=torch.long).reshape(-1)
        if groups.shape[0] != ids.shape[0]:
            raise ValueError("query_group_ids must contain one value per descriptor pair")
    valid = (ids >= 0) & torch.isfinite(query_descriptors).all(dim=1) & torch.isfinite(support_descriptors).all(dim=1)
    if not torch.any(valid):
        return None, {"landmark_retrieval_valid_count": 0.0}
    query = F.normalize(query_descriptors[valid], dim=1)
    support = support_descriptors[valid]
    ids = ids[valid]
    if groups is not None:
        groups = groups[valid]
    xyz_rows = None if track_xyz is None else track_xyz.to(device=query.device, dtype=torch.float32).reshape(-1, 3)[valid]
    unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    current_prototypes, current_counts = _aggregate_track_rows(support, inverse, int(unique_ids.numel()))
    current_xyz = _aggregate_track_xyz(xyz_rows, inverse, int(unique_ids.numel()))

    history_found = torch.zeros((unique_ids.numel(),), dtype=torch.bool, device=query.device)
    history_counts = np.zeros((unique_ids.numel(),), dtype=np.int64)
    positive_prototypes = current_prototypes
    if memory_bank is not None and len(memory_bank) > 0:
        history, history_found, history_counts, _history_xyz = memory_bank.lookup(unique_ids.detach().cpu().numpy())
        mix = float(cfg.prototype_history_mix)
        weights = history_found.to(dtype=current_prototypes.dtype)[:, None] * mix
        positive_prototypes = F.normalize((1.0 - weights) * current_prototypes + weights * history, dim=1)

    memory_descriptors = torch.zeros((0, query.shape[1]), dtype=query.dtype, device=query.device)
    memory_xyz = torch.zeros((0, 3), dtype=torch.float32, device=query.device)
    selected_memory_indices: list[int] = []
    semantic_selected: list[int] = []
    geometry_selected: list[int] = []
    random_selected: list[int] = []
    if memory_bank is not None and len(memory_bank) > 0:
        memory_descriptors, _memory_track_ids, memory_xyz = memory_bank.candidates(
            exclude_track_ids=unique_ids.detach().cpu().numpy(),
            max_count=int(cfg.memory_candidate_pool_size),
            seed=int(seed),
        )
    if memory_descriptors.shape[0] > 0:
        detached_scores = query.detach() @ memory_descriptors.T
        semantic_k = min(int(cfg.semantic_hard_negatives_per_query), int(memory_descriptors.shape[0]))
        if semantic_k > 0:
            semantic_selected = torch.topk(detached_scores, k=semantic_k, dim=1).indices.reshape(-1).detach().cpu().tolist()
        geometry_k = min(int(cfg.geometry_hard_negatives_per_track), int(memory_descriptors.shape[0]))
        finite_current = torch.isfinite(current_xyz).all(dim=1)
        finite_memory = torch.isfinite(memory_xyz).all(dim=1)
        if geometry_k > 0 and torch.any(finite_current) and torch.any(finite_memory):
            finite_memory_indices = torch.nonzero(finite_memory, as_tuple=False).reshape(-1)
            distances = torch.cdist(current_xyz[finite_current], memory_xyz[finite_memory])
            local_k = min(geometry_k, int(finite_memory_indices.numel()))
            geometry_local = torch.topk(distances, k=local_k, dim=1, largest=False).indices.reshape(-1)
            geometry_selected = finite_memory_indices[geometry_local].detach().cpu().tolist()
        random_k = min(int(cfg.random_negatives), int(memory_descriptors.shape[0]))
        if random_k > 0:
            rng = np.random.default_rng(int(seed) + 7919)
            random_selected = rng.choice(int(memory_descriptors.shape[0]), size=random_k, replace=False).tolist()
        selected_memory_indices = _ordered_unique(
            [*semantic_selected, *geometry_selected, *random_selected],
            limit=int(cfg.max_memory_negatives),
        )

    candidate_descriptors = positive_prototypes
    if selected_memory_indices:
        selected = torch.as_tensor(selected_memory_indices, dtype=torch.long, device=query.device)
        candidate_descriptors = torch.cat([candidate_descriptors, memory_descriptors[selected]], dim=0)
    cosine_scores = query @ candidate_descriptors.T
    logits = cosine_scores / float(cfg.temperature)
    if cfg.dustbin_logit is not None and np.isfinite(float(cfg.dustbin_logit)):
        dustbin = torch.full((logits.shape[0], 1), float(cfg.dustbin_logit), dtype=logits.dtype, device=logits.device)
        logits = torch.cat([logits, dustbin], dim=1)
    labels = inverse
    same_group_track_mask = torch.zeros(
        (int(query.shape[0]), int(unique_ids.numel())),
        dtype=torch.bool,
        device=query.device,
    )
    same_group_track_mask[torch.arange(query.shape[0], device=query.device), labels] = True
    if groups is not None:
        same_group_rows = groups[:, None] == groups[None, :]
        row_track_membership = F.one_hot(inverse, num_classes=int(unique_ids.numel())).to(dtype=torch.float32)
        same_group_track_mask = (same_group_rows.to(dtype=torch.float32) @ row_track_membership) > 0
    denominator_keep = torch.ones_like(logits, dtype=torch.bool)
    denominator_keep[:, : int(unique_ids.numel())] = ~same_group_track_mask
    denominator_keep[torch.arange(logits.shape[0], device=logits.device), labels] = True
    masked_logits = logits.masked_fill(~denominator_keep, float("-inf"))
    loss = F.cross_entropy(masked_logits, labels)

    with torch.no_grad():
        positive_scores = cosine_scores[torch.arange(cosine_scores.shape[0], device=query.device), labels]
        negative_mask = torch.ones_like(cosine_scores, dtype=torch.bool)
        negative_mask[torch.arange(cosine_scores.shape[0], device=query.device), labels] = False
        if torch.any(negative_mask):
            hardest_negative = cosine_scores.masked_fill(~negative_mask, -1.0).max(dim=1).values
            score_gap = positive_scores - hardest_negative
            hardest_negative_mean = float(hardest_negative.mean().item())
            score_gap_mean = float(score_gap.mean().item())
        else:
            hardest_negative_mean = float("nan")
            score_gap_mean = float("nan")
        metrics: dict[str, float] = {
            "landmark_retrieval_loss": float(loss.detach().cpu().item()),
            "landmark_retrieval_valid_count": float(query.shape[0]),
            "landmark_retrieval_track_count": float(unique_ids.numel()),
            "landmark_retrieval_candidate_count": float(logits.shape[1]),
            "landmark_retrieval_memory_negative_count": float(len(selected_memory_indices)),
            "landmark_retrieval_semantic_hard_negative_count": float(len(set(semantic_selected))),
            "landmark_retrieval_geometry_hard_negative_count": float(len(set(geometry_selected))),
            "landmark_retrieval_random_negative_count": float(len(set(random_selected))),
            "landmark_retrieval_positive_score_mean": float(positive_scores.mean().item()),
            "landmark_retrieval_hardest_negative_score_mean": hardest_negative_mean,
            "landmark_retrieval_score_gap_mean": score_gap_mean,
            "landmark_retrieval_history_positive_fraction": float(history_found.float().mean().item()),
            "landmark_retrieval_mean_prototype_observation_count": float(
                np.mean(current_counts.detach().cpu().numpy() + history_counts)
            ),
            "landmark_retrieval_geometry_available_fraction": float(torch.isfinite(current_xyz).all(dim=1).float().mean().item()),
            "landmark_retrieval_same_cell_false_negatives_excluded_mean": float(
                (same_group_track_mask.sum(dim=1) - 1).clamp_min(0).float().mean().item()
            ),
            "landmark_retrieval_memory_size_before": float(0 if memory_bank is None else len(memory_bank)),
        }
        ranking = torch.argsort(logits, dim=1, descending=True)
        valid_group_candidates = torch.zeros_like(logits, dtype=torch.bool)
        valid_group_candidates[:, : int(unique_ids.numel())] = same_group_track_mask
        for k in (1, 5, 20):
            take = min(int(k), int(ranking.shape[1]))
            hit = torch.any(ranking[:, :take] == labels[:, None], dim=1)
            metrics[f"landmark_retrieval_recall_at_{k}"] = float(hit.float().mean().item())
            valid_hit = torch.gather(valid_group_candidates, 1, ranking[:, :take]).any(dim=1)
            metrics[f"landmark_retrieval_same_cell_valid_recall_at_{k}"] = float(valid_hit.float().mean().item())

    if memory_bank is not None and bool(update_memory):
        memory_bank.update(
            track_ids=unique_ids.detach().cpu().numpy(),
            descriptors=current_prototypes,
            observation_counts=current_counts.detach().cpu().numpy().astype(np.int64),
            xyz=current_xyz,
        )
    metrics["landmark_retrieval_memory_size_after"] = float(0 if memory_bank is None else len(memory_bank))
    return loss, metrics
