"""Full-map query-to-landmark retrieval supervision for joint localization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.landmark_feature_aggregation import LandmarkAggregationConfig, TrackPrototypeBuilder


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
    dustbin_loss_weight: float = 0.25
    dustbin_detach_descriptors: bool = False
    prototype_aggregation_method: str = "mean"
    prototype_l2_normalize_observations: bool = False
    normalize_final_prototypes: bool = True
    prototype_min_support_observations: int = 1
    set_valued_cell_positives: bool = True

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
        if str(self.prototype_aggregation_method) not in {
            "mean",
            "cosine_weighted_mean",
            "geometry_weighted",
        }:
            raise ValueError("unsupported training-time prototype aggregation method")
        if int(self.prototype_min_support_observations) <= 0:
            raise ValueError("prototype_min_support_observations must be positive")
        if self.dustbin_logit is not None and not np.isfinite(float(self.dustbin_logit)):
            raise ValueError("dustbin_logit must be finite or None")
        if float(self.dustbin_loss_weight) < 0.0:
            raise ValueError("dustbin_loss_weight must be non-negative")


class LandmarkPrototypeMemoryBank:
    """Bounded EMA bank of detached, multi-observation track prototypes."""

    def __init__(
        self,
        *,
        capacity: int,
        descriptor_dim: int,
        device: torch.device | str,
        momentum: float = 0.9,
        frozen: bool = False,
        source_path: str = "",
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
        self.frozen = bool(frozen)
        self.source_path = str(source_path)
        self.snapshot_metadata: dict[str, object] = {}
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

    @classmethod
    def from_projected_landmark_npz(
        cls,
        path: Path,
        *,
        device: torch.device | str,
        expected_descriptor_dim: int,
    ) -> "LandmarkPrototypeMemoryBank":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            track_ids = np.asarray(data["track_ids"], dtype=np.int64).reshape(-1)
            descriptors = np.asarray(data["features"], dtype=np.float32)
            xyz = np.asarray(data["xyz"], dtype=np.float32).reshape(-1, 3)
            counts = np.asarray(data["observation_counts"], dtype=np.int64).reshape(-1)
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        if descriptors.ndim != 2 or descriptors.shape[0] != track_ids.shape[0]:
            raise ValueError("frozen landmark snapshot has inconsistent track_ids/features")
        if int(descriptors.shape[1]) != int(expected_descriptor_dim):
            raise ValueError(
                "frozen landmark snapshot descriptor dimension mismatch: "
                f"snapshot={descriptors.shape[1]}, model={int(expected_descriptor_dim)}"
            )
        if len(set(track_ids.tolist())) != int(track_ids.size):
            raise ValueError("frozen landmark snapshot contains duplicate track ids")
        if xyz.shape[0] != track_ids.shape[0] or counts.shape[0] != track_ids.shape[0]:
            raise ValueError("frozen landmark snapshot xyz/count arrays do not match track ids")
        bank = cls(
            capacity=max(1, int(track_ids.size)),
            descriptor_dim=int(expected_descriptor_dim),
            device=device,
            momentum=0.0,
            frozen=True,
            source_path=str(source),
        )
        bank._size = int(track_ids.size)
        bank.track_ids[: bank._size] = track_ids
        bank.observation_counts[: bank._size] = counts
        bank.descriptors[: bank._size] = F.normalize(
            torch.as_tensor(descriptors, dtype=torch.float32, device=bank.device),
            dim=1,
        )
        bank.xyz[: bank._size] = torch.as_tensor(xyz, dtype=torch.float32, device=bank.device)
        bank._slot_by_track = {int(track_id): int(slot) for slot, track_id in enumerate(track_ids.tolist())}
        bank.snapshot_metadata = dict(metadata)
        return bank

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
        if bool(self.frozen):
            return
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
    config: LandmarkRetrievalLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    builder = TrackPrototypeBuilder(
        aggregation=LandmarkAggregationConfig(
            method=str(config.prototype_aggregation_method),
            min_observations=1,
            l2_normalize_observations=bool(config.prototype_l2_normalize_observations),
        ),
        normalize_final_prototypes=bool(config.normalize_final_prototypes),
    )
    return builder.aggregate_torch(descriptors, inverse, int(track_count))


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
    query_image_group_ids: torch.Tensor | None = None,
    dustbin_logits: torch.Tensor | None = None,
    unmatched_query_descriptors: torch.Tensor | None = None,
    unmatched_dustbin_logits: torch.Tensor | None = None,
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
    image_groups = None
    if query_image_group_ids is not None:
        image_groups = query_image_group_ids.to(device=query_descriptors.device, dtype=torch.long).reshape(-1)
        if image_groups.shape[0] != ids.shape[0]:
            raise ValueError("query_image_group_ids must contain one value per descriptor pair")
    valid = (ids >= 0) & torch.isfinite(query_descriptors).all(dim=1) & torch.isfinite(support_descriptors).all(dim=1)
    if not torch.any(valid):
        return None, {"landmark_retrieval_valid_count": 0.0}
    query = F.normalize(query_descriptors[valid], dim=1)
    valid_dustbin_logits = None
    if dustbin_logits is not None:
        all_dustbin_logits = dustbin_logits.to(device=query.device, dtype=query.dtype).reshape(-1)
        if all_dustbin_logits.shape[0] != valid.shape[0]:
            raise ValueError("dustbin_logits must contain one value per query descriptor")
        valid_dustbin_logits = all_dustbin_logits[valid]
    support = support_descriptors[valid]
    ids = ids[valid]
    if groups is not None:
        groups = groups[valid]
    if image_groups is not None:
        image_groups = image_groups[valid]
    xyz_rows = None if track_xyz is None else track_xyz.to(device=query.device, dtype=torch.float32).reshape(-1, 3)[valid]
    all_unique_ids, all_inverse = torch.unique(ids, sorted=True, return_inverse=True)
    all_prototypes, all_counts = _aggregate_track_rows(
        support,
        all_inverse,
        int(all_unique_ids.numel()),
        cfg,
    )
    all_xyz = _aggregate_track_xyz(xyz_rows, all_inverse, int(all_unique_ids.numel()))
    eligible_tracks = all_counts >= float(cfg.prototype_min_support_observations)
    if not torch.any(eligible_tracks):
        return None, {
            "landmark_retrieval_valid_count": 0.0,
            "landmark_retrieval_support_observation_count": float(query.shape[0]),
            "landmark_retrieval_eligible_track_count": 0.0,
            "landmark_retrieval_dropped_singleton_track_count": float(all_unique_ids.numel()),
        }
    old_to_new = torch.full((all_unique_ids.numel(),), -1, dtype=torch.long, device=query.device)
    old_to_new[eligible_tracks] = torch.arange(int(torch.count_nonzero(eligible_tracks)), device=query.device)
    eligible_rows = eligible_tracks[all_inverse]
    query = query[eligible_rows]
    if valid_dustbin_logits is not None:
        valid_dustbin_logits = valid_dustbin_logits[eligible_rows]
    ids = ids[eligible_rows]
    labels = old_to_new[all_inverse[eligible_rows]]
    if groups is not None:
        groups = groups[eligible_rows]
    if image_groups is not None:
        image_groups = image_groups[eligible_rows]
    unique_ids = all_unique_ids[eligible_tracks]
    current_prototypes = all_prototypes[eligible_tracks]
    current_counts = all_counts[eligible_tracks]
    current_xyz = all_xyz[eligible_tracks]
    total_valid_support_count = int(support.shape[0])
    eligible_support_count = int(torch.count_nonzero(eligible_rows).item())
    input_query_count = int(query.shape[0])
    if image_groups is not None:
        keep_rows: list[int] = []
        seen_query_tracks: set[tuple[int, int]] = set()
        for row, (image_group, track_id) in enumerate(
            zip(image_groups.detach().cpu().tolist(), ids.detach().cpu().tolist())
        ):
            key = (int(image_group), int(track_id))
            if key in seen_query_tracks:
                continue
            seen_query_tracks.add(key)
            keep_rows.append(int(row))
        keep = torch.as_tensor(keep_rows, dtype=torch.long, device=query.device)
        query = query[keep]
        if valid_dustbin_logits is not None:
            valid_dustbin_logits = valid_dustbin_logits[keep]
        labels = labels[keep]
        image_groups = image_groups[keep]
        if groups is not None:
            groups = groups[keep]

    history_found = torch.zeros((unique_ids.numel(),), dtype=torch.bool, device=query.device)
    history_counts = np.zeros((unique_ids.numel(),), dtype=np.int64)
    positive_prototypes = current_prototypes
    if memory_bank is not None and len(memory_bank) > 0 and not bool(memory_bank.frozen):
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
    unmatched_query = query.new_zeros((0, query.shape[1]))
    valid_unmatched_dustbin_logits = None
    if unmatched_query_descriptors is not None:
        unmatched_values = unmatched_query_descriptors.to(device=query.device, dtype=query.dtype).reshape(
            -1, query.shape[1]
        )
        unmatched_valid = torch.isfinite(unmatched_values).all(dim=1)
        unmatched_query = F.normalize(unmatched_values[unmatched_valid], dim=1)
        if unmatched_dustbin_logits is not None:
            all_unmatched_dustbin_logits = unmatched_dustbin_logits.to(
                device=query.device,
                dtype=query.dtype,
            ).reshape(-1)
            if all_unmatched_dustbin_logits.shape[0] != unmatched_values.shape[0]:
                raise ValueError("unmatched_dustbin_logits must contain one value per unmatched descriptor")
            valid_unmatched_dustbin_logits = all_unmatched_dustbin_logits[unmatched_valid]
    elif unmatched_dustbin_logits is not None:
        raise ValueError("unmatched_dustbin_logits requires unmatched_query_descriptors")
    if memory_descriptors.shape[0] > 0:
        semantic_queries = query.detach()
        if unmatched_query.shape[0] > 0:
            semantic_queries = torch.cat([semantic_queries, unmatched_query.detach()], dim=0)
        detached_scores = semantic_queries @ memory_descriptors.T
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
    if valid_dustbin_logits is not None:
        logits = torch.cat([logits, valid_dustbin_logits[:, None]], dim=1)
    elif cfg.dustbin_logit is not None and np.isfinite(float(cfg.dustbin_logit)):
        dustbin = torch.full((logits.shape[0], 1), float(cfg.dustbin_logit), dtype=logits.dtype, device=logits.device)
        logits = torch.cat([logits, dustbin], dim=1)
    same_group_track_mask = torch.zeros(
        (int(query.shape[0]), int(unique_ids.numel())),
        dtype=torch.bool,
        device=query.device,
    )
    same_group_track_mask[torch.arange(query.shape[0], device=query.device), labels] = True
    if groups is not None:
        same_group_rows = groups[:, None] == groups[None, :]
        row_track_membership = F.one_hot(labels, num_classes=int(unique_ids.numel())).to(dtype=torch.float32)
        same_group_track_mask = (same_group_rows.to(dtype=torch.float32) @ row_track_membership) > 0
    if bool(cfg.set_valued_cell_positives):
        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        positive_mask[:, : int(unique_ids.numel())] = same_group_track_mask
        positive_logsumexp = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
        retrieval_loss = (torch.logsumexp(logits, dim=1) - positive_logsumexp).mean()
    else:
        denominator_keep = torch.ones_like(logits, dtype=torch.bool)
        denominator_keep[:, : int(unique_ids.numel())] = ~same_group_track_mask
        denominator_keep[torch.arange(logits.shape[0], device=logits.device), labels] = True
        masked_logits = logits.masked_fill(~denominator_keep, float("-inf"))
        retrieval_loss = F.cross_entropy(masked_logits, labels)

    unmatched_logits = logits.new_zeros((0, int(candidate_descriptors.shape[0]) + 1))
    dustbin_loss = logits.new_tensor(0.0)
    if unmatched_query.shape[0] > 0:
        unmatched_for_scores = unmatched_query.detach() if bool(cfg.dustbin_detach_descriptors) else unmatched_query
        candidates_for_scores = (
            candidate_descriptors.detach() if bool(cfg.dustbin_detach_descriptors) else candidate_descriptors
        )
        unmatched_logits = (unmatched_for_scores @ candidates_for_scores.T) / float(cfg.temperature)
        if valid_unmatched_dustbin_logits is not None:
            unmatched_logits = torch.cat([unmatched_logits, valid_unmatched_dustbin_logits[:, None]], dim=1)
        elif cfg.dustbin_logit is not None and np.isfinite(float(cfg.dustbin_logit)):
            dustbin = torch.full(
                (unmatched_logits.shape[0], 1),
                float(cfg.dustbin_logit),
                dtype=unmatched_logits.dtype,
                device=unmatched_logits.device,
            )
            unmatched_logits = torch.cat([unmatched_logits, dustbin], dim=1)
        else:
            raise ValueError("unmatched landmark queries require a learned or fixed dustbin logit")
        dustbin_targets = torch.full(
            (unmatched_logits.shape[0],),
            int(unmatched_logits.shape[1] - 1),
            dtype=torch.long,
            device=unmatched_logits.device,
        )
        dustbin_loss = F.cross_entropy(unmatched_logits, dustbin_targets)
    loss = retrieval_loss + float(cfg.dustbin_loss_weight) * dustbin_loss

    with torch.no_grad():
        positive_scores = cosine_scores[torch.arange(cosine_scores.shape[0], device=query.device), labels]
        negative_mask = torch.ones_like(cosine_scores, dtype=torch.bool)
        if bool(cfg.set_valued_cell_positives):
            negative_mask[:, : int(unique_ids.numel())] &= ~same_group_track_mask
        else:
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
            "landmark_retrieval_positive_loss": float(retrieval_loss.detach().cpu().item()),
            "landmark_retrieval_dustbin_loss": float(dustbin_loss.detach().cpu().item()),
            "landmark_retrieval_dustbin_positive_count": float(unmatched_query.shape[0]),
            "landmark_retrieval_valid_count": float(query.shape[0]),
            "landmark_retrieval_support_observation_count": float(input_query_count),
            "landmark_retrieval_deduplicated_query_count": float(input_query_count - int(query.shape[0])),
            "landmark_retrieval_total_valid_support_observation_count": float(total_valid_support_count),
            "landmark_retrieval_eligible_support_observation_count": float(eligible_support_count),
            "landmark_retrieval_eligible_track_count": float(unique_ids.numel()),
            "landmark_retrieval_dropped_singleton_track_count": float(
                all_unique_ids.numel() - unique_ids.numel()
            ),
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
            "landmark_retrieval_set_valued_cell_positives": float(bool(cfg.set_valued_cell_positives)),
        }
        if valid_dustbin_logits is not None:
            valid_dustbin_prob = torch.softmax(logits, dim=1)[:, -1]
            metrics["landmark_retrieval_valid_dustbin_probability_mean"] = float(valid_dustbin_prob.mean().item())
            metrics["landmark_retrieval_valid_accept_rate_0p5"] = float(
                (valid_dustbin_prob < 0.5).float().mean().item()
            )
        if unmatched_query.shape[0] > 0:
            unmatched_dustbin_prob = torch.softmax(unmatched_logits, dim=1)[:, -1]
            metrics["landmark_retrieval_unmatched_dustbin_probability_mean"] = float(
                unmatched_dustbin_prob.mean().item()
            )
            metrics["landmark_retrieval_dustbin_recall_0p5"] = float(
                (unmatched_dustbin_prob >= 0.5).float().mean().item()
            )
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
