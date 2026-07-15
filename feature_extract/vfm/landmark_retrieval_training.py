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
    # Zero searches the complete frozen landmark bank. A positive limit keeps
    # the historical deterministic random pre-sampling behavior.
    memory_candidate_pool_size: int = 0
    semantic_hard_negatives_per_query: int = 16
    geometry_hard_negatives_per_track: int = 8
    random_negatives: int = 128
    max_memory_negatives: int = 2048
    memory_negative_merge_policy: str = "source_balanced_round_robin"
    system_hard_negative_margin: float = 0.05
    system_hard_negative_margin_weight: float = 0.0
    dustbin_logit: float | None = 0.0
    dustbin_loss_weight: float = 0.25
    dustbin_detach_descriptors: bool = False
    prototype_aggregation_method: str = "mean"
    prototype_l2_normalize_observations: bool = False
    normalize_final_prototypes: bool = True
    prototype_min_support_observations: int = 1
    positive_prototype_source: str = "episode_support_observations"
    set_valued_cell_positives: bool = True
    exclude_known_cell_positives_from_memory: bool = True

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
        if str(self.memory_negative_merge_policy) not in {
            "legacy_source_concat",
            "source_balanced_round_robin",
        }:
            raise ValueError("unsupported memory negative merge policy")
        if float(self.system_hard_negative_margin) < 0.0:
            raise ValueError("system_hard_negative_margin must be non-negative")
        if float(self.system_hard_negative_margin_weight) < 0.0:
            raise ValueError("system_hard_negative_margin_weight must be non-negative")
        if str(self.prototype_aggregation_method) not in {
            "mean",
            "cosine_weighted_mean",
            "geometry_weighted",
        }:
            raise ValueError("unsupported training-time prototype aggregation method")
        if int(self.prototype_min_support_observations) <= 0:
            raise ValueError("prototype_min_support_observations must be positive")
        if str(self.positive_prototype_source) not in {
            "episode_support_observations",
            "query_disjoint_frozen_bank",
        }:
            raise ValueError("unsupported landmark positive prototype source")
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
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
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
                np.zeros((0,), dtype=np.int64),
            )
        slot_array = np.asarray(slots, dtype=np.int64)
        slot_tensor = torch.as_tensor(slot_array, dtype=torch.long, device=self.device)
        return (
            self.descriptors[slot_tensor].detach().clone(),
            self.track_ids[slot_array].copy(),
            self.xyz[slot_tensor].detach().clone(),
            slot_array,
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


def _merge_memory_negative_sources(
    *,
    semantic: Sequence[int],
    geometry: Sequence[int],
    random: Sequence[int],
    limit: int,
    policy: str,
) -> tuple[list[int], dict[str, int]]:
    """Merge hard-negative sources without starving rows or source types."""

    source_values = {
        "semantic": _ordered_unique(semantic, limit=0),
        "geometry": _ordered_unique(geometry, limit=0),
        "random": _ordered_unique(random, limit=0),
    }
    maximum = int(limit)
    if maximum < 0:
        raise ValueError("memory negative limit must be non-negative")
    if str(policy) == "legacy_source_concat":
        selected = _ordered_unique(
            [
                *source_values["semantic"],
                *source_values["geometry"],
                *source_values["random"],
            ],
            limit=maximum,
        )
        selected_set = set(selected)
        # Legacy attribution follows source order, matching the actual concat.
        attributed: set[int] = set()
        counts: dict[str, int] = {}
        for source in ("semantic", "geometry", "random"):
            admitted = [
                index
                for index in source_values[source]
                if index in selected_set and index not in attributed
            ]
            attributed.update(admitted)
            counts[source] = int(len(admitted))
        return selected, counts
    if str(policy) != "source_balanced_round_robin":
        raise ValueError("unsupported memory negative merge policy")

    # Semantic confusers receive two slots per cycle. Geometry confusers and
    # random regularizers each receive one, so a large semantic list cannot
    # silently consume the complete shared budget.
    schedule = ("semantic", "semantic", "geometry", "random")
    cursors = {source: 0 for source in source_values}
    selected: list[int] = []
    selected_set: set[int] = set()
    counts = {source: 0 for source in source_values}
    while maximum == 0 or len(selected) < maximum:
        made_progress = False
        for source in schedule:
            if maximum > 0 and len(selected) >= maximum:
                break
            values = source_values[source]
            while cursors[source] < len(values):
                index = int(values[cursors[source]])
                cursors[source] += 1
                if index in selected_set:
                    continue
                selected.append(index)
                selected_set.add(index)
                counts[source] += 1
                made_progress = True
                break
        if not made_progress:
            break
    return selected, counts


def _known_positive_candidate_pairs(
    known_positive_track_ids: torch.Tensor | None,
    *,
    memory_bank: LandmarkPrototypeMemoryBank,
    candidate_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map padded per-query track ids to sparse candidate score coordinates."""

    if known_positive_track_ids is None or known_positive_track_ids.numel() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    slots = np.asarray(candidate_slots, dtype=np.int64).reshape(-1)
    candidate_row_by_slot = np.full((int(memory_bank.capacity),), -1, dtype=np.int64)
    candidate_row_by_slot[slots] = np.arange(slots.size, dtype=np.int64)
    known = known_positive_track_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    query_rows: list[int] = []
    candidate_rows: list[int] = []
    for query_row, values in enumerate(known):
        selected: set[int] = set()
        for track_id in values.tolist():
            if int(track_id) < 0:
                continue
            slot = memory_bank._slot_by_track.get(int(track_id))
            if slot is None:
                continue
            candidate_row = int(candidate_row_by_slot[int(slot)])
            if candidate_row >= 0:
                selected.add(candidate_row)
        for candidate_row in sorted(selected):
            query_rows.append(int(query_row))
            candidate_rows.append(int(candidate_row))
    return np.asarray(query_rows, dtype=np.int64), np.asarray(candidate_rows, dtype=np.int64)


def landmark_retrieval_loss(
    query_descriptors: torch.Tensor,
    support_descriptors: torch.Tensor,
    track_ids: torch.Tensor,
    *,
    track_xyz: torch.Tensor | None = None,
    query_group_ids: torch.Tensor | None = None,
    query_image_group_ids: torch.Tensor | None = None,
    # Coarse-cell ambiguity set. These tracks are excluded from negatives but
    # are not automatically valid pose-level positives.
    known_positive_track_ids: torch.Tensor | None = None,
    # Strict pixel-radius alternatives may enter the set-valued numerator only
    # when their prototype is built from this episode's independent support
    # observations. A frozen full-map bank may contain the query observation
    # itself, so matching bank tracks are ambiguity exclusions, never positives.
    strict_positive_track_ids: torch.Tensor | None = None,
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
    known_positives = None
    if known_positive_track_ids is not None:
        known_positives = known_positive_track_ids.to(
            device=query_descriptors.device, dtype=torch.long
        )
        if known_positives.ndim != 2 or known_positives.shape[0] != ids.shape[0]:
            raise ValueError(
                "known_positive_track_ids must have shape (descriptor pair count, padded track count)"
            )
    strict_positives = None
    if strict_positive_track_ids is not None:
        strict_positives = strict_positive_track_ids.to(
            device=query_descriptors.device, dtype=torch.long
        )
        if strict_positives.ndim != 2 or strict_positives.shape[0] != ids.shape[0]:
            raise ValueError(
                "strict_positive_track_ids must have shape (descriptor pair count, padded track count)"
            )
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
    if known_positives is not None:
        known_positives = known_positives[valid]
    if strict_positives is not None:
        strict_positives = strict_positives[valid]
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
    if known_positives is not None:
        known_positives = known_positives[eligible_rows]
    if strict_positives is not None:
        strict_positives = strict_positives[eligible_rows]
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
        if known_positives is not None:
            known_positives = known_positives[keep]
        if strict_positives is not None:
            strict_positives = strict_positives[keep]

    history_found = torch.zeros((unique_ids.numel(),), dtype=torch.bool, device=query.device)
    history_counts = np.zeros((unique_ids.numel(),), dtype=np.int64)
    positive_observation_counts = current_counts.detach().cpu().numpy().astype(np.int64)
    positive_prototypes = current_prototypes
    if str(cfg.positive_prototype_source) == "query_disjoint_frozen_bank":
        if memory_bank is None or not bool(memory_bank.frozen):
            raise ValueError(
                "query_disjoint_frozen_bank positives require a frozen projected landmark bank"
            )
        bank_prototypes, bank_found, bank_counts, _bank_xyz = memory_bank.lookup(
            unique_ids.detach().cpu().numpy()
        )
        if not bool(torch.all(bank_found)):
            missing = unique_ids[~bank_found].detach().cpu().tolist()
            raise ValueError(
                "query-disjoint frozen bank is missing episode-positive tracks: "
                f"count={len(missing)}, preview={missing[:10]!r}"
            )
        positive_prototypes = bank_prototypes
        positive_observation_counts = bank_counts
    elif memory_bank is not None and len(memory_bank) > 0 and not bool(memory_bank.frozen):
        history, history_found, history_counts, _history_xyz = memory_bank.lookup(unique_ids.detach().cpu().numpy())
        mix = float(cfg.prototype_history_mix)
        weights = history_found.to(dtype=current_prototypes.dtype)[:, None] * mix
        positive_prototypes = F.normalize((1.0 - weights) * current_prototypes + weights * history, dim=1)
        positive_observation_counts = current_counts.detach().cpu().numpy().astype(np.int64) + history_counts

    memory_descriptors = torch.zeros((0, query.shape[1]), dtype=query.dtype, device=query.device)
    memory_xyz = torch.zeros((0, 3), dtype=torch.float32, device=query.device)
    selected_memory_indices: list[int] = []
    selected_memory_negative_count = 0
    semantic_selected: list[int] = []
    geometry_selected: list[int] = []
    random_selected: list[int] = []
    admitted_source_counts = {"semantic": 0, "geometry": 0, "random": 0}
    semantic_top_indices = torch.zeros((0, 0), dtype=torch.long, device=query.device)
    memory_candidate_slots = np.zeros((0,), dtype=np.int64)
    known_memory_query_rows = np.zeros((0,), dtype=np.int64)
    known_memory_candidate_rows = np.zeros((0,), dtype=np.int64)
    strict_memory_query_rows = np.zeros((0,), dtype=np.int64)
    strict_memory_candidate_rows = np.zeros((0,), dtype=np.int64)
    known_positive_raw_top1_count = 0
    if memory_bank is not None and len(memory_bank) > 0:
        memory_descriptors, _memory_track_ids, memory_xyz, memory_candidate_slots = memory_bank.candidates(
            exclude_track_ids=unique_ids.detach().cpu().numpy(),
            max_count=int(cfg.memory_candidate_pool_size),
            seed=int(seed),
        )
        known_memory_query_rows, known_memory_candidate_rows = (
            _known_positive_candidate_pairs(
                known_positives,
                memory_bank=memory_bank,
                candidate_slots=memory_candidate_slots,
            )
        )
        strict_memory_query_rows, strict_memory_candidate_rows = (
            _known_positive_candidate_pairs(
                strict_positives,
                memory_bank=memory_bank,
                candidate_slots=memory_candidate_slots,
            )
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
        ambiguous_query_rows = np.concatenate(
            [known_memory_query_rows, strict_memory_query_rows], axis=0
        )
        ambiguous_candidate_rows = np.concatenate(
            [known_memory_candidate_rows, strict_memory_candidate_rows], axis=0
        )
        if ambiguous_query_rows.size:
            raw_top1 = torch.argmax(detached_scores[: query.shape[0]], dim=1).detach().cpu().numpy()
            known_pairs = set(
                zip(ambiguous_query_rows.tolist(), ambiguous_candidate_rows.tolist())
            )
            known_positive_raw_top1_count = int(
                sum(
                    (int(row), int(candidate_row)) in known_pairs
                    for row, candidate_row in enumerate(raw_top1.tolist())
                )
            )
            if bool(cfg.exclude_known_cell_positives_from_memory):
                detached_scores[
                    torch.as_tensor(ambiguous_query_rows, dtype=torch.long, device=query.device),
                    torch.as_tensor(ambiguous_candidate_rows, dtype=torch.long, device=query.device),
                ] = float("-inf")
        semantic_k = min(int(cfg.semantic_hard_negatives_per_query), int(memory_descriptors.shape[0]))
        if semantic_k > 0:
            semantic_top_indices = torch.topk(detached_scores, k=semantic_k, dim=1).indices
            # Rank-major flattening admits every query's nearest confuser
            # before adding rank-2 candidates. Row-major flattening starved
            # later query observations when max_memory_negatives was reached.
            semantic_selected = (
                semantic_top_indices.T.reshape(-1).detach().cpu().tolist()
            )
        geometry_k = min(int(cfg.geometry_hard_negatives_per_track), int(memory_descriptors.shape[0]))
        finite_current = torch.isfinite(current_xyz).all(dim=1)
        finite_memory = torch.isfinite(memory_xyz).all(dim=1)
        if geometry_k > 0 and torch.any(finite_current) and torch.any(finite_memory):
            finite_memory_indices = torch.nonzero(finite_memory, as_tuple=False).reshape(-1)
            distances = torch.cdist(current_xyz[finite_current], memory_xyz[finite_memory])
            local_k = min(geometry_k, int(finite_memory_indices.numel()))
            geometry_local = torch.topk(distances, k=local_k, dim=1, largest=False).indices.reshape(-1)
            geometry_selected = (
                finite_memory_indices[geometry_local.reshape(-1, local_k).T.reshape(-1)]
                .detach()
                .cpu()
                .tolist()
            )
        random_k = min(int(cfg.random_negatives), int(memory_descriptors.shape[0]))
        if random_k > 0:
            rng = np.random.default_rng(int(seed) + 7919)
            random_selected = rng.choice(int(memory_descriptors.shape[0]), size=random_k, replace=False).tolist()
        selected_memory_indices, admitted_source_counts = _merge_memory_negative_sources(
            semantic=semantic_selected,
            geometry=geometry_selected,
            random=random_selected,
            limit=int(cfg.max_memory_negatives),
            policy=str(cfg.memory_negative_merge_policy),
        )
        selected_memory_negative_count = int(len(selected_memory_indices))

    selected_known_positive_mask = torch.zeros(
        (int(query.shape[0]), int(len(selected_memory_indices))),
        dtype=torch.bool,
        device=query.device,
    )
    selected_strict_ambiguity_mask = torch.zeros_like(selected_known_positive_mask)
    if selected_memory_indices and known_memory_query_rows.size:
        selected_position_by_candidate = np.full(
            (int(memory_descriptors.shape[0]),), -1, dtype=np.int64
        )
        selected_position_by_candidate[np.asarray(selected_memory_indices, dtype=np.int64)] = np.arange(
            len(selected_memory_indices), dtype=np.int64
        )
        selected_positions = selected_position_by_candidate[known_memory_candidate_rows]
        admitted_known = selected_positions >= 0
        if np.any(admitted_known):
            selected_known_positive_mask[
                torch.as_tensor(
                    known_memory_query_rows[admitted_known], dtype=torch.long, device=query.device
                ),
                torch.as_tensor(
                    selected_positions[admitted_known], dtype=torch.long, device=query.device
                ),
            ] = True
    if selected_memory_indices and strict_memory_query_rows.size:
        selected_position_by_candidate = np.full(
            (int(memory_descriptors.shape[0]),), -1, dtype=np.int64
        )
        selected_position_by_candidate[np.asarray(selected_memory_indices, dtype=np.int64)] = np.arange(
            len(selected_memory_indices), dtype=np.int64
        )
        selected_positions = selected_position_by_candidate[strict_memory_candidate_rows]
        admitted_strict = selected_positions >= 0
        if np.any(admitted_strict):
            selected_strict_ambiguity_mask[
                torch.as_tensor(
                    strict_memory_query_rows[admitted_strict], dtype=torch.long, device=query.device
                ),
                torch.as_tensor(
                    selected_positions[admitted_strict], dtype=torch.long, device=query.device
                ),
            ] = True

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
    own_track_mask = torch.zeros(
        (int(query.shape[0]), int(unique_ids.numel())),
        dtype=torch.bool,
        device=query.device,
    )
    own_track_mask[torch.arange(query.shape[0], device=query.device), labels] = True
    coarse_ambiguity_track_mask = own_track_mask.clone()
    if groups is not None:
        same_group_rows = groups[:, None] == groups[None, :]
        row_track_membership = F.one_hot(labels, num_classes=int(unique_ids.numel())).to(dtype=torch.float32)
        coarse_ambiguity_track_mask = (
            same_group_rows.to(dtype=torch.float32) @ row_track_membership
        ) > 0
    strict_track_mask = own_track_mask.clone()
    if strict_positives is not None and strict_positives.numel() > 0:
        valid_strict = strict_positives >= 0
        strict_track_mask |= torch.any(
            (strict_positives[:, :, None] == unique_ids[None, None, :])
            & valid_strict[:, :, None],
            dim=1,
        )
    denominator_keep = torch.ones_like(logits, dtype=torch.bool)
    if bool(cfg.exclude_known_cell_positives_from_memory):
        denominator_keep[:, : int(unique_ids.numel())] &= ~(
            coarse_ambiguity_track_mask & ~strict_track_mask
        )
    if (
        bool(cfg.exclude_known_cell_positives_from_memory)
        and selected_known_positive_mask.numel() > 0
    ):
        start = int(unique_ids.numel())
        denominator_keep[
            :, start : start + int(selected_known_positive_mask.shape[1])
        ] &= ~(selected_known_positive_mask | selected_strict_ambiguity_mask)
    if bool(cfg.set_valued_cell_positives):
        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        positive_mask[:, : int(unique_ids.numel())] = strict_track_mask
        positive_logsumexp = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
        denominator_logsumexp = torch.logsumexp(
            logits.masked_fill(~denominator_keep, float("-inf")), dim=1
        )
        retrieval_loss = (denominator_logsumexp - positive_logsumexp).mean()
    else:
        denominator_keep[:, : int(unique_ids.numel())] = ~coarse_ambiguity_track_mask
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
    system_hard_negative_margin_loss = logits.new_tensor(0.0)
    system_hard_negative_margin_count = 0
    if (
        float(cfg.system_hard_negative_margin_weight) > 0.0
        and semantic_top_indices.shape[0] >= query.shape[0]
        and semantic_top_indices.shape[1] > 0
    ):
        matched_semantic_indices = semantic_top_indices[: query.shape[0]]
        hard_descriptors = memory_descriptors[matched_semantic_indices].detach()
        hard_scores = torch.einsum("qc,qkc->qk", query, hard_descriptors)
        hardest_system_negative = hard_scores.max(dim=1).values
        current_positive_scores = query @ positive_prototypes.T
        best_positive_score = current_positive_scores.masked_fill(
            ~strict_track_mask,
            float("-inf"),
        ).max(dim=1).values
        margin_rows = torch.isfinite(best_positive_score) & torch.isfinite(
            hardest_system_negative
        )
        if torch.any(margin_rows):
            system_hard_negative_margin_loss = F.relu(
                hardest_system_negative[margin_rows]
                - best_positive_score[margin_rows]
                + float(cfg.system_hard_negative_margin)
            ).mean()
            system_hard_negative_margin_count = int(
                torch.count_nonzero(margin_rows).item()
            )
    loss = (
        retrieval_loss
        + float(cfg.dustbin_loss_weight) * dustbin_loss
        + float(cfg.system_hard_negative_margin_weight)
        * system_hard_negative_margin_loss
    )

    with torch.no_grad():
        positive_scores = cosine_scores[torch.arange(cosine_scores.shape[0], device=query.device), labels]
        negative_mask = torch.ones_like(cosine_scores, dtype=torch.bool)
        if bool(cfg.set_valued_cell_positives):
            if bool(cfg.exclude_known_cell_positives_from_memory):
                negative_mask[:, : int(unique_ids.numel())] &= ~coarse_ambiguity_track_mask
            else:
                negative_mask[:, : int(unique_ids.numel())] &= ~strict_track_mask
        else:
            negative_mask[torch.arange(cosine_scores.shape[0], device=query.device), labels] = False
        if (
            bool(cfg.exclude_known_cell_positives_from_memory)
            and selected_known_positive_mask.numel() > 0
        ):
            start = int(unique_ids.numel())
            negative_mask[
                :, start : start + int(selected_known_positive_mask.shape[1])
            ] &= ~(selected_known_positive_mask | selected_strict_ambiguity_mask)
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
            "landmark_retrieval_memory_negative_count": float(selected_memory_negative_count),
            "landmark_retrieval_memory_strict_ambiguity_candidate_count": float(
                torch.count_nonzero(selected_strict_ambiguity_mask.any(dim=0)).item()
                if selected_strict_ambiguity_mask.numel() > 0
                else 0
            ),
            "landmark_retrieval_memory_candidate_pool_count": float(memory_descriptors.shape[0]),
            "landmark_retrieval_global_exact_semantic_mining": float(
                bool(memory_bank is not None)
                and int(cfg.memory_candidate_pool_size) == 0
            ),
            "landmark_retrieval_semantic_hard_negative_count": float(
                admitted_source_counts["semantic"]
            ),
            "landmark_retrieval_geometry_hard_negative_count": float(
                admitted_source_counts["geometry"]
            ),
            "landmark_retrieval_random_negative_count": float(
                admitted_source_counts["random"]
            ),
            "landmark_retrieval_semantic_hard_negative_raw_unique_count": float(
                len(set(semantic_selected))
            ),
            "landmark_retrieval_geometry_hard_negative_raw_unique_count": float(
                len(set(geometry_selected))
            ),
            "landmark_retrieval_random_negative_raw_unique_count": float(
                len(set(random_selected))
            ),
            "landmark_retrieval_system_hard_negative_margin_loss": float(
                system_hard_negative_margin_loss.detach().cpu().item()
            ),
            "landmark_retrieval_system_hard_negative_margin_count": float(
                system_hard_negative_margin_count
            ),
            "landmark_retrieval_positive_score_mean": float(positive_scores.mean().item()),
            "landmark_retrieval_hardest_negative_score_mean": hardest_negative_mean,
            "landmark_retrieval_score_gap_mean": score_gap_mean,
            "landmark_retrieval_history_positive_fraction": float(history_found.float().mean().item()),
            "landmark_retrieval_mean_prototype_observation_count": float(
                np.mean(positive_observation_counts)
            ),
            "landmark_retrieval_positive_source_episode_support": float(
                str(cfg.positive_prototype_source) == "episode_support_observations"
            ),
            "landmark_retrieval_positive_source_query_disjoint_frozen_bank": float(
                str(cfg.positive_prototype_source) == "query_disjoint_frozen_bank"
            ),
            "landmark_retrieval_geometry_available_fraction": float(torch.isfinite(current_xyz).all(dim=1).float().mean().item()),
            "landmark_retrieval_same_cell_false_negatives_excluded_mean": float(
                (coarse_ambiguity_track_mask.sum(dim=1) - 1).clamp_min(0).float().mean().item()
            ),
            "landmark_retrieval_memory_size_before": float(0 if memory_bank is None else len(memory_bank)),
            "landmark_retrieval_set_valued_cell_positives": float(bool(cfg.set_valued_cell_positives)),
            "landmark_retrieval_excludes_known_cell_positives_from_memory": float(
                bool(cfg.exclude_known_cell_positives_from_memory)
            ),
            "landmark_retrieval_known_positive_memory_candidate_count": float(
                known_memory_query_rows.size
            ),
            "landmark_retrieval_known_positive_memory_candidates_per_query_mean": float(
                known_memory_query_rows.size / max(int(query.shape[0]), 1)
            ),
            "landmark_retrieval_known_positive_raw_top1_count": float(
                known_positive_raw_top1_count
            ),
            "landmark_retrieval_known_positive_raw_top1_fraction": float(
                known_positive_raw_top1_count / max(int(query.shape[0]), 1)
            ),
            "landmark_retrieval_known_positive_selected_negative_count": float(
                (selected_known_positive_mask | selected_strict_ambiguity_mask).sum().item()
                if bool(cfg.exclude_known_cell_positives_from_memory)
                else 0.0
            ),
            "landmark_retrieval_known_positive_selected_candidate_count": float(
                selected_known_positive_mask.sum().item()
            ),
            "landmark_retrieval_strict_ambiguity_memory_pair_count": float(
                strict_memory_query_rows.size
            ),
            "landmark_retrieval_strict_ambiguity_selected_pair_count": float(
                selected_strict_ambiguity_mask.sum().item()
            ),
            "landmark_retrieval_memory_negative_merge_policy_balanced": float(
                str(cfg.memory_negative_merge_policy)
                == "source_balanced_round_robin"
            ),
        }
        if semantic_top_indices.shape[0] > 0:
            selected_memory_set = set(selected_memory_indices)
            admitted = torch.as_tensor(
                [
                    int(index) in selected_memory_set
                    for index in semantic_top_indices.reshape(-1).detach().cpu().tolist()
                ],
                dtype=torch.float32,
            ).reshape(semantic_top_indices.shape)
            metrics["landmark_retrieval_semantic_top1_admitted_fraction"] = float(
                admitted[:, 0].mean().item()
            )
            metrics["landmark_retrieval_semantic_topk_admitted_per_query_mean"] = float(
                admitted.sum(dim=1).mean().item()
            )
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
        valid_group_candidates[:, : int(unique_ids.numel())] = coarse_ambiguity_track_mask
        if selected_known_positive_mask.numel() > 0:
            start = int(unique_ids.numel())
            valid_group_candidates[
                :, start : start + int(selected_known_positive_mask.shape[1])
            ] = selected_known_positive_mask
        strict_group_candidates = torch.zeros_like(logits, dtype=torch.bool)
        strict_group_candidates[:, : int(unique_ids.numel())] = strict_track_mask
        if selected_strict_ambiguity_mask.numel() > 0:
            start = int(unique_ids.numel())
            strict_group_candidates[
                :, start : start + int(selected_strict_ambiguity_mask.shape[1])
            ] = selected_strict_ambiguity_mask
        for k in (1, 5, 20):
            take = min(int(k), int(ranking.shape[1]))
            hit = torch.any(ranking[:, :take] == labels[:, None], dim=1)
            metrics[f"landmark_retrieval_recall_at_{k}"] = float(hit.float().mean().item())
            valid_hit = torch.gather(valid_group_candidates, 1, ranking[:, :take]).any(dim=1)
            metrics[f"landmark_retrieval_same_cell_valid_recall_at_{k}"] = float(valid_hit.float().mean().item())
            strict_hit = torch.gather(strict_group_candidates, 1, ranking[:, :take]).any(dim=1)
            metrics[f"landmark_retrieval_strict_valid_recall_at_{k}"] = float(
                strict_hit.float().mean().item()
            )

    if memory_bank is not None and bool(update_memory):
        memory_bank.update(
            track_ids=unique_ids.detach().cpu().numpy(),
            descriptors=current_prototypes,
            observation_counts=current_counts.detach().cpu().numpy().astype(np.int64),
            xyz=current_xyz,
        )
    metrics["landmark_retrieval_memory_size_after"] = float(0 if memory_bank is None else len(memory_bank))
    return loss, metrics
