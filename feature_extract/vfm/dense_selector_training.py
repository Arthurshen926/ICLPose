"""Dense-token selector training for fixed candidate banks."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.hypotheses import CandidateHypothesis
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.losses import basin_bce_loss, group_sparsity_loss, listwise_pose_rank_loss
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord
from feature_extract.vfm.track_utility_training import (
    TrackUtilityTrainingConfig,
    _build_track_supervision_groups,
    _loss_for_track_groups,
    _pairwise_track_separation,
    _utility_target_correlation,
)


@dataclass(frozen=True)
class DenseSelectorTrainingConfig:
    steps: int = 100
    batch_size: int = 8
    output_dim: int = 64
    group_size: int = 16
    seed: int = 0
    device: str = "cpu"
    lr: float = 1e-3
    eval_split_fraction: float = 0.2
    rank_temperature: float = 0.1
    sparsity_weight: float = 0.005
    layer_name: str = "radio_final"
    spatial_samples: int = 256
    sampling_seed: int | None = None
    init_checkpoint: str = ""
    diagnostic_query_limit: int = 0
    preload_sampled_features: bool = False
    sample_cache: str = ""
    write_sample_cache: str = ""
    sample_cache_dtype: str = "float16"
    basin_bce_weight: float = 0.0
    hard_negative_weight: float = 0.0
    basin_translation_threshold_m: float = 5.0
    basin_rotation_threshold_deg: float = 10.0
    hard_negative_margin: float = 0.1
    utility_weighted_pooling: bool = False
    init_anchor_weight: float = 0.0
    track_supervision_weight: float = 0.0
    track_batch_size: int = 64
    track_min_observations: int = 2
    track_temperature: float = 0.1
    track_consistency_weight: float = 1.0
    track_contrastive_weight: float = 1.0
    track_utility_weight: float = 0.5


@dataclass(frozen=True)
class DenseSelectorTrainingSummary:
    initial_loss: float
    final_loss: float
    raw_train_top1_acc: float
    raw_eval_top1_acc: float
    train_top1_acc: float
    eval_top1_acc: float
    query_count: int
    train_query_count: int
    eval_query_count: int
    train_diagnostic_query_count: int
    eval_diagnostic_query_count: int
    loaded_feature_count: int
    preloaded_feature_count: int
    written_feature_count: int
    train_cost_max_translation_m: float
    train_cost_max_rotation_deg: float
    track_supervision_track_count: int
    track_supervision_observation_count: int
    initial_track_positive_similarity: float
    initial_track_negative_similarity: float
    final_track_positive_similarity: float
    final_track_negative_similarity: float
    initial_track_utility_target_correlation: float
    final_track_utility_target_correlation: float


@dataclass(frozen=True)
class DenseSelectorTrainingRun:
    selector: LocalizableFeatureSelector
    summary: DenseSelectorTrainingSummary


@dataclass(frozen=True)
class _DenseQueryGroup:
    query_id: str
    query_path: str
    candidate_paths: Tuple[str, ...]
    translation_m: np.ndarray
    rotation_deg: np.ndarray
    costs: np.ndarray


_FeatureCacheKey = Tuple[str, str, int, int]
_FeatureCache = Mapping[_FeatureCacheKey, np.ndarray]
_SAMPLE_CACHE_FORMAT = "vfm_sampled_dense_feature_cache_v1"


def _read_dense_feature(path: str, layer_name: str) -> np.ndarray:
    with np.load(path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {path}")
        feature = np.asarray(data[layer_name], dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("dense token feature must have shape (C, H, W)")
    return feature


def _record_index(manifest: TokenBankManifest) -> dict[str, TokenBankRecord]:
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


def _build_dense_groups(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
) -> List[_DenseQueryGroup]:
    query_index = _record_index(query_manifest)
    map_index = _record_index(map_manifest)
    grouped: dict[str, list[CandidateHypothesis]] = {}
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_index:
            raise ValueError(f"query token not found: {candidate.query_id}")
        if candidate.reference_image not in map_index:
            raise ValueError(f"reference token not found: {candidate.reference_image}")
        grouped.setdefault(candidate.query_id, []).append(candidate)
    if not grouped:
        raise ValueError("candidate bank has no candidates")

    groups: list[_DenseQueryGroup] = []
    for query_id in sorted(grouped):
        candidates = grouped[query_id]
        if len(candidates) < 2:
            raise ValueError(f"query {query_id} needs at least 2 candidates")
        candidate_paths = []
        translations = []
        rotations = []
        for candidate in candidates:
            assert candidate.reference_image is not None
            assert candidate.pose_error is not None
            candidate_paths.append(str(map_index[candidate.reference_image].token_path))
            translations.append(float(candidate.pose_error.translation_m))
            rotations.append(float(candidate.pose_error.rotation_deg))
        groups.append(
            _DenseQueryGroup(
                query_id=query_id,
                query_path=str(query_index[query_id].token_path),
                candidate_paths=tuple(candidate_paths),
                translation_m=np.asarray(translations, dtype=np.float32),
                rotation_deg=np.asarray(rotations, dtype=np.float32),
                costs=np.zeros((len(candidate_paths),), dtype=np.float32),
            )
        )
    return groups


def _pose_cost_normalizer(groups: Sequence[_DenseQueryGroup]) -> tuple[float, float]:
    if not groups:
        raise ValueError("at least one dense group is required for cost normalization")
    translations = np.concatenate([group.translation_m for group in groups], axis=0)
    rotations = np.concatenate([group.rotation_deg for group in groups], axis=0)
    return max(float(np.max(translations)), 1e-6), max(float(np.max(rotations)), 1e-6)


def _normalize_dense_groups(
    groups: Sequence[_DenseQueryGroup],
    normalizer: tuple[float, float],
) -> List[_DenseQueryGroup]:
    max_translation_m, max_rotation_deg = normalizer
    if max_translation_m <= 0.0 or max_rotation_deg <= 0.0:
        raise ValueError("normalizers must be positive")
    normalized: list[_DenseQueryGroup] = []
    for group in groups:
        translation_cost = np.maximum(group.translation_m, 0.0) / max_translation_m
        rotation_cost = np.maximum(group.rotation_deg, 0.0) / max_rotation_deg
        costs = (0.5 * (translation_cost + rotation_cost)).astype(np.float32, copy=False)
        normalized.append(replace(group, costs=costs))
    return normalized


def _split_groups(
    groups: Sequence[_DenseQueryGroup],
    eval_split_fraction: float,
    seed: int,
) -> Tuple[List[_DenseQueryGroup], List[_DenseQueryGroup]]:
    if not 0.0 <= eval_split_fraction < 1.0:
        raise ValueError("eval_split_fraction must be in [0, 1)")
    order = list(groups)
    random.Random(seed).shuffle(order)
    eval_count = int(round(len(order) * eval_split_fraction))
    if eval_split_fraction > 0.0 and len(order) > 1:
        eval_count = max(1, eval_count)
    eval_count = min(eval_count, len(order) - 1)
    return order[eval_count:], order[:eval_count]


def _limit_diagnostic_groups(
    groups: Sequence[_DenseQueryGroup],
    limit: int,
    seed: int,
) -> List[_DenseQueryGroup]:
    if limit <= 0 or limit >= len(groups):
        return list(groups)
    order = list(groups)
    random.Random(seed).shuffle(order)
    return order[:limit]


def _sample_spatial(feature: np.ndarray, spatial_samples: int, seed: int) -> np.ndarray:
    channels, height, width = feature.shape
    flat = feature.reshape(channels, height * width)
    if spatial_samples <= 0 or spatial_samples >= flat.shape[1]:
        return flat.reshape(channels, height, width)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(flat.shape[1], size=spatial_samples, replace=False))
    return flat[:, indices].reshape(channels, spatial_samples, 1)


def _stable_spatial_seed(identifier: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _effective_sampling_seed(config: DenseSelectorTrainingConfig) -> int:
    return config.seed if config.sampling_seed is None else int(config.sampling_seed)


@lru_cache(maxsize=256)
def _load_sampled_dense_feature(
    path: str,
    layer_name: str,
    spatial_samples: int,
    seed: int,
) -> np.ndarray:
    feature = _sample_spatial(_read_dense_feature(path, layer_name), spatial_samples, seed)
    return np.asarray(feature, dtype=np.float32)


def _feature_cache_key(path: str, layer_name: str, spatial_samples: int, seed: int) -> _FeatureCacheKey:
    return (str(path), str(layer_name), int(spatial_samples), int(seed))


def _write_sampled_dense_feature_cache(
    cache: _FeatureCache,
    path: Path,
    storage_dtype: str = "float16",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(storage_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("sample cache storage dtype must be float16 or float32")
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for idx, (key, feature) in enumerate(sorted(cache.items(), key=lambda item: item[0])):
        array_name = f"feature_{idx:06d}"
        arrays[array_name] = np.asarray(feature, dtype=dtype)
        feature_path, layer_name, spatial_samples, seed = key
        entries.append(
            {
                "array": array_name,
                "path": feature_path,
                "layer_name": layer_name,
                "spatial_samples": int(spatial_samples),
                "seed": int(seed),
                "shape": list(arrays[array_name].shape),
                "dtype": str(arrays[array_name].dtype),
            }
        )
    metadata = {
        "format": _SAMPLE_CACHE_FORMAT,
        "entry_count": len(entries),
        "storage_dtype": str(dtype),
        "entries": entries,
    }
    np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)


def _load_sampled_dense_feature_cache(path: Path) -> Dict[_FeatureCacheKey, np.ndarray]:
    cache_path = Path(path)
    with np.load(cache_path) as data:
        if "metadata" not in data:
            raise ValueError(f"sample cache {cache_path} is missing metadata")
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("format") != _SAMPLE_CACHE_FORMAT:
            raise ValueError(f"unsupported sample cache format in {cache_path}")
        cache: Dict[_FeatureCacheKey, np.ndarray] = {}
        for entry in metadata.get("entries", []):
            array_name = str(entry["array"])
            if array_name not in data:
                raise ValueError(f"sample cache {cache_path} is missing array {array_name}")
            key = (
                str(entry["path"]),
                str(entry["layer_name"]),
                int(entry["spatial_samples"]),
                int(entry["seed"]),
            )
            cache[key] = np.asarray(data[array_name], dtype=np.float32)
    return cache


def _feature_tensor(
    path: str,
    layer_name: str,
    spatial_samples: int,
    seed: int,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> torch.Tensor:
    key = _feature_cache_key(path, layer_name, spatial_samples, seed)
    if feature_cache is not None and key in feature_cache:
        feature = feature_cache[key]
    else:
        feature = _load_sampled_dense_feature(path, layer_name, spatial_samples, seed)
    return torch.as_tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)


def _query_feature_key(group: _DenseQueryGroup, config: DenseSelectorTrainingConfig) -> _FeatureCacheKey:
    sampling_seed = _effective_sampling_seed(config)
    return _feature_cache_key(
        group.query_path,
        config.layer_name,
        config.spatial_samples,
        _stable_spatial_seed(group.query_id, sampling_seed),
    )


def _candidate_feature_key(path: str, config: DenseSelectorTrainingConfig) -> _FeatureCacheKey:
    sampling_seed = _effective_sampling_seed(config)
    return _feature_cache_key(
        path,
        config.layer_name,
        config.spatial_samples,
        _stable_spatial_seed(path, sampling_seed),
    )


def _preload_sampled_dense_features(
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    existing_cache: _FeatureCache | None = None,
) -> Dict[_FeatureCacheKey, np.ndarray]:
    cache: Dict[_FeatureCacheKey, np.ndarray] = dict(existing_cache or {})
    for group in groups:
        keys = [_query_feature_key(group, config)]
        keys.extend(_candidate_feature_key(path, config) for path in group.candidate_paths)
        for key in keys:
            if key in cache:
                continue
            path, layer_name, spatial_samples, seed = key
            cache[key] = _load_sampled_dense_feature(path, layer_name, spatial_samples, seed)
    return cache


def _selected_descriptor(
    selector: LocalizableFeatureSelector,
    tokens: torch.Tensor,
    use_utility_weighted_pooling: bool = False,
) -> torch.Tensor:
    output = selector(tokens)
    selected = output.selected
    if use_utility_weighted_pooling:
        weights = output.utility.clamp_min(1e-6)
        descriptor = (selected * weights).flatten(2).sum(dim=2)
        normalizer = weights.flatten(2).sum(dim=2).clamp_min(1e-6)
        descriptor = descriptor / normalizer
    else:
        descriptor = selected.flatten(2).mean(dim=2)
    return F.normalize(descriptor, dim=-1, eps=1e-6)


def _raw_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    descriptor = tokens.flatten(2).mean(dim=2)
    return F.normalize(descriptor, dim=-1, eps=1e-6)


def _score_group(
    selector: LocalizableFeatureSelector,
    group: _DenseQueryGroup,
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> torch.Tensor:
    query_tokens = _feature_tensor(
        group.query_path,
        config.layer_name,
        config.spatial_samples,
        seed=_stable_spatial_seed(group.query_id, _effective_sampling_seed(config)),
        device=device,
        feature_cache=feature_cache,
    )
    candidate_tokens = [
        _feature_tensor(
            path,
            config.layer_name,
            config.spatial_samples,
            seed=_stable_spatial_seed(path, _effective_sampling_seed(config)),
            device=device,
            feature_cache=feature_cache,
        )
        for path in group.candidate_paths
    ]
    query_descriptor = _selected_descriptor(
        selector,
        query_tokens,
        use_utility_weighted_pooling=config.utility_weighted_pooling,
    )
    candidate_descriptors = torch.cat(
        [
            _selected_descriptor(
                selector,
                tokens,
                use_utility_weighted_pooling=config.utility_weighted_pooling,
            )
            for tokens in candidate_tokens
        ],
        dim=0,
    )
    return torch.matmul(query_descriptor, candidate_descriptors.T)


def _score_groups_batched(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> List[torch.Tensor]:
    if not groups:
        return []
    results: list[torch.Tensor | None] = [None] * len(groups)
    buckets: dict[int, list[tuple[int, _DenseQueryGroup]]] = {}
    for idx, group in enumerate(groups):
        buckets.setdefault(len(group.candidate_paths), []).append((idx, group))

    for candidate_count, bucket in buckets.items():
        query_tokens = torch.cat(
            [
                _feature_tensor(
                    group.query_path,
                    config.layer_name,
                    config.spatial_samples,
                    seed=_stable_spatial_seed(group.query_id, _effective_sampling_seed(config)),
                    device=device,
                    feature_cache=feature_cache,
                )
                for _idx, group in bucket
            ],
            dim=0,
        )
        candidate_tokens = torch.cat(
            [
                _feature_tensor(
                    path,
                    config.layer_name,
                    config.spatial_samples,
                    seed=_stable_spatial_seed(path, _effective_sampling_seed(config)),
                    device=device,
                    feature_cache=feature_cache,
                )
                for _idx, group in bucket
                for path in group.candidate_paths
            ],
            dim=0,
        )
        query_descriptors = _selected_descriptor(
            selector,
            query_tokens,
            use_utility_weighted_pooling=config.utility_weighted_pooling,
        )
        candidate_descriptors = _selected_descriptor(
            selector,
            candidate_tokens,
            use_utility_weighted_pooling=config.utility_weighted_pooling,
        ).view(
            len(bucket),
            candidate_count,
            -1,
        )
        scores = torch.einsum("bd,bnd->bn", query_descriptors, candidate_descriptors)
        for row, (idx, _group) in enumerate(bucket):
            results[idx] = scores[row : row + 1]

    return [result for result in results if result is not None]


def _raw_scores(
    group: _DenseQueryGroup,
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> torch.Tensor:
    query_tokens = _feature_tensor(
        group.query_path,
        config.layer_name,
        config.spatial_samples,
        seed=_stable_spatial_seed(group.query_id, _effective_sampling_seed(config)),
        device=device,
        feature_cache=feature_cache,
    )
    candidate_tokens = [
        _feature_tensor(
            path,
            config.layer_name,
            config.spatial_samples,
            seed=_stable_spatial_seed(path, _effective_sampling_seed(config)),
            device=device,
            feature_cache=feature_cache,
        )
        for path in group.candidate_paths
    ]
    query_descriptor = _raw_descriptor(query_tokens)
    candidate_descriptors = torch.cat([_raw_descriptor(tokens) for tokens in candidate_tokens], dim=0)
    return torch.matmul(query_descriptor, candidate_descriptors.T)


def _raw_scores_batched(
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> List[torch.Tensor]:
    if not groups:
        return []
    results: list[torch.Tensor | None] = [None] * len(groups)
    buckets: dict[int, list[tuple[int, _DenseQueryGroup]]] = {}
    for idx, group in enumerate(groups):
        buckets.setdefault(len(group.candidate_paths), []).append((idx, group))

    for candidate_count, bucket in buckets.items():
        query_tokens = torch.cat(
            [
                _feature_tensor(
                    group.query_path,
                    config.layer_name,
                    config.spatial_samples,
                    seed=_stable_spatial_seed(group.query_id, _effective_sampling_seed(config)),
                    device=device,
                    feature_cache=feature_cache,
                )
                for _idx, group in bucket
            ],
            dim=0,
        )
        candidate_tokens = torch.cat(
            [
                _feature_tensor(
                    path,
                    config.layer_name,
                    config.spatial_samples,
                    seed=_stable_spatial_seed(path, _effective_sampling_seed(config)),
                    device=device,
                    feature_cache=feature_cache,
                )
                for _idx, group in bucket
                for path in group.candidate_paths
            ],
            dim=0,
        )
        query_descriptors = _raw_descriptor(query_tokens)
        candidate_descriptors = _raw_descriptor(candidate_tokens).view(len(bucket), candidate_count, -1)
        scores = torch.einsum("bd,bnd->bn", query_descriptors, candidate_descriptors)
        for row, (idx, _group) in enumerate(bucket):
            results[idx] = scores[row : row + 1]

    return [result for result in results if result is not None]


def _basin_labels_for_group(
    group: _DenseQueryGroup,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: torch.device,
) -> torch.Tensor:
    labels = np.logical_and(
        group.translation_m <= float(translation_threshold_m),
        group.rotation_deg <= float(rotation_threshold_deg),
    ).astype(np.float32, copy=False)
    return torch.as_tensor(labels, dtype=torch.float32, device=device).view(1, -1)


def _hard_negative_margin_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    losses = []
    for row_scores, row_labels in zip(scores, labels):
        positive = row_labels > 0.5
        negative = ~positive
        if not bool(torch.any(positive)) or not bool(torch.any(negative)):
            continue
        best_positive = torch.max(row_scores[positive])
        best_negative = torch.max(row_scores[negative])
        losses.append(F.relu(best_negative + float(margin) - best_positive))
    if not losses:
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    return torch.stack(losses).mean()


def _loss_for_groups(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> torch.Tensor:
    losses = []
    scores_by_group = _score_groups_batched(selector, groups, config, device, feature_cache=feature_cache)
    for group, scores in zip(groups, scores_by_group):
        costs = torch.as_tensor(group.costs, dtype=torch.float32, device=device).view(1, -1)
        group_loss = listwise_pose_rank_loss(scores, costs, temperature=config.rank_temperature)
        if config.basin_bce_weight > 0.0 or config.hard_negative_weight > 0.0:
            labels = _basin_labels_for_group(
                group,
                translation_threshold_m=config.basin_translation_threshold_m,
                rotation_threshold_deg=config.basin_rotation_threshold_deg,
                device=device,
            )
            if config.basin_bce_weight > 0.0:
                group_loss = group_loss + config.basin_bce_weight * basin_bce_loss(scores, labels)
            if config.hard_negative_weight > 0.0:
                group_loss = group_loss + config.hard_negative_weight * _hard_negative_margin_loss(
                    scores,
                    labels,
                    margin=config.hard_negative_margin,
                )
        losses.append(group_loss)
    data_loss = torch.stack(losses).mean()
    gates = torch.sigmoid(selector.group_logits)
    return data_loss + config.sparsity_weight * group_sparsity_loss(gates)


def _track_training_config(config: DenseSelectorTrainingConfig) -> TrackUtilityTrainingConfig:
    return TrackUtilityTrainingConfig(
        steps=1,
        batch_size=config.track_batch_size,
        output_dim=config.output_dim,
        group_size=config.group_size,
        seed=config.seed,
        device=config.device,
        lr=config.lr,
        min_observations=config.track_min_observations,
        temperature=config.track_temperature,
        consistency_weight=config.track_consistency_weight,
        contrastive_weight=config.track_contrastive_weight,
        utility_weight=config.track_utility_weight,
        sparsity_weight=0.0,
    )


def _selector_anchor_loss(
    selector: LocalizableFeatureSelector,
    anchor_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, value in selector.named_parameters():
        if name not in anchor_state:
            continue
        anchor = anchor_state[name].to(device=value.device, dtype=value.dtype)
        losses.append(torch.mean((value - anchor) ** 2))
    if not losses:
        return torch.zeros((), dtype=selector.group_logits.dtype, device=selector.group_logits.device)
    return torch.stack(losses).mean()


def _joint_loss_for_groups(
    selector: LocalizableFeatureSelector,
    dense_groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
    track_groups=None,
    rng: random.Random | None = None,
    anchor_state: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    loss = _loss_for_groups(selector, dense_groups, config, device, feature_cache=feature_cache)
    if track_groups:
        track_config = _track_training_config(config)
        track_batch_size = min(config.track_batch_size, len(track_groups))
        batch = (rng or random).sample(track_groups, track_batch_size)
        loss = loss + config.track_supervision_weight * _loss_for_track_groups(
            selector,
            batch,
            track_config,
            device,
        )
    if config.init_anchor_weight > 0.0:
        if anchor_state is None:
            raise ValueError("anchor_state is required when init_anchor_weight > 0")
        loss = loss + config.init_anchor_weight * _selector_anchor_loss(selector, anchor_state)
    return loss


def _top1_acc(
    selector: LocalizableFeatureSelector,
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> float:
    if not groups:
        return 0.0
    correct = 0
    with torch.no_grad():
        scores_by_group = _score_groups_batched(selector, groups, config, device, feature_cache=feature_cache)
        for group, scores in zip(groups, scores_by_group):
            if int(scores.argmax(dim=1).item()) == int(np.argmin(group.costs)):
                correct += 1
    return float(correct / len(groups))


def _raw_top1_acc(
    groups: Sequence[_DenseQueryGroup],
    config: DenseSelectorTrainingConfig,
    device: torch.device,
    feature_cache: _FeatureCache | None = None,
) -> float:
    if not groups:
        return 0.0
    correct = 0
    with torch.no_grad():
        scores_by_group = _raw_scores_batched(groups, config, device, feature_cache=feature_cache)
        for group, scores in zip(groups, scores_by_group):
            if int(scores.argmax(dim=1).item()) == int(np.argmin(group.costs)):
                correct += 1
    return float(correct / len(groups))


def _infer_input_dim(
    groups: Sequence[_DenseQueryGroup],
    layer_name: str,
    config: DenseSelectorTrainingConfig,
    feature_cache: _FeatureCache | None = None,
) -> int:
    if not groups:
        raise ValueError("at least one dense group is required")
    if feature_cache is not None:
        query_key = _query_feature_key(groups[0], config)
        if query_key in feature_cache:
            return int(feature_cache[query_key].shape[0])
        for feature in feature_cache.values():
            return int(feature.shape[0])
    return int(_read_dense_feature(groups[0].query_path, layer_name).shape[0])


def run_dense_selector_training(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    config: DenseSelectorTrainingConfig,
    track_observations: Sequence[TrackObservation] | None = None,
) -> DenseSelectorTrainingRun:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.diagnostic_query_limit < 0:
        raise ValueError("diagnostic_query_limit must be non-negative")
    if config.sampling_seed is not None and config.sampling_seed < 0:
        raise ValueError("sampling_seed must be non-negative")
    if config.basin_bce_weight < 0.0:
        raise ValueError("basin_bce_weight must be non-negative")
    if config.hard_negative_weight < 0.0:
        raise ValueError("hard_negative_weight must be non-negative")
    if config.basin_translation_threshold_m <= 0.0:
        raise ValueError("basin_translation_threshold_m must be positive")
    if config.basin_rotation_threshold_deg <= 0.0:
        raise ValueError("basin_rotation_threshold_deg must be positive")
    if config.hard_negative_margin < 0.0:
        raise ValueError("hard_negative_margin must be non-negative")
    if config.init_anchor_weight < 0.0:
        raise ValueError("init_anchor_weight must be non-negative")
    if config.track_supervision_weight < 0.0:
        raise ValueError("track_supervision_weight must be non-negative")
    if config.track_batch_size <= 0:
        raise ValueError("track_batch_size must be positive")
    if config.track_supervision_weight > 0.0 and not track_observations:
        raise ValueError("track_observations are required when track_supervision_weight > 0")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = random.Random(config.seed)
    device = torch.device(config.device)

    groups = _build_dense_groups(bank, query_manifest, map_manifest)
    train_groups, eval_groups = _split_groups(groups, config.eval_split_fraction, config.seed)
    cost_normalizer = _pose_cost_normalizer(train_groups)
    train_groups = _normalize_dense_groups(train_groups, cost_normalizer)
    eval_groups = _normalize_dense_groups(eval_groups, cost_normalizer)
    track_groups = (
        _build_track_supervision_groups(
            list(track_observations or ()),
            min_observations=config.track_min_observations,
        )
        if config.track_supervision_weight > 0.0
        else []
    )
    train_diagnostic_groups = _limit_diagnostic_groups(
        train_groups,
        config.diagnostic_query_limit,
        config.seed + 1_000_003,
    )
    eval_diagnostic_groups = _limit_diagnostic_groups(
        eval_groups,
        config.diagnostic_query_limit,
        config.seed + 2_000_003,
    )
    loaded_feature_count = 0
    written_feature_count = 0
    feature_cache: Dict[_FeatureCacheKey, np.ndarray] | None = None
    if config.sample_cache:
        feature_cache = _load_sampled_dense_feature_cache(Path(config.sample_cache))
        loaded_feature_count = len(feature_cache)
    if config.preload_sampled_features or config.write_sample_cache:
        feature_cache = _preload_sampled_dense_features(
            [*train_groups, *eval_diagnostic_groups],
            config,
            existing_cache=feature_cache,
        )
    if config.write_sample_cache:
        if feature_cache is None:
            feature_cache = {}
        _write_sampled_dense_feature_cache(
            feature_cache,
            Path(config.write_sample_cache),
            storage_dtype=config.sample_cache_dtype,
        )
        written_feature_count = len(feature_cache)
    preloaded_feature_count = len(feature_cache) if feature_cache is not None else 0
    input_dim = _infer_input_dim(groups, config.layer_name, config, feature_cache=feature_cache)
    if config.init_checkpoint:
        selector = load_selector_from_checkpoint(
            Path(config.init_checkpoint),
            input_dim=input_dim,
            device=config.device,
        )
    else:
        selector = LocalizableFeatureSelector(
            input_dim=input_dim,
            output_dim=config.output_dim,
            group_size=config.group_size,
        ).to(device)
    anchor_state = (
        {name: value.detach().clone().cpu() for name, value in selector.state_dict().items()}
        if config.init_anchor_weight > 0.0
        else None
    )
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.lr)

    with torch.no_grad():
        initial_loss = float(
            _joint_loss_for_groups(
                selector,
                train_diagnostic_groups,
                config,
                device,
                feature_cache=feature_cache,
                track_groups=track_groups,
                rng=rng,
                anchor_state=anchor_state,
            )
            .detach()
            .cpu()
        )
        if track_groups:
            initial_track_positive_similarity, initial_track_negative_similarity = _pairwise_track_separation(
                selector,
                track_groups,
                device,
            )
            initial_track_utility_target_correlation = _utility_target_correlation(selector, track_groups, device)
        else:
            initial_track_positive_similarity = 0.0
            initial_track_negative_similarity = 0.0
            initial_track_utility_target_correlation = 0.0

    for _ in range(config.steps):
        batch_size = min(config.batch_size, len(train_groups))
        batch = rng.sample(train_groups, batch_size)
        loss = _joint_loss_for_groups(
            selector,
            batch,
            config,
            device,
            feature_cache=feature_cache,
            track_groups=track_groups,
            rng=rng,
            anchor_state=anchor_state,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(
            _joint_loss_for_groups(
                selector,
                train_diagnostic_groups,
                config,
                device,
                feature_cache=feature_cache,
                track_groups=track_groups,
                rng=rng,
                anchor_state=anchor_state,
            )
            .detach()
            .cpu()
        )
        if track_groups:
            final_track_positive_similarity, final_track_negative_similarity = _pairwise_track_separation(
                selector,
                track_groups,
                device,
            )
            final_track_utility_target_correlation = _utility_target_correlation(selector, track_groups, device)
        else:
            final_track_positive_similarity = 0.0
            final_track_negative_similarity = 0.0
            final_track_utility_target_correlation = 0.0
        raw_train_top1_acc = _raw_top1_acc(train_diagnostic_groups, config, device, feature_cache=feature_cache)
        raw_eval_top1_acc = (
            _raw_top1_acc(eval_diagnostic_groups, config, device, feature_cache=feature_cache)
            if eval_diagnostic_groups
            else raw_train_top1_acc
        )
        train_top1_acc = _top1_acc(selector, train_diagnostic_groups, config, device, feature_cache=feature_cache)
        eval_top1_acc = (
            _top1_acc(selector, eval_diagnostic_groups, config, device, feature_cache=feature_cache)
            if eval_diagnostic_groups
            else train_top1_acc
        )

    summary = DenseSelectorTrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        raw_train_top1_acc=raw_train_top1_acc,
        raw_eval_top1_acc=raw_eval_top1_acc,
        train_top1_acc=train_top1_acc,
        eval_top1_acc=eval_top1_acc,
        query_count=len(groups),
        train_query_count=len(train_groups),
        eval_query_count=len(eval_groups),
        train_diagnostic_query_count=len(train_diagnostic_groups),
        eval_diagnostic_query_count=len(eval_diagnostic_groups),
        loaded_feature_count=loaded_feature_count,
        preloaded_feature_count=preloaded_feature_count,
        written_feature_count=written_feature_count,
        train_cost_max_translation_m=float(cost_normalizer[0]),
        train_cost_max_rotation_deg=float(cost_normalizer[1]),
        track_supervision_track_count=len(track_groups),
        track_supervision_observation_count=int(sum(group.features.shape[0] for group in track_groups)),
        initial_track_positive_similarity=initial_track_positive_similarity,
        initial_track_negative_similarity=initial_track_negative_similarity,
        final_track_positive_similarity=final_track_positive_similarity,
        final_track_negative_similarity=final_track_negative_similarity,
        initial_track_utility_target_correlation=initial_track_utility_target_correlation,
        final_track_utility_target_correlation=final_track_utility_target_correlation,
    )
    return DenseSelectorTrainingRun(selector=selector, summary=summary)


def build_dense_selector_sample_cache(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    config: DenseSelectorTrainingConfig,
    output: Path,
) -> dict[str, object]:
    if config.diagnostic_query_limit < 0:
        raise ValueError("diagnostic_query_limit must be non-negative")
    if config.sampling_seed is not None and config.sampling_seed < 0:
        raise ValueError("sampling_seed must be non-negative")
    groups = _build_dense_groups(bank, query_manifest, map_manifest)
    train_groups, eval_groups = _split_groups(groups, config.eval_split_fraction, config.seed)
    eval_diagnostic_groups = _limit_diagnostic_groups(
        eval_groups,
        config.diagnostic_query_limit,
        config.seed + 2_000_003,
    )
    existing_cache = _load_sampled_dense_feature_cache(Path(config.sample_cache)) if config.sample_cache else None
    cache = _preload_sampled_dense_features(
        [*train_groups, *eval_diagnostic_groups],
        config,
        existing_cache=existing_cache,
    )
    _write_sampled_dense_feature_cache(cache, Path(output), storage_dtype=config.sample_cache_dtype)
    return {
        "query_count": len(groups),
        "train_query_count": len(train_groups),
        "eval_query_count": len(eval_groups),
        "eval_diagnostic_query_count": len(eval_diagnostic_groups),
        "loaded_feature_count": len(existing_cache) if existing_cache is not None else 0,
        "sample_cache_entry_count": len(cache),
        "sample_cache": str(output),
    }


def train_dense_selector(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    config: DenseSelectorTrainingConfig,
) -> DenseSelectorTrainingSummary:
    return run_dense_selector_training(bank, query_manifest, map_manifest, config).summary
