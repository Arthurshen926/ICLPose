"""Lightweight descriptors derived from dense token-bank records."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.score_table import ScoreRow
from feature_extract.vfm.tokens import TokenBankManifest


@dataclass(frozen=True)
class TokenDescriptorBank:
    """Per-image descriptors for fast fixed-candidate scoring."""

    image_ids: Tuple[str, ...]
    descriptors: np.ndarray
    layer_name: str
    pooling: str
    normalized: bool = True
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_ids", tuple(self.image_ids))
        object.__setattr__(self, "descriptors", np.asarray(self.descriptors, dtype=np.float32))
        if self.descriptors.ndim != 2:
            raise ValueError("descriptors must have shape (N, C)")
        if len(self.image_ids) != self.descriptors.shape[0]:
            raise ValueError("image_ids length must match descriptor rows")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("duplicate image_id in descriptor bank")

    def index(self) -> dict[str, int]:
        return {image_id: idx for idx, image_id in enumerate(self.image_ids)}

    def get(self, image_id: str) -> np.ndarray:
        index = self.index()
        if image_id not in index:
            raise ValueError(f"descriptor not found: {image_id}")
        return self.descriptors[index[image_id]]

    def to_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "layer_name": self.layer_name,
            "pooling": self.pooling,
            "normalized": self.normalized,
            "metadata": dict(self.metadata or {}),
        }
        np.savez_compressed(
            path,
            image_ids=np.asarray(self.image_ids, dtype=object),
            descriptors=self.descriptors.astype(np.float32, copy=False),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def from_npz(cls, path: Path) -> "TokenDescriptorBank":
        with np.load(Path(path), allow_pickle=True) as data:
            image_ids = tuple(str(item) for item in data["image_ids"].tolist())
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            metadata = json.loads(str(data["metadata_json"].tolist()))
        return cls(
            image_ids=image_ids,
            descriptors=descriptors,
            layer_name=str(metadata["layer_name"]),
            pooling=str(metadata["pooling"]),
            normalized=bool(metadata["normalized"]),
            metadata=dict(metadata.get("metadata", {})),
        )


@dataclass(frozen=True)
class PCAWhiteningTransform:
    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    whitening_epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float32))
        object.__setattr__(self, "components", np.asarray(self.components, dtype=np.float32))
        object.__setattr__(self, "scales", np.asarray(self.scales, dtype=np.float32))
        if self.mean.ndim != 1:
            raise ValueError("PCA mean must have shape (C,)")
        if self.components.ndim != 2:
            raise ValueError("PCA components must have shape (D, C)")
        if self.scales.ndim != 1:
            raise ValueError("PCA scales must have shape (D,)")
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError("PCA components input dimension must match mean")
        if self.components.shape[0] != self.scales.shape[0]:
            raise ValueError("PCA scales length must match output dimension")


def _signed_gem_pool(feature: np.ndarray, gem_power: float) -> np.ndarray:
    if gem_power <= 0:
        raise ValueError("gem_power must be positive")
    flat = feature.reshape(feature.shape[0], -1)
    if np.isclose(gem_power, 3.0):
        signed_power = flat * flat
        signed_power *= flat
    else:
        signed_power = np.sign(flat) * np.power(np.abs(flat), gem_power)
    mean_power = signed_power.mean(axis=1)
    return np.sign(mean_power) * np.power(np.abs(mean_power), 1.0 / gem_power)


def _pool_feature(feature: np.ndarray, pooling: str, gem_power: float = 3.0) -> np.ndarray:
    if feature.ndim < 2:
        raise ValueError("token feature must have at least channel and spatial dimensions")
    if pooling == "mean":
        return feature.reshape(feature.shape[0], -1).mean(axis=1)
    if pooling == "gem":
        return _signed_gem_pool(feature, gem_power=gem_power)
    raise ValueError(f"unsupported descriptor pooling: {pooling}")


def _normalize_tokens(feature: np.ndarray) -> np.ndarray:
    flat = feature.reshape(feature.shape[0], -1)
    norms = np.linalg.norm(flat, axis=0, keepdims=True)
    normalized = flat / np.maximum(norms, 1e-6)
    return normalized.reshape(feature.shape)


def _normalize_rows(descriptors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-6)


def _power_normalize_rows(descriptors: np.ndarray, power: float) -> np.ndarray:
    if float(power) <= 0.0:
        raise ValueError("descriptor_power must be positive")
    if np.isclose(float(power), 1.0):
        return descriptors.astype(np.float32, copy=False)
    return (np.sign(descriptors) * np.power(np.abs(descriptors), float(power))).astype(np.float32, copy=False)


def _feature_tokens(feature: np.ndarray) -> np.ndarray:
    return np.asarray(feature, dtype=np.float32).reshape(feature.shape[0], -1).T


def _fit_vlad_codebook(
    manifest: TokenBankManifest,
    layer_name: str,
    *,
    clusters: int,
    iterations: int,
    max_tokens: int,
    max_tokens_per_image: int,
    max_images: int,
    normalize_tokens: bool,
    seed: int,
) -> np.ndarray:
    if int(clusters) <= 0:
        raise ValueError("vlad_clusters must be positive")
    if int(iterations) <= 0:
        raise ValueError("vlad_iterations must be positive")
    if int(max_tokens) <= 0:
        raise ValueError("vlad_max_tokens must be positive")
    if int(max_tokens_per_image) < 0:
        raise ValueError("vlad_codebook_tokens_per_image must be non-negative")
    if int(max_images) < 0:
        raise ValueError("vlad_codebook_max_images must be non-negative")
    rng = np.random.default_rng(int(seed))
    records = tuple(manifest.records)
    if int(max_images) > 0 and len(records) > int(max_images):
        choice = rng.choice(len(records), size=int(max_images), replace=False)
        records = tuple(records[int(index)] for index in np.sort(choice))
    samples: list[np.ndarray] = []
    sampled_count = 0
    for record in records:
        with np.load(record.token_path) as data:
            if layer_name not in data:
                raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
            feature = np.asarray(data[layer_name], dtype=np.float32)
        if normalize_tokens:
            feature = _normalize_tokens(feature)
        tokens_for_record = _feature_tokens(feature)
        per_image_limit = int(max_tokens_per_image)
        if per_image_limit > 0 and tokens_for_record.shape[0] > per_image_limit:
            choice = rng.choice(tokens_for_record.shape[0], size=per_image_limit, replace=False)
            tokens_for_record = tokens_for_record[choice]
        if tokens_for_record.shape[0] > int(max_tokens):
            choice = rng.choice(tokens_for_record.shape[0], size=int(max_tokens), replace=False)
            tokens_for_record = tokens_for_record[choice]
        samples.append(tokens_for_record.astype(np.float32, copy=False))
        sampled_count += int(tokens_for_record.shape[0])
        if sampled_count > int(max_tokens) * 2:
            merged = np.concatenate(samples, axis=0)
            choice = rng.choice(merged.shape[0], size=int(max_tokens), replace=False)
            samples = [merged[choice].astype(np.float32, copy=False)]
            sampled_count = int(max_tokens)
    tokens = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    if tokens.shape[0] < int(clusters):
        raise ValueError("vlad_clusters cannot exceed the number of available tokens")
    if tokens.shape[0] > int(max_tokens):
        choice = rng.choice(tokens.shape[0], size=int(max_tokens), replace=False)
        tokens = tokens[choice]

    centroids = np.empty((int(clusters), tokens.shape[1]), dtype=np.float32)
    first = int(np.argmax(np.linalg.norm(tokens, axis=1)))
    centroids[0] = tokens[first]
    min_dist = np.sum((tokens - centroids[0][None, :]) ** 2, axis=1)
    for cluster_idx in range(1, int(clusters)):
        next_idx = int(np.argmax(min_dist))
        centroids[cluster_idx] = tokens[next_idx]
        dist = np.sum((tokens - centroids[cluster_idx][None, :]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)

    for _ in range(int(iterations)):
        dists = np.sum((tokens[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignment = np.argmin(dists, axis=1)
        updated = centroids.copy()
        for cluster_idx in range(int(clusters)):
            mask = assignment == cluster_idx
            if np.any(mask):
                updated[cluster_idx] = tokens[mask].mean(axis=0)
        if np.allclose(updated, centroids):
            break
        centroids = updated.astype(np.float32, copy=False)
    return centroids.astype(np.float32, copy=False)


def _vlad_pool_feature(
    feature: np.ndarray,
    codebook: np.ndarray,
    *,
    normalize_tokens: bool,
    max_tokens_per_image: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if normalize_tokens:
        feature = _normalize_tokens(feature)
    tokens = _feature_tokens(feature)
    max_tokens = int(max_tokens_per_image)
    if max_tokens > 0 and tokens.shape[0] > max_tokens:
        generator = rng if rng is not None else np.random.default_rng(0)
        choice = generator.choice(tokens.shape[0], size=max_tokens, replace=False)
        tokens = tokens[choice]
    centroids = np.asarray(codebook, dtype=np.float32)
    dists = np.sum((tokens[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    assignment = np.argmin(dists, axis=1)
    residuals = np.zeros((centroids.shape[0], centroids.shape[1]), dtype=np.float32)
    for cluster_idx in range(centroids.shape[0]):
        mask = assignment == cluster_idx
        if np.any(mask):
            residuals[cluster_idx] = np.sum(tokens[mask] - centroids[cluster_idx][None, :], axis=0)
    residuals = _normalize_rows(residuals)
    descriptor = residuals.reshape(-1)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-6:
        return descriptor.astype(np.float32, copy=False)
    return (descriptor / norm).astype(np.float32, copy=False)


def fit_vlad_codebook_from_manifest(
    manifest: TokenBankManifest,
    layer_name: str,
    *,
    clusters: int = 32,
    iterations: int = 20,
    max_tokens: int = 200000,
    max_tokens_per_image: int = 0,
    max_images: int = 0,
    normalize_tokens: bool = False,
    seed: int = 0,
) -> np.ndarray:
    manifest.validate(verify_checksums=False)
    return _fit_vlad_codebook(
        manifest,
        layer_name,
        clusters=int(clusters),
        iterations=int(iterations),
        max_tokens=int(max_tokens),
        max_tokens_per_image=int(max_tokens_per_image),
        max_images=int(max_images),
        normalize_tokens=bool(normalize_tokens),
        seed=int(seed),
    )


def fit_vlad_codebook_from_manifests(
    manifests: Sequence[TokenBankManifest],
    layer_name: str,
    *,
    clusters: int = 32,
    iterations: int = 20,
    max_tokens: int = 200000,
    max_tokens_per_image: int = 0,
    max_images: int = 0,
    normalize_tokens: bool = False,
    seed: int = 0,
) -> np.ndarray:
    records = []
    for manifest_idx, manifest in enumerate(tuple(manifests)):
        manifest.validate(verify_checksums=False)
        for record_idx, record in enumerate(manifest.records):
            records.append(replace(record, image_id=f"manifest{manifest_idx:03d}:{record_idx:06d}:{record.image_id}"))
    if not records:
        raise ValueError("at least one token manifest is required")
    return _fit_vlad_codebook(
        TokenBankManifest(records=tuple(records)),
        layer_name,
        clusters=int(clusters),
        iterations=int(iterations),
        max_tokens=int(max_tokens),
        max_tokens_per_image=int(max_tokens_per_image),
        max_images=int(max_images),
        normalize_tokens=bool(normalize_tokens),
        seed=int(seed),
    )


def save_vlad_codebook_npz(
    path: Path,
    centroids: np.ndarray,
    *,
    layer_name: str,
    normalize_tokens: bool,
    metadata: Mapping[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_name": str(layer_name),
        "normalize_tokens": bool(normalize_tokens),
        "metadata": dict(metadata or {}),
    }
    np.savez_compressed(
        output,
        centroids=np.asarray(centroids, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )


def load_vlad_codebook_npz(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(Path(path), allow_pickle=True) as data:
        centroids = np.asarray(data["centroids"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].tolist()))
    if centroids.ndim != 2:
        raise ValueError("VLAD codebook centroids must have shape (K, C)")
    return centroids, dict(metadata)


def fit_pca_whitening_transform(
    descriptors: np.ndarray,
    output_dim: int,
    whitening_epsilon: float = 1e-6,
) -> PCAWhiteningTransform:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    if int(output_dim) <= 0 or int(output_dim) > values.shape[1]:
        raise ValueError("output_dim must be in [1, C]")
    if values.shape[0] < 2:
        raise ValueError("at least two descriptors are required to fit PCA whitening")
    mean = values.mean(axis=0)
    centered = values - mean[None, :]
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: int(output_dim)].astype(np.float32, copy=False)
    variances = (singular_values[: int(output_dim)] ** 2) / max(values.shape[0] - 1, 1)
    scales = (1.0 / np.sqrt(variances + float(whitening_epsilon))).astype(np.float32, copy=False)
    return PCAWhiteningTransform(
        mean=mean.astype(np.float32, copy=False),
        components=components,
        scales=scales,
        whitening_epsilon=float(whitening_epsilon),
    )


def apply_pca_whitening_transform(
    descriptors: np.ndarray,
    transform: PCAWhiteningTransform,
    *,
    descriptor_power: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    projected = (values - transform.mean[None, :]) @ transform.components.T
    projected = projected * transform.scales[None, :]
    projected = _power_normalize_rows(projected.astype(np.float32, copy=False), descriptor_power)
    if normalize:
        projected = _normalize_rows(projected)
    return projected.astype(np.float32, copy=False)


def save_pca_whitening_transform_npz(path: Path, transform: PCAWhiteningTransform) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        mean=transform.mean.astype(np.float32, copy=False),
        components=transform.components.astype(np.float32, copy=False),
        scales=transform.scales.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps({"whitening_epsilon": float(transform.whitening_epsilon)}, sort_keys=True)),
    )


def load_pca_whitening_transform_npz(path: Path) -> PCAWhiteningTransform:
    with np.load(Path(path), allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].tolist()))
        return PCAWhiteningTransform(
            mean=np.asarray(data["mean"], dtype=np.float32),
            components=np.asarray(data["components"], dtype=np.float32),
            scales=np.asarray(data["scales"], dtype=np.float32),
            whitening_epsilon=float(metadata["whitening_epsilon"]),
        )


def combine_token_descriptor_banks(
    banks: Sequence[TokenDescriptorBank],
    *,
    mode: str = "concat",
    normalize: bool = True,
) -> TokenDescriptorBank:
    values = tuple(banks)
    if not values:
        raise ValueError("at least one descriptor bank is required")
    if mode not in {"concat", "mean"}:
        raise ValueError("mode must be one of: concat, mean")
    image_ids = values[0].image_ids
    aligned = []
    for bank in values:
        index = bank.index()
        if set(index) != set(image_ids):
            raise ValueError("all descriptor banks must contain the same image ids")
        aligned.append(np.stack([bank.descriptors[index[image_id]] for image_id in image_ids], axis=0))
    if mode == "concat":
        descriptors = np.concatenate(aligned, axis=1).astype(np.float32, copy=False)
    else:
        dims = {array.shape[1] for array in aligned}
        if len(dims) != 1:
            raise ValueError("mean fusion requires matching descriptor dimensions")
        descriptors = np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32, copy=False)
    if normalize:
        descriptors = _normalize_rows(descriptors)
    return TokenDescriptorBank(
        image_ids=image_ids,
        descriptors=descriptors,
        layer_name="+".join(bank.layer_name for bank in values),
        pooling=f"{mode}:" + "+".join(bank.pooling for bank in values),
        normalized=bool(normalize),
        metadata={
            "source_bank_count": len(values),
            "source_pooling": [bank.pooling for bank in values],
        },
    )


def build_token_descriptor_bank(
    manifest: TokenBankManifest,
    layer_name: str,
    pooling: str = "mean",
    gem_power: float = 3.0,
    normalize_tokens: bool = False,
    normalize: bool = True,
    metadata: Mapping[str, object] | None = None,
    vlad_clusters: int = 32,
    vlad_iterations: int = 20,
    vlad_max_tokens: int = 200000,
    seed: int = 0,
    vlad_codebook: np.ndarray | None = None,
    vlad_tokens_per_image: int = 0,
    descriptor_power: float = 1.0,
) -> TokenDescriptorBank:
    """Build a per-image descriptor bank by pooling dense token files once."""

    manifest.validate(verify_checksums=False)
    codebook = None
    if pooling == "vlad":
        codebook = (
            np.asarray(vlad_codebook, dtype=np.float32)
            if vlad_codebook is not None
            else fit_vlad_codebook_from_manifest(
                manifest,
                layer_name,
                clusters=int(vlad_clusters),
                iterations=int(vlad_iterations),
                max_tokens=int(vlad_max_tokens),
                max_tokens_per_image=0,
                max_images=0,
                normalize_tokens=bool(normalize_tokens),
                seed=int(seed),
            )
        )
        if codebook.ndim != 2:
            raise ValueError("vlad_codebook must have shape (K, C)")
    image_ids: list[str] = []
    descriptors: list[np.ndarray] = []
    if int(vlad_tokens_per_image) < 0:
        raise ValueError("vlad_tokens_per_image must be non-negative")
    for record_idx, record in enumerate(manifest.records):
        with np.load(record.token_path) as data:
            if layer_name not in data:
                raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
            feature = np.asarray(data[layer_name], dtype=np.float32)
        if pooling == "vlad":
            assert codebook is not None
            descriptor = _vlad_pool_feature(
                feature,
                codebook,
                normalize_tokens=bool(normalize_tokens),
                max_tokens_per_image=int(vlad_tokens_per_image),
                rng=np.random.default_rng(int(seed) + int(record_idx)),
            )
        else:
            if normalize_tokens:
                feature = _normalize_tokens(feature)
            descriptor = _pool_feature(feature, pooling, gem_power=gem_power)
        image_ids.append(record.image_id)
        descriptors.append(descriptor)
    descriptor_array = np.stack(descriptors, axis=0).astype(np.float32, copy=False)
    descriptor_array = _power_normalize_rows(descriptor_array, float(descriptor_power))
    if normalize:
        descriptor_array = _normalize_rows(descriptor_array)
    descriptor_metadata = dict(metadata or {})
    if pooling == "gem":
        descriptor_metadata["gem_power"] = float(gem_power)
    if pooling == "vlad":
        assert codebook is not None
        descriptor_metadata["vlad_clusters"] = int(codebook.shape[0])
        descriptor_metadata["vlad_iterations"] = int(vlad_iterations)
        descriptor_metadata["vlad_max_tokens"] = int(vlad_max_tokens)
        descriptor_metadata["vlad_seed"] = int(seed)
        if int(vlad_tokens_per_image) > 0:
            descriptor_metadata["vlad_tokens_per_image"] = int(vlad_tokens_per_image)
    if normalize_tokens:
        descriptor_metadata["normalize_tokens"] = True
    if not np.isclose(float(descriptor_power), 1.0):
        descriptor_metadata["descriptor_power"] = float(descriptor_power)
    return TokenDescriptorBank(
        image_ids=tuple(image_ids),
        descriptors=descriptor_array,
        layer_name=layer_name,
        pooling=pooling,
        normalized=normalize,
        metadata=descriptor_metadata,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_candidate_bank_by_descriptor_cosine(
    bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> list[ScoreRow]:
    if query_descriptors.layer_name != map_descriptors.layer_name:
        raise ValueError("query and map descriptor banks must use the same layer")
    query_index = query_descriptors.index()
    map_index = map_descriptors.index()
    rows: list[ScoreRow] = []
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
        query_feature = query_descriptors.descriptors[query_index[candidate.query_id]]
        map_feature = map_descriptors.descriptors[map_index[candidate.reference_image]]
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=_cosine(query_feature, map_feature),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
            )
        )
    return rows
