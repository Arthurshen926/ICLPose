"""Lightweight descriptors derived from dense token-bank records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

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


def build_token_descriptor_bank(
    manifest: TokenBankManifest,
    layer_name: str,
    pooling: str = "mean",
    gem_power: float = 3.0,
    normalize_tokens: bool = False,
    normalize: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> TokenDescriptorBank:
    """Build a per-image descriptor bank by pooling dense token files once."""

    manifest.validate(verify_checksums=False)
    image_ids: list[str] = []
    descriptors: list[np.ndarray] = []
    for record in manifest.records:
        with np.load(record.token_path) as data:
            if layer_name not in data:
                raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
            feature = np.asarray(data[layer_name], dtype=np.float32)
        if normalize_tokens:
            feature = _normalize_tokens(feature)
        image_ids.append(record.image_id)
        descriptors.append(_pool_feature(feature, pooling, gem_power=gem_power))
    descriptor_array = np.stack(descriptors, axis=0).astype(np.float32, copy=False)
    if normalize:
        descriptor_array = _normalize_rows(descriptor_array)
    descriptor_metadata = dict(metadata or {})
    if pooling == "gem":
        descriptor_metadata["gem_power"] = float(gem_power)
    if normalize_tokens:
        descriptor_metadata["normalize_tokens"] = True
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
