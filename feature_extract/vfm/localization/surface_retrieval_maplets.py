"""Compact anchor-free RADIO-final maplets used only for region retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class SurfaceRetrievalMapletBank:
    """A bounded RADIO-final mixture and metric region per maplet.

    Maplets are regional retrieval units, not stable point identities. The
    mixture has no component/view identity. The artifact deliberately has no
    per-view descriptors, image identifiers,
    anchor identifiers, observations, or point correspondences.
    """

    maplet_ids: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    extents: np.ndarray
    descriptor_offsets: np.ndarray
    descriptors: np.ndarray
    descriptor_weights: np.ndarray
    quality_scores: np.ndarray
    descriptor_uncertainties: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.maplet_ids, dtype=np.int64).reshape(-1)
        count = int(ids.size)
        if len(np.unique(ids)) != count:
            raise ValueError("maplet_ids must be unique")
        object.__setattr__(self, "maplet_ids", ids)
        for name in ("centers", "normals", "extents"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (count, 3):
                raise ValueError(f"{name} must have shape (N, 3)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "normals", _normalize_rows(self.normals))
        offsets = np.asarray(self.descriptor_offsets, dtype=np.int64).reshape(-1)
        descriptors = _normalize_rows(self.descriptors)
        weights = np.asarray(self.descriptor_weights, dtype=np.float32).reshape(-1)
        if (
            offsets.shape != (count + 1,)
            or offsets[0] != 0
            or offsets[-1] != descriptors.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("descriptor_offsets must define at least one component per maplet")
        if weights.shape != (descriptors.shape[0],) or np.any(weights < 0.0):
            raise ValueError("descriptor_weights must be non-negative per component")
        normalized_weights = weights.copy()
        for row in range(count):
            begin, end = int(offsets[row]), int(offsets[row + 1])
            total = float(np.sum(normalized_weights[begin:end]))
            if total <= 0.0:
                raise ValueError("each maplet mixture must have positive total weight")
            normalized_weights[begin:end] /= total
        object.__setattr__(self, "descriptor_offsets", offsets)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "descriptor_weights", normalized_weights)
        for name in ("quality_scores", "descriptor_uncertainties"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata or {})
        for key in (
            "stores_mapping_rgb",
            "stores_mapping_image_paths",
            "stores_mapping_image_ids",
            "uses_alike_descriptors",
            "uses_radio_intermediate",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_stable_anchor_identity",
            "uses_point_correspondences",
        ):
            if bool(metadata.get(key, False)):
                raise ValueError(f"retrieval maplet bank violates contract: {key}")
        if metadata.get("vfm_layer", "radio_final") != "radio_final":
            raise ValueError("retrieval maplets must use RADIO-final features")
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        return int(self.maplet_ids.size)

    def save_npz(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(path),
            maplet_ids=self.maplet_ids,
            centers=self.centers,
            normals=self.normals,
            extents=self.extents,
            descriptor_offsets=self.descriptor_offsets,
            descriptors=self.descriptors,
            descriptor_weights=self.descriptor_weights,
            quality_scores=self.quality_scores,
            descriptor_uncertainties=self.descriptor_uncertainties,
            metadata_json=np.asarray(json.dumps(dict(self.metadata or {}), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "SurfaceRetrievalMapletBank":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {
                "maplet_ids",
                "centers",
                "normals",
                "extents",
                "descriptor_offsets",
                "descriptors",
                "descriptor_weights",
                "quality_scores",
                "descriptor_uncertainties",
                "metadata_json",
            }
            extra = set(data.files) - expected
            missing = expected - set(data.files)
            if extra or missing:
                raise ValueError(
                    f"non-canonical retrieval maplet artifact; extra={sorted(extra)}, "
                    f"missing={sorted(missing)}"
                )
            return cls(
                maplet_ids=data["maplet_ids"],
                centers=data["centers"],
                normals=data["normals"],
                extents=data["extents"],
                descriptor_offsets=data["descriptor_offsets"],
                descriptors=data["descriptors"],
                descriptor_weights=data["descriptor_weights"],
                quality_scores=data["quality_scores"],
                descriptor_uncertainties=data["descriptor_uncertainties"],
                metadata=json.loads(str(data["metadata_json"].item())),
            )


def retrieve_surface_maplets(
    query_descriptors: np.ndarray,
    bank: SurfaceRetrievalMapletBank,
    *,
    maximum_maplets: int = 12,
    candidates_per_region: int = 8,
    temperature: float = 0.08,
    null_logit: float = 0.0,
    mixture_temperature: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Accumulate soft regional RADIO evidence without point matching."""

    query = _normalize_rows(query_descriptors)
    if query.shape[1] != bank.descriptors.shape[1]:
        raise ValueError("query and maplet feature dimensions differ")
    if len(bank) == 0 or query.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {
            "retrieved_maplet_count": 0,
            "mean_null_probability": 1.0,
        }
    logits = np.empty((query.shape[0], len(bank)), dtype=np.float32)
    for row in range(len(bank)):
        begin = int(bank.descriptor_offsets[row])
        end = int(bank.descriptor_offsets[row + 1])
        component_scores = query @ bank.descriptors[begin:end].T
        maximum = np.max(component_scores, axis=1)
        weights = bank.descriptor_weights[begin:end]
        if float(mixture_temperature) <= 0.0:
            logits[:, row] = maximum
        else:
            logits[:, row] = maximum + float(mixture_temperature) * np.log(
                np.sum(
                    weights[None]
                    * np.exp(
                        (component_scores - maximum[:, None])
                        / float(mixture_temperature)
                    ),
                    axis=1,
                )
                + 1e-12
            )
    logits += 0.15 * np.log(np.clip(bank.quality_scores[None], 1e-4, 1.0))
    logits -= 0.10 * np.clip(bank.descriptor_uncertainties[None], 0.0, 10.0)
    keep = min(int(candidates_per_region), len(bank))
    columns = np.argpartition(-logits, kth=keep - 1, axis=1)[:, :keep]
    candidate_logits = np.take_along_axis(logits, columns, axis=1) / max(
        float(temperature), 1e-4
    )
    row_max = np.maximum(
        np.max(candidate_logits, axis=1, keepdims=True), float(null_logit)
    )
    numerator = np.exp(candidate_logits - row_max)
    null_numerator = np.exp(float(null_logit) - row_max[:, 0])
    denominator = np.sum(numerator, axis=1) + null_numerator
    probabilities = numerator / denominator[:, None]
    null_probabilities = null_numerator / denominator
    evidence = np.zeros((len(bank),), dtype=np.float64)
    np.add.at(evidence, columns.reshape(-1), probabilities.reshape(-1))
    ranked_rows = np.argsort(-evidence, kind="mergesort")
    ranked_rows = ranked_rows[evidence[ranked_rows] > 0.0][: int(maximum_maplets)]
    return bank.maplet_ids[ranked_rows], {
        "retrieved_maplet_count": int(ranked_rows.size),
        "mean_null_probability": float(np.mean(null_probabilities)),
        "regional_evidence_sum": float(np.sum(evidence[ranked_rows])),
        "representation": "compact_radio_final_mixture_per_maplet",
    }
