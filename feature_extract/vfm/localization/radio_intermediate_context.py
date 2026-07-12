"""Compact observation-referenced RADIO intermediate context features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.query_to_3d_matching import normalize_rows


@dataclass(frozen=True)
class RadioIntermediateContextCache:
    support_descriptors: np.ndarray
    query_anchor_descriptors: np.ndarray
    query_context_descriptors: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        support = np.asarray(self.support_descriptors, dtype=np.float32)
        anchors = np.asarray(self.query_anchor_descriptors, dtype=np.float32)
        context = np.asarray(self.query_context_descriptors, dtype=np.float32)
        mean = np.asarray(self.pca_mean, dtype=np.float32).reshape(-1)
        components = np.asarray(self.pca_components, dtype=np.float32)
        if components.ndim != 2 or components.shape[1] != len(mean):
            raise ValueError("PCA components and mean have incompatible dimensions")
        output_dim = int(components.shape[0])
        for name, values in (
            ("support", support),
            ("query anchor", anchors),
            ("query context", context),
        ):
            if values.ndim != 2 or values.shape[1] != output_dim:
                raise ValueError(f"{name} RADIO descriptors have the wrong shape")
            normalized, valid = normalize_rows(values)
            if not np.all(valid) or np.max(np.abs(normalized - values)) > 5e-3:
                raise ValueError(f"{name} RADIO descriptors must be finite and L2 normalized")
        object.__setattr__(self, "support_descriptors", support)
        object.__setattr__(self, "query_anchor_descriptors", anchors)
        object.__setattr__(self, "query_context_descriptors", context)
        object.__setattr__(self, "pca_mean", mean)
        object.__setattr__(self, "pca_components", components)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def descriptor_dim(self) -> int:
        return int(self.pca_components.shape[0])


def fit_normalized_pca(
    samples: np.ndarray,
    *,
    output_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit deterministic PCA after row normalization."""

    from sklearn.decomposition import PCA

    values, valid = normalize_rows(np.asarray(samples, dtype=np.float32))
    if not np.all(valid):
        raise ValueError("PCA samples contain invalid RADIO descriptors")
    if int(output_dim) <= 0 or int(output_dim) >= int(values.shape[1]):
        raise ValueError("PCA output dimension must be smaller than its input")
    if int(values.shape[0]) < max(int(output_dim) * 2, int(output_dim) + 1):
        raise ValueError("too few samples to fit the requested PCA projection")
    estimator = PCA(
        n_components=int(output_dim),
        svd_solver="randomized",
        random_state=int(seed),
    )
    estimator.fit(values)
    return (
        np.asarray(estimator.mean_, dtype=np.float32),
        np.asarray(estimator.components_, dtype=np.float32),
        {
            "explained_variance_ratio_sum": float(
                np.sum(estimator.explained_variance_ratio_)
            ),
            "explained_variance_ratio_min": float(
                np.min(estimator.explained_variance_ratio_)
            ),
        },
    )


@torch.no_grad()
def project_and_sample_radio_map(
    feature_map: torch.Tensor | np.ndarray,
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """Project a full map once, then endpoint-sample requested image pixels."""

    target = torch.device(device)
    values = torch.as_tensor(feature_map, dtype=torch.float32, device=target)
    if values.ndim != 3:
        raise ValueError("RADIO feature map must have shape (C, H, W)")
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    components = torch.as_tensor(pca_components, dtype=torch.float32, device=target)
    mean = torch.as_tensor(pca_mean, dtype=torch.float32, device=target).reshape(-1)
    if int(values.shape[0]) != int(components.shape[1]) or len(mean) != int(values.shape[0]):
        raise ValueError("RADIO map and PCA projection dimensions differ")
    if len(coordinates) == 0:
        return np.zeros((0, int(components.shape[0])), dtype=np.float32)
    flat = values.permute(1, 2, 0).reshape(-1, int(values.shape[0]))
    flat = F.normalize(flat, p=2, dim=1)
    projected = (flat - mean[None]) @ components.T
    projected = F.normalize(projected, p=2, dim=1)
    projected_map = projected.reshape(
        int(values.shape[1]), int(values.shape[2]), int(components.shape[0])
    ).permute(2, 0, 1)[None]
    grid = torch.as_tensor(coordinates, dtype=torch.float32, device=target)
    grid_x = 2.0 * grid[:, 0] / max(float(image_width - 1), 1.0) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(float(image_height - 1), 1.0) - 1.0
    normalized_grid = torch.stack([grid_x, grid_y], dim=1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        projected_map,
        normalized_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, 0].T
    sampled = F.normalize(sampled, p=2, dim=1)
    return sampled.cpu().numpy().astype(np.float32)


def save_radio_intermediate_context_cache(
    cache: RadioIntermediateContextCache,
    path: Path,
    *,
    cache_dtype: str,
) -> None:
    if str(cache_dtype) not in {"float16", "float32"}:
        raise ValueError("cache_dtype must be float16 or float32")
    dtype = np.float16 if str(cache_dtype) == "float16" else np.float32
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "radio_intermediate_observation_context_cache_v1",
        "descriptor_dim": int(cache.descriptor_dim),
        "cache_dtype": str(cache_dtype),
        **dict(cache.metadata),
    }
    np.savez(
        output,
        support_descriptors=cache.support_descriptors.astype(dtype),
        query_anchor_descriptors=cache.query_anchor_descriptors.astype(dtype),
        query_context_descriptors=cache.query_context_descriptors.astype(dtype),
        pca_mean=cache.pca_mean.astype(np.float32),
        pca_components=cache.pca_components.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def load_radio_intermediate_context_cache(
    path: Path,
    *,
    expected_metadata: Mapping[str, object] | None = None,
) -> RadioIntermediateContextCache:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != "radio_intermediate_observation_context_cache_v1":
            raise ValueError("unsupported RADIO intermediate context cache")
        if expected_metadata:
            mismatches = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"stale RADIO intermediate context cache: {json.dumps(mismatches, sort_keys=True)}"
                )
        return RadioIntermediateContextCache(
            support_descriptors=np.asarray(data["support_descriptors"], dtype=np.float32),
            query_anchor_descriptors=np.asarray(
                data["query_anchor_descriptors"], dtype=np.float32
            ),
            query_context_descriptors=np.asarray(
                data["query_context_descriptors"], dtype=np.float32
            ),
            pca_mean=np.asarray(data["pca_mean"], dtype=np.float32),
            pca_components=np.asarray(data["pca_components"], dtype=np.float32),
            metadata=metadata,
        )
