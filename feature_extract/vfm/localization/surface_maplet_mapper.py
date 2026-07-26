"""RADIO-final mapper specialized for 2DGS surface-maplet retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extract.vfm.gaussian_raw_landmarks import vfm_token_saliency
from feature_extract.vfm.localization.schemas import MappedFeatureMap


@dataclass(frozen=True)
class SurfaceMapletMapperConfig:
    input_dim: int = 1280
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if int(self.input_dim) <= 0 or int(self.hidden_dim) <= 0 or int(self.output_dim) <= 0:
            raise ValueError("surface maplet mapper dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SurfaceMapletMapper(nn.Module):
    """Residual 1x1 full-map adapter; spatial context is pooled afterward."""

    def __init__(self, config: SurfaceMapletMapperConfig = SurfaceMapletMapperConfig()):
        super().__init__()
        self.config = config
        self.input_norm = nn.GroupNorm(1, int(config.input_dim))
        self.mlp = nn.Sequential(
            nn.Conv2d(int(config.input_dim), int(config.hidden_dim), kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(float(config.dropout)),
            nn.Conv2d(int(config.hidden_dim), int(config.output_dim), kernel_size=1),
        )
        self.shortcut = nn.Conv2d(int(config.input_dim), int(config.output_dim), kernel_size=1, bias=False)
        self.output_scale = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        if feature_maps.ndim != 4 or int(feature_maps.shape[1]) != int(self.config.input_dim):
            raise ValueError("feature_maps must have shape (B, input_dim, H, W)")
        normalized_input = self.input_norm(feature_maps)
        output = self.shortcut(feature_maps) + self.output_scale * self.mlp(normalized_input)
        return F.normalize(output, p=2, dim=1, eps=1e-8)


def surface_maplet_contrastive_loss(
    descriptors: torch.Tensor,
    maplet_labels: torch.Tensor,
    image_labels: torch.Tensor,
    quality_weights: torch.Tensor | None = None,
    maplet_centers: torch.Tensor | None = None,
    temperature: float = 0.07,
    hard_negative_radius: float = 0.75,
    hard_negative_margin: float = 0.25,
    hard_negative_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Cross-view set contrastive loss using only persistent 2DGS maplet identity.

    Same-maplet observations from different real mapping views are positives.
    Observations of other maplets are negatives; nearby maplets receive an
    additional separation margin because they are the most damaging aliases for
    local pose estimation. Same-maplet observations from the same image are
    excluded rather than incorrectly treated as negatives.
    """

    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape (N, C)")
    count = int(descriptors.shape[0])
    labels = maplet_labels.reshape(-1)
    images = image_labels.reshape(-1)
    if labels.shape != (count,) or images.shape != (count,):
        raise ValueError("maplet_labels and image_labels must have shape (N,)")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if count < 2:
        return descriptors.sum() * 0.0, {"valid_anchor_count": 0, "positive_pair_count": 0}

    normalized = F.normalize(descriptors, p=2, dim=1, eps=1e-8)
    similarity = normalized @ normalized.T
    identity = torch.eye(count, dtype=torch.bool, device=descriptors.device)
    same_maplet = labels[:, None] == labels[None, :]
    different_view = images[:, None] != images[None, :]
    positive = same_maplet & different_view
    negative = ~same_maplet
    denominator = (positive | negative) & ~identity
    valid = torch.any(positive, dim=1) & torch.any(denominator, dim=1)
    if not bool(torch.any(valid)):
        return descriptors.sum() * 0.0, {"valid_anchor_count": 0, "positive_pair_count": 0}

    logits = similarity / float(temperature)
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logsumexp = torch.logsumexp(
        torch.where(positive, logits, negative_infinity),
        dim=1,
    )
    denominator_logsumexp = torch.logsumexp(
        torch.where(denominator, logits, negative_infinity),
        dim=1,
    )
    per_anchor = denominator_logsumexp - positive_logsumexp
    if quality_weights is None:
        weights = torch.ones((count,), dtype=descriptors.dtype, device=descriptors.device)
    else:
        weights = torch.clamp(
            quality_weights.reshape(-1).to(device=descriptors.device, dtype=descriptors.dtype),
            min=0.0,
        )
        if weights.shape != (count,):
            raise ValueError("quality_weights must have shape (N,)")
    valid_weights = weights[valid]
    contrastive = torch.sum(per_anchor[valid] * valid_weights) / torch.clamp(
        torch.sum(valid_weights),
        min=1e-8,
    )

    hard_negative = descriptors.sum() * 0.0
    hard_pair_count = 0
    if (
        maplet_centers is not None
        and float(hard_negative_weight) > 0.0
        and float(hard_negative_radius) > 0.0
    ):
        centers = maplet_centers.to(device=descriptors.device, dtype=descriptors.dtype)
        if centers.shape != (count, 3):
            raise ValueError("maplet_centers must have shape (N, 3)")
        distance = torch.cdist(centers, centers)
        hard_mask = negative & (distance <= float(hard_negative_radius)) & ~identity
        hard_pair_count = int(torch.sum(hard_mask).detach().cpu().item())
        if hard_pair_count > 0:
            pair_weight = torch.sqrt(torch.clamp(weights[:, None] * weights[None, :], min=0.0))
            penalties = F.relu(similarity - float(hard_negative_margin))
            hard_negative = torch.sum(penalties[hard_mask] * pair_weight[hard_mask]) / torch.clamp(
                torch.sum(pair_weight[hard_mask]),
                min=1e-8,
            )
    loss = contrastive + float(hard_negative_weight) * hard_negative
    return loss, {
        "valid_anchor_count": int(torch.sum(valid).detach().cpu().item()),
        "positive_pair_count": int(torch.sum(positive).detach().cpu().item()),
        "hard_negative_pair_count": int(hard_pair_count),
        "contrastive_loss": float(contrastive.detach().cpu().item()),
        "hard_negative_loss": float(hard_negative.detach().cpu().item()),
    }


def maplet_prototype_retrieval_metrics(
    descriptors: np.ndarray,
    maplet_labels: np.ndarray,
    train_mask: np.ndarray,
    query_mask: np.ndarray,
    quality_weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate held-out real views against prototypes built from train views."""

    feature = np.asarray(descriptors, dtype=np.float32)
    labels = np.asarray(maplet_labels, dtype=np.int64).reshape(-1)
    train = np.asarray(train_mask, dtype=bool).reshape(-1)
    query = np.asarray(query_mask, dtype=bool).reshape(-1)
    count = int(feature.shape[0]) if feature.ndim == 2 else -1
    if count < 0 or labels.shape != (count,) or train.shape != (count,) or query.shape != (count,):
        raise ValueError("retrieval arrays must share observation dimension")
    if np.any(train & query):
        raise ValueError("train and query masks must be disjoint")
    if quality_weights is None:
        weights = np.ones((count,), dtype=np.float32)
    else:
        weights = np.maximum(np.asarray(quality_weights, dtype=np.float32).reshape(-1), 0.0)
        if weights.shape != (count,):
            raise ValueError("quality_weights must have shape (N,)")

    prototype_labels = np.unique(labels[train])
    prototypes: list[np.ndarray] = []
    retained_labels: list[int] = []
    for label in prototype_labels.tolist():
        rows = np.flatnonzero(train & (labels == int(label)))
        if rows.size == 0:
            continue
        row_weights = weights[rows]
        if float(np.sum(row_weights)) <= 0.0:
            row_weights = np.ones_like(row_weights)
        prototype = np.average(feature[rows], axis=0, weights=row_weights)
        norm = float(np.linalg.norm(prototype))
        if norm <= 1e-8:
            continue
        prototypes.append((prototype / norm).astype(np.float32, copy=False))
        retained_labels.append(int(label))
    eligible = query & np.isin(labels, np.asarray(retained_labels, dtype=np.int64))
    rows = np.flatnonzero(eligible)
    if not prototypes or rows.size == 0:
        return {
            "prototype_count": int(len(prototypes)),
            "query_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "median_rank": 0.0,
            "mean_reciprocal_rank": 0.0,
        }
    prototype_matrix = np.stack(prototypes, axis=0)
    prototype_label_array = np.asarray(retained_labels, dtype=np.int64)
    query_feature = feature[rows]
    query_feature /= np.maximum(np.linalg.norm(query_feature, axis=1, keepdims=True), 1e-8)
    scores = query_feature @ prototype_matrix.T
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked_labels = prototype_label_array[order]
    ranks = np.argmax(ranked_labels == labels[rows, None], axis=1) + 1
    return {
        "prototype_count": int(len(prototypes)),
        "query_count": int(rows.size),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
    }


def select_spatially_balanced_radio_final_regions(
    raw_feature_map: np.ndarray,
    grid_rows: int = 8,
    grid_cols: int = 8,
    regions_per_cell: int = 2,
    saliency_mode: str = "local_contrast",
) -> tuple[np.ndarray, np.ndarray]:
    """Select query regions from RADIO final without geometry or pose input."""

    feature = np.asarray(raw_feature_map, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("raw_feature_map must have shape (C, H, W)")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0 or int(regions_per_cell) <= 0:
        raise ValueError("query region grid parameters must be positive")
    saliency = vfm_token_saliency(feature, mode=str(saliency_mode))
    height, width = saliency.shape
    y_edges = np.linspace(0, height, int(grid_rows) + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, int(grid_cols) + 1, dtype=np.int64)
    selected: list[int] = []
    for row in range(int(grid_rows)):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(int(grid_cols)):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            if y1 <= y0 or x1 <= x0:
                continue
            local = saliency[y0:y1, x0:x1].reshape(-1)
            order = np.argsort(-local, kind="mergesort")[: int(regions_per_cell)]
            for local_index in order.tolist():
                y = y0 + int(local_index) // (x1 - x0)
                x = x0 + int(local_index) % (x1 - x0)
                selected.append(y * width + x)
    token_indices = np.asarray(sorted(set(selected)), dtype=np.int64)
    token_xy = np.stack(
        [token_indices % width, token_indices // width],
        axis=1,
    ).astype(np.float32)
    return token_indices, token_xy


@dataclass
class SurfaceMapletFeatureMapper:
    model: SurfaceMapletMapper
    device: str = "cpu"

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        feature = np.asarray(feature_map, dtype=np.float32)
        if feature.ndim != 3:
            raise ValueError("feature_map must have shape (C, H, W)")
        torch_device = torch.device(
            self.device
            if torch.cuda.is_available() or not str(self.device).startswith("cuda")
            else "cpu"
        )
        was_training = self.model.training
        model = self.model.to(torch_device).eval()
        with torch.no_grad():
            output = model(torch.as_tensor(feature[None], device=torch_device))[0]
        if was_training:
            self.model.train()
        descriptors = output.detach().cpu().numpy().astype(np.float32, copy=False)
        return MappedFeatureMap(
            coarse_descriptors=descriptors,
            measurement_context=descriptors,
            offset_logits=None,
            heatmap=None,
        )


def save_surface_maplet_mapper(
    path: Path,
    model: SurfaceMapletMapper,
    metadata: dict[str, object] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "surface_maplet_mapper",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        Path(path),
    )


def load_surface_maplet_mapper(
    path: Path,
    device: str = "cpu",
) -> tuple[SurfaceMapletFeatureMapper, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    if str(payload.get("artifact_type", "")) != "surface_maplet_mapper":
        raise ValueError("checkpoint is not a surface-maplet mapper")
    config = SurfaceMapletMapperConfig(**dict(payload["config"]))
    model = SurfaceMapletMapper(config)
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    model.eval()
    metadata = dict(payload.get("metadata", {}))
    if bool(metadata.get("uses_radio_intermediate", False)):
        raise ValueError("surface-maplet mapper illegally uses RADIO intermediate")
    if bool(metadata.get("uses_sfm_points", False)):
        raise ValueError("surface-maplet mapper illegally uses SfM points")
    if bool(metadata.get("uses_sfm_tracks", False)):
        raise ValueError("surface-maplet mapper illegally uses SfM tracks")
    return SurfaceMapletFeatureMapper(model=model, device=str(device)), metadata


def pool_radio_final_context_torch(
    mapped_feature_map: torch.Tensor,
    token_xy: torch.Tensor,
    pool_sizes: tuple[int, ...] = (1, 3, 5, 9),
    pool_weights: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10),
) -> torch.Tensor:
    """Differentiable equivalent of final-layer multi-scale region encoding."""

    if mapped_feature_map.ndim != 3:
        raise ValueError("mapped_feature_map must have shape (C, H, W)")
    if token_xy.ndim != 2 or int(token_xy.shape[1]) != 2:
        raise ValueError("token_xy must have shape (N, 2)")
    if len(pool_sizes) != len(pool_weights):
        raise ValueError("pool sizes and weights must align")
    _channels, height, width = mapped_feature_map.shape
    x = torch.clamp(torch.round(token_xy[:, 0]).long(), 0, width - 1)
    y = torch.clamp(torch.round(token_xy[:, 1]).long(), 0, height - 1)
    descriptor = torch.zeros(
        (token_xy.shape[0], mapped_feature_map.shape[0]),
        dtype=mapped_feature_map.dtype,
        device=mapped_feature_map.device,
    )
    total_weight = 0.0
    source = mapped_feature_map[None]
    for size, weight in zip(pool_sizes, pool_weights):
        if int(size) <= 0 or int(size) % 2 == 0 or float(weight) < 0.0:
            raise ValueError("pool sizes must be positive odd and weights non-negative")
        if float(weight) == 0.0:
            continue
        pooled = F.avg_pool2d(
            source,
            kernel_size=int(size),
            stride=1,
            padding=int(size) // 2,
            count_include_pad=False,
        )[0]
        values = pooled[:, y, x].T
        descriptor = descriptor + float(weight) * F.normalize(values, p=2, dim=1, eps=1e-8)
        total_weight += float(weight)
    if total_weight <= 0.0:
        raise ValueError("at least one pooling weight must be positive")
    return F.normalize(descriptor / total_weight, p=2, dim=1, eps=1e-8)
