"""Patch-to-pixel 2D measurement refinement for VFM 2D-3D matches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch
from feature_extract.vfm.rendered_map_verifier import project_xyz_to_image
from feature_extract.vfm.tokens import TokenBankManifest


@dataclass(frozen=True)
class PatchOffsetTrainingSet:
    query_windows: np.ndarray
    landmark_features: np.ndarray
    match_stats: np.ndarray
    target_offsets: np.ndarray
    labels: np.ndarray
    stride_px: np.ndarray
    refine_labels: np.ndarray | None = None
    sample_weights: np.ndarray | None = None
    target_bins: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        windows = np.asarray(self.query_windows, dtype=np.float32)
        landmarks = np.asarray(self.landmark_features, dtype=np.float32)
        stats = np.asarray(self.match_stats, dtype=np.float32)
        targets = np.asarray(self.target_offsets, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.float32).reshape(-1)
        stride = np.asarray(self.stride_px, dtype=np.float32).reshape(-1)
        refine_labels = labels if self.refine_labels is None else np.asarray(self.refine_labels, dtype=np.float32).reshape(-1)
        sample_weights = np.ones_like(labels, dtype=np.float32) if self.sample_weights is None else np.asarray(self.sample_weights, dtype=np.float32).reshape(-1)
        target_bins = None if self.target_bins is None else np.asarray(self.target_bins, dtype=np.int64).reshape(-1)
        if windows.ndim != 4:
            raise ValueError("query_windows must have shape (N, K, K, D)")
        if landmarks.ndim != 2 or landmarks.shape[0] != windows.shape[0] or landmarks.shape[1] != windows.shape[3]:
            raise ValueError("landmark_features must have shape (N, D) matching query_windows")
        if stats.ndim != 2 or stats.shape[0] != windows.shape[0]:
            raise ValueError("match_stats must have shape (N, S)")
        if targets.shape != (windows.shape[0], 2):
            raise ValueError("target_offsets must have shape (N, 2)")
        if labels.shape[0] != windows.shape[0] or stride.shape[0] != windows.shape[0]:
            raise ValueError("labels and stride_px must have length N")
        if refine_labels.shape[0] != windows.shape[0] or sample_weights.shape[0] != windows.shape[0]:
            raise ValueError("refine_labels and sample_weights must have length N")
        if target_bins is not None and target_bins.shape[0] != windows.shape[0]:
            raise ValueError("target_bins must have length N")
        if windows.shape[1] != windows.shape[2] or windows.shape[1] % 2 != 1:
            raise ValueError("query_windows must use odd square windows")
        object.__setattr__(self, "query_windows", windows)
        object.__setattr__(self, "landmark_features", landmarks)
        object.__setattr__(self, "match_stats", stats)
        object.__setattr__(self, "target_offsets", targets)
        object.__setattr__(self, "labels", np.clip(labels, 0.0, 1.0))
        object.__setattr__(self, "stride_px", np.clip(stride, 1e-6, None))
        object.__setattr__(self, "refine_labels", np.clip(refine_labels, 0.0, 1.0))
        object.__setattr__(self, "sample_weights", np.clip(sample_weights, 0.0, None))
        object.__setattr__(self, "target_bins", target_bins)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def sample_count(self) -> int:
        return int(self.query_windows.shape[0])

    @property
    def window_size(self) -> int:
        return int(self.query_windows.shape[1]) if self.query_windows.ndim == 4 else 0

    @property
    def feature_dim(self) -> int:
        return int(self.query_windows.shape[3]) if self.query_windows.ndim == 4 else 0

    @property
    def stats_dim(self) -> int:
        return int(self.match_stats.shape[1]) if self.match_stats.ndim == 2 else 0


@dataclass(frozen=True)
class PatchOffsetRefinerConfig:
    feature_dim: int
    stats_dim: int
    window_size: int = 3
    hidden_dim: int = 128
    steps: int = 800
    batch_size: int = 512
    lr: float = 1e-3
    max_offset_stride: float = 0.5
    confidence_loss_weight: float = 0.5
    anchor_zero_weight: float = 0.0
    eval_split_fraction: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.stats_dim < 0:
            raise ValueError("feature_dim must be positive and stats_dim must be non-negative")
        if self.window_size <= 0 or self.window_size % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        if self.hidden_dim <= 0 or self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("hidden_dim, steps and batch_size must be positive")
        if self.lr <= 0.0 or self.max_offset_stride <= 0.0:
            raise ValueError("lr and max_offset_stride must be positive")
        if self.confidence_loss_weight < 0.0 or self.anchor_zero_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= float(self.eval_split_fraction) < 1.0:
            raise ValueError("eval_split_fraction must be in [0, 1)")


@dataclass(frozen=True)
class HeatmapPatchOffsetRefinerConfig(PatchOffsetRefinerConfig):
    bin_count: int = 8
    residual_loss_weight: float = 0.25
    bin_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.bin_count) <= 1:
            raise ValueError("bin_count must be greater than 1")
        if float(self.residual_loss_weight) < 0.0 or float(self.bin_loss_weight) < 0.0:
            raise ValueError("heatmap loss weights must be non-negative")


@dataclass(frozen=True)
class PatchOffsetTrainingSummary:
    initial_loss: float
    final_loss: float
    initial_positive_offset_mae_px: float
    final_positive_offset_mae_px: float
    raw_positive_offset_mae_px: float
    train_positive_count: int
    eval_positive_count: int
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    feature_dim: int
    stats_dim: int
    window_size: int
    steps: int
    batch_size: int
    max_offset_stride: float


@dataclass(frozen=True)
class PatchOffsetRefinerRun:
    model: "PatchOffsetRefiner"
    summary: PatchOffsetTrainingSummary


class PatchOffsetRefiner(nn.Module):
    """Tiny local-correlation head that predicts sub-token 2D offsets."""

    def __init__(self, feature_dim: int, stats_dim: int, window_size: int = 3, hidden_dim: int = 128, max_offset_stride: float = 0.5) -> None:
        super().__init__()
        if int(window_size) <= 0 or int(window_size) % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        self.feature_dim = int(feature_dim)
        self.stats_dim = int(stats_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.max_offset_stride = float(max_offset_stride)
        input_dim = self.window_size * self.window_size + self.stats_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(self, query_windows: torch.Tensor, landmark_features: torch.Tensor, match_stats: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        windows = F.normalize(query_windows, dim=-1, eps=1e-8)
        landmarks = F.normalize(landmark_features, dim=-1, eps=1e-8)
        corr = torch.sum(windows * landmarks[:, None, None, :], dim=-1)
        flat = corr.reshape(corr.shape[0], -1)
        if self.stats_dim:
            if match_stats is None:
                raise ValueError("match_stats is required when stats_dim > 0")
            flat = torch.cat([flat, match_stats], dim=1)
        raw = self.head(flat)
        offset = torch.tanh(raw[:, :2]) * float(self.max_offset_stride)
        log_sigma = torch.clamp(raw[:, 2:3], min=-4.0, max=4.0)
        confidence_logit = raw[:, 3:4]
        return {"offset": offset, "log_sigma": log_sigma, "confidence_logit": confidence_logit}


class HeatmapPatchOffsetRefiner(nn.Module):
    """Patch-bin classifier with a small residual offset head."""

    def __init__(
        self,
        feature_dim: int,
        stats_dim: int,
        window_size: int = 3,
        hidden_dim: int = 128,
        max_offset_stride: float = 0.5,
        bin_count: int = 8,
    ) -> None:
        super().__init__()
        if int(window_size) <= 0 or int(window_size) % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        if int(bin_count) <= 1:
            raise ValueError("bin_count must be greater than 1")
        self.feature_dim = int(feature_dim)
        self.stats_dim = int(stats_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.max_offset_stride = float(max_offset_stride)
        self.bin_count = int(bin_count)
        input_dim = self.window_size * self.window_size + self.stats_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.heatmap_head = nn.Linear(self.hidden_dim, self.bin_count * self.bin_count)
        self.residual_head = nn.Linear(self.hidden_dim, 2)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        self.log_sigma_head = nn.Linear(self.hidden_dim, 1)

    def _to_tensor(self, value: torch.Tensor | np.ndarray, *, device: torch.device | None = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value if device is None else value.to(device)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def _bin_centers(self, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(
            -float(self.max_offset_stride),
            float(self.max_offset_stride),
            steps=int(self.bin_count),
            dtype=torch.float32,
            device=device,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def forward(
        self,
        query_windows: torch.Tensor | np.ndarray,
        landmark_features: torch.Tensor | np.ndarray,
        match_stats: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        windows = self._to_tensor(query_windows)
        landmarks = self._to_tensor(landmark_features, device=windows.device)
        stats = None if match_stats is None else self._to_tensor(match_stats, device=windows.device)
        windows = F.normalize(windows, dim=-1, eps=1e-8)
        landmarks = F.normalize(landmarks, dim=-1, eps=1e-8)
        corr = torch.sum(windows * landmarks[:, None, None, :], dim=-1)
        flat = corr.reshape(corr.shape[0], -1)
        if self.stats_dim:
            if stats is None:
                raise ValueError("match_stats is required when stats_dim > 0")
            flat = torch.cat([flat, stats], dim=1)
        hidden = self.trunk(flat)
        logits = self.heatmap_head(hidden)
        probs = torch.softmax(logits, dim=1)
        centers = self._bin_centers(logits.device)
        coarse = probs @ centers
        bin_width = 2.0 * float(self.max_offset_stride) / max(int(self.bin_count) - 1, 1)
        residual = torch.tanh(self.residual_head(hidden)) * (0.5 * float(bin_width))
        offset = torch.clamp(coarse + residual, -float(self.max_offset_stride), float(self.max_offset_stride))
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1, keepdim=True)
        entropy_norm = entropy / float(np.log(self.bin_count * self.bin_count))
        learned_log_sigma = torch.clamp(self.log_sigma_head(hidden), min=-4.0, max=4.0)
        entropy_log_sigma = torch.log(torch.clamp(0.25 + entropy_norm * 2.0, min=1e-4))
        return {
            "offset": offset,
            "heatmap_logits": logits,
            "heatmap_entropy": entropy.reshape(-1),
            "log_sigma": torch.maximum(learned_log_sigma, entropy_log_sigma),
            "confidence_logit": self.confidence_head(hidden),
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _query_feature_for_token(feature_map: np.ndarray, token_index: int) -> tuple[np.ndarray, int, int]:
    values = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = values.shape
    y_idx, x_idx = divmod(int(token_index), int(width))
    if y_idx < 0 or y_idx >= height:
        raise ValueError(f"token_index {token_index} is outside feature map")
    return values[:, y_idx, x_idx].reshape(channels), y_idx, x_idx


def extract_query_window(feature_map: np.ndarray, token_index: int, window_size: int = 3) -> np.ndarray:
    values = np.asarray(feature_map, dtype=np.float32)
    channels, height, width = values.shape
    _center, y_idx, x_idx = _query_feature_for_token(values, int(token_index))
    radius = int(window_size) // 2
    padded = np.pad(values, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    patch = padded[:, y_idx : y_idx + int(window_size), x_idx : x_idx + int(window_size)]
    return np.moveaxis(patch, 0, -1).astype(np.float32, copy=True)


def _stride_for_feature_map(feature_map: np.ndarray, camera: ColmapCamera) -> float:
    _channels, height, width = np.asarray(feature_map).shape
    stride_x = float(camera.width - 1) / max(float(width - 1), 1.0)
    stride_y = float(camera.height - 1) / max(float(height - 1), 1.0)
    return float(max(stride_x, stride_y))


def _match_stats_from_row(row: Mapping[str, Any]) -> np.ndarray:
    margin = row.get("similarity_margin", 0.0)
    if margin is None:
        margin = 0.0
    values = [
        float(row.get("similarity", 0.0) or 0.0),
        float(margin),
        float(row.get("landmark_variance", 0.0) or 0.0),
        float(row.get("landmark_reprojection_error", 0.0) or 0.0),
        float(np.log1p(float(row.get("observation_count", 0.0) or 0.0))),
        float(row.get("landmark_quality", 0.0) or 0.0),
    ]
    return np.asarray(values, dtype=np.float32)


def _label_from_row(row: Mapping[str, Any], gt_error_stride: float, positive_stride: float, negative_stride: float) -> tuple[float, bool]:
    if bool(row.get("ignore_label", False)):
        return 0.0, True
    if bool(row.get("patch_positive_label", False)) or bool(row.get("stride_positive_label", False)) or gt_error_stride <= float(positive_stride):
        return 1.0, False
    if bool(row.get("hard_negative_label", False)) or gt_error_stride > float(negative_stride):
        return 0.0, False
    return 0.0, True


def _target_bins_for_offsets(offsets: np.ndarray, max_offset_stride: float, bin_count: int) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    clipped = np.clip(values, -float(max_offset_stride), float(max_offset_stride))
    scaled = (clipped + float(max_offset_stride)) / max(2.0 * float(max_offset_stride), 1e-6)
    idx = np.rint(scaled * (int(bin_count) - 1)).astype(np.int64)
    idx = np.clip(idx, 0, int(bin_count) - 1)
    return (idx[:, 1] * int(bin_count) + idx[:, 0]).astype(np.int64)


def _row_is_positive(row: Mapping[str, Any], gt_error_stride: float, mode: str, positive_stride: float, bounded_positive_stride: float) -> bool:
    mode = str(mode)
    patch_positive = bool(row.get("patch_positive_label", row.get("patch_correct", False)))
    stride_positive = bool(row.get("stride_positive_label", False)) or gt_error_stride <= float(positive_stride)
    bounded_positive = gt_error_stride <= float(bounded_positive_stride)
    if mode == "default":
        return patch_positive or stride_positive
    if mode == "patch":
        return patch_positive
    if mode == "bounded":
        return bounded_positive
    if mode == "patch_or_bounded":
        return patch_positive or bounded_positive
    if mode == "patch_and_bounded":
        return patch_positive and bounded_positive
    raise ValueError("positive_label_mode must be one of default, patch, bounded, patch_or_bounded, patch_and_bounded")


def build_patch_offset_samples_from_rows(
    match_jsonl: str | Path,
    query_manifest: str | Path,
    landmark_bank: str | Path,
    pose_by_query: Mapping[str, np.ndarray],
    camera: ColmapCamera,
    layer_name: str = "radio_final",
    window_size: int = 3,
    max_samples: int = 0,
    positive_stride: float = 1.0,
    negative_stride: float = 2.0,
    max_target_offset_stride: float = 1.0,
    require_pnp_inlier: bool = False,
    positive_label_mode: str = "default",
    bounded_positive_stride: float = 0.5,
    negative_sample_weight: float = 1.0,
    positive_sample_weight: float = 1.0,
    heatmap_bin_count: int = 8,
    seed: int = 0,
) -> tuple[PatchOffsetTrainingSet, dict[str, object]]:
    rows = _load_jsonl(Path(match_jsonl))
    manifest = TokenBankManifest.from_json(Path(query_manifest))
    manifest.validate(verify_checksums=False)
    record_by_query = {record.image_id: record for record in manifest.records}
    bank = load_selected_track_bank_npz(Path(landmark_bank))
    track_features = {int(track_id): np.asarray(track.mean_feature, dtype=np.float32) for track_id, track in bank.tracks.items()}
    query_cache: dict[str, np.ndarray] = {}
    samples = []
    skipped = 0
    positive_count = 0
    negative_count = 0
    pnp_inlier_count = 0
    non_refine_inlier_count = 0
    for row in rows:
        if bool(require_pnp_inlier) and not bool(row.get("pnp_inlier", False)):
            skipped += 1
            continue
        if bool(row.get("pnp_inlier", False)):
            pnp_inlier_count += 1
        query_id = str(row.get("query_id", ""))
        track_id = int(row.get("track_id", -1))
        if query_id not in record_by_query or query_id not in pose_by_query or track_id not in track_features:
            skipped += 1
            continue
        xy = np.asarray(row.get("xy", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        xyz = np.asarray(row.get("xyz", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(xyz)):
            skipped += 1
            continue
        projected = project_xyz_to_image(xyz, np.asarray(pose_by_query[query_id], dtype=np.float64), camera)
        if projected is None:
            skipped += 1
            continue
        if query_id not in query_cache:
            with np.load(record_by_query[query_id].token_path) as data:
                query_cache[query_id] = np.asarray(data[layer_name], dtype=np.float32)
        stride = _stride_for_feature_map(query_cache[query_id], camera)
        gt_xy = np.asarray(projected, dtype=np.float64)
        delta = (gt_xy - xy) / max(float(stride), 1e-6)
        gt_error_stride = float(np.linalg.norm(delta))
        if bool(require_pnp_inlier):
            ignore = False
            label = 1.0 if _row_is_positive(row, gt_error_stride, positive_label_mode, positive_stride, bounded_positive_stride) else 0.0
        else:
            label, ignore = _label_from_row(row, gt_error_stride, positive_stride, negative_stride)
        if ignore:
            skipped += 1
            continue
        if label > 0.5:
            positive_count += 1
        else:
            negative_count += 1
            if bool(require_pnp_inlier):
                non_refine_inlier_count += 1
        target = np.clip(delta, -float(max_target_offset_stride), float(max_target_offset_stride)).astype(np.float32)
        samples.append(
            (
                extract_query_window(query_cache[query_id], int(row.get("token_index", 0)), int(window_size)),
                track_features[track_id],
                _match_stats_from_row(row),
                target,
                float(label),
                float(stride),
                float(label),
                float(positive_sample_weight if label > 0.5 else negative_sample_weight),
            )
        )
    rng = np.random.default_rng(int(seed))
    if max_samples > 0 and len(samples) > int(max_samples):
        keep = sorted(rng.choice(len(samples), size=int(max_samples), replace=False).tolist())
        samples = [samples[int(idx)] for idx in keep]
    if samples:
        windows, landmarks, stats, targets, labels, strides, refine_labels, sample_weights = zip(*samples)
        target_offsets = np.stack(targets, axis=0)
        dataset = PatchOffsetTrainingSet(
            query_windows=np.stack(windows, axis=0),
            landmark_features=np.stack(landmarks, axis=0),
            match_stats=np.stack(stats, axis=0),
            target_offsets=target_offsets,
            labels=np.asarray(labels, dtype=np.float32),
            stride_px=np.asarray(strides, dtype=np.float32),
            refine_labels=np.asarray(refine_labels, dtype=np.float32),
            sample_weights=np.asarray(sample_weights, dtype=np.float32),
            target_bins=_target_bins_for_offsets(target_offsets, float(max_target_offset_stride), int(heatmap_bin_count)),
            metadata={
                "source_match_jsonl": str(match_jsonl),
                "window_size": int(window_size),
                "positive_stride": float(positive_stride),
                "negative_stride": float(negative_stride),
                "require_pnp_inlier": bool(require_pnp_inlier),
                "positive_label_mode": str(positive_label_mode),
                "bounded_positive_stride": float(bounded_positive_stride),
                "heatmap_bin_count": int(heatmap_bin_count),
            },
        )
    else:
        dataset = PatchOffsetTrainingSet(
            query_windows=np.zeros((0, int(window_size), int(window_size), bank.feature_dim), dtype=np.float32),
            landmark_features=np.zeros((0, bank.feature_dim), dtype=np.float32),
            match_stats=np.zeros((0, 6), dtype=np.float32),
            target_offsets=np.zeros((0, 2), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.float32),
            stride_px=np.zeros((0,), dtype=np.float32),
            refine_labels=np.zeros((0,), dtype=np.float32),
            sample_weights=np.zeros((0,), dtype=np.float32),
            target_bins=np.zeros((0,), dtype=np.int64),
        )
    return dataset, {
        "sample_count": int(dataset.sample_count),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "skipped_count": int(skipped),
        "pnp_inlier_count": int(pnp_inlier_count),
        "non_refine_inlier_count": int(non_refine_inlier_count),
        "window_size": int(window_size),
        "stats_dim": int(dataset.stats_dim),
        "positive_label_mode": str(positive_label_mode),
        "require_pnp_inlier": bool(require_pnp_inlier),
    }


def _split_indices(count: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    eval_count = int(round(count * float(eval_fraction)))
    if eval_fraction > 0.0 and count > 1:
        eval_count = max(1, eval_count)
    eval_count = min(eval_count, max(count - 1, 0))
    return indices[eval_count:], indices[:eval_count]


def _subset(samples: PatchOffsetTrainingSet, indices: np.ndarray) -> PatchOffsetTrainingSet:
    return PatchOffsetTrainingSet(
        query_windows=samples.query_windows[indices],
        landmark_features=samples.landmark_features[indices],
        match_stats=samples.match_stats[indices],
        target_offsets=samples.target_offsets[indices],
        labels=samples.labels[indices],
        stride_px=samples.stride_px[indices],
        refine_labels=samples.refine_labels[indices],
        sample_weights=samples.sample_weights[indices],
        target_bins=None if samples.target_bins is None else samples.target_bins[indices],
        metadata=samples.metadata,
    )


def _offset_loss(model: PatchOffsetRefiner, samples: PatchOffsetTrainingSet, config: PatchOffsetRefinerConfig, device: torch.device) -> torch.Tensor:
    windows = torch.as_tensor(samples.query_windows, dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(samples.landmark_features, dtype=torch.float32, device=device)
    stats = torch.as_tensor(samples.match_stats, dtype=torch.float32, device=device)
    targets = torch.as_tensor(samples.target_offsets, dtype=torch.float32, device=device)
    labels = torch.as_tensor(samples.labels, dtype=torch.float32, device=device)
    refine_labels = torch.as_tensor(samples.refine_labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(samples.sample_weights, dtype=torch.float32, device=device)
    pred = model(windows, landmarks, stats)
    pos = refine_labels > 0.5
    loss = torch.zeros((), dtype=torch.float32, device=device)
    if bool(pos.any().item()):
        residual = F.smooth_l1_loss(pred["offset"][pos], targets[pos], reduction="none").sum(dim=1, keepdim=True)
        sigma = torch.exp(pred["log_sigma"][pos])
        nll = residual / torch.clamp(sigma, min=1e-4) + pred["log_sigma"][pos]
        pos_weights = weights[pos].reshape(-1, 1)
        loss = loss + torch.sum(nll * pos_weights) / torch.clamp(torch.sum(pos_weights), min=1e-6)
    if float(config.confidence_loss_weight) > 0.0:
        bce = F.binary_cross_entropy_with_logits(pred["confidence_logit"].reshape(-1), labels, weight=weights, reduction="sum")
        bce = bce / torch.clamp(torch.sum(weights), min=1e-6)
        loss = loss + float(config.confidence_loss_weight) * bce
    if float(config.anchor_zero_weight) > 0.0:
        loss = loss + float(config.anchor_zero_weight) * torch.mean(pred["offset"] ** 2)
    return loss


def _heatmap_offset_loss(
    model: HeatmapPatchOffsetRefiner,
    samples: PatchOffsetTrainingSet,
    config: HeatmapPatchOffsetRefinerConfig,
    device: torch.device,
) -> torch.Tensor:
    windows = torch.as_tensor(samples.query_windows, dtype=torch.float32, device=device)
    landmarks = torch.as_tensor(samples.landmark_features, dtype=torch.float32, device=device)
    stats = torch.as_tensor(samples.match_stats, dtype=torch.float32, device=device)
    targets = torch.as_tensor(samples.target_offsets, dtype=torch.float32, device=device)
    labels = torch.as_tensor(samples.labels, dtype=torch.float32, device=device)
    refine_labels = torch.as_tensor(samples.refine_labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(samples.sample_weights, dtype=torch.float32, device=device)
    bins_np = _target_bins_for_offsets(samples.target_offsets, float(config.max_offset_stride), int(config.bin_count))
    target_bins = torch.as_tensor(bins_np, dtype=torch.long, device=device)
    pred = model(windows, landmarks, stats)
    pos = refine_labels > 0.5
    loss = torch.zeros((), dtype=torch.float32, device=device)
    if bool(pos.any().item()):
        pos_weights = weights[pos]
        ce = F.cross_entropy(pred["heatmap_logits"][pos], target_bins[pos], reduction="none")
        loss = loss + float(config.bin_loss_weight) * torch.sum(ce * pos_weights) / torch.clamp(torch.sum(pos_weights), min=1e-6)
        residual = F.smooth_l1_loss(pred["offset"][pos], targets[pos], reduction="none").sum(dim=1)
        loss = loss + float(config.residual_loss_weight) * torch.sum(residual * pos_weights) / torch.clamp(torch.sum(pos_weights), min=1e-6)
        sigma = torch.exp(pred["log_sigma"][pos]).reshape(-1)
        nll = residual / torch.clamp(sigma, min=1e-4) + torch.log(torch.clamp(sigma, min=1e-4))
        loss = loss + 0.1 * torch.sum(nll * pos_weights) / torch.clamp(torch.sum(pos_weights), min=1e-6)
    if float(config.confidence_loss_weight) > 0.0:
        bce = F.binary_cross_entropy_with_logits(pred["confidence_logit"].reshape(-1), labels, weight=weights, reduction="sum")
        loss = loss + float(config.confidence_loss_weight) * bce / torch.clamp(torch.sum(weights), min=1e-6)
    if float(config.anchor_zero_weight) > 0.0:
        loss = loss + float(config.anchor_zero_weight) * torch.mean(pred["offset"] ** 2)
    return loss


def _positive_offset_mae_px(model: PatchOffsetRefiner | None, samples: PatchOffsetTrainingSet, device: torch.device) -> float:
    if samples.sample_count == 0 or not np.any(samples.refine_labels > 0.5):
        return 0.0
    pos = samples.refine_labels > 0.5
    targets = samples.target_offsets[pos]
    stride = samples.stride_px[pos].reshape(-1, 1)
    if model is None:
        pred = np.zeros_like(targets, dtype=np.float32)
    else:
        model.eval()
        with torch.no_grad():
            out = model(
                torch.as_tensor(samples.query_windows[pos], dtype=torch.float32, device=device),
                torch.as_tensor(samples.landmark_features[pos], dtype=torch.float32, device=device),
                torch.as_tensor(samples.match_stats[pos], dtype=torch.float32, device=device),
            )
            pred = out["offset"].detach().cpu().numpy().astype(np.float32)
    return float(np.mean(np.linalg.norm((pred - targets) * stride, axis=1)))


def _loss_value(model: PatchOffsetRefiner, samples: PatchOffsetTrainingSet, config: PatchOffsetRefinerConfig, device: torch.device) -> float:
    if samples.sample_count == 0:
        return 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, min(int(config.batch_size), 2048)):
            end = min(start + min(int(config.batch_size), 2048), samples.sample_count)
            loss = _offset_loss(model, _subset(samples, np.arange(start, end)), config, device)
            total += float(loss.detach().cpu()) * (end - start)
            count += end - start
    return float(total / max(count, 1))


def _heatmap_loss_value(
    model: HeatmapPatchOffsetRefiner,
    samples: PatchOffsetTrainingSet,
    config: HeatmapPatchOffsetRefinerConfig,
    device: torch.device,
) -> float:
    if samples.sample_count == 0:
        return 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, samples.sample_count, min(int(config.batch_size), 2048)):
            end = min(start + min(int(config.batch_size), 2048), samples.sample_count)
            loss = _heatmap_offset_loss(model, _subset(samples, np.arange(start, end)), config, device)
            total += float(loss.detach().cpu()) * (end - start)
            count += end - start
    return float(total / max(count, 1))


def train_patch_offset_refiner(samples: PatchOffsetTrainingSet, config: PatchOffsetRefinerConfig | None = None) -> PatchOffsetRefinerRun:
    if samples.sample_count == 0:
        raise ValueError("at least one offset training sample is required")
    config = config or PatchOffsetRefinerConfig(feature_dim=samples.feature_dim, stats_dim=samples.stats_dim)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = PatchOffsetRefiner(
        feature_dim=int(config.feature_dim),
        stats_dim=int(config.stats_dim),
        window_size=int(config.window_size),
        hidden_dim=int(config.hidden_dim),
        max_offset_stride=float(config.max_offset_stride),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))
    initial_loss = _loss_value(model, train_samples, config, device)
    initial_mae = _positive_offset_mae_px(model, train_samples, device)
    for _step in range(int(config.steps)):
        count = min(int(config.batch_size), train_samples.sample_count)
        batch = rng.choice(train_samples.sample_count, size=count, replace=False)
        loss = _offset_loss(model, _subset(train_samples, batch), config, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = _loss_value(model, train_samples, config, device)
    final_mae = _positive_offset_mae_px(model, train_samples, device)
    summary = PatchOffsetTrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_positive_offset_mae_px=initial_mae,
        final_positive_offset_mae_px=final_mae,
        raw_positive_offset_mae_px=_positive_offset_mae_px(None, train_samples, device),
        train_positive_count=int(np.sum(train_samples.labels > 0.5)),
        eval_positive_count=int(np.sum(eval_samples.labels > 0.5)),
        sample_count=int(samples.sample_count),
        train_sample_count=int(train_samples.sample_count),
        eval_sample_count=int(eval_samples.sample_count),
        feature_dim=int(samples.feature_dim),
        stats_dim=int(samples.stats_dim),
        window_size=int(samples.window_size),
        steps=int(config.steps),
        batch_size=int(config.batch_size),
        max_offset_stride=float(config.max_offset_stride),
    )
    return PatchOffsetRefinerRun(model=model.cpu(), summary=summary)


def train_heatmap_patch_offset_refiner(
    samples: PatchOffsetTrainingSet,
    config: HeatmapPatchOffsetRefinerConfig | None = None,
) -> PatchOffsetRefinerRun:
    if samples.sample_count == 0:
        raise ValueError("at least one offset training sample is required")
    config = config or HeatmapPatchOffsetRefinerConfig(feature_dim=samples.feature_dim, stats_dim=samples.stats_dim)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    device = torch.device(config.device)
    train_idx, eval_idx = _split_indices(samples.sample_count, float(config.eval_split_fraction), int(config.seed))
    train_samples = _subset(samples, train_idx)
    eval_samples = _subset(samples, eval_idx) if eval_idx.size else _subset(samples, train_idx[:0])
    model = HeatmapPatchOffsetRefiner(
        feature_dim=int(config.feature_dim),
        stats_dim=int(config.stats_dim),
        window_size=int(config.window_size),
        hidden_dim=int(config.hidden_dim),
        max_offset_stride=float(config.max_offset_stride),
        bin_count=int(config.bin_count),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(config.seed))
    initial_loss = _heatmap_loss_value(model, train_samples, config, device)
    initial_mae = _positive_offset_mae_px(model, train_samples, device)
    for _step in range(int(config.steps)):
        count = min(int(config.batch_size), train_samples.sample_count)
        batch = rng.choice(train_samples.sample_count, size=count, replace=False)
        loss = _heatmap_offset_loss(model, _subset(train_samples, batch), config, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = _heatmap_loss_value(model, train_samples, config, device)
    final_mae = _positive_offset_mae_px(model, train_samples, device)
    summary = PatchOffsetTrainingSummary(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_positive_offset_mae_px=initial_mae,
        final_positive_offset_mae_px=final_mae,
        raw_positive_offset_mae_px=_positive_offset_mae_px(None, train_samples, device),
        train_positive_count=int(np.sum(train_samples.refine_labels > 0.5)),
        eval_positive_count=int(np.sum(eval_samples.refine_labels > 0.5)),
        sample_count=int(samples.sample_count),
        train_sample_count=int(train_samples.sample_count),
        eval_sample_count=int(eval_samples.sample_count),
        feature_dim=int(samples.feature_dim),
        stats_dim=int(samples.stats_dim),
        window_size=int(samples.window_size),
        steps=int(config.steps),
        batch_size=int(config.batch_size),
        max_offset_stride=float(config.max_offset_stride),
    )
    return PatchOffsetRefinerRun(model=model.cpu(), summary=summary)


def predict_patch_offsets_for_matches(
    query_feature: np.ndarray,
    landmark_index: LandmarkMapIndex,
    matches: Sequence[QueryTo3DMatch],
    model: PatchOffsetRefiner | HeatmapPatchOffsetRefiner,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_to_row = {int(track_id): idx for idx, track_id in enumerate(landmark_index.track_ids.tolist())}
    windows = []
    landmarks = []
    stats = []
    valid_positions = []
    for idx, match in enumerate(matches):
        row = track_to_row.get(int(match.track_id))
        if row is None:
            continue
        windows.append(extract_query_window(query_feature, int(match.token_index), int(model.window_size)))
        landmarks.append(np.asarray(landmark_index.features[row], dtype=np.float32))
        stats.append(
            np.asarray(
                [
                    float(match.similarity),
                    0.0 if match.similarity_margin is None else float(match.similarity_margin),
                    float(match.landmark_variance),
                    0.0 if match.landmark_reprojection_error is None else float(match.landmark_reprojection_error),
                    float(np.log1p(0 if match.observation_count is None else match.observation_count)),
                    0.0 if match.landmark_quality is None else float(match.landmark_quality),
                ],
                dtype=np.float32,
            )
        )
        valid_positions.append(idx)
    offsets = np.zeros((len(matches), 2), dtype=np.float32)
    confidences = np.zeros((len(matches),), dtype=np.float32)
    sigmas = np.ones((len(matches),), dtype=np.float32)
    if not windows:
        return offsets, confidences, sigmas
    torch_device = torch.device(device)
    was_training = model.training
    model = model.to(torch_device)
    model.eval()
    windows_arr = np.stack(windows, axis=0)
    landmarks_arr = np.stack(landmarks, axis=0)
    stats_arr = np.stack(stats, axis=0)
    chunks_offset = []
    chunks_conf = []
    chunks_sigma = []
    with torch.no_grad():
        for start in range(0, windows_arr.shape[0], int(batch_size)):
            end = min(start + int(batch_size), windows_arr.shape[0])
            out = model(
                torch.as_tensor(windows_arr[start:end], dtype=torch.float32, device=torch_device),
                torch.as_tensor(landmarks_arr[start:end], dtype=torch.float32, device=torch_device),
                torch.as_tensor(stats_arr[start:end], dtype=torch.float32, device=torch_device),
            )
            chunks_offset.append(out["offset"].detach().cpu().numpy().astype(np.float32))
            chunks_conf.append(torch.sigmoid(out["confidence_logit"]).reshape(-1).detach().cpu().numpy().astype(np.float32))
            chunks_sigma.append(torch.exp(out["log_sigma"]).reshape(-1).detach().cpu().numpy().astype(np.float32))
    if was_training:
        model.train()
    pred_offsets = np.concatenate(chunks_offset, axis=0)
    pred_conf = np.concatenate(chunks_conf, axis=0)
    pred_sigma = np.concatenate(chunks_sigma, axis=0)
    for out_idx, match_idx in enumerate(valid_positions):
        offsets[int(match_idx)] = pred_offsets[out_idx]
        confidences[int(match_idx)] = pred_conf[out_idx]
        sigmas[int(match_idx)] = pred_sigma[out_idx]
    return offsets, confidences, sigmas


def apply_predicted_patch_offsets(
    matches: Sequence[QueryTo3DMatch],
    offsets: np.ndarray,
    confidences: np.ndarray,
    stride_px: float,
    confidence_threshold: float = 0.5,
    inlier_mask: np.ndarray | None = None,
    max_offset_stride: float = 0.5,
    sigmas: np.ndarray | None = None,
    max_sigma: float | None = None,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    offsets = np.asarray(offsets, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if offsets.shape != (len(matches), 2) or confidences.shape[0] != len(matches):
        raise ValueError("offset and confidence arrays must match matches")
    gate = confidences >= float(confidence_threshold)
    if inlier_mask is not None:
        gate &= np.asarray(inlier_mask, dtype=bool).reshape(-1)
    rejected_by_sigma = 0
    if sigmas is not None and max_sigma is not None:
        sigma_values = np.asarray(sigmas, dtype=np.float32).reshape(-1)
        if sigma_values.shape[0] != len(matches):
            raise ValueError("sigmas must have one value per match")
        sigma_gate = sigma_values <= float(max_sigma)
        rejected_by_sigma = int(np.sum(gate & ~sigma_gate))
        gate &= sigma_gate
    refined_count = 0
    offset_norms = []
    for idx, match in enumerate(matches):
        if not bool(gate[idx]):
            continue
        delta = np.clip(offsets[idx], -float(max_offset_stride), float(max_offset_stride)) * float(stride_px)
        refined[idx] = replace(match, xy=np.asarray(match.xy, dtype=np.float64).reshape(2) + delta.astype(np.float64))
        refined_count += 1
        offset_norms.append(float(np.linalg.norm(delta)))
    return refined, {
        "mode": "learned",
        "refined_count": int(refined_count),
        "mean_offset_px": None if not offset_norms else float(np.mean(offset_norms)),
        "mean_confidence": float(np.mean(confidences)) if confidences.size else 0.0,
        "confidence_threshold": float(confidence_threshold),
        "max_sigma": None if max_sigma is None else float(max_sigma),
        "rejected_by_sigma_count": int(rejected_by_sigma),
    }


def refine_matches_with_oracle_offsets(
    matches: Sequence[QueryTo3DMatch],
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    stride_px: float,
    inlier_mask: np.ndarray | None = None,
    max_offset_stride: float | None = 2.0,
    bound_metric: str = "l2",
    patch_positive_by_token: Mapping[int, set[int]] | None = None,
    require_patch_positive: bool = False,
    noise_sigma_px: float = 0.0,
    rng_seed: int = 0,
) -> tuple[list[QueryTo3DMatch], dict[str, object]]:
    refined = list(matches)
    gate = np.ones((len(matches),), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if gate.shape[0] != len(matches):
        raise ValueError("inlier_mask must have one value per match")
    metric = str(bound_metric).lower()
    if metric not in {"l2", "linf"}:
        raise ValueError("bound_metric must be 'l2' or 'linf'")
    patch_positive_by_token = patch_positive_by_token or {}
    rng = np.random.default_rng(int(rng_seed))
    refined_count = 0
    rejected_by_gate = 0
    rejected_by_projection = 0
    rejected_by_bound = 0
    rejected_by_patch_positive = 0
    offset_norms = []
    for idx, match in enumerate(matches):
        if not bool(gate[idx]):
            rejected_by_gate += 1
            continue
        if bool(require_patch_positive):
            positives = patch_positive_by_token.get(int(match.token_index), set())
            if int(match.track_id) not in positives:
                rejected_by_patch_positive += 1
                continue
        projected = project_xyz_to_image(np.asarray(match.xyz, dtype=np.float64), pose_w2c, camera)
        if projected is None:
            rejected_by_projection += 1
            continue
        target = np.asarray(projected, dtype=np.float64).reshape(2)
        delta = target - np.asarray(match.xy, dtype=np.float64).reshape(2)
        if max_offset_stride is not None:
            threshold = float(max_offset_stride) * float(stride_px)
            offset_size = float(np.linalg.norm(delta, ord=np.inf if metric == "linf" else 2))
            if offset_size > threshold:
                rejected_by_bound += 1
                continue
        if float(noise_sigma_px) > 0.0:
            target = target + rng.normal(0.0, float(noise_sigma_px), size=(2,))
            delta = target - np.asarray(match.xy, dtype=np.float64).reshape(2)
        refined[idx] = replace(match, xy=target)
        refined_count += 1
        offset_norms.append(float(np.linalg.norm(delta)))
    return refined, {
        "mode": "oracle",
        "refined_count": int(refined_count),
        "mean_offset_px": None if not offset_norms else float(np.mean(offset_norms)),
        "bounded": max_offset_stride is not None,
        "max_offset_stride": None if max_offset_stride is None else float(max_offset_stride),
        "bound_metric": metric,
        "noise_sigma_px": float(noise_sigma_px),
        "require_patch_positive": bool(require_patch_positive),
        "rejected_by_gate_count": int(rejected_by_gate),
        "rejected_by_projection_count": int(rejected_by_projection),
        "rejected_by_bound_count": int(rejected_by_bound),
        "rejected_by_patch_positive_count": int(rejected_by_patch_positive),
    }


def save_patch_offset_refiner_checkpoint(run: PatchOffsetRefinerRun, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    is_heatmap = isinstance(run.model, HeatmapPatchOffsetRefiner)
    torch.save(
        {
            "format": "vfm_patch_offset_refiner_v2" if is_heatmap else "vfm_patch_offset_refiner_v1",
            "model_type": "heatmap" if is_heatmap else "regression",
            "model_config": {
                "feature_dim": int(run.model.feature_dim),
                "stats_dim": int(run.model.stats_dim),
                "window_size": int(run.model.window_size),
                "hidden_dim": int(run.model.hidden_dim),
                "max_offset_stride": float(run.model.max_offset_stride),
                **({"bin_count": int(run.model.bin_count)} if is_heatmap else {}),
            },
            "state_dict": run.model.state_dict(),
            "summary": asdict(run.summary),
        },
        output,
    )


def load_patch_offset_refiner_checkpoint(path: str | Path, device: str = "cpu") -> PatchOffsetRefinerRun:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") not in {"vfm_patch_offset_refiner_v1", "vfm_patch_offset_refiner_v2"}:
        raise ValueError("unsupported patch offset refiner checkpoint format")
    cfg = payload["model_config"]
    if payload.get("model_type") == "heatmap":
        model = HeatmapPatchOffsetRefiner(
            feature_dim=int(cfg["feature_dim"]),
            stats_dim=int(cfg["stats_dim"]),
            window_size=int(cfg["window_size"]),
            hidden_dim=int(cfg["hidden_dim"]),
            max_offset_stride=float(cfg["max_offset_stride"]),
            bin_count=int(cfg.get("bin_count", 8)),
        )
    else:
        model = PatchOffsetRefiner(
            feature_dim=int(cfg["feature_dim"]),
            stats_dim=int(cfg["stats_dim"]),
            window_size=int(cfg["window_size"]),
            hidden_dim=int(cfg["hidden_dim"]),
            max_offset_stride=float(cfg["max_offset_stride"]),
        )
    model.load_state_dict(payload["state_dict"], strict=False)
    summary = PatchOffsetTrainingSummary(**payload["summary"])
    return PatchOffsetRefinerRun(model=model, summary=summary)
