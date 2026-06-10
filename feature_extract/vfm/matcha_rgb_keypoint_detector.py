"""MATCHA-style RGB-local 65-bin keypoint detector.

This module mirrors MATCHA's ALIKE distillation head: an RGB image is encoded
with a shallow convolutional stem, each 8x8 window is unfolded into one cell
descriptor, and the head predicts 64 sub-cell bins plus one non-keypoint bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.matcha_keypoint_distillation import NON_KEYPOINT_LABEL


class BasicConvLayer(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU, matching MATCHA's BasicLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                int(kernel_size),
                padding=int(padding),
                stride=int(stride),
                dilation=int(dilation),
                bias=bool(bias),
            ),
            nn.BatchNorm2d(int(out_channels), affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


class MatchaRgbKeypointDetector(nn.Module):
    """MATCHA-style 65-bin detector trained by ALIKE keypoint distillation."""

    def __init__(self, window_size: int = 8, stem_channels: int = 16, hidden_channels: int = 64) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.stem_channels = int(stem_channels)
        self.hidden_channels = int(hidden_channels)
        self.keypoint_encoder = BasicConvLayer(3, int(stem_channels), 3, padding=1)
        self.keypoint_head = nn.Sequential(
            BasicConvLayer(int(stem_channels) * int(window_size) ** 2, int(hidden_channels), 1, padding=0),
            BasicConvLayer(int(hidden_channels), int(hidden_channels), 1, padding=0),
            BasicConvLayer(int(hidden_channels), int(hidden_channels), 1, padding=0),
            nn.Conv2d(int(hidden_channels), 65, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (B, 3, H, W)")
        window = int(self.window_size)
        if int(image.shape[2]) % window != 0 or int(image.shape[3]) % window != 0:
            raise ValueError("image height and width must be divisible by window_size")
        features = self.keypoint_encoder(image)
        batch, channels, height, width = features.shape
        features = (
            features.unfold(2, window, window)
            .unfold(3, window, window)
            .reshape(batch, channels, height // window, width // window, window**2)
        )
        features = features.permute(0, 1, 4, 2, 3).reshape(batch, channels * window**2, height // window, width // window)
        return self.keypoint_head(features)


@dataclass(frozen=True)
class MatchaAlikeDistillationMetrics:
    loss: float
    acc: float
    positive_acc: float
    non_keypoint_acc: float
    positive_count: int
    non_keypoint_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": float(self.loss),
            "acc": float(self.acc),
            "positive_acc": float(self.positive_acc),
            "non_keypoint_acc": float(self.non_keypoint_acc),
            "positive_count": int(self.positive_count),
            "non_keypoint_count": int(self.non_keypoint_count),
        }


def matcha_alike_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    non_keypoint_divisor: int = 32,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute MATCHA-style ALIKE distillation CE over positive cells and sampled dustbins."""

    if logits.ndim != 4 or logits.shape[1] != 65:
        raise ValueError("logits must have shape (B, 65, H, W)")
    if labels.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("labels must have shape (B, H, W)")
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, 65)
    flat_labels = labels.reshape(-1).long()
    if torch.any(flat_labels < 0) or torch.any(flat_labels > NON_KEYPOINT_LABEL):
        raise ValueError("labels must be in [0, 64]")
    positive = torch.nonzero(flat_labels < NON_KEYPOINT_LABEL, as_tuple=False).flatten()
    negative = torch.nonzero(flat_labels == NON_KEYPOINT_LABEL, as_tuple=False).flatten()
    if positive.numel() == 0:
        return flat_logits.sum() * 0.0, MatchaAlikeDistillationMetrics(0.0, 0.0, 0.0, 0.0, 0, 0).as_dict()
    divisor = max(int(non_keypoint_divisor), 1)
    count = min(int(negative.numel()), int(positive.numel()) // divisor)
    if count == 0:
        keep_negative = negative[:0]
    else:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=negative.device)
            generator.manual_seed(int(seed))
        order = torch.randperm(int(negative.numel()), device=negative.device, generator=generator)[:count]
        keep_negative = negative[order]
    keep = torch.cat([positive, keep_negative], dim=0)
    selected_logits = flat_logits[keep]
    selected_labels = flat_labels[keep]
    loss = F.cross_entropy(selected_logits, selected_labels)
    with torch.no_grad():
        pred = torch.argmax(selected_logits, dim=1)
        selected_positive = selected_labels < NON_KEYPOINT_LABEL
        selected_non_keypoint = selected_labels == NON_KEYPOINT_LABEL

        def acc(mask: torch.Tensor) -> float:
            if not torch.any(mask):
                return 0.0
            return float(torch.mean((pred[mask] == selected_labels[mask]).float()).item())

        metrics = MatchaAlikeDistillationMetrics(
            loss=float(loss.detach().cpu().item()),
            acc=float(torch.mean((pred == selected_labels).float()).item()),
            positive_acc=acc(selected_positive),
            non_keypoint_acc=acc(selected_non_keypoint),
            positive_count=int(torch.count_nonzero(selected_positive).item()),
            non_keypoint_count=int(torch.count_nonzero(selected_non_keypoint).item()),
        )
    return loss, metrics.as_dict()


def matcha_keypoint_position_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_points_xy: torch.Tensor,
    target_points_xy: torch.Tensor,
    *,
    point_batch_indices: torch.Tensor | None = None,
    softmax_temp: float = 1.0,
) -> tuple[torch.Tensor, float, dict[str, int]]:
    """MATCHA keypoint offset roundtrip loss for a 65-bin detector.

    The 65th dustbin is excluded from the coordinate classifier. Source cells
    predicted as dustbin are not used as roundtrip anchors; target labels are
    derived from the corresponding target point offsets, matching MATCHA's
    original ``keypoint_position_loss`` logic.
    """

    if source_logits.ndim != 4 or target_logits.ndim != 4:
        raise ValueError("source_logits and target_logits must have shape (B, 65, H, W)")
    if source_logits.shape != target_logits.shape:
        raise ValueError("source_logits and target_logits must have matching shape")
    if int(source_logits.shape[1]) != 65:
        raise ValueError("source_logits and target_logits must have 65 channels")
    points1 = torch.as_tensor(source_points_xy, dtype=torch.float32, device=source_logits.device).reshape(-1, 2)
    points2 = torch.as_tensor(target_points_xy, dtype=torch.float32, device=source_logits.device).reshape(-1, 2)
    if points1.shape != points2.shape:
        raise ValueError("source_points_xy and target_points_xy must have matching shape")
    batch, _channels, height, width = source_logits.shape
    if point_batch_indices is None:
        batches = torch.zeros((points1.shape[0],), dtype=torch.long, device=source_logits.device)
    else:
        batches = torch.as_tensor(point_batch_indices, dtype=torch.long, device=source_logits.device).reshape(-1)
        if batches.shape[0] != points1.shape[0]:
            raise ValueError("point_batch_indices must contain one value per point pair")
    source_spatial = source_logits[:, :NON_KEYPOINT_LABEL].permute(0, 2, 3, 1) * float(softmax_temp)
    target_spatial = target_logits[:, :NON_KEYPOINT_LABEL].permute(0, 2, 3, 1) * float(softmax_temp)
    full_source_labels = torch.argmax(source_logits, dim=1)
    source_candidate_count = int(torch.count_nonzero(full_source_labels < NON_KEYPOINT_LABEL).detach().cpu().item())

    selected_source_logits = []
    selected_source_labels = []
    selected_target_logits = []
    selected_target_labels = []
    with torch.no_grad():
        cell_x, cell_y = torch.meshgrid(
            torch.arange(width, device=source_logits.device),
            torch.arange(height, device=source_logits.device),
            indexing="xy",
        )
        cell_origins = torch.stack([cell_x, cell_y], dim=-1).long() * 8

    for batch_index in range(int(batch)):
        point_mask = batches == int(batch_index)
        batch_points1 = points1[point_mask]
        batch_points2 = points2[point_mask]
        with torch.no_grad():
            hashmap = torch.full((height * 8, width * 8, 2), -1, dtype=torch.long, device=source_logits.device)
            if batch_points1.numel() > 0:
                src_xy = batch_points1.long()
                tgt_xy = batch_points2.long()
                in_source = (
                    (src_xy[:, 0] >= 0)
                    & (src_xy[:, 0] < width * 8)
                    & (src_xy[:, 1] >= 0)
                    & (src_xy[:, 1] < height * 8)
                )
                if torch.any(in_source):
                    hashmap[src_xy[in_source, 1], src_xy[in_source, 0], :] = tgt_xy[in_source]

            predicted = full_source_labels[batch_index]
            source_mask = predicted < NON_KEYPOINT_LABEL
            if not torch.any(source_mask):
                continue
            source_labels = predicted[source_mask].long()
            offset_xy = torch.stack([source_labels % 8, source_labels // 8], dim=-1)
            source_coords = cell_origins[source_mask] + offset_xy
            gt_target = hashmap[source_coords[:, 1], source_coords[:, 0]]
            valid = torch.all(gt_target >= 0, dim=-1)
            valid &= (gt_target[:, 0] >= 0) & (gt_target[:, 0] < width * 8)
            valid &= (gt_target[:, 1] >= 0) & (gt_target[:, 1] < height * 8)
            if not torch.any(valid):
                continue
            gt_target = gt_target[valid]
            target_labels = (gt_target[:, 0] % 8) + 8 * (gt_target[:, 1] % 8)
            target_rows = gt_target[:, 1] // 8
            target_cols = gt_target[:, 0] // 8
            source_rows, source_cols = torch.nonzero(source_mask, as_tuple=True)
            source_rows = source_rows[valid]
            source_cols = source_cols[valid]

        source_selected = source_spatial[batch_index, source_rows, source_cols]
        target_selected = target_spatial[batch_index, target_rows, target_cols]
        with torch.no_grad():
            source_keep_labels = torch.argmax(source_selected, dim=-1)
        selected_source_logits.append(source_selected)
        selected_source_labels.append(source_keep_labels)
        selected_target_logits.append(target_selected)
        selected_target_labels.append(target_labels)

    if not selected_target_logits:
        zero = source_logits.sum() * 0.0 + target_logits.sum() * 0.0
        return zero, 0.0, {"valid_count": 0, "source_candidate_count": source_candidate_count}

    source_selected_logits = torch.cat(selected_source_logits, dim=0)
    source_labels = torch.cat(selected_source_labels, dim=0)
    target_selected_logits = torch.cat(selected_target_logits, dim=0)
    target_labels = torch.cat(selected_target_labels, dim=0)
    source_loss = F.nll_loss(F.log_softmax(source_selected_logits, dim=-1), source_labels, reduction="mean")
    target_loss = F.nll_loss(F.log_softmax(target_selected_logits, dim=-1), target_labels, reduction="mean")
    with torch.no_grad():
        predicted_target = torch.argmax(target_selected_logits, dim=-1)
        correct = torch.count_nonzero(predicted_target == target_labels)
        valid_count = int(target_labels.numel())
        acc = float((correct.float() / max(valid_count, 1)).detach().cpu().item())
    return source_loss + target_loss, acc, {"valid_count": valid_count, "source_candidate_count": source_candidate_count}


def decode_keypoints_from_logits(
    logits: torch.Tensor | np.ndarray,
    *,
    image_width: int,
    image_height: int,
    top_k: int = 2048,
    threshold: float = 0.0,
    offset_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image of 65-bin logits into keypoints, scores, and offset labels."""

    tensor = torch.as_tensor(logits, dtype=torch.float32)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("batched decode expects batch size 1")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 65:
        raise ValueError("logits must have shape (65, H, W) or (1, 65, H, W)")
    probabilities = torch.softmax(tensor, dim=0)
    confidence = 1.0 - probabilities[NON_KEYPOINT_LABEL]
    labels = torch.argmax(tensor, dim=0)
    mask = (labels < NON_KEYPOINT_LABEL) & (confidence >= float(threshold))
    rows, cols = torch.nonzero(mask, as_tuple=True)
    if rows.numel() == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    scores = confidence[rows, cols]
    order = torch.argsort(scores, descending=True)
    if int(top_k) > 0:
        order = order[: int(top_k)]
    rows = rows[order]
    cols = cols[order]
    scores = scores[order]
    selected_labels = labels[rows, cols].long()
    bins = int(offset_bins)
    x_bin = selected_labels % bins
    y_bin = selected_labels // bins
    cell_w = float(image_width) / float(tensor.shape[2])
    cell_h = float(image_height) / float(tensor.shape[1])
    xy = torch.stack(
        [
            (cols.float() + (x_bin.float() + 0.5) / float(bins)) * cell_w,
            (rows.float() + (y_bin.float() + 0.5) / float(bins)) * cell_h,
        ],
        dim=1,
    )
    return (
        xy.detach().cpu().numpy().astype(np.float32, copy=False),
        scores.detach().cpu().numpy().astype(np.float32, copy=False),
        selected_labels.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def keypoint_xy_to_feature_cell_indices(
    keypoints_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    feature_grid_width: int,
    feature_grid_height: int,
) -> np.ndarray:
    """Map image-space keypoints to flattened descriptor feature-grid cells."""

    xy = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image_width and image_height must be positive")
    if int(feature_grid_width) <= 0 or int(feature_grid_height) <= 0:
        raise ValueError("feature grid size must be positive")
    valid = (
        np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < float(image_width))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < float(image_height))
    )
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)
    compact = xy[valid]
    cols = np.floor(compact[:, 0] / float(image_width) * int(feature_grid_width)).astype(np.int64)
    rows = np.floor(compact[:, 1] / float(image_height) * int(feature_grid_height)).astype(np.int64)
    cols = np.clip(cols, 0, int(feature_grid_width) - 1)
    rows = np.clip(rows, 0, int(feature_grid_height) - 1)
    indices = rows * int(feature_grid_width) + cols
    return np.unique(indices.astype(np.int64, copy=False))
