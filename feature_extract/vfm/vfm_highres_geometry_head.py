"""High-resolution depth and normal decoders on frozen RADIO/VFM tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GeometryHeadOutput:
    depth: torch.Tensor
    normal: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class GeometrySupportPrediction:
    """Metric geometry read at identity-free VFM support locations."""

    depth: np.ndarray
    normal: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class GeometryMapPrediction:
    """Dense query geometry in the feature-map coordinate system."""

    depth: np.ndarray
    normal: np.ndarray
    confidence: np.ndarray


class RadioHighResGeometryHead(nn.Module):
    """Decode low-resolution RADIO tokens into high-resolution geometry maps."""

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_channels: int = 128,
        min_depth: float = 1e-3,
        architecture: str = "shared",
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if float(min_depth) <= 0.0:
            raise ValueError("min_depth must be positive")
        if str(architecture) not in {"shared", "separate_decoders"}:
            raise ValueError("architecture must be 'shared' or 'separate_decoders'")
        self.min_depth = float(min_depth)
        self.architecture = str(architecture)
        mid = max(int(hidden_channels) // 2, 32)
        self.mid_channels = int(mid)
        self.stem = nn.Sequential(
            nn.GroupNorm(1, int(in_channels)),
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine = self._make_decoder(int(hidden_channels), int(mid))
        if self.architecture == "separate_decoders":
            self.depth_refine = self._make_decoder(int(hidden_channels), int(mid))
            self.normal_refine = self._make_decoder(int(hidden_channels), int(mid))
        else:
            self.depth_refine = self.refine
            self.normal_refine = self.refine
        self.depth_head = nn.Conv2d(mid, 1, kernel_size=3, padding=1)
        self.normal_head = nn.Conv2d(mid, 3, kernel_size=3, padding=1)
        self.confidence_head = nn.Conv2d(mid, 1, kernel_size=3, padding=1)

    @staticmethod
    def _make_decoder(hidden_channels: int, mid_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(mid_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor, output_size: tuple[int, int]) -> GeometryHeadOutput:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape (B, C, H, W)")
        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size must be positive")
        x = self.stem(tokens)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        depth_features = self.depth_refine(x)
        normal_features = self.normal_refine(x)
        log_depth = self.depth_head(depth_features).squeeze(1)
        depth = torch.exp(torch.clamp(log_depth, min=float(np.log(self.min_depth)), max=float(np.log(1e4))))
        normal = F.normalize(self.normal_head(normal_features), dim=1, eps=1e-6)
        confidence = torch.sigmoid(self.confidence_head(depth_features)).squeeze(1)
        return GeometryHeadOutput(depth=depth, normal=normal, confidence=confidence)


def load_radio_highres_geometry_head(
    path: Path,
    device: str = "cpu",
) -> tuple[RadioHighResGeometryHead, dict[str, object]]:
    payload = torch.load(Path(path), map_location=str(device), weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("not a RADIO high-resolution geometry-head checkpoint")
    model = RadioHighResGeometryHead(
        in_channels=int(payload["in_channels"]),
        hidden_channels=int(payload["hidden_channels"]),
        architecture=str(payload.get("architecture", "shared")),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(str(device)).eval()
    return model, payload


def predict_radio_geometry_at_normalized_xy(
    model: RadioHighResGeometryHead,
    radio_final: np.ndarray,
    xy_normalized: np.ndarray,
    *,
    output_size: tuple[int, int] | None = None,
) -> GeometrySupportPrediction:
    """Decode once, then bilinearly read geometry at normalized image points."""

    raw = np.asarray(radio_final, dtype=np.float32)
    xy = np.asarray(xy_normalized, dtype=np.float32).reshape(-1, 2)
    if raw.ndim != 3:
        raise ValueError("radio_final must have shape (C, H, W)")
    if np.any(~np.isfinite(xy)) or np.any((xy < 0.0) | (xy > 1.0)):
        raise ValueError("xy_normalized must be finite and lie in [0, 1]")
    if output_size is None:
        output_size = (2 * int(raw.shape[1]), 2 * int(raw.shape[2]))
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(
            torch.as_tensor(raw[None], dtype=torch.float32, device=device),
            output_size=(int(output_size[0]), int(output_size[1])),
        )
        if xy.shape[0] == 0:
            return GeometrySupportPrediction(
                depth=np.zeros((0,), dtype=np.float32),
                normal=np.zeros((0, 3), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
            )
        # grid_sample with align_corners=False uses image-edge normalized
        # coordinates, exactly matching the support coordinates in [0, 1].
        grid = torch.as_tensor(2.0 * xy - 1.0, dtype=torch.float32, device=device).reshape(1, -1, 1, 2)
        depth = F.grid_sample(
            output.depth[:, None], grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        ).reshape(-1)
        normal = F.grid_sample(
            output.normal, grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        ).reshape(3, -1).T
        normal = F.normalize(normal, dim=1, eps=1e-6)
        confidence = F.grid_sample(
            output.confidence[:, None], grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        ).reshape(-1)
    return GeometrySupportPrediction(
        depth=depth.cpu().numpy().astype(np.float32, copy=False),
        normal=normal.cpu().numpy().astype(np.float32, copy=False),
        confidence=confidence.cpu().numpy().astype(np.float32, copy=False),
    )


def predict_radio_geometry_map(
    model: RadioHighResGeometryHead,
    radio_final: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> GeometryMapPrediction:
    """Decode a dense depth/normal/confidence map once for pose likelihood."""

    raw = np.asarray(radio_final, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError("radio_final must have shape (C, H, W)")
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(
            torch.as_tensor(raw[None], dtype=torch.float32, device=device),
            output_size=(int(output_size[0]), int(output_size[1])),
        )
    return GeometryMapPrediction(
        depth=output.depth[0].cpu().numpy().astype(np.float32, copy=False),
        normal=output.normal[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False),
        confidence=output.confidence[0].cpu().numpy().astype(np.float32, copy=False),
    )


def masked_log_depth_l1(pred_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    if not bool(torch.any(valid)):
        return pred_depth.sum() * 0.0
    pred = torch.clamp(pred_depth[valid], min=1e-6)
    target = torch.clamp(target_depth[valid], min=1e-6)
    return torch.mean(torch.abs(torch.log(pred) - torch.log(target)))


def masked_scale_invariant_log_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize within-image shape after marginalizing global depth scale."""

    valid = valid_mask.bool()
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    losses = []
    for batch in range(int(pred_depth.shape[0])):
        mask = valid[batch]
        if not bool(torch.any(mask)):
            continue
        residual = (
            torch.log(torch.clamp(pred_depth[batch][mask], min=1e-6))
            - torch.log(torch.clamp(target_depth[batch][mask], min=1e-6))
        )
        residual = residual - torch.mean(residual)
        losses.append(torch.mean(torch.abs(residual)))
    if not losses:
        return pred_depth.sum() * 0.0
    return torch.mean(torch.stack(losses))


def masked_log_depth_gradient_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Match local log-depth differences that determine surface shape."""

    valid = valid_mask.bool()
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    pred_log = torch.log(torch.clamp(pred_depth, min=1e-6))
    target_log = torch.log(torch.clamp(target_depth, min=1e-6))
    residual = pred_log - target_log
    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    values = []
    if bool(torch.any(horizontal_valid)):
        values.append(torch.mean(torch.abs(
            (residual[:, :, 1:] - residual[:, :, :-1])[horizontal_valid]
        )))
    if bool(torch.any(vertical_valid)):
        values.append(torch.mean(torch.abs(
            (residual[:, 1:, :] - residual[:, :-1, :])[vertical_valid]
        )))
    if not values:
        return pred_depth.sum() * 0.0
    return torch.mean(torch.stack(values))


def masked_normal_cosine_loss(pred_normal: torch.Tensor, target_normal: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if pred_normal.shape != target_normal.shape:
        raise ValueError("pred_normal and target_normal must have matching shapes")
    if pred_normal.ndim != 4 or pred_normal.shape[1] != 3:
        raise ValueError("normal tensors must have shape (B, 3, H, W)")
    if valid.shape != pred_normal.shape[0:1] + pred_normal.shape[2:4]:
        raise ValueError("valid_mask must have shape (B, H, W)")
    if not bool(torch.any(valid)):
        return pred_normal.sum() * 0.0
    pred = F.normalize(pred_normal.permute(0, 2, 3, 1)[valid], dim=-1, eps=1e-6)
    target = F.normalize(target_normal.permute(0, 2, 3, 1)[valid], dim=-1, eps=1e-6)
    return torch.mean(1.0 - torch.sum(pred * target, dim=-1))


def geometry_confidence_loss(
    pred_confidence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise the advertised geometry confidence as a valid-surface probability."""

    if pred_confidence.shape != valid_mask.shape:
        raise ValueError("pred_confidence and valid_mask must have matching shapes")
    # Probability-space BCE is intentionally retained because the public head
    # returns calibrated probabilities.  PyTorch forbids this operator inside
    # autocast, so evaluate only this small term in FP32.
    with torch.cuda.amp.autocast(enabled=False):
        probability = torch.clamp(
            pred_confidence.float(), min=1e-6, max=1.0 - 1e-6,
        )
        return F.binary_cross_entropy(
            probability, valid_mask.to(dtype=torch.float32),
        )


def masked_geometry_loss(
    pred_depth: torch.Tensor,
    pred_normal: torch.Tensor,
    target_depth: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    normal_weight: float = 0.25,
    pred_confidence: torch.Tensor | None = None,
    confidence_weight: float = 0.0,
    absolute_depth_weight: float = 1.0,
    scale_invariant_depth_weight: float = 0.0,
    depth_gradient_weight: float = 0.0,
) -> torch.Tensor:
    loss = float(absolute_depth_weight) * masked_log_depth_l1(
        pred_depth, target_depth, valid_mask,
    ) + float(normal_weight) * masked_normal_cosine_loss(
        pred_normal,
        target_normal,
        valid_mask,
    )
    if float(scale_invariant_depth_weight) > 0.0:
        loss = loss + float(scale_invariant_depth_weight) * masked_scale_invariant_log_depth_loss(
            pred_depth, target_depth, valid_mask,
        )
    if float(depth_gradient_weight) > 0.0:
        loss = loss + float(depth_gradient_weight) * masked_log_depth_gradient_loss(
            pred_depth, target_depth, valid_mask,
        )
    if float(confidence_weight) > 0.0:
        if pred_confidence is None:
            raise ValueError("confidence supervision requires pred_confidence")
        loss = loss + float(confidence_weight) * geometry_confidence_loss(
            pred_confidence, valid_mask,
        )
    return loss
