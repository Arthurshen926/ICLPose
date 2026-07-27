"""RADIO-conditioned stride-4 metric decoder with a shallow RGB detail stem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HighresSurfaceMetricDecoderConfig:
    radio_dim: int = 1280
    hidden_dim: int = 96
    rgb_dim: int = 48
    output_dim: int = 64


class _SpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.net(value))


class HighresSurfaceMetricDecoder(nn.Module):
    """Independent metric branch; it never consumes retrieval-mapped features."""

    def __init__(
        self,
        config: HighresSurfaceMetricDecoderConfig = HighresSurfaceMetricDecoderConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        rgb_dim = int(config.rgb_dim)
        output = int(config.output_dim)
        if output < 2 or output % 2:
            raise ValueError("output_dim must be an even value of at least two")
        self.radio_projection = nn.Sequential(
            nn.Conv2d(int(config.radio_dim), hidden, 1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        self.coarse_block = _SpatialBlock(hidden)
        self.middle_block = _SpatialBlock(hidden)
        self.rgb_stem = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, rgb_dim, 1),
            nn.GroupNorm(1, rgb_dim),
            nn.GELU(),
            nn.Conv2d(rgb_dim, rgb_dim, 1),
        )
        self.fine_fusion = nn.Sequential(
            nn.Conv2d(hidden + rgb_dim, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            _SpatialBlock(hidden),
        )
        self.coarse_head = nn.Conv2d(hidden, output, 1)
        self.middle_head = nn.Conv2d(hidden, output, 1)
        self.fine_radio_head = nn.Conv2d(hidden, output // 2, 1)
        self.fine_rgb_head = nn.Conv2d(rgb_dim, output // 2, 1)
        self.matchability_head = nn.Conv2d(hidden, 1, 1)
        self.uncertainty_head = nn.Conv2d(hidden, 1, 1)

    def forward(
        self, radio_final: torch.Tensor, rgb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if radio_final.ndim != 4 or int(radio_final.shape[1]) != int(
            self.config.radio_dim
        ):
            raise ValueError("radio_final must have shape (B, radio_dim, H/16, W/16)")
        if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
            raise ValueError("rgb must have shape (B, 3, H, W)")
        coarse_context = self.coarse_block(self.radio_projection(radio_final.float()))
        middle_context = self.middle_block(
            F.interpolate(
                coarse_context, scale_factor=2.0, mode="bilinear", align_corners=False
            )
        )
        fine_context = F.interpolate(
            middle_context, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        rgb_context = self.rgb_stem(rgb.float())
        if rgb_context.shape[-2:] != fine_context.shape[-2:]:
            rgb_context = F.interpolate(
                rgb_context,
                size=fine_context.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused_context = self.fine_fusion(
            torch.cat([fine_context, rgb_context], dim=1)
        )
        # Preserve the division of labour explicitly.  The RADIO half carries
        # cross-view semantic identity; the shallow RGB half carries the local
        # phase that the coarse VFM grid cannot reconstruct.
        fine = F.normalize(
            torch.cat(
                [
                    F.normalize(self.fine_radio_head(fine_context), dim=1),
                    F.normalize(self.fine_rgb_head(rgb_context), dim=1),
                ],
                dim=1,
            ),
            dim=1,
        )
        return {
            "coarse": F.normalize(self.coarse_head(coarse_context), dim=1),
            "middle": F.normalize(self.middle_head(middle_context), dim=1),
            "fine": fine,
            "matchability": torch.sigmoid(self.matchability_head(fused_context)),
            "uncertainty": F.softplus(self.uncertainty_head(fused_context)),
        }


def save_highres_surface_metric_decoder(
    path: Path,
    model: HighresSurfaceMetricDecoder,
    metadata: dict[str, object],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "highres_surface_metric_decoder",
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_highres_surface_metric_decoder(
    path: Path, *, device: str = "cpu"
) -> tuple[HighresSurfaceMetricDecoder, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("artifact_type") != "highres_surface_metric_decoder":
        raise ValueError("not a high-resolution surface metric decoder")
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "uses_alike_descriptors",
        "uses_radio_intermediate",
        "uses_sfm_points",
        "uses_sfm_tracks",
    ):
        if bool(metadata.get(key, False)):
            raise ValueError(f"metric decoder violates contract: {key}")
    model = HighresSurfaceMetricDecoder(
        HighresSurfaceMetricDecoderConfig(**dict(payload["config"]))
    )
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    torch_device = torch.device(
        device
        if torch.cuda.is_available() or not str(device).startswith("cuda")
        else "cpu"
    )
    return model.to(torch_device).eval(), metadata


def decode_highres_surface_metric(
    model: HighresSurfaceMetricDecoder,
    raw_radio_final: np.ndarray,
    rgb: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    torch_device = torch.device(
        device
        if torch.cuda.is_available() or not str(device).startswith("cuda")
        else "cpu"
    )
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if float(np.max(image, initial=0.0)) > 1.5:
        image = image / 255.0
    with torch.no_grad():
        output = model.to(torch_device).eval()(
            torch.as_tensor(
                np.asarray(raw_radio_final, dtype=np.float32)[None],
                device=torch_device,
            ),
            torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))[None],
                device=torch_device,
            ),
        )
    return {
        key: value[0].detach().cpu().numpy().astype(np.float32, copy=False)
        for key, value in output.items()
    }
