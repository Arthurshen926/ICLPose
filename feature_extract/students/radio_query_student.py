import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = ConvNormAct(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.pw = ConvNormAct(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        return x + self.pw(self.dw(x))


def _softplus_inverse(x: float) -> float:
    if x <= 0:
        raise ValueError(f"softplus inverse expects positive input, got {x}")
    return math.log(math.expm1(x))


class RadioQueryStudent(nn.Module):
    """Minimal dual-head RGB encoder for RADIO-style query features."""

    def __init__(
        self,
        in_channels=3,
        feature_dim=64,
        base_channels=32,
        stage_dims=(32, 64, 96, 128),
        output_hw=(68, 120),
        input_hw=(1088, 1920),
        dropout=0.0,
        l2_normalize=True,
        predict_magnitude=False,
        fine_init_norm=1.0,
        coarse_init_norm=1.0,
        magnitude_min=1e-4,
        retrieval_dim=None,
        retrieval_hidden_dim=None,
        retrieval_dropout=0.0,
        retrieval_l2_normalize=True,
    ):
        super().__init__()
        stage_dims = tuple(stage_dims)
        if len(stage_dims) != 4:
            raise ValueError("stage_dims must have four entries")

        self.output_hw = tuple(output_hw) if output_hw is not None else None
        self.input_hw = tuple(input_hw) if input_hw is not None else None
        self.feature_dim = feature_dim
        self.l2_normalize = l2_normalize
        self.predict_magnitude = bool(predict_magnitude)
        self.fine_init_norm = float(fine_init_norm)
        self.coarse_init_norm = float(coarse_init_norm)
        self.magnitude_min = float(magnitude_min)
        self.retrieval_dim = int(retrieval_dim) if retrieval_dim else 0
        self.retrieval_l2_normalize = retrieval_l2_normalize

        stem_dim = stage_dims[0] or base_channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_dim, kernel_size=5, stride=2),
            ResidualDepthwiseBlock(stem_dim),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(stem_dim, stage_dims[1], stride=2),
            ResidualDepthwiseBlock(stage_dims[1]),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stage_dims[1], stage_dims[2], stride=2),
            ResidualDepthwiseBlock(stage_dims[2]),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(stage_dims[2], stage_dims[3], stride=2),
            ResidualDepthwiseBlock(stage_dims[3]),
            ResidualDepthwiseBlock(stage_dims[3]),
        )

        self.fine_fuse = nn.Sequential(
            ConvNormAct(stage_dims[2] + stage_dims[3], stage_dims[3], kernel_size=1),
            ResidualDepthwiseBlock(stage_dims[3]),
        )
        self.coarse_refine = nn.Sequential(
            ResidualDepthwiseBlock(stage_dims[3]),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.fine_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], feature_dim, kernel_size=1),
        )
        self.coarse_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], feature_dim, kernel_size=1),
        )
        if self.predict_magnitude:
            self.fine_norm_head = nn.Conv2d(stage_dims[3], 1, kernel_size=1)
            self.coarse_norm_head = nn.Conv2d(stage_dims[3], 1, kernel_size=1)
        else:
            self.fine_norm_head = None
            self.coarse_norm_head = None

        if self.retrieval_dim > 0:
            retrieval_hidden_dim = int(retrieval_hidden_dim or stage_dims[3])
            self.retrieval_head = nn.Sequential(
                nn.Linear(stage_dims[3], retrieval_hidden_dim),
                nn.GELU(),
                nn.Dropout(retrieval_dropout) if retrieval_dropout > 0 else nn.Identity(),
                nn.Linear(retrieval_hidden_dim, self.retrieval_dim),
            )
        else:
            self.retrieval_head = None

        self._init_weights()
        if self.predict_magnitude:
            self._init_magnitude_head(self.fine_norm_head, self.fine_init_norm)
            self._init_magnitude_head(self.coarse_norm_head, self.coarse_init_norm)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_magnitude_head(self, head, init_norm):
        nn.init.zeros_(head.weight)
        init_mag = max(float(init_norm) - self.magnitude_min, 1e-4)
        nn.init.constant_(head.bias, _softplus_inverse(init_mag))

    def forward(self, x):
        if self.input_hw is not None and tuple(x.shape[-2:]) != self.input_hw:
            x = F.interpolate(x, self.input_hw, mode="bilinear", align_corners=False)

        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        s3_to_s4 = F.avg_pool2d(s3, kernel_size=2, stride=2)
        fine_latent = self.fine_fuse(torch.cat([s4, s3_to_s4], dim=1))
        coarse_latent = self.coarse_refine(s4)

        fine = self.fine_head(fine_latent)
        coarse = self.coarse_head(coarse_latent)
        fine_mag = None
        coarse_mag = None
        if self.predict_magnitude:
            fine_mag = F.softplus(self.fine_norm_head(fine_latent)) + self.magnitude_min
            coarse_mag = F.softplus(self.coarse_norm_head(coarse_latent)) + self.magnitude_min

        if self.output_hw is not None:
            fine = F.interpolate(fine, self.output_hw, mode="bilinear", align_corners=False)
            coarse = F.interpolate(coarse, self.output_hw, mode="bilinear", align_corners=False)
            if fine_mag is not None:
                fine_mag = F.interpolate(fine_mag, self.output_hw, mode="bilinear", align_corners=False)
                coarse_mag = F.interpolate(coarse_mag, self.output_hw, mode="bilinear", align_corners=False)

        if self.predict_magnitude:
            fine = F.normalize(fine, dim=1) * fine_mag
            coarse = F.normalize(coarse, dim=1) * coarse_mag
        elif self.l2_normalize:
            fine = F.normalize(fine, dim=1)
            coarse = F.normalize(coarse, dim=1)

        outputs = {
            "fine": fine,
            "coarse": coarse,
            "backbone_features": {
                "stage2": s2,
                "stage3": s3,
                "stage4": s4,
            },
        }
        if self.predict_magnitude:
            outputs["magnitude"] = {
                "fine": fine_mag,
                "coarse": coarse_mag,
            }

        if self.retrieval_head is not None:
            retrieval = self.retrieval_head(coarse_latent.mean(dim=(-1, -2)))
            if self.retrieval_l2_normalize:
                retrieval = F.normalize(retrieval, dim=1)
            outputs["retrieval"] = retrieval

        return outputs
