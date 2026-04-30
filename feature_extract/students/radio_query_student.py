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
        fine_feature_dim=None,
        coarse_feature_dim=None,
        base_channels=32,
        stage_dims=(32, 64, 96, 128),
        output_hw=(68, 120),
        coarse_output_hw=None,
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
        fine_low_level_skip=False,
        fine_low_level_init=0.0,
        fine_highres_skip=False,
        fine_highres_source="stage2",
        fine_highres_init=0.0,
        fine_highres_zero_init=False,
        fine_loc_head=False,
        fine_loc_init=1.0,
        fine_loc_zero_init=True,
        fine_loc_detach_base=False,
        fine_loc_highres_source=None,
        fine_loc_highres_init=1.0,
        fine_loc_highres_zero_init=True,
        fine_loc_highres_detach=True,
        teacher_fine_condition=False,
        teacher_fine_init=1.0,
        teacher_fine_zero_init=True,
        teacher_fine_detach=True,
        scene_coord_head=False,
        scene_coord_zero_init=True,
        scene_coord_detach_base=False,
        scene_coord_use_pixel_grid=False,
        scene_coord_global_context=False,
    ):
        super().__init__()
        stage_dims = tuple(stage_dims)
        if len(stage_dims) != 4:
            raise ValueError("stage_dims must have four entries")

        self.output_hw = tuple(output_hw) if output_hw is not None else None
        self.coarse_output_hw = tuple(coarse_output_hw) if coarse_output_hw is not None else self.output_hw
        self.input_hw = tuple(input_hw) if input_hw is not None else None
        self.feature_dim = int(feature_dim)
        self.fine_feature_dim = int(fine_feature_dim) if fine_feature_dim is not None else self.feature_dim
        self.coarse_feature_dim = int(coarse_feature_dim) if coarse_feature_dim is not None else self.feature_dim
        self.l2_normalize = l2_normalize
        self.predict_magnitude = bool(predict_magnitude)
        self.fine_init_norm = float(fine_init_norm)
        self.coarse_init_norm = float(coarse_init_norm)
        self.magnitude_min = float(magnitude_min)
        self.retrieval_dim = int(retrieval_dim) if retrieval_dim else 0
        self.retrieval_l2_normalize = retrieval_l2_normalize
        self.fine_low_level_skip = bool(fine_low_level_skip)
        self.fine_low_level_init = float(fine_low_level_init)
        self.fine_highres_skip = bool(fine_highres_skip)
        self.fine_highres_source = str(fine_highres_source).lower()
        self.fine_highres_init = float(fine_highres_init)
        self.fine_highres_zero_init = bool(fine_highres_zero_init)
        self.fine_loc_highres_source = (
            str(fine_loc_highres_source).lower()
            if fine_loc_highres_source is not None
            else None
        )
        self.use_fine_loc_head = bool(fine_loc_head or self.fine_loc_highres_source is not None)
        self.fine_loc_init = float(fine_loc_init)
        self.fine_loc_zero_init = bool(fine_loc_zero_init)
        self.fine_loc_detach_base = bool(fine_loc_detach_base)
        self.fine_loc_highres_init = float(fine_loc_highres_init)
        self.fine_loc_highres_zero_init = bool(fine_loc_highres_zero_init)
        self.fine_loc_highres_detach = bool(fine_loc_highres_detach)
        self.teacher_fine_condition = bool(teacher_fine_condition)
        self.teacher_fine_init = float(teacher_fine_init)
        self.teacher_fine_zero_init = bool(teacher_fine_zero_init)
        self.teacher_fine_detach = bool(teacher_fine_detach)
        self.use_scene_coord_head = bool(scene_coord_head)
        self.scene_coord_zero_init = bool(scene_coord_zero_init)
        self.scene_coord_detach_base = bool(scene_coord_detach_base)
        self.scene_coord_use_pixel_grid = bool(scene_coord_use_pixel_grid)
        self.scene_coord_global_context = bool(scene_coord_global_context)

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
        self.fine_low_fuse = (
            nn.Sequential(
                ConvNormAct(stage_dims[1], stage_dims[3], kernel_size=1),
                ResidualDepthwiseBlock(stage_dims[3]),
            )
            if self.fine_low_level_skip
            else None
        )
        self.fine_low_scale = (
            nn.Parameter(torch.tensor(self.fine_low_level_init, dtype=torch.float32))
            if self.fine_low_level_skip
            else None
        )
        highres_channels = {
            "stage1": stage_dims[0],
            "stem": stage_dims[0],
            "stage2": stage_dims[1],
            "stage3": stage_dims[2],
        }
        if self.fine_highres_skip and self.fine_highres_source not in highres_channels:
            raise ValueError(
                f"fine_highres_source must be one of {sorted(highres_channels)}, "
                f"got {self.fine_highres_source!r}"
            )
        if self.fine_loc_highres_source is not None and self.fine_loc_highres_source not in highres_channels:
            raise ValueError(
                f"fine_loc_highres_source must be one of {sorted(highres_channels)}, "
                f"got {self.fine_loc_highres_source!r}"
            )
        if self.fine_highres_skip:
            highres_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            self.fine_highres_fuse = nn.Sequential(
                ConvNormAct(highres_channels[self.fine_highres_source], highres_hidden, kernel_size=3),
                ResidualDepthwiseBlock(highres_hidden),
                nn.Conv2d(highres_hidden, self.fine_feature_dim, kernel_size=1),
            )
            self.fine_highres_scale = nn.Parameter(
                torch.tensor(self.fine_highres_init, dtype=torch.float32)
            )
        else:
            self.fine_highres_fuse = None
            self.fine_highres_scale = None
        if self.teacher_fine_condition:
            teacher_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            self.teacher_fine_fuse = nn.Sequential(
                ConvNormAct(self.fine_feature_dim, teacher_hidden, kernel_size=3),
                ResidualDepthwiseBlock(teacher_hidden),
                nn.Conv2d(teacher_hidden, self.fine_feature_dim, kernel_size=1),
            )
            self.teacher_fine_scale = nn.Parameter(
                torch.tensor(self.teacher_fine_init, dtype=torch.float32)
            )
        else:
            self.teacher_fine_fuse = None
            self.teacher_fine_scale = None
        if self.use_scene_coord_head:
            scene_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            scene_in_channels = self.fine_feature_dim
            if self.scene_coord_use_pixel_grid:
                scene_in_channels += 2
            if self.scene_coord_global_context:
                scene_in_channels += scene_hidden
                self.scene_context_proj = nn.Conv2d(stage_dims[3], scene_hidden, kernel_size=1)
            else:
                self.scene_context_proj = None
            self.scene_coord_head = nn.Sequential(
                ConvNormAct(scene_in_channels, scene_hidden, kernel_size=3),
                ResidualDepthwiseBlock(scene_hidden),
                nn.Conv2d(scene_hidden, 3, kernel_size=1),
            )
        else:
            self.scene_context_proj = None
            self.scene_coord_head = None
        self.coarse_refine = nn.Sequential(
            ResidualDepthwiseBlock(stage_dims[3]),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.fine_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], self.fine_feature_dim, kernel_size=1),
        )
        self.coarse_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], self.coarse_feature_dim, kernel_size=1),
        )
        if self.use_fine_loc_head:
            loc_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            if fine_loc_head:
                self.fine_loc_head = nn.Sequential(
                    ConvNormAct(self.fine_feature_dim, loc_hidden, kernel_size=3),
                    ResidualDepthwiseBlock(loc_hidden),
                    nn.Conv2d(loc_hidden, self.fine_feature_dim, kernel_size=1),
                )
            else:
                self.fine_loc_head = None
            if self.fine_loc_highres_source is not None:
                self.fine_loc_highres_fuse = nn.Sequential(
                    ConvNormAct(highres_channels[self.fine_loc_highres_source], loc_hidden, kernel_size=3),
                    ResidualDepthwiseBlock(loc_hidden),
                    nn.Conv2d(loc_hidden, self.fine_feature_dim, kernel_size=1),
                )
                self.fine_loc_highres_scale = nn.Parameter(
                    torch.tensor(self.fine_loc_highres_init, dtype=torch.float32)
                )
            else:
                self.fine_loc_highres_fuse = None
                self.fine_loc_highres_scale = None
            self.fine_loc_scale = nn.Parameter(
                torch.tensor(self.fine_loc_init, dtype=torch.float32)
            )
        else:
            self.fine_loc_head = None
            self.fine_loc_scale = None
            self.fine_loc_highres_fuse = None
            self.fine_loc_highres_scale = None
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
        if self.fine_highres_zero_init and self.fine_highres_fuse is not None:
            nn.init.zeros_(self.fine_highres_fuse[-1].weight)
            if self.fine_highres_fuse[-1].bias is not None:
                nn.init.zeros_(self.fine_highres_fuse[-1].bias)
        if self.fine_loc_zero_init and self.fine_loc_head is not None:
            nn.init.zeros_(self.fine_loc_head[-1].weight)
            if self.fine_loc_head[-1].bias is not None:
                nn.init.zeros_(self.fine_loc_head[-1].bias)
        if self.fine_loc_highres_zero_init and self.fine_loc_highres_fuse is not None:
            nn.init.zeros_(self.fine_loc_highres_fuse[-1].weight)
            if self.fine_loc_highres_fuse[-1].bias is not None:
                nn.init.zeros_(self.fine_loc_highres_fuse[-1].bias)
        if self.teacher_fine_zero_init and self.teacher_fine_fuse is not None:
            nn.init.zeros_(self.teacher_fine_fuse[-1].weight)
            if self.teacher_fine_fuse[-1].bias is not None:
                nn.init.zeros_(self.teacher_fine_fuse[-1].bias)
        if self.scene_coord_zero_init and self.scene_coord_head is not None:
            nn.init.zeros_(self.scene_coord_head[-1].weight)
            if self.scene_coord_head[-1].bias is not None:
                nn.init.zeros_(self.scene_coord_head[-1].bias)
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

    def forward(self, x, teacher_fine=None):
        if self.input_hw is not None and tuple(x.shape[-2:]) != self.input_hw:
            x = F.interpolate(x, self.input_hw, mode="bilinear", align_corners=False)

        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        s3_to_s4 = F.avg_pool2d(s3, kernel_size=2, stride=2)
        fine_latent = self.fine_fuse(torch.cat([s4, s3_to_s4], dim=1))
        if self.fine_low_fuse is not None:
            s2_to_s4 = F.adaptive_avg_pool2d(s2, output_size=s4.shape[-2:])
            fine_latent = fine_latent + self.fine_low_scale * self.fine_low_fuse(s2_to_s4)
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
            if fine_mag is not None:
                fine_mag = F.interpolate(fine_mag, self.output_hw, mode="bilinear", align_corners=False)
        if self.fine_highres_fuse is not None:
            highres_sources = {
                "stage1": s1,
                "stem": s1,
                "stage2": s2,
                "stage3": s3,
            }
            fine_skip = self.fine_highres_fuse(highres_sources[self.fine_highres_source])
            if fine_skip.shape[-2:] != fine.shape[-2:]:
                fine_skip = F.interpolate(fine_skip, fine.shape[-2:], mode="bilinear", align_corners=False)
            fine = fine + self.fine_highres_scale * fine_skip
        if self.teacher_fine_fuse is not None and teacher_fine is not None:
            teacher = teacher_fine.float()
            if self.teacher_fine_detach:
                teacher = teacher.detach()
            if teacher.shape[-2:] != fine.shape[-2:]:
                teacher = F.interpolate(teacher, fine.shape[-2:], mode="bilinear", align_corners=False)
            fine = fine + self.teacher_fine_scale * self.teacher_fine_fuse(teacher)
        scene_coord = None
        if self.scene_coord_head is not None:
            scene_input = fine.detach() if self.scene_coord_detach_base else fine
            scene_inputs = [scene_input]
            if self.scene_coord_use_pixel_grid:
                B, _C, H, W = scene_input.shape
                yy, xx = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, H, device=scene_input.device, dtype=scene_input.dtype),
                    torch.linspace(-1.0, 1.0, W, device=scene_input.device, dtype=scene_input.dtype),
                    indexing="ij",
                )
                grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
                scene_inputs.append(grid)
            if self.scene_context_proj is not None:
                context = F.adaptive_avg_pool2d(s4, output_size=1)
                context = self.scene_context_proj(context)
                context = context.expand(-1, -1, scene_input.shape[-2], scene_input.shape[-1])
                scene_inputs.append(context)
            if len(scene_inputs) > 1:
                scene_input = torch.cat(scene_inputs, dim=1)
            scene_coord = self.scene_coord_head(scene_input)
        if self.coarse_output_hw is not None:
            coarse = F.interpolate(coarse, self.coarse_output_hw, mode="bilinear", align_corners=False)
            if coarse_mag is not None:
                coarse_mag = F.interpolate(coarse_mag, self.coarse_output_hw, mode="bilinear", align_corners=False)

        if self.predict_magnitude:
            fine = F.normalize(fine, dim=1) * fine_mag
            coarse = F.normalize(coarse, dim=1) * coarse_mag
        elif self.l2_normalize:
            fine = F.normalize(fine, dim=1)
            coarse = F.normalize(coarse, dim=1)

        fine_loc = None
        if self.use_fine_loc_head:
            fine_loc_base = fine.detach() if self.fine_loc_detach_base else fine
            loc_delta = torch.zeros_like(fine_loc_base)
            if self.fine_loc_head is not None:
                loc_delta = loc_delta + self.fine_loc_scale * self.fine_loc_head(fine_loc_base)
            if self.fine_loc_highres_fuse is not None:
                highres_sources = {
                    "stage1": s1,
                    "stem": s1,
                    "stage2": s2,
                    "stage3": s3,
                }
                loc_src = highres_sources[self.fine_loc_highres_source]
                if self.fine_loc_highres_detach:
                    loc_src = loc_src.detach()
                loc_skip = self.fine_loc_highres_fuse(loc_src)
                if loc_skip.shape[-2:] != fine_loc_base.shape[-2:]:
                    loc_skip = F.interpolate(
                        loc_skip,
                        fine_loc_base.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loc_delta = loc_delta + self.fine_loc_highres_scale * loc_skip
            fine_loc = fine_loc_base + loc_delta
            if self.predict_magnitude:
                fine_loc = F.normalize(fine_loc, dim=1) * fine_mag
            elif self.l2_normalize:
                fine_loc = F.normalize(fine_loc, dim=1)

        outputs = {
            "fine": fine,
            "coarse": coarse,
            "backbone_features": {
                "stage2": s2,
                "stage3": s3,
                "stage4": s4,
            },
        }
        if fine_loc is not None:
            outputs["fine_loc"] = fine_loc
        if scene_coord is not None:
            outputs["scene_coord"] = scene_coord
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
