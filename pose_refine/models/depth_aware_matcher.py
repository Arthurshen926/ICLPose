from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_refine.utils.geometry_solver import compute_image_jacobian


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = ConvNormAct(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.pw = ConvNormAct(channels, channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.dw(x))


class DepthAwareLocalMatcher(nn.Module):
    """Residual local-correlation refiner conditioned on rendered geometry."""

    def __init__(
        self,
        radius: int = 4,
        hidden_dim: int = 64,
        zero_init: bool = True,
        residual_scale: float = 1.0,
        context_mode: str = "basic",
    ):
        super().__init__()
        self.radius = int(radius)
        self.corr_channels = (2 * self.radius + 1) ** 2
        self.context_mode = str(context_mode or "basic").lower()
        if self.context_mode not in {"basic", "observability"}:
            raise ValueError(
                "context_mode must be one of {'basic', 'observability'}, "
                f"got {context_mode!r}"
            )
        context_channels = 5 + (4 if self.context_mode == "observability" else 0)
        self.refine = nn.Sequential(
            ConvNormAct(self.corr_channels + context_channels, hidden_dim, kernel_size=3),
            ResidualDepthwiseBlock(hidden_dim),
            nn.Conv2d(hidden_dim, self.corr_channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.refine[-1].weight)
            if self.refine[-1].bias is not None:
                nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _masked_standardize(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            mean = value.mean(dim=(-2, -1), keepdim=True)
            std = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp(min=1e-6)
            return (value - mean) / std
        mask = mask.float()
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / denom
        var = ((value - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denom
        return (value - mean) / torch.sqrt(var + 1e-6)

    @staticmethod
    def _resize_intrinsics(
        intrinsics,
        *,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ):
        if intrinsics is None:
            return None
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        sx = float(dst_w) / max(float(src_w), 1.0)
        sy = float(dst_h) / max(float(src_h), 1.0)
        if isinstance(intrinsics, torch.Tensor):
            scaled = intrinsics.clone()
            scaled[..., 0] = scaled[..., 0] * sx
            scaled[..., 1] = scaled[..., 1] * sy
            scaled[..., 2] = scaled[..., 2] * sx
            scaled[..., 3] = scaled[..., 3] * sy
            return scaled
        return {
            "fx": intrinsics["fx"] * sx,
            "fy": intrinsics["fy"] * sy,
            "cx": intrinsics["cx"] * sx,
            "cy": intrinsics["cy"] * sy,
        }

    def _build_observability_context(
        self,
        depth: torch.Tensor | None,
        intrinsics,
        target_hw: tuple[int, int],
        valid_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if depth is None or intrinsics is None:
            return None
        H, W = target_hw
        depth_f = depth.float()
        if depth_f.ndim == 4:
            depth_f = depth_f.squeeze(1)
        src_hw = depth_f.shape[-2:]
        if src_hw != (H, W):
            depth_f = F.interpolate(
                depth_f.unsqueeze(1),
                (H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        scaled_intrinsics = self._resize_intrinsics(
            intrinsics,
            src_hw=src_hw,
            dst_hw=(H, W),
        )
        Ju, Jv, jac_valid = compute_image_jacobian(depth_f, scaled_intrinsics)
        B = depth_f.shape[0]
        trans_obs = torch.sqrt(
            Ju[..., :3].square().sum(dim=-1) + Jv[..., :3].square().sum(dim=-1) + 1e-6
        ).reshape(B, 1, H, W)
        rot_obs = torch.sqrt(
            Ju[..., 3:].square().sum(dim=-1) + Jv[..., 3:].square().sum(dim=-1) + 1e-6
        ).reshape(B, 1, H, W)
        yaw_obs = torch.sqrt(
            Ju[..., 5].square() + Jv[..., 5].square() + 1e-6
        ).reshape(B, 1, H, W)
        balance = torch.log((rot_obs + 1e-3) / (trans_obs + 1e-3))
        jac_valid = jac_valid.reshape(B, 1, H, W).float()
        if valid_mask is not None:
            vm = valid_mask.float()
            if vm.ndim == 3:
                vm = vm.unsqueeze(1)
            if vm.shape[-2:] != (H, W):
                vm = F.interpolate(vm, (H, W), mode="nearest")
            jac_valid = jac_valid * vm
        return torch.cat(
            [
                self._masked_standardize(trans_obs, jac_valid),
                self._masked_standardize(rot_obs, jac_valid),
                self._masked_standardize(yaw_obs, jac_valid),
                self._masked_standardize(balance, jac_valid),
            ],
            dim=1,
        ).to(dtype=dtype)

    def _build_context(
        self,
        corr: torch.Tensor,
        depth: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        intrinsics=None,
    ) -> torch.Tensor:
        B, _C, H, W = corr.shape
        device = corr.device
        dtype = corr.dtype
        if depth is None:
            depth_ch = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            inv_depth_ch = torch.zeros_like(depth_ch)
            depth_valid = torch.ones_like(depth_ch)
        else:
            depth_ch = depth.float()
            if depth_ch.ndim == 3:
                depth_ch = depth_ch.unsqueeze(1)
            if depth_ch.shape[-2:] != (H, W):
                depth_ch = F.interpolate(depth_ch, (H, W), mode="bilinear", align_corners=False)
            depth_valid = (depth_ch > 0.05).float()
            log_depth = torch.log(depth_ch.clamp(min=0.05))
            inv_depth = 1.0 / depth_ch.clamp(min=0.05)
            depth_ch = self._masked_standardize(log_depth, depth_valid).to(dtype=dtype)
            inv_depth_ch = self._masked_standardize(inv_depth, depth_valid).to(dtype=dtype)

        if valid_mask is not None:
            valid = valid_mask.float()
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            if valid.shape[-2:] != (H, W):
                valid = F.interpolate(valid, (H, W), mode="nearest")
            depth_valid = depth_valid * valid

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        context = [depth_ch, inv_depth_ch, xy, depth_valid.to(dtype=dtype)]
        if self.context_mode == "observability":
            obs_context = self._build_observability_context(
                depth,
                intrinsics,
                (H, W),
                depth_valid,
                dtype,
            )
            if obs_context is None:
                obs_context = torch.zeros(B, 4, H, W, device=device, dtype=dtype)
            context.append(obs_context)
        return torch.cat(context, dim=1)

    def forward(
        self,
        corr: torch.Tensor,
        depth: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        intrinsics=None,
    ) -> torch.Tensor:
        if corr.shape[1] != self.corr_channels:
            raise ValueError(
                f"DepthAwareLocalMatcher expected {self.corr_channels} channels "
                f"for radius={self.radius}, got {corr.shape[1]}"
            )
        context = self._build_context(
            corr,
            depth=depth,
            valid_mask=valid_mask,
            intrinsics=intrinsics,
        )
        residual = self.refine(torch.cat([corr.float(), context.float()], dim=1))
        return corr + self.residual_scale.to(dtype=corr.dtype) * residual.to(dtype=corr.dtype)


class DepthAwareLocalFlowHead(nn.Module):
    """Predict local flow/confidence from correlation plus rendered geometry."""

    def __init__(
        self,
        radius: int = 4,
        hidden_dim: int = 64,
        zero_init: bool = True,
        max_flow: float | None = None,
        base_flow_mode: str = "none",
        base_temperature: float = 0.05,
        context_mode: str = "basic",
    ):
        super().__init__()
        self.radius = int(radius)
        self.corr_channels = (2 * self.radius + 1) ** 2
        self.max_flow = float(max_flow) if max_flow is not None else float(self.radius)
        self.context_mode = str(context_mode or "basic").lower()
        if self.context_mode not in {"basic", "observability"}:
            raise ValueError(
                "context_mode must be one of {'basic', 'observability'}, "
                f"got {context_mode!r}"
            )
        self.base_flow_mode = str(base_flow_mode or "none").lower()
        if self.base_flow_mode not in {"none", "softargmax", "argmax"}:
            raise ValueError(
                "base_flow_mode must be one of {'none', 'softargmax', 'argmax'}, "
                f"got {base_flow_mode!r}"
            )
        self.base_temperature = float(base_temperature)
        context_channels = 8 + (4 if self.context_mode == "observability" else 0)
        self.predict = nn.Sequential(
            ConvNormAct(self.corr_channels + context_channels, hidden_dim, kernel_size=3),
            ResidualDepthwiseBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        if zero_init:
            nn.init.zeros_(self.predict[-1].weight)
            if self.predict[-1].bias is not None:
                nn.init.zeros_(self.predict[-1].bias)

    _masked_standardize = staticmethod(DepthAwareLocalMatcher._masked_standardize)
    _resize_intrinsics = staticmethod(DepthAwareLocalMatcher._resize_intrinsics)
    _build_observability_context = DepthAwareLocalMatcher._build_observability_context

    def _softargmax_flow(self, corr: torch.Tensor, *, temperature: float) -> torch.Tensor:
        window = 2 * self.radius + 1
        yy, xx = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            indexing="ij",
        )
        dx = xx.reshape(1, window * window, 1, 1)
        dy = yy.reshape(1, window * window, 1, 1)
        probs = torch.softmax(corr.float() / max(float(temperature), 1e-6), dim=1).to(dtype=corr.dtype)
        return torch.cat(
            [
                (probs * dx).sum(dim=1, keepdim=True),
                (probs * dy).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def _argmax_flow(self, corr: torch.Tensor) -> torch.Tensor:
        window = 2 * self.radius + 1
        yy, xx = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            indexing="ij",
        )
        dx = xx.reshape(1, window * window, 1, 1)
        dy = yy.reshape(1, window * window, 1, 1)
        idx = corr.argmax(dim=1, keepdim=True)
        dx_map = dx.expand(corr.shape[0], -1, corr.shape[2], corr.shape[3]).gather(1, idx)
        dy_map = dy.expand(corr.shape[0], -1, corr.shape[2], corr.shape[3]).gather(1, idx)
        return torch.cat([dx_map, dy_map], dim=1)

    def _base_flow(self, corr: torch.Tensor) -> torch.Tensor:
        if self.base_flow_mode == "none":
            return corr.new_zeros(corr.shape[0], 2, corr.shape[2], corr.shape[3])
        if self.base_flow_mode == "softargmax":
            return self._softargmax_flow(corr, temperature=self.base_temperature)
        return self._argmax_flow(corr)

    def _build_context(
        self,
        corr: torch.Tensor,
        depth: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        intrinsics=None,
    ) -> torch.Tensor:
        geometry_context = DepthAwareLocalMatcher._build_context(
            self,
            corr,
            depth=depth,
            valid_mask=valid_mask,
            intrinsics=intrinsics,
        )
        prior_flow = self._softargmax_flow(corr, temperature=self.base_temperature)
        top2 = torch.topk(corr.float(), k=min(2, corr.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            peak_margin = torch.zeros_like(top2[:, :1])
        else:
            peak_margin = top2[:, :1] - top2[:, 1:2]
        return torch.cat(
            [
                geometry_context,
                prior_flow.to(dtype=geometry_context.dtype),
                peak_margin.to(dtype=geometry_context.dtype),
            ],
            dim=1,
        )

    def forward(
        self,
        corr: torch.Tensor,
        depth: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        intrinsics=None,
    ) -> dict[str, torch.Tensor]:
        if corr.shape[1] != self.corr_channels:
            raise ValueError(
                f"DepthAwareLocalFlowHead expected {self.corr_channels} channels "
                f"for radius={self.radius}, got {corr.shape[1]}"
            )
        context = self._build_context(
            corr,
            depth=depth,
            valid_mask=valid_mask,
            intrinsics=intrinsics,
        )
        raw = self.predict(torch.cat([corr.float(), context.float()], dim=1))
        base_flow = self._base_flow(corr)
        residual_flow = torch.tanh(raw[:, :2]) * self.max_flow
        flow = (base_flow.float() + residual_flow).clamp(-self.max_flow, self.max_flow)
        confidence_logits = raw[:, 2:3]
        confidence = torch.sigmoid(confidence_logits)
        return {
            "flow": flow.to(dtype=corr.dtype),
            "confidence": confidence.to(dtype=corr.dtype),
            "confidence_logits": confidence_logits.to(dtype=corr.dtype),
        }
