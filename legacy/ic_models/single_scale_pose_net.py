"""
SingleScalePoseNet: 单尺度 Flow-Based 位姿求解网络 (v2: global+local)
===================================================
单一分辨率 DA3 64d 特征，两阶段迭代求解：

核心流程:
  1. Decode query/render features → 64d (ScaleDecoder, 共享 Q/R)
  2. Global correlation at low-res (17×30) → initial coarse flow
  3. Context network → 初始隐状态
  4. N 次 GRU 迭代 (RAFT-style):
     - guided_multiscale_correlation (dilations=1,2,4) → 243ch
     - corr_encoder → ConvGRU → flow_head → Δflow + confidence
  5. Geometry solver: Image Jacobian + 加权最小二乘 → Δξ (6-DOF)

Global correlation 负责覆盖大位移(8°+ noise)，
Dilated local correlation 负责亚像素精细匹配。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve


# ==============================================================================
#  Building Blocks (reuse compatible interfaces from MSFlowPoseNet)
# ==============================================================================

class ScaleDecoder(nn.Module):
    """单尺度 1×1 conv 解码器: in_dim → out_dim, L2 normalized."""

    def __init__(self, in_dim: int, out_dim: int = 64, mid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=1)


class ConvGRU(nn.Module):
    """Convolutional GRU with fused z+r gate."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 64):
        super().__init__()
        self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = zr.chunk(2, dim=1)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# ==============================================================================
#  Correlation Functions
# ==============================================================================

# Grid cache for guided correlation
_GRID_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_base_grid(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached (grid_x, grid_y) each shaped (1, H, W)."""
    key = (H, W, device)
    if key not in _GRID_CACHE:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        _GRID_CACHE[key] = (grid_x.unsqueeze(0), grid_y.unsqueeze(0))
    return _GRID_CACHE[key]


def local_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
) -> torch.Tensor:
    """局部 (2r+1)² correlation using unfold."""
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    fmap2_unfold = fmap2_pad.unfold(2, d, 1).unfold(3, d, 1)
    fmap2_unfold = fmap2_unfold.reshape(B, C, H, W, d * d)
    corr = torch.einsum('bchwn,bchw->bhwn', fmap2_unfold, fmap1)
    return corr.permute(0, 3, 1, 2).contiguous()


def dilated_local_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
    dilation: int = 2,
) -> torch.Tensor:
    """Dilated local correlation: effective radius = radius * dilation."""
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    effective_r = radius * dilation
    fmap2_pad = F.pad(fmap2, [effective_r] * 4, mode='constant', value=0)
    offsets = torch.arange(-radius, radius + 1, device=fmap1.device) * dilation + effective_r
    patches = []
    for dy in offsets:
        for dx in offsets:
            patches.append(fmap2_pad[:, :, dy:dy + H, dx:dx + W])
    fmap2_dilated = torch.stack(patches, dim=-1)
    corr = torch.einsum('bchwn,bchw->bhwn', fmap2_dilated, fmap1)
    return corr.permute(0, 3, 1, 2).contiguous()


def guided_multiscale_correlation(
    fmap_q: torch.Tensor,
    fmap_r: torch.Tensor,
    flow: torch.Tensor,
    radius: int = 4,
    dilations: Tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """
    Warp-guided multi-scale correlation.
    Warp fmap_r using flow, then compute local correlation at multiple dilation levels.
    Returns (B, len(dilations) * (2r+1)², H, W).
    """
    B, C, H, W = fmap_q.shape
    grid_x, grid_y = _get_base_grid(H, W, flow.device)
    grid_x = (grid_x + flow[:, 0]) / (W - 1) * 2.0 - 1.0
    grid_y = (grid_y + flow[:, 1]) / (H - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped_r = F.grid_sample(
        fmap_r, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    corrs = []
    for d in dilations:
        if d == 1:
            corrs.append(local_correlation(fmap_q, warped_r, radius=radius))
        else:
            corrs.append(dilated_local_correlation(fmap_q, warped_r, radius=radius, dilation=d))
    return torch.cat(corrs, dim=1)


def global_correlation(
    fmap_q: torch.Tensor,
    fmap_r: torch.Tensor,
) -> torch.Tensor:
    """
    Global correlation: each query pixel correlates with ALL reference pixels.
    Suitable for low-res feature maps (e.g., 17×30 = 510 pixels).
    
    Args:
        fmap_q: (B, C, H, W) query features
        fmap_r: (B, C, H, W) reference features
    Returns:
        corr: (B, H*W, H, W)  — for each query pixel, correlation with all ref pixels
    """
    B, C, H, W = fmap_q.shape
    q_flat = fmap_q.reshape(B, C, -1)        # (B, C, N)
    r_flat = fmap_r.reshape(B, C, -1)        # (B, C, N)
    corr = torch.einsum('bcm,bcn->bmn', q_flat, r_flat)  # (B, N, N)
    corr = corr.reshape(B, H * W, H, W)
    return corr


class CoarseFlowDecoder(nn.Module):
    """Decode initial flow from global correlation volume at low resolution."""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
#  Flow Refinement Head
# ==============================================================================

class FlowRefinementHead(nn.Module):
    """
    Correlation → encoder → ConvGRU → flow + confidence.
    """

    def __init__(
        self,
        corr_channels: int,
        hidden_dim: int = 128,
        context_dim: int = 64,
        conf_dim: int = 1,
    ):
        super().__init__()
        self.conf_dim = conf_dim

        # corr + flow(2) + conf → context_dim
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels + 2 + conf_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, context_dim, 3, padding=1),
            nn.GELU(),
        )

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim)

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2 + conf_dim, 3, padding=1),
        )

    def forward(
        self,
        corr: torch.Tensor,
        hidden: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([corr, flow, confidence], dim=1)
        inp_encoded = self.corr_encoder(inp)
        new_hidden = self.gru(hidden, inp_encoded)

        out = self.flow_head(new_hidden)
        delta_flow = out[:, :2]
        new_conf = torch.sigmoid(out[:, 2:])
        flow_out = flow + delta_flow

        return delta_flow, new_conf, new_hidden, flow_out


# ==============================================================================
#  Main Network
# ==============================================================================

class SingleScalePoseNet(nn.Module):
    """
    单尺度 Flow Pose Network.

    输入: 64d DA3 fine 特征 (query + render), 深度图
    输出: 6-DOF pose update (se3)

    Args:
        hidden_dim: GRU 隐状态维度
        decode_dim: decoder 输出维度
        feat_in_dim: 输入特征维度 (DA3 = 64)
        local_radius: correlation 搜索半径
        corr_dilations: multi-scale dilation 因子
        fine_iters: GRU 迭代次数
        damping: LM solver 阻尼
        fine_hw: 特征分辨率
        intrinsics: 相机内参
        img_hw: 原始图像分辨率
        irls_iters: IRLS 鲁棒估计迭代次数
        robust_kernel: 鲁棒核类型
    """

    DEFAULT_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    DEFAULT_IMG_HW = (480, 640)

    def __init__(
        self,
        hidden_dim: int = 128,
        decode_dim: int = 64,
        feat_in_dim: int = 64,
        local_radius: int = 4,
        corr_dilations: Tuple[int, ...] = (1, 2, 4),
        fine_iters: int = 8,
        damping: float = 1e-3,
        fine_hw: Tuple[int, int] = (69, 121),
        intrinsics: Optional[Dict[str, float]] = None,
        img_hw: Optional[Tuple[int, int]] = None,
        irls_iters: int = 3,
        irls_huber_k: float = 1.345,
        robust_kernel: str = 'huber',
        gnc_mu_init: float = 1.0,
        gnc_mu_step: float = 1.4,
        pixel_stride: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decode_dim = decode_dim
        self.local_radius = local_radius
        self.corr_dilations = corr_dilations
        self.fine_iters = fine_iters
        self.damping = damping
        self.irls_iters = irls_iters
        self.irls_huber_k = irls_huber_k
        self.robust_kernel = robust_kernel
        self.gnc_mu_init = gnc_mu_init
        self.gnc_mu_step = gnc_mu_step
        self.pixel_stride = pixel_stride

        self.FINE_HW = tuple(fine_hw)
        self.BASE_INTRINSICS = intrinsics if intrinsics is not None else self.DEFAULT_INTRINSICS.copy()
        self.IMG_HW = img_hw if img_hw is not None else self.DEFAULT_IMG_HW

        # Correlation channels: (2r+1)² per dilation level
        d = 2 * local_radius + 1
        corr_ch = d * d * len(corr_dilations)  # 81 × 3 = 243

        # ── Feature decoder (shared Q/R) ──
        # If input is already decode_dim, use lightweight decoder;
        # otherwise use full projection
        if feat_in_dim == decode_dim:
            # Lightweight: just normalize + tiny refinement
            self.feat_dec = ScaleDecoder(feat_in_dim, decode_dim, mid_dim=64)
        else:
            self.feat_dec = ScaleDecoder(feat_in_dim, decode_dim, mid_dim=128)

        # ── Context network: query feat → initial GRU hidden ──
        self.context_net = nn.Sequential(
            nn.Conv2d(decode_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # ── Flow refinement head (GRU-based) ──
        self.flow_head = FlowRefinementHead(
            corr_channels=corr_ch,
            hidden_dim=hidden_dim,
            context_dim=64,
            conf_dim=1,
        )

        # ── Intrinsics at fine resolution ──
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        """Scale base intrinsics to target resolution."""
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        render_feats: Dict[str, torch.Tensor],
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_feats: {'fine': (B, feat_in_dim, H, W)}
            render_feats: {'fine': (B, feat_in_dim, H, W)}
            depth: (B, H, W) rendered depth

        Returns:
            dict with:
                'delta_xi': (B, 6)
                'flow_fine': (B, 2, H, W)
                'conf_fine': (B, 1, H, W)
                'fine_flow_preds': list of flow per GRU iteration
                'hidden_fine': (B, hidden_dim, H, W)
        """
        q_raw = query_feats['fine']
        r_raw = render_feats['fine']
        B = q_raw.shape[0]
        device = q_raw.device

        # ══════════════════════════════════════════════
        #  1. Decode features → 64d
        # ══════════════════════════════════════════════
        q_fine = self._match_spatial(self.feat_dec(q_raw), self.FINE_HW)
        r_fine = self._match_spatial(self.feat_dec(r_raw), self.FINE_HW)

        # ══════════════════════════════════════════════
        #  2. Initialize hidden state and flow
        # ══════════════════════════════════════════════
        h = self.context_net(q_fine)  # (B, hidden_dim, H, W)

        fH, fW = self.FINE_HW
        flow = torch.zeros(B, 2, fH, fW, device=device)
        conf = torch.ones(B, 1, fH, fW, device=device) * 0.5

        # ══════════════════════════════════════════════
        #  3. GRU iterations with dilated correlation
        # ══════════════════════════════════════════════
        flow_preds = []

        for _iter in range(self.fine_iters):
            corr = guided_multiscale_correlation(
                q_fine, r_fine, flow,
                radius=self.local_radius,
                dilations=self.corr_dilations,
            )

            _, conf, h, flow = self.flow_head(corr, h, flow, conf)
            flow_preds.append(flow)

        # ══════════════════════════════════════════════
        #  4. Geometry Solver
        # ══════════════════════════════════════════════
        result = {
            'flow_fine': flow,
            'conf_fine': conf,
            'hidden_fine': h,
            'fine_flow_preds': flow_preds,
            'decoded_q_fine': q_fine,
        }

        if depth is not None:
            with torch.amp.autocast('cuda', enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow.float()
                conf_f32 = conf.float()

                # Resize depth to match flow if needed
                if depth_f32.shape[-2:] != flow_f32.shape[-2:]:
                    need_squeeze = (depth_f32.ndim == 3)
                    if need_squeeze:
                        depth_f32 = depth_f32.unsqueeze(1)
                    depth_f32 = F.interpolate(
                        depth_f32, size=flow_f32.shape[-2:],
                        mode='bilinear', align_corners=False)
                    if need_squeeze:
                        depth_f32 = depth_f32.squeeze(1)

                Ju, Jv, valid = compute_image_jacobian(
                    depth_f32, self.fine_intrinsics)

                delta_xi = diff_pose_solve(
                    flow_f32, conf_f32, Ju, Jv, valid,
                    damping=self.damping,
                    irls_iters=self.irls_iters,
                    irls_huber_k=self.irls_huber_k,
                    pixel_stride=self.pixel_stride,
                    robust_kernel=self.robust_kernel,
                    gnc_mu_init=self.gnc_mu_init,
                    gnc_mu_step=self.gnc_mu_step,
                )

            result['delta_xi'] = delta_xi

        return result

    @staticmethod
    def _match_spatial(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != tuple(target_hw):
            return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GT flow (用于 flow supervision).
        Args:
            pose_init: (B, 4, 4) w2c
            pose_gt:   (B, 4, 4) w2c
            depth:     (B, H, W) depth at init pose
            resolution: (tH, tW)
        Returns:
            flow_gt: (B, 2, tH, tW)
            valid:   (B, 1, tH, tW)
        """
        B, H, W = depth.shape
        device = depth.device
        intrinsics = self._scale_intrinsics(H, W)
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth
        Y = (v_coords - cy) / fy * depth
        Z = depth

        pts_init = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts_init.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = pose_gt @ torch.linalg.inv(pose_init)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat)
        pts_gt = pts_gt.reshape(B, 3, H, W)

        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([u_gt - u_coords.unsqueeze(1),
                              v_gt - v_coords.unsqueeze(1)], dim=1)

        valid = (
            (depth.unsqueeze(1) > 0.05) &
            (Z_gt_raw > 0.1) &
            (u_gt > -0.5) & (u_gt < W - 0.5) &
            (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()

        flow_gt = flow_gt * valid

        tH, tW = resolution
        if tH != H or tW != W:
            scale_x = tW / W
            scale_y = tH / H
            flow_gt = F.interpolate(flow_gt, size=(tH, tW), mode='bilinear',
                                     align_corners=False)
            flow_gt[:, 0] *= scale_x
            flow_gt[:, 1] *= scale_y
            valid = F.interpolate(valid, size=(tH, tW), mode='nearest')

        return flow_gt, valid
