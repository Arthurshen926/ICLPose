"""
MSFlowPoseNet: SD-Primary Coarse-to-Fine Flow Pose Network
============================================================
核心思路:
  - 单次渲染 (不迭代) → 多尺度 decode → 全局/局部 correlation →
    coarse→mid→fine 流场预测 → 几何求解

层级结构 (Q/R 共享 decoder, 每层独立 FlowHead):
  coarse: SD s5 1280d @  8×10  → 64d, global all-pairs correlation (80ch)
  mid:    SD s4 1280d @ 16×20  → 64d, warp-guided local corr r=4 (81ch)
  fine:   SD s3 640d @32×40 + DINO 768d @35×46 → dual-branch → 64d @35×46, local corr r=4 (81ch)

Fine 层支持 RAFT 风格多次 GRU 迭代: 每次迭代重新计算 warp-guided correlation,
用更新后的 flow 进行渐进细化. fine_iters 默认 4 次.

最终 flow 在 35×46 分辨率 → Image Jacobian + 加权最小二乘 → 几何求解
35×46 = 1,610 像素约束 vs 6 未知数 (268:1), 无需上采样.

参数量估算:
  Decoder × 4 (shared Q/R) + FlowHead × 3 + misc ≈ 2M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve


# ==============================================================================
#  Building Blocks
# ==============================================================================

class ScaleDecoder(nn.Module):
    """
    单尺度 CNN 解码器: high-dim → 64d, 纯逐像素 (不改分辨率).

    1×1 conv 逐步降维, 保留空间结构.
    Query 和 Render 共享同一实例, 确保映射到相同特征空间.
    """

    def __init__(self, in_dim: int, out_dim: int = 64, mid_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) → (B, 64, H, W), L2 normalized"""
        out = self.net(x)
        return F.normalize(out, p=2, dim=1)


class FineDualDecoder(nn.Module):
    """
    Fine 层双分支解码器:
      - SD s3 → 64d via 1×1 conv → interpolate to target_hw (如有必要)
      - DINO  → 64d via 1×1 conv
      - concat 128d → 1×1 fuse → 64d

    Query 和 Render 共享同一实例.
    当 SD 和 DINO 分辨率相同时自动跳过插值.
    """

    def __init__(self, sd_dim: int = 640, dino_dim: int = 768,
                 out_dim: int = 64, mid_dim: int = 256,
                 target_hw: Tuple[int, int] = (35, 46)):
        super().__init__()
        self.target_hw = target_hw

        # SD s3 branch
        self.sd_branch = nn.Sequential(
            nn.Conv2d(sd_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

        # DINO branch
        self.dino_branch = nn.Sequential(
            nn.Conv2d(dino_dim, mid_dim, 1),
            nn.GroupNorm(16, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
        )

        # Fusion: 128d → 64d
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(
        self,
        sd_feat: torch.Tensor,    # (B, 640, Hs, Ws)
        dino_feat: torch.Tensor,  # (B, 768, Hd, Wd)
    ) -> torch.Tensor:
        """→ (B, 64, tH, tW) L2 normalized"""
        sd_out = self.sd_branch(sd_feat)
        # 仅当 SD 分辨率与目标不同时才 interpolate
        if sd_out.shape[2:] != torch.Size(self.target_hw):
            sd_out = F.interpolate(
                sd_out, size=self.target_hw,
                mode='bilinear', align_corners=False,
            )

        dino_out = self.dino_branch(dino_feat)
        if dino_out.shape[2:] != torch.Size(self.target_hw):
            dino_out = F.interpolate(
                dino_out, size=self.target_hw,
                mode='bilinear', align_corners=False,
            )

        fused = self.fuse(torch.cat([sd_out, dino_out], dim=1))
        return F.normalize(fused, p=2, dim=1)


# ==============================================================================
#  Correlation Functions
# ==============================================================================

def global_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
) -> torch.Tensor:
    """
    全局 All-Pairs Correlation (coarse 层专用).

    对每个 fmap1 像素, 计算与 fmap2 所有像素的点积相关.
    结果是一个 (H1*W1) 维的相关向量 — 每个查询位置获得全局匹配信息.

    Args:
        fmap1: (B, C, H1, W1) 查询特征
        fmap2: (B, C, H2, W2) 渲染特征

    Returns:
        corr: (B, H2*W2, H1, W1) — channel dim = 所有 render 位置
    """
    B, C, H1, W1 = fmap1.shape
    _, _, H2, W2 = fmap2.shape

    fmap1_flat = fmap1.reshape(B, C, H1 * W1)   # (B, C, N1)
    fmap2_flat = fmap2.reshape(B, C, H2 * W2)   # (B, C, N2)

    # dot: (B, N1, N2) = fmap1^T @ fmap2
    corr = torch.bmm(fmap1_flat.transpose(1, 2), fmap2_flat)  # (B, N1, N2)

    # reshape to (B, N2, H1, W1) — 每个 query 像素有 N2 个 channel
    corr = corr.permute(0, 2, 1).reshape(B, H2 * W2, H1, W1)
    return corr


def local_correlation(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
) -> torch.Tensor:
    """
    局部 Correlation (mid/fine 层).
    在 fmap2 上对每个 fmap1 像素的 (2r+1)² 邻域计算相关.

    Args:
        fmap1: (B, C, H, W)
        fmap2: (B, C, H, W) — 必须与 fmap1 同分辨率
        radius: 搜索半径

    Returns:
        corr: (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1

    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)

    # 预分配输出
    corr = fmap1.new_empty(B, d * d, H, W)
    idx = 0
    for dy in range(-radius, radius + 1):
        strip = fmap2_pad.narrow(2, radius + dy, H)
        for dx in range(-radius, radius + 1):
            shifted = strip.narrow(3, radius + dx, W)
            corr[:, idx] = (fmap1 * shifted).sum(1)
            idx += 1

    return corr


def guided_local_correlation(
    fmap_q: torch.Tensor,
    fmap_r: torch.Tensor,
    flow: torch.Tensor,
    radius: int = 4,
) -> torch.Tensor:
    """
    Warp-guided 局部 correlation.

    先用 flow 将 fmap_r 向 fmap_q 方向 warp, 然后做标准局部 correlation.
    这样局部搜索窗口就以上一级预测的位移为中心, 避免大位移被 local window 截断.

    Args:
        fmap_q: (B, C, H, W)
        fmap_r: (B, C, H, W)
        flow:   (B, 2, H, W) 上一级上采样并缩放后的 flow [du, dv]
        radius: 搜索半径

    Returns:
        corr: (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap_q.shape

    # 构建 warp grid: base_grid + flow → normalized [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=torch.float32),
        torch.arange(W, device=flow.device, dtype=torch.float32),
        indexing='ij',
    )
    # flow[:, 0] = du (x-dir), flow[:, 1] = dv (y-dir)
    grid_x = (grid_x.unsqueeze(0) + flow[:, 0]) / (W - 1) * 2.0 - 1.0
    grid_y = (grid_y.unsqueeze(0) + flow[:, 1]) / (H - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    warped_r = F.grid_sample(
        fmap_r, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return local_correlation(fmap_q, warped_r, radius=radius)


# ==============================================================================
#  ConvGRU (from CorrPoseNet)
# ==============================================================================

class ConvGRU(nn.Module):
    """Convolutional GRU, 3×3 卷积门控."""

    def __init__(self, hidden_dim: int = 128, input_dim: int = 64):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# ==============================================================================
#  Flow Refinement Head (per-scale)
# ==============================================================================

class FlowRefinementHead(nn.Module):
    """
    单尺度 flow 细化头:
      correlation → encoder → ConvGRU 更新 hidden → 输出 flow + confidence

    每个尺度独立一个 Head (不共享权重).

    Args:
        corr_channels: correlation 的 channel 数
          - coarse(global): H2*W2 = 80
          - mid/fine(local r=4): (2*4+1)² = 81
        hidden_dim: GRU 隐状态维度
        flow_dim: flow + confidence 输出维度 (3: du, dv, conf)
    """

    def __init__(
        self,
        corr_channels: int,
        hidden_dim: int = 128,
        context_dim: int = 64,
    ):
        super().__init__()

        # Correlation encoder: corr + flow(2) + conf(1) → context_dim
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels + 3, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, context_dim, 3, padding=1),
            nn.GELU(),
        )

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim)

        # Flow head: hidden → flow(2) + confidence(1)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),  # du, dv, raw_conf
        )

    def forward(
        self,
        corr: torch.Tensor,     # (B, corr_ch, H, W)
        hidden: torch.Tensor,   # (B, hidden_dim, H, W)
        flow: torch.Tensor,     # (B, 2, H, W) current flow estimate
        confidence: torch.Tensor,  # (B, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_flow: (B, 2, H, W) flow update
            new_conf:   (B, 1, H, W) updated confidence
            new_hidden: (B, hidden_dim, H, W)
            flow_out:   (B, 2, H, W) = flow + delta_flow
        """
        # Encode: concat corr + current flow + conf
        inp = torch.cat([corr, flow, confidence], dim=1)
        inp_encoded = self.corr_encoder(inp)

        # GRU update
        new_hidden = self.gru(hidden, inp_encoded)

        # Predict flow delta + conf
        out = self.flow_head(new_hidden)
        delta_flow = out[:, :2]          # (B, 2, H, W)
        raw_conf = out[:, 2:3]           # (B, 1, H, W)
        new_conf = torch.sigmoid(raw_conf)

        flow_out = flow + delta_flow

        return delta_flow, new_conf, new_hidden, flow_out


# ==============================================================================
#  Main Network
# ==============================================================================

class MSFlowPoseNet(nn.Module):
    """
    SD-Primary Coarse-to-Fine Multi-Scale Flow Pose Network.

    Architecture:
      1. Decode query/render features at each scale to 64d (shared decoder)
      2. Coarse: global all-pairs correlation → flow head → initial flow
      3. Mid: upsample coarse flow → warp-guided local corr → flow head
      4. Fine: upsample mid flow → dual-branch decode → local corr → flow head
      5. Geometry solver: Image Jacobian + weighted least squares → Δξ

    Resolution 通过构造函数参数配置, 同时支持 v1/v2 特征:
      v1: coarse 7×10, mid 15×20, fine_sd 35×46, fine 35×46
      v2: coarse 8×10, mid 16×20, fine_sd 32×40, fine 35×46

    Args:
        hidden_dim: GRU hidden state dimension
        decode_dim: decoder output dimension (common across scales)
        local_radius: local correlation search radius
        damping: LM solver damping
        coarse_hw: coarse 特征分辨率 (H, W)
        mid_hw: mid 特征分辨率
        fine_hw: fine 输出分辨率 (= DINO patch grid)
        fine_iters: fine 层 GRU 迭代次数 (RAFT 风格, 默认 4)
    """

    # Intrinsics (Replica room_0 base)
    BASE_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    IMG_HW = (480, 640)

    def __init__(
        self,
        hidden_dim: int = 128,
        decode_dim: int = 64,
        local_radius: int = 4,
        damping: float = 1e-3,
        coarse_hw: Tuple[int, int] = (7, 10),
        mid_hw: Tuple[int, int] = (15, 20),
        fine_hw: Tuple[int, int] = (35, 46),
        fine_iters: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decode_dim = decode_dim
        self.local_radius = local_radius
        self.damping = damping
        self.fine_iters = fine_iters

        # 分辨率配置 (实例属性, 不再是类常量)
        self.COARSE_HW = tuple(coarse_hw)
        self.MID_HW = tuple(mid_hw)
        self.FINE_HW = tuple(fine_hw)

        d = 2 * local_radius + 1
        local_corr_ch = d * d  # 81

        # ── Decoders (Q/R shared) ──
        self.coarse_dec = ScaleDecoder(1280, decode_dim)
        self.mid_dec = ScaleDecoder(1280, decode_dim)
        self.fine_dec = FineDualDecoder(
            640, 768, decode_dim, target_hw=self.FINE_HW)

        # ── Flow Heads ──
        coarse_global_ch = self.COARSE_HW[0] * self.COARSE_HW[1]
        self.coarse_head = FlowRefinementHead(
            corr_channels=coarse_global_ch,
            hidden_dim=hidden_dim,
        )
        self.mid_head = FlowRefinementHead(
            corr_channels=local_corr_ch,
            hidden_dim=hidden_dim,
        )
        self.fine_head = FlowRefinementHead(
            corr_channels=local_corr_ch,
            hidden_dim=hidden_dim,
        )

        # ── Context encoder: query feature → initial hidden state ──
        self.context_net = nn.Sequential(
            nn.Conv2d(decode_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # Intrinsics at fine resolution for geometry solving
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        """从 640×480 基准缩放内参到目标分辨率."""
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    def _init_flow(self, B: int, H: int, W: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化零 flow 和均匀 confidence."""
        flow = torch.zeros(B, 2, H, W, device=device)
        conf = torch.ones(B, 1, H, W, device=device) * 0.5
        return flow, conf

    def _upsample_flow(
        self, flow: torch.Tensor, conf: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双线性上采样 flow 和 confidence 到目标分辨率, 并缩放 flow 值.
        """
        _, _, sH, sW = flow.shape
        tH, tW = target_hw
        scale_x = tW / sW
        scale_y = tH / sH

        flow_up = F.interpolate(flow, size=target_hw, mode='bilinear',
                                 align_corners=False)
        flow_up[:, 0] *= scale_x
        flow_up[:, 1] *= scale_y

        conf_up = F.interpolate(conf, size=target_hw, mode='bilinear',
                                 align_corners=False)
        return flow_up, conf_up

    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        render_feats: Dict[str, torch.Tensor],
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            query_feats: {
                'coarse':    (B, 1280, 8, 10),
                'mid':       (B, 1280, 16, 20),
                'fine_sd':   (B, 640, 32, 40),
                'fine_dino': (B, 768, 35, 46),
            }
            render_feats: same structure (from 3DGS rendering)
            depth: (B, 35, 46) rendered depth at fine resolution

        Returns:
            dict:
                'delta_xi': (B, 6) se(3) update — only if depth provided
                'flow_fine': (B, 2, 35, 46) final fine-resolution flow
                'conf_fine': (B, 1, 35, 46)
                'flow_coarse': (B, 2, 8, 10)
                'flow_mid':    (B, 2, 16, 20)
                'hidden_fine': (B, hidden_dim, 35, 46)
        """
        B = query_feats['coarse'].shape[0]
        device = query_feats['coarse'].device

        # ══════════════════════════════════════════════
        #  1. Decode all scales to 64d (shared Q/R decoders)
        # ══════════════════════════════════════════════
        q_coarse = self.coarse_dec(query_feats['coarse'])    # (B, 64, 8, 10)
        r_coarse = self.coarse_dec(render_feats['coarse'])

        q_mid = self.mid_dec(query_feats['mid'])             # (B, 64, 16, 20)
        r_mid = self.mid_dec(render_feats['mid'])

        q_fine = self.fine_dec(                               # (B, 64, 35, 46)
            query_feats['fine_sd'], query_feats['fine_dino'])
        r_fine = self.fine_dec(
            render_feats['fine_sd'], render_feats['fine_dino'])

        # ══════════════════════════════════════════════
        #  2. Coarse: Global correlation → initial flow
        # ══════════════════════════════════════════════
        coarse_corr = global_correlation(q_coarse, r_coarse)  # (B, 80, 8, 10)

        # Context → initial hidden (at coarse resolution)
        h_coarse = self.context_net(q_coarse)  # (B, hidden_dim, 8, 10)

        flow_c, conf_c = self._init_flow(
            B, *self.COARSE_HW, device)

        _, conf_c, h_coarse, flow_c = self.coarse_head(
            coarse_corr, h_coarse, flow_c, conf_c)

        # ══════════════════════════════════════════════
        #  3. Mid: Warp-guided local correlation
        # ══════════════════════════════════════════════
        flow_m, conf_m = self._upsample_flow(
            flow_c, conf_c, self.MID_HW)

        mid_corr = guided_local_correlation(
            q_mid, r_mid, flow_m, radius=self.local_radius)  # (B, 81, 16, 20)

        # Initialize mid hidden from upsampled coarse hidden
        h_mid = F.interpolate(
            h_coarse, size=self.MID_HW, mode='bilinear', align_corners=False)

        _, conf_m, h_mid, flow_m = self.mid_head(
            mid_corr, h_mid, flow_m, conf_m)

        # ══════════════════════════════════════════════
        #  4. Fine: RAFT-style iterative refinement
        #     多次 GRU 迭代, 每次重新计算 warp-guided correlation
        # ══════════════════════════════════════════════
        flow_f, conf_f = self._upsample_flow(
            flow_m, conf_m, self.FINE_HW)

        h_fine = F.interpolate(
            h_mid, size=self.FINE_HW, mode='bilinear', align_corners=False)

        fine_flow_preds = []   # 存储每次迭代的 flow, 用于 sequence loss

        for _iter in range(self.fine_iters):
            fine_corr = guided_local_correlation(
                q_fine, r_fine, flow_f, radius=self.local_radius)

            _, conf_f, h_fine, flow_f = self.fine_head(
                fine_corr, h_fine, flow_f, conf_f)

            fine_flow_preds.append(flow_f)

        # ══════════════════════════════════════════════
        #  5. Geometry Solver (if depth available)
        #     直接在 35×46 分辨率求解, 无需上采样
        # ══════════════════════════════════════════════
        result = {
            'flow_coarse': flow_c,
            'conf_coarse': conf_c,
            'flow_mid': flow_m,
            'conf_mid': conf_m,
            'flow_fine': flow_f,
            'conf_fine': conf_f,
            'hidden_fine': h_fine,
            'fine_flow_preds': fine_flow_preds,  # 每次迭代的 flow, 用于 sequence loss
        }

        if depth is not None:
            # Geometry solver must run in fp32 (linalg.solve doesn't support fp16)
            with torch.cuda.amp.autocast(enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow_f.float()
                conf_f32 = conf_f.float()

                Ju, Jv, valid = compute_image_jacobian(
                    depth_f32, self.fine_intrinsics)

                delta_xi = diff_pose_solve(
                    flow_f32, conf_f32, Ju, Jv, valid,
                    damping=self.damping,
                )
            result['delta_xi'] = delta_xi

        return result

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据位姿差异和深度, 计算 GT flow (用于 flow supervision).

        将 init 位姿下的 3D 点投影到 GT 视角, 得到像素位移.
        同时返回有效性 mask: 排除 depth 无效、Z_gt 太小或投影出界的像素.

        Args:
            pose_init: (B, 4, 4) 初始 w2c
            pose_gt:   (B, 4, 4) GT w2c
            depth:     (B, H, W) 在 init 视角下的深度
            resolution: (tH, tW) 目标 flow 分辨率

        Returns:
            flow_gt: (B, 2, tH, tW) GT 像素位移 (无效像素设为 0)
            valid:   (B, 1, tH, tW) 有效像素 mask (0/1 float)
        """
        B, H, W = depth.shape
        device = depth.device
        intrinsics = self._scale_intrinsics(H, W)
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        # 像素坐标 → 3D (init camera frame)
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

        pts_init = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # (B, H, W, 4)
        pts_flat = pts_init.reshape(B, -1, 4).permute(0, 2, 1)  # (B, 4, N)

        # init_cam → world → gt_cam
        T_rel = pose_gt @ torch.linalg.inv(pose_init)  # (B, 4, 4)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat)  # (B, 3, N)
        pts_gt = pts_gt.reshape(B, 3, H, W)

        # Project
        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([u_gt - u_coords.unsqueeze(1),
                              v_gt - v_coords.unsqueeze(1)], dim=1)  # (B, 2, H, W)

        # Validity mask: depth > 0.05, Z_gt > 0.1, projected coords within image bounds
        valid = (
            (depth.unsqueeze(1) > 0.05) &           # 原始深度有效
            (Z_gt_raw > 0.1) &                       # GT 视角下在相机前方
            (u_gt > -0.5) & (u_gt < W - 0.5) &      # 投影在图像内
            (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()  # (B, 1, H, W)

        # 将无效像素的 flow 设为 0 (避免 NaN/Inf 传播)
        flow_gt = flow_gt * valid

        # Resize to target
        tH, tW = resolution
        if tH != H or tW != W:
            scale_x = tW / W
            scale_y = tH / H
            flow_gt = F.interpolate(flow_gt, size=(tH, tW), mode='bilinear',
                                     align_corners=False)
            flow_gt[:, 0] *= scale_x
            flow_gt[:, 1] *= scale_y
            # mask 用 nearest 以避免边缘模糊
            valid = F.interpolate(valid, size=(tH, tW), mode='nearest')

        return flow_gt, valid
