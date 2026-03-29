"""
TransformerPoseNet: Transformer-based single-scale pose estimation
==================================================================
用 self-attention + cross-attention 替代 correlation volume，直接解码 flow。

核心优势:
  - Cross-attention 天然具有全局感受野，不受 local correlation 搜索半径限制
  - 可以处理大位移 (8°+ noise → ~40px displacement)
  - 无需显式 correlation volume 计算

架构:
  ScaleDecoder(64d) → Downsample(attn_hw) → PE
  → K iterations (weight-shared):
      flow_cond(flow+conf → 64d, additive)
      → N × [Self-Attn(Q) → Cross-Attn(Q,R) → FFN]
      → flow_head → Δflow + conf
  → Upsample(fine_hw) → Geometry Solver → Δξ(6)

参数量估计 (默认配置):
  ScaleDecoder: ~16K
  Transformer (4 layers, 8 heads, 64d, ffn=256): ~460K
  flow_cond + flow_head: ~25K
  context_proj: ~8K
  Total: ~510K params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve


# ==============================================================================
#  Building Blocks
# ==============================================================================

class ScaleDecoder(nn.Module):
    """1×1 conv decoder: in_dim → out_dim, L2 normalized."""

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


class SinusoidalPE2D(nn.Module):
    """2D sinusoidal positional encoding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns (1, H*W, d_model)."""
        d = self.d_model
        half = d // 2
        div = torch.exp(
            torch.arange(0, half, 2, device=device, dtype=torch.float32) *
            -(math.log(10000.0) / half)
        )
        pos_h = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(H, W, d, device=device)
        pe[:, :, 0:half:2] = torch.sin(pos_h * div).unsqueeze(1).expand(-1, W, -1)
        pe[:, :, 1:half:2] = torch.cos(pos_h * div).unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half::2] = torch.sin(pos_w * div).unsqueeze(0).expand(H, -1, -1)
        pe[:, :, half + 1::2] = torch.cos(pos_w * div).unsqueeze(0).expand(H, -1, -1)
        return pe.reshape(1, H * W, d)


class TransformerMatchingBlock(nn.Module):
    """
    Weight-shared Transformer matching block for iterative flow refinement.

    Each iteration:
      1. Inject flow conditioning into Q tokens (additive)
      2. N layers of pre-norm: Self-Attn(Q) → Cross-Attn(Q→R) → FFN
      3. Flow head: predict delta_flow + confidence
    """

    def __init__(
        self,
        feat_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Flow conditioning: current flow(2) + conf(1) → feat_dim
        self.flow_cond = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        # Transformer layers
        self.self_attn = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm_sa = nn.ModuleList()   # pre-norm for self-attn
        self.norm_ca = nn.ModuleList()   # pre-norm for cross-attn
        self.norm_ff = nn.ModuleList()   # pre-norm for FFN

        for _ in range(n_layers):
            self.self_attn.append(
                nn.MultiheadAttention(
                    feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn.append(
                nn.MultiheadAttention(
                    feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffn.append(nn.Sequential(
                nn.Linear(feat_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, feat_dim),
                nn.Dropout(dropout),
            ))
            self.norm_sa.append(nn.LayerNorm(feat_dim))
            self.norm_ca.append(nn.LayerNorm(feat_dim))
            self.norm_ff.append(nn.LayerNorm(feat_dim))

        # Flow prediction head
        self.flow_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),  # delta_flow(2) + confidence(1)
        )

    def forward(
        self,
        q_tokens: torch.Tensor,   # (B, N, C) query tokens + PE
        r_tokens: torch.Tensor,   # (B, N, C) reference tokens + PE
        flow: torch.Tensor,       # (B, 2, H, W) current flow at attention resolution
        conf: torch.Tensor,       # (B, 1, H, W) current confidence
        H: int, W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_flow: (B, 2, H, W)
            new_conf: (B, 1, H, W)
            flow_out: (B, 2, H, W)
        """
        B = q_tokens.shape[0]

        # Flow conditioning
        flow_inp = torch.cat([flow, conf], dim=1)  # (B, 3, H, W)
        flow_tokens = flow_inp.flatten(2).permute(0, 2, 1)  # (B, N, 3)
        x = q_tokens + self.flow_cond(flow_tokens)

        # Transformer layers
        for i in range(self.n_layers):
            # Self-attention
            xn = self.norm_sa[i](x)
            x = x + self.self_attn[i](xn, xn, xn, need_weights=False)[0]
            # Cross-attention
            xn = self.norm_ca[i](x)
            rn = self.norm_ca[i](r_tokens)
            x = x + self.cross_attn[i](xn, rn, rn, need_weights=False)[0]
            # FFN
            xn = self.norm_ff[i](x)
            x = x + self.ffn[i](xn)

        # Predict flow
        out = self.flow_head(x)  # (B, N, 3)
        out_2d = out.permute(0, 2, 1).reshape(B, 3, H, W)
        delta_flow = out_2d[:, :2]
        new_conf = torch.sigmoid(out_2d[:, 2:3])
        flow_out = flow + delta_flow

        return delta_flow, new_conf, flow_out


# ==============================================================================
#  Main Network
# ==============================================================================

class TransformerPoseNet(nn.Module):
    """
    Transformer-based single-scale flow pose network.

    用 cross-attention 替代 correlation volume 进行特征匹配。
    在 attention resolution 做全局注意力匹配，上采样后经几何求解器得到位姿。

    Args:
        feat_dim: working feature dimension (must match ScaleDecoder output)
        feat_in_dim: input feature dimension (DA3 = 64)
        n_heads: number of attention heads
        n_layers: number of transformer layers per iteration
        ffn_dim: FFN hidden dimension
        fine_iters: number of iterative refinement passes
        attn_hw: attention resolution (downsampled from fine_hw)
        fine_hw: output flow resolution
        damping: LM solver damping
        intrinsics: camera intrinsics dict
        img_hw: original image resolution
        irls_iters: IRLS robust estimation iterations
        robust_kernel: robust kernel type for IRLS
    """

    DEFAULT_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    DEFAULT_IMG_HW = (480, 640)

    def __init__(
        self,
        feat_dim: int = 64,
        feat_in_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        fine_iters: int = 4,
        attn_hw: Tuple[int, int] = (35, 61),
        fine_hw: Tuple[int, int] = (69, 121),
        damping: float = 1e-3,
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
        self.feat_dim = feat_dim
        self.fine_iters = fine_iters
        self.ATTN_HW = tuple(attn_hw)
        self.FINE_HW = tuple(fine_hw)
        self.damping = damping
        self.irls_iters = irls_iters
        self.irls_huber_k = irls_huber_k
        self.robust_kernel = robust_kernel
        self.gnc_mu_init = gnc_mu_init
        self.gnc_mu_step = gnc_mu_step
        self.pixel_stride = pixel_stride

        self.BASE_INTRINSICS = intrinsics if intrinsics is not None else self.DEFAULT_INTRINSICS.copy()
        self.IMG_HW = img_hw if img_hw is not None else self.DEFAULT_IMG_HW

        # ── Feature decoder (shared Q/R) ──
        if feat_in_dim == feat_dim:
            self.feat_dec = ScaleDecoder(feat_in_dim, feat_dim, mid_dim=64)
        else:
            self.feat_dec = ScaleDecoder(feat_in_dim, feat_dim, mid_dim=128)

        # ── Positional encoding ──
        self.pe = SinusoidalPE2D(feat_dim)

        # ── Transformer matching block (weight-shared across iterations) ──
        self.matcher = TransformerMatchingBlock(
            feat_dim=feat_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # ── Intrinsics at fine resolution ──
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    @staticmethod
    def _match_spatial(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != tuple(target_hw):
            return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

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
                'delta_xi': (B, 6) if depth provided
                'flow_fine': (B, 2, fH, fW) final flow at fine resolution
                'conf_fine': (B, 1, fH, fW) final confidence
                'fine_flow_preds': list of flow at fine res per iteration
        """
        q_raw = query_feats['fine']
        r_raw = render_feats['fine']
        B = q_raw.shape[0]
        device = q_raw.device

        # ══════════════════════════════════════════════
        #  1. Decode features → feat_dim, match to fine resolution
        # ══════════════════════════════════════════════
        q_fine = self._match_spatial(self.feat_dec(q_raw), self.FINE_HW)
        r_fine = self._match_spatial(self.feat_dec(r_raw), self.FINE_HW)

        # ══════════════════════════════════════════════
        #  2. Downsample to attention resolution
        # ══════════════════════════════════════════════
        aH, aW = self.ATTN_HW
        q_attn = self._match_spatial(q_fine, self.ATTN_HW)  # (B, C, aH, aW)
        r_attn = self._match_spatial(r_fine, self.ATTN_HW)

        # Tokenize + PE
        q_tokens = q_attn.flatten(2).permute(0, 2, 1)  # (B, N, C)
        r_tokens = r_attn.flatten(2).permute(0, 2, 1)
        pe = self.pe(aH, aW, device=device)
        q_tokens = q_tokens + pe
        r_tokens = r_tokens + pe

        # ══════════════════════════════════════════════
        #  3. Iterative transformer matching at attn resolution
        # ══════════════════════════════════════════════
        flow = torch.zeros(B, 2, aH, aW, device=device)
        conf = torch.ones(B, 1, aH, aW, device=device) * 0.5

        flow_preds_fine = []

        for _iter in range(self.fine_iters):
            delta_flow, conf, flow = self.matcher(
                q_tokens, r_tokens, flow, conf, aH, aW)

            # Upsample to fine resolution for supervision/output
            fH, fW = self.FINE_HW
            flow_fine = F.interpolate(flow, size=(fH, fW), mode='bilinear', align_corners=False)
            # Scale flow values for resolution change
            flow_fine = flow_fine.clone()
            flow_fine[:, 0] *= fW / aW
            flow_fine[:, 1] *= fH / aH
            flow_preds_fine.append(flow_fine)

        # Final confidence at fine resolution
        conf_fine = F.interpolate(conf, size=self.FINE_HW, mode='bilinear', align_corners=False)

        # ══════════════════════════════════════════════
        #  4. Geometry Solver
        # ══════════════════════════════════════════════
        result = {
            'flow_fine': flow_preds_fine[-1],
            'conf_fine': conf_fine,
            'fine_flow_preds': flow_preds_fine,
            'decoded_q_fine': q_fine,
        }

        if depth is not None:
            with torch.amp.autocast('cuda', enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow_preds_fine[-1].float()
                conf_f32 = conf_fine.float()

                # Resize depth to match flow
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

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GT flow for supervision."""
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
