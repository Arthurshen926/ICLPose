"""
PoseRefiner — Single-scale iterative pose estimation from feature maps.

Architecture:
    Query + Ref features → DomainProjection → CrossAttention →
    GlobalCorrelation → CoarseGRU → Upsample →
    GuidedLocalCorrelation → FineGRU×N → ConvexUpsample(4×) →
    Depth-Normalized GeometrySolver → se(3) pose update

Key design decisions:
  - Single feature type (e.g. RADIO 64d), two processing resolutions (coarse/fine)
  - Coarse: global correlation for large-displacement matching
  - Fine: guided local correlation + 8 GRU iterations for sub-pixel accuracy
  - Convex upsampling to 4× solver resolution for translation sensitivity
  - Depth normalization: equalizes rotation/translation Jacobian magnitudes
  - Optional TranslationHead: MLP on detached hidden state (gradient-isolated)

Usage:
    model = PoseRefiner(config)
    result = model(query_feat, ref_feat, depth)
    # result: flow_coarse, flow_fine, conf_fine, delta_xi, fine_flow_preds, ...
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from modules.lie_algebra import se3_exp
from modules.geometry_solver import (
    compute_image_jacobian,
    diff_pose_solve,
    diff_pose_solve_sequential,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Correlation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def global_correlation(fmap_q: torch.Tensor, fmap_r: torch.Tensor) -> torch.Tensor:
    """All-pairs dot-product correlation.
    (B, C, Hq, Wq) × (B, C, Hr, Wr) → (B, Hr*Wr, Hq, Wq)
    """
    B, C, Hq, Wq = fmap_q.shape
    Hr, Wr = fmap_r.shape[2:]
    f_q = fmap_q.reshape(B, C, Hq * Wq)
    f_r = fmap_r.reshape(B, C, Hr * Wr)
    corr = torch.einsum('bcn,bcm->bmn', f_q, f_r)
    return corr.reshape(B, Hr * Wr, Hq, Wq)


def guided_local_correlation(
    fmap_q: torch.Tensor, fmap_r: torch.Tensor,
    flow: torch.Tensor, radius: int = 4,
) -> torch.Tensor:
    """Warp-guided local correlation (RAFT-style).
    Warps reference features by current flow, then computes local dot products
    in a (2r+1)×(2r+1) neighborhood. → (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap_q.shape
    d = 2 * radius + 1

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=torch.float32),
        torch.arange(W, device=flow.device, dtype=torch.float32),
        indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    coords = grid + flow

    coords_norm = torch.stack([
        2.0 * coords[:, 0] / max(W - 1, 1) - 1.0,
        2.0 * coords[:, 1] / max(H - 1, 1) - 1.0,
    ], dim=-1)

    fmap_r_warped = F.grid_sample(
        fmap_r, coords_norm, mode='bilinear',
        padding_mode='zeros', align_corners=True)

    fmap_r_pad = F.pad(fmap_r_warped, [radius] * 4, mode='constant', value=0)
    fmap_r_unfold = fmap_r_pad.unfold(2, d, 1).unfold(3, d, 1)
    fmap_r_unfold = fmap_r_unfold.reshape(B, C, H, W, d * d)

    fmap_q_exp = fmap_q.reshape(B, C, H, W, 1)
    corr = (fmap_q_exp * fmap_r_unfold).sum(dim=1)
    return corr.permute(0, 3, 1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class DomainProjection(nn.Module):
    """Shared 1×1 conv projection → L2 normalized matching space."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class SinusoidalPE2D(nn.Module):
    """2D sinusoidal positional encoding for cross-attention tokens."""

    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 256):
        super().__init__()
        pe = torch.zeros(d_model, max_h, max_w)
        d_half = d_model // 2
        n_freqs = d_half // 2
        div = torch.exp(torch.arange(0, n_freqs).float() * -(math.log(10000.0) / max(n_freqs, 1)))

        pos_h = torch.arange(max_h).float().unsqueeze(1)
        pos_w = torch.arange(max_w).float().unsqueeze(1)

        pe_h = torch.zeros(d_half, max_h)
        pe_h[0::2, :] = torch.sin(pos_h * div).T
        pe_h[1::2, :] = torch.cos(pos_h * div).T

        pe_w = torch.zeros(d_half, max_w)
        pe_w[0::2, :] = torch.sin(pos_w * div).T
        pe_w[1::2, :] = torch.cos(pos_w * div).T

        pe[:d_half, :, :] = pe_h.unsqueeze(2).expand(-1, -1, max_w)
        pe[d_half:, :, :] = pe_w.unsqueeze(1).expand(-1, max_h, -1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns (1, N, C) positional encoding for H×W spatial layout."""
        C = self.pe.shape[1]
        return self.pe[:, :C, :H, :W].reshape(1, C, H * W).permute(0, 2, 1)


class CrossAttention(nn.Module):
    """Symmetric multi-layer cross-attention with self-attention and FFN.
    Pre-norm architecture with gated residual (warmstart-safe)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pe = SinusoidalPE2D(d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

        self.q_self_attn = nn.ModuleList()
        self.q_cross_attn = nn.ModuleList()
        self.q_ffn = nn.ModuleList()
        self.q_norm1 = nn.ModuleList()
        self.q_norm2 = nn.ModuleList()
        self.q_norm3 = nn.ModuleList()

        self.r_self_attn = nn.ModuleList()
        self.r_cross_attn = nn.ModuleList()
        self.r_ffn = nn.ModuleList()
        self.r_norm1 = nn.ModuleList()
        self.r_norm2 = nn.ModuleList()
        self.r_norm3 = nn.ModuleList()

        for _ in range(n_layers):
            self.q_self_attn.append(nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.q_cross_attn.append(nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.q_ffn.append(nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)))
            self.q_norm1.append(nn.LayerNorm(d_model))
            self.q_norm2.append(nn.LayerNorm(d_model))
            self.q_norm3.append(nn.LayerNorm(d_model))

            self.r_self_attn.append(nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.r_cross_attn.append(nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.r_ffn.append(nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)))
            self.r_norm1.append(nn.LayerNorm(d_model))
            self.r_norm2.append(nn.LayerNorm(d_model))
            self.r_norm3.append(nn.LayerNorm(d_model))

        self.last_attn_q2r: Optional[torch.Tensor] = None
        self.last_attn_r2q: Optional[torch.Tensor] = None

    def forward(self, q_feat: torch.Tensor, r_feat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = q_feat.shape
        pe = self.pe(H, W, q_feat.device)

        q = q_feat.flatten(2).permute(0, 2, 1) + pe
        r = r_feat.flatten(2).permute(0, 2, 1) + pe
        q_res, r_res = q, r

        for i in range(self.n_layers):
            save_w = (i == self.n_layers - 1)
            # Q branch
            qn = self.q_norm1[i](q)
            q = q + self.q_self_attn[i](qn, qn, qn, need_weights=False)[0]
            qn = self.q_norm2[i](q)
            rn = self.r_norm2[i](r)
            out, w = self.q_cross_attn[i](qn, rn, rn, need_weights=save_w, average_attn_weights=True)
            q = q + out
            if save_w:
                self.last_attn_q2r = w.detach()
            q = q + self.q_ffn[i](self.q_norm3[i](q))

            # R branch
            rn = self.r_norm1[i](r)
            r = r + self.r_self_attn[i](rn, rn, rn, need_weights=False)[0]
            rn2 = self.r_norm2[i](r)
            qn2 = self.q_norm2[i](q)
            out, w = self.r_cross_attn[i](rn2, qn2, qn2, need_weights=save_w, average_attn_weights=True)
            r = r + out
            if save_w:
                self.last_attn_r2q = w.detach()
            r = r + self.r_ffn[i](self.r_norm3[i](r))

        gate = torch.sigmoid(self.gate)
        q = q_res + gate * (q - q_res)
        r = r_res + gate * (r - r_res)
        return q.permute(0, 2, 1).reshape(B, C, H, W), r.permute(0, 2, 1).reshape(B, C, H, W)


class ConvGRU(nn.Module):
    """Convolutional GRU with fused z+r gates."""

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
        self.conv_h = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = zr.chunk(2, dim=1)
        rh_x = torch.cat([r * h, x], dim=1)
        h_hat = torch.tanh(self.conv_h(rh_x))
        return (1 - z) * h + z * h_hat


class FlowRefinementHead(nn.Module):
    """Correlation → encoder → GRU update → flow delta + confidence."""

    def __init__(self, corr_channels: int, hidden_dim: int = 128,
                 conf_dim: int = 1, conf_floor: float = 0.0):
        super().__init__()
        self.conf_floor = conf_floor
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels + 2 + conf_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.gru = ConvGRU(hidden_dim, hidden_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2 + conf_dim, 3, padding=1),
        )

    def forward(self, corr, hidden, flow, confidence):
        inp = torch.cat([corr, flow, confidence], dim=1)
        inp_encoded = self.corr_encoder(inp)
        new_hidden = self.gru(hidden, inp_encoded)
        out = self.flow_head(new_hidden)
        delta_flow = out[:, :2]
        raw_conf = torch.sigmoid(out[:, 2:])
        if self.conf_floor > 0:
            new_conf = self.conf_floor + (1.0 - self.conf_floor) * raw_conf
        else:
            new_conf = raw_conf
        return delta_flow, new_conf, new_hidden, flow + delta_flow


class ContextAdapter(nn.Module):
    """Fuse upsampled hidden state with scale-specific query features."""

    def __init__(self, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + feat_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, h_up: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_up, q_feat], dim=1))


class ConvexUpsampler(nn.Module):
    """RAFT-style learned convex upsampling for flow + confidence."""

    def __init__(self, hidden_dim: int = 128, scale_factor: int = 4,
                 conf_channels: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        self.mask_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, scale_factor * scale_factor * 9, 1),
        )

    def forward(self, flow: torch.Tensor, conf: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = flow.shape
        s = self.scale_factor

        mask = self.mask_net(hidden)
        mask = mask.view(B, 1, s * s, 9, H, W)
        mask = torch.softmax(mask, dim=3)

        up_data = torch.cat([flow, conf], dim=1)
        C_total = up_data.shape[1]
        up_padded = F.pad(up_data, [1, 1, 1, 1], mode='replicate')
        up_unf = F.unfold(up_padded, kernel_size=3, padding=0)
        up_unf = up_unf.view(B, C_total, 9, H, W).unsqueeze(2)

        weighted = (mask * up_unf).sum(dim=3)
        weighted = weighted.view(B, C_total, s, s, H, W)
        weighted = weighted.permute(0, 1, 4, 2, 5, 3).contiguous()
        weighted = weighted.view(B, C_total, H * s, W * s)

        flow_up = weighted[:, :2].clone()
        conf_up = weighted[:, 2:]
        flow_up[:, 0] *= s
        flow_up[:, 1] *= s
        return flow_up, conf_up


class TranslationHead(nn.Module):
    """Direct translation regression from GRU hidden state (gradient-isolated)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(hidden))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class PoseRefiner(nn.Module):
    """
    Single-scale iterative correspondence-based pose estimator.

    Takes reference feature map + depth (from map) and query feature map,
    predicts 6-DOF relative pose via:
      1. Feature projection + cross-attention interaction
      2. Dense correlation → iterative GRU flow refinement
      3. Convex upsampling → high-res geometry solver with depth normalization
    """

    def __init__(
        self,
        in_dim: int = 64,
        match_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        ffn_dim: int = 128,
        local_radius: int = 4,
        fine_iters: int = 8,
        damping: float = 1e-3,
        coarse_hw: Tuple[int, int] = (17, 30),
        fine_hw: Tuple[int, int] = (34, 60),
        solver_upsample: int = 4,
        solver_hw: Optional[Tuple[int, int]] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        img_hw: Tuple[int, int] = (1080, 1920),
        depth_normalize: bool = True,
        sequential_solve: bool = True,
        detach_conf: bool = True,
        conf_floor: float = 0.1,
        use_trans_head: bool = False,
        trans_head_mode: str = 'replace',
        solver_trans_scale: float = 0.0,
        use_flow_scale_head: bool = False,
        raw_coarse_corr: bool = False,
    ):
        super().__init__()

        self.raw_coarse_corr = raw_coarse_corr
        self.COARSE_HW = tuple(coarse_hw)
        self.FINE_HW = tuple(fine_hw)
        # Solver resolution: explicit solver_hw takes priority, else fine_hw * upsample
        if solver_hw is not None:
            self.SOLVER_HW = tuple(solver_hw)
        else:
            self.SOLVER_HW = (fine_hw[0] * solver_upsample, fine_hw[1] * solver_upsample)
        self.IMG_HW = tuple(img_hw)
        self.BASE_INTRINSICS = intrinsics or {
            'fx': 1663.12, 'fy': 1663.12, 'cx': 960.0, 'cy': 540.0
        }
        self.local_radius = local_radius
        self.fine_iters = fine_iters
        self.damping = damping
        self.depth_normalize = depth_normalize
        self.sequential_solve = sequential_solve
        self.detach_conf = detach_conf
        self.conf_floor = conf_floor
        self.solver_trans_scale = solver_trans_scale
        self.use_trans_head = use_trans_head
        self.trans_head_mode = trans_head_mode
        self.update_damping = False  # scale delta_xi by mean confidence
        self.flow_scale = 1.0  # multiplicative scale applied to flow before solver
        self.use_flow_scale_head = use_flow_scale_head

        # Feature projection (shared for Q and R)
        self.projection = DomainProjection(in_dim, match_dim)

        # Cross-attention
        self.cross_attn = CrossAttention(match_dim, n_heads, n_attn_layers, ffn_dim)

        # Coarse pathway
        coarse_corr_ch = coarse_hw[0] * coarse_hw[1]
        self.context_net = nn.Sequential(
            nn.Conv2d(match_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.coarse_head = FlowRefinementHead(
            coarse_corr_ch, hidden_dim, conf_floor=conf_floor)

        # Fine pathway
        fine_corr_ch = (2 * local_radius + 1) ** 2
        self.fine_context = ContextAdapter(hidden_dim, match_dim)
        self.fine_head = FlowRefinementHead(
            fine_corr_ch, hidden_dim, conf_floor=conf_floor)

        # Convex upsampler: fine → solver resolution
        self.upsampler = ConvexUpsampler(hidden_dim, solver_upsample)

        # Optional translation head
        if use_trans_head:
            self.trans_head = TranslationHead(hidden_dim)

        # Optional learned flow scale head: predicts per-sample alpha ∈ (0.1, 1.0)
        if use_flow_scale_head:
            self.flow_scale_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),      # (B, hidden_dim, 1, 1)
                nn.Flatten(),                  # (B, hidden_dim)
                nn.Linear(hidden_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),                  # → (B, 1) in [0, 1]
            )

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
        query_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_feat: (B, C, H_in, W_in) query features
            ref_feat:   (B, C, H_in, W_in) reference features (rendered at current pose)
            depth:      (B, H_d, W_d) depth at reference pose (optional, needed for solver)

        Returns:
            dict with: flow_coarse, flow_fine, conf_fine, delta_xi,
                       fine_flow_preds, hidden_fine, attn_weights_*, etc.
        """
        B = query_feat.shape[0]
        device = query_feat.device

        # ── 1. Normalize + Downsample + Project per scale (v15 order) ──
        q = F.normalize(query_feat, dim=1)
        r = F.normalize(ref_feat, dim=1)

        q_coarse = self.projection(F.interpolate(q, self.COARSE_HW, mode='bilinear', align_corners=False))
        r_coarse = self.projection(F.interpolate(r, self.COARSE_HW, mode='bilinear', align_corners=False))
        q_fine = self.projection(F.interpolate(q, self.FINE_HW, mode='bilinear', align_corners=False))
        r_fine = self.projection(F.interpolate(r, self.FINE_HW, mode='bilinear', align_corners=False))

        # ── 2. Cross-attention at coarse resolution ──
        q_enhanced, r_enhanced = self.cross_attn(q_coarse, r_coarse)

        # ── 3. Coarse: global correlation → single GRU step ──
        if self.raw_coarse_corr:
            coarse_corr = global_correlation(q_coarse, r_coarse)
        else:
            coarse_corr = global_correlation(q_enhanced, r_enhanced)
        h_coarse = self.context_net(q_coarse)
        flow_c = torch.zeros(B, 2, *self.COARSE_HW, device=device)
        conf_c = torch.ones(B, 1, *self.COARSE_HW, device=device) * 0.5

        _, conf_c, h_coarse, flow_c = self.coarse_head(
            coarse_corr, h_coarse, flow_c, conf_c)

        # ── 4. Fine: upsample + guided local correlation × N iters ──
        scale_x = self.FINE_HW[1] / self.COARSE_HW[1]
        scale_y = self.FINE_HW[0] / self.COARSE_HW[0]
        flow_f = F.interpolate(flow_c, self.FINE_HW, mode='bilinear', align_corners=False)
        flow_f[:, 0] *= scale_x
        flow_f[:, 1] *= scale_y
        conf_f = F.interpolate(conf_c, self.FINE_HW, mode='bilinear', align_corners=False)

        h_fine_up = F.interpolate(h_coarse, self.FINE_HW, mode='bilinear', align_corners=False)
        h_fine = self.fine_context(h_fine_up, q_fine)

        fine_flow_preds = []
        for _ in range(self.fine_iters):
            fine_corr = guided_local_correlation(
                q_fine, r_fine, flow_f, radius=self.local_radius)
            _, conf_f, h_fine, flow_f = self.fine_head(
                fine_corr, h_fine, flow_f, conf_f)
            fine_flow_preds.append(flow_f)

        result = {
            'flow_coarse': flow_c,
            'conf_coarse': conf_c,
            'flow_fine': flow_f,
            'conf_fine': conf_f,
            'fine_flow_preds': fine_flow_preds,
            'hidden_fine': h_fine,
            'attn_weights_q2r': self.cross_attn.last_attn_q2r,
            'attn_weights_r2q': self.cross_attn.last_attn_r2q,
            'q_coarse_proj': q_coarse,
            'r_coarse_proj': r_coarse,
        }

        # ── 5. Geometry Solver ──
        if depth is not None:
            with torch.cuda.amp.autocast(enabled=False):
                flow_f32 = flow_f.float()
                conf_f32 = conf_f.float()
                depth_f32 = depth.float()

                # Apply flow scale (test-time scaling for flow magnitude adjustment)
                if self.use_flow_scale_head:
                    # Learned per-sample flow scale from hidden state
                    alpha = self.flow_scale_predictor(h_fine.float())  # (B, 1)
                    alpha = 0.1 + 0.9 * alpha  # remap to [0.1, 1.0]
                    flow_f32 = flow_f32 * alpha.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                    result['flow_scale_alpha'] = alpha.squeeze(-1)  # (B,) for logging
                elif self.flow_scale != 1.0:
                    flow_f32 = flow_f32 * self.flow_scale

                # Bilinear upsample flow/conf to solver resolution (v15 approach)
                sH, sW = self.SOLVER_HW
                if (sH, sW) != self.FINE_HW:
                    scale_x = sW / self.FINE_HW[1]
                    scale_y = sH / self.FINE_HW[0]
                    solve_flow = F.interpolate(
                        flow_f32, (sH, sW), mode='bilinear', align_corners=False)
                    solve_flow[:, 0] *= scale_x
                    solve_flow[:, 1] *= scale_y
                    solve_conf = F.interpolate(
                        conf_f32, (sH, sW), mode='bilinear', align_corners=False)
                else:
                    solve_flow = flow_f32
                    solve_conf = conf_f32

                # Interpolate depth to solver resolution
                if depth_f32.ndim == 3:
                    depth_f32 = depth_f32.unsqueeze(1)
                depth_solve = F.interpolate(
                    depth_f32, (sH, sW),
                    mode='bilinear', align_corners=False).squeeze(1)

                # Depth normalization (only useful when translation is active)
                depth_median = torch.ones(1, device=device)
                if self.depth_normalize and self.solver_trans_scale > 0:
                    valid_depths = depth_solve[depth_solve > 0.05]
                    if valid_depths.numel() > 0:
                        depth_median = valid_depths.median().clamp(min=1.0)
                    depth_solve = depth_solve / depth_median

                # Image Jacobian
                solver_intrinsics = self._scale_intrinsics(sH, sW)
                Ju, Jv, valid = compute_image_jacobian(depth_solve, solver_intrinsics)

                # Detach confidence to prevent pose loss → confidence collapse
                conf_detached = solve_conf.detach() if self.detach_conf else solve_conf

                # Weighted least-squares solve
                if self.sequential_solve:
                    delta_xi = diff_pose_solve_sequential(
                        solve_flow, conf_detached, Ju, Jv, valid, damping=self.damping)
                else:
                    delta_xi = diff_pose_solve(
                        solve_flow, conf_detached, Ju, Jv, valid, damping=self.damping)

                # Rescale translation after depth normalization
                if self.depth_normalize and self.solver_trans_scale > 0:
                    delta_xi = torch.cat([
                        delta_xi[:, :3] * depth_median,
                        delta_xi[:, 3:],
                    ], dim=1)

                # Confidence-proportional update damping
                # When confidence is low → small update, prevents overshoot
                if self.update_damping:
                    conf_mean = solve_conf.mean(dim=(1, 2, 3), keepdim=False)  # (B,)
                    damping_scale = conf_mean.clamp(0.1, 1.0).unsqueeze(1)  # (B, 1)
                    delta_xi = delta_xi * damping_scale

                # Scale translation (0.0 = disable, 1.0 = full)
                if self.solver_trans_scale != 1.0:
                    delta_xi = torch.cat([
                        delta_xi[:, :3] * self.solver_trans_scale,
                        delta_xi[:, 3:],
                    ], dim=1)

                # Optional: replace/augment translation with MLP head
                if self.use_trans_head:
                    delta_t = self.trans_head(h_fine.detach().float())
                    if self.trans_head_mode == 'replace':
                        delta_xi = torch.cat([delta_t, delta_xi[:, 3:]], dim=1)
                    else:  # 'residual'
                        delta_xi = torch.cat([
                            delta_xi[:, :3] + delta_t, delta_xi[:, 3:]], dim=1)

                result['delta_xi'] = delta_xi

        return result

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GT flow from pose difference and depth for supervision.

        Args:
            pose_init: (B, 4, 4) current w2c pose
            pose_gt:   (B, 4, 4) GT w2c pose
            depth:     (B, H, W) depth at pose_init viewpoint
            target_hw: (tH, tW) target flow resolution

        Returns:
            flow_gt:    (B, 2, tH, tW) ground-truth optical flow
            valid_mask: (B, 1, tH, tW) validity mask
        """
        B, H, W = depth.shape
        device = depth.device
        intrinsics = self._scale_intrinsics(H, W)
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij')
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth
        Y = (v_coords - cy) / fy * depth
        Z = depth

        pts = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = pose_gt @ torch.linalg.inv(pose_init)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)

        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([
            u_gt - u_coords.unsqueeze(1),
            v_gt - v_coords.unsqueeze(1),
        ], dim=1)

        valid = (
            (depth.unsqueeze(1) > 0.05) &
            (Z_gt_raw > 0.1) &
            (u_gt > -0.5) & (u_gt < W - 0.5) &
            (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()

        flow_gt = flow_gt * valid

        tH, tW = target_hw
        if tH != H or tW != W:
            sx, sy = tW / W, tH / H
            flow_gt = F.interpolate(flow_gt, (tH, tW), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= sx
            flow_gt[:, 1] *= sy
            valid = F.interpolate(valid, (tH, tW), mode='nearest')

        return flow_gt, valid
