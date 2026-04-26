"""
RadioPoseNet: Transformer Cross-Attention + Correlation + GRU Pose Estimation
=============================================================================

为 RADIO 64d 特征 + 2DGS 渲染设计的位姿估计网络。

Architecture:
  1. FeatureAdapter: 64d → 64d 特征域对齐 (shared Q/R)
  2. Coarse (17×30): Transformer cross-attention 特征交互
     → global correlation → FlowRefinementHead (GRU 1 iter) → flow_coarse
     - 保存注意力图用于 heatmap 可视化
  3. Fine (34×60): upsample flow → guided local correlation
     → GRU × N iters → flow_fine + confidence
  4. Geometry Solver: Image Jacobian + WLS → Δξ (se3)

Input:
  - query_feat:  RADIO PCA 64d @ 68×120 (from image encoder)
  - render_feat: 2DGS rendered 64d @ any resolution (from gaussian feature model)
  - depth:       2DGS rendered depth

Key design choices:
  - Transformer cross-attention replaces simple dot-product at coarse level,
    providing learned feature selection (vs. fixed inner product)
  - Attention weights stored for heatmap visualization (feature selection interpretability)
  - GRU iterative updates at fine level (RAFT-style, proven effective)
  - Geometry solver (Image Jacobian + WLS) for physically-grounded pose estimation

Estimated params: ~1.2M
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve


# ==============================================================================
#  Building Blocks
# ==============================================================================

class SinusoidalPE2D(nn.Module):
    """2D sinusoidal positional encoding with caching."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self._cache: Dict[Tuple, torch.Tensor] = {}

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key in self._cache:
            return self._cache[key]
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
        result = pe.reshape(1, H * W, d)
        self._cache[key] = result
        return result


class FeatureAdapter(nn.Module):
    """Lightweight shared adapter (legacy, kept for checkpoint compat)."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class DomainProjection(nn.Module):
    """Domain-specific feature projection for matching.
    
    Separate projections for query and reference features allow each
    domain to learn its own mapping into a discriminative matching space.
    Combined with contrastive loss, this prevents the feature collapse
    that occurs with shared adapters.
    
    Output is L2-normalized to keep correlation values bounded.
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class CrossAttentionBlock(nn.Module):
    """
    Symmetric Transformer cross-attention for Q↔R feature interaction.

    Stores attention weights for heatmap visualization.
    Uses pre-norm architecture + residual gating (warmstart-safe).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.pe = SinusoidalPE2D(d_model)

        # Q: self-attention + cross-attention + FFN
        self.q_self_attn = nn.ModuleList()
        self.q_cross_attn = nn.ModuleList()
        self.q_ffn = nn.ModuleList()
        self.q_norm1 = nn.ModuleList()
        self.q_norm2 = nn.ModuleList()
        self.q_norm3 = nn.ModuleList()

        # R: self-attention + cross-attention + FFN (symmetric)
        self.r_self_attn = nn.ModuleList()
        self.r_cross_attn = nn.ModuleList()
        self.r_ffn = nn.ModuleList()
        self.r_norm1 = nn.ModuleList()
        self.r_norm2 = nn.ModuleList()
        self.r_norm3 = nn.ModuleList()

        for _ in range(n_layers):
            self.q_self_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.q_cross_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.q_ffn.append(nn.Sequential(
                nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)))
            self.q_norm1.append(nn.LayerNorm(d_model))
            self.q_norm2.append(nn.LayerNorm(d_model))
            self.q_norm3.append(nn.LayerNorm(d_model))

            self.r_self_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.r_cross_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.r_ffn.append(nn.Sequential(
                nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)))
            self.r_norm1.append(nn.LayerNorm(d_model))
            self.r_norm2.append(nn.LayerNorm(d_model))
            self.r_norm3.append(nn.LayerNorm(d_model))

        # Residual gate: starts at 0 → Transformer does nothing initially (warmstart-safe)
        self.gate = nn.Parameter(torch.tensor(0.0))

        # Storage for attention maps (populated during forward)
        self.last_attn_weights_q2r: Optional[torch.Tensor] = None  # Q attends to R
        self.last_attn_weights_r2q: Optional[torch.Tensor] = None  # R attends to Q

    def forward(
        self, q_feat: torch.Tensor, r_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q_feat: (B, C, H, W) query features
            r_feat: (B, C, H, W) reference features
        Returns:
            q_out: (B, C, H, W) enhanced query
            r_out: (B, C, H, W) enhanced reference
        """
        B, C, H, W = q_feat.shape

        q_tokens = q_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        r_tokens = r_feat.flatten(2).permute(0, 2, 1)

        pe = self.pe(H, W, q_feat.device)
        q_tokens = q_tokens + pe
        r_tokens = r_tokens + pe

        q_res = q_tokens
        r_res = r_tokens

        for i in range(self.n_layers):
            # Q self-attention
            qn = self.q_norm1[i](q_tokens)
            q_tokens = q_tokens + self.q_self_attn[i](qn, qn, qn, need_weights=False)[0]

            # Q cross-attention (Q attends to R) — save weights for heatmap
            qn = self.q_norm2[i](q_tokens)
            rn = self.r_norm2[i](r_tokens)  # use R's norm for keys
            q_cross, attn_w = self.q_cross_attn[i](
                qn, rn, rn, need_weights=True, average_attn_weights=True)
            q_tokens = q_tokens + q_cross

            # Q FFN
            qn = self.q_norm3[i](q_tokens)
            q_tokens = q_tokens + self.q_ffn[i](qn)

            # R self-attention
            rn = self.r_norm1[i](r_tokens)
            r_tokens = r_tokens + self.r_self_attn[i](rn, rn, rn, need_weights=False)[0]

            # R cross-attention (R attends to Q) — save weights for heatmap
            rn2 = self.r_norm2[i](r_tokens)
            qn2 = self.q_norm2[i](q_tokens)
            r_cross, attn_w_r2q = self.r_cross_attn[i](
                rn2, qn2, qn2, need_weights=True, average_attn_weights=True)
            r_tokens = r_tokens + r_cross

            # R FFN
            rn = self.r_norm3[i](r_tokens)
            r_tokens = r_tokens + self.r_ffn[i](rn)

        # Store last-layer attention weights for visualization
        self.last_attn_weights_q2r = attn_w.detach()       # (B, N_q, N_r)
        self.last_attn_weights_r2q = attn_w_r2q.detach()   # (B, N_r, N_q)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        q_out = q_res + gate * (q_tokens - q_res)
        r_out = r_res + gate * (r_tokens - r_res)

        q_out = q_out.permute(0, 2, 1).reshape(B, C, H, W)
        r_out = r_out.permute(0, 2, 1).reshape(B, C, H, W)
        return q_out, r_out


class ConvGRU(nn.Module):
    """Convolutional GRU with fused z+r gates (from RAFT)."""

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
    """Per-scale: correlation → encoder → GRU update → flow + confidence."""

    def __init__(self, corr_channels: int, hidden_dim: int = 128, conf_dim: int = 1,
                 conf_floor: float = 0.0):
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
        # Apply confidence floor to prevent pathological collapse
        if self.conf_floor > 0:
            new_conf = self.conf_floor + (1.0 - self.conf_floor) * raw_conf
        else:
            new_conf = raw_conf
        return delta_flow, new_conf, new_hidden, flow + delta_flow


class ContextAdapter(nn.Module):
    """Upsample hidden state + inject scale-specific query features."""

    def __init__(self, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + feat_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, hidden_up: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([hidden_up, q_feat], dim=1))


# ==============================================================================
#  Correlation Functions
# ==============================================================================

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
    Warps reference features by current flow, then computes local dot products.
    """
    B, C, H, W = fmap_q.shape
    d = 2 * radius + 1

    # Create sampling grid from flow
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=torch.float32),
        torch.arange(W, device=flow.device, dtype=torch.float32),
        indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    coords = grid + flow  # (B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    coords_norm = torch.stack([
        2.0 * coords[:, 0] / max(W - 1, 1) - 1.0,
        2.0 * coords[:, 1] / max(H - 1, 1) - 1.0,
    ], dim=-1)  # (B, H, W, 2)

    fmap_r_warped = F.grid_sample(
        fmap_r, coords_norm, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Local correlation via unfold
    fmap_r_pad = F.pad(fmap_r_warped, [radius] * 4, mode='constant', value=0)
    fmap_r_unfold = fmap_r_pad.unfold(2, d, 1).unfold(3, d, 1)  # (B, C, H, W, d, d)
    fmap_r_unfold = fmap_r_unfold.reshape(B, C, H, W, d * d)

    fmap_q_exp = fmap_q.reshape(B, C, H, W, 1)
    corr = (fmap_q_exp * fmap_r_unfold).sum(dim=1)  # (B, H, W, d*d)
    return corr.permute(0, 3, 1, 2)  # (B, d*d, H, W)


# ==============================================================================
#  Main Network
# ==============================================================================

class RadioPoseNet(nn.Module):
    """
    Transformer Cross-Attention + Correlation + GRU Pose Estimation
    for RADIO 64d features with 2DGS rendering.

    Two-scale architecture:
      Coarse (17×30): Transformer cross-attention → global correlation → GRU → flow
      Fine (34×60):   upsample flow → guided local correlation → GRU × N → flow + conf
      Geometry:       Image Jacobian + WLS → Δξ (se3)

    Args:
        feat_dim:      input/working feature dimension (64 for RADIO PCA)
        hidden_dim:    GRU hidden state dimension
        n_heads:       Transformer attention heads
        n_attn_layers: number of Transformer cross-attention layers
        ffn_dim:       Transformer FFN hidden dimension
        local_radius:  local correlation search radius
        fine_iters:    GRU iterations at fine scale
        damping:       LM geometry solver damping
        coarse_hw:     coarse resolution (H, W)
        fine_hw:       fine resolution (H, W)
        intrinsics:    camera intrinsics dict {fx, fy, cx, cy}
        img_hw:        original input image size (H, W) for intrinsic scaling
    """

    DEFAULT_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    DEFAULT_IMG_HW = (480, 640)

    def __init__(
        self,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        ffn_dim: int = 128,
        local_radius: int = 4,
        fine_iters: int = 4,
        damping: float = 1e-3,
        coarse_hw: Tuple[int, int] = (17, 30),
        fine_hw: Tuple[int, int] = (34, 60),
        intrinsics: Optional[Dict[str, float]] = None,
        img_hw: Optional[Tuple[int, int]] = None,
        irls_iters: int = 0,
        irls_huber_k: float = 1.345,
        robust_kernel: str = 'huber',
        conf_floor: float = 0.0,
        detach_conf_in_solver: bool = False,
        solver_hw: Optional[Tuple[int, int]] = None,
        sequential_solve: bool = False,
        solver_trans_scale: float = 1.0,
        shared_projection: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.local_radius = local_radius
        self.fine_iters = fine_iters
        self.damping = damping
        self.irls_iters = irls_iters
        self.irls_huber_k = irls_huber_k
        self.robust_kernel = robust_kernel
        self.detach_conf_in_solver = detach_conf_in_solver
        self.SOLVER_HW = tuple(solver_hw) if solver_hw else None
        self.sequential_solve = sequential_solve
        self.solver_trans_scale = solver_trans_scale
        self.COARSE_HW = tuple(coarse_hw)
        self.FINE_HW = tuple(fine_hw)
        self.BASE_INTRINSICS = intrinsics or self.DEFAULT_INTRINSICS.copy()
        self.IMG_HW = img_hw or self.DEFAULT_IMG_HW

        d = 2 * local_radius + 1
        local_corr_ch = d * d  # 81

        # ── Domain projections for Q and R ──
        # shared_projection=True: same network for both domains (guarantees alignment)
        # shared_projection=False: separate networks (can diverge if contrastive loss fails)
        self.proj_q = DomainProjection(feat_dim, feat_dim)
        self.proj_r = self.proj_q if shared_projection else DomainProjection(feat_dim, feat_dim)
        # Legacy adapter kept for checkpoint compatibility
        self.adapter = FeatureAdapter(feat_dim, feat_dim)

        # ── Transformer cross-attention at coarse level ──
        self.cross_attn = CrossAttentionBlock(
            d_model=feat_dim, n_heads=n_heads,
            n_layers=n_attn_layers, ffn_dim=ffn_dim)

        # ── Coarse flow head: global correlation → GRU ──
        coarse_global_ch = coarse_hw[0] * coarse_hw[1]  # 17*30 = 510
        self.coarse_head = FlowRefinementHead(
            corr_channels=coarse_global_ch, hidden_dim=hidden_dim,
            conf_floor=conf_floor)

        # ── Fine flow head: local correlation → GRU (iterative) ──
        self.fine_head = FlowRefinementHead(
            corr_channels=local_corr_ch, hidden_dim=hidden_dim,
            conf_floor=conf_floor)

        # ── Context encoder: query features → initial GRU hidden ──
        self.context_net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # ── Fine context: upsample coarse hidden + inject fine query ──
        self.fine_context = ContextAdapter(hidden_dim, feat_dim)

        # Intrinsics at fine resolution
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        """Scale base intrinsics to target resolution."""
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    def _upsample_flow(self, flow, conf, target_hw):
        _, _, sH, sW = flow.shape
        tH, tW = target_hw
        flow_up = F.interpolate(flow, size=target_hw, mode='bilinear', align_corners=False)
        flow_up[:, 0] *= tW / sW
        flow_up[:, 1] *= tH / sH
        conf_up = F.interpolate(conf, size=target_hw, mode='bilinear', align_corners=False)
        return flow_up, conf_up

    def forward(
        self,
        query_feat: torch.Tensor,
        render_feat: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_feat:  (B, 64, H_in, W_in) RADIO PCA features
            render_feat: (B, 64, H_in, W_in) 2DGS rendered features
            depth:       (B, H_d, W_d) rendered depth map

        Returns:
            dict with:
              'delta_xi': (B, 6) se3 pose update (if depth given)
              'flow_coarse': (B, 2, cH, cW) coarse flow
              'flow_fine': (B, 2, fH, fW) final fine flow
              'conf_fine': (B, 1, fH, fW) flow confidence
              'fine_flow_preds': list of flow per GRU iteration
              'attn_weights': (B, cH*cW, cH*cW) cross-attention heatmap
        """
        B = query_feat.shape[0]
        device = query_feat.device

        # ══════════════════════════════════════════════
        #  1. Normalize + Downsample + Project features
        # ══════════════════════════════════════════════
        # L2 normalize to unit vectors: eliminates scale mismatch between
        # query RADIO features (range ~[-20,20]) and 2DGS rendered features (range ~[-1,1])
        query_feat = F.normalize(query_feat, dim=1)
        render_feat = F.normalize(render_feat, dim=1)

        q_coarse = F.interpolate(query_feat, self.COARSE_HW, mode='bilinear', align_corners=False)
        r_coarse = F.interpolate(render_feat, self.COARSE_HW, mode='bilinear', align_corners=False)
        q_fine = F.interpolate(query_feat, self.FINE_HW, mode='bilinear', align_corners=False)
        r_fine = F.interpolate(render_feat, self.FINE_HW, mode='bilinear', align_corners=False)

        # Separate domain projections → L2-normalized matching space
        q_coarse_p = self.proj_q(q_coarse)
        r_coarse_p = self.proj_r(r_coarse)
        q_fine_p = self.proj_q(q_fine)
        r_fine_p = self.proj_r(r_fine)

        # ══════════════════════════════════════════════
        #  2. Coarse: Transformer + Global Correlation → GRU
        # ══════════════════════════════════════════════
        # Transformer feature interaction (saves attention weights)
        q_enhanced, r_enhanced = self.cross_attn(q_coarse_p, r_coarse_p)

        # Global correlation on transformer-enhanced features
        coarse_corr = global_correlation(q_enhanced, r_enhanced)

        # Context → initial hidden
        h_coarse = self.context_net(q_coarse_p)

        # Zero flow + uniform confidence
        flow_c = torch.zeros(B, 2, *self.COARSE_HW, device=device)
        conf_c = torch.ones(B, 1, *self.COARSE_HW, device=device) * 0.5

        _, conf_c, h_coarse, flow_c = self.coarse_head(
            coarse_corr, h_coarse, flow_c, conf_c)

        # ══════════════════════════════════════════════
        #  3. Fine: Guided Local Correlation + GRU Iterations
        # ══════════════════════════════════════════════
        flow_f, conf_f = self._upsample_flow(flow_c, conf_c, self.FINE_HW)

        h_fine_up = F.interpolate(h_coarse, self.FINE_HW, mode='bilinear', align_corners=False)
        h_fine = self.fine_context(h_fine_up, q_fine_p)

        fine_flow_preds = []
        for _ in range(self.fine_iters):
            fine_corr = guided_local_correlation(
                q_fine_p, r_fine_p, flow_f, radius=self.local_radius)
            _, conf_f, h_fine, flow_f = self.fine_head(
                fine_corr, h_fine, flow_f, conf_f)
            fine_flow_preds.append(flow_f)

        # ══════════════════════════════════════════════
        #  4. Geometry Solver
        # ══════════════════════════════════════════════
        result = {
            'flow_coarse': flow_c,
            'conf_coarse': conf_c,
            'flow_fine': flow_f,
            'conf_fine': conf_f,
            'fine_flow_preds': fine_flow_preds,
            'hidden_fine': h_fine,
            'attn_weights_q2r': self.cross_attn.last_attn_weights_q2r,
            'attn_weights_r2q': self.cross_attn.last_attn_weights_r2q,
            # Expose projected features for contrastive loss
            'q_coarse_proj': q_coarse_p,
            'r_coarse_proj': r_coarse_p,
        }

        if depth is not None:
            with torch.amp.autocast('cuda', enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow_f.float()
                conf_f32 = conf_f.float()

                # Optionally upsample flow/conf to higher resolution for
                # the geometry solver — improves translation sensitivity
                # by increasing effective focal length.
                if self.SOLVER_HW is not None:
                    sH, sW = self.SOLVER_HW
                    scale_x = sW / flow_f32.shape[-1]
                    scale_y = sH / flow_f32.shape[-2]
                    flow_f32 = F.interpolate(
                        flow_f32, size=(sH, sW),
                        mode='bilinear', align_corners=False)
                    flow_f32[:, 0] *= scale_x
                    flow_f32[:, 1] *= scale_y
                    conf_f32 = F.interpolate(
                        conf_f32, size=(sH, sW),
                        mode='bilinear', align_corners=False)
                    solver_intrinsics = self._scale_intrinsics(sH, sW)
                else:
                    solver_intrinsics = self.fine_intrinsics

                if depth_f32.ndim == 3:
                    depth_f32 = depth_f32.unsqueeze(1)
                if depth_f32.shape[-2:] != flow_f32.shape[-2:]:
                    depth_f32 = F.interpolate(
                        depth_f32, size=flow_f32.shape[-2:],
                        mode='bilinear', align_corners=False)
                depth_f32 = depth_f32.squeeze(1)

                Ju, Jv, valid = compute_image_jacobian(
                    depth_f32, solver_intrinsics)

                # Detach confidence in geometry solver to prevent
                # pose loss from driving confidence → 0.  Confidence
                # is trained only via calibration / regularization loss.
                conf_detached = conf_f32.detach() if self.detach_conf_in_solver else conf_f32

                if self.sequential_solve:
                    from modules.geometry_solver import diff_pose_solve_sequential
                    delta_xi = diff_pose_solve_sequential(
                        flow_f32, conf_detached, Ju, Jv, valid,
                        damping=self.damping,
                    )
                else:
                    delta_xi = diff_pose_solve(
                        flow_f32, conf_detached, Ju, Jv, valid,
                        damping=self.damping,
                        irls_iters=self.irls_iters,
                        irls_huber_k=self.irls_huber_k,
                        robust_kernel=self.robust_kernel,
                    )

            # Scale down translation to prevent overshoot at large depths
            if self.solver_trans_scale != 1.0:
                delta_xi = torch.cat([
                    delta_xi[:, :3] * self.solver_trans_scale,
                    delta_xi[:, 3:],
                ], dim=1)

            result['delta_xi'] = delta_xi

        return result

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GT flow from pose difference and depth for supervision."""
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

        pts_init = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts_init.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = pose_gt @ torch.linalg.inv(pose_init)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)

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
            flow_gt = F.interpolate(flow_gt, size=(tH, tW), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= scale_x
            flow_gt[:, 1] *= scale_y
            valid = F.interpolate(valid, size=(tH, tW), mode='nearest')

        return flow_gt, valid
