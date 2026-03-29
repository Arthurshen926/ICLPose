"""
TransformerPoseNetV2: LoFTR-style Transformer Pose Estimation
=============================================================
改进版 Transformer 位姿估计网络，解决 V1 发散问题。

V1 发散原因:
  1. L2 normalization 杀死 transformer 学习能力
  2. 仅 Q tokens 有 self-attention, R tokens 静态不更新
  3. 单向 cross-attention (Q→R, 没有 R→Q)
  4. 参数量不足 (0.51M vs MSFlowPoseNet 的 4.17M)
  5. MLP flow head 无空间结构

V2 改进:
  1. GroupNorm 替代 L2 norm — 保留幅值信息
  2. LoFTR 风格双向注意力: self(Q) + self(R) + cross(Q→R) + cross(R→Q)
  3. Conv-based flow decoder + 可选 GRU 迭代细化
  4. 更大容量: ~2M params
  5. 可学习 2D 位置编码

架构:
  FeatureEncoder(64→D, no L2) → Downsample(attn_hw) 
  → Learnable PE
  → N × LoFTRBlock: [self(Q) + self(R) + cross(Q↔R) + FFN]
  → Reshape(B, D, aH, aW) → ConvFlowDecoder → flow(2) + conf(1)
  → Upsample(fine_hw) → Geometry Solver → Δξ(6)

参数量 (默认 D=128, 6 layers):
  Encoder: ~25K
  PE: ~273K
  LoFTR blocks: ~2.1M
  Flow decoder: ~150K
  Total: ~2.55M
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

class FeatureEncoder(nn.Module):
    """Project input features WITHOUT L2 normalization.
    
    Uses GroupNorm for stability instead of L2 norm — preserves magnitude
    information essential for transformer attention scaling.
    """

    def __init__(self, in_dim: int, out_dim: int, mid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoFTRBlock(nn.Module):
    """LoFTR-style attention block with bidirectional cross-attention.
    
    每个 block 包含:
      1. Self-Attention(Q) — Q tokens 建立空间上下文
      2. Self-Attention(R) — R tokens 建立空间上下文  
      3. Cross-Attention(Q→R) — Q 查询 R 的对应关系
      4. Cross-Attention(R→Q) — R 查询 Q 的对应关系 (双向)
      5. FFN(Q) + FFN(R) — 非线性变换
      
    所有注意力使用 pre-norm (LayerNorm before attention).
    Q 和 R tokens 都会被更新.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Self-attention for Q
        self.norm_sq = nn.LayerNorm(d_model)
        self.self_attn_q = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Self-attention for R
        self.norm_sr = nn.LayerNorm(d_model)
        self.self_attn_r = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention Q→R
        self.norm_cq = nn.LayerNorm(d_model)
        self.norm_cr_for_cq = nn.LayerNorm(d_model)
        self.cross_attn_qr = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention R→Q
        self.norm_cr = nn.LayerNorm(d_model)
        self.norm_cq_for_cr = nn.LayerNorm(d_model)
        self.cross_attn_rq = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        
        # FFN for Q
        self.norm_fq = nn.LayerNorm(d_model)
        self.ffn_q = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        
        # FFN for R
        self.norm_fr = nn.LayerNorm(d_model)
        self.ffn_r = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,   # (B, N, D)
        r: torch.Tensor,   # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Self-attention on Q
        qn = self.norm_sq(q)
        q = q + self.self_attn_q(qn, qn, qn, need_weights=False)[0]
        
        # 2. Self-attention on R
        rn = self.norm_sr(r)
        r = r + self.self_attn_r(rn, rn, rn, need_weights=False)[0]
        
        # 3. Cross-attention Q→R (Q queries, R keys/values)
        qn = self.norm_cq(q)
        rn = self.norm_cr_for_cq(r)
        q = q + self.cross_attn_qr(qn, rn, rn, need_weights=False)[0]
        
        # 4. Cross-attention R→Q (R queries, Q keys/values)
        rn = self.norm_cr(r)
        qn = self.norm_cq_for_cr(q)
        r = r + self.cross_attn_rq(rn, qn, qn, need_weights=False)[0]
        
        # 5. FFN
        q = q + self.ffn_q(self.norm_fq(q))
        r = r + self.ffn_r(self.norm_fr(r))
        
        return q, r


class ConvFlowDecoder(nn.Module):
    """Conv-based flow decoder for spatial coherence.
    
    Takes fused Q+R features in 2D, outputs flow(2) + confidence(1).
    Uses 3×3 convolutions to maintain spatial structure.
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 3, 3, padding=1),  # flow(2) + conf(1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        flow = out[:, :2]
        conf = torch.sigmoid(out[:, 2:3])
        return flow, conf


# ==============================================================================
#  Main Network
# ==============================================================================

class TransformerPoseNetV2(nn.Module):
    """
    LoFTR-style Transformer for single-scale flow-based pose estimation.
    
    核心改进:
      - 双向 cross-attention (Q↔R)
      - 双方向都有 self-attention
      - 无 L2 normalization
      - Conv-based flow decoder
      - 更大容量 (~2.5M params)
    
    前向流程:
      1. Encode query/render features → d_model dim (no L2 norm)
      2. Downsample to attn_hw for efficient attention
      3. Add learnable 2D positional encoding
      4. N × LoFTR blocks (bidirectional self + cross attention)
      5. Fuse Q and R → Conv flow decoder → flow + confidence
      6. Upsample to fine_hw
      7. Geometry solver → Δξ (6-DOF pose update)
    """

    DEFAULT_INTRINSICS = {'fx': 320.0, 'fy': 320.0, 'cx': 319.5, 'cy': 239.5}
    DEFAULT_IMG_HW = (480, 640)

    def __init__(
        self,
        d_model: int = 128,
        feat_in_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 6,
        ffn_dim: int = 256,
        dropout: float = 0.0,
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
        self.d_model = d_model
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

        # ── Feature encoder (shared for Q and R, NO L2 norm) ──
        self.encoder = FeatureEncoder(feat_in_dim, d_model, mid_dim=max(d_model, 128))

        # ── Learnable 2D positional encoding ──
        aH, aW = self.ATTN_HW
        self.pos_embed = nn.Parameter(torch.randn(1, aH * aW, d_model) * 0.02)

        # ── LoFTR blocks ──
        self.blocks = nn.ModuleList([
            LoFTRBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # ── Final norm ──
        self.final_norm_q = nn.LayerNorm(d_model)
        self.final_norm_r = nn.LayerNorm(d_model)

        # ── Conv flow decoder ──
        # Input: concatenation of Q and element-wise product Q*R
        self.flow_decoder = ConvFlowDecoder(d_model * 2, hidden=d_model)

        # ── Intrinsics at fine resolution ──
        self.fine_intrinsics = self._scale_intrinsics(*self.FINE_HW)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize flow decoder final layer to near-zero
        # so initial predictions are small (near-zero flow)
        final_conv = self.flow_decoder.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

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
                'fine_flow_preds': [flow_fine] (single-element list for compatibility)
        """
        q_raw = query_feats['fine']
        r_raw = render_feats['fine']
        B = q_raw.shape[0]
        device = q_raw.device
        aH, aW = self.ATTN_HW

        # ══════════════════════════════════════════════
        #  1. Encode features (NO L2 normalization)
        # ══════════════════════════════════════════════
        q_feat = self.encoder(q_raw)
        r_feat = self.encoder(r_raw)

        # ══════════════════════════════════════════════
        #  2. Downsample to attention resolution
        # ══════════════════════════════════════════════
        q_attn = self._match_spatial(q_feat, self.ATTN_HW)  # (B, D, aH, aW)
        r_attn = self._match_spatial(r_feat, self.ATTN_HW)

        # ══════════════════════════════════════════════
        #  3. Tokenize + learnable positional encoding
        # ══════════════════════════════════════════════
        q_tok = q_attn.flatten(2).permute(0, 2, 1)  # (B, N, D)
        r_tok = r_attn.flatten(2).permute(0, 2, 1)
        q_tok = q_tok + self.pos_embed
        r_tok = r_tok + self.pos_embed

        # ══════════════════════════════════════════════
        #  4. LoFTR blocks (bidirectional attention)
        # ══════════════════════════════════════════════
        for block in self.blocks:
            q_tok, r_tok = block(q_tok, r_tok)

        # Final norm
        q_tok = self.final_norm_q(q_tok)
        r_tok = self.final_norm_r(r_tok)

        # ══════════════════════════════════════════════
        #  5. Reshape to 2D + flow decode
        # ══════════════════════════════════════════════
        q_2d = q_tok.permute(0, 2, 1).reshape(B, self.d_model, aH, aW)
        r_2d = r_tok.permute(0, 2, 1).reshape(B, self.d_model, aH, aW)

        # Fuse: concat [Q, Q*R] — element-wise product captures matching score
        fused = torch.cat([q_2d, q_2d * r_2d], dim=1)  # (B, 2D, aH, aW)
        flow_attn, conf_attn = self.flow_decoder(fused)

        # ══════════════════════════════════════════════
        #  6. Upsample to fine resolution
        # ══════════════════════════════════════════════
        fH, fW = self.FINE_HW
        flow_fine = F.interpolate(flow_attn, size=(fH, fW), mode='bilinear', align_corners=False)
        flow_fine = flow_fine.clone()
        flow_fine[:, 0] *= fW / aW
        flow_fine[:, 1] *= fH / aH
        conf_fine = F.interpolate(conf_attn, size=(fH, fW), mode='bilinear', align_corners=False)

        # ══════════════════════════════════════════════
        #  7. Geometry Solver
        # ══════════════════════════════════════════════
        result = {
            'flow_fine': flow_fine,
            'conf_fine': conf_fine,
            'fine_flow_preds': [flow_fine],  # single prediction for loss compatibility
            'decoded_q_fine': self._match_spatial(q_feat, self.FINE_HW),
        }

        if depth is not None:
            with torch.amp.autocast('cuda', enabled=False):
                depth_f32 = depth.float()
                flow_f32 = flow_fine.float()
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
