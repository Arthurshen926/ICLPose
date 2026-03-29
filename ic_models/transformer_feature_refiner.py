"""
TransformerFeatureRefiner: Cross-attention feature enhancement for flow estimation
==================================================================================
在 correlation 计算前，用 Transformer 增强 query/reference 特征的匹配能力。

设计要点:
  - Self-attention: 增强局部特征一致性 (消除噪声、增强边缘)
  - Cross-attention: Q 特征吸收 R 上下文 (建立跨视角关联)
  - 输出增强后的 Q/R 特征，然后走正常 correlation 流程
  - Residual + gate 机制: warmstart safe，gate 初始为 0

支持两种模式:
  - 'full': 全注意力 (coarse/mid 尺度, tokens < 2000)
  - 'windowed': 窗口注意力 (fine 尺度, tokens > 10000)

参数量估计 (per scale):
  feat_dim=64, n_heads=4, n_layers=2, ffn=128
  → ~2×(4×64×64 + 64×128×2) ~= 66K per layer, ×2 layers = ~132K
  → 3 scales = ~400K 额外参数 (总模型 ~4.6M)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SinusoidalPE2D(nn.Module):
    """Lightweight 2D sinusoidal positional encoding."""

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


class TransformerFeatureRefiner(nn.Module):
    """
    Cross-attention feature enhancement module.

    Refines Q and R feature maps by:
    1. Self-attention within Q (intra-feature consistency)
    2. Cross-attention: Q attends to R (cross-view information transfer)
    3. Symmetric: also refines R by attending to Q

    Gate mechanism ensures warmstart safety.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        mode: str = 'full',       # 'full' or 'windowed'
        window_size: int = 7,     # for windowed mode
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mode = mode
        self.window_size = window_size

        self.pe = SinusoidalPE2D(feat_dim)

        # Self-attention + Cross-attention layers for Q
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()  # pre-norm for self-attn
        self.norm2 = nn.ModuleList()  # pre-norm for cross-attn
        self.norm3 = nn.ModuleList()  # pre-norm for ffn

        for _ in range(n_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn_layers.append(
                nn.MultiheadAttention(feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(feat_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, feat_dim),
            ))
            self.norm1.append(nn.LayerNorm(feat_dim))
            self.norm2.append(nn.LayerNorm(feat_dim))
            self.norm3.append(nn.LayerNorm(feat_dim))

        # Output projection (zero-init for warmstart safety)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Residual gate starts at 0
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _refine_one_side(
        self, q_tokens: torch.Tensor, r_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Refine q by attending to itself and r. Returns refined q tokens."""
        x = q_tokens
        for i in range(self.n_layers):
            # Self-attention (pre-norm)
            x_norm = self.norm1[i](x)
            x = x + self.self_attn_layers[i](x_norm, x_norm, x_norm)[0]
            # Cross-attention (pre-norm)
            x_norm = self.norm2[i](x)
            r_norm = self.norm2[i](r_tokens)  # share normalization stats
            x = x + self.cross_attn_layers[i](x_norm, r_norm, r_norm)[0]
            # FFN (pre-norm)
            x_norm = self.norm3[i](x)
            x = x + self.ffn_layers[i](x_norm)
        return x

    def _windowed_refine(
        self, q: torch.Tensor, r: torch.Tensor,
        H: int, W: int,
    ) -> torch.Tensor:
        """
        Windowed cross-attention for large feature maps.
        Splits feature map into non-overlapping windows, processes each independently.
        """
        B, N, C = q.shape
        wH = wW = self.window_size

        # Pad to multiples of window size
        pad_h = (wH - H % wH) % wH
        pad_w = (wW - W % wW) % wW
        Hp, Wp = H + pad_h, W + pad_w

        # Reshape to spatial, pad, and partition into windows
        q_2d = q.reshape(B, H, W, C)
        r_2d = r.reshape(B, H, W, C)

        if pad_h > 0 or pad_w > 0:
            q_2d = F.pad(q_2d, (0, 0, 0, pad_w, 0, pad_h))
            r_2d = F.pad(r_2d, (0, 0, 0, pad_w, 0, pad_h))

        nH, nW = Hp // wH, Wp // wW
        # (B, nH, wH, nW, wW, C)
        q_win = q_2d.reshape(B, nH, wH, nW, wW, C).permute(0, 1, 3, 2, 4, 5)
        r_win = r_2d.reshape(B, nH, wH, nW, wW, C).permute(0, 1, 3, 2, 4, 5)
        # (B*nH*nW, wH*wW, C)
        Bw = B * nH * nW
        q_win = q_win.reshape(Bw, wH * wW, C)
        r_win = r_win.reshape(Bw, wH * wW, C)

        # Apply transformer within windows
        q_refined = self._refine_one_side(q_win, r_win)

        # Unpartition
        q_refined = q_refined.reshape(B, nH, nW, wH, wW, C)
        q_refined = q_refined.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            q_refined = q_refined[:, :H, :W, :]

        return q_refined.reshape(B, N, C)

    def forward(
        self, q_feat: torch.Tensor, r_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine Q and R features via cross-attention.

        Args:
            q_feat: (B, C, H, W)
            r_feat: (B, C, H, W)

        Returns:
            q_refined: (B, C, H, W) — enhanced query features
            r_refined: (B, C, H, W) — enhanced reference features
        """
        B, C, H, W = q_feat.shape

        # Flatten to tokens + add PE
        q_tokens = q_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        r_tokens = r_feat.flatten(2).permute(0, 2, 1)
        pe = self.pe(H, W, device=q_feat.device)
        q_tokens = q_tokens + pe
        r_tokens = r_tokens + pe

        if self.mode == 'windowed' and H * W > 2000:
            q_refined = self._windowed_refine(q_tokens, r_tokens, H, W)
            r_refined = self._windowed_refine(r_tokens, q_tokens, H, W)
        else:
            q_refined = self._refine_one_side(q_tokens, r_tokens)
            r_refined = self._refine_one_side(r_tokens, q_tokens)

        # Project and gate
        q_out = self.out_proj(q_refined).permute(0, 2, 1).reshape(B, C, H, W)
        r_out = self.out_proj(r_refined).permute(0, 2, 1).reshape(B, C, H, W)

        # Residual: original + gate * refined
        g = torch.tanh(self.gate)
        q_final = q_feat + g * q_out
        r_final = r_feat + g * r_out

        return q_final, r_final


class TransformerFlowDecoder(nn.Module):
    """
    全 Transformer 流场解码器，完全替代 correlation + FlowRefinementHead。

    将 Q/R 特征通过 cross-attention 直接预测流场，无需显式 correlation volume。
    适用于 coarse 和 mid 尺度 (tokens < 1500)。

    设计灵感: GMFlow (Xu et al., CVPR 2022), FlowFormer (Huang et al., ECCV 2022)

    工作流程:
      1. Q/R tokens + 2D PE
      2. N layers of: self-attn(Q) → cross-attn(Q→R) → FFN
      3. Q tokens → flow + confidence head
      4. 支持 flow conditioning: 当前 flow + conf 注入到 Q tokens

    接口: 与 FlowRefinementHead 对齐
    """

    def __init__(
        self,
        feat_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        conf_dim: int = 1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.conf_dim = conf_dim

        self.pe = SinusoidalPE2D(feat_dim)

        # Flow conditioning: project current flow+conf to feat_dim
        self.flow_cond = nn.Sequential(
            nn.Linear(2 + conf_dim, feat_dim),
            nn.GELU(),
        )

        # Self+Cross attention layers
        self.self_attn = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.norm3 = nn.ModuleList()

        for _ in range(n_layers):
            self.self_attn.append(
                nn.MultiheadAttention(feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn.append(
                nn.MultiheadAttention(feat_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffn.append(nn.Sequential(
                nn.Linear(feat_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, feat_dim),
            ))
            self.norm1.append(nn.LayerNorm(feat_dim))
            self.norm2.append(nn.LayerNorm(feat_dim))
            self.norm3.append(nn.LayerNorm(feat_dim))

        # Output: project to hidden_dim for compatibility, then predict flow+conf
        self.to_hidden = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2 + conf_dim, 3, padding=1),
        )

    def forward(
        self,
        q_feat: torch.Tensor,    # (B, C, H, W) decoded query
        r_feat: torch.Tensor,    # (B, C, H, W) decoded reference
        flow: torch.Tensor,      # (B, 2, H, W) current flow
        conf: torch.Tensor,      # (B, conf_dim, H, W) current confidence
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns same interface as FlowRefinementHead:
            delta_flow, new_conf, hidden, flow_out
        """
        B, C, H, W = q_feat.shape

        # Flatten + PE
        q_tokens = q_feat.flatten(2).permute(0, 2, 1)  # (B, N, C)
        r_tokens = r_feat.flatten(2).permute(0, 2, 1)
        pe = self.pe(H, W, device=q_feat.device)
        q_tokens = q_tokens + pe
        r_tokens = r_tokens + pe

        # Flow conditioning: add flow/conf info to Q tokens
        flow_cond_inp = torch.cat([flow, conf], dim=1)  # (B, 2+conf_dim, H, W)
        flow_cond_tokens = flow_cond_inp.flatten(2).permute(0, 2, 1)  # (B, N, 2+c)
        q_tokens = q_tokens + self.flow_cond(flow_cond_tokens)

        # Transformer layers
        x = q_tokens
        for i in range(len(self.self_attn)):
            xn = self.norm1[i](x)
            x = x + self.self_attn[i](xn, xn, xn)[0]
            xn = self.norm2[i](x)
            rn = self.norm2[i](r_tokens)
            x = x + self.cross_attn[i](xn, rn, rn)[0]
            xn = self.norm3[i](x)
            x = x + self.ffn[i](xn)

        # Project to hidden dim
        hidden = self.to_hidden(x)  # (B, N, hidden_dim)
        hidden = hidden.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)

        # Predict flow delta + conf
        out = self.flow_head(hidden)
        delta_flow = out[:, :2]
        raw_conf = out[:, 2:]
        new_conf = torch.sigmoid(raw_conf)
        flow_out = flow + delta_flow

        return delta_flow, new_conf, hidden, flow_out
