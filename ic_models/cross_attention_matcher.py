"""
Cross-Attention Coarse Matcher
================================
用 Transformer cross-attention 替代 global correlation 作为 coarse 匹配。

动机:
  - global_correlation 计算所有像素对的点积，对特征质量要求高
  - OH 的 SD coarse 特征高度相似 (correlation 接近平坦)
  - Cross-attention 可学习更强的匹配模式

接口兼容:
  输入: query [B, C, Hq, Wq], reference [B, C, Hr, Wr]
  输出: [B, Hr*Wr, Hq, Wq] (与 global_correlation 输出形状完全一致)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionMatcher(nn.Module):
    """
    Transformer-based cross-attention matcher.
    
    Works as RESIDUAL enhancement to global_correlation:
    output = global_correlation(q, r) + gate * attention_score(q, r)
    
    Gate starts at 0 (warmstart-safe) and grows during training,
    preserving the strong inductive bias of dot-product correlation.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        output_mode: str = 'correlation',   # 'correlation' or 'attention'
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        self.output_mode = output_mode

        # Lightweight position encoding (2D sinusoidal, added to tokens)
        self.pos_enc = LearnablePositionalEncoding2D(feat_dim)

        # Cross-attention layers: query attends to reference
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Project attended features to correlation-like scores
        # Initialize to zero so residual starts at 0
        self.score_proj = nn.Linear(feat_dim, feat_dim)
        nn.init.zeros_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)

        # Residual gate: starts at 0, grows during training
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute residual attention correlation.

        Args:
            query: (B, C, Hq, Wq) decoded query features
            reference: (B, C, Hr, Wr) decoded reference features

        Returns:
            corr: (B, Hr*Wr, Hq, Wq) — global_correlation + gate * attention_residual
        """
        B, C, Hq, Wq = query.shape
        _, _, Hr, Wr = reference.shape

        # Standard global correlation (baseline)
        f1 = query.reshape(B, C, Hq * Wq)
        f2 = reference.reshape(B, C, Hr * Wr)
        base_corr = torch.einsum('bcn,bcm->bmn', f1, f2)  # (B, Hr*Wr, Hq*Wq)

        # Attention residual
        q_tokens = query.flatten(2).permute(0, 2, 1)  # (B, Hq*Wq, C)
        r_tokens = reference.flatten(2).permute(0, 2, 1)  # (B, Hr*Wr, C)

        q_tokens = q_tokens + self.pos_enc(Hq, Wq, device=query.device)
        r_tokens = r_tokens + self.pos_enc(Hr, Wr, device=reference.device)

        q_attended = self.cross_attn(q_tokens, r_tokens)  # (B, Hq*Wq, C)

        q_proj = self.score_proj(q_attended)  # (B, Hq*Wq, C)
        q_proj = F.normalize(q_proj, p=2, dim=-1)
        r_norm = F.normalize(r_tokens, p=2, dim=-1)

        attn_corr = torch.bmm(q_proj, r_norm.permute(0, 2, 1))  # (B, Hq*Wq, Hr*Wr)
        attn_corr = attn_corr.permute(0, 2, 1)  # (B, Hr*Wr, Hq*Wq)

        # Combine: base + gate * residual
        combined = base_corr + self.gate * attn_corr
        return combined.reshape(B, Hr * Wr, Hq, Wq)


class LearnablePositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding with learnable scaling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self._cache = {}

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns (1, H*W, d_model) positional encoding."""
        key = (H, W, device)
        if key not in self._cache:
            pe = self._build_pe(H, W, device)
            self._cache[key] = pe
        return self._cache[key] * self.scale

    def _build_pe(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        d = self.d_model
        pe = torch.zeros(H, W, d, device=device)

        # Use half channels for y, half for x
        half_d = d // 2
        div_term = torch.exp(
            torch.arange(0, half_d, 2, device=device, dtype=torch.float32) *
            -(math.log(10000.0) / half_d)
        )

        pos_h = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(1)

        pe[:, :, 0:half_d:2] = torch.sin(pos_h * div_term).unsqueeze(1).expand(-1, W, -1)
        pe[:, :, 1:half_d:2] = torch.cos(pos_h * div_term).unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half_d::2] = torch.sin(pos_w * div_term).unsqueeze(0).expand(H, -1, -1)
        pe[:, :, half_d+1::2] = torch.cos(pos_w * div_term).unsqueeze(0).expand(H, -1, -1)

        return pe.reshape(1, H * W, d)
