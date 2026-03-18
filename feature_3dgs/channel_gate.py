"""
Channel Gate for Feature Embedding
====================================
可学习的通道级门控，自动识别并抑制噪声通道。

每个尺度的特征有独立的 gate_logits，经 sigmoid 后对渲染特征
逐通道加权。配合 L1 稀疏正则，自动发现对定位有用的通道子集。

用法:
    gate = ChannelGate(scale_dims={'fine_sd': 64, 'fine_dino': 64, 'mid': 64, 'coarse': 32})
    gated_feats = gate(rendered_feats)  # dict of [C, H, W]
    sparsity_loss = gate.sparsity_loss()
"""

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    """Learnable per-channel gating for each feature scale."""

    def __init__(self, scale_dims: dict, init_bias: float = 2.0):
        """
        Args:
            scale_dims: e.g. {'fine_sd': 64, 'fine_dino': 64, 'mid': 64, 'coarse': 32}
            init_bias: initial logit value (sigmoid(2.0) ≈ 0.88, all channels start open)
        """
        super().__init__()
        self.scale_dims = scale_dims
        self.gates = nn.ParameterDict()
        for name, dim in scale_dims.items():
            self.gates[name] = nn.Parameter(torch.full((dim,), init_bias))

    def forward(self, feats: dict) -> dict:
        """
        Apply channel gating.
        
        Args:
            feats: dict of [C, H, W] or [B, C, H, W] per scale
        Returns:
            gated feats with same structure
        """
        result = {}
        for name, feat in feats.items():
            if name not in self.gates:
                result[name] = feat
                continue
            g = torch.sigmoid(self.gates[name])
            if feat.ndim == 3:  # [C, H, W]
                result[name] = feat * g[:, None, None]
            elif feat.ndim == 4:  # [B, C, H, W]
                result[name] = feat * g[None, :, None, None]
            else:
                result[name] = feat
        return result

    def sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity loss on gate activations (encourages channel pruning)."""
        total = 0.0
        count = 0
        for g in self.gates.values():
            total = total + torch.sigmoid(g).sum()
            count += g.numel()
        return total / count

    def get_gate_stats(self) -> dict:
        """Get readable gate statistics for logging."""
        stats = {}
        for name, g in self.gates.items():
            activated = torch.sigmoid(g)
            stats[name] = {
                'mean': activated.mean().item(),
                'min': activated.min().item(),
                'max': activated.max().item(),
                'active_channels': (activated > 0.5).sum().item(),
                'total_channels': g.numel(),
            }
        return stats
