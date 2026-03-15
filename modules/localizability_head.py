"""
Localizability Scoring Head (C1)
=================================
Learns a per-pixel prior for how localizable each pixel is, based on
correlation volume statistics rather than just flow confidence.

Key insight: the existing confidence output only captures "how sure is
the flow head about its prediction", not "is this pixel inherently
useful for localization".  A flat/multi-peak correlation → ambiguous
region (repeated texture); sharp uni-modal peak → distinctive region.

Signals used:
  1. Correlation peak value  (max of correlation volume)
  2. Peak sharpness          (max - second_max, or max - mean)
  3. Correlation entropy      (-Σ p log p)

These are computed from the correlation volume *without* additional
neural network parameters, then fed through a lightweight 3-layer MLP
that outputs a [0, 1] localizability score per pixel.

Supervision: soft label from detached flow error (low error → high score).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LocalizabilityHead(nn.Module):
    """
    Per-pixel localizability scoring from correlation volume statistics.

    Input:  correlation volume (B, corr_channels, H, W)
    Output: localizability score (B, 1, H, W) in [0, 1]
    """

    def __init__(self, corr_channels: int, hidden_dim: int = 32):
        """
        Args:
            corr_channels: number of channels in the correlation volume
                           (e.g. (2*r+1)^2 = 81 for radius=4)
            hidden_dim: hidden layer dimension
        """
        super().__init__()
        # Statistics extractor: 3 scalar features per pixel
        #   [peak_value, sharpness, entropy]
        stat_dim = 3

        self.mlp = nn.Sequential(
            nn.Conv2d(stat_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Zero-init output layer for warmstart safety:
        # initially outputs ~0 → sigmoid → 0.5 (neutral)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.corr_channels = corr_channels

    def _compute_stats(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Extract per-pixel statistics from correlation volume.

        Args:
            corr: (B, C, H, W) raw correlation values

        Returns:
            stats: (B, 3, H, W) — [peak, sharpness, entropy]
        """
        B, C, H, W = corr.shape

        # 1. Peak value
        peak, _ = corr.max(dim=1, keepdim=True)  # (B, 1, H, W)

        # 2. Sharpness: peak - mean (how much the peak stands out)
        mean_corr = corr.mean(dim=1, keepdim=True)
        sharpness = peak - mean_corr  # (B, 1, H, W)

        # 3. Entropy of softmax distribution (normalized correlation)
        probs = F.softmax(corr, dim=1)  # (B, C, H, W)
        log_probs = torch.log(probs.clamp(min=1e-8))
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        # Normalize entropy to ~[0, 1] by dividing by max possible entropy
        max_entropy = torch.log(torch.tensor(float(C), device=corr.device))
        entropy = entropy / max_entropy.clamp(min=1e-8)

        return torch.cat([peak, sharpness, entropy], dim=1)  # (B, 3, H, W)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            corr: (B, C, H, W) correlation volume

        Returns:
            score: (B, 1, H, W) localizability score in [0, 1]
        """
        stats = self._compute_stats(corr)
        logit = self.mlp(stats)
        return torch.sigmoid(logit)


def localizability_loss(
    loc_score: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    error_clamp: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervision for localizability head using flow error as soft target.

    Pixels with low flow error → high localizability target.
    Pixels with high flow error → low localizability target.

    Args:
        loc_score: (B, 1, H, W) predicted localizability in [0, 1]
        flow_pred: (B, 2, H, W) predicted flow (detached)
        flow_gt: (B, 2, H, W) ground truth flow
        mask: (B, 1, H, W) valid pixel mask
        error_clamp: max flow error for normalization (pixels)

    Returns:
        loss: scalar
        metrics: dict with diagnostic values
    """
    with torch.no_grad():
        flow_err = torch.norm(flow_pred.detach() - flow_gt.detach(), dim=1, keepdim=True)
        # Normalize to [0, 1], invert so low error → high target
        err_norm = (flow_err / error_clamp).clamp(0, 1)
        target = 1.0 - err_norm  # (B, 1, H, W)

    # Disable autocast for BCE (it's unsafe under AMP autocast)
    with torch.cuda.amp.autocast(enabled=False):
        loss = F.binary_cross_entropy(loc_score.float(), target.float(), reduction='none')

    if mask is not None:
        n_valid = mask.sum().clamp(min=1.0)
        loss = (loss * mask).sum() / n_valid
    else:
        loss = loss.mean()

    metrics = {
        'loc_score_mean': loc_score.mean().item(),
        'loc_score_std': loc_score.std().item(),
        'loc_target_mean': target.mean().item(),
        'loc_loss': loss.item(),
    }

    return loss, metrics
