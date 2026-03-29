"""
Loss functions for Deferred Cascaded Feature Field training.

Three core losses:
  1. RGB reconstruction: L1 + SSIM (standard 2DGS)
  2. Fine feature alignment: cosine + L1 against RADIO_geo targets
  3. Coarse feature alignment: cosine + L1 against RADIO_sem targets

Plus regularizers:
  - Hash grid total variation (spatial smoothness)
  - 2DGS geometry: normal consistency, distortion, scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
    """Mean cosine distance: 1 - cos_sim, averaged over valid pixels.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    """
    pred_n = F.normalize(pred, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    cos_sim = (pred_n * target_n).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    if mask is not None:
        loss = (1.0 - cos_sim) * mask
        n_valid = mask.sum().clamp(min=1)
        return loss.sum() / n_valid
    return (1.0 - cos_sim).mean()


def l1_feature_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
    """L1 loss on L2-normalized features.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    """
    pred_n = F.normalize(pred, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    diff = (pred_n - target_n).abs()

    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum().clamp(min=1) * pred.shape[1]
        return diff.sum() / n_valid
    return diff.mean()


def ssim_loss(img1, img2, window_size=11):
    """1 - SSIM loss for RGB reconstruction."""
    C = img1.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    window = torch.outer(g, g)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


class DCFFLoss(nn.Module):
    """Combined loss manager for DCFF training.

    Manages phase-dependent loss computation:
      Phase 1 (geometry): RGB only
      Phase 2 (+fine): RGB + fine feature alignment
      Phase 3 (+coarse): RGB + fine + coarse feature alignment
    """

    def __init__(
        self,
        lambda_dssim: float = 0.2,
        lambda_fine_cos: float = 1.0,
        lambda_fine_l1: float = 0.5,
        lambda_coarse_cos: float = 1.0,
        lambda_coarse_l1: float = 0.5,
        lambda_tv: float = 0.01,
        lambda_normal: float = 0.05,
        lambda_dist: float = 0.01,
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_fine_cos = lambda_fine_cos
        self.lambda_fine_l1 = lambda_fine_l1
        self.lambda_coarse_cos = lambda_coarse_cos
        self.lambda_coarse_l1 = lambda_coarse_l1
        self.lambda_tv = lambda_tv
        self.lambda_normal = lambda_normal
        self.lambda_dist = lambda_dist

    def rgb_loss(self, rendered: torch.Tensor, gt: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """(1-λ)*L1 + λ*SSIM combined RGB loss."""
        if mask is not None:
            rendered = rendered * mask
            gt = gt * mask

        l1 = F.l1_loss(rendered, gt)
        ss = ssim_loss(rendered, gt)
        return (1 - self.lambda_dssim) * l1 + self.lambda_dssim * ss

    def fine_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor = None) -> dict:
        """Fine feature alignment loss."""
        cos = cosine_loss(pred, target, mask)
        l1 = l1_feature_loss(pred, target, mask)
        total = self.lambda_fine_cos * cos + self.lambda_fine_l1 * l1
        return {'fine_cos': cos, 'fine_l1': l1, 'fine_total': total}

    def coarse_loss(self, pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor = None) -> dict:
        """Coarse feature alignment loss."""
        cos = cosine_loss(pred, target, mask)
        l1 = l1_feature_loss(pred, target, mask)
        total = self.lambda_coarse_cos * cos + self.lambda_coarse_l1 * l1
        return {'coarse_cos': cos, 'coarse_l1': l1, 'coarse_total': total}

    def compute(
        self,
        render_result: dict,
        gt_rgb: torch.Tensor,
        radio_geo: torch.Tensor = None,
        radio_sem: torch.Tensor = None,
        hash_grid=None,
        phase: int = 1,
        mask: torch.Tensor = None,
    ) -> dict:
        """Compute all active losses for the current phase.

        Args:
            render_result: output dict from DeferredCascadedRenderer.forward()
            gt_rgb: [1, 3, H, W] ground truth RGB
            radio_geo: [1, 64, fH, fW] RADIO fine geometric targets
            radio_sem: [1, 64, fH, fW] RADIO coarse semantic targets
            hash_grid: SpatialHashGrid (for TV loss)
            phase: 1=RGB only, 2=+fine, 3=+fine+coarse
            mask: [1, 1, H, W] optional mask

        Returns:
            dict of loss name → value
        """
        losses = {}

        # RGB reconstruction (all phases)
        losses['rgb'] = self.rgb_loss(render_result['rgb'], gt_rgb, mask)
        total = losses['rgb']

        # Geometry regularization
        if render_result.get('normals') is not None and render_result.get('surf_normals') is not None:
            normals = render_result['normals']
            surf_normals = render_result['surf_normals']
            if surf_normals is not None and normals is not None:
                if surf_normals.dim() == 4 and normals.dim() == 4:
                    sn = surf_normals.permute(0, 3, 1, 2) if surf_normals.shape[-1] == 3 else surf_normals
                    nn_out = normals
                    normal_loss = (1.0 - (nn_out * sn).sum(dim=1)).mean()
                    if not torch.isnan(normal_loss):
                        losses['normal'] = normal_loss
                        total = total + self.lambda_normal * normal_loss

        if render_result.get('distort') is not None:
            distort = render_result['distort']
            if distort is not None:
                dist_loss = distort.mean()
                if not torch.isnan(dist_loss):
                    losses['distort'] = dist_loss
                    total = total + self.lambda_dist * dist_loss

        # Fine feature alignment (phase 2+)
        if phase >= 2 and radio_geo is not None and render_result.get('fine_features') is not None:
            fine = render_result['fine_features']
            # Resize to match RADIO target resolution
            if fine.shape[-2:] != radio_geo.shape[-2:]:
                fine = F.interpolate(fine, radio_geo.shape[-2:],
                                     mode='bilinear', align_corners=False)
            alpha_mask = None
            if render_result.get('alpha') is not None:
                am = render_result['alpha']
                if am.shape[-2:] != radio_geo.shape[-2:]:
                    am = F.interpolate(am, radio_geo.shape[-2:],
                                       mode='bilinear', align_corners=False)
                alpha_mask = (am > 0.5).float()

            fl = self.fine_loss(fine, radio_geo, alpha_mask)
            losses.update(fl)
            total = total + fl['fine_total']

        # Coarse feature alignment (phase 3)
        if phase >= 3 and radio_sem is not None and render_result.get('coarse_features') is not None:
            coarse = render_result['coarse_features']
            if coarse.shape[-2:] != radio_sem.shape[-2:]:
                coarse = F.interpolate(coarse, radio_sem.shape[-2:],
                                       mode='bilinear', align_corners=False)
            alpha_mask = None
            if render_result.get('alpha') is not None:
                am = render_result['alpha']
                if am.shape[-2:] != radio_sem.shape[-2:]:
                    am = F.interpolate(am, radio_sem.shape[-2:],
                                       mode='bilinear', align_corners=False)
                alpha_mask = (am > 0.5).float()

            cl = self.coarse_loss(coarse, radio_sem, alpha_mask)
            losses.update(cl)
            total = total + cl['coarse_total']

        # Hash grid TV regularization (phase 3)
        if phase >= 3 and hash_grid is not None:
            tv = hash_grid.total_variation_loss()
            losses['tv'] = tv
            total = total + self.lambda_tv * tv

        losses['total'] = total
        return losses
