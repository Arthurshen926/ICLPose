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


def infonce_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 0.07,
    n_samples: int = 512,
) -> torch.Tensor:
    """Pixel-level InfoNCE contrastive loss.

    For each sampled pixel, the predicted feature should be more similar to
    the target feature at the same location than to target features at other
    locations.  This encourages spatially discriminative DCFF features.

    Args:
        pred:   [B, C, H, W] predicted (rendered) features
        target: [B, C, H, W] supervision (RADIO) features
        mask:   [B, 1, H, W] optional valid-pixel mask
        temperature: softmax temperature (lower → sharper)
        n_samples:   pixels to sample per image (memory: O(n²))
    """
    B, C, H, W = pred.shape
    N = H * W

    pred_n = F.normalize(pred.float(), p=2, dim=1).flatten(2)    # (B, C, N)
    target_n = F.normalize(target.float(), p=2, dim=1).flatten(2)

    total_loss = torch.tensor(0.0, device=pred.device)
    count = 0

    for b in range(B):
        if mask is not None:
            valid_idx = (mask[b].flatten() > 0.5).nonzero(as_tuple=True)[0]
        else:
            valid_idx = torch.arange(N, device=pred.device)

        n_valid = len(valid_idx)
        if n_valid < 32:
            continue

        k = min(n_samples, n_valid)
        perm = torch.randperm(n_valid, device=pred.device)[:k]
        idx = valid_idx[perm]

        p = pred_n[b, :, idx]    # (C, k)
        t = target_n[b, :, idx]  # (C, k)

        sim = torch.mm(p.T, t) / temperature   # (k, k)
        labels = torch.arange(k, device=pred.device)
        total_loss = total_loss + F.cross_entropy(sim, labels)
        count += 1

    return total_loss / max(count, 1)


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


def channel_standardized_loss(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor = None) -> torch.Tensor:
    """L1 loss after per-channel spatial standardization.

    Per-channel: subtract mean, divide by std. This addresses rank-1 RADIO features
    where normalization destroys spatial structure. Makes loss pay equal attention
    to all channels regardless of their magnitude.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    """
    def _standardize(x, m=None):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        if m is not None:
            m_flat = m.expand_as(x).reshape(B, C, -1)
            n_valid = m_flat.sum(-1, keepdim=True).clamp(min=1)
            mu = (x_flat * m_flat).sum(-1, keepdim=True) / n_valid
            var = ((x_flat - mu) ** 2 * m_flat).sum(-1, keepdim=True) / n_valid
        else:
            mu = x_flat.mean(-1, keepdim=True)
            var = x_flat.var(-1, keepdim=True)
        sigma = var.sqrt().clamp(min=1e-6)
        return ((x_flat - mu) / sigma).reshape(B, C, H, W)

    pred_s = _standardize(pred, mask)
    target_s = _standardize(target, mask)
    diff = (pred_s - target_s).abs()
    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum().clamp(min=1) * pred.shape[1]
        return diff.sum() / n_valid
    return diff.mean()


def feature_gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor = None) -> torch.Tensor:
    """Match screen-space feature gradients to preserve teacher structure."""
    pred_n = F.normalize(pred.float(), p=2, dim=1)
    target_n = F.normalize(target.float(), p=2, dim=1)

    pred_dx = pred_n[:, :, :, 1:] - pred_n[:, :, :, :-1]
    target_dx = target_n[:, :, :, 1:] - target_n[:, :, :, :-1]
    pred_dy = pred_n[:, :, 1:, :] - pred_n[:, :, :-1, :]
    target_dy = target_n[:, :, 1:, :] - target_n[:, :, :-1, :]

    dx = (pred_dx - target_dx).abs()
    dy = (pred_dy - target_dy).abs()

    if mask is None:
        return 0.5 * (dx.mean() + dy.mean())

    mask = mask.float()
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    dx = dx * mask_x
    dy = dy * mask_y
    denom_x = (mask_x.sum() * pred.shape[1]).clamp(min=1.0)
    denom_y = (mask_y.sum() * pred.shape[1]).clamp(min=1.0)
    return 0.5 * (dx.sum() / denom_x + dy.sum() / denom_y)


def screen_space_tv_loss(feat: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """L1 screen-space TV on normalized feature maps.

    Encourages neighboring pixels to have similar coarse features while
    respecting the rendered foreground mask so the loss does not blur across
    empty/background regions.
    """
    feat_n = F.normalize(feat, p=2, dim=1)
    dx = (feat_n[:, :, :, 1:] - feat_n[:, :, :, :-1]).abs()
    dy = (feat_n[:, :, 1:, :] - feat_n[:, :, :-1, :]).abs()

    if mask is None:
        return 0.5 * (dx.mean() + dy.mean())

    mask = mask.float()
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    dx = dx * mask_x
    dy = dy * mask_y

    denom_x = (mask_x.sum() * feat.shape[1]).clamp(min=1.0)
    denom_y = (mask_y.sum() * feat.shape[1]).clamp(min=1.0)
    return 0.5 * (dx.sum() / denom_x + dy.sum() / denom_y)


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
        lambda_coarse_screen_tv: float = 0.0,
        lambda_normal: float = 0.05,
        lambda_dist: float = 0.01,
        lambda_channel_std: float = 0.0,
        lambda_fine_grad: float = 0.0,
        lambda_coarse_grad: float = 0.0,
        lambda_fine_nce: float = 0.0,
        lambda_coarse_nce: float = 0.0,
        nce_temperature: float = 0.07,
        nce_samples: int = 512,
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_fine_cos = lambda_fine_cos
        self.lambda_fine_l1 = lambda_fine_l1
        self.lambda_coarse_cos = lambda_coarse_cos
        self.lambda_coarse_l1 = lambda_coarse_l1
        self.lambda_tv = lambda_tv
        self.lambda_coarse_screen_tv = lambda_coarse_screen_tv
        self.lambda_normal = lambda_normal
        self.lambda_dist = lambda_dist
        self.lambda_channel_std = lambda_channel_std
        self.lambda_fine_grad = lambda_fine_grad
        self.lambda_coarse_grad = lambda_coarse_grad
        self.lambda_fine_nce = lambda_fine_nce
        self.lambda_coarse_nce = lambda_coarse_nce
        self.nce_temperature = nce_temperature
        self.nce_samples = nce_samples

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
        cs = channel_standardized_loss(pred, target, mask) if self.lambda_channel_std > 0 else torch.tensor(0.0, device=pred.device)
        grad = feature_gradient_loss(pred, target, mask) if self.lambda_fine_grad > 0 else torch.tensor(0.0, device=pred.device)
        total = self.lambda_fine_cos * cos + self.lambda_fine_l1 * l1 + self.lambda_channel_std * cs + self.lambda_fine_grad * grad
        return {'fine_cos': cos, 'fine_l1': l1, 'fine_cs': cs, 'fine_grad': grad, 'fine_total': total}

    def coarse_loss(self, pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor = None) -> dict:
        """Coarse feature alignment loss."""
        cos = cosine_loss(pred, target, mask)
        l1 = l1_feature_loss(pred, target, mask)
        cs = channel_standardized_loss(pred, target, mask) if self.lambda_channel_std > 0 else torch.tensor(0.0, device=pred.device)
        grad = feature_gradient_loss(pred, target, mask) if self.lambda_coarse_grad > 0 else torch.tensor(0.0, device=pred.device)
        total = self.lambda_coarse_cos * cos + self.lambda_coarse_l1 * l1 + self.lambda_channel_std * cs + self.lambda_coarse_grad * grad
        return {'coarse_cos': cos, 'coarse_l1': l1, 'coarse_cs': cs, 'coarse_grad': grad, 'coarse_total': total}

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
        if (
            self.lambda_normal > 0
            and render_result.get('normals') is not None
            and render_result.get('surf_normals') is not None
        ):
            normals = render_result['normals']
            surf_normals = render_result['surf_normals']
            if surf_normals is not None and normals is not None:
                if surf_normals.dim() == 4 and normals.dim() == 4:
                    sn = surf_normals.permute(0, 3, 1, 2) if surf_normals.shape[-1] == 3 else surf_normals
                    nn_out = normals
                    normal_loss = (1.0 - (nn_out * sn).sum(dim=1)).mean()
                    if torch.isfinite(normal_loss):
                        losses['normal'] = normal_loss
                        total = total + self.lambda_normal * normal_loss

        if self.lambda_dist > 0 and render_result.get('distort') is not None:
            distort = render_result['distort']
            if distort is not None:
                dist_loss = distort.mean()
                if torch.isfinite(dist_loss):
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
            if self.lambda_fine_nce > 0:
                fine_nce = infonce_contrastive_loss(
                    fine, radio_geo, alpha_mask,
                    temperature=self.nce_temperature,
                    n_samples=self.nce_samples,
                )
                if not torch.isnan(fine_nce):
                    losses['fine_nce'] = fine_nce
                    total = total + self.lambda_fine_nce * fine_nce

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
            if self.lambda_coarse_nce > 0:
                coarse_nce = infonce_contrastive_loss(
                    coarse, radio_sem, alpha_mask,
                    temperature=self.nce_temperature,
                    n_samples=self.nce_samples,
                )
                if not torch.isnan(coarse_nce):
                    losses['coarse_nce'] = coarse_nce
                    total = total + self.lambda_coarse_nce * coarse_nce

            if self.lambda_coarse_screen_tv > 0:
                coarse_tv = screen_space_tv_loss(coarse, alpha_mask)
                if not torch.isnan(coarse_tv):
                    losses['coarse_tv'] = coarse_tv
                    total = total + self.lambda_coarse_screen_tv * coarse_tv

        # Hash grid TV regularization (phase 3)
        if phase >= 3 and hash_grid is not None:
            tv = hash_grid.total_variation_loss()
            losses['tv'] = tv
            total = total + self.lambda_tv * tv

        losses['total'] = total
        return losses
