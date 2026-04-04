# Source Generated with Decompyle++
# File: losses.cpython-39.pyc (Python 3.9)

'''
Loss functions for Deferred Cascaded Feature Field training.

Three core losses:
  1. RGB reconstruction: L1 + SSIM (standard 2DGS)
  2. Fine feature alignment: cosine + L1 against RADIO_geo targets
  3. Coarse feature alignment: cosine + L1 against RADIO_sem targets

Plus regularizers:
  - Hash grid total variation (spatial smoothness)
  - 2DGS geometry: normal consistency, distortion, scale
'''
import torch
from torch.nn import nn
import torch.nn.functional
F = functional
nn

def cosine_loss(pred = None, target = None, mask = None):
    '''Mean cosine distance: 1 - cos_sim, averaged over valid pixels.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    '''
    pred_n = F.normalize(pred, 2, 1, **('p', 'dim'))
    target_n = F.normalize(target, 2, 1, **('p', 'dim'))
    cos_sim = (pred_n * target_n).sum(1, True, **('dim', 'keepdim'))
    if mask is not None:
        loss = (1 - cos_sim) * mask
        n_valid = mask.sum().clamp(1, **('min',))
        return loss.sum() / n_valid
    return (None - cos_sim).mean()


def l1_feature_loss(pred = None, target = None, mask = None):
    '''L1 loss on L2-normalized features.

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    '''
    pred_n = F.normalize(pred, 2, 1, **('p', 'dim'))
    target_n = F.normalize(target, 2, 1, **('p', 'dim'))
    diff = (pred_n - target_n).abs()
    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum().clamp(1, **('min',)) * pred.shape[1]
        return diff.sum() / n_valid
    return None.mean()


def raw_l1_loss(pred = None, target = None, mask = None):
    '''L1 loss on raw (unnormalized) features.

    Forces the model to match per-pixel magnitudes, not just directions.
    Critical when teacher features carry spatial info in magnitude (e.g.
    RADIO features which are near rank-1 in direction).

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    '''
    diff = (pred - target).abs()
    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum().clamp(1, **('min',)) * pred.shape[1]
        return diff.sum() / n_valid
    return None.mean()


def channel_standardized_loss(pred = None, target = None, mask = None):
    '''L1 loss after per-channel spatial standardization.

    Scale-invariant: does not require pred and target to have the same
    magnitude range.  Preserves spatial structure by ensuring the per-channel
    spatial *distribution* of pred matches target (mean=0, std=1).

    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        mask: [B, 1, H, W] optional validity mask
    '''
    
    def _standardize(x, m = (None,)):
        (B, C, H, W) = x.shape
        if m is not None:
            m_flat = m.expand_as(x).reshape(B, C, -1)
            x_flat = x.reshape(B, C, -1)
            n_valid = m_flat.sum(-1, True, **('keepdim',)).clamp(1, **('min',))
            mu = (x_flat * m_flat).sum(-1, True, **('keepdim',)) / n_valid
            var = ((x_flat - mu) ** 2 * m_flat).sum(-1, True, **('keepdim',)) / n_valid
            sigma = var.sqrt().clamp(1e-06, **('min',))
            return ((x_flat - mu) / sigma).reshape(B, C, H, W)
        x_flat = None.reshape(B, C, -1)
        mu = x_flat.mean(-1, True, **('keepdim',))
        sigma = x_flat.std(-1, True, **('keepdim',)).clamp(1e-06, **('min',))
        return ((x_flat - mu) / sigma).reshape(B, C, H, W)

    pred_s = _standardize(pred, mask)
    target_s = _standardize(target, mask)
    diff = (pred_s - target_s).abs()
    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum().clamp(1, **('min',)) * pred.shape[1]
        return diff.sum() / n_valid
    return None.mean()


def ssim_loss(img1, img2, window_size = (11,)):
    '''1 - SSIM loss for RGB reconstruction.'''
    C = img1.shape[1]
    coords = torch.arange(window_size, torch.float32, img1.device, **('dtype', 'device')) - window_size // 2
    g = torch.exp(-coords ** 2 / 4.5)
    window = torch.outer(g, g)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, pad, C, **('padding', 'groups'))
    mu2 = F.conv2d(img2, window, pad, C, **('padding', 'groups'))
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, window, pad, C, **('padding', 'groups')) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, pad, C, **('padding', 'groups')) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, pad, C, **('padding', 'groups')) - mu1_mu2
    (C1, C2) = (0.0001, 0.0009)
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


class DCFFLoss(nn.Module):
    '''Combined loss manager for DCFF training.

    Manages phase-dependent loss computation:
      Phase 1 (geometry): RGB only
      Phase 2 (+fine): RGB + fine feature alignment
      Phase 3 (+coarse): RGB + fine + coarse feature alignment
    '''
    
    def __init__(self = None, lambda_dssim = None, lambda_fine_cos = None, lambda_fine_l1 = None, lambda_coarse_cos = None, lambda_coarse_l1 = None, lambda_tv = None, lambda_normal = None, lambda_dist = None, lambda_fine_raw = None, lambda_coarse_raw = None, lambda_fine_std = (0.2, 1, 0.5, 1, 0.5, 0.01, 0.05, 0.01, 0, 0, 0, 0), lambda_coarse_std = ({
        'lambda_dssim': float,
        'lambda_fine_cos': float,
        'lambda_fine_l1': float,
        'lambda_coarse_cos': float,
        'lambda_coarse_l1': float,
        'lambda_tv': float,
        'lambda_normal': float,
        'lambda_dist': float,
        'lambda_fine_raw': float,
        'lambda_coarse_raw': float,
        'lambda_fine_std': float,
        'lambda_coarse_std': float },)):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_fine_cos = lambda_fine_cos
        self.lambda_fine_l1 = lambda_fine_l1
        self.lambda_coarse_cos = lambda_coarse_cos
        self.lambda_coarse_l1 = lambda_coarse_l1
        self.lambda_tv = lambda_tv
        self.lambda_normal = lambda_normal
        self.lambda_dist = lambda_dist
        self.lambda_fine_raw = lambda_fine_raw
        self.lambda_coarse_raw = lambda_coarse_raw
        self.lambda_fine_std = lambda_fine_std
        self.lambda_coarse_std = lambda_coarse_std

    
    def rgb_loss(self = None, rendered = None, gt = None, mask = (None,)):
        '''(1-λ)*L1 + λ*SSIM combined RGB loss.'''
        if mask is not None:
            rendered = rendered * mask
            gt = gt * mask
        l1 = F.l1_loss(rendered, gt)
        ss = ssim_loss(rendered, gt)
        return (1 - self.lambda_dssim) * l1 + self.lambda_dssim * ss

    
    def fine_loss(self = None, pred = None, target = None, mask = (None,)):
        '''Fine feature alignment loss.'''
        d = { }
        total = torch.tensor(0, pred.device, **('device',))
        if self.lambda_fine_cos > 0:
            cos = cosine_loss(pred, target, mask)
            d['fine_cos'] = cos
            total = total + self.lambda_fine_cos * cos
        if self.lambda_fine_l1 > 0:
            l1 = l1_feature_loss(pred, target, mask)
            d['fine_l1'] = l1
            total = total + self.lambda_fine_l1 * l1
        if self.lambda_fine_raw > 0:
            rl1 = raw_l1_loss(pred, target, mask)
            d['fine_raw'] = rl1
            total = total + self.lambda_fine_raw * rl1
        if self.lambda_fine_std > 0:
            sl1 = channel_standardized_loss(pred, target, mask)
            d['fine_std'] = sl1
            total = total + self.lambda_fine_std * sl1
        d['fine_total'] = total
        return d

    
    def coarse_loss(self = None, pred = None, target = None, mask = (None,)):
        '''Coarse feature alignment loss.'''
        d = { }
        total = torch.tensor(0, pred.device, **('device',))
        if self.lambda_coarse_cos > 0:
            cos = cosine_loss(pred, target, mask)
            d['coarse_cos'] = cos
            total = total + self.lambda_coarse_cos * cos
        if self.lambda_coarse_l1 > 0:
            l1 = l1_feature_loss(pred, target, mask)
            d['coarse_l1'] = l1
            total = total + self.lambda_coarse_l1 * l1
        if self.lambda_coarse_raw > 0:
            rl1 = raw_l1_loss(pred, target, mask)
            d['coarse_raw'] = rl1
            total = total + self.lambda_coarse_raw * rl1
        if self.lambda_coarse_std > 0:
            sl1 = channel_standardized_loss(pred, target, mask)
            d['coarse_std'] = sl1
            total = total + self.lambda_coarse_std * sl1
        d['coarse_total'] = total
        return d

    
    def compute(self, render_result, gt_rgb, radio_geo = None, radio_sem = None, hash_grid = None, phase = (None, None, None, 1, None), mask = {
        'render_result': dict,
        'gt_rgb': torch.Tensor,
        'radio_geo': torch.Tensor,
        'radio_sem': torch.Tensor,
        'phase': int,
        'mask': torch.Tensor,
        'return': dict }):
        '''Compute all active losses for the current phase.

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
        '''
        losses = { }
        losses['rgb'] = self.rgb_loss(render_result['rgb'], gt_rgb, mask)
        total = losses['rgb']
        if render_result.get('normals') is not None and render_result.get('surf_normals') is not None:
            normals = render_result['normals']
            surf_normals = render_result['surf_normals']
            if surf_normals is not None and normals is not None and surf_normals.dim() == 4 and normals.dim() == 4:
                sn = surf_normals.permute(0, 3, 1, 2) if surf_normals.shape[-1] == 3 else surf_normals
                nn_out = normals
                normal_loss = (1 - (nn_out * sn).sum(1, **('dim',))).mean()
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
        if phase >= 2 and radio_geo is not None and render_result.get('fine_features') is not None:
            fine = render_result['fine_features']
            if fine.shape[-2:] != radio_geo.shape[-2:]:
                fine = F.interpolate(fine, radio_geo.shape[-2:], 'bilinear', False, **('mode', 'align_corners'))
            alpha_mask = None
            if render_result.get('alpha') is not None:
                am = render_result['alpha']
                if am.shape[-2:] != radio_geo.shape[-2:]:
                    am = F.interpolate(am, radio_geo.shape[-2:], 'bilinear', False, **('mode', 'align_corners'))
                alpha_mask = (am > 0.5).float()
            fl = self.fine_loss(fine, radio_geo, alpha_mask)
            losses.update(fl)
            total = total + fl['fine_total']
        if phase >= 3 and radio_sem is not None and render_result.get('coarse_features') is not None:
            coarse = render_result['coarse_features']
            if coarse.shape[-2:] != radio_sem.shape[-2:]:
                coarse = F.interpolate(coarse, radio_sem.shape[-2:], 'bilinear', False, **('mode', 'align_corners'))
            alpha_mask = None
            if render_result.get('alpha') is not None:
                am = render_result['alpha']
                if am.shape[-2:] != radio_sem.shape[-2:]:
                    am = F.interpolate(am, radio_sem.shape[-2:], 'bilinear', False, **('mode', 'align_corners'))
                alpha_mask = (am > 0.5).float()
            cl = self.coarse_loss(coarse, radio_sem, alpha_mask)
            losses.update(cl)
            total = total + cl['coarse_total']
        if phase >= 3 and hash_grid is not None:
            tv = hash_grid.total_variation_loss()
            losses['tv'] = tv
            total = total + self.lambda_tv * tv
        losses['total'] = total
        return losses

    __classcell__ = None

