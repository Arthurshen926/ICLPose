"""Loss functions for RADIO-GS feature distillation.

Supervises distillation of RADIO (1280d) features into 3DGS, including
feature reconstruction, multi-view consistency, adaptor alignment,
and spatial smoothness regularization.
"""

from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Core feature distillation loss between decoded and GT RADIO features.

    Combines pixel-wise L2 (or Huber) with cosine similarity loss,
    optionally masked by alpha visibility.  Supports channel-standardized
    loss to address rank-1 RADIO features (where normalization destroys
    spatial structure).
    """

    def __init__(
        self,
        l2_weight: float = 1.0,
        cosine_weight: float = 0.5,
        huber_weight: float = 0.0,
        huber_delta: float = 0.1,
        normalize_features: bool = True,
        channel_std_weight: float = 0.0,
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.cosine_weight = cosine_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        self.normalize_features = normalize_features
        self.channel_std_weight = channel_std_weight

        if huber_weight > 0:
            self.huber_loss = nn.HuberLoss(reduction='none', delta=huber_delta)

    @staticmethod
    def _channel_standardized_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """L1 loss after per-channel spatial standardization.

        Scale-invariant: preserves spatial structure by ensuring the per-channel
        spatial distribution of pred matches target (mean=0, std=1).
        Addresses rank-1 RADIO features where >99.9% variance is in one direction.
        """
        def _standardize(x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def forward(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            decoded: [B, 1280, H, W] reconstructed RADIO features.
            target: [B, 1280, H, W] ground truth RADIO features.
            mask: [B, 1, H, W] optional validity mask (e.g. alpha > threshold).

        Returns:
            Dict with 'total', 'l2', 'cosine', and optionally 'channel_std' loss tensors.
        """
        # Channel-standardized loss operates on raw (unnormalized) features
        channel_std_loss = torch.tensor(0.0, device=decoded.device)
        if self.channel_std_weight > 0:
            channel_std_loss = self._channel_standardized_loss(decoded, target, mask)

        if self.normalize_features:
            decoded = F.normalize(decoded, p=2, dim=1)
            target = F.normalize(target, p=2, dim=1)

        # --- L2 / Huber loss: pixel-wise MSE over the channel dimension ---
        if self.huber_weight > 0:
            pixel_loss = self.huber_loss(decoded, target).mean(dim=1, keepdim=True)
        else:
            pixel_loss = (decoded - target).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]

        if mask is not None:
            l2_loss = (pixel_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            l2_loss = pixel_loss.mean()

        # --- Cosine similarity loss ---
        cosine_sim = F.cosine_similarity(decoded, target, dim=1).unsqueeze(1)  # [B,1,H,W]
        cosine_dist = 1.0 - cosine_sim

        if mask is not None:
            cosine_loss = (cosine_dist * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            cosine_loss = cosine_dist.mean()

        # --- Combine ---
        if self.huber_weight > 0:
            total = (self.huber_weight * l2_loss) + (self.cosine_weight * cosine_loss)
        else:
            total = (self.l2_weight * l2_loss) + (self.cosine_weight * cosine_loss)

        total = total + self.channel_std_weight * channel_std_loss

        result = {
            'total': total,
            'l2': l2_loss,
            'cosine': cosine_loss,
        }
        if self.channel_std_weight > 0:
            result['channel_std'] = channel_std_loss
        return result


class CompactDistillationLoss(nn.Module):
    """Loss for supervising the compact (pre-HCD-decoder) feature embeddings.

    Applied directly on the low-dimensional 3DGS rendered features before
    they are decoded back to 1280d, encouraging faithful compression.
    """

    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        if loss_type not in ('l2', 'cosine'):
            raise ValueError(f"Unsupported loss_type '{loss_type}', expected 'l2' or 'cosine'")
        self.loss_type = loss_type

    def forward(
        self,
        rendered_compact: torch.Tensor,
        gt_compact: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute compact-space loss.

        Args:
            rendered_compact: [B, D_compact, H, W] rendered compact features.
            gt_compact: [B, D_compact, H, W] ground truth compact features.
            mask: [B, 1, H, W] optional validity mask.

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == 'l2':
            pixel_loss = (rendered_compact - gt_compact).pow(2).mean(dim=1, keepdim=True)
        else:
            cosine_sim = F.cosine_similarity(rendered_compact, gt_compact, dim=1)
            pixel_loss = (1.0 - cosine_sim).unsqueeze(1)  # [B,1,H,W]

        if mask is not None:
            return (pixel_loss * mask).sum() / mask.sum().clamp(min=1.0)
        return pixel_loss.mean()


class MultiViewConsistencyLoss(nn.Module):
    """FeatSharp-3D inspired multi-view consistency loss.

    Warps source-view features into the reference frame using depth and
    relative pose, then penalises feature disagreement at visible pixels.
    """

    def __init__(
        self,
        warp_mode: str = 'bilinear',
        consistency_weight: float = 1.0,
    ):
        super().__init__()
        self.warp_mode = warp_mode
        self.consistency_weight = consistency_weight

    @staticmethod
    def warp_features(
        feat_src: torch.Tensor,
        depth_ref: torch.Tensor,
        viewmat_ref: torch.Tensor,
        viewmat_src: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple:
        """Warp source features into the reference view.

        Args:
            feat_src: [B, C, H, W] source view features.
            depth_ref: [B, H, W] reference view depth.
            viewmat_ref: [B, 4, 4] reference world-to-camera matrix.
            viewmat_src: [B, 4, 4] source world-to-camera matrix.
            K: [3, 3] camera intrinsics.

        Returns:
            warped_feat: [B, C, H, W] source features warped to ref frame.
            valid_mask: [B, 1, H, W] boolean mask of valid warp coords.
        """
        B, C, H, W = feat_src.shape
        device = feat_src.device

        # Build pixel grid in reference view
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=depth_ref.dtype),
            torch.arange(W, device=device, dtype=depth_ref.dtype),
            indexing='ij',
        )
        ones = torch.ones_like(u)
        pixel_coords = torch.stack([u, v, ones], dim=0)  # [3, H, W]

        # Unproject reference pixels to 3-D (camera frame)
        K_inv = torch.inverse(K.to(device))  # [3, 3]
        cam_points = K_inv @ pixel_coords.reshape(3, -1)  # [3, HW]
        cam_points = cam_points * depth_ref.reshape(B, 1, -1)  # [B, 3, HW]

        # Homogeneous coordinates
        ones_h = torch.ones(B, 1, H * W, device=device, dtype=cam_points.dtype)
        cam_points_h = torch.cat([cam_points, ones_h], dim=1)  # [B, 4, HW]

        # Reference camera → world → source camera
        ref_to_world = torch.inverse(viewmat_ref)  # [B, 4, 4]
        relative = viewmat_src @ ref_to_world  # [B, 4, 4]
        src_points = relative @ cam_points_h  # [B, 4, HW]
        src_points = src_points[:, :3, :]  # [B, 3, HW]

        # Project into source image
        K_batch = K.unsqueeze(0).expand(B, -1, -1).to(device)
        proj = K_batch @ src_points  # [B, 3, HW]
        z = proj[:, 2:3, :].clamp(min=1e-6)
        uv = proj[:, :2, :] / z  # [B, 2, HW]

        # Normalise to [-1, 1] for grid_sample
        grid_x = 2.0 * uv[:, 0, :] / (W - 1) - 1.0
        grid_y = 2.0 * uv[:, 1, :] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, H, W, 2)

        # Validity: within image bounds and positive depth
        valid = (
            (grid_x > -1) & (grid_x < 1)
            & (grid_y > -1) & (grid_y < 1)
            & (z.squeeze(1) > 0)
        ).reshape(B, 1, H, W).float()

        warped = F.grid_sample(
            feat_src, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
        )
        return warped, valid

    def forward(
        self,
        features_ref: torch.Tensor,
        features_src: torch.Tensor,
        depth_ref: torch.Tensor,
        viewmat_ref: torch.Tensor,
        viewmat_src: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-view consistency loss.

        Args:
            features_ref: [B, C, H, W] features rendered from reference view.
            features_src: [B, C, H, W] features rendered from source view.
            depth_ref: [B, H, W] depth from reference view.
            viewmat_ref: [B, 4, 4] reference camera matrix.
            viewmat_src: [B, 4, 4] source camera matrix.
            K: [3, 3] intrinsics matrix.

        Returns:
            Scalar consistency loss.
        """
        warped_src, valid_mask = self.warp_features(
            features_src, depth_ref, viewmat_ref, viewmat_src, K,
        )

        diff = (features_ref - warped_src).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
        loss = (diff * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        return self.consistency_weight * loss


class AdaptorAlignmentLoss(nn.Module):
    """Ensures decoded features are compatible with RADIO's pre-trained adaptors.

    Passes both decoded and ground-truth features through the same adaptor
    network, then penalises output disagreement.
    """

    def __init__(self, adaptor_type: str = 'siglip2', weight: float = 0.1):
        super().__init__()
        self.adaptor_type = adaptor_type
        self.weight = weight

    def forward(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
        adaptor_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute adaptor alignment loss.

        Args:
            decoded: [B, 1280, H, W] decoded features.
            target: [B, 1280, H, W] ground truth RADIO features.
            adaptor_fn: callable mapping [B, 1280, H, W] → adaptor output.

        Returns:
            Scalar loss weighted by ``self.weight``.
        """
        out_decoded = adaptor_fn(decoded)
        out_target = adaptor_fn(target)
        loss = F.mse_loss(out_decoded, out_target)
        return self.weight * loss


class TotalVariationLoss(nn.Module):
    """Spatial smoothness regularisation on rendered feature maps."""

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Compute TV loss.

        Args:
            feature_map: [B, C, H, W] rendered features.

        Returns:
            Scalar TV loss.
        """
        diff_h = (feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :]).abs().mean()
        diff_w = (feature_map[:, :, :, 1:] - feature_map[:, :, :, :-1]).abs().mean()
        return diff_h + diff_w


class GradientWeightedLoss(nn.Module):
    """Edge-aware distillation loss that upweights high-gradient (boundary) regions.

    Uses Sobel filtering on GT features (channel-mean) to detect edges, then
    produces a per-pixel weight map that focuses the L2/cosine loss on boundaries
    where sharpness matters most.
    """

    def __init__(self, base_weight: float = 1.0, edge_multiplier: float = 3.0):
        super().__init__()
        self.base_weight = base_weight
        self.edge_multiplier = edge_multiplier
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_edge_weight(self, target: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel edge weight from GT features.

        Args:
            target: [B, C, H, W] ground truth features.

        Returns:
            [B, 1, H, W] weight map in [base_weight, base_weight + edge_multiplier].
        """
        gray = target.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = (gx.pow(2) + gy.pow(2)).sqrt()  # [B, 1, H, W]
        # Normalize to [0, 1] per-image
        mag_max = magnitude.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
        norm_mag = magnitude / mag_max
        return self.base_weight + self.edge_multiplier * norm_mag

    def forward(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient-weighted L2 loss focusing on boundary regions.

        Args:
            decoded: [B, C, H, W] predicted features.
            target: [B, C, H, W] ground truth features.

        Returns:
            Scalar loss.
        """
        weight = self._compute_edge_weight(target.detach())  # [B, 1, H, W]
        pixel_err = (decoded - target).pow(2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        return (pixel_err * weight).mean()


class GeometricEdgeAlignmentLoss(nn.Module):
    """Align rendered feature boundaries with geometric depth / alpha edges."""

    def __init__(self, alpha_weight: float = 0.5):
        super().__init__()
        self.alpha_weight = alpha_weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()
        mag_max = mag.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
        return mag / mag_max

    def forward(
        self,
        features: torch.Tensor,
        geom_depth: torch.Tensor,
        alpha_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat_scalar = features.float().norm(dim=1, keepdim=True)
        depth = geom_depth.float()
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth.mean(dim=1, keepdim=True)
        if alpha_map is not None:
            alpha = alpha_map.float()
            if alpha.dim() == 3:
                alpha = alpha.unsqueeze(1)
            if alpha.dim() == 4 and alpha.shape[1] != 1:
                alpha = alpha.mean(dim=1, keepdim=True)
        else:
            alpha = None

        feat_edges = self._edge_map(feat_scalar)
        depth_edges = self._edge_map(depth)
        edge_weight = 1.0 + depth_edges
        edge_loss = (feat_edges - depth_edges).abs() * edge_weight

        if alpha is not None:
            alpha_edges = self._edge_map(alpha.clamp(0.0, 1.0))
            reliability = alpha.clamp(0.0, 1.0) + self.alpha_weight * alpha_edges
            edge_loss = edge_loss * reliability.clamp(min=0.1)

        return edge_loss.mean()


class DepthGuidedFeatureLoss(nn.Module):
    """Enforce feature spatial structure to match geometry depth edges.

    Where the geometric depth is smooth, features should also be smooth.
    Where the geometric depth has edges, features are allowed to be sharp.
    This transfers 3DGS geometric quality into the feature field.
    """

    def __init__(self, smoothness_weight: float = 1.0, epsilon: float = 1e-3):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.epsilon = epsilon

    def forward(
        self,
        features: torch.Tensor,
        geom_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Compute depth-guided feature smoothness loss.

        Args:
            features: [B, C, H, W] rendered feature map (compact or decoded).
            geom_depth: [B, 1, H, W] geometric depth from 3DGS rendering.

        Returns:
            Scalar loss.
        """
        # Feature gradients (horizontal and vertical)
        feat_dx = (features[:, :, :, 1:] - features[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        feat_dy = (features[:, :, 1:, :] - features[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        # Depth gradients → edge weight (high gradient = edge = allow feature variation)
        depth_dx = (geom_depth[:, :, :, 1:] - geom_depth[:, :, :, :-1]).abs()
        depth_dy = (geom_depth[:, :, 1:, :] - geom_depth[:, :, :-1, :]).abs()

        # Exponential weighting: smooth depth → high weight → penalize feature gradients
        weight_x = torch.exp(-depth_dx / (depth_dx.mean() + self.epsilon))
        weight_y = torch.exp(-depth_dy / (depth_dy.mean() + self.epsilon))

        loss = (weight_x * feat_dx).mean() + (weight_y * feat_dy).mean()
        return self.smoothness_weight * loss


class BoundaryAwareFeatureLoss(nn.Module):
    """Enforce feature sharpness at depth boundaries while preserving smoothness elsewhere.

    Addresses the fundamental alpha-blending smoothing problem:
    3DGS blends features across depth discontinuities, producing blurred boundaries.
    This loss directly penalizes the *mismatch* between feature gradients and depth
    gradients, encouraging the rendered feature field to be sharp where geometry
    has edges.  Unlike DepthGuidedFeatureLoss (which only penalizes feature gradients
    in smooth regions), this also *encourages* feature gradients at depth boundaries.

    The loss has two terms:
        1. Sharpness term: at depth edges, feature gradients should be large
        2. Smoothness term: at smooth depth regions, feature gradients should be small

    Args:
        sharpness_weight: Scale for the boundary sharpness encouragement term.
        smoothness_weight: Scale for the smooth-region feature penalty.
        edge_threshold: Normalized depth gradient above which a pixel is considered a boundary.
        temperature: Softness of the edge/smooth boundary (higher = sharper transition).
    """

    def __init__(
        self,
        sharpness_weight: float = 1.0,
        smoothness_weight: float = 1.0,
        edge_threshold: float = 0.1,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.sharpness_weight = sharpness_weight
        self.smoothness_weight = smoothness_weight
        self.edge_threshold = edge_threshold
        self.temperature = temperature
        # Sobel filters for robust gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel filters.

        Args:
            x: [B, 1, H, W] single-channel map.
        Returns:
            [B, 1, H, W] gradient magnitude.
        """
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()

    def forward(
        self,
        features: torch.Tensor,
        gt_features: torch.Tensor,
        geom_depth: torch.Tensor,
        alpha_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute boundary-aware feature loss.

        Args:
            features: [B, C, H, W] rendered/decoded feature map.
            gt_features: [B, C, H, W] ground truth feature map.
            geom_depth: [B, 1, H, W] geometric depth from 3DGS.
            alpha_map: [B, 1, H, W] optional alpha/opacity map.

        Returns:
            Scalar loss.
        """
        # Compute depth edge map (normalized)
        depth = geom_depth.float()
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] != 1:
            depth = depth.mean(dim=1, keepdim=True)
        depth_grad = self._compute_gradients(depth)
        # Normalize to [0, 1]
        dg_max = depth_grad.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
        depth_edge = depth_grad / dg_max  # [B, 1, H, W]

        # Soft edge mask via sigmoid
        edge_mask = torch.sigmoid(self.temperature * (depth_edge - self.edge_threshold))
        smooth_mask = 1.0 - edge_mask

        # Apply alpha masking if available
        if alpha_map is not None:
            alpha = alpha_map.float()
            if alpha.dim() == 3:
                alpha = alpha.unsqueeze(1)
            if alpha.shape[-2:] != features.shape[-2:]:
                alpha = F.interpolate(alpha, size=features.shape[-2:], mode='bilinear', align_corners=False)
            visibility = (alpha > 0.05).float()
            edge_mask = edge_mask * visibility
            smooth_mask = smooth_mask * visibility

        # Feature error map: per-pixel L1 between predicted and GT features
        feat_error = (features - gt_features).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Feature gradient magnitude (channel-mean)
        feat_scalar = features.float().norm(dim=1, keepdim=True)
        feat_grad = self._compute_gradients(feat_scalar)
        # Normalize
        fg_max = feat_grad.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1).clamp(min=1e-6)
        feat_edge = feat_grad / fg_max

        # Term 1: Sharpness — at depth boundaries, encourage feature gradients
        # Penalize *low* feature gradients at depth edges
        sharpness_loss = (edge_mask * (1.0 - feat_edge)).mean()

        # Term 2: Smoothness — at smooth depth, penalize feature error
        smoothness_loss = (smooth_mask * feat_error).mean()

        return (
            self.sharpness_weight * sharpness_loss
            + self.smoothness_weight * smoothness_loss
        )


class RadioGSLoss(nn.Module):
    """Combined loss manager for RADIO-GS training.

    Wraps all individual loss components and applies config-driven weights.

    Config dict keys (all optional, defaults shown):
        distillation_l2_weight: 1.0
        distillation_cosine_weight: 0.5
        distillation_huber_weight: 0.0
        distillation_huber_delta: 0.1
        normalize_features: True
        compact_loss_type: 'l2'
        compact_weight: 0.5
        consistency_weight: 1.0
        consistency_warp_mode: 'bilinear'
        adaptor_type: 'siglip2'
        adaptor_weight: 0.1
        tv_weight: 0.01
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.distillation = DistillationLoss(
            l2_weight=config.get('distillation_l2_weight', 1.0),
            cosine_weight=config.get('distillation_cosine_weight', 0.5),
            huber_weight=config.get('distillation_huber_weight', 0.0),
            huber_delta=config.get('distillation_huber_delta', 0.1),
            normalize_features=config.get('normalize_features', True),
        )
        self.compact_weight = config.get('compact_weight', 0.5)
        self.compact_loss = CompactDistillationLoss(
            loss_type=config.get('compact_loss_type', 'l2'),
        )
        self.consistency = MultiViewConsistencyLoss(
            warp_mode=config.get('consistency_warp_mode', 'bilinear'),
            consistency_weight=config.get('consistency_weight', 1.0),
        )
        self.adaptor_loss = AdaptorAlignmentLoss(
            adaptor_type=config.get('adaptor_type', 'siglip2'),
            weight=config.get('adaptor_weight', 0.1),
        )
        self.tv_weight = config.get('tv_weight', 0.01)
        self.tv_loss = TotalVariationLoss()

    def forward(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
        rendered_compact: Optional[torch.Tensor] = None,
        gt_compact: Optional[torch.Tensor] = None,
        features_src: Optional[torch.Tensor] = None,
        depth_ref: Optional[torch.Tensor] = None,
        viewmats: Optional[tuple] = None,
        K: Optional[torch.Tensor] = None,
        adaptor_fn: Optional[Callable] = None,
        feature_map: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all applicable losses.

        Args:
            decoded: [B, 1280, H, W] decoded features.
            target: [B, 1280, H, W] GT RADIO features.
            rendered_compact: [B, D_compact, H, W] optional compact features.
            gt_compact: [B, D_compact, H, W] optional GT compact features.
            features_src: [B, C, H, W] source-view features for consistency.
            depth_ref: [B, H, W] reference depth for warping.
            viewmats: (viewmat_ref, viewmat_src) each [B, 4, 4].
            K: [3, 3] intrinsics.
            adaptor_fn: callable for adaptor alignment.
            feature_map: [B, C, H, W] rendered map for TV regularisation.
            mask: [B, 1, H, W] validity mask.

        Returns:
            Dict with 'total' and per-component loss tensors.
        """
        device = decoded.device
        losses: Dict[str, torch.Tensor] = {}

        # Core distillation
        dist = self.distillation(decoded, target, mask=mask)
        losses['distillation'] = dist['total']
        losses['distillation_l2'] = dist['l2']
        losses['distillation_cosine'] = dist['cosine']
        total = dist['total']

        # Compact-space supervision
        if rendered_compact is not None and gt_compact is not None:
            c_loss = self.compact_loss(rendered_compact, gt_compact, mask=mask)
            losses['compact'] = c_loss
            total = total + self.compact_weight * c_loss

        # Multi-view consistency
        if (
            features_src is not None
            and depth_ref is not None
            and viewmats is not None
            and K is not None
        ):
            viewmat_ref, viewmat_src = viewmats
            mv_loss = self.consistency(
                decoded, features_src, depth_ref, viewmat_ref, viewmat_src, K,
            )
            losses['consistency'] = mv_loss
            total = total + mv_loss

        # Adaptor alignment
        if adaptor_fn is not None:
            a_loss = self.adaptor_loss(decoded, target, adaptor_fn)
            losses['adaptor'] = a_loss
            total = total + a_loss

        # Total variation
        fm = feature_map if feature_map is not None else decoded
        tv = self.tv_loss(fm)
        losses['tv'] = tv
        total = total + self.tv_weight * tv

        losses['total'] = total
        return losses
