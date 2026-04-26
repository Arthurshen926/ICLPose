"""
Query-side Feature Extractor + Pose Head for ICLPose-loc.

This module handles query image feature extraction and pose regression:

  1. ViT-Ti Feature Extractor: Lightweight ViT-Tiny to extract query features
  2. Cross-Attention Matcher: Match query features against rendered DCFF features
  3. Pose Head: 6-DoF pose regression (rotation + translation)

Architecture:
  Query Image → ViT-Ti → Query Features
                           ↓
  Rendered DCFF Features ← Cross-Attention → Matched Features
                                                ↓
                                           Pose Head → [R|t]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class ViTQueryExtractor(nn.Module):
    """Lightweight ViT-Tiny query feature extractor.

    Uses timm's ViT-Tiny with custom projection to match DCFF feature dimension.
    Produces both per-patch features and a global CLS feature.

    Args:
        img_size: Input image size (default 224)
        patch_size: Patch size (default 16)
        feature_dim: Output feature dimension (default 64)
        pretrained: Use ImageNet pretrained weights (default True)
        freeze_backbone: Freeze ViT backbone (default False)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        feature_dim: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim

        try:
            import timm
            self.vit = timm.create_model(
                f'vit_tiny_patch{patch_size}_{img_size}',
                pretrained=pretrained,
                num_classes=0,
            )
            self.use_timm = True
            vit_dim = self.vit.embed_dim
        except Exception:
            self.use_timm = False
            vit_dim = self._build_custom_vit(img_size, patch_size)

        self.proj = nn.Linear(vit_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"  [ViTQueryExtractor] img={img_size}, patch={patch_size}, "
            f"vit_dim={vit_dim}, out={feature_dim}d, params={n_params:,}"
        )

    def _build_custom_vit(self, img_size, patch_size):
        """Fallback custom ViT-Tiny if timm unavailable."""
        embed_dim = 192
        depth = 12
        num_heads = 3
        mlp_ratio = 4.0
        return embed_dim

    def forward(self, x: torch.Tensor) -> dict:
        """Extract query features.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            dict with:
                'features': [B, feature_dim, H//patch, W//patch] per-patch features
                'cls_feature': [B, feature_dim] global CLS feature
                'patch_grid': [B, H//patch, W//patch] number of patches per dim
        """
        B, C, H, W = x.shape

        if self.use_timm:
            x_resized = x
            if H != self.img_size or W != self.img_size:
                x_resized = F.interpolate(x, (self.img_size, self.img_size),
                                         mode='bilinear', align_corners=False)

            vit_out = self.vit(x_resized)
            if vit_out.dim() == 2:
                cls_token = vit_out
                patch_tokens = None
            else:
                cls_token = vit_out[:, 0]
                patch_tokens = vit_out[:, 1:]

            H_p = W_p = self.img_size // self.patch_size
            if patch_tokens is not None:
                patch_tokens = patch_tokens.reshape(B, H_p, W_p, -1)
                patch_tokens = patch_tokens.permute(0, 3, 1, 2)
            else:
                patch_tokens = None
        else:
            raise NotImplementedError("Custom ViT fallback not implemented")

        cls_feat = self.norm(self.proj(cls_token))
        if patch_tokens is not None:
            patch_feat = self.proj(patch_tokens.permute(0, 3, 1, 2))
            patch_feat = self.norm(patch_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            patch_feat = None

        return {
            'features': patch_feat,
            'cls_feature': cls_feat,
            'patch_grid': (H // self.patch_size, W // self.patch_size),
        }


class CrossAttentionMatcher(nn.Module):
    """Cross-attention matcher between query and map features.

    Matches query image features against rendered DCFF map features
    using multi-head cross-attention. Produces matched features that
    capture both query-map correspondence and local context.

    Args:
        feature_dim: Feature channel dimension (default 64)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of attention layers (default 2)
        hidden_dim: Hidden dimension for FFN (default 128)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.proj_q = nn.Linear(feature_dim, feature_dim)
        self.proj_k = nn.Linear(feature_dim, feature_dim)
        self.proj_v = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"  [CrossAttentionMatcher] dim={feature_dim}, heads={num_heads}, "
            f"layers={num_layers}, params={n_params:,}"
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        map_feat: torch.Tensor,
        query_mask: torch.Tensor = None,
        map_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Match query features against map features.

        Args:
            query_feat: [B, C, Hq, Wq] or [B, C] query features
            map_feat: [B, C, Hm, Wm] rendered map features
            query_mask: [B, 1, Hq, Wq] optional query mask
            map_mask: [B, 1, Hm, Wm] optional map mask

        Returns:
            matched: [B, C] matched feature vector
        """
        B = query_feat.shape[0]

        if query_feat.dim() == 4:
            q = query_feat.flatten(2).permute(0, 2, 1)
        else:
            q = query_feat.unsqueeze(1)

        if map_feat.dim() == 4:
            k = map_feat.flatten(2).permute(0, 2, 1)
            v = k
        else:
            k = map_feat.unsqueeze(1)
            v = k

        if query_mask is not None:
            if query_mask.dim() == 4:
                q_mask = query_mask.flatten(2)
            else:
                q_mask = query_mask.unsqueeze(1)
        else:
            q_mask = None

        if map_mask is not None:
            k_mask = map_mask.flatten(2)[:, 0, :]
        else:
            k_mask = None

        Q = self.proj_q(q)
        K = self.proj_k(k)
        V = self.proj_v(v)

        for layer in self.layers:
            Q = layer(Q, K, V, key_padding_mask=k_mask)

        out = Q.mean(dim=1)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: MHCA + FFN."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, k, v, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(k), self.norm1(v),
                         key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PoseHead(nn.Module):
    """6-DoF pose regression head (rotation + translation).

    Takes matched features and predicts camera rotation (quaternion) and
    translation (3D vector). Uses separate heads for rotation and translation
    with auxiliary losses for geometric consistency.

    Args:
        feature_dim: Input feature dimension (default 64)
        hidden_dim: Hidden dimension for pose MLP (default 256)
        num_layers: Number of MLP layers (default 3)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        prev = feature_dim
        for i in range(num_layers - 1):
            layers.extend([nn.Linear(prev, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            prev = hidden_dim
        self.mlp = nn.Sequential(*layers)

        self.rotation_head = nn.Linear(prev, 4)
        self.translation_head = nn.Linear(prev, 3)

        self.rotation_scale = nn.Parameter(torch.tensor(0.1))
        self.translation_scale = nn.Parameter(torch.tensor(0.01))

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"  [PoseHead] in={feature_dim}, hidden={hidden_dim}, "
            f"layers={num_layers}, params={n_params:,}"
        )

    def _init_weights(self):
        xavier_uniform_(self.rotation_head.weight)
        xavier_uniform_(self.translation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)
        nn.init.zeros_(self.translation_head.bias)

    def forward(self, features: torch.Tensor) -> dict:
        """Predict 6-DoF pose.

        Args:
            features: [B, C] matched features

        Returns:
            dict with:
                'rotation': [B, 4] quaternion (w, x, y, z)
                'translation': [B, 3] translation vector
                'features': [B, hidden] intermediate features for auxiliary losses
        """
        x = self.mlp(features)

        rot = self.rotation_head(x)
        rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-6)

        trans = self.translation_head(x) * self.translation_scale.exp()

        return {
            'rotation': rot,
            'translation': trans,
            'features': x,
        }


class QueryPoseNetwork(nn.Module):
    """End-to-end query feature extraction + matching + pose regression.

    Combines:
      1. ViTQueryExtractor: Extract query image features
      2. CrossAttentionMatcher: Match query vs rendered map features
      3. PoseHead: 6-DoF pose prediction

    Args:
        feature_dim: Feature dimension (must match DCFF)
        img_size: Query image size
        matcher_config: Config dict for CrossAttentionMatcher
        pose_config: Config dict for PoseHead
    """

    def __init__(
        self,
        feature_dim: int = 64,
        img_size: int = 224,
        matcher_num_heads: int = 4,
        matcher_num_layers: int = 2,
        pose_hidden_dim: int = 256,
        pose_num_layers: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.extractor = ViTQueryExtractor(
            img_size=img_size,
            patch_size=16,
            feature_dim=feature_dim,
            pretrained=True,
            freeze_backbone=False,
        )

        self.matcher = CrossAttentionMatcher(
            feature_dim=feature_dim,
            num_heads=matcher_num_heads,
            num_layers=matcher_num_layers,
            hidden_dim=feature_dim * 2,
        )

        self.pose_head = PoseHead(
            feature_dim=feature_dim,
            hidden_dim=pose_hidden_dim,
            num_layers=pose_num_layers,
        )

    def forward(
        self,
        query_image: torch.Tensor,
        map_features: torch.Tensor,
        map_mask: torch.Tensor = None,
    ) -> dict:
        """Predict pose for query image given rendered map features.

        Args:
            query_image: [B, 3, H, W] query image
            map_features: [B, C, Hm, Wm] rendered DCFF features
            map_mask: [B, 1, Hm, Wm] optional validity mask

        Returns:
            dict with pose prediction and intermediate results
        """
        query_feat = self.extractor(query_image)

        matched_feat = self.matcher(
            query_feat['cls_feature'],
            map_features,
            map_mask=map_mask,
        )

        pose = self.pose_head(matched_feat)

        return {
            'rotation': pose['rotation'],
            'translation': pose['translation'],
            'query_cls': query_feat['cls_feature'],
            'query_patches': query_feat['features'],
            'matched_features': matched_feat,
        }


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        q: [..., 4] quaternion

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def pose_loss(
    pred_rotation: torch.Tensor,
    pred_translation: torch.Tensor,
    gt_rotation: torch.Tensor,
    gt_translation: torch.Tensor,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> dict:
    """Pose regression loss: geodesic rotation + L1 translation.

    Args:
        pred_rotation: [B, 4] predicted quaternion
        pred_translation: [B, 3] predicted translation
        gt_rotation: [B, 4] ground truth quaternion
        gt_translation: [B, 3] ground truth translation
        rotation_weight: Weight for rotation loss
        translation_weight: Weight for translation loss

    Returns:
        dict with individual and total losses
    """
    rot_loss = geodesic_rotation_loss(pred_rotation, gt_rotation)
    trans_loss = F.l1_loss(pred_translation, gt_translation)
    total = rotation_weight * rot_loss + translation_weight * trans_loss
    return {'rotation_loss': rot_loss, 'translation_loss': trans_loss, 'total': total}


def geodesic_rotation_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between quaternions as rotation loss.

    Uses the chordal distance on the unit quaternion sphere, which is
    a smooth proxy for geodesic distance.

    Args:
        q_pred: [B, 4] predicted quaternion (w, x, y, z)
        q_gt: [B, 4] ground truth quaternion

    Returns:
        scalar loss
    """
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + 1e-6)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + 1e-6)

    dot = (q_pred * q_gt).sum(dim=-1)
    dot = dot.abs().clamp_(0, 1)
    chordal = (2 * dot ** 2 - 1).clamp_(-1, 1)
    loss = (1 - chordal).mean()
    return loss
