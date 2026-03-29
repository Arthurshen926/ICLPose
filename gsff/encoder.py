"""
2D Feature Encoder for GSFFs.

Paper uses a ViT with patch size 14 (DINOv2) for coarse features and 
full-resolution features for fine level. Both output d=16 dimensional features.

Coarse encoder: DINOv2 ViT-B/14 backbone → linear head → d=16 @ (H/14, W/14)
Fine encoder: Lightweight CNN that upsamples ViT features to full resolution → d=16

Both are jointly trained with the triplane feature field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseEncoder(nn.Module):
    """
    Coarse 2D feature encoder based on DINOv2 ViT-B/14.
    
    Extracts patch-level features at (H/14, W/14) resolution, projected to d dimensions.
    The backbone can be frozen or fine-tuned.
    """
    
    def __init__(self, feature_dim: int = 16, backbone_name: str = 'dinov2_vitb14',
                 freeze_backbone: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = 14
        self.freeze_backbone = freeze_backbone
        
        # Load DINOv2 backbone
        self.backbone = self._load_backbone(backbone_name)
        self.backbone_dim = 768  # ViT-B/14
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        # Projection head: 768 → d
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim // 4),
            nn.GELU(),
            nn.Linear(self.backbone_dim // 4, feature_dim),
        )
    
    def _load_backbone(self, name: str):
        """Load DINOv2 from local cache or hub."""
        import os
        local_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        if os.path.isdir(local_dir):
            model = torch.hub.load(local_dir, name, source='local')
        else:
            model = torch.hub.load('facebookresearch/dinov2', name)
        return model
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract coarse features from images.
        
        Args:
            images: [B, 3, H, W] normalized images (ImageNet normalization)
            
        Returns:
            features: [B, D, H', W'] coarse feature maps where H'=H/14, W'=W/14
        """
        B, C, H, W = images.shape
        
        # Pad to be divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = images.shape[2], images.shape[3]
        h_patches = H_pad // self.patch_size
        w_patches = W_pad // self.patch_size
        
        # Extract patch features
        if self.freeze_backbone:
            with torch.no_grad():
                patch_features = self.backbone.forward_features(images)
                patch_tokens = patch_features['x_norm_patchtokens']  # [B, N, 768]
        else:
            patch_features = self.backbone.forward_features(images)
            patch_tokens = patch_features['x_norm_patchtokens']
        
        # Project to target dimension
        features = self.proj(patch_tokens)  # [B, N, D]
        
        # Reshape to spatial
        features = features.permute(0, 2, 1).reshape(B, self.feature_dim, h_patches, w_patches)
        
        return features
    
    def train(self, mode=True):
        """Override to keep backbone frozen when needed."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class FineEncoder(nn.Module):
    """
    Fine 2D feature encoder with DINOv2 feature guidance.
    
    Combines low-level image features from a CNN with upsampled DINOv2 patch 
    tokens for semantic guidance. This gives spatially precise features that 
    are also semantically meaningful.
    """
    
    def __init__(self, feature_dim: int = 16, backbone_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Low-level CNN branch: captures edges, textures at full resolution
        self.low_level = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
        )
        
        # DINOv2 feature projection (will be upsampled to full res)
        self.dino_proj = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.GELU(),
        )
        
        # Fusion: combine low-level (64) + upsampled DINOv2 (64) → feature_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, feature_dim, 1),
        )
    
    def forward(self, images: torch.Tensor, dino_patch_tokens: torch.Tensor = None,
                patch_h: int = 0, patch_w: int = 0) -> torch.Tensor:
        """
        Extract fine features from images with optional DINOv2 guidance.
        
        Args:
            images: [B, 3, H, W] normalized images
            dino_patch_tokens: [B, N, 768] DINOv2 patch tokens (optional)
            patch_h, patch_w: spatial dimensions of patch tokens
            
        Returns:
            features: [B, D, H, W] full-resolution feature maps
        """
        B, C, H, W = images.shape
        low_feat = self.low_level(images)  # [B, 64, H, W]
        
        if dino_patch_tokens is not None:
            # Project and reshape DINOv2 tokens to spatial
            dino_feat = self.dino_proj(dino_patch_tokens)  # [B, N, 64]
            dino_feat = dino_feat.permute(0, 2, 1).reshape(B, 64, patch_h, patch_w)
            # Upsample to full resolution
            dino_up = F.interpolate(dino_feat, (H, W), mode='bilinear', align_corners=False)
            # Concatenate and fuse
            fused = torch.cat([low_feat, dino_up], dim=1)  # [B, 128, H, W]
            return self.fusion(fused)
        else:
            # Fallback: CNN-only (no DINOv2 guidance)
            return self.fusion(torch.cat([low_feat, torch.zeros_like(low_feat)], dim=1))


class OldFineEncoder(nn.Module):
    """Original fine encoder (v1) for backward compatibility with old checkpoints."""
    
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, feature_dim, 1),
        )
    
    def forward(self, images, dino_patch_tokens=None, patch_h=0, patch_w=0):
        return self.encoder(images)


class DualScaleEncoder(nn.Module):
    """Combined coarse + fine 2D encoder with shared DINOv2 backbone."""
    
    def __init__(self, feature_dim: int = 16, freeze_backbone: bool = True,
                 use_old_fine_encoder: bool = False):
        super().__init__()
        self.coarse_encoder = CoarseEncoder(feature_dim, freeze_backbone=freeze_backbone)
        if use_old_fine_encoder:
            self.fine_encoder = OldFineEncoder(feature_dim)
        else:
            self.fine_encoder = FineEncoder(feature_dim, backbone_dim=self.coarse_encoder.backbone_dim)
    
    def forward(self, images: torch.Tensor):
        """
        Returns:
            coarse_features: [B, D, H/14, W/14]
            fine_features: [B, D, H, W]
        """
        B, C, H, W = images.shape
        ps = self.coarse_encoder.patch_size
        
        # Pad for DINOv2
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        images_padded = images
        if pad_h > 0 or pad_w > 0:
            images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = images_padded.shape[2], images_padded.shape[3]
        h_patches = H_pad // ps
        w_patches = W_pad // ps
        
        # Extract DINOv2 patch tokens (shared between coarse and fine)
        if self.coarse_encoder.freeze_backbone:
            with torch.no_grad():
                patch_features = self.coarse_encoder.backbone.forward_features(images_padded)
                patch_tokens = patch_features['x_norm_patchtokens']
        else:
            patch_features = self.coarse_encoder.backbone.forward_features(images_padded)
            patch_tokens = patch_features['x_norm_patchtokens']
        
        # Coarse: project patch tokens
        coarse_feat = self.coarse_encoder.proj(patch_tokens)
        coarse_feat = coarse_feat.permute(0, 2, 1).reshape(
            B, self.coarse_encoder.feature_dim, h_patches, w_patches)
        
        # Fine: CNN + upsampled DINOv2 guidance
        fine_feat = self.fine_encoder(images, patch_tokens.detach(), h_patches, w_patches)
        
        return coarse_feat, fine_feat
    
    def train(self, mode=True):
        super().train(mode)
        self.coarse_encoder.train(mode)
        return self


class SegmentationHead(nn.Module):
    """
    Lightweight classification head for privacy-preserving segmentation.
    Predicts K-class segmentation from 2D encoder features.
    """
    
    def __init__(self, feature_dim: int = 16, num_classes: int = 34):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D, H, W]
        Returns:
            logits: [B, K, H, W]
        """
        return self.head(features)
