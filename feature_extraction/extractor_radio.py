"""
RADIO (C-RADIOv4-H) Feature Extractor

Extracts:
  - Local features: [1280, H/16, W/16] spatial patch tokens
  - Global summary:  [2560] CLS-like summary vector

Model: C-RADIOv4-H (ViT-H/16, 653M params)
  Input: RGB [0,1], resolution divisible by 16 (auto-adjusted)
  Output local:  1280-d spatial features at H/16 × W/16
  Output global: 2560-d summary vector (concatenated CLS tokens)
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path


class RADIOFeatureExtractor:
    """Extract features from NVIDIA RADIO model."""

    def __init__(self, version='c-radio_v4-h', device='cuda',
                 radio_repo='/root/RADIO'):
        self.device = torch.device(device)
        self.radio_repo = radio_repo

        print(f"Loading RADIO {version}...")
        self.model = torch.hub.load(
            radio_repo, 'radio_model',
            version=version, source='local', skip_validation=True
        )
        self.model = self.model.to(self.device).eval()
        self.patch_size = self.model.patch_size  # 16 for v4-H
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  RADIO loaded: {n_params:.0f}M params, patch_size={self.patch_size}")

    @torch.no_grad()
    def extract(self, image_tensor):
        """
        Extract local + global features from a single image.

        Args:
            image_tensor: (1, 3, H, W) float tensor in [0, 1]

        Returns:
            dict with:
              'local':   (1280, Hp, Wp) float32 tensor, Hp=H/16, Wp=W/16
              'summary': (2560,) float32 tensor
        """
        _, _, H, W = image_tensor.shape

        # Adjust to nearest supported resolution (divisible by patch_size)
        nearest = self.model.get_nearest_supported_resolution(H, W)
        target_H, target_W = nearest.height, nearest.width
        if target_H != H or target_W != W:
            image_tensor = F.interpolate(
                image_tensor, (target_H, target_W),
                mode='bilinear', align_corners=False
            )

        image_tensor = image_tensor.to(self.device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            summary, features = self.model(image_tensor, feature_fmt='NCHW')

        # features: (1, 1280, Hp, Wp), summary: (1, 2560)
        local_feat = features.squeeze(0).float()   # (1280, Hp, Wp)
        summary_vec = summary.squeeze(0).float()    # (2560,)

        return {
            'local': local_feat,
            'summary': summary_vec,
        }

    @torch.no_grad()
    def extract_batch(self, image_tensors):
        """
        Extract features from a batch of images (same resolution).

        Args:
            image_tensors: (B, 3, H, W) float tensor in [0, 1]

        Returns:
            dict with:
              'local':   (B, 1280, Hp, Wp) float32
              'summary': (B, 2560) float32
        """
        B, _, H, W = image_tensors.shape

        nearest = self.model.get_nearest_supported_resolution(H, W)
        target_H, target_W = nearest.height, nearest.width
        if target_H != H or target_W != W:
            image_tensors = F.interpolate(
                image_tensors, (target_H, target_W),
                mode='bilinear', align_corners=False
            )

        image_tensors = image_tensors.to(self.device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            summary, features = self.model(image_tensors, feature_fmt='NCHW')

        return {
            'local': features.float(),     # (B, 1280, Hp, Wp)
            'summary': summary.float(),     # (B, 2560)
        }
