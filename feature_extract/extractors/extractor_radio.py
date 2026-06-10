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
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from pathlib import Path

from feature_extract.utils.radio_loader import load_radio_model


def _radio_output_features(output):
    """Return the feature tensor from RADIO's final/intermediate output forms."""

    if hasattr(output, "features"):
        return output.features
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        return output[1]
    return output


class RADIOFeatureExtractor:
    """Extract features from NVIDIA RADIO model."""

    def __init__(self, version='c-radio_v4-h', device='cuda',
                 radio_repo='feature_extract/checkpoints/RADIO'):
        self.device = torch.device(device)
        self.radio_repo = radio_repo

        print(f"Loading RADIO {version}...")
        self.model = load_radio_model(version=version, radio_repo=radio_repo)
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

        autocast_context = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else nullcontext()
        )
        with autocast_context:
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

        autocast_context = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else nullcontext()
        )
        with autocast_context:
            summary, features = self.model(image_tensors, feature_fmt='NCHW')

        return {
            'local': features.float(),     # (B, 1280, Hp, Wp)
            'summary': summary.float(),     # (B, 2560)
        }

    @torch.no_grad()
    def extract_dual(
        self,
        image_tensor,
        *,
        fine_intermediate_index: int = -6,
        coarse_source: str = "final",
        coarse_intermediate_index: int = -1,
        norm_intermediates: bool = True,
        aggregation: str = "sparse",
    ):
        """Extract RADIO intermediate/final maps as `[fine, coarse, dual]`.

        The spatial stride remains RADIO's patch stride. The split is semantic:
        an earlier/middle layer is used as the local geometry branch and the
        final or late layer is used as the coarse semantic branch.
        """

        _, _, H, W = image_tensor.shape
        nearest = self.model.get_nearest_supported_resolution(H, W)
        target_H, target_W = nearest.height, nearest.width
        if target_H != H or target_W != W:
            image_tensor = F.interpolate(
                image_tensor, (target_H, target_W),
                mode='bilinear', align_corners=False
            )
        image_tensor = image_tensor.to(self.device)
        autocast_context = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else nullcontext()
        )
        with autocast_context:
            if str(coarse_source) == "intermediate":
                requested = [int(fine_intermediate_index)]
                if int(coarse_intermediate_index) != int(fine_intermediate_index):
                    requested.append(int(coarse_intermediate_index))
                outputs = self.model.forward_intermediates(
                    image_tensor,
                    indices=requested,
                    norm=bool(norm_intermediates),
                    stop_early=True,
                    output_fmt='NCHW',
                    intermediates_only=True,
                    aggregation=str(aggregation),
                )
                values = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
                block_count = None
                inner_model = getattr(self.model, "model", None)
                blocks = getattr(inner_model, "blocks", None)
                if blocks is not None:
                    try:
                        block_count = len(blocks)
                    except TypeError:
                        block_count = None

                def resolve(index: int) -> int:
                    return int(block_count + index) if block_count is not None and int(index) < 0 else int(index)

                ordered = sorted(requested, key=resolve)
                by_index = {int(index): _radio_output_features(value) for index, value in zip(ordered, values)}
                fine = by_index[int(fine_intermediate_index)]
                coarse = by_index[int(coarse_intermediate_index)]
            else:
                final, outputs = self.model.forward_intermediates(
                    image_tensor,
                    indices=[int(fine_intermediate_index)],
                    norm=bool(norm_intermediates),
                    stop_early=False,
                    output_fmt='NCHW',
                    intermediates_only=False,
                    aggregation=str(aggregation),
                )
                values = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
                fine = _radio_output_features(values[0])
                coarse = _radio_output_features(final)
        fine = fine.squeeze(0).float()
        coarse = coarse.squeeze(0).float()
        if fine.shape[-2:] != coarse.shape[-2:]:
            coarse = F.interpolate(coarse.unsqueeze(0), size=fine.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        return {
            'fine': fine,
            'coarse': coarse,
            'dual': torch.cat([fine, coarse], dim=0),
        }
