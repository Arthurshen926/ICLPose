"""
FlowFeat Multi-Scale Feature Extractor

Uses the FINAL output of FlowFeat's DPT decoder (full 4-level refinement +
output convolutions + LayerNorm) and downsamples to multiple scales.

FlowFeat architecture (DINOv2 ViT-B/14 backbone + DPT decoder):
  Encoder: 4 hooked ViT blocks → postprocess → scratch projection (128d)
  Decoder: refinenet4→3→2→1 (each 2× upsample + residual fusion)
           → output_conv0 → bilinear upsample to input res
           → output_conv1 (Conv-ReLU-Conv) → LayerNorm
  Final output: 128d @ input resolution (e.g. 560×980 for OH 0.5×)

Multi-scale is created by downsampling the single final output:
  - coarse: 128d @ target_resolutions['coarse']  (e.g. 20×35)
  - mid:    128d @ target_resolutions['mid']      (e.g. 40×70)
  - fine:   128d @ target_resolutions['fine']     (e.g. 80×140)

Usage:
    extractor = FlowFeatExtractor(device='cuda:0')
    feats = extractor.extract(image_tensor)
    # feats = {'coarse': (128, 20, 35), 'mid': (128, 40, 70), 'fine': (128, 80, 140)}
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# FlowFeat repo
FLOWFEAT_ROOT = os.environ.get("FLOWFEAT_ROOT", "/root/flowfeat")
if FLOWFEAT_ROOT not in sys.path:
    sys.path.insert(0, FLOWFEAT_ROOT)


class FlowFeatExtractor:
    """
    Multi-scale feature extractor using FlowFeat's pretrained DPT decoder
    over a frozen DINOv2 ViT-B/14 encoder.

    Extracts the FINAL 128d output (full 4-level DPT fusion + LayerNorm)
    and downsamples to 3 target scales via adaptive average pooling.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Default target resolutions for OldHospital (1920×1080 @ 0.5×)
    DEFAULT_RESOLUTIONS = {
        'coarse': (20, 35),
        'mid': (40, 70),
        'fine': (80, 140),
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14_yt",
        device: str = "cuda:0",
        target_resolutions: dict = None,
    ):
        """
        Args:
            model_name: FlowFeat hub model name (e.g. 'dinov2_vitb14_yt')
            device: torch device
            target_resolutions: Dict mapping scale names to (H, W) tuples.
                Features are adaptively pooled to these sizes.
                e.g. {'coarse': (20, 35), 'mid': (40, 70), 'fine': (80, 140)}
        """
        self.device = torch.device(device)
        self.patch_size = 14
        self.target_resolutions = target_resolutions or self.DEFAULT_RESOLUTIONS

        # Load pretrained FlowFeat
        from hubconf import flowfeat
        self.model = flowfeat(model_name, pretrained=True, map_location=device)
        self.model = self.model.to(self.device).eval()

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.normalize = T.Normalize(
            mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
        )

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[FlowFeatExtractor] Loaded {model_name}: {n_params:.1f}M params on {device}")

    def _pad_to_patch(self, h: int, w: int):
        """Compute padding to make h, w divisible by patch_size,
        and ensure the patch grid has even dimensions (for DPT stride-2 layer)."""
        ps = self.patch_size
        # Pad to patch_size-divisible
        new_h = math.ceil(h / ps) * ps
        new_w = math.ceil(w / ps) * ps
        # Ensure even patch grid (needed for DPT postprocess4 stride-2)
        grid_h = new_h // ps
        grid_w = new_w // ps
        if grid_h % 2 != 0:
            new_h += ps
        if grid_w % 2 != 0:
            new_w += ps
        return new_h, new_w

    def load_image(self, image_path: str, resize_factor: float = 0.5) -> torch.Tensor:
        """Load and preprocess image.

        Args:
            image_path: path to image file
            resize_factor: scale factor relative to original resolution

        Returns:
            Normalized tensor (1, 3, H_padded, W_padded) on device
        """
        img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img.size

        # Resize
        h_new = int(h_orig * resize_factor)
        w_new = int(w_orig * resize_factor)
        img = img.resize((w_new, h_new), Image.BILINEAR)

        # Convert to tensor and normalize
        x = T.functional.to_tensor(img)  # (3, h_new, w_new)
        x = self.normalize(x)

        # Pad to patch-aligned
        h_pad, w_pad = self._pad_to_patch(h_new, w_new)
        if h_pad != h_new or w_pad != w_new:
            x = F.pad(x, (0, w_pad - w_new, 0, h_pad - h_new), mode="reflect")

        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> dict:
        """Extract multi-scale features from a preprocessed image tensor.

        Runs the full FlowFeat DPT decoder (4-level refinement + output conv +
        LayerNorm) to produce a single 128d feature map at input resolution,
        then downsamples to target scales via adaptive average pooling.

        Args:
            x: (1, 3, H, W) normalized image tensor. H, W must be
               divisible by patch_size, with even patch grid dimensions.

        Returns:
            Dict with keys 'coarse', 'mid', 'fine', each a (128, H_s, W_s) CPU tensor.
        """
        assert x.dim() == 4 and x.shape[0] == 1

        # Full FlowFeat forward: encoder + complete DPT decoder
        enc = self.model.encoder
        dec = self.model.decoder
        enc(x)
        final = dec(enc, x.shape[2:], with_norm=True)  # (1, 128, H, W)

        # Downsample to target resolutions
        feats = {}
        for scale, (th, tw) in self.target_resolutions.items():
            feats[scale] = F.adaptive_avg_pool2d(final, (th, tw)).squeeze(0)

        return {k: v.cpu() for k, v in feats.items()}

    @torch.no_grad()
    def extract_from_path(
        self, image_path: str, resize_factor: float = 0.5
    ) -> dict:
        """Convenience: load image + extract features.

        Args:
            image_path: path to image file
            resize_factor: resize relative to original (0.5 = half-res)

        Returns:
            Dict with 'coarse', 'mid', 'fine' feature tensors.
        """
        x = self.load_image(image_path, resize_factor=resize_factor)
        return self.extract(x)
