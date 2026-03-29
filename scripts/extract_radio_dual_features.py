"""
Extract dual-scale RADIO features (shallow geometric + deep semantic).

For each training image, extracts:
  - RADIO_geo (shallow layers ~8-12): high-frequency geometric/edge features → PCA 64d
  - RADIO_sem (deep layers, final): abstract semantic features → PCA 64d

Usage:
    python scripts/extract_radio_dual_features.py \
        --source_dir dataset/OldHospital \
        --output_dir output/features_radio_dual/OldHospital \
        --target_dim 64 \
        --radio_repo /root/RADIO
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


class DualScaleRADIOExtractor:
    """Extract both shallow (geometric) and deep (semantic) RADIO features.

    ViT-H/16 has 32 transformer blocks. We hook:
      - Shallow (block ~10): captures edges, textures, geometric boundaries
      - Deep (final output): captures abstract semantics, lighting-invariant
    """

    def __init__(self, version='c-radio_v4-h', device='cuda',
                 radio_repo='/root/RADIO',
                 shallow_block=10):
        self.device = torch.device(device)
        self.shallow_block = shallow_block

        print(f"Loading RADIO {version}...")
        self.model = torch.hub.load(
            radio_repo, 'radio_model',
            version=version, source='local', skip_validation=True,
        )
        self.model = self.model.to(self.device).eval()
        self.patch_size = self.model.patch_size

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  RADIO loaded: {n_params:.0f}M params, patch_size={self.patch_size}")

        # Discover model structure and register hooks
        self._shallow_features = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hook on the shallow transformer block."""
        # RADIO model structure: model.model.blocks[i] or similar
        # Try common ViT structures
        blocks = None
        model_inner = self.model

        # Try to find the transformer blocks
        for attr_path in [
            'model.blocks', 'model.encoder.blocks', 'model.encoder.layers',
            'blocks', 'encoder.blocks', 'encoder.layers',
        ]:
            obj = model_inner
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                if hasattr(obj, '__len__') and len(obj) > self.shallow_block:
                    blocks = obj
                    print(f"  Found {len(blocks)} transformer blocks at .{attr_path}")
                    break
            except AttributeError:
                continue

        if blocks is None:
            print("  WARNING: Could not find transformer blocks for intermediate extraction.")
            print("  Will use final-layer features for both scales with different PCA.")
            self._use_single_layer = True
            return

        self._use_single_layer = False
        n_blocks = len(blocks)
        shallow_idx = min(self.shallow_block, n_blocks - 1)
        print(f"  Hooking shallow features at block {shallow_idx}/{n_blocks-1}")

        def hook_fn(module, input, output):
            # ViT block output is typically (B, N_tokens, D) or a tuple
            if isinstance(output, tuple):
                self._shallow_features = output[0].detach()
            else:
                self._shallow_features = output.detach()

        blocks[shallow_idx].register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract(self, image_tensor):
        """Extract dual-scale features from a single image.

        Args:
            image_tensor: (1, 3, H, W) float tensor in [0, 1]

        Returns:
            dict with:
                'geo': (D, Hp, Wp) shallow geometric features
                'sem': (D, Hp, Wp) deep semantic features
                'summary': (2560,) global summary
        """
        _, _, H, W = image_tensor.shape
        nearest = self.model.get_nearest_supported_resolution(H, W)
        tH, tW = nearest.height, nearest.width
        if tH != H or tW != W:
            image_tensor = F.interpolate(image_tensor, (tH, tW),
                                         mode='bilinear', align_corners=False)
        image_tensor = image_tensor.to(self.device)

        Hp = tH // self.patch_size
        Wp = tW // self.patch_size

        self._shallow_features = None

        with torch.autocast('cuda', dtype=torch.bfloat16):
            summary, features = self.model(image_tensor, feature_fmt='NCHW')

        # Deep (semantic): final layer output
        sem = features.squeeze(0).float()  # (D, Hp, Wp)

        # Shallow (geometric): from hook
        if hasattr(self, '_use_single_layer') and self._use_single_layer:
            # Fallback: use same features for both
            geo = sem.clone()
        else:
            shallow = self._shallow_features  # (1, N_tokens, D)
            if shallow is not None:
                shallow = shallow.squeeze(0).float()  # (N_tokens, D)
                # May include CLS token — check
                if shallow.shape[0] == Hp * Wp + 1:
                    shallow = shallow[1:]  # remove CLS
                elif shallow.shape[0] != Hp * Wp:
                    # Best effort: take last Hp*Wp tokens
                    shallow = shallow[-Hp * Wp:]
                geo = shallow.T.reshape(-1, Hp, Wp)  # (D, Hp, Wp)
            else:
                geo = sem.clone()

        return {
            'geo': geo,
            'sem': sem,
            'summary': summary.squeeze(0).float(),
        }


def fit_pca(features_list, target_dim, desc=""):
    """Fit PCA on sampled features using SVD."""
    all_pixels = []
    sample_interval = max(1, len(features_list) // 100)
    for i in range(0, len(features_list), sample_interval):
        feat = features_list[i]  # (C, H, W)
        C, H, W = feat.shape
        pixels = feat.reshape(C, -1).T  # (HW, C)
        n_sample = min(500, pixels.shape[0])
        indices = torch.randperm(pixels.shape[0])[:n_sample]
        all_pixels.append(pixels[indices])

    all_pixels = torch.cat(all_pixels, dim=0).float()
    mean = all_pixels.mean(dim=0)
    centered = all_pixels - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:target_dim]

    var_explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  PCA {desc}: {all_pixels.shape[1]}d → {target_dim}d, "
          f"variance explained = {var_explained:.3f}")
    return components, mean


def apply_pca(feat, pca_matrix, pca_mean):
    """Apply PCA: (C, H, W) → (target_dim, H, W)"""
    C, H, W = feat.shape
    pixels = feat.reshape(C, -1).T.float()
    centered = pixels - pca_mean.to(pixels.device)
    transformed = centered @ pca_matrix.T.to(pixels.device)
    return transformed.T.reshape(-1, H, W)


def main():
    parser = argparse.ArgumentParser(description='Extract dual-scale RADIO features')
    parser.add_argument('--source_dir', required=True, help='Dataset directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--target_dim', type=int, default=64, help='PCA target dimension')
    parser.add_argument('--radio_repo', default='/root/RADIO', help='RADIO repo path')
    parser.add_argument('--shallow_block', type=int, default=10,
                        help='Transformer block index for shallow features')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find images
    img_patterns = ['seq*/*.png', 'images/*.png', 'images/*.jpg', '*.png', '*.jpg']
    image_paths = []
    for pat in img_patterns:
        found = sorted(glob.glob(os.path.join(args.source_dir, pat)))
        if found:
            image_paths = found
            break

    if not image_paths:
        print(f"ERROR: No images found in {args.source_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Initialize extractor
    extractor = DualScaleRADIOExtractor(
        radio_repo=args.radio_repo,
        shallow_block=args.shallow_block,
    )

    # Phase 1: Extract raw features
    print("\n[Phase 1] Extracting raw dual-scale features...")
    geo_features = []
    sem_features = []
    summaries = []

    for img_path in tqdm(image_paths, desc="Extracting"):
        from torchvision import transforms
        from PIL import Image

        img = Image.open(img_path).convert('RGB')
        tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W] in [0, 1]

        result = extractor.extract(tensor)
        geo_features.append(result['geo'].cpu())
        sem_features.append(result['sem'].cpu())
        summaries.append(result['summary'].cpu())

    # Phase 2: Fit PCA separately for geo and sem
    print("\n[Phase 2] Fitting PCA...")
    geo_pca, geo_mean = fit_pca(geo_features, args.target_dim, desc="geo (shallow)")
    sem_pca, sem_mean = fit_pca(sem_features, args.target_dim, desc="sem (deep)")

    # Phase 3: Apply PCA and save
    print("\n[Phase 3] Applying PCA and saving...")
    geo_dir = os.path.join(args.output_dir, 'fine_geo')
    sem_dir = os.path.join(args.output_dir, 'coarse_sem')
    pca_dir = os.path.join(args.output_dir, 'pca_params')
    summary_dir = os.path.join(args.output_dir, 'summary')
    for d in [geo_dir, sem_dir, pca_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)

    for i, img_path in enumerate(tqdm(image_paths, desc="Saving")):
        geo_pca_feat = apply_pca(geo_features[i], geo_pca, geo_mean)
        sem_pca_feat = apply_pca(sem_features[i], sem_pca, sem_mean)
        D_g, Hg, Wg = geo_pca_feat.shape
        D_s, Hs, Ws = sem_pca_feat.shape

        torch.save(geo_pca_feat.half(),
                   os.path.join(geo_dir, f'rgb_{i}_fine_geo_{D_g}x{Hg}x{Wg}.pt'))
        torch.save(sem_pca_feat.half(),
                   os.path.join(sem_dir, f'rgb_{i}_coarse_sem_{D_s}x{Hs}x{Ws}.pt'))
        torch.save(summaries[i].half(),
                   os.path.join(summary_dir, f'rgb_{i}_summary_2560.pt'))

    # Save PCA parameters
    torch.save({
        'components': geo_pca, 'mean': geo_mean,
        'source_dim': geo_features[0].shape[0], 'target_dim': args.target_dim,
    }, os.path.join(pca_dir, 'fine_geo_pca.pt'))

    torch.save({
        'components': sem_pca, 'mean': sem_mean,
        'source_dim': sem_features[0].shape[0], 'target_dim': args.target_dim,
    }, os.path.join(pca_dir, 'coarse_sem_pca.pt'))

    # Save stacked summary matrix
    summary_matrix = torch.stack(summaries, dim=0)
    torch.save(summary_matrix, os.path.join(args.output_dir, 'summary_matrix.pt'))

    print(f"\nDone! Saved {len(image_paths)} frames to {args.output_dir}")
    print(f"  fine_geo:   {D_g}d @ {Wg}x{Hg}")
    print(f"  coarse_sem: {D_s}d @ {Ws}x{Hs}")


if __name__ == '__main__':
    main()
