"""Extract 128d PCA dual-RADIO features and produce stacked train/test tensors.

This script:
1. Extracts raw RADIO features (geo from block 10, sem from final layer)
2. Fits PCA to target_dim (128) using training images
3. Produces stacked tensors: fine_geo_train.pt, fine_geo_test.pt, coarse_sem_train.pt, coarse_sem_test.pt
4. Also produces summary_matrix.pt (indexed by all images)

Usage:
    python -m feature_extract.extract_and_stack_128d \
        --source_dir dataset/OldHospital \
        --output_dir output/feature_extract/features_radio_dual_128/OldHospital_pilot \
        --target_dim 128 \
        --device cuda:5
"""

import os
import sys
import argparse
import glob as glob_module
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from feature_extract.utils.radio_loader import load_radio_model


class DualScaleRADIOExtractor:
    """Extract both shallow (geometric) and deep (semantic) RADIO features."""

    def __init__(self, version='c-radio_v4-h', device='cuda',
                 radio_repo='feature_extract/checkpoints/RADIO',
                 shallow_block=10):
        self.device = torch.device(device)
        self.shallow_block = shallow_block
        self.intermediate_aggregation = 'dense'
        self.intermediate_norm_alpha_scheme = 'post-alpha'

        print(f"Loading RADIO {version}...")
        self.model = load_radio_model(version=version, radio_repo=radio_repo)
        self.model = self.model.to(self.device).eval()
        self.patch_size = self.model.patch_size

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  RADIO loaded: {n_params:.0f}M params, patch_size={self.patch_size}")

        self._shallow_features = None
        self._use_forward_intermediates = hasattr(self.model, 'forward_intermediates')
        if self._use_forward_intermediates:
            print(
                f"  Using RADIO forward_intermediates() for shallow block {self.shallow_block} "
                f"(aggregation={self.intermediate_aggregation})"
            )
        else:
            self._register_hooks()

    def _register_hooks(self):
        blocks = None
        model_inner = self.model
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
            print("  WARNING: Could not find transformer blocks.")
            self._use_single_layer = True
            return

        self._use_single_layer = False
        n_blocks = len(blocks)
        shallow_idx = min(self.shallow_block, n_blocks - 1)
        print(f"  Hooking shallow features at block {shallow_idx}/{n_blocks-1}")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._shallow_features = output[0].detach()
            else:
                self._shallow_features = output.detach()

        blocks[shallow_idx].register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract(self, image_tensor):
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

        autocast_context = (
            torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else nullcontext()
        )
        with autocast_context:
            if self._use_forward_intermediates:
                final, intermediates = self.model.forward_intermediates(
                    image_tensor,
                    indices=[self.shallow_block],
                    return_prefix_tokens=False,
                    norm=False,
                    stop_early=False,
                    output_fmt='NCHW',
                    intermediates_only=False,
                    aggregation=self.intermediate_aggregation,
                    norm_alpha_scheme=self.intermediate_norm_alpha_scheme,
                )
                final_features = final.features if hasattr(final, 'features') else final[1]
                final_summary = final.summary if hasattr(final, 'summary') else final[0]
                sem = final_features.squeeze(0).float()
                summary = final_summary.squeeze(0).float()
                if intermediates:
                    geo = intermediates[0].squeeze(0).float()
                else:
                    geo = sem.clone()
            else:
                summary, features = self.model(image_tensor, feature_fmt='NCHW')

                sem = features.squeeze(0).float()  # (D, Hp, Wp)

                if hasattr(self, '_use_single_layer') and self._use_single_layer:
                    geo = sem.clone()
                else:
                    shallow = self._shallow_features
                    if shallow is not None:
                        shallow = shallow.squeeze(0).float()
                        if shallow.shape[0] == Hp * Wp + 1:
                            shallow = shallow[1:]
                        elif shallow.shape[0] != Hp * Wp:
                            shallow = shallow[-Hp * Wp:]
                        geo = shallow.T.reshape(-1, Hp, Wp)
                    else:
                        geo = sem.clone()
                summary = summary.squeeze(0).float()

        return {
            'geo': geo,
            'sem': sem,
            'summary': summary,
        }


def fit_pca(features_list, target_dim, desc=""):
    """Fit PCA on sampled features using SVD."""
    all_pixels = []
    sample_interval = max(1, len(features_list) // 100)
    for i in range(0, len(features_list), sample_interval):
        feat = features_list[i]
        C, H, W = feat.shape
        pixels = feat.reshape(C, -1).T
        n_sample = min(500, pixels.shape[0])
        indices = torch.randperm(pixels.shape[0])[:n_sample]
        all_pixels.append(pixels[indices])

    all_pixels = torch.cat(all_pixels, dim=0).float()
    mean = all_pixels.mean(dim=0)
    centered = all_pixels - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:target_dim]

    var_explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  PCA {desc}: {all_pixels.shape[1]}d -> {target_dim}d, "
          f"variance explained = {var_explained:.3f}")
    return components, mean


def apply_pca(feat, pca_matrix, pca_mean):
    """Apply PCA: (C, H, W) -> (target_dim, H, W)"""
    C, H, W = feat.shape
    pixels = feat.reshape(C, -1).T.float()
    centered = pixels - pca_mean
    transformed = centered @ pca_matrix.T
    return transformed.T.reshape(-1, H, W)


def parse_dataset_file(dataset_file):
    """Parse Cambridge Landmarks dataset file, return list of image names."""
    names = []
    with open(dataset_file) as f:
        lines = f.readlines()
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) >= 8:
            names.append(parts[0])
    return names


def main():
    parser = argparse.ArgumentParser(description='Extract 128d dual-RADIO features (stacked)')
    parser.add_argument('--source_dir', required=True, help='Dataset directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--target_dim', type=int, default=128, help='PCA target dimension')
    parser.add_argument('--radio_repo', default='feature_extract/checkpoints/RADIO')
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--shallow_block', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse train/test splits
    train_names = parse_dataset_file(os.path.join(args.source_dir, 'dataset_train.txt'))
    test_names = parse_dataset_file(os.path.join(args.source_dir, 'dataset_test.txt'))
    print(f"Train: {len(train_names)} images, Test: {len(test_names)} images")

    # Build ordered list of all unique images
    all_names = train_names + test_names
    # Build name -> list index
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Initialize extractor
    extractor = DualScaleRADIOExtractor(
        device=args.device,
        radio_repo=args.radio_repo,
        shallow_block=args.shallow_block,
    )

    # Phase 1: Extract raw features for all images
    print(f"\n[Phase 1] Extracting raw features for {len(all_names)} images...")
    geo_features = []
    sem_features = []
    summaries = []

    transform = transforms.ToTensor()

    for img_name in tqdm(all_names, desc="Extracting"):
        img_path = os.path.join(args.source_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)

        result = extractor.extract(tensor)
        geo_features.append(result['geo'].cpu())
        sem_features.append(result['sem'].cpu())
        summaries.append(result['summary'].cpu())

    # Phase 2: Fit PCA using ALL images (train+test, same as original)
    print(f"\n[Phase 2] Fitting PCA to {args.target_dim}d...")
    geo_pca, geo_mean = fit_pca(geo_features, args.target_dim, desc="geo (shallow)")
    sem_pca, sem_mean = fit_pca(sem_features, args.target_dim, desc="sem (deep)")

    # Phase 3: Apply PCA and stack into train/test tensors
    print("\n[Phase 3] Applying PCA and stacking...")

    # Apply PCA to all
    geo_pca_all = []
    sem_pca_all = []
    for i in tqdm(range(len(all_names)), desc="PCA transform"):
        geo_pca_all.append(apply_pca(geo_features[i], geo_pca, geo_mean).half())
        sem_pca_all.append(apply_pca(sem_features[i], sem_pca, sem_mean).half())

    # Free raw features
    del geo_features, sem_features

    # Split into train/test
    n_train = len(train_names)
    n_test = len(test_names)

    geo_train = torch.stack(geo_pca_all[:n_train])  # (895, 128, 68, 120)
    geo_test = torch.stack(geo_pca_all[n_train:])   # (182, 128, 68, 120)
    sem_train = torch.stack(sem_pca_all[:n_train])
    sem_test = torch.stack(sem_pca_all[n_train:])

    summary_all = torch.stack(summaries)  # (1077, 2560)

    # Save
    print(f"\n[Phase 4] Saving to {args.output_dir}...")
    torch.save(geo_train, os.path.join(args.output_dir, 'fine_geo_train.pt'))
    torch.save(geo_test, os.path.join(args.output_dir, 'fine_geo_test.pt'))
    torch.save(sem_train, os.path.join(args.output_dir, 'coarse_sem_train.pt'))
    torch.save(sem_test, os.path.join(args.output_dir, 'coarse_sem_test.pt'))
    torch.save(summary_all, os.path.join(args.output_dir, 'summary_matrix.pt'))

    # Save PCA parameters
    pca_dir = os.path.join(args.output_dir, 'pca_params')
    os.makedirs(pca_dir, exist_ok=True)
    torch.save({
        'components': geo_pca, 'mean': geo_mean,
        'source_dim': geo_pca_all[0].shape[0] if len(geo_pca_all) > 0 else args.target_dim,
        'target_dim': args.target_dim,
    }, os.path.join(pca_dir, 'fine_geo_pca.pt'))
    torch.save({
        'components': sem_pca, 'mean': sem_mean,
        'source_dim': sem_pca_all[0].shape[0] if len(sem_pca_all) > 0 else args.target_dim,
        'target_dim': args.target_dim,
    }, os.path.join(pca_dir, 'coarse_sem_pca.pt'))

    print(f"\nDone!")
    print(f"  fine_geo_train: {geo_train.shape}")
    print(f"  fine_geo_test:  {geo_test.shape}")
    print(f"  coarse_sem_train: {sem_train.shape}")
    print(f"  coarse_sem_test:  {sem_test.shape}")
    print(f"  summary_matrix: {summary_all.shape}")
    print(f"  Geo PCA variance explained logged above")
    print(f"  Sem PCA variance explained logged above")


if __name__ == '__main__':
    main()
