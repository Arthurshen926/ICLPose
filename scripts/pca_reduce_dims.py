#!/usr/bin/env python3
"""
Reduce PCA feature dimensions by slicing top-k components.

Since PCA components are ordered by explained variance, we can reduce
64d PCA features to 16d by simply taking the first 16 channels.
This makes 3DGS reconstruction much easier (64d total → fewer dims per Gaussian).

Usage:
    python scripts/pca_reduce_dims.py \
        --input_dir output/features_multiscale_pca/OldHospital_indexed \
        --output_dir output/features_multiscale_pca16/OldHospital_indexed \
        --target_dim 16
"""
import argparse, os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm


SCALES = ['coarse', 'mid', 'fine_sd', 'fine_dino']
ORIGINAL_DIMS = {'coarse': 32, 'mid': 64, 'fine_sd': 64, 'fine_dino': 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_dim', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy trajectory
    traj_src = os.path.join(args.input_dir, 'traj_w_c.txt')
    traj_dst = os.path.join(args.output_dir, 'traj_w_c.txt')
    if os.path.exists(traj_src) and not os.path.exists(traj_dst):
        shutil.copy2(traj_src, traj_dst)

    td = args.target_dim

    # Update PCA params
    pca_in = os.path.join(args.input_dir, 'pca_params')
    pca_out = os.path.join(args.output_dir, 'pca_params')
    os.makedirs(pca_out, exist_ok=True)

    for scale in SCALES:
        orig_dim = ORIGINAL_DIMS[scale]
        actual_dim = min(td, orig_dim)
        print(f"\n=== {scale}: {orig_dim}d → {actual_dim}d ===")

        # Slice PCA params
        params = np.load(os.path.join(pca_in, f'{scale}_pca.npz'))
        np.savez(
            os.path.join(pca_out, f'{scale}_pca.npz'),
            mean=params['mean'],
            components=params['components'][:actual_dim],
            singular_values=params['singular_values'][:actual_dim],
        )

        # Slice features
        in_dir = os.path.join(args.input_dir, scale)
        out_dir = os.path.join(args.output_dir, scale)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(in_dir) if f.endswith('.pt'))
        for fn in tqdm(files, desc=f"Slicing {scale}"):
            feat = torch.load(os.path.join(in_dir, fn), map_location='cpu')
            C, H, W = feat.shape
            sliced = feat[:actual_dim]

            # Rename file: rgb_0_coarse_32x15x26.pt → rgb_0_coarse_16x15x26.pt
            new_fn = fn.replace(f'{orig_dim}x', f'{actual_dim}x')
            torch.save(sliced, os.path.join(out_dir, new_fn))

        # Quick distinctiveness check
        sample = torch.load(os.path.join(out_dir, sorted(os.listdir(out_dir))[0]), map_location='cpu')
        import torch.nn.functional as F
        fn_norm = F.normalize(sample.unsqueeze(0), dim=1)
        C, H, W = sample.shape
        q = fn_norm.reshape(1, C, -1).permute(0, 2, 1)
        r = fn_norm.reshape(1, C, -1)
        corr = torch.bmm(q, r).squeeze(0)
        off_diag = (corr.sum() - torch.diagonal(corr).sum()) / (H*W*(H*W-1))
        gap = 1.0 - off_diag.item()
        print(f"  Distinctiveness gap: {gap:.1%} ({actual_dim}d)")

    # Report total per-Gaussian dim
    total = sum(min(td, ORIGINAL_DIMS[s]) for s in SCALES)
    print(f"\n✓ Done! Total per-Gaussian: {total}d (was {sum(ORIGINAL_DIMS.values())}d)")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
