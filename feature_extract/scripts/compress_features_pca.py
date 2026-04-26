#!/usr/bin/env python3
"""
Replace AE compression with PCA for OldHospital features.

The autoencoder compression destroys spatial distinctiveness:
  - Original DINO 768d: 65.8% gap → AE 64d: 6.8% gap (10× worse)
  - PCA 64d: 70.8% gap (actually BETTER than original!)

This script:
1. Loads all original uncompressed features for each scale
2. Fits PCA to target dimension
3. Saves PCA-compressed features in the same format as AE-compressed
4. Saves PCA parameters (mean, components) for use in 3DGS embedding

Usage:
    python scripts/compress_features_pca.py \
        --input_dir output/features_multiscale_stride7/OldHospital \
        --output_dir output/features_multiscale_pca/OldHospital_indexed \
        --traj_path output/features_multiscale_stride7_compressed/OldHospital_indexed/traj_w_c.txt
"""
import argparse, os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm


TARGET_DIMS = {
    'coarse': 32,
    'mid': 64,
    'fine_sd': 64,
    'fine_dino': 64,
}

# Target resolutions (must match compressed feature files — NOT the pose net's internal resolution)
# The pose network resizes internally; feature files stay at these resolutions.
TARGET_RES = {
    'coarse': (15, 26),
    'mid': (30, 53),
    'fine_sd': (69, 121),
    'fine_dino': (69, 121),
}


def load_all_features(feat_dir, scale):
    """Load all feature files for a scale, return dict {frame_stem: tensor}."""
    scale_dir = os.path.join(feat_dir, scale)
    files = sorted(os.listdir(scale_dir))
    features = {}
    for fn in tqdm(files, desc=f"Loading {scale}"):
        if not fn.endswith('.pt'):
            continue
        feat = torch.load(os.path.join(scale_dir, fn), map_location='cpu')
        if isinstance(feat, dict):
            feat = list(feat.values())[0]
        # Extract frame stem from filename
        # e.g., seq1_frame00001_coarse_512x15x26.pt → seq1_frame00001
        parts = fn.rsplit(f'_{scale}_', 1)
        stem = parts[0] if len(parts) > 1 else fn.replace('.pt', '')
        features[stem] = feat
    return features


def fit_pca(features_dict, target_dim, max_samples=50000):
    """Fit PCA on all features.
    
    Returns: mean (C,), components (target_dim, C)
    """
    # Collect all pixel features
    all_pixels = []
    for feat in features_dict.values():
        C, H, W = feat.shape
        all_pixels.append(feat.reshape(C, -1).T)  # (HW, C)
    
    X = torch.cat(all_pixels, dim=0).numpy()  # (N_total, C)
    print(f"  Total pixels for PCA: {X.shape[0]}")
    
    # Subsample for efficiency
    rng = np.random.RandomState(42)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    # Center
    mean = X_sub.mean(axis=0)
    X_centered = X_sub - mean
    
    # SVD
    print(f"  Computing SVD on {X_centered.shape}...")
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    components = Vt[:target_dim]  # (target_dim, C)
    explained = (S[:target_dim]**2).sum() / (S**2).sum()
    print(f"  Explained variance: {explained*100:.1f}%")
    print(f"  Component shape: {components.shape}")
    
    return mean, components, S


def project_features(feat, mean, components, target_res):
    """Project a single feature tensor through PCA.
    
    feat: (C_in, H_in, W_in) original features
    mean: (C_in,) PCA mean
    components: (C_out, C_in) PCA components
    target_res: (H_out, W_out) target spatial resolution
    
    Returns: (C_out, H_out, W_out) PCA-compressed features
    """
    import torch.nn.functional as F
    
    C_in, H_in, W_in = feat.shape
    C_out = components.shape[0]
    H_out, W_out = target_res
    
    # Resize spatial dimensions if needed
    if (H_in, W_in) != (H_out, W_out):
        feat = F.interpolate(
            feat.unsqueeze(0), size=(H_out, W_out), mode='bilinear', align_corners=False
        ).squeeze(0)
    
    # Flatten: (C_in, H*W) → (H*W, C_in)
    pixels = feat.reshape(C_out if C_in == C_out else feat.shape[0], -1).T  # Actually preserve C_in
    pixels = feat.reshape(feat.shape[0], -1).T  # (H*W, C_in)
    
    # Center and project
    mean_t = torch.from_numpy(mean).float()
    comp_t = torch.from_numpy(components).float()
    
    centered = pixels - mean_t
    projected = centered @ comp_t.T  # (H*W, C_out)
    
    # Reshape back to (C_out, H, W)
    result = projected.T.reshape(C_out, H_out, W_out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Uncompressed features dir')
    parser.add_argument('--output_dir', required=True, help='Output PCA-compressed dir')
    parser.add_argument('--traj_path', required=True, help='Trajectory file to copy')
    parser.add_argument('--max_pca_samples', type=int, default=50000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy trajectory file
    traj_out = os.path.join(args.output_dir, 'traj_w_c.txt')
    if not os.path.exists(traj_out):
        shutil.copy2(args.traj_path, traj_out)
        print(f"Copied trajectory to {traj_out}")

    # Build frame index mapping: seq1_frame00001 → 0, seq1_frame00002 → 1, ...
    # We need to match the indexing used in the compressed features
    # The indexed format uses rgb_N where N is a sequential index
    
    # First, collect all unique stems across scales
    all_stems = set()
    for scale in TARGET_DIMS:
        scale_dir = os.path.join(args.input_dir, scale)
        for fn in os.listdir(scale_dir):
            if fn.endswith('.pt'):
                parts = fn.rsplit(f'_{scale}_', 1)
                stem = parts[0] if len(parts) > 1 else fn.replace('.pt', '')
                all_stems.add(stem)
    
    stems_sorted = sorted(all_stems)
    stem_to_idx = {s: i for i, s in enumerate(stems_sorted)}
    print(f"Total frames: {len(stems_sorted)}")
    
    # Save PCA params for later use
    pca_params_dir = os.path.join(args.output_dir, 'pca_params')
    os.makedirs(pca_params_dir, exist_ok=True)
    
    for scale, target_dim in TARGET_DIMS.items():
        print(f"\n{'='*60}")
        print(f"Processing {scale} (target: {target_dim}d)")
        print(f"{'='*60}")
        
        # Load all features
        features = load_all_features(args.input_dir, scale)
        print(f"  Loaded {len(features)} feature files")
        
        # Fit PCA
        mean, components, S = fit_pca(features, target_dim, args.max_pca_samples)
        
        # Save PCA params
        np.savez(
            os.path.join(pca_params_dir, f'{scale}_pca.npz'),
            mean=mean,
            components=components,
            singular_values=S[:target_dim],
        )
        
        # Project and save all features
        out_scale_dir = os.path.join(args.output_dir, scale)
        os.makedirs(out_scale_dir, exist_ok=True)
        
        target_res = TARGET_RES[scale]
        
        for stem, feat in tqdm(features.items(), desc=f"Projecting {scale}"):
            projected = project_features(feat, mean, components, target_res)
            
            # Save with indexed naming: rgb_N_{scale}_{dim}x{H}x{W}.pt
            idx = stem_to_idx[stem]
            out_name = f'rgb_{idx}_{scale}_{target_dim}x{target_res[0]}x{target_res[1]}.pt'
            torch.save(projected, os.path.join(out_scale_dir, out_name))
        
        # Verify distinctiveness
        sample_stem = list(features.keys())[0]
        proj_sample = project_features(features[sample_stem], mean, components, target_res)
        import torch.nn.functional as F
        fn = F.normalize(proj_sample.unsqueeze(0), dim=1)
        C, H, W = proj_sample.shape
        q = fn.reshape(1, C, -1).permute(0, 2, 1)
        r = fn.reshape(1, C, -1)
        corr = torch.bmm(q, r).squeeze(0)
        off_diag = (corr.sum() - torch.diagonal(corr).sum()) / (H*W*(H*W-1))
        print(f"  Verification: off_diag={off_diag:.4f}, gap={1.0-off_diag.item():.4f}")
    
    print(f"\n✓ PCA compression complete! Output: {args.output_dir}")
    print(f"  PCA params saved to: {pca_params_dir}")


if __name__ == '__main__':
    main()
