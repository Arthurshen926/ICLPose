import torch
import numpy as np
import os
import glob
from pathlib import Path

BASE = "/root/ICLPose-loc/output/feature_extract/features_radio_dual/OldHospital_pilot"
FINE_DIR = f"{BASE}/fine_geo"
COARSE_DIR = f"{BASE}/coarse_sem"
PCA_DIR = f"{BASE}/pca_params"
DATASET = "/root/ICLPose-loc/dataset/OldHospital"

# ── Reconstruct global image ordering (same as extraction script) ──
all_images = sorted(glob.glob(os.path.join(DATASET, 'seq*/*.png')))
# Build map: relative path -> index
name_to_idx = {}
for i, p in enumerate(all_images):
    rel = os.path.relpath(p, DATASET)  # e.g. seq1/frame00001.png
    name_to_idx[rel] = i

print(f"Total images in dataset: {len(all_images)}")
print(f"Total feature files: {len(os.listdir(FINE_DIR))}")

# ── Parse train/test splits ──
def parse_split(path):
    names = []
    with open(path) as f:
        for line in f.readlines()[3:]:
            parts = line.strip().split()
            if parts:
                names.append(parts[0])
    return names

train_imgs = parse_split(f"{DATASET}/dataset_train.txt")
test_imgs = parse_split(f"{DATASET}/dataset_test.txt")
print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

# Map to indices
train_indices = [name_to_idx[n] for n in train_imgs]
test_indices = [name_to_idx[n] for n in test_imgs]
print(f"Train idx range: {min(train_indices)}-{max(train_indices)}")
print(f"Test idx range: {min(test_indices)}-{max(test_indices)}")

# Verify all exist
max_idx = max(max(train_indices), max(test_indices))
print(f"Max index needed: {max_idx}, files available: {len(os.listdir(FINE_DIR))}")

# ── Load sample patch tokens for statistics ──
def load_pt(directory, idx, scale):
    fname = f"rgb_{idx}_{scale}_64x68x120.pt"
    return torch.load(os.path.join(directory, fname), map_location='cpu')

n_sample = 10
sample_indices = np.linspace(0, len(train_indices)-1, n_sample, dtype=int)
sample_global = [train_indices[i] for i in sample_indices]

print("\n" + "="*60)
print("PATCH TOKEN STATISTICS")
print("="*60)

for label, directory, scale in [("fine_geo", FINE_DIR, "fine_geo"), ("coarse_sem", COARSE_DIR, "coarse_sem")]:
    samples = [load_pt(directory, idx, scale).float() for idx in sample_global]
    stack = torch.stack(samples)
    print(f"\n--- {label} ---")
    print(f"  Shape per image: {samples[0].shape}, dtype: torch.float16 (loaded as float32)")
    print(f"  Mean:  {stack.mean().item():.6f}")
    print(f"  Std:   {stack.std().item():.6f}")
    print(f"  Min:   {stack.min().item():.6f}")
    print(f"  Max:   {stack.max().item():.6f}")
    
    ch_mean = stack.mean(dim=(0,2,3))
    ch_std = stack.std(dim=(0,2,3))
    print(f"  Per-channel mean range: [{ch_mean.min():.4f}, {ch_mean.max():.4f}]")
    print(f"  Per-channel std range:  [{ch_std.min():.4f}, {ch_std.max():.4f}]")
    
    spatial_stds = [s.std(dim=(1,2)).mean().item() for s in samples]
    cross_img_std = stack.mean(dim=(2,3)).std(dim=0).mean().item()
    print(f"  Avg within-image spatial std: {np.mean(spatial_stds):.6f}")
    print(f"  Cross-image std (of spatial means): {cross_img_std:.6f}")
    
    flat = stack.view(n_sample, 64, -1)
    means = flat.mean(dim=2)
    means_norm = means / means.norm(dim=1, keepdim=True)
    cos_sim = means_norm @ means_norm.T
    triu = torch.triu_indices(n_sample, n_sample, offset=1)
    print(f"  Avg pairwise cosine sim (mean repr): {cos_sim[triu[0], triu[1]].mean().item():.4f}")

# ── PCA variance analysis ──
print("\n" + "="*60)
print("PCA PARAMS ANALYSIS")
print("="*60)

for label in ["fine_geo", "coarse_sem"]:
    pca_data = torch.load(f"{PCA_DIR}/{label}_pca.pt", map_location='cpu')
    print(f"\n--- {label}_pca.pt ---")
    for k, v in pca_data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    
    # The PCA was fit via SVD. We have components (64, D_orig) and mean (D_orig,)
    # To estimate variance explained, we need the singular values which aren't saved.
    # Instead, let's compute empirically from the projected data.
    # The projected features should have unit-like variance if PCA is good.
    comps = pca_data['components'].float()
    mean = pca_data['mean'].float()
    print(f"  Components norm (per dim): min={comps.norm(dim=1).min():.4f}, max={comps.norm(dim=1).max():.4f}")

# ── Empirical PCA variance from projected features ──
print("\n" + "="*60)
print("EMPIRICAL VARIANCE ANALYSIS (from projected features)")
print("="*60)

for label, directory, scale in [("fine_geo", FINE_DIR, "fine_geo"), ("coarse_sem", COARSE_DIR, "coarse_sem")]:
    samples = torch.stack([load_pt(directory, idx, scale).float() for idx in sample_global])
    flat = samples.reshape(-1, 64).T  # (64, N_pixels)
    flat_centered = flat - flat.mean(dim=1, keepdim=True)
    cov = (flat_centered @ flat_centered.T) / flat.shape[1]
    eigvals = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    eigvals_norm = eigvals / eigvals.sum()
    cum_var = eigvals_norm.cumsum(0)
    entropy = -(eigvals_norm * (eigvals_norm + 1e-10).log()).sum().item()
    eff_rank = np.exp(entropy)
    print(f"\n--- {label} ---")
    print(f"  Effective rank: {eff_rank:.1f} / 64")
    for d in [1, 5, 10, 20, 32, 48, 64]:
        print(f"    Top {d:3d} dims: {cum_var[d-1].item()*100:.2f}%")

# ── Spatial structure variation ──
print("\n" + "="*60)
print("SPATIAL STRUCTURE VARIATION")
print("="*60)

for label, directory, scale in [("fine_geo", FINE_DIR, "fine_geo"), ("coarse_sem", COARSE_DIR, "coarse_sem")]:
    samples = torch.stack([load_pt(directory, idx, scale).float() for idx in sample_global])
    spatial_maps = samples.mean(dim=1).reshape(n_sample, -1)
    spatial_norm = spatial_maps / spatial_maps.norm(dim=1, keepdim=True)
    sim = spatial_norm @ spatial_norm.T
    triu = torch.triu_indices(n_sample, n_sample, offset=1)
    avg_sim = sim[triu[0], triu[1]].mean().item()
    std_sim = sim[triu[0], triu[1]].std().item()
    print(f"  {label} spatial map cosine sim: {avg_sim:.4f} +/- {std_sim:.4f}")

# ── summary_matrix ──
print("\n" + "="*60)
print("SUMMARY MATRIX")
print("="*60)
summary = torch.load(f"{BASE}/summary_matrix.pt", map_location='cpu')
print(f"  Shape: {summary.shape}, dtype: {summary.dtype}")

# ══════════════════════════════════════════════════════════════
# BUILD TRAIN/TEST MATRICES
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("BUILDING TRAIN/TEST PATCH TOKEN MATRICES")
print("="*60)

for label, directory, scale in [("fine_geo", FINE_DIR, "fine_geo"), ("coarse_sem", COARSE_DIR, "coarse_sem")]:
    for split_name, indices in [("train", train_indices), ("test", test_indices)]:
        N = len(indices)
        print(f"\nBuilding {label}_{split_name}: {N} x 64 x 68 x 120 ...")
        
        # Pre-allocate as float16 to save memory
        matrix = torch.zeros(N, 64, 68, 120, dtype=torch.float16)
        for j, idx in enumerate(indices):
            matrix[j] = load_pt(directory, idx, scale)
            if (j+1) % 200 == 0:
                print(f"  Loaded {j+1}/{N}")
        
        out_path = f"{BASE}/{label}_{split_name}.pt"
        torch.save(matrix, out_path)
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {matrix.shape}, dtype: {matrix.dtype}")
        print(f"  File size: {size_mb:.1f} MB")

print("\nDone!")
