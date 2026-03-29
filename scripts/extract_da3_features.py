#!/usr/bin/env python3
"""
DA3 (Depth Anything 3) 多尺度特征提取脚本

从 RGB 图像通过 DA3-Base DPT decoder 提取多尺度特征:
  - fine:   64d @ target fine resolution (output_conv1, after DPT refinement)
  - mid:    128d → PCA to mid_dim @ mid resolution (refinenet4, fused deep features)
  - coarse: 128d → PCA to coarse_dim @ coarse resolution (l4_rn, deepest features)

DA3 架构: DINOv2 ViT-B/14 backbone + DualDPT decoder (features=128)
  - backbone layers [5,7,9,11] → project → resize → layer_rn (all 128d)
  - l4_rn: 128d @ ph/2 × pw/2 (deepest, most abstract)
  - refinenet4: fuse l4_rn with l3_rn → 128d @ ph × pw
  - output_conv1: 128d → 64d @ 2ph × 2pw (after refinenet1)

用法:
    CUDA_VISIBLE_DEVICES=4 python scripts/extract_da3_features.py \
        --image_dir dataset/OldHospital \
        --output_dir output/features_da3/OldHospital_indexed \
        --traj_source output/features_selected_pca/OldHospital_indexed/traj_w_c.txt \
        --fine_hw 69 121 --mid_hw 30 53 --coarse_hw 15 26 \
        --fine_dim 64 --mid_dim 64 --coarse_dim 32
"""

import os
import sys
import argparse
import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_images(image_dir: str) -> list:
    """Find all sequence images and return sorted (idx, path) pairs.
    Supports: seq*/frame*.png (Cambridge), Sequence_*/rgb/ (room)"""
    images = []
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        global_idx = 0
        for seq_dir in seq_dirs:
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    images.append((global_idx, os.path.join(seq_dir, fname)))
                    global_idx += 1
        if images:
            return images

    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "Sequence_*")))
    if seq_dirs:
        global_idx = 0
        for seq_dir in seq_dirs:
            rgb_dir = os.path.join(seq_dir, "rgb")
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    images.append((global_idx, os.path.join(rgb_dir, fname)))
                    global_idx += 1
        return images

    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            images.append((len(images), os.path.join(image_dir, fname)))
    return images


def load_da3_model(device):
    """Load DA3-Base model with pre-trained weights."""
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.model.da3 import DepthAnything3Net

    config_path = "/root/Depth-Anything-3/src/depth_anything_3/configs/da3-base.yaml"
    config = load_config(config_path)
    model = create_object(config)

    weights_path = os.path.expanduser("~/.cache/da3-base/model.safetensors")
    if not os.path.exists(weights_path):
        print("Downloading DA3-Base weights...")
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="depth-anything/DA3-BASE",
            filename="model.safetensors",
            cache_dir=os.path.expanduser("~/.cache/da3-base"),
            local_dir=os.path.expanduser("~/.cache/da3-base"),
        )

    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    # Strip 'model.' prefix if present
    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")

    model = model.to(device).eval()
    print(f"DA3-Base loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def extract_da3_multiscale(model, image_tensor, device):
    """
    Extract multi-scale features from DA3's DPT decoder.

    Args:
        model: DA3 DepthAnything3Net
        image_tensor: (1, 3, H, W) in [0, 1], H/W divisible by 14

    Returns:
        dict with keys 'fine', 'mid', 'coarse':
          - fine: (64, H_fine, W_fine) from output_conv1
          - mid: (128, H_mid, W_mid) from refinenet4
          - coarse: (128, H_coarse, W_coarse) from l4_rn
    """
    B, _, H, W = image_tensor.shape
    patch_size = 14
    ph, pw = H // patch_size, W // patch_size

    with torch.no_grad():
        # 1) Run backbone: input (B, S=1, 3, H, W)
        x = image_tensor.unsqueeze(1)  # (B, 1, 3, H, W)
        feats, _ = model.backbone(x, cam_token=None, export_feat_layers=[])
        # feats: list of 4 tuples, each [0] is (B, S, N, C)
        # With cat_token=True: N=ph*pw (no CLS position), C=1536

        # 2) Process through DPT head manually to extract intermediates
        head = model.head  # DualDPT
        patch_start_idx = 0  # cat_token mode: no CLS token in sequence

        BS = B  # B*S=1
        C = feats[0][0].shape[-1]  # 1536

        resized_feats = []
        for stage_idx in range(4):
            feat = feats[stage_idx][0]  # (B, S, N, C)
            feat = feat.reshape(BS, -1, C)  # (B*S, N, C)
            x_feat = feat[:, patch_start_idx:]
            x_feat = head.norm(x_feat)
            x_feat = x_feat.permute(0, 2, 1).reshape(BS, C, ph, pw)
            x_feat = head.projects[stage_idx](x_feat)
            if head.pos_embed:
                x_feat = head._add_pos_embed(x_feat, W, H)
            x_feat = head.resize_layers[stage_idx](x_feat)
            resized_feats.append(x_feat)

        # 3) Run main fusion chain
        l1, l2, l3, l4 = resized_feats
        scratch = head.scratch

        l1_rn = scratch.layer1_rn(l1)  # 128d @ 4*ph x 4*pw
        l2_rn = scratch.layer2_rn(l2)  # 128d @ 2*ph x 2*pw
        l3_rn = scratch.layer3_rn(l3)  # 128d @ ph x pw
        l4_rn = scratch.layer4_rn(l4)  # 128d @ ~ph/2 x pw/2

        # Main fusion
        out4 = scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])   # 128d @ ph x pw
        out3 = scratch.refinenet3(out4, l3_rn, size=l2_rn.shape[2:])
        out2 = scratch.refinenet2(out3, l2_rn, size=l1_rn.shape[2:])
        out1 = scratch.refinenet1(out2, l1_rn)  # 128d @ 4*ph x 4*pw
        fine_feat = scratch.output_conv1(out1)   # 64d @ 4*ph x 4*pw

    return {
        'fine': fine_feat.squeeze(0),     # (64, 4*ph, 4*pw)
        'mid': out4.squeeze(0),           # (128, ph, pw)
        'coarse': l4_rn.squeeze(0),       # (128, ~ph/2, ~pw/2)
    }


def fit_pca(features_list, target_dim, desc=""):
    """Fit PCA on sampled features.
    
    Args:
        features_list: list of (C, H, W) tensors
        target_dim: target PCA dimension
    
    Returns:
        pca_matrix: (target_dim, C), pca_mean: (C,)
    """
    # Sample pixels for PCA fitting
    all_pixels = []
    sample_interval = max(1, len(features_list) // 100)  # ~100 frames
    for i in range(0, len(features_list), sample_interval):
        feat = features_list[i]  # (C, H, W)
        C, H, W = feat.shape
        pixels = feat.reshape(C, -1).T  # (H*W, C)
        # Subsample pixels
        n_sample = min(500, pixels.shape[0])
        indices = torch.randperm(pixels.shape[0])[:n_sample]
        all_pixels.append(pixels[indices])

    all_pixels = torch.cat(all_pixels, dim=0).float()  # (N, C)
    print(f"  PCA {desc}: {all_pixels.shape[0]} samples, {all_pixels.shape[1]}d → {target_dim}d")

    # Compute PCA (no mean subtraction — keeps features comparable)
    mean = all_pixels.mean(dim=0)
    centered = all_pixels - mean
    # SVD on covariance
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:target_dim]  # (target_dim, C)

    # Check explained variance
    var_explained = (S[:target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"  PCA {desc}: explained variance = {var_explained:.3f}")

    return components, mean


def apply_pca(feat, pca_matrix, pca_mean):
    """Apply PCA transform.
    
    Args:
        feat: (C, H, W)
        pca_matrix: (target_dim, C)
        pca_mean: (C,)
    
    Returns:
        (target_dim, H, W)
    """
    C, H, W = feat.shape
    pixels = feat.reshape(C, -1).T.float()  # (H*W, C)
    centered = pixels - pca_mean.to(pixels.device)
    transformed = centered @ pca_matrix.T.to(pixels.device)  # (H*W, target_dim)
    return transformed.T.reshape(-1, H, W)  # (target_dim, H, W)


def main():
    parser = argparse.ArgumentParser(description='Extract DA3 multi-scale features')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Dataset image directory (e.g., dataset/OldHospital)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output feature directory')
    parser.add_argument('--traj_source', type=str, default=None,
                        help='Copy trajectory file from this path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)

    # Target resolutions (match existing pipeline)
    parser.add_argument('--fine_hw', type=int, nargs=2, default=[69, 121])
    parser.add_argument('--mid_hw', type=int, nargs=2, default=[30, 53])
    parser.add_argument('--coarse_hw', type=int, nargs=2, default=[15, 26])

    # Target PCA dimensions
    parser.add_argument('--fine_dim', type=int, default=64,
                        help='Fine feature dim (no PCA needed, already 64d)')
    parser.add_argument('--mid_dim', type=int, default=64,
                        help='Mid feature dim after PCA (from 128d)')
    parser.add_argument('--coarse_dim', type=int, default=32,
                        help='Coarse feature dim after PCA (from 128d)')

    parser.add_argument('--no_pca', action='store_true',
                        help='Skip PCA, save raw 128d features')

    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # Create output directories
    for sub in ['fine', 'mid', 'coarse', 'pca_params']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(args.image_dir)
    print(f"Found {len(images)} images in {args.image_dir}")

    # If trajectory file has fewer entries, limit images to match
    if args.traj_source and os.path.exists(args.traj_source):
        n_traj = sum(1 for _ in open(args.traj_source))
        if n_traj < len(images):
            print(f"Limiting to {n_traj} images (matching trajectory)")
            images = images[:n_traj]

    # Load DA3 model
    model = load_da3_model(device)

    # ====== Pass 1: Extract raw features at target resolutions ======
    print("\n=== Pass 1: Extracting DA3 features ===")
    raw_fine = []
    raw_mid = []
    raw_coarse = []

    fine_H, fine_W = args.fine_hw
    mid_H, mid_W = args.mid_hw
    coarse_H, coarse_W = args.coarse_hw

    for idx, img_path in tqdm(images, desc="Extracting"):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Resize to nearest multiple of 14
        _, _, H_orig, W_orig = img_tensor.shape
        H_target = round(H_orig / 14) * 14
        W_target = round(W_orig / 14) * 14
        if H_target != H_orig or W_target != W_orig:
            img_tensor = F.interpolate(img_tensor, size=(H_target, W_target),
                                       mode='bilinear', align_corners=False)

        img_tensor = img_tensor.to(device)

        # Extract features
        with torch.no_grad():
            feats = extract_da3_multiscale(model, img_tensor, device)

        # Resize to target resolutions
        fine_feat = F.interpolate(
            feats['fine'].unsqueeze(0), size=(fine_H, fine_W),
            mode='bilinear', align_corners=False
        ).squeeze(0).cpu()  # (64, fine_H, fine_W)

        mid_feat = F.interpolate(
            feats['mid'].unsqueeze(0), size=(mid_H, mid_W),
            mode='bilinear', align_corners=False
        ).squeeze(0).cpu()  # (128, mid_H, mid_W)

        coarse_feat = F.interpolate(
            feats['coarse'].unsqueeze(0), size=(coarse_H, coarse_W),
            mode='bilinear', align_corners=False
        ).squeeze(0).cpu()  # (128, coarse_H, coarse_W)

        raw_fine.append(fine_feat)
        raw_mid.append(mid_feat)
        raw_coarse.append(coarse_feat)

    # ====== Pass 2: PCA compression ======
    if args.no_pca:
        print("\n=== Skipping PCA (--no_pca) ===")
        final_fine = raw_fine
        final_mid = raw_mid
        final_coarse = raw_coarse
        actual_fine_dim = raw_fine[0].shape[0]
        actual_mid_dim = raw_mid[0].shape[0]
        actual_coarse_dim = raw_coarse[0].shape[0]
    else:
        print("\n=== Pass 2: PCA compression ===")

        # Fine: already 64d, no PCA needed if target_dim == 64
        actual_fine_dim = args.fine_dim
        if raw_fine[0].shape[0] == args.fine_dim:
            print(f"  Fine: already {args.fine_dim}d, no PCA needed")
            final_fine = raw_fine
        else:
            pca_fine, mean_fine = fit_pca(raw_fine, args.fine_dim, "fine")
            torch.save({'components': pca_fine, 'mean': mean_fine},
                       output_dir / 'pca_params' / 'fine_pca.pt')
            final_fine = [apply_pca(f, pca_fine, mean_fine) for f in tqdm(raw_fine, desc="PCA fine")]

        # Mid: 128d → mid_dim
        actual_mid_dim = args.mid_dim
        if raw_mid[0].shape[0] == args.mid_dim:
            final_mid = raw_mid
        else:
            pca_mid, mean_mid = fit_pca(raw_mid, args.mid_dim, "mid")
            torch.save({'components': pca_mid, 'mean': mean_mid},
                       output_dir / 'pca_params' / 'mid_pca.pt')
            final_mid = [apply_pca(f, pca_mid, mean_mid) for f in tqdm(raw_mid, desc="PCA mid")]

        # Coarse: 128d → coarse_dim
        actual_coarse_dim = args.coarse_dim
        if raw_coarse[0].shape[0] == args.coarse_dim:
            final_coarse = raw_coarse
        else:
            pca_coarse, mean_coarse = fit_pca(raw_coarse, args.coarse_dim, "coarse")
            torch.save({'components': pca_coarse, 'mean': mean_coarse},
                       output_dir / 'pca_params' / 'coarse_pca.pt')
            final_coarse = [apply_pca(f, pca_coarse, mean_coarse)
                           for f in tqdm(raw_coarse, desc="PCA coarse")]

    # ====== Pass 3: Save features ======
    print("\n=== Saving features ===")
    for i, (idx, img_path) in enumerate(tqdm(images, desc="Saving")):
        # Fine
        f = final_fine[i].half()  # Save as fp16 to reduce disk space
        fname = f"rgb_{idx}_fine_{actual_fine_dim}x{fine_H}x{fine_W}.pt"
        torch.save(f, output_dir / 'fine' / fname)

        # Mid
        f = final_mid[i].half()
        fname = f"rgb_{idx}_mid_{actual_mid_dim}x{mid_H}x{mid_W}.pt"
        torch.save(f, output_dir / 'mid' / fname)

        # Coarse
        f = final_coarse[i].half()
        fname = f"rgb_{idx}_coarse_{actual_coarse_dim}x{coarse_H}x{coarse_W}.pt"
        torch.save(f, output_dir / 'coarse' / fname)

    # Copy trajectory file
    if args.traj_source and os.path.exists(args.traj_source):
        import shutil
        shutil.copy2(args.traj_source, output_dir / 'traj_w_c.txt')
        print(f"Copied trajectory from {args.traj_source}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Frames: {len(images)}")
    print(f"  Fine:   {actual_fine_dim}d @ {fine_H}×{fine_W}")
    print(f"  Mid:    {actual_mid_dim}d @ {mid_H}×{mid_W}")
    print(f"  Coarse: {actual_coarse_dim}d @ {coarse_H}×{coarse_W}")
    print(f"  Output: {output_dir}")

    # Estimate disk usage
    total_bytes = 0
    for sub in ['fine', 'mid', 'coarse']:
        sub_dir = output_dir / sub
        sub_bytes = sum(f.stat().st_size for f in sub_dir.glob('*.pt'))
        total_bytes += sub_bytes
        print(f"  {sub}: {sub_bytes / 1e6:.1f} MB")
    print(f"  Total: {total_bytes / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
