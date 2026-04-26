#!/usr/bin/env python3
"""
DINOv2 + ADE20K Linear Head → Semantic Masks
=============================================
使用预训练 DINOv2 ViT-B/14 + ADE20K 线性分割头，将每帧图像分割为 5 类:
  0: other
  1: building
  2: vegetation
  3: sky
  4: ground

输出: dataset/{scene}/semantic_masks/{image_name}_sem.pt  (uint8 [H_p, W_p])
      dataset/{scene}/semantic_masks_viz/               (可视化 PNG)

用法:
  python scripts/generate_semantic_masks.py \
      --scene_dir dataset/OldHospital \
      --output_dir dataset/OldHospital/semantic_masks \
      [--device cuda:0] [--batch_size 8] [--viz]
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
# ADE20K class → 5-class mapping (0-indexed ADE20K labels)
# ════════════════════════════════════════════════════════════════════════════

ADE20K_VEGETATION = {4, 9, 17, 29, 55, 60, 66}  # tree, grass, plant, field, flower, palm, bush
ADE20K_BUILDING = {1, 25}                         # building, house
ADE20K_SKY = {2}                                   # sky
ADE20K_GROUND = {6, 11, 13}                        # road, sidewalk, earth/ground

LABEL_NAMES = ['other', 'building', 'vegetation', 'sky', 'ground']
LABEL_COLORS = np.array([
    [128, 128, 128],  # other: gray
    [180, 120, 60],   # building: brown
    [0, 180, 0],      # vegetation: green
    [135, 206, 235],  # sky: light blue
    [160, 100, 40],   # ground: dark brown
], dtype=np.uint8)


def build_ade20k_to_5class():
    """Build mapping from 150 ADE20K classes → 5 merged classes."""
    mapping = torch.zeros(150, dtype=torch.long)  # default: 0 = other
    for idx in ADE20K_BUILDING:
        mapping[idx] = 1
    for idx in ADE20K_VEGETATION:
        mapping[idx] = 2
    for idx in ADE20K_SKY:
        mapping[idx] = 3
    for idx in ADE20K_GROUND:
        mapping[idx] = 4
    return mapping


# ════════════════════════════════════════════════════════════════════════════
# DINOv2 + ADE20K Linear Head
# ════════════════════════════════════════════════════════════════════════════

def load_dinov2_backbone(device='cuda'):
    """Load DINOv2 ViT-B/14 backbone from local cache or hub."""
    local_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
    if os.path.isdir(local_dir):
        print(f"  Loading DINOv2 from local cache: {local_dir}")
        model = torch.hub.load(local_dir, 'dinov2_vitb14', source='local')
    else:
        print("  Downloading DINOv2 from hub...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval().to(device)
    return model


def load_ade20k_head(device='cuda'):
    """Download and build ADE20K linear segmentation head.

    Architecture: BN(768) → Linear(768, 150)
    Weights from: dinov2_vitb14_ade20k_linear_head.pth
    """
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_head.pth"
    cache_dir = os.path.expanduser('~/.cache/torch/hub/dinov2_heads')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'dinov2_vitb14_ade20k_linear_head.pth')

    if not os.path.exists(cache_path):
        print(f"  Downloading ADE20K linear head to {cache_path}...")
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu',
                                                         model_dir=cache_dir)
    else:
        print(f"  Loading ADE20K head from cache: {cache_path}")
        checkpoint = torch.load(cache_path, map_location='cpu', weights_only=False)

    # Extract state_dict from checkpoint wrapper
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # Keys: decode_head.bn.{weight,bias,running_mean,running_var,num_batches_tracked}
    #        decode_head.conv_seg.{weight,bias}

    # Build head: BN2d → Conv2d(1×1)  (works on spatial [B, 768, H, W] input)
    # decode_head.bn is SyncBN2d, conv_seg is Conv2d(768, 150, kernel_size=1)
    head = nn.Sequential(
        nn.BatchNorm2d(768),         # decode_head.bn (SyncBN → BN for single GPU)
        nn.Conv2d(768, 150, 1),      # decode_head.conv_seg (1×1 conv)
    )

    # Remap keys: decode_head.bn.* → 0.*, decode_head.conv_seg.* → 1.*
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('decode_head.bn.'):
            new_sd['0.' + k.split('decode_head.bn.')[1]] = v
        elif k.startswith('decode_head.conv_seg.'):
            new_sd['1.' + k.split('decode_head.conv_seg.')[1]] = v
    head.load_state_dict(new_sd, strict=True)
    head.eval().to(device)
    print(f"  ADE20K head loaded: BN(768) → Linear(768, 150)")
    return head


# ════════════════════════════════════════════════════════════════════════════
# Image discovery
# ════════════════════════════════════════════════════════════════════════════

def find_images(scene_dir):
    """Find all images in scene directory. Returns list of (rel_name, abs_path)."""
    results = []

    # Cambridge-style: seq*/frame*.png
    seq_dirs = sorted(glob.glob(os.path.join(scene_dir, "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        for seq_dir in seq_dirs:
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.join(os.path.basename(seq_dir), fname)
                    results.append((rel, os.path.join(seq_dir, fname)))
        if results:
            return results

    # Replica-style: Sequence_*/rgb/
    seq_dirs = sorted(glob.glob(os.path.join(scene_dir, "Sequence_*")))
    if seq_dirs:
        for seq_dir in seq_dirs:
            rgb_dir = os.path.join(seq_dir, "rgb")
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.relpath(os.path.join(rgb_dir, fname), scene_dir)
                    results.append((rel, os.path.join(rgb_dir, fname)))
        return results

    # Flat: images/ or root
    img_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(img_dir):
        img_dir = scene_dir
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            results.append((fname, os.path.join(img_dir, fname)))

    return results


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_masks(args):
    device = args.device 
    patch_size = 14  # DINOv2 ViT-B/14

    # Load models
    print("Loading DINOv2 backbone...")
    backbone = load_dinov2_backbone(device)
    print("Loading ADE20K linear head...")
    head = load_ade20k_head(device)

    # 5-class mapping
    ade2merged = build_ade20k_to_5class().to(device)

    # ImageNet normalization (DINOv2 standard)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Find images
    images = find_images(args.scene_dir)
    print(f"Found {len(images)} images in {args.scene_dir}")
    if not images:
        print("ERROR: No images found!")
        return

    # Output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.viz:
        viz_dir = output_dir.parent / 'semantic_masks_viz'
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    stats = {i: 0 for i in range(5)}

    for rel_name, abs_path in tqdm(images, desc="Generating semantic masks"):
        # Load and preprocess image
        img = Image.open(abs_path).convert('RGB')
        W_orig, H_orig = img.size

        # Resize to be divisible by patch_size (14)
        H_new = (H_orig // patch_size) * patch_size
        W_new = (W_orig // patch_size) * patch_size
        img_resized = img.resize((W_new, H_new), Image.BILINEAR)

        # To tensor + normalize
        img_tensor = transforms.ToTensor()(img_resized)  # [3, H_new, W_new]
        img_tensor = normalize(img_tensor).unsqueeze(0).to(device)  # [1, 3, H, W]

        # DINOv2 forward → patch tokens
        # output.shape = [1, N_patches, 768] where N_patches = (H/14) * (W/14)
        features = backbone.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']  # [1, N, 768]

        H_p = H_new // patch_size
        W_p = W_new // patch_size

        # Reshape to spatial: [1, N, 768] → [1, 768, H_p, W_p] for BN2d + Conv2d
        tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(1, 768, H_p, W_p)  # [1, 768, H_p, W_p]
        logits = head(tokens_spatial)  # [1, 150, H_p, W_p]

        # ADE20K 150-class → 5-class
        ade_labels = logits.argmax(dim=1).squeeze(0)  # [H_p, W_p], values in [0, 149]
        merged_labels = ade2merged[ade_labels]          # [H_p, W_p], values in [0, 4]

        # Save as uint8 tensor
        save_name = rel_name.replace('/', '_').replace('\\', '_')
        save_name = os.path.splitext(save_name)[0] + '_sem.pt'
        torch.save(merged_labels.cpu().to(torch.uint8), str(output_dir / save_name))

        # Stats
        for c in range(5):
            stats[c] += (merged_labels == c).sum().item()

        # Visualization
        if args.viz:
            labels_np = merged_labels.cpu().numpy()
            vis = LABEL_COLORS[labels_np]  # [H_p, W_p, 3]
            # Upscale to original resolution for overlay
            vis_img = Image.fromarray(vis).resize((W_orig, H_orig), Image.NEAREST)

            # Blend with original
            orig_arr = np.array(img)
            vis_arr = np.array(vis_img)
            blended = (0.5 * orig_arr + 0.5 * vis_arr).astype(np.uint8)

            viz_name = os.path.splitext(save_name)[0] + '.png'
            Image.fromarray(blended).save(str(viz_dir / viz_name))

    # Print statistics
    total_px = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"Semantic mask generation complete!")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {len(images)}")
    print(f"\n  Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        pct = 100 * stats[i] / max(total_px, 1)
        print(f"    {name:12s}: {pct:5.1f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate semantic masks using DINOv2 + ADE20K')
    parser.add_argument('--scene_dir', required=True, help='Scene directory (e.g., dataset/OldHospital)')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: scene_dir/semantic_masks)')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--viz', action='store_true', help='Generate visualization PNGs')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir, 'semantic_masks')

    generate_masks(args)
