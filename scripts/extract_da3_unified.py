#!/usr/bin/env python3
"""
统一 DA3 特征+深度+语义分割 提取脚本

从单个 DA3 模型的一次前向传播中同时提取:
  1. Fine 特征: 64d @ target resolution (用于 3DGS 嵌入 + 定位网络)
  2. 度量深度: metric depth via exp activation (替代 DPT-Large)
  3. 语义分割: 5-class mask (利用 DA3 backbone 的 DINOv2 特征 + ADE20K head)

注意: DA3 backbone 被深度任务微调后，特征与标准 DINOv2 差异很大，
      ADE20K 线性分割头无法直接使用。因此仍需加载轻量 DINOv2 backbone (85M) 来生成语义 mask。
      但总共只需 DA3 + DINOv2 两个模型（替代了原来的 DA3 + DPT-Large + DINOv2 三个模型）。
      深度完全由 DA3 提供（替代 DPT-Large），特征也由 DA3 提供。

用法:
    CUDA_VISIBLE_DEVICES=4 python scripts/extract_da3_unified.py \
        --image_dir dataset/OldHospital \
        --output_dir output/features_da3_unified/OldHospital \
        --traj_source output/features_selected_pca/OldHospital_indexed/traj_w_c.txt \
        --fine_hw 69 121
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════════════════════════════════════════════════════════════════════
# Image discovery (same as extract_da3_features.py)
# ════════════════════════════════════════════════════════════════════════════

def find_images(image_dir: str) -> list:
    """Find all sequence images and return sorted (idx, path) pairs."""
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


# ════════════════════════════════════════════════════════════════════════════
# ADE20K 6-class mapping (extended from generate_semantic_masks.py)
# Class 0: other, 1: building, 2: vegetation, 3: sky, 4: ground, 5: dynamic
# ════════════════════════════════════════════════════════════════════════════

ADE20K_VEGETATION = {4, 9, 17, 29, 55, 60, 66}
ADE20K_BUILDING = {1, 25}
ADE20K_SKY = {2}
ADE20K_GROUND = {6, 11, 13}
# Dynamic objects: person, car, bus, truck, bicycle, motorcycle, van, boat, minibike, animal
ADE20K_DYNAMIC = {12, 20, 76, 80, 83, 102, 116, 126, 127, 130}


def build_ade20k_to_5class():
    mapping = torch.zeros(150, dtype=torch.long)
    for idx in ADE20K_BUILDING:
        mapping[idx] = 1
    for idx in ADE20K_VEGETATION:
        mapping[idx] = 2
    for idx in ADE20K_SKY:
        mapping[idx] = 3
    for idx in ADE20K_GROUND:
        mapping[idx] = 4
    for idx in ADE20K_DYNAMIC:
        mapping[idx] = 5
    return mapping


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_da3_model(device):
    """Load DA3-Base model with pre-trained weights."""
    sys.path.insert(0, '/root/Depth-Anything-3/src')
    from depth_anything_3.cfg import create_object, load_config

    config_path = "/root/Depth-Anything-3/src/depth_anything_3/configs/da3-base.yaml"
    config = load_config(config_path)
    model = create_object(config)

    weights_path = os.path.expanduser("~/.cache/da3-base/model.safetensors")
    if not os.path.exists(weights_path):
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="depth-anything/DA3-BASE",
            filename="model.safetensors",
            cache_dir=os.path.expanduser("~/.cache/da3-base"),
            local_dir=os.path.expanduser("~/.cache/da3-base"),
        )

    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"DA3-Base loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def load_dinov2_backbone(device):
    """Load standard DINOv2 ViT-B/14 for segmentation."""
    local_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
    if os.path.isdir(local_dir):
        model = torch.hub.load(local_dir, 'dinov2_vitb14', source='local')
    else:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval().to(device)
    print(f"DINOv2 ViT-B/14 loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def load_ade20k_head(device='cuda'):
    """Load ADE20K linear segmentation head: BN(768) → Conv2d(768, 150, 1)."""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/dinov2_heads')
    cache_path = os.path.join(cache_dir, 'dinov2_vitb14_ade20k_linear_head.pth')

    if not os.path.exists(cache_path):
        url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_head.pth"
        os.makedirs(cache_dir, exist_ok=True)
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', model_dir=cache_dir)
    else:
        checkpoint = torch.load(cache_path, map_location='cpu', weights_only=False)

    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    head = nn.Sequential(
        nn.BatchNorm2d(768),
        nn.Conv2d(768, 150, 1),
    )
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('decode_head.bn.'):
            new_sd['0.' + k.split('decode_head.bn.')[1]] = v
        elif k.startswith('decode_head.conv_seg.'):
            new_sd['1.' + k.split('decode_head.conv_seg.')[1]] = v
    head.load_state_dict(new_sd, strict=True)
    head.eval().to(device)
    print("ADE20K seg head loaded: BN(768) → Conv2d(768, 150)")
    return head


# ════════════════════════════════════════════════════════════════════════════
# Unified extraction
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_all(model, dinov2, seg_head, ade2merged, image_tensor, device, fine_hw):
    """
    Unified extraction: fine features + metric depth + semantic mask.

    Uses DA3 for features and depth, standard DINOv2 for segmentation.

    Args:
        model: DA3 model
        dinov2: standard DINOv2 ViT-B/14 backbone
        seg_head: ADE20K linear head (BN+Conv2d)
        ade2merged: (150,) ADE20K→5class mapping (CPU)
        image_tensor: (1, 3, H, W) in [0, 1], H/W divisible by 14
        device: torch device
        fine_hw: (H_fine, W_fine) target resolution for features

    Returns:
        fine_feat: (64, H_fine, W_fine) float16
        depth: (H_fine, W_fine) float32 metric depth
        mask: (H_p, W_p) uint8 5-class semantic label
    """
    B, _, H, W = image_tensor.shape
    patch_size = 14
    ph, pw = H // patch_size, W // patch_size

    # ── 1) DA3 backbone forward ──
    x = image_tensor.unsqueeze(1)  # (B, S=1, 3, H, W)
    feats, _ = model.backbone(x, cam_token=None, export_feat_layers=[])

    # ── 2) Semantic mask via standard DINOv2 ──
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_norm = normalize(image_tensor.squeeze(0)).unsqueeze(0)  # (1, 3, H, W)
    dino_out = dinov2.forward_features(img_norm)
    patch_tokens = dino_out['x_norm_patchtokens']  # (1, N, 768)
    tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(1, 768, ph, pw)
    seg_logits = seg_head(tokens_spatial)  # (1, 150, ph, pw)
    ade_labels = seg_logits.argmax(dim=1).squeeze(0)  # (ph, pw)
    mask = ade2merged[ade_labels.cpu()]  # (ph, pw) 5-class

    # ── 3) DPT head: features + depth ──
    head = model.head
    C = feats[0][0].shape[-1]  # 1536
    BS = B

    resized_feats = []
    for stage_idx in range(4):
        feat = feats[stage_idx][0].reshape(BS, -1, C)  # (B, N, 1536)
        x_feat = feat  # all patch tokens (no CLS in cat_token mode)
        x_feat = head.norm(x_feat)
        x_feat = x_feat.permute(0, 2, 1).reshape(BS, C, ph, pw)
        x_feat = head.projects[stage_idx](x_feat)
        if head.pos_embed:
            x_feat = head._add_pos_embed(x_feat, W, H)
        x_feat = head.resize_layers[stage_idx](x_feat)
        resized_feats.append(x_feat)

    # Run fusion chain → 64d fine features + depth
    fused_main, _ = head._fuse(resized_feats)
    # fused_main: (B, 64, 4*ph, 4*pw) after output_conv1

    # Resize fine features to target resolution
    fine_H, fine_W = fine_hw
    fine_feat = F.interpolate(fused_main, size=(fine_H, fine_W),
                              mode='bilinear', align_corners=False)
    fine_feat = fine_feat.squeeze(0).half()  # (64, fine_H, fine_W)

    # Depth: output_conv2 → 2ch → exp activation
    # Upsample fused_main to depth resolution (same as fine features)
    fused_for_depth = F.interpolate(fused_main, size=(fine_H, fine_W),
                                    mode='bilinear', align_corners=True)
    if head.pos_embed:
        fused_for_depth = head._add_pos_embed(fused_for_depth, W, H)
    depth_logits = head.scratch.output_conv2(fused_for_depth)  # (B, 2, H, W)
    depth_map = depth_logits[:, 0:1, :, :]  # first channel = depth logit
    depth_map = torch.exp(depth_map)  # metric depth (exp activation)
    depth_map = depth_map.squeeze(0).squeeze(0).float()  # (fine_H, fine_W)

    return fine_feat, depth_map, mask.to(torch.uint8)


def main():
    parser = argparse.ArgumentParser(description='Unified DA3 extraction: features + depth + masks')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--traj_source', type=str, default=None,
                        help='Copy trajectory file from this path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fine_hw', type=int, nargs=2, default=[69, 121],
                        help='Target resolution for fine features and depth')

    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    fine_H, fine_W = args.fine_hw
    fine_dim = 64  # DA3 output_conv1 always produces 64d

    # Create output directories
    for sub in ['fine', 'depth', 'masks']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(args.image_dir)
    print(f"Found {len(images)} images in {args.image_dir}")
    if not images:
        print("ERROR: No images found!")
        return

    # Limit to trajectory count
    if args.traj_source and os.path.exists(args.traj_source):
        n_traj = sum(1 for _ in open(args.traj_source))
        if n_traj < len(images):
            print(f"Limiting to {n_traj} images (matching trajectory)")
            images = images[:n_traj]

    # Load models
    print("Loading DA3 model...")
    model = load_da3_model(device)
    print("Loading DINOv2 backbone for segmentation...")
    dinov2 = load_dinov2_backbone(device)
    print("Loading ADE20K segmentation head...")
    seg_head = load_ade20k_head(device)
    ade2merged = build_ade20k_to_5class()  # CPU

    # Copy trajectory file
    if args.traj_source and os.path.exists(args.traj_source):
        import shutil
        dest = output_dir / 'traj_w_c.txt'
        shutil.copy2(args.traj_source, str(dest))
        print(f"Copied trajectory to {dest}")

    # Extract all
    print(f"\n=== Extracting: fine={fine_dim}d@{fine_H}×{fine_W}, depth, masks ===")

    for idx, img_path in tqdm(images, desc="Extracting"):
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

        fine_feat, depth, mask = extract_all(
            model, dinov2, seg_head, ade2merged, img_tensor, device, args.fine_hw
        )

        # Save fine features: rgb_{idx}_fine_{dim}x{H}x{W}.pt
        feat_name = f"rgb_{idx}_fine_{fine_dim}x{fine_H}x{fine_W}.pt"
        torch.save(fine_feat.cpu(), str(output_dir / 'fine' / feat_name))

        # Save depth: rgb_{idx}_depth.pt
        depth_name = f"rgb_{idx}_depth.pt"
        torch.save(depth.cpu(), str(output_dir / 'depth' / depth_name))

        # Save mask: rgb_{idx}_mask.pt
        mask_name = f"rgb_{idx}_mask.pt"
        torch.save(mask.cpu(), str(output_dir / 'masks' / mask_name))

    # Save metadata
    meta = {
        'n_images': len(images),
        'fine_dim': fine_dim,
        'fine_hw': [fine_H, fine_W],
        'depth_hw': [fine_H, fine_W],
        'mask_type': '6class_ade20k',
        'depth_type': 'metric_exp',
    }
    import json
    with open(str(output_dir / 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Done! Saved {len(images)} frames to {output_dir}/")
    print(f"  fine/   : {fine_dim}d @ {fine_H}×{fine_W} (float16)")
    print(f"  depth/  : metric depth @ {fine_H}×{fine_W} (float32)")
    print(f"  masks/  : 6-class @ patch resolution (uint8)")
    print(f"           0=other 1=building 2=vegetation 3=sky 4=ground 5=dynamic")


if __name__ == '__main__':
    main()
