#!/usr/bin/env python3
"""
中间量可视化脚本
================
生成 OldHospital 管线中各种中间量的可视化面板:
  1. RGB 原图
  2. 单目深度估计 (mono depth, viridis colormap)
  3. 语义分割 mask (5类: building/sky/vegetation/ground/other)
  4. masks.pkl 三通道叠加 (obj_mask / sky_mask / distort_mask)
  5. DA3 特征 PCA 可视化 (coarse / mid / fine)

用法:
    python scripts/visualize_intermediates.py [--num_samples 8] [--output_dir output/intermediate_vis]
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "OldHospital"
DA3_FEATURE_DIR = PROJECT_ROOT / "output" / "features_da3" / "OldHospital_indexed"
MASKS_PKL = DATASET_DIR / "masks.pkl"

# 语义类别颜色 (building=0, sky=1, vegetation=2, ground=3, other=4)
SEM_COLORS = np.array([
    [0.6, 0.6, 0.6],   # building: gray
    [0.3, 0.6, 1.0],   # sky: blue
    [0.2, 0.8, 0.2],   # vegetation: green
    [0.8, 0.7, 0.4],   # ground: brown/tan
    [1.0, 0.4, 0.4],   # other: red
])
SEM_LABELS = ['building', 'sky', 'vegetation', 'ground', 'other']


def find_images(image_dir):
    """按 seq 排序找到所有图像，返回 (global_idx, path) 列表。"""
    images = []
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "seq*")))
    global_idx = 0
    for seq_dir in seq_dirs:
        if not os.path.isdir(seq_dir):
            continue
        for fname in sorted(os.listdir(seq_dir)):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                images.append((global_idx, os.path.join(seq_dir, fname)))
                global_idx += 1
    return images


def pca_colorize(feat_map):
    """将 [D, H, W] 特征图 PCA 降到 3 通道 RGB [H, W, 3]，归一化到 [0,1]。"""
    D, H, W = feat_map.shape
    flat = feat_map.reshape(D, -1).T  # [HW, D]
    if isinstance(flat, torch.Tensor):
        flat = flat.cpu().float().numpy()

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat)  # [HW, 3]
    for c in range(3):
        lo, hi = np.percentile(rgb[:, c], [2, 98])
        if hi - lo < 1e-6:
            rgb[:, c] = 0.5
        else:
            rgb[:, c] = np.clip((rgb[:, c] - lo) / (hi - lo), 0, 1)
    return rgb.reshape(H, W, 3)


def get_frame_key(image_path):
    """从完整路径中提取 masks.pkl 的 key，如 'seq1/frame00001.png'。"""
    parts = Path(image_path).parts
    # 找到 seq* 部分
    for i, p in enumerate(parts):
        if p.startswith('seq'):
            return '/'.join(parts[i:])
    return None


def load_mono_depth(image_path):
    """加载对应的单目深度图。"""
    parts = Path(image_path).parts
    for i, p in enumerate(parts):
        if p.startswith('seq'):
            seq = p
            fname = parts[i + 1]
            break
    else:
        return None

    depth_path = DATASET_DIR / "mono_depth" / seq / fname.replace('.png', '.npy').replace('.jpg', '.npy')
    if depth_path.exists():
        return np.load(str(depth_path))
    return None


def load_semantic_mask(image_path):
    """加载对应的语义 mask。"""
    parts = Path(image_path).parts
    for i, p in enumerate(parts):
        if p.startswith('seq'):
            seq = p
            fname = parts[i + 1]
            break
    else:
        return None

    stem = fname.rsplit('.', 1)[0]
    sem_path = DATASET_DIR / "semantic_masks" / f"{seq}_{stem}_sem.pt"
    if sem_path.exists():
        return torch.load(str(sem_path), map_location='cpu').numpy()
    return None


def load_da3_features(global_idx):
    """加载指定 index 的 DA3 多尺度特征。"""
    feats = {}
    for scale in ['coarse', 'mid', 'fine']:
        scale_dir = DA3_FEATURE_DIR / scale
        # 文件名格式: rgb_{idx}_{scale}_{CxHxW}.pt
        matches = list(scale_dir.glob(f"rgb_{global_idx}_{scale}_*.pt"))
        if matches:
            feat = torch.load(str(matches[0]), map_location='cpu').float()
            feats[scale] = feat
    return feats


def visualize_frame(image_path, global_idx, masks_data, output_dir, frame_label):
    """为单帧生成完整的中间量可视化面板。"""
    # 1. RGB
    rgb = np.array(Image.open(image_path))

    # 2. Mono depth
    depth = load_mono_depth(image_path)

    # 3. Semantic mask
    sem = load_semantic_mask(image_path)

    # 4. masks.pkl
    frame_key = get_frame_key(image_path)
    mask_tuple = masks_data.get(frame_key) if masks_data and frame_key else None

    # 5. DA3 features
    da3_feats = load_da3_features(global_idx)

    # 计算面板数
    n_panels = 2  # RGB + depth always
    if sem is not None:
        n_panels += 1
    if mask_tuple is not None:
        n_panels += 1
    n_panels += len(da3_feats)  # coarse/mid/fine

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')

    panel_idx = 0

    # Panel 1: RGB
    axes[panel_idx].imshow(rgb)
    axes[panel_idx].set_title(f'RGB ({rgb.shape[1]}×{rgb.shape[0]})', fontsize=11)
    panel_idx += 1

    # Panel 2: Mono Depth
    if depth is not None:
        im = axes[panel_idx].imshow(depth, cmap='viridis')
        axes[panel_idx].set_title(f'Mono Depth ({depth.shape[1]}×{depth.shape[0]})\nrange [{depth.min():.3f}, {depth.max():.3f}]', fontsize=11)
        plt.colorbar(im, ax=axes[panel_idx], fraction=0.046, pad=0.04)
    else:
        axes[panel_idx].set_title('Mono Depth (N/A)', fontsize=11)
    panel_idx += 1

    # Panel 3: Semantic Mask
    if sem is not None:
        sem_rgb = SEM_COLORS[np.clip(sem, 0, 4)]
        axes[panel_idx].imshow(sem_rgb)
        classes_present = np.unique(sem)
        legend_text = ', '.join([f'{SEM_LABELS[c]}({c})' for c in classes_present if c < 5])
        axes[panel_idx].set_title(f'Semantic Mask ({sem.shape[1]}×{sem.shape[0]})\n{legend_text}', fontsize=10)
    else:
        axes[panel_idx].set_title('Semantic Mask (N/A)', fontsize=11)
    panel_idx += 1

    # Panel 4: Masks overlay (obj=R, sky=B, distort=G)
    if mask_tuple is not None:
        obj_mask, sky_mask, distort_mask = mask_tuple
        h, w = obj_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        if isinstance(obj_mask, torch.Tensor):
            obj_mask = obj_mask.numpy()
            sky_mask = sky_mask.numpy()
            distort_mask = distort_mask.numpy()
        overlay[..., 0] = obj_mask.astype(np.float32)      # R = obj
        overlay[..., 2] = sky_mask.astype(np.float32)       # B = sky
        overlay[..., 1] = distort_mask.astype(np.float32)   # G = distort
        axes[panel_idx].imshow(overlay)
        n_obj = obj_mask.sum()
        n_sky = sky_mask.sum()
        n_dist = distort_mask.sum()
        total = h * w
        axes[panel_idx].set_title(
            f'Masks ({w}×{h})\nobj(R)={n_obj/total:.1%}  sky(B)={n_sky/total:.1%}  distort(G)={n_dist/total:.1%}',
            fontsize=10
        )
    else:
        axes[panel_idx].set_title('Masks (N/A)', fontsize=11)
    panel_idx += 1

    # Panels 5-7: DA3 features PCA
    for scale in ['coarse', 'mid', 'fine']:
        if scale in da3_feats:
            feat = da3_feats[scale]
            pca_rgb = pca_colorize(feat)
            D, H, W = feat.shape
            axes[panel_idx].imshow(pca_rgb)
            axes[panel_idx].set_title(f'DA3 {scale} PCA\n({D}d @ {W}×{H})', fontsize=11)
        else:
            axes[panel_idx].set_title(f'DA3 {scale} (N/A)', fontsize=11)
        panel_idx += 1

    fig.suptitle(f'{frame_label}  (idx={global_idx})', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    safe_label = frame_label.replace('/', '_').replace('.png', '')
    out_path = os.path.join(output_dir, f'{safe_label}.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='中间量可视化')
    parser.add_argument('--num_samples', type=int, default=8, help='采样帧数')
    parser.add_argument('--output_dir', type=str, default='output/intermediate_vis')
    parser.add_argument('--specific_indices', type=int, nargs='+', default=None,
                        help='指定全局索引 (覆盖 num_samples)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 列出所有图像
    images = find_images(str(DATASET_DIR))
    print(f'Found {len(images)} images')

    # 2. 选择采样帧
    if args.specific_indices:
        sample_indices = args.specific_indices
    else:
        # 均匀采样 + 确保覆盖不同序列
        n = min(args.num_samples, len(images))
        step = len(images) // n
        sample_indices = [i * step for i in range(n)]

    # 3. 加载 masks.pkl (一次性)
    masks_data = None
    if MASKS_PKL.exists():
        print(f'Loading masks.pkl ({MASKS_PKL.stat().st_size / 1e9:.1f} GB)...')
        with open(str(MASKS_PKL), 'rb') as f:
            masks_data = pickle.load(f)
        print(f'  Loaded {len(masks_data)} mask entries')
    else:
        print('masks.pkl not found, skipping mask visualization')

    # 4. 逐帧生成可视化
    saved_paths = []
    for idx in sample_indices:
        if idx >= len(images):
            print(f'  [SKIP] idx={idx} out of range ({len(images)} images)')
            continue
        global_idx, image_path = images[idx]
        frame_key = get_frame_key(image_path)
        print(f'\nProcessing idx={global_idx}: {frame_key}')
        out = visualize_frame(
            image_path, global_idx, masks_data, args.output_dir, frame_key
        )
        if out:
            saved_paths.append(out)

    print(f'\n{"="*60}')
    print(f'Done! {len(saved_paths)} visualizations saved to {args.output_dir}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
