#!/usr/bin/env python3
"""
多尺度特征 PCA 降维模块

对 extract_multiscale_features.py 提取的原始高维特征进行 PCA 降维:
  - Fine   : 1408d → 128d
  - Mid    : 1280d → 64d
  - Coarse : 1280d → 32d

步骤:
  1. 从所有帧采样像素，拟合 PCA
  2. 将 PCA 变换后的特征保存到新目录
  3. 保存 PCA 模型供推理使用

用法:
    python scripts/compress_multiscale_pca.py \
        --input_dir output/features_multiscale/room_0 \
        --output_dir output/features_multiscale_compressed/room_0 \
        --fine_dim 128 --mid_dim 64 --coarse_dim 32 \
        --max_samples 200000
"""
import os
import sys
import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_samples(feat_dir: Path, pattern: str, max_samples: int,
                    desc: str = "采样") -> np.ndarray:
    """
    从特征目录中随机采样像素，用于 PCA 拟合。

    Args:
        feat_dir: 特征文件目录
        pattern:  glob 匹配模式
        max_samples: 最大采样像素数
        desc: tqdm 描述

    Returns:
        samples: [N, C] numpy array
    """
    feat_files = sorted(glob.glob(str(feat_dir / pattern)))
    if not feat_files:
        raise FileNotFoundError(f"在 {feat_dir} 中未找到匹配 '{pattern}' 的文件")

    print(f"  {desc}: 从 {len(feat_files)} 个文件中采样 (目标 {max_samples} 个像素)")

    # 先算每个文件平均该采多少
    samples_per_file = max(1, max_samples // len(feat_files))

    all_samples = []
    total = 0
    for fp in tqdm(feat_files, desc=f"  {desc}", leave=False):
        feat = torch.load(fp, map_location='cpu')  # [C, H, W]
        C, H, W = feat.shape
        num_pixels = H * W
        k = min(samples_per_file, num_pixels)
        # 随机选 k 个像素
        indices = torch.randperm(num_pixels)[:k]
        flat = feat.reshape(C, -1).T  # [H*W, C]
        sampled = flat[indices].numpy()  # [k, C]
        all_samples.append(sampled)
        total += k
        if total >= max_samples:
            break

    samples = np.concatenate(all_samples, axis=0)
    if len(samples) > max_samples:
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = samples[idx]

    print(f"    → 采样完毕: {samples.shape[0]} 个像素, 维度 {samples.shape[1]}")
    return samples


def fit_and_transform(feat_dir: Path, pattern: str, output_dir: Path,
                      target_dim: int, max_samples: int,
                      level_name: str) -> PCA:
    """
    对某层特征: 采样→拟合 PCA→变换所有帧→保存

    Returns:
        pca: 已拟合的 PCA 模型
    """
    print(f"\n{'='*60}")
    print(f"  [{level_name}]  原始维度 → {target_dim} 维")
    print(f"{'='*60}")

    # 1. 采样并拟合 PCA
    samples = collect_samples(feat_dir, pattern, max_samples, desc=f"{level_name} 采样")
    print(f"  拟合 PCA (n_components={target_dim})...")
    pca = PCA(n_components=target_dim, random_state=42)
    pca.fit(samples)
    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"    → 保留方差: {explained_var:.1f}%")

    # 2. 变换所有帧
    feat_files = sorted(glob.glob(str(feat_dir / pattern)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(feat_files, desc=f"  {level_name} 变换", leave=False):
        feat = torch.load(fp, map_location='cpu')  # [C, H, W]
        C, H, W = feat.shape
        flat = feat.reshape(C, -1).T.numpy()       # [H*W, C]
        transformed = pca.transform(flat)           # [H*W, target_dim]

        # L2 归一化
        norms = np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8
        transformed = transformed / norms

        result = torch.from_numpy(
            transformed.reshape(H, W, target_dim).transpose(2, 0, 1)
        ).float()  # [target_dim, H, W]

        # 重命名输出文件
        orig_name = Path(fp).name
        # e.g. rgb_0000_fine_1408x35x46.pt → rgb_0000_fine_128x35x46.pt
        new_name = orig_name
        for old_dim in [str(C)]:
            new_name = new_name.replace(f"{old_dim}x", f"{target_dim}x", 1)
        torch.save(result, output_dir / new_name)

    print(f"    → 保存 {len(feat_files)} 个文件到 {output_dir}")
    return pca


def main():
    parser = argparse.ArgumentParser(
        description='多尺度特征 PCA 降维'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='extract_multiscale_features.py 的输出目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='降维后特征的输出目录')
    parser.add_argument('--fine_dim', type=int, default=128,
                        help='Fine 层目标维度')
    parser.add_argument('--mid_dim', type=int, default=64,
                        help='Mid 层目标维度')
    parser.add_argument('--coarse_dim', type=int, default=32,
                        help='Coarse 层目标维度')
    parser.add_argument('--max_samples', type=int, default=200000,
                        help='PCA 拟合的最大采样像素数')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 验证输入目录
    for sub in ['fine', 'mid', 'coarse']:
        assert (input_dir / sub).exists(), f"输入目录不存在: {input_dir / sub}"

    # 降维各层
    pca_models = {}

    pca_models['fine'] = fit_and_transform(
        feat_dir=input_dir / 'fine',
        pattern='*_fine_*.pt',
        output_dir=output_dir / 'fine',
        target_dim=args.fine_dim,
        max_samples=args.max_samples,
        level_name='Fine (s3+DINO → 1408d)',
    )

    pca_models['mid'] = fit_and_transform(
        feat_dir=input_dir / 'mid',
        pattern='*_mid_*.pt',
        output_dir=output_dir / 'mid',
        target_dim=args.mid_dim,
        max_samples=args.max_samples,
        level_name='Mid (s4 → 1280d)',
    )

    pca_models['coarse'] = fit_and_transform(
        feat_dir=input_dir / 'coarse',
        pattern='*_coarse_*.pt',
        output_dir=output_dir / 'coarse',
        target_dim=args.coarse_dim,
        max_samples=args.max_samples,
        level_name='Coarse (s5 → 1280d)',
    )

    # 直接复制 CLS tokens (已经是 768d, 无需降维)
    cls_src = input_dir / 'cls'
    cls_dst = output_dir / 'cls'
    if cls_src.exists():
        cls_dst.mkdir(parents=True, exist_ok=True)
        cls_files = sorted(glob.glob(str(cls_src / '*.pt')))
        for fp in cls_files:
            import shutil
            shutil.copy2(fp, cls_dst / Path(fp).name)
        print(f"\n  CLS tokens: 复制 {len(cls_files)} 个文件到 {cls_dst}")

    # 保存 PCA 模型 (推理时需要)
    pca_save_path = output_dir / 'pca_models.pkl'
    with open(pca_save_path, 'wb') as f:
        pickle.dump({
            'fine': pca_models['fine'],
            'mid': pca_models['mid'],
            'coarse': pca_models['coarse'],
            'config': {
                'fine_dim': args.fine_dim,
                'mid_dim': args.mid_dim,
                'coarse_dim': args.coarse_dim,
            }
        }, f)
    print(f"\n  PCA 模型保存至: {pca_save_path}")

    # 最终汇总
    total_dim = args.fine_dim + args.mid_dim + args.coarse_dim
    print(f"\n{'='*60}")
    print(f"  ✓ 多尺度 PCA 降维完成!")
    print(f"    Fine   : 1408d → {args.fine_dim}d  (保留方差: {pca_models['fine'].explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"    Mid    : 1280d → {args.mid_dim}d  (保留方差: {pca_models['mid'].explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"    Coarse : 1280d → {args.coarse_dim}d  (保留方差: {pca_models['coarse'].explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"    CLS    : 768d  → 768d  (无需降维)")
    print(f"    3DGS 总嵌入维度: {total_dim}d  (< 原始单层 256d 或 768d)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
