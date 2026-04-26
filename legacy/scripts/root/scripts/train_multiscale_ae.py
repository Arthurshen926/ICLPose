#!/usr/bin/env python3
"""
多尺度特征 AutoEncoder 训练与压缩

对 extract_multiscale_features.py 提取的原始高维特征进行 AutoEncoder 降维:
  - fine_sd   (SD s3)  : 512d  → 64d   (v2_fine_sd)
  - fine_dino (DINO)   : 768d  → 64d   (v2_fine_dino)
  - mid       (SD s4)  : 512d  → 64d   (v2_mid)
  - coarse    (SD s5)  : 512d  → 32d   (v2_coarse)

嵌入 3DGS 时: fine = fine_sd(64d) + fine_dino(64d) = 128d, mid = 64d, coarse = 32d
总计 224d, 比原始单层 256d 还小

用法:
    # 训练所有 4 个 AutoEncoder
    python scripts/train_multiscale_ae.py \
        --input_dir output/features_multiscale/room_0 \
        --output_dir output/features_multiscale_compressed/room_0 \
        --ae_save_dir output/ae_models/room_0 \
        --epochs 50 --batch_size 4096 --lr 1e-3

    # 仅压缩 (使用已训练好的权重)
    python scripts/train_multiscale_ae.py \
        --input_dir output/features_multiscale/room_0 \
        --output_dir output/features_multiscale_compressed/room_0 \
        --ae_save_dir output/ae_models/room_0 \
        --compress_only
"""
import os
import sys
import argparse
import glob
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_compression.autoencoder import AutoencoderFlexible, FEATURE_CONFIGS


# ═══════════════════════════════════════════════════════════
#  v2 多尺度压缩的 4 个任务定义
# ═══════════════════════════════════════════════════════════
COMPRESS_TASKS = [
    {
        'name': 'fine_sd',
        'config_key': 'v2_fine_sd',
        'subdir': 'fine_sd',
        'pattern': '*_fine_sd_*.pt',
        'desc': 'Fine SD (s3): 512d → 64d',
    },
    {
        'name': 'fine_dino',
        'config_key': 'v2_fine_dino',
        'subdir': 'fine_dino',
        'pattern': '*_fine_dino_*.pt',
        'desc': 'Fine DINO: 768d → 64d',
    },
    {
        'name': 'mid',
        'config_key': 'v2_mid',
        'subdir': 'mid',
        'pattern': '*_mid_*.pt',
        'desc': 'Mid (s4): 512d → 64d',
    },
    {
        'name': 'coarse',
        'config_key': 'v2_coarse',
        'subdir': 'coarse',
        'pattern': '*_coarse_*.pt',
        'desc': 'Coarse (s5): 512d → 32d',
    },
]


def collect_pixel_samples(feat_dir: Path, pattern: str,
                          max_samples: int = 500000) -> torch.Tensor:
    """
    从特征目录中收集像素样本用于 AE 训练

    Returns:
        samples: [N, C] float32 tensor
    """
    feat_files = sorted(glob.glob(str(feat_dir / pattern)))
    if not feat_files:
        raise FileNotFoundError(f"在 {feat_dir} 中未找到匹配 '{pattern}' 的文件")

    samples_per_file = max(1, max_samples // len(feat_files))
    all_samples = []
    total = 0

    for fp in feat_files:
        feat = torch.load(fp, map_location='cpu')  # [C, H, W]
        C, H, W = feat.shape
        num_pixels = H * W
        k = min(samples_per_file, num_pixels)
        indices = torch.randperm(num_pixels)[:k]
        flat = feat.reshape(C, -1).T  # [H*W, C]
        all_samples.append(flat[indices])
        total += k
        if total >= max_samples:
            break

    samples = torch.cat(all_samples, dim=0)
    if len(samples) > max_samples:
        idx = torch.randperm(len(samples))[:max_samples]
        samples = samples[idx]

    return samples.float()


def train_single_ae(config_key: str, samples: torch.Tensor,
                    feat_mean: torch.Tensor, feat_std: torch.Tensor,
                    save_path: Path, device: str,
                    epochs: int = 50, batch_size: int = 4096,
                    lr: float = 1e-3) -> AutoencoderFlexible:
    """
    训练单个 AutoEncoder

    Args:
        config_key: FEATURE_CONFIGS 中的 key
        samples: [N, C] 训练样本 (原始尺度, 未归一化)
        feat_mean: [C] per-channel 均值, 注入模型用于推理时一致归一化
        feat_std:  [C] per-channel 标准差
        save_path: 模型保存路径
        device: 计算设备
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率

    Returns:
        训练好的模型
    """
    config = FEATURE_CONFIGS[config_key]
    input_dim = config['input_dim']
    output_dim = config['encoder_hidden_dims'][-1]

    assert samples.shape[1] == input_dim, \
        f"样本维度 {samples.shape[1]} != 配置输入维度 {input_dim}"

    model = AutoencoderFlexible(
        input_dim=config['input_dim'],
        encoder_hidden_dims=config['encoder_hidden_dims'],
        decoder_hidden_dims=config['decoder_hidden_dims'],
    ).to(device)

    # 注入 per-channel 归一化参数, 消除特征尺度差异 (如 SD fine 均值~9.5 vs coarse ~0.7)
    model.set_input_norm(feat_mean, feat_std)
    print(f"    输入归一化已注入: mean∈[{feat_mean.min():.3f},{feat_mean.max():.3f}], "
          f"std∈[{feat_std.min():.3f},{feat_std.max():.3f}]")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)

            # AE 前向: normalize → encode → L2 norm → decode (在归一化空间重建)
            reconstructed = model(batch)

            # 目标: 归一化后的 batch (消除不同特征尺度的 MSE 差异)
            target = model._normalize_input(batch)

            # 重建损失: MSE + Cosine (方向+幅度都要对齐), 在归一化空间计算
            loss_mse = F_torch.mse_loss(reconstructed, target)
            cos_sim = F_torch.cosine_similarity(reconstructed, target, dim=1).mean()
            loss_cos = 1.0 - cos_sim
            loss = loss_mse + 0.5 * loss_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"(MSE={loss_mse.item():.4f}, cos={loss_cos.item():.4f})  "
                  f"best={best_loss:.6f}")

    # 加载最佳权重
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Calibrate: 用训练数据统计全局 bottleneck min/max
    model.calibrate(loader, device=device)
    torch.save(model.state_dict(), save_path)  # 保存含 calibration 数据的权重

    print(f"    → 最佳 loss: {best_loss:.6f}")
    print(f"    → 模型保存: {save_path}")
    return model


@torch.no_grad()
def compress_all_files(model: AutoencoderFlexible, input_dir: Path,
                       pattern: str, output_dir: Path,
                       device: str, target_dim: int):
    """
    用训练好的 AE 压缩目录中所有特征文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_files = sorted(glob.glob(str(input_dir / pattern)))

    for fp in tqdm(feat_files, desc="    压缩", leave=False):
        feat = torch.load(fp, map_location='cpu')  # [C, H, W]
        C, H, W = feat.shape
        flat = feat.reshape(C, -1).T.to(device)     # [H*W, C]

        compressed = model.encode(flat)              # [H*W, target_dim]
        result = compressed.reshape(H, W, target_dim).permute(2, 0, 1).cpu()  # [target_dim, H, W]

        # 重命名: 把原始通道数替换为压缩后通道数
        orig_name = Path(fp).name
        new_name = orig_name.replace(f"{C}x", f"{target_dim}x", 1)
        torch.save(result, output_dir / new_name)

    print(f"    → 压缩 {len(feat_files)} 个文件到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='多尺度特征 AutoEncoder 训练与压缩'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='extract_multiscale_features.py 的输出目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='压缩后特征输出目录')
    parser.add_argument('--ae_save_dir', type=str, required=True,
                        help='AutoEncoder 模型保存目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=500000,
                        help='训练采样像素数')
    parser.add_argument('--compress_only', action='store_true',
                        help='仅压缩, 使用已训练好的 AE 权重')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ae_save_dir = Path(args.ae_save_dir)
    ae_save_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for task in COMPRESS_TASKS:
        name = task['name']
        config_key = task['config_key']
        config = FEATURE_CONFIGS[config_key]
        input_dim = config['input_dim']
        target_dim = config['encoder_hidden_dims'][-1]
        feat_dir = input_dir / task['subdir']
        out_dir = output_dir / task['subdir']
        ae_path = ae_save_dir / f"ae_{name}.pth"

        print(f"\n{'='*60}")
        print(f"  [{task['desc']}]")
        print(f"{'='*60}")

        if not feat_dir.exists():
            print(f"  ⚠ 跳过: 目录不存在 {feat_dir}")
            continue

        if args.compress_only:
            # 仅压缩模式: 加载已有权重
            if not ae_path.exists():
                print(f"  ⚠ 跳过: 权重不存在 {ae_path}")
                continue
            model = AutoencoderFlexible(
                input_dim=config['input_dim'],
                encoder_hidden_dims=config['encoder_hidden_dims'],
                decoder_hidden_dims=config['decoder_hidden_dims'],
            ).to(args.device)
            model.load_state_dict(torch.load(ae_path, map_location=args.device), strict=False)
            model.eval()
            print(f"  加载已训练权重: {ae_path}")
        else:
            # 训练 + 压缩模式
            print(f"  采样训练数据...")
            samples = collect_pixel_samples(
                feat_dir, task['pattern'], args.max_samples
            )
            print(f"  训练样本: {samples.shape[0]} 个像素, {samples.shape[1]} 维")

            # 计算 per-channel 归一化统计并打印诊断信息
            feat_mean = samples.mean(dim=0)
            feat_std = samples.std(dim=0)
            print(f"  特征统计: mean=[{feat_mean.min():.3f}, {feat_mean.max():.3f}], "
                  f"std=[{feat_std.min():.3f}, {feat_std.max():.3f}], "
                  f"L2 norm (per pixel) = {samples.norm(dim=1).mean():.2f}")

            model = train_single_ae(
                config_key=config_key,
                samples=samples,
                feat_mean=feat_mean,
                feat_std=feat_std,
                save_path=ae_path,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )

        # 压缩所有文件
        compress_all_files(
            model=model,
            input_dir=feat_dir,
            pattern=task['pattern'],
            output_dir=out_dir,
            device=args.device,
            target_dim=target_dim,
        )

    # 复制 CLS tokens (无需压缩)
    cls_src = input_dir / 'cls'
    cls_dst = output_dir / 'cls'
    if cls_src.exists():
        cls_dst.mkdir(parents=True, exist_ok=True)
        cls_files = sorted(glob.glob(str(cls_src / '*.pt')))
        for fp in cls_files:
            shutil.copy2(fp, cls_dst / Path(fp).name)
        print(f"\n  CLS tokens: 复制 {len(cls_files)} 个文件到 {cls_dst}")

    elapsed = time.time() - t_start

    # 汇总
    print(f"\n{'='*60}")
    print(f"  ✓ 多尺度 AutoEncoder 降维完成! ({elapsed:.1f}s)")
    for task in COMPRESS_TASKS:
        config = FEATURE_CONFIGS[task['config_key']]
        in_d = config['input_dim']
        out_d = config['encoder_hidden_dims'][-1]
        print(f"    {task['name']:12s}: {in_d}d → {out_d}d")
    fine_total = (FEATURE_CONFIGS['v2_fine_sd']['encoder_hidden_dims'][-1] +
                  FEATURE_CONFIGS['v2_fine_dino']['encoder_hidden_dims'][-1])
    mid_d = FEATURE_CONFIGS['v2_mid']['encoder_hidden_dims'][-1]
    coarse_d = FEATURE_CONFIGS['v2_coarse']['encoder_hidden_dims'][-1]
    total_3dgs = fine_total + mid_d + coarse_d
    print(f"    {'CLS':12s}: 768d → 768d (不嵌入 3DGS)")
    print(f"    3DGS 总嵌入维度: fine({fine_total}d) + mid({mid_d}d) + coarse({coarse_d}d) = {total_3dgs}d")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
