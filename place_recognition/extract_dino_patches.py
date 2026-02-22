#!/usr/bin/env python3
"""
DINO Patch Token 提取 (用于 VLAD 场景检索)
=============================================
从图像提取 DINO patch-level 特征 (768d @ H/14 × W/14)。
比完整多尺度提取轻量很多，只需要 DINO 模型。

用法:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/extract_dino_patches.py \
        --input_dir dataset/room_0/Sequence_2/rgb \
        --output_dir output/features_multiscale/room_0_seq2/fine_dino

也可以同时提取 CLS token:
    --save_cls --cls_output_dir output/features_multiscale/room_0_seq2/cls
"""
import argparse
import math
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Extract DINO patch tokens')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='图像目录 (包含 rgb_*.png)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录 (fine_dino patch features)')
    parser.add_argument('--save_cls', action='store_true',
                        help='同时保存 CLS token')
    parser.add_argument('--cls_output_dir', type=str, default=None,
                        help='CLS token 输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_height', type=int, default=480,
                        help='目标图像高度 (影响 patch grid 大小)')
    parser.add_argument('--target_width', type=int, default=640,
                        help='目标图像宽度')
    args = parser.parse_args()

    from feature_extraction.extractor_dino import ViTExtractor
    from PIL import Image

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_cls:
        cls_dir = Path(args.cls_output_dir) if args.cls_output_dir else out_dir.parent / 'cls'
        cls_dir.mkdir(parents=True, exist_ok=True)

    # 加载 DINO
    print("[DINO Patch Extractor] 加载 DINOv2 模型...")
    extractor = ViTExtractor('dinov2_vitb14', stride=14, device=args.device)
    print("  ✓ DINOv2 加载完成")

    # 计算 patch grid: 640×480 → 644×490 (nearest 14-alg), grid = 46×35
    dino_w = int(math.ceil(args.target_width / 14) * 14)
    dino_h = int(math.ceil(args.target_height / 14) * 14)
    grid_w = dino_w // 14
    grid_h = dino_h // 14
    print(f"  输入: {args.target_width}×{args.target_height}")
    print(f"  DINO: {dino_w}×{dino_h} → patch grid: {grid_w}×{grid_h}")
    print(f"  Patch 数量: {grid_w * grid_h} × 768d")

    # 扫描图像
    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob('rgb_*.png'))
    if not image_files:
        image_files = sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg'))
    print(f"  找到 {len(image_files)} 张图像")

    torch.set_grad_enabled(False)

    for img_path in tqdm(image_files, desc="提取 DINO patches"):
        # 解析 frame id
        stem = img_path.stem
        parts = stem.split('_')
        fid = parts[1] if len(parts) >= 2 else parts[0]

        # 加载并预处理
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((dino_w, dino_h), Image.Resampling.BILINEAR)
        img_batch = extractor.preprocess_pil(img_resized)

        # 提取 (含 CLS)
        feats = extractor.extract_descriptors(
            img_batch.to(args.device), layer=11, facet='token',
            include_cls=True
        )
        # feats: [1, 1, 1+grid_h*grid_w, 768]

        cls_token = feats[:, :, 0, :].squeeze()  # [768]
        patch_tokens = feats[:, :, 1:, :].squeeze()  # [grid_h*grid_w, 768]

        # Reshape to spatial: [768, grid_h, grid_w]
        patch_map = patch_tokens.reshape(grid_h, grid_w, -1).permute(2, 0, 1)  # [768, H, W]
        # 不做 L2 归一化: VLAD 需要原始特征 (magnitude 携带信息)
        # AnyLoc 论文也使用原始 DINOv2 patch tokens
        patch_map = patch_map.float().cpu()

        D, H, W = patch_map.shape
        save_path = out_dir / f'rgb_{fid}_fine_dino_{D}x{H}x{W}.pt'
        torch.save(patch_map, save_path)

        if args.save_cls:
            cls_norm = F.normalize(cls_token.float(), p=2, dim=-1).cpu()
            cls_save = cls_dir / f'rgb_{fid}_cls_768.pt'
            torch.save(cls_norm, cls_save)

    print(f"\n完成! {len(image_files)} 帧 DINO patch features 保存至 {out_dir}")
    if args.save_cls:
        print(f"  CLS tokens 保存至 {cls_dir}")


if __name__ == '__main__':
    main()
