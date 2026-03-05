#!/usr/bin/env python3
"""
多尺度特征提取脚本 v2 (SD-Primary Coarse-to-Fine)

与 v1 的关键区别:
  - SD 特征保留 UNet 零填充后的原生分辨率 (不裁剪), 形成干净的 2× 层级:
      coarse: 1280d @  8×10  (SD s5, padded)
      mid:    1280d @ 16×20  (SD s4, padded)
      fine_sd:  640d @ 32×40  (SD s3, padded, 不上采样到 DINO 网格)
  - DINO 保持原生 patch 网格: 768d @ 35×46
  - 这样 3DGS 训练时各尺度直接匹配, 无需对齐

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_v2.py \\
        --input_dir dataset/room_0/Sequence_1/rgb \\
        --output_dir output/features_v2/room_0

输出 (每帧 4 个文件):
    sd_s5/rgb_0000_sd_s5_1280x8x10.pt
    sd_s4/rgb_0000_sd_s4_1280x16x20.pt
    sd_s3/rgb_0000_sd_s3_640x32x40.pt
    dino/rgb_0000_dino_768x35x46.pt
"""
import os
import sys
import math
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("HF_HOME", "/home/yons/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/home/yons/.cache/torch")


def extract_v2(sd_model, sd_aug, extractor_vit, img_path, device='cuda'):
    """
    提取 v2 格式的多尺度特征 (SD 保留 padded 原生分辨率).

    Returns:
        dict with keys: 'sd_s3', 'sd_s4', 'sd_s5', 'dino'
        每个 value 是 CPU tensor [C, H, W]
    """
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size  # 640, 480

    # ── SD ──
    sd_image_size = 480
    scale = sd_image_size / min(orig_w, orig_h)
    sd_w = int(round(orig_w * scale))
    sd_h = int(round(orig_h * scale))
    img_sd = img.resize((sd_w, sd_h), Image.Resampling.LANCZOS)

    from feature_extraction.extractor_sd import process_features_and_mask
    feats_sd = process_features_and_mask(
        sd_model, sd_aug, img_sd, mask=False, raw=True
    )

    # 不裁剪! 保留 UNet 零填充后的完整分辨率
    # s3: [1, 640, 32, 40], s4: [1, 1280, 16, 20], s5: [1, 1280, 8, 10]
    sd_s3 = feats_sd['s3'].squeeze(0).cpu()  # [640, 32, 40]
    sd_s4 = feats_sd['s4'].squeeze(0).cpu()  # [1280, 16, 20]
    sd_s5 = feats_sd['s5'].squeeze(0).cpu()  # [1280, 8, 10]

    # ── DINO ──
    dino_w = int(math.ceil(sd_w / 14) * 14)  # 644
    dino_h = int(math.ceil(sd_h / 14) * 14)  # 490
    tokens_h, tokens_w = dino_h // 14, dino_w // 14  # 35, 46

    img_dino = img.resize((dino_w, dino_h), Image.Resampling.BILINEAR)
    img_batch = extractor_vit.preprocess_pil(img_dino)
    feats_dino = extractor_vit.extract_descriptors(
        img_batch.to(device), layer=11, facet='token', include_cls=False
    )
    dino = feats_dino.permute(0, 1, 3, 2).reshape(
        1, -1, tokens_h, tokens_w
    ).squeeze(0).cpu()  # [768, 35, 46]

    return {
        'sd_s3': sd_s3,
        'sd_s4': sd_s4,
        'sd_s5': sd_s5,
        'dino': dino,
    }


def main():
    parser = argparse.ArgumentParser(
        description='v2 多尺度特征提取 (SD padded 原生分辨率 + DINO 原生)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='RGB 图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出根目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始帧 (断点续提)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 子目录: 使用 SD 层名 (避免与 v1 的 fine_sd/mid/coarse 混淆)
    subdirs = {}
    for sub in ['sd_s3', 'sd_s4', 'sd_s5', 'dino']:
        d = output_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        subdirs[sub] = d

    # 查找图像
    image_paths = sorted(
        glob.glob(str(input_dir / '*.png')) +
        glob.glob(str(input_dir / '*.jpg'))
    )
    print(f"找到 {len(image_paths)} 张图像")
    if not image_paths:
        print(f"错误: 未找到图像 in {input_dir}")
        return

    # 加载模型
    print("\n加载 SD 模型...")
    from feature_extraction.extractor_sd import load_model
    sd_model, sd_aug = load_model(
        diffusion_ver='v1-5', image_size=480,
        num_timesteps=50, block_indices=[2, 5, 8, 11]
    )
    print("加载 DINO 模型...")
    from feature_extraction.extractor_dino import ViTExtractor
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device=args.device)
    print("模型加载完成\n")

    # 提取
    for i, img_path in enumerate(tqdm(image_paths, desc="提取 v2 特征")):
        if i < args.start_idx:
            continue

        stem = Path(img_path).stem
        feats = extract_v2(sd_model, sd_aug, extractor_vit, img_path, args.device)

        # 保存
        for key, tensor in feats.items():
            shape_str = 'x'.join(str(s) for s in tensor.shape)
            torch.save(tensor, subdirs[key] / f"{stem}_{key}_{shape_str}.pt")

        if i == 0:
            print(f"\n  特征尺寸 (v2 native):")
            for key, tensor in feats.items():
                print(f"    {key:8s}: {list(tensor.shape)}")
            print()

    print(f"\n✓ 完成! 共 {len(image_paths)} 帧")
    for key, d in subdirs.items():
        print(f"  {key:8s} → {d}")


if __name__ == '__main__':
    main()
