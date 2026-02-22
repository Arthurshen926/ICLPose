#!/usr/bin/env python3
"""
轻量 DINO CLS Token 提取 (仅加载 DINO, 不需要 SD)
====================================================
用于快速为新序列生成 CLS tokens, 用于跨序列检索验证。

用法:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/extract_cls_only.py \
        --input_dir dataset/room_0/Sequence_2/rgb \
        --output_dir output/features_multiscale_compressed/room_0_seq2/cls
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
    parser = argparse.ArgumentParser(description='Extract DINO CLS tokens only')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from feature_extraction.extractor_dino import ViTExtractor
    from PIL import Image

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 DINO 模型
    print("[CLS Extractor] 加载 DINOv2 模型...")
    extractor = ViTExtractor('dinov2_vitb14', stride=14, device=args.device)
    print("  ✓ DINOv2 加载完成")

    # 扫描图像
    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob('rgb_*.png'))
    if not image_files:
        image_files = sorted(input_dir.glob('*.png')) + sorted(input_dir.glob('*.jpg'))
    print(f"  找到 {len(image_files)} 张图像")

    torch.set_grad_enabled(False)

    for img_path in tqdm(image_files, desc="提取 CLS tokens"):
        # 解析 frame id
        stem = img_path.stem  # e.g. 'rgb_42'
        parts = stem.split('_')
        fid = parts[1] if len(parts) >= 2 else parts[0]

        # 加载并预处理
        img = Image.open(img_path).convert('RGB')
        w, h = img.size  # 640, 480
        dino_w = int(math.ceil(w / 14) * 14)
        dino_h = int(math.ceil(h / 14) * 14)
        img_resized = img.resize((dino_w, dino_h), Image.Resampling.BILINEAR)
        img_batch = extractor.preprocess_pil(img_resized)

        # 提取 CLS token
        feats = extractor.extract_descriptors(
            img_batch.to(args.device), layer=11, facet='token',
            include_cls=True
        )
        cls_token = feats[:, :, 0, :].squeeze()  # [768]
        cls_token = F.normalize(cls_token.float(), p=2, dim=-1).cpu()

        # 保存
        save_path = out_dir / f'rgb_{fid}_cls_768.pt'
        torch.save(cls_token, save_path)

    print(f"\n完成! {len(image_files)} 个 CLS tokens 保存至 {out_dir}")


if __name__ == '__main__':
    main()
