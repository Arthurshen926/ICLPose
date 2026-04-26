"""
下载 SegFormer 模型到本地缓存
==============================
在有网络时运行一次即可，之后离线也能使用。

用法:
  # 默认下载 SegFormer-B2 ADE20K
  python scripts/data_prep/download_segformer.py

  # 下载到指定目录 (便于离线迁移)
  python scripts/data_prep/download_segformer.py --save_dir models/segformer-b2-ade

  # 下载后验证
  python scripts/data_prep/download_segformer.py --verify

模型信息:
  - nvidia/segformer-b2-finetuned-ade-512-512
  - 24.7M 参数, ADE20K 150类语义分割
  - 推理速度: ~30ms/帧 (RTX 3090, 512x512)
  - 硬盘占用: ~100MB
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="下载 SegFormer 模型")
    parser.add_argument("--model_name", default="nvidia/segformer-b2-finetuned-ade-512-512",
                        help="HuggingFace 模型名称")
    parser.add_argument("--save_dir", default=None,
                        help="保存到指定本地目录 (可选, 默认使用 HF cache)")
    parser.add_argument("--verify", action="store_true",
                        help="下载后验证模型能否加载")
    parser.add_argument("--mirror", default=None,
                        help="HuggingFace 镜像 URL (如 https://hf-mirror.com)")
    args = parser.parse_args()

    # 设置镜像
    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror
        print(f"[Mirror] 使用镜像: {args.mirror}")

    try:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
    except ImportError:
        print("[ERROR] 请先安装 transformers: pip install transformers")
        sys.exit(1)

    print(f"[Download] 正在下载 {args.model_name} ...")

    # 下载 processor
    print("  → ImageProcessor ...")
    processor = SegformerImageProcessor.from_pretrained(args.model_name)

    # 下载 model
    print("  → Model weights ...")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Download] 完成! ({n_params:.1f}M 参数)")

    # 保存到指定目录
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        processor.save_pretrained(args.save_dir)
        model.save_pretrained(args.save_dir)
        print(f"[Save] 模型已保存到 {args.save_dir}")
        print(f"[使用] 预处理时指定: --segformer_model {args.save_dir}")

    # 验证
    if args.verify:
        import torch
        import numpy as np

        print("\n[Verify] 运行推理测试 ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        # 创建测试图像
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            seg_map = torch.nn.functional.interpolate(
                logits, size=(480, 640), mode="bilinear", align_corners=False
            ).argmax(dim=1).squeeze(0)

        print(f"  输入: {dummy_image.shape}")
        print(f"  输出: {seg_map.shape}, 类别范围: [{seg_map.min()}, {seg_map.max()}]")
        print(f"  设备: {device}")
        print("[Verify] 通过! ✓")

    # 使用提示 (如果镜像可用)
    print("\n" + "=" * 50)
    print("如果默认 HuggingFace 无法访问, 尝试镜像:")
    print("  python scripts/data_prep/download_segformer.py \\")
    print("    --mirror https://hf-mirror.com \\")
    print("    --save_dir models/segformer-b2-ade --verify")
    print("=" * 50)


if __name__ == "__main__":
    main()
