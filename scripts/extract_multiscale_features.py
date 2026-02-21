#!/usr/bin/env python3
"""
多尺度特征提取脚本 (v2-iterative-routing)

从 RGB 图像提取多尺度特征金字塔 + DINO CLS Token:
  - fine_sd  : SD s3 (640d)  @ 35×46 (上采样到 DINO 网格)
  - fine_dino: DINO Patch (768d) @ 35×46
  - mid      : SD s4 (1280d) @ ~15×20
  - coarse   : SD s5 (1280d) @ ~8×10
  - cls      : DINO CLS Token (768d)

注意: fine_sd 和 fine_dino 分布差异大, 不直接拼接.
      后续由各自的 AutoEncoder 独立压缩后再拼接嵌入 3DGS.

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_multiscale_features.py \\
        --input_dir dataset/room_0/Sequence_1/rgb \\
        --output_dir output/features_multiscale/room_0 \\
        --visualize --vis_interval 50

输出结构 (output_dir/):
    fine_sd/rgb_0000_fine_sd_640x35x46.pt     # [640 , 35, 46]
    fine_dino/rgb_0000_fine_dino_768x35x46.pt # [768 , 35, 46]
    mid/rgb_0000_mid_1280x15x20.pt            # [1280, 15, 20]
    coarse/rgb_0000_coarse_1280x8x10.pt       # [1280,  8, 10]
    cls/rgb_0000_cls_768.pt                   # [768]
    vis/rgb_0000_vis.png                      # (可选) PCA 可视化
"""
import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_multiscale_pca(features, rgb_image, output_path):
    """
    对三层特征分别做 PCA→RGB 可视化, 上采样到原始分辨率对比
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from PIL import Image
        import torch.nn.functional as F

        if isinstance(rgb_image, (str, Path)):
            pil_rgb = Image.open(rgb_image).convert('RGB')
        else:
            pil_rgb = Image.fromarray(rgb_image)
        rgb_arr = np.array(pil_rgb)
        orig_H, orig_W = rgb_arr.shape[:2]

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(
            f'Multi-Scale Feature Pyramid  |  RGB {orig_W}×{orig_H}',
            fontsize=12, fontweight='bold'
        )

        # RGB
        axes[0].imshow(rgb_arr)
        axes[0].set_title(f'RGB ({orig_W}×{orig_H})', fontsize=10)
        axes[0].axis('off')

        level_names = ['Fine SD (s3)', 'Fine DINO', 'Mid (s4)', 'Coarse (s5)']
        level_keys = ['fine_sd', 'fine_dino', 'mid', 'coarse']

        for idx, (name, key) in enumerate(zip(level_names, level_keys)):
            feat = features[key]  # [C, H, W]
            C, fH, fW = feat.shape
            feat_flat = feat.reshape(C, -1).T.cpu().numpy()
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat).reshape(fH, fW, 3)
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)

            # 上采样到原始分辨率
            pca_t = torch.from_numpy(feat_pca).permute(2, 0, 1).unsqueeze(0).float()
            pca_up = F.interpolate(pca_t, size=(orig_H, orig_W),
                                   mode='bilinear', align_corners=False)
            pca_up = pca_up.squeeze(0).permute(1, 2, 0).numpy()

            axes[idx + 1].imshow(np.clip(pca_up, 0, 1))
            axes[idx + 1].set_title(f'{name}\n{C}d @ {fW}×{fH}', fontsize=10)
            axes[idx + 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

    except ImportError as e:
        print(f"  警告: 无法生成可视化 ({e})")


def main():
    parser = argparse.ArgumentParser(
        description='提取多尺度特征金字塔 (Fine/Mid/Coarse + CLS Token)'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='RGB 图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出特征根目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize', action='store_true',
                        help='生成 PCA 可视化')
    parser.add_argument('--vis_interval', type=int, default=50,
                        help='可视化间隔 (每 N 帧)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 创建子目录
    dirs = {}
    for sub in ['fine_sd', 'fine_dino', 'mid', 'coarse', 'cls']:
        d = output_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        dirs[sub] = d
    if args.visualize:
        dirs['vis'] = output_dir / 'vis'
        dirs['vis'].mkdir(parents=True, exist_ok=True)

    # 查找图像
    image_paths = sorted(
        glob.glob(str(input_dir / '*.png')) +
        glob.glob(str(input_dir / '*.jpg'))
    )
    print(f"找到 {len(image_paths)} 张图像")
    if not image_paths:
        print(f"错误: 在 {input_dir} 中未找到图像")
        return

    # 初始化提取器
    print("\n初始化多尺度特征提取器...")
    from feature_extraction.multiscale_extractor import MultiScaleFeatureExtractor
    extractor = MultiScaleFeatureExtractor(device=args.device)

    # 提取特征
    print("\n开始提取多尺度特征...")
    vis_count = 0
    for i, img_path in enumerate(tqdm(image_paths, desc="提取特征")):
        img_path = Path(img_path)
        stem = img_path.stem

        ms = extractor.extract(img_path)

        # 保存各层特征 (fine_sd 和 fine_dino 独立保存, 不拼接)
        def _save(tensor, subdir, tag):
            shape_str = 'x'.join(str(s) for s in tensor.shape)
            torch.save(tensor, dirs[subdir] / f"{stem}_{tag}_{shape_str}.pt")

        _save(ms.fine_sd,   'fine_sd',   'fine_sd')
        _save(ms.fine_dino, 'fine_dino', 'fine_dino')
        _save(ms.mid,       'mid',       'mid')
        _save(ms.coarse,    'coarse',    'coarse')
        torch.save(ms.cls_token, dirs['cls'] / f"{stem}_cls_{ms.cls_token.shape[0]}.pt")

        # 可视化
        if args.visualize and i % args.vis_interval == 0:
            vis_path = dirs['vis'] / f"{stem}_multiscale_vis.png"
            visualize_multiscale_pca(
                {'fine_sd': ms.fine_sd, 'fine_dino': ms.fine_dino,
                 'mid': ms.mid, 'coarse': ms.coarse},
                str(img_path),
                vis_path,
            )
            vis_count += 1

        # 第一帧打印尺寸信息
        if i == 0:
            print(f"\n  特征尺寸:")
            print(f"    Fine SD   : {list(ms.fine_sd.shape)}  (SD s3, 上采样到 DINO 网格)")
            print(f"    Fine DINO : {list(ms.fine_dino.shape)}  (DINO Patch Tokens)")
            print(f"    Mid       : {list(ms.mid.shape)}  (SD s4)")
            print(f"    Coarse    : {list(ms.coarse.shape)}  (SD s5)")
            print(f"    CLS       : {list(ms.cls_token.shape)}  (DINO CLS Token)")
            print()

    print(f"\n✓ 完成! 共提取 {len(image_paths)} 帧多尺度特征")
    print(f"  Fine SD   → {dirs['fine_sd']}")
    print(f"  Fine DINO → {dirs['fine_dino']}")
    print(f"  Mid       → {dirs['mid']}")
    print(f"  Coarse    → {dirs['coarse']}")
    print(f"  CLS       → {dirs['cls']}")
    if args.visualize:
        print(f"  可视化 → {dirs['vis']}  ({vis_count} 张)")


if __name__ == '__main__':
    main()
