#!/usr/bin/env python3
"""
融合特征提取脚本
从RGB图像提取DINO+SD融合特征

使用方法:
    python scripts/extract_fused_features.py \
        --input_dir dataset/room_0/Synthetic_Train/rgb \
        --output_dir dataset/room_0/Synthetic_Train/features_raw \
        --visualize --vis_interval 50
"""
import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_features_pca(features, rgb_image, output_path):
    """使用PCA可视化特征，并将特征图上采样到原始RGB分辨率进行空间对齐对比。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from sklearn.decomposition import PCA
        from PIL import Image
        import torch.nn.functional as F

        # 加载 RGB
        if isinstance(rgb_image, (str, Path)):
            pil_rgb = Image.open(rgb_image).convert('RGB')
        else:
            pil_rgb = Image.fromarray(rgb_image)
        rgb_arr = np.array(pil_rgb)          # (orig_H, orig_W, 3)
        orig_H, orig_W = rgb_arr.shape[:2]

        # features: (C, H, W) token grid → PCA → [H, W, 3]
        C, fH, fW = features.shape
        feat_flat = features.reshape(C, -1).T.cpu().numpy()  # (fH*fW, C)
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_flat).reshape(fH, fW, 3)  # (fH, fW, 3)
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)

        # 上采样 PCA 结果和 norm 热图到原始 RGB 分辨率，确保空间对齐
        pca_t = torch.from_numpy(feat_pca).permute(2, 0, 1).unsqueeze(0).float()
        pca_up = F.interpolate(pca_t, size=(orig_H, orig_W),
                               mode='bilinear', align_corners=False)
        pca_up = pca_up.squeeze(0).permute(1, 2, 0).numpy()  # (orig_H, orig_W, 3)

        feat_norm = torch.from_numpy(
            np.linalg.norm(features.cpu().numpy(), axis=0)
        ).unsqueeze(0).unsqueeze(0).float()
        norm_up = F.interpolate(feat_norm, size=(orig_H, orig_W),
                                mode='bilinear', align_corners=False)
        norm_up = norm_up.squeeze().numpy()  # (orig_H, orig_W)

        # 绘图：3列等宽，相同 axis 尺寸
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f'Fused Feature Visualization  |  '
            f'RGB {orig_W}×{orig_H}  |  Token Grid {fW}×{fH}',
            fontsize=11, fontweight='bold'
        )

        axes[0].imshow(rgb_arr)
        axes[0].set_title(f'RGB Image  ({orig_W}×{orig_H})', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(np.clip(pca_up, 0, 1))
        axes[1].set_title(
            f'Fused Feature PCA→RGB\n(token grid {fW}×{fH}, upsampled to {orig_W}×{orig_H})',
            fontsize=10
        )
        axes[1].axis('off')

        im = axes[2].imshow(norm_up, cmap='viridis')
        axes[2].set_title('Feature L2-Norm Heatmap\n(upsampled to original res)', fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

    except ImportError as e:
        print(f"警告: 无法生成可视化 ({e})")


def main():
    parser = argparse.ArgumentParser(description='提取DINO+SD融合特征')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='RGB图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出特征目录')
    parser.add_argument('--aggregator_weights', type=str, default=None,
                        help='AggregationNetwork权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--visualize', action='store_true',
                        help='生成PCA可视化')
    parser.add_argument('--vis_interval', type=int, default=50,
                        help='可视化间隔 (每N帧可视化一次)')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='可视化输出目录 (默认: output_dir/visualizations)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化目录
    if args.visualize:
        vis_dir = Path(args.vis_dir) if args.vis_dir else output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找图像
    image_paths = sorted(
        glob.glob(str(input_dir / '*.png')) + 
        glob.glob(str(input_dir / '*.jpg'))
    )
    print(f"找到 {len(image_paths)} 张图像")
    
    if len(image_paths) == 0:
        print(f"错误: 在 {input_dir} 中未找到图像")
        return
    
    # 初始化提取器
    print("\n初始化特征提取器...")
    from feature_extraction import FusedFeatureExtractor
    extractor = FusedFeatureExtractor(
        device=args.device,
        aggregator_weights=args.aggregator_weights
    )
    
    # 提取特征
    print("\n开始提取特征...")
    vis_count = 0
    for i, img_path in enumerate(tqdm(image_paths, desc="提取特征")):
        img_path = Path(img_path)
        
        # 提取
        features = extractor.extract(img_path)  # (768, H, W)
        
        # 保存
        stem = img_path.stem
        output_path = output_dir / f"{stem}_fused_{features.shape[0]}x{features.shape[1]}x{features.shape[2]}.pt"
        torch.save(features.unsqueeze(0), output_path)  # 保存为 (1, C, H, W)
        
        # 可视化
        if args.visualize and i % args.vis_interval == 0:
            vis_path = vis_dir / f"{stem}_vis.png"
            visualize_features_pca(features, str(img_path), vis_path)
            vis_count += 1
    
    print(f"\n✓ 完成! 特征保存在: {output_dir}")
    print(f"  共 {len(image_paths)} 个特征文件")
    if args.visualize:
        print(f"  共 {vis_count} 个可视化文件在: {vis_dir}")


if __name__ == '__main__':
    main()

