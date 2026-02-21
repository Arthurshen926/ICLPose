#!/usr/bin/env python3
"""
特征压缩脚本
将768维融合特征压缩为256维

使用方法:
    python scripts/compress_features.py \
        --input_dir dataset/room_0/Synthetic_Train/features_raw \
        --output_dir dataset/room_0/Synthetic_Train/fused_feat \
        --model_path dataset/room_0/Sequence_1/ae_models/ae_fused_256.pth \
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


def visualize_compression(original, compressed, output_path):
    """可视化压缩前后特征对比"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始特征 (768维)
        C1, H1, W1 = original.shape
        orig_flat = original.reshape(C1, -1).T  # (H*W, C)
        if isinstance(orig_flat, torch.Tensor):
            orig_flat = orig_flat.cpu().numpy()
        
        # PCA降维
        pca_orig = PCA(n_components=3)
        orig_pca = pca_orig.fit_transform(orig_flat)
        orig_pca = (orig_pca - orig_pca.min()) / (orig_pca.max() - orig_pca.min() + 1e-8)
        orig_pca = orig_pca.reshape(H1, W1, 3)
        
        # 压缩特征 (256维)
        if isinstance(compressed, torch.Tensor):
            compressed = compressed.cpu().numpy()
        C2, H2, W2 = compressed.shape
        comp_flat = compressed.reshape(C2, -1).T  # (H*W, C)
        
        pca_comp = PCA(n_components=3)
        comp_pca = pca_comp.fit_transform(comp_flat)
        comp_pca = (comp_pca - comp_pca.min()) / (comp_pca.max() - comp_pca.min() + 1e-8)
        comp_pca = comp_pca.reshape(H2, W2, 3)
        
        # 第一行: 原始特征
        axes[0, 0].imshow(orig_pca)
        axes[0, 0].set_title(f'Original PCA ({C1} dims)')
        axes[0, 0].axis('off')
        
        orig_norm = np.linalg.norm(original.cpu().numpy() if isinstance(original, torch.Tensor) else original, axis=0)
        im1 = axes[0, 1].imshow(orig_norm, cmap='viridis')
        axes[0, 1].set_title('Original Norm')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 原始特征通道分布
        axes[0, 2].hist(orig_flat.flatten(), bins=50, alpha=0.7, label='Original')
        axes[0, 2].set_title('Original Feature Distribution')
        axes[0, 2].set_xlabel('Value')
        axes[0, 2].set_ylabel('Frequency')
        
        # 第二行: 压缩特征
        axes[1, 0].imshow(comp_pca)
        axes[1, 0].set_title(f'Compressed PCA ({C2} dims)')
        axes[1, 0].axis('off')
        
        comp_norm = np.linalg.norm(compressed, axis=0)
        im2 = axes[1, 1].imshow(comp_norm, cmap='viridis')
        axes[1, 1].set_title('Compressed Norm')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 压缩特征通道分布
        axes[1, 2].hist(comp_flat.flatten(), bins=50, alpha=0.7, color='orange', label='Compressed')
        axes[1, 2].set_title('Compressed Feature Distribution')
        axes[1, 2].set_xlabel('Value')
        axes[1, 2].set_ylabel('Frequency')
        
        # 压缩比
        compression_ratio = C1 / C2
        plt.suptitle(f'Feature Compression: {C1}→{C2} dims (ratio: {compression_ratio:.2f}x)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
    except ImportError as e:
        print(f"警告: 无法生成可视化 ({e})")


def main():
    parser = argparse.ArgumentParser(description='压缩融合特征')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='原始特征目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='压缩特征输出目录')
    parser.add_argument('--model_path', type=str, required=True,
                        help='AutoEncoder模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--save_npy', action='store_true', default=True,
                        help='保存为.npy格式 (默认)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成压缩前后对比可视化')
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
    
    # 查找特征文件
    feature_files = sorted(glob.glob(str(input_dir / '*_fused_*.pt')))
    print(f"找到 {len(feature_files)} 个特征文件")
    
    if len(feature_files) == 0:
        print(f"错误: 在 {input_dir} 中未找到特征文件")
        return
    
    # 初始化压缩器
    print("\n初始化特征压缩器...")
    from feature_compression import FeatureCompressor
    compressor = FeatureCompressor(
        model_path=args.model_path,
        device=args.device
    )
    
    # 压缩特征
    print("\n开始压缩特征...")
    vis_count = 0
    for i, feat_path in enumerate(tqdm(feature_files, desc="压缩特征")):
        feat_path = Path(feat_path)
        
        # 加载原始特征
        features = torch.load(feat_path)
        if features.dim() == 4:
            features = features.squeeze(0)  # (C, H, W)
        
        # 压缩
        compressed = compressor.compress(features, return_numpy=True)  # (C', H, W)
        
        # 确定输出文件名
        stem = feat_path.stem.split('_fused_')[0]
        
        # 匹配现有数据格式: fused_feat_XXXX.npy
        # 从stem中提取编号
        if stem.startswith('frame_'):
            frame_id = int(stem.split('_')[1])
        elif stem.startswith('rgb_'):
            frame_id = int(stem.split('_')[1])
        else:
            # 尝试解析数字
            try:
                frame_id = int(''.join(filter(str.isdigit, stem)))
            except:
                frame_id = abs(hash(stem)) % 10000
        
        output_name = f"fused_feat_{frame_id:04d}.npy"
        output_path = output_dir / output_name
        
        # 保存
        np.save(str(output_path), compressed)
        
        # 可视化
        if args.visualize and i % args.vis_interval == 0:
            vis_path = vis_dir / f"compress_vis_{frame_id:04d}.png"
            visualize_compression(features, compressed, vis_path)
            vis_count += 1
    
    print(f"\n✓ 完成! 压缩特征保存在: {output_dir}")
    print(f"  原始维度: 768")
    print(f"  压缩维度: {compressor.output_dim}")
    print(f"  共 {len(feature_files)} 个文件")
    if args.visualize:
        print(f"  共 {vis_count} 个可视化文件在: {vis_dir}")


if __name__ == '__main__':
    main()

