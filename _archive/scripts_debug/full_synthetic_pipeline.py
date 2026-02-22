#!/usr/bin/env python3
"""
完整合成数据生成流水线

整合以下步骤:
1. 位姿生成 (扰动 + 插值)
2. 3DGS RGB渲染
3. DINO+SD融合特征提取
4. AutoEncoder特征压缩

使用方法:
    python scripts/full_synthetic_pipeline.py \
        --sequence_dir dataset/room_0/Sequence_1 \
        --output_dir dataset/room_0/Synthetic_Train \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import time


def run_step(step_name, cmd, cwd=None):
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n✗ 步骤 '{step_name}' 失败 (耗时 {elapsed:.1f}s)")
        return False
    
    print(f"\n✓ 步骤 '{step_name}' 完成 (耗时 {elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description='完整合成数据生成流水线')
    parser.add_argument('--sequence_dir', type=str, required=True,
                        help='原始序列目录 (包含traj_tum.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--ply_path', type=str, required=True,
                        help='3DGS PLY文件路径')
    parser.add_argument('--ae_model_path', type=str, default=None,
                        help='AutoEncoder模型路径 (默认使用序列目录中的)')
    
    # 位姿生成参数
    parser.add_argument('--sample_step', type=int, default=5,
                        help='采样步长')
    parser.add_argument('--num_perturbed', type=int, default=3,
                        help='每个基础位姿生成的扰动数')
    parser.add_argument('--num_interpolated', type=int, default=2,
                        help='每对相邻位姿间的插值数')
    
    # 相机参数
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    
    # 跳过步骤
    parser.add_argument('--skip_poses', action='store_true',
                        help='跳过位姿生成')
    parser.add_argument('--skip_render', action='store_true',
                        help='跳过RGB渲染')
    parser.add_argument('--skip_features', action='store_true',
                        help='跳过特征提取')
    parser.add_argument('--skip_compress', action='store_true',
                        help='跳过特征压缩')
    
    args = parser.parse_args()
    
    # 路径处理
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    scripts_dir = Path(__file__).parent
    project_dir = scripts_dir.parent
    
    # 默认AE模型路径
    if args.ae_model_path is None:
        args.ae_model_path = str(sequence_dir / 'ae_models' / 'ae_fused.pth')
    
    print("=" * 60)
    print("合成数据生成流水线")
    print("=" * 60)
    print(f"输入序列: {sequence_dir}")
    print(f"输出目录: {output_dir}")
    print(f"PLY文件: {args.ply_path}")
    print(f"AE模型: {args.ae_model_path}")
    
    total_start = time.time()
    
    # ==================== Step 1: 位姿生成 ====================
    if not args.skip_poses:
        cmd = [
            'python', str(scripts_dir / 'generate_synthetic_data.py'),
            '--sample_step', str(args.sample_step),
            '--num_perturbed', str(args.num_perturbed),
            '--num_interpolated', str(args.num_interpolated),
        ]
        if not run_step("位姿生成", cmd, cwd=str(project_dir)):
            return 1
    
    poses_file = output_dir / 'traj_tum.txt'
    if not poses_file.exists():
        print(f"✗ 位姿文件不存在: {poses_file}")
        return 1
    
    # ==================== Step 2: 3DGS渲染 ====================
    rgb_dir = output_dir / 'rgb'
    if not args.skip_render:
        cmd = [
            'python', str(scripts_dir / 'render_3dgs.py'),
            '--poses_file', str(poses_file),
            '--output_dir', str(rgb_dir),
            '--ply_path', args.ply_path,
            '--width', str(args.width),
            '--height', str(args.height),
            '--fx', str(args.fx),
            '--fy', str(args.fy),
            '--cx', str(args.cx),
            '--cy', str(args.cy),
        ]
        if not run_step("3DGS RGB渲染", cmd, cwd=str(project_dir)):
            return 1
    
    # ==================== Step 3: 特征提取 ====================
    features_dir = output_dir / 'features_raw'
    if not args.skip_features:
        # 设置离线模式
        env_prefix = 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'
        cmd = [
            'python', str(scripts_dir / 'extract_fused_features.py'),
            '--input_dir', str(rgb_dir),
            '--output_dir', str(features_dir),
        ]
        # 使用shell=True来设置环境变量
        full_cmd = f"{env_prefix} {' '.join(cmd)}"
        print(f"\n{'='*60}")
        print(f"步骤: DINO+SD特征提取")
        print(f"{'='*60}")
        print(f"命令: {full_cmd}")
        start_time = time.time()
        result = subprocess.run(full_cmd, shell=True, cwd=str(project_dir))
        elapsed = time.time() - start_time
        if result.returncode != 0:
            print(f"\n✗ 步骤 '特征提取' 失败 (耗时 {elapsed:.1f}s)")
            return 1
        print(f"\n✓ 步骤 '特征提取' 完成 (耗时 {elapsed:.1f}s)")
    
    # ==================== Step 4: 特征压缩 ====================
    features_compressed_dir = output_dir / 'features_compressed'
    if not args.skip_compress:
        cmd = [
            'python', str(scripts_dir / 'compress_features.py'),
            '--input_dir', str(features_dir),
            '--output_dir', str(features_compressed_dir),
            '--model_path', args.ae_model_path,
        ]
        if not run_step("特征压缩", cmd, cwd=str(project_dir)):
            return 1
    
    # ==================== 完成 ====================
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✓ 流水线完成!")
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── traj_tum.txt (位姿文件)")
    print(f"    ├── rgb/ (渲染的RGB图像)")
    print(f"    ├── features_raw/ (原始融合特征)")
    print(f"    └── features_compressed/ (压缩特征)")
    
    # 统计
    n_poses = sum(1 for l in open(poses_file) if l.strip() and not l.startswith('#'))
    n_rgb = len(list(rgb_dir.glob('*.png'))) if rgb_dir.exists() else 0
    n_feat = len(list(features_compressed_dir.glob('*.npy'))) if features_compressed_dir.exists() else 0
    
    print(f"\n统计:")
    print(f"  位姿数量: {n_poses}")
    print(f"  RGB图像: {n_rgb}")
    print(f"  压缩特征: {n_feat}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
