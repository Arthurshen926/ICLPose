"""
诊断训练问题的脚本
分析loss、位姿误差、特征质量等
"""

import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader


def analyze_initial_pose_difficulty():
    """分析初始位姿的难度是否合理"""
    print("=" * 80)
    print("1. 分析初始位姿难度")
    print("=" * 80)
    
    config_path = Path(__file__).parent / "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['dataset']
    
    # 创建数据集
    dataset = CorrespondenceDataset(
        data_root=data_cfg['data_root'],
        scene_name=data_cfg['train_scene'],
        image_size=tuple(data_cfg['image_size']),
        augment=True,
        max_samples=50,
        use_depth=False,
        gaussian_path=data_cfg.get('gaussian_path'),
        fx=data_cfg['fx'],
        fy=data_cfg['fy'],
        cx=data_cfg['cx'],
        cy=data_cfg['cy'],
        sample_step=1,
        use_initial_pose=True,
        pose_noise_rot_deg=data_cfg.get('pose_noise_rot_deg', 5.0),
        pose_noise_trans_m=data_cfg.get('pose_noise_trans_m', 0.1),
    )
    
    print(f"\n配置的噪声水平:")
    print(f"  旋转噪声: {data_cfg.get('pose_noise_rot_deg', 5.0)}°")
    print(f"  平移噪声: {data_cfg.get('pose_noise_trans_m', 0.1)}m")
    
    # 统计实际噪声
    rot_errors = []
    trans_errors = []
    
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        pose_gt = sample['pose'].numpy()
        pose_init = sample['initial_pose'].numpy()
        
        R_gt = pose_gt[:3, :3]
        R_init = pose_init[:3, :3]
        t_gt = pose_gt[:3, 3]
        t_init = pose_init[:3, 3]
        
        # 旋转误差
        R_diff = R_init.T @ R_gt
        trace = np.trace(R_diff)
        angle_diff_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        angle_diff_deg = np.degrees(angle_diff_rad)
        
        # 平移误差
        trans_diff = np.linalg.norm(t_init - t_gt)
        
        rot_errors.append(angle_diff_deg)
        trans_errors.append(trans_diff)
    
    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)
    
    print(f"\n实际噪声统计 (n={len(rot_errors)}):")
    print(f"  旋转误差: 均值={rot_errors.mean():.2f}°, 标准差={rot_errors.std():.2f}°, "
          f"最大={rot_errors.max():.2f}°")
    print(f"  平移误差: 均值={trans_errors.mean():.3f}m, 标准差={trans_errors.std():.3f}m, "
          f"最大={trans_errors.max():.3f}m")
    
    # 评估难度
    print(f"\n难度评估:")
    if rot_errors.mean() > 5.0:
        print(f"  ⚠️  旋转噪声较大 ({rot_errors.mean():.1f}°)，可能导致视锥裁剪不准确")
    else:
        print(f"  ✓ 旋转噪声合理 ({rot_errors.mean():.1f}°)")
    
    if trans_errors.mean() > 0.2:
        print(f"  ⚠️  平移噪声较大 ({trans_errors.mean():.2f}m)，室内场景可能过大")
    else:
        print(f"  ✓ 平移噪声合理 ({trans_errors.mean():.2f}m)")


def analyze_training_log():
    """分析训练日志"""
    print("\n" + "=" * 80)
    print("2. 分析训练日志")
    print("=" * 80)
    
    log_path = Path("output/exp010/training.log")
    if not log_path.exists():
        print("找不到训练日志")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # 提取训练loss
    train_losses = []
    val_losses = []
    val_rot_errors = []
    val_trans_errors = []
    
    for line in lines:
        if "[训练] Loss:" in line:
            try:
                loss = float(line.split("Loss:")[1].split("|")[0].strip())
                train_losses.append(loss)
            except:
                pass
        elif "[验证] Loss:" in line:
            try:
                loss = float(line.split("Loss:")[1].split("|")[0].strip())
                val_losses.append(loss)
            except:
                pass
        elif "角度误差:" in line:
            try:
                # 提取类似 "7.84°" 的值
                error_str = line.split("角度误差:")[1].split("°")[0].strip()
                error = float(error_str)
                val_rot_errors.append(error)
            except:
                pass
        elif "平移误差:" in line:
            try:
                # 提取类似 "1.5992m" 的值
                error_str = line.split("平移误差:")[1].split("m")[0].strip()
                error = float(error_str)
                val_trans_errors.append(error)
            except:
                pass
    
    print(f"\n训练统计 (共{len(train_losses)}个epoch):")
    if len(train_losses) > 0:
        train_losses = np.array(train_losses)
        print(f"  初始Loss: {train_losses[0]:.2f}")
        print(f"  最终Loss: {train_losses[-1]:.2f}")
        print(f"  最小Loss: {train_losses.min():.2f}")
        print(f"  最近100 epoch平均: {train_losses[-100:].mean():.2f}")
        print(f"  最近100 epoch标准差: {train_losses[-100:].std():.2f}")
        
        # 判断是否收敛
        if train_losses[-100:].std() > 1.0:
            print(f"  ⚠️  Loss波动大，训练不稳定")
        
        if train_losses[-1] > train_losses.min() * 1.5:
            print(f"  ⚠️  最终Loss远高于最小Loss，可能未收敛")
    
    print(f"\n验证统计 (共{len(val_losses)}次):")
    if len(val_losses) > 0:
        val_losses = np.array(val_losses)
        print(f"  初始Loss: {val_losses[0]:.2f}")
        print(f"  最终Loss: {val_losses[-1]:.2f}")
        print(f"  最小Loss: {val_losses.min():.2f}")
        
        if val_losses[-1] > 50:
            print(f"  ❌ 验证Loss异常高 ({val_losses[-1]:.1f})，说明验证数据处理可能有问题")
    
    if len(val_rot_errors) > 0:
        val_rot_errors = np.array(val_rot_errors)
        val_trans_errors = np.array(val_trans_errors)
        print(f"\n验证精度:")
        print(f"  旋转误差: 最好={val_rot_errors.min():.2f}°, 最差={val_rot_errors.max():.2f}°, "
              f"最终={val_rot_errors[-1]:.2f}°")
        print(f"  平移误差: 最好={val_trans_errors.min():.3f}m, 最差={val_trans_errors.max():.3f}m, "
              f"最终={val_trans_errors[-1]:.3f}m")


def check_visualization():
    """检查可视化结果"""
    print("\n" + "=" * 80)
    print("3. 检查可视化结果")
    print("=" * 80)
    
    vis_dir = Path("output/exp010/visualizations")
    if not vis_dir.exists():
        print("找不到可视化目录")
        return
    
    # 找最新的epoch
    epoch_dirs = sorted(vis_dir.glob("epoch_*"))
    if len(epoch_dirs) == 0:
        print("没有可视化结果")
        return
    
    print(f"\n找到 {len(epoch_dirs)} 个epoch的可视化")
    
    # 检查最早和最晚的
    for epoch_dir in [epoch_dirs[0], epoch_dirs[-1]]:
        print(f"\n检查 {epoch_dir.name}:")
        
        heatmap = epoch_dir / "confidence_heatmap_" + epoch_dir.name.replace("epoch_", "epoch") + ".png"
        if heatmap.exists():
            img = Image.open(heatmap)
            img_array = np.array(img)
            print(f"  置信度热力图: {img_array.shape}, 均值={img_array.mean():.1f}")
            
            # 简单检查是否全黑或全白
            if img_array.mean() < 10:
                print(f"    ⚠️  图像几乎全黑，可能置信度过低")
            elif img_array.mean() > 245:
                print(f"    ⚠️  图像几乎全白")


def analyze_relative_pose_logic():
    """检查相对位姿逻辑"""
    print("\n" + "=" * 80)
    print("4. 检查相对位姿逻辑")
    print("=" * 80)
    
    config_path = Path("configs/train_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    use_relative = config['loss'].get('use_relative_pose', False)
    print(f"\nuse_relative_pose: {use_relative}")
    
    if use_relative:
        print("  ✓ 启用相对位姿模式")
        print("  预期行为:")
        print("    - 数据集应生成 initial_pose（带噪声）")
        print("    - 训练目标: 从 initial_pose 修正到 GT")
        print("    - 评估: 恢复绝对位姿后计算误差")
    else:
        print("  使用绝对位姿模式")


def main():
    print("\n" + "=" * 80)
    print("训练问题诊断")
    print("=" * 80)
    
    analyze_initial_pose_difficulty()
    analyze_training_log()
    check_visualization()
    analyze_relative_pose_logic()
    
    print("\n" + "=" * 80)
    print("诊断建议")
    print("=" * 80)
    
    print("\n基于以上分析，建议:")
    print("1. ✅ 已将初始位姿噪声降低到合理水平 (3°/0.05m)")
    print("2. ✅ 已降低batch size到32，提高梯度稳定性")
    print("3. ✅ 已降低模型复杂度 (queries: 128→64, layers: 8→4)")
    print("4. ✅ 已提高学习率到1e-4，加速收敛")
    print("5. ✅ 已调整Kendall loss初始权重")
    print("\n下一步:")
    print("  - 清空旧的输出目录或使用新的exp编号")
    print("  - 重新开始训练，观察前10个epoch的loss下降趋势")
    print("  - 如果loss仍然不下降，考虑:")
    print("    a) 禁用use_relative_pose，先用绝对位姿训练")
    print("    b) 检查数据集是否正确加载")
    print("    c) 检查SplatLoc特征解码器是否正常工作")


if __name__ == '__main__':
    main()
