#!/usr/bin/env python3
"""
快速验证脚本
检查ICL-I2PReg集成模块是否正确安装
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("1. 测试模块导入...")
    print("=" * 60)
    
    try:
        from implicit_correspondence.modules import (
            TransformerLayer, CrossModalFusionModule, PoseRegressor
        )
        print("✅ modules 导入成功")
    except Exception as e:
        print(f"❌ modules 导入失败: {e}")
        return False
    
    try:
        from implicit_correspondence.ic_models import ICPoseNet, ICPoseNetSimple
        print("✅ models 导入成功")
    except Exception as e:
        print(f"❌ models 导入失败: {e}")
        return False
    
    try:
        from implicit_correspondence.data import CorrespondenceDataset, collate_fn
        print("✅ data 导入成功")
    except Exception as e:
        print(f"❌ data 导入失败: {e}")
        return False
    
    try:
        from implicit_correspondence.losses import PoseLoss
        print("✅ losses 导入成功")
    except Exception as e:
        print(f"❌ losses 导入失败: {e}")
        return False
    
    return True


def test_network():
    """测试网络前向传播"""
    print("\n" + "=" * 60)
    print("2. 测试网络前向传播...")
    print("=" * 60)
    
    try:
        import torch
        from implicit_correspondence.ic_models import ICPoseNetSimple
        
        # 创建模型
        model = ICPoseNetSimple(
            feature_dim=256,
            hidden_dim=512,
            num_layers=2,
            num_heads=8,
            dropout=0.1
        )
        
        print(f"✅ 模型创建成功")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 2
        num_img_feats = 100
        num_pcd_feats = 200
        feature_dim = 256
        
        img_feats = torch.randn(batch_size, num_img_feats, feature_dim)
        pcd_feats = torch.randn(batch_size, num_pcd_feats, feature_dim)
        
        with torch.no_grad():
            pose_matrix, rotation, translation = model(img_feats, pcd_feats)
        
        print(f"✅ 前向传播成功")
        print(f"   pose_matrix shape: {pose_matrix.shape}")
        print(f"   rotation shape: {rotation.shape}")
        print(f"   translation shape: {translation.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("3. 测试损失函数...")
    print("=" * 60)
    
    try:
        import torch
        from implicit_correspondence.losses import PoseLoss
        
        # 创建损失函数
        loss_fn = PoseLoss(
            rotation_loss_type='geodesic',
            translation_loss_type='l2',
            rotation_weight=1.0,
            translation_weight=1.0
        )
        
        print("✅ 损失函数创建成功")
        
        # 测试损失计算
        batch_size = 2
        # 创建随机位姿矩阵
        pred_pose = torch.eye(4).unsqueeze(0).expand(batch_size, 4, 4).clone()
        pred_pose[:, :3, :3] = torch.randn(batch_size, 3, 3)
        pred_pose[:, :3, 3] = torch.randn(batch_size, 3)
        
        gt_pose = torch.eye(4).unsqueeze(0).expand(batch_size, 4, 4).clone()
        gt_pose[:, :3, :3] = torch.randn(batch_size, 3, 3)
        gt_pose[:, :3, 3] = torch.randn(batch_size, 3)
        
        loss = loss_fn(pred_pose, gt_pose)
        
        print(f"✅ 损失计算成功")
        print(f"   total_loss: {loss['loss'].item():.4f}")
        if 'rotation_loss' in loss:
            print(f"   rotation_loss: {loss['rotation_loss'].item():.4f}")
        if 'translation_loss' in loss:
            print(f"   translation_loss: {loss['translation_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_files():
    """检查文件结构"""
    print("\n" + "=" * 60)
    print("4. 检查文件结构...")
    print("=" * 60)
    
    ic_root = Path(__file__).parent
    
    required_files = [
        "modules/transformer.py",
        "modules/fusion_module.py",
        "modules/pose_regressor.py",
        "modules/__init__.py",
        "ic_models/ic_pose_net.py",
        "ic_models/__init__.py",
        "data/dataset.py",
        "data/__init__.py",
        "losses/pose_loss.py",
        "losses/__init__.py",
        "train.py",
        "configs/train_config.yaml",
        "README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = ic_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("ICL-I2PReg 集成模块验证")
    print("🚀" * 30 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("网络测试", test_network()))
    results.append(("损失函数", test_loss()))
    results.append(("文件结构", check_files()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！模块已正确安装。")
        print("\n下一步:")
        print("  1. 修改 configs/train_config.yaml 中的路径")
        print("  2. 运行 python check_environment.py 检查环境")
        print("  3. 运行 ./run_train.sh 开始训练")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查安装。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
