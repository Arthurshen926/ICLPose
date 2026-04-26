"""
使用示例：训练隐式对应关系模型
演示如何使用CorrespondenceDataset和PoseLoss
"""

import torch
from torch.utils.data import DataLoader
from implicit_correspondence.data import CorrespondenceDataset, collate_fn
from implicit_correspondence.losses import PoseLoss


def main():
    """主函数：演示如何使用数据集和损失函数"""
    
    print("=" * 60)
    print("隐式对应关系模型 - 使用示例")
    print("=" * 60)
    
    # ==========================
    # 1. 创建数据集
    # ==========================
    print("\n[步骤 1] 创建数据集...")
    
    data_root = "/home/yons/Projects/data/room_0"
    scene_name = "Sequence_1"
    
    train_dataset = CorrespondenceDataset(
        data_root=data_root,
        scene_name=scene_name,
        image_size=(640, 480),
        augment=True,           # 训练时开启数据增强
        max_samples=100,        # 使用前100张图像
        use_depth=False,        # 不使用深度图
        fx=320.0, fy=320.0,
        cx=319.5, cy=239.5,
    )
    
    val_dataset = CorrespondenceDataset(
        data_root=data_root,
        scene_name=scene_name,
        image_size=(640, 480),
        augment=False,          # 验证时不使用数据增强
        max_samples=20,         # 使用20张图像验证
        use_depth=False,
        fx=320.0, fy=320.0,
        cx=319.5, cy=239.5,
    )
    
    # ==========================
    # 2. 创建DataLoader
    # ==========================
    print("\n[步骤 2] 创建DataLoader...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    
    # ==========================
    # 3. 创建损失函数
    # ==========================
    print("\n[步骤 3] 创建损失函数...")
    
    # 方案1: 测地距离 + L2平移
    pose_loss_geodesic = PoseLoss(
        rotation_loss_type='geodesic',
        translation_loss_type='l2',
        rotation_weight=1.0,
        translation_weight=1.0,
        reduction='mean',
    )
    
    # 方案2: 四元数 + L2平移（权重调整）
    pose_loss_quaternion = PoseLoss(
        rotation_loss_type='quaternion',
        translation_loss_type='l2',
        rotation_weight=10.0,      # 旋转损失权重更大
        translation_weight=1.0,
        reduction='mean',
    )
    
    # ==========================
    # 4. 模拟训练循环
    # ==========================
    print("\n[步骤 4] 模拟训练循环...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将损失函数移到设备上
    pose_loss_geodesic = pose_loss_geodesic.to(device)
    
    # 模拟一个epoch
    print("\n开始训练（模拟）...")
    num_batches = min(3, len(train_loader))  # 只处理前3个batch
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # 移到设备
        images = batch['image'].to(device)
        poses_gt = batch['pose'].to(device)
        points_2d = batch['points_2d'].to(device)
        points_3d = batch['points_3d'].to(device)
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}:")
        print(f"  图像形状: {images.shape}")
        print(f"  位姿形状: {poses_gt.shape}")
        print(f"  2D点数量: {points_2d.shape[0]}")
        print(f"  3D点数量: {points_3d.shape[0]}")
        
        # 模拟网络预测（这里用真值加噪声代替）
        noise_R = torch.randn_like(poses_gt[:, :3, :3]) * 0.1
        noise_t = torch.randn_like(poses_gt[:, :3, 3]) * 0.1
        
        poses_pred = poses_gt.clone()
        poses_pred[:, :3, :3] = poses_gt[:, :3, :3] + noise_R
        poses_pred[:, :3, 3] = poses_gt[:, :3, 3] + noise_t
        
        # 计算损失
        losses = pose_loss_geodesic(
            poses_pred, 
            poses_gt, 
            return_components=True
        )
        
        print(f"  总损失: {losses['loss'].item():.4f}")
        print(f"  旋转损失: {losses['rotation_loss'].item():.4f} (度)")
        print(f"  平移损失: {losses['translation_loss'].item():.4f} (米)")
        
        # 在真实训练中，这里会进行反向传播和优化
        # loss.backward()
        # optimizer.step()
    
    # ==========================
    # 5. 比较不同损失函数
    # ==========================
    print("\n[步骤 5] 比较不同损失函数...")
    
    # 获取一个batch
    sample_batch = next(iter(val_loader))
    poses_gt = sample_batch['pose'].to(device)
    
    # 生成预测（添加不同程度的噪声）
    poses_pred_small = poses_gt.clone()
    poses_pred_small[:, :3, :3] += torch.randn_like(poses_gt[:, :3, :3]) * 0.05
    poses_pred_small[:, :3, 3] += torch.randn_like(poses_gt[:, :3, 3]) * 0.05
    
    poses_pred_large = poses_gt.clone()
    poses_pred_large[:, :3, :3] += torch.randn_like(poses_gt[:, :3, :3]) * 0.3
    poses_pred_large[:, :3, 3] += torch.randn_like(poses_gt[:, :3, 3]) * 0.3
    
    loss_types = [
        ('geodesic', 'l2'),
        ('l2', 'l2'),
        ('cosine', 'l1'),
        ('quaternion', 'l2'),
    ]
    
    print("\n小误差情况:")
    for rot_type, trans_type in loss_types:
        loss_fn = PoseLoss(rot_type, trans_type, 1.0, 1.0).to(device)
        losses = loss_fn(poses_pred_small, poses_gt, return_components=True)
        print(f"  {rot_type:12} + {trans_type:10} = {losses['loss'].item():.4f}")
    
    print("\n大误差情况:")
    for rot_type, trans_type in loss_types:
        loss_fn = PoseLoss(rot_type, trans_type, 1.0, 1.0).to(device)
        losses = loss_fn(poses_pred_large, poses_gt, return_components=True)
        print(f"  {rot_type:12} + {trans_type:10} = {losses['loss'].item():.4f}")
    
    # ==========================
    # 6. 总结
    # ==========================
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("\n使用建议:")
    print("1. 对于位姿估计任务，推荐使用 geodesic + l2")
    print("2. 如果训练不稳定，可以尝试 quaternion + smooth_l1")
    print("3. 根据任务需求调整 rotation_weight 和 translation_weight")
    print("4. 数据增强可以提高模型的鲁棒性")
    print("5. 可以通过 use_depth=True 来使用真实深度信息")
    print("\n完整的训练代码需要:")
    print("- 定义编码器/解码器网络")
    print("- 实现特征提取和匹配")
    print("- 添加优化器和学习率调度")
    print("- 实现验证和保存逻辑")


if __name__ == '__main__':
    main()
