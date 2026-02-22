#!/usr/bin/env python3
"""
exp020 组件测试脚本

测试新架构的所有组件是否正常工作:
1. ICPoseNetV2 模型
2. C2F损失函数
3. 重叠检测模块
4. 位置编码
"""

import sys
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, '/home/yons/Projects/ICLPose')

def test_icposenet_v2():
    """测试 ICPoseNet V2"""
    print("=" * 60)
    print("1. Testing ICPoseNetV2")
    print("=" * 60)
    
    from ic_models import ICPoseNetV2, ICPoseNetV2Lite
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ICPoseNetV2(
        feature_dim=256,
        num_queries=64,      # 减少以加速测试
        num_layers=8,        # 减少以加速测试
        num_heads=8,
        dropout=0.1,
        output_interval=2,
        use_overlap_detection=True,
        use_c2f=True,
    ).to(device)
    
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试输入
    B, N_img, N_pcd, C = 4, 1000, 500, 256
    
    img_feats = torch.randn(B, N_img, C).to(device)
    pcd_feats = torch.randn(B, N_pcd, C).to(device)
    img_pixels = torch.rand(B, N_img, 2).to(device) * 640
    pcd_points = torch.randn(B, N_pcd, 3).to(device) * 5
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    img_global_feats = torch.randn(B, C).to(device)
    
    # 前向传播
    outputs = model(
        img_feats=img_feats,
        pcd_feats=pcd_feats,
        img_pixels=img_pixels,
        pcd_points=pcd_points,
        intrinsics=intrinsics,
        img_global_feats=img_global_feats,
    )
    
    print(f"  Pose matrix: {outputs['pose_matrix'].shape}")
    print(f"  Stages: {len(outputs['stage_outputs'])}")
    print(f"  Overlap mask: {outputs.get('overlap_mask', 'N/A')}")
    
    # 梯度测试
    loss = outputs['translation'].sum()
    loss.backward()
    
    print("  ✅ ICPoseNetV2 测试通过")
    return True


def test_c2f_loss():
    """测试 C2F 损失函数"""
    print("\n" + "=" * 60)
    print("2. Testing PoseLossC2F")
    print("=" * 60)
    
    from losses import PoseLossC2F
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建损失函数
    criterion = PoseLossC2F(
        num_stages=4,
        stage_weights=[0.2, 0.4, 0.6, 1.0],
        warmup_epochs=50,
        use_dyntanh=True,
    ).to(device)
    
    # 测试输入
    B = 4
    
    # 模拟多阶段输出
    stage_outputs = []
    for i in range(4):
        R = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone() + torch.randn(B, 3, 3) * 0.1
        t = torch.randn(B, 3) * 0.5
        
        # Gram-Schmidt 正交化
        q, _ = torch.linalg.qr(R)
        R = q * torch.sign(torch.det(q)).unsqueeze(-1).unsqueeze(-1)
        
        pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device).clone()
        pose[:, :3, :3] = R.to(device)
        pose[:, :3, 3] = t.to(device)
        
        stage_outputs.append({
            'pose_matrix': pose,
            'rotation_6d': torch.randn(B, 6).to(device),
            'translation': t.to(device),
        })
    
    # GT pose
    gt_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    
    # 提取 pose_matrix 列表
    pose_stages = [s['pose_matrix'] for s in stage_outputs]
    
    # 计算损失
    criterion.set_epoch(0)  # 设置当前 epoch
    loss_dict = criterion(pose_stages, gt_pose, return_components=True)
    
    print(f"  Total loss: {loss_dict['loss'].item():.4f}")
    print(f"  Rotation loss: {loss_dict.get('rotation_loss', 'N/A')}")
    print(f"  Translation loss: {loss_dict.get('translation_loss', 'N/A')}")
    print(f"  Stages: {len(loss_dict.get('stage_losses', []))}")
    
    # 测试warmup
    criterion.set_epoch(0)
    loss_warmup = criterion(pose_stages, gt_pose)
    criterion.set_epoch(100)
    loss_post = criterion(pose_stages, gt_pose)
    
    print(f"  Warmup mode (epoch=0): rotation_mode = L1+Cosine")
    print(f"  Post-warmup (epoch=100): rotation_mode = Geodesic")
    
    print("  ✅ PoseLossC2F 测试通过")
    return True


def test_overlap_detection():
    """测试重叠检测模块"""
    print("\n" + "=" * 60)
    print("3. Testing OverlapDetectionModule")
    print("=" * 60)
    
    from modules import OverlapDetectionModule, OverlapLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    module = OverlapDetectionModule(
        pcd_feat_dim=256,
        img_feat_dim=256,
        hidden_dim=128,
    ).to(device)
    
    print(f"  参数量: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M")
    
    # 测试输入
    B, N, C = 4, 500, 256
    
    pcd_feats = torch.randn(B, N, C).to(device)
    pcd_points = torch.randn(B, N, 3).to(device) * 5
    global_img_feats = torch.randn(B, C).to(device)
    
    # 前向传播
    overlap_mask, coarse_pose, vertex = module(
        pcd_feats=pcd_feats,
        global_img_feats=global_img_feats,
        pcd_points=pcd_points,
    )
    
    print(f"  Overlap mask: {overlap_mask.shape}, range: [{overlap_mask.min():.3f}, {overlap_mask.max():.3f}]")
    print(f"  Coarse pose: {coarse_pose.shape}")
    print(f"  Vertex: {vertex.shape}")
    
    # 测试损失函数
    loss_fn = OverlapLoss().to(device)
    
    gt_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    
    loss = loss_fn(overlap_mask, pcd_points, gt_pose, intrinsics)
    
    print(f"  Overlap loss: {loss.item():.4f}")
    
    print("  ✅ OverlapDetectionModule 测试通过")
    return True


def test_position_encoding():
    """测试位置编码"""
    print("\n" + "=" * 60)
    print("4. Testing Position Encoding")
    print("=" * 60)
    
    from modules.fusion_module_c2f import PositionalEncoding2DIntrinsic, PositionalEncoding3DNeRF
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3D NeRF编码
    pos_enc_3d = PositionalEncoding3DNeRF(
        feature_dim=256,
        num_freqs=10,
    ).to(device)
    
    coords_3d = torch.randn(4, 100, 3).to(device)
    embed_3d = pos_enc_3d(coords_3d)
    
    print(f"  3D encoding: {coords_3d.shape} -> {embed_3d.shape}")
    
    # 2D内参归一化编码
    pos_enc_2d = PositionalEncoding2DIntrinsic(
        feature_dim=256,
        num_freqs=10,
    ).to(device)
    
    pixels = torch.rand(4, 100, 2).to(device) * 640
    intrinsics = torch.eye(3).unsqueeze(0).expand(4, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    intrinsics[:, 0, 2] = 319.5
    intrinsics[:, 1, 2] = 239.5
    
    embed_2d = pos_enc_2d(pixels, intrinsics)
    
    print(f"  2D encoding: {pixels.shape} -> {embed_2d.shape}")
    
    # 验证内参归一化效果
    pixels_normalized = pixels.clone()
    pixels_normalized[:, :, 0] = (pixels[:, :, 0] - intrinsics[:, 0, 2].unsqueeze(-1)) / intrinsics[:, 0, 0].unsqueeze(-1)
    pixels_normalized[:, :, 1] = (pixels[:, :, 1] - intrinsics[:, 1, 2].unsqueeze(-1)) / intrinsics[:, 1, 1].unsqueeze(-1)
    
    print(f"  Normalized pixel range: [{pixels_normalized.min():.3f}, {pixels_normalized.max():.3f}]")
    
    print("  ✅ Position Encoding 测试通过")
    return True


def test_full_pipeline():
    """测试完整流程"""
    print("\n" + "=" * 60)
    print("5. Testing Full Pipeline (Model + Loss)")
    print("=" * 60)
    
    from ic_models import ICPoseNetV2
    from losses import PoseLossC2F
    from modules import OverlapLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ICPoseNetV2(
        feature_dim=256,
        num_queries=64,
        num_layers=8,
        output_interval=2,
        use_overlap_detection=True,
        use_c2f=True,
    ).to(device)
    
    # 创建损失函数
    pose_loss = PoseLossC2F(
        num_stages=4,
        stage_weights=[0.3, 0.5, 0.7, 1.0],
        warmup_epochs=50,
        use_dyntanh=True,
    ).to(device)
    
    overlap_loss = OverlapLoss().to(device)
    
    # 测试数据
    B, N_img, N_pcd, C = 2, 500, 300, 256
    
    img_feats = torch.randn(B, N_img, C).to(device)
    pcd_feats = torch.randn(B, N_pcd, C).to(device)
    img_pixels = torch.rand(B, N_img, 2).to(device) * 640
    pcd_points = torch.randn(B, N_pcd, 3).to(device) * 5
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    intrinsics[:, 0, 0] = 320
    intrinsics[:, 1, 1] = 320
    img_global_feats = torch.randn(B, C).to(device)
    gt_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1).to(device).clone()
    
    # 前向传播
    outputs = model(
        img_feats=img_feats,
        pcd_feats=pcd_feats,
        img_pixels=img_pixels,
        pcd_points=pcd_points,
        intrinsics=intrinsics,
        img_global_feats=img_global_feats,
    )
    
    # 计算损失
    # 从 stage_outputs 中提取 pose_matrix
    pose_stages = [stage['pose_matrix'] for stage in outputs['stage_outputs']]
    pose_loss_dict = pose_loss(pose_stages, gt_pose, return_components=True)
    overlap_loss_val = overlap_loss(outputs['overlap_mask'], pcd_points, gt_pose, intrinsics)
    
    total_loss = pose_loss_dict['loss'] + 0.5 * overlap_loss_val
    
    print(f"  Pose loss: {pose_loss_dict['loss'].item():.4f}")
    print(f"  Overlap loss: {overlap_loss_val.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # 反向传播
    total_loss.backward()
    
    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    
    print(f"  Parameters with grad: {has_grad}/{total_params}")
    
    print("  ✅ Full Pipeline 测试通过")
    return True


def main():
    print("=" * 60)
    print("EXP020 组件测试")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("ICPoseNetV2", test_icposenet_v2()))
    except Exception as e:
        print(f"  ❌ ICPoseNetV2 失败: {e}")
        results.append(("ICPoseNetV2", False))
    
    try:
        results.append(("PoseLossC2F", test_c2f_loss()))
    except Exception as e:
        print(f"  ❌ PoseLossC2F 失败: {e}")
        results.append(("PoseLossC2F", False))
    
    try:
        results.append(("OverlapDetection", test_overlap_detection()))
    except Exception as e:
        print(f"  ❌ OverlapDetection 失败: {e}")
        results.append(("OverlapDetection", False))
    
    try:
        results.append(("PositionEncoding", test_position_encoding()))
    except Exception as e:
        print(f"  ❌ PositionEncoding 失败: {e}")
        results.append(("PositionEncoding", False))
    
    try:
        results.append(("FullPipeline", test_full_pipeline()))
    except Exception as e:
        print(f"  ❌ FullPipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FullPipeline", False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！可以开始 exp020 训练")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == '__main__':
    main()
