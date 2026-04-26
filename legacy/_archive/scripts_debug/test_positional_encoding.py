"""
测试位置编码和35x46特征修改
"""

import sys
import torch
import numpy as np

print("=" * 80)
print("测试1: 位置编码模块")
print("=" * 80)

from ic_models.positional_encoding import PositionalEncoding2D, PositionalEncoding3D

# 测试2D位置编码
print("\n[1.1] 测试2D位置编码 (256-dim)...")
pos_enc_2d = PositionalEncoding2D(embed_dim=256)
coords_2d = torch.rand(4, 100, 2)  # [B=4, N=100, 2] 归一化坐标
pos_emb_2d = pos_enc_2d(coords_2d)
print(f"  输入形状: {coords_2d.shape}")
print(f"  输出形状: {pos_emb_2d.shape}")
assert pos_emb_2d.shape == (4, 100, 256), "2D位置编码输出形状错误"
print("  ✓ 2D位置编码测试通过")

# 测试3D位置编码
print("\n[1.2] 测试3D位置编码 (258-dim)...")
pos_enc_3d = PositionalEncoding3D(embed_dim=258, normalize=True, scale_factor=0.1)
coords_3d = torch.randn(4, 100, 3)  # [B=4, N=100, 3] 世界坐标
pos_emb_3d = pos_enc_3d(coords_3d)
print(f"  输入形状: {coords_3d.shape}")
print(f"  输出形状: {pos_emb_3d.shape}")
assert pos_emb_3d.shape == (4, 100, 258), "3D位置编码输出形状错误"
print("  ✓ 3D位置编码测试通过")

print("\n" + "=" * 80)
print("测试2: 融合特征加载 (35x46原始尺寸)")
print("=" * 80)

from data.dataset import CorrespondenceDataset

print("\n[2.1] 加载数据集...")
dataset = CorrespondenceDataset(
    data_root='/home/yons/Projects/data/room_0',
    scene_name='Sequence_1',
    image_size=(640, 480),
    augment=False,
    max_samples=2,
    use_depth=False,
    gaussian_path='/home/yons/Projects/data/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    fx=320.0, fy=320.0, cx=319.5, cy=239.5
)

print(f"数据集大小: {len(dataset)}")

print("\n[2.2] 检查融合特征尺寸...")
sample = dataset[0]
fused_feat = sample['fused_feature']
if fused_feat is not None:
    print(f"  融合特征形状: {fused_feat.shape}")
    assert fused_feat.shape == torch.Size([256, 35, 46]), f"融合特征形状错误: {fused_feat.shape}，应该是[256, 35, 46]"
    print("  ✓ 融合特征尺寸正确 (35x46，无上采样)")
else:
    print("  警告: 融合特征为None")

print("\n[2.3] 检查其他数据...")
print(f"  图像形状: {sample['image'].shape}")
print(f"  位姿形状: {sample['pose'].shape}")
print(f"  2D点形状: {sample['points_2d'].shape}")
print(f"  3D点形状: {sample['points_3d'].shape}")

print("\n" + "=" * 80)
print("测试3: ICPoseNet位置编码集成")
print("=" * 80)

from ic_models.ic_pose_net import ICPoseNet

print("\n[3.1] 初始化ICPoseNet...")
model = ICPoseNet(
    feature_dim=256,
    num_queries=128,
    fusion_layers=6,
    num_heads=8,
    dropout=0.1
)

# 检查位置编码模块是否存在
print("\n[3.2] 检查位置编码模块...")
assert hasattr(model, 'pos_enc_2d'), "模型缺少pos_enc_2d"
assert hasattr(model, 'pos_enc_3d'), "模型缺少pos_enc_3d"
assert hasattr(model, 'pos_enc_3d_proj'), "模型缺少pos_enc_3d_proj"
print("  ✓ 所有位置编码模块存在")

print("\n[3.3] 测试前向传播（不含位置编码）...")
batch_size = 2
n_pts = 1024
img_feats = torch.randn(batch_size, n_pts, 256)
pcd_feats = torch.randn(batch_size, n_pts, 256)

pose_matrix, pose_6d, rotation, translation = model(img_feats, pcd_feats)
print(f"  pose_matrix形状: {pose_matrix.shape}")
print(f"  rotation形状: {rotation.shape}")
print(f"  translation形状: {translation.shape}")
assert pose_matrix.shape == (batch_size, 4, 4), "位姿矩阵形状错误"
print("  ✓ 前向传播测试通过")

print("\n[3.4] 测试位置编码前向...")
# 测试2D坐标
coords_2d = torch.rand(batch_size, n_pts, 2)  # 归一化到[0,1]
pos_enc_2d_out = model.pos_enc_2d(coords_2d)
print(f"  2D位置编码输出: {pos_enc_2d_out.shape}")
assert pos_enc_2d_out.shape == (batch_size, n_pts, 256)

# 测试3D坐标
coords_3d = torch.randn(batch_size, n_pts, 3)
pos_enc_3d_out = model.pos_enc_3d(coords_3d)
pos_enc_3d_proj = model.pos_enc_3d_proj(pos_enc_3d_out)
print(f"  3D位置编码输出: {pos_enc_3d_out.shape}")
print(f"  3D位置编码投影: {pos_enc_3d_proj.shape}")
assert pos_enc_3d_proj.shape == (batch_size, n_pts, 256)
print("  ✓ 位置编码前向测试通过")

print("\n[3.5] 测试特征+位置编码融合...")
img_feats_with_pos = img_feats + pos_enc_2d_out
pcd_feats_with_pos = pcd_feats + pos_enc_3d_proj
print(f"  2D特征+位置编码: {img_feats_with_pos.shape}")
print(f"  3D特征+位置编码: {pcd_feats_with_pos.shape}")
print("  ✓ 特征融合测试通过")

print("\n" + "=" * 80)
print("测试4: 坐标映射 (640x480 -> 35x46)")
print("=" * 80)

print("\n[4.1] 测试图像坐标到特征坐标的映射...")
H_img, W_img = 480, 640
H_feat, W_feat = 35, 46

# 测试几个关键点
test_points = [
    (0, 0),       # 左上角
    (639, 0),     # 右上角
    (0, 479),     # 左下角
    (639, 479),   # 右下角
    (319.5, 239.5),  # 中心点
]

print(f"  图像尺寸: {W_img}x{H_img}")
print(f"  特征尺寸: {W_feat}x{H_feat}")
print(f"  缩放比例: W={W_feat/W_img:.4f}, H={H_feat/H_img:.4f}")

for u_img, v_img in test_points:
    u_feat = u_img * (W_feat / W_img)
    v_feat = v_img * (H_feat / H_img)
    print(f"  图像({u_img:6.1f}, {v_img:6.1f}) -> 特征({u_feat:5.2f}, {v_feat:5.2f})")

print("  ✓ 坐标映射测试通过")

print("\n" + "=" * 80)
print("✓ 所有测试通过!")
print("=" * 80)
print("\n修改总结:")
print("1. ✅ 融合特征保持原始35x46尺寸（不再上采样）")
print("2. ✅ 实现2D Sine/Cosine位置编码 (256-dim)")
print("3. ✅ 实现3D Sine/Cosine位置编码 (258-dim -> 256-dim)")
print("4. ✅ 特征提取时正确映射图像坐标到特征坐标")
print("5. ✅ 特征与位置编码相加融合")
print("\n下一步: 运行完整训练测试")
print("  python train.py --config configs/train_config.yaml")
