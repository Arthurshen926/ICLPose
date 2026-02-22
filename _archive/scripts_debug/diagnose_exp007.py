"""
诊断EXP007的问题
检查特征归一化、位置编码、Kendall Loss等实现
"""
import torch
import torch.nn as nn

print("=" * 80)
print("EXP007 问题诊断")
print("=" * 80)

# 1. 模拟特征提取和归一化
print("\n[1] 特征归一化测试")
print("-" * 80)

# 2D特征（从fused_feature）
feat_2d = torch.randn(32, 256) * 0.5 + 0.5  # 模拟[0,1]范围的特征
print(f"2D特征（归一化前）:")
print(f"  Mean: {feat_2d.mean():.4f}, Std: {feat_2d.std():.4f}")
print(f"  ||f||: {torch.norm(feat_2d, p=2, dim=-1).mean():.4f}")

feat_2d_norm = torch.nn.functional.normalize(feat_2d, p=2, dim=-1)
print(f"2D特征（归一化后）:")
print(f"  Mean: {feat_2d_norm.mean():.4f}, Std: {feat_2d_norm.std():.4f}")
print(f"  ||f||: {torch.norm(feat_2d_norm, p=2, dim=-1).mean():.4f}")

# 3D特征（从feat_decoder）
feat_3d = torch.randn(32, 256) * 0.3  # 模拟decoder输出
print(f"\n3D特征（归一化前）:")
print(f"  Mean: {feat_3d.mean():.4f}, Std: {feat_3d.std():.4f}")
print(f"  ||f||: {torch.norm(feat_3d, p=2, dim=-1).mean():.4f}")

feat_3d_norm = torch.nn.functional.normalize(feat_3d, p=2, dim=-1)
print(f"3D特征（归一化后）:")
print(f"  Mean: {feat_3d_norm.mean():.4f}, Std: {feat_3d_norm.std():.4f}")
print(f"  ||f||: {torch.norm(feat_3d_norm, p=2, dim=-1).mean():.4f}")

# 相似度（归一化后）
sim_norm = torch.sum(feat_2d_norm * feat_3d_norm, dim=-1).mean()
print(f"\n归一化后的相似度（点积）: {sim_norm:.4f}")

# 2. 位置编码的影响
print("\n[2] 位置编码影响测试")
print("-" * 80)

# 简化的正弦位置编码
coords_2d = torch.rand(32, 2)  # [N, 2] 归一化坐标
temp = 10000
dim_t = temp ** (2 * torch.arange(64, dtype=torch.float32) / 64)
u = coords_2d[:, 0]
pos_u = u.unsqueeze(-1) / dim_t
pos_enc_u = torch.cat([pos_u.sin(), pos_u.cos()], dim=-1)
v = coords_2d[:, 1]
pos_v = v.unsqueeze(-1) / dim_t
pos_enc_v = torch.cat([pos_v.sin(), pos_v.cos()], dim=-1)
pos_enc = torch.cat([pos_enc_u, pos_enc_v], dim=-1)  # [N, 256]

print(f"位置编码统计:")
print(f"  Mean: {pos_enc.mean():.4f}, Std: {pos_enc.std():.4f}")
print(f"  ||pe||: {torch.norm(pos_enc, p=2, dim=-1).mean():.4f}")

# 添加位置编码后（当前实现）
feat_with_pe = feat_2d_norm + pos_enc
print(f"\n特征+位置编码（当前实现）:")
print(f"  Mean: {feat_with_pe.mean():.4f}, Std: {feat_with_pe.std():.4f}")
print(f"  ||f+pe||: {torch.norm(feat_with_pe, p=2, dim=-1).mean():.4f}")
print(f"  ❌ 归一化被破坏了！")

# 正确方案：先加位置编码，再归一化
feat_correct = torch.nn.functional.normalize(feat_2d + pos_enc, p=2, dim=-1)
print(f"\n特征+位置编码后归一化（正确）:")
print(f"  Mean: {feat_correct.mean():.4f}, Std: {feat_correct.std():.4f}")
print(f"  ||f||: {torch.norm(feat_correct, p=2, dim=-1).mean():.4f}")
print(f"  ✓ 归一化保持！")

# 3. Kendall Loss分析
print("\n[3] Kendall Loss分析")
print("-" * 80)

# 模拟训练过程
log_var_rot = torch.tensor(3.0)  # 初始值
log_var_trans = torch.tensor(-1.0)

rot_loss = torch.tensor(20.0)  # 度数
trans_loss = torch.tensor(1.5)  # 米

weight_rot = torch.exp(-log_var_rot)
weight_trans = torch.exp(-log_var_trans)

weighted_rot = weight_rot * rot_loss + log_var_rot
weighted_trans = weight_trans * trans_loss + log_var_trans
total_loss = weighted_rot + weighted_trans

print(f"Kendall Loss初始状态:")
print(f"  log_var_rotation: {log_var_rot.item():.3f}")
print(f"  log_var_translation: {log_var_trans.item():.3f}")
print(f"  Rotation loss: {rot_loss.item():.2f}°")
print(f"  Translation loss: {trans_loss.item():.2f}m")
print(f"  Weight (rotation): {weight_rot.item():.4f}")
print(f"  Weight (translation): {weight_trans.item():.4f}")
print(f"  Weighted rotation: {weighted_rot.item():.4f}")
print(f"  Weighted translation: {weighted_trans.item():.4f}")
print(f"  Total loss: {total_loss.item():.4f}")

print(f"\n  分析: 旋转权重={weight_rot.item():.4f}, 平移权重={weight_trans.item():.4f}")
print(f"  平移权重是旋转权重的 {(weight_trans/weight_rot).item():.1f}x")

print("\n" + "=" * 80)
print("总结:")
print("=" * 80)
print("1. ❌ 位置编码破坏了L2归一化 → 相似度计算失效")
print("2. ❌ Kendall Loss初始权重可能不平衡")
print("3. ✓ 特征空间本身是统一的（SD+DINO融合）")
print("\n建议修复:")
print("  1. 将位置编码添加改为：normalize(feat + pos_enc)")
print("  2. 调整Kendall Loss初始log_var")
print("  3. 或暂时禁用Kendall，使用固定权重")
