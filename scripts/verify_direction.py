#!/usr/bin/env python3
"""
精确验证整个管道的方向一致性:
1. compute_gt_flow 的 flow 方向
2. FlowToPose 从 flow 恢复的 xi 方向
3. se3_exp(xi) @ pose 的更新方向
4. 最终 Δ 是正还是负

如果方向对齐正确, 那么:
  model输出的flow越接近GT flow → FlowToPose得到的xi越准确 → pose correction越好 → Δ > 0
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from modules.lie_algebra import se3_exp, compute_gt_flow
from modules.flow_to_pose import flow_to_pose_weighted_lstsq
from losses.sequence_loss import rotation_geodesic_loss

torch.set_printoptions(precision=6)

intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
H, W = 35, 46

# === Step 1: 创建已知场景 ===
# GT pose = identity
pose_gt = torch.eye(4).unsqueeze(0)

# 扰动: 绕 Y 轴旋转 3° + X 平移 0.02m
# xi_perturbation 使得 perturbed = exp(xi_pert) @ gt
xi_pert = torch.tensor([[0.02, 0.0, 0.0, 0.0, 0.0524, 0.0]])  # 3° ≈ 0.0524 rad
delta_T = se3_exp(xi_pert)
perturbed_pose = delta_T @ pose_gt  # initial_pose

# 验证初始误差
init_rot_err = rotation_geodesic_loss(
    perturbed_pose[:, :3, :3], pose_gt[:, :3, :3]
).item() * 180 / np.pi
print(f"=== 初始误差: {init_rot_err:.2f}° ===\n")

# Depth: Z=2m uniform
depth = torch.ones(1, H, W) * 2.0

# === Step 2: compute_gt_flow 方向测试 ===
# SequenceLoss 中: compute_gt_flow(depth_gt, pose_gt, P_before, intrinsics)
# P_before = poses[k] = initial_pose (修正前)
flow_data = compute_gt_flow(depth, pose_gt, perturbed_pose, intrinsics)
flow_gt = flow_data['flow']  # (1, 2, H, W)

print(f"Step 2: compute_gt_flow(depth, pose_gt, perturbed_pose)")
print(f"  GT flow mean: u={flow_gt[0,0].mean():.4f}, v={flow_gt[0,1].mean():.4f}")
print(f"  GT flow意义: 在GT图像上, 像素应该移动到哪里才能匹配perturbed图像的对应点\n")

# === Step 3: FlowToPose 方向测试 ===
log_conf = torch.zeros(1, 1, H, W)

# FlowToPose(negate=True) 输出: -solve(J, flow)
# 即: 如果 flow表示 GT→perturbed 的运动, 取反得到 perturbed→GT 的修正
xi_recovered = flow_to_pose_weighted_lstsq(flow_gt, log_conf, depth, intrinsics)
xi_correction = -xi_recovered  # negate_output=True 的效果

print(f"Step 3: FlowToPose")
print(f"  xi_recovered (raw):  {xi_recovered[0].tolist()}")
print(f"  xi_correction (-xi): {xi_correction[0].tolist()}")

# === Step 4: 应用修正 ===
# model中: current_pose = se3_exp(xi) @ current_pose
# xi 是 DualHead 输出, 当 FlowToPose(negate=True) 时, xi = -solve(J, flow_pred)  
corrected_pose = se3_exp(xi_correction) @ perturbed_pose

final_rot_err = rotation_geodesic_loss(
    corrected_pose[:, :3, :3], pose_gt[:, :3, :3]
).item() * 180 / np.pi

print(f"\nStep 4: 应用修正")
print(f"  corrected = exp(xi_correction) @ perturbed_pose")
print(f"  初始误差: {init_rot_err:.4f}°")
print(f"  修正后误差: {final_rot_err:.4f}°")
print(f"  Δ = {init_rot_err - final_rot_err:+.4f}°")

if final_rot_err < init_rot_err:
    print(f"  ✓ 方向正确! GT flow → FlowToPose → 修正成功")
else:
    print(f"  ✗ 方向错误! 修正反而增大了误差")

# === Step 5: 测试如果 flow 全零 (模型初始状态) ===
print(f"\n=== Step 5: 零 flow (模型初始状态) ===")
zero_flow = torch.zeros(1, 2, H, W)
xi_zero = flow_to_pose_weighted_lstsq(zero_flow, log_conf, depth, intrinsics)
xi_zero_correction = -xi_zero
print(f"  xi from zero flow: {xi_zero[0].tolist()}")
print(f"  应该为全零 (无修正)")

# === Step 6: 测试训练中实际发生的事 ===
print(f"\n=== Step 6: 训练管道端到端模拟 ===")
# 模拟: FlowHead 输出 flow_pred = GT_flow * 0.1 (学到了10%的正确flow)
flow_pred_10pct = flow_gt * 0.1
xi_10 = flow_to_pose_weighted_lstsq(flow_pred_10pct, log_conf, depth, intrinsics)
xi_10_correction = -xi_10
corrected_10 = se3_exp(xi_10_correction) @ perturbed_pose
err_10 = rotation_geodesic_loss(
    corrected_10[:, :3, :3], pose_gt[:, :3, :3]
).item() * 180 / np.pi
print(f"  10% GT flow → rot err: {err_10:.4f}° (初始 {init_rot_err:.2f}°) Δ={init_rot_err-err_10:+.4f}°")

# 50% of GT flow
flow_pred_50pct = flow_gt * 0.5
xi_50 = flow_to_pose_weighted_lstsq(flow_pred_50pct, log_conf, depth, intrinsics)
xi_50_correction = -xi_50
corrected_50 = se3_exp(xi_50_correction) @ perturbed_pose
err_50 = rotation_geodesic_loss(
    corrected_50[:, :3, :3], pose_gt[:, :3, :3]
).item() * 180 / np.pi
print(f"  50% GT flow → rot err: {err_50:.4f}° (初始 {init_rot_err:.2f}°) Δ={init_rot_err-err_50:+.4f}°")

# === Step 7: 实际 flow loss 值 ===
from losses.sequence_loss import masked_flow_l1_loss
valid_mask = flow_data['valid_mask']

fl_0 = masked_flow_l1_loss(zero_flow, flow_gt, valid_mask).item()
fl_10 = masked_flow_l1_loss(flow_pred_10pct, flow_gt, valid_mask).item()
fl_50 = masked_flow_l1_loss(flow_pred_50pct, flow_gt, valid_mask).item()
fl_100 = masked_flow_l1_loss(flow_gt, flow_gt, valid_mask).item()

print(f"\n=== Flow Loss 对比 ===")
print(f"  zero flow:  fl={fl_0:.4f}, Δ=0°")
print(f"  10% GT:     fl={fl_10:.4f}, Δ={init_rot_err-err_10:+.4f}°")
print(f"  50% GT:     fl={fl_50:.4f}, Δ={init_rot_err-err_50:+.4f}°")
print(f"  100% GT:    fl={fl_100:.4f} (should be 0)")
print(f"  当前训练 fl≈4.8 说明预测和GT差距很大")

# === Step 8: 检查 GT flow 在 SequenceLoss 中用 poses[k] vs P_k ===
print(f"\n=== Step 8: poses[k] vs poses[k+1] 的 GT flow 对比 ===")
# poses[0] = initial_pose = perturbed_pose (修正前)
# poses[1] = P_1 = corrected_pose (修正后, 接近GT if flow is perfect)
flow_data_before = compute_gt_flow(depth, pose_gt, perturbed_pose, intrinsics)
flow_data_after = compute_gt_flow(depth, pose_gt, corrected_pose.detach(), intrinsics)
fg_before = flow_data_before['flow']
fg_after = flow_data_after['flow']
print(f"  GT flow (poses[0]=perturbed): mean magnitude = {fg_before.abs().mean():.4f}")
print(f"  GT flow (poses[1]=corrected): mean magnitude = {fg_after.abs().mean():.4f}")
print(f"  如果用 poses[1] 做GT, flow_gt≈0, 网络学到zero flow就最优 → 永远没修正!")
