#!/usr/bin/env python3
"""
诊断 FlowHead 光流预测质量
加载 v8 checkpoint, 对单个样本检查:
1. 预测 flow vs GT flow 的分布
2. flow 方向正确性
3. FlowToPose 恢复精度
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from ic_models.ic_pose_net_v3 import ICPoseNetV3
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import compute_gt_flow
from modules.flow_to_pose import flow_to_pose_weighted_lstsq
from data.dataset_v3 import PoseDatasetV3
from losses.sequence_loss import rotation_geodesic_loss
import os

device = torch.device('cuda')
intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

# Load checkpoint
ckpt_path = 'output/v3_train_v8_flow2pose/best_model.pth'
if not os.path.exists(ckpt_path):
    print(f"No checkpoint at {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=device)
scale_configs = ckpt['scale_configs']
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

# Build model
model = ICPoseNetV3(
    scale_configs=scale_configs,
    hidden_dim=128,
    output_resolution=(35, 46),
    num_iters=1,
    residual_mode='concat',
    residual_out_dim=64,
    flow_to_pose_intrinsics=intrinsics,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Build renderer
ply_path = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
renderer = MultiScaleRenderer(
    ply_path=ply_path,
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

# Load a sample
dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(900)),
    scale_names=['fine_sd', 'fine_dino'],
    noise_rot_deg=2.0,
    noise_trans_m=0.03,
    is_train=True,
    depth_resize=(35, 46),
)

# Test on single sample
torch.manual_seed(42)
sample = dataset[100]

query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
depth = sample['depth'].unsqueeze(0).to(device)

# Initial error
with torch.no_grad():
    init_rot = rotation_geodesic_loss(
        initial_pose[:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi
    print(f"\nInitial rotation error: {init_rot:.2f}°")
    
    # Forward pass
    pred = model(
        query_feats=query_feats,
        initial_pose=initial_pose,
        renderer=renderer,
        num_iters=1,
        depth_gt=depth,
    )
    
    xi = pred['xi_list'][0]
    flow_pred = pred['flow_list'][0]
    log_conf = pred['log_conf_list'][0]
    final_pose = pred['final_pose']
    
    final_rot = rotation_geodesic_loss(
        final_pose[:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi
    
    print(f"Final rotation error: {final_rot:.2f}°")
    print(f"Improvement: {init_rot - final_rot:+.2f}°")
    print(f"xi: {xi[0].cpu().tolist()}")
    print(f"xi_norm: {xi[0].norm().item():.6f} rad = {xi[0].norm().item()*180/np.pi:.2f}°")
    
    # GT flow
    flow_data = compute_gt_flow(depth, pose_gt, initial_pose, intrinsics)
    flow_gt = flow_data['flow']
    valid = flow_data['valid_mask']
    
    # Flow statistics
    valid_mask = valid.squeeze(1).bool()  # (1, H, W)
    fp = flow_pred[0]  # (2, H, W)
    fg = flow_gt[0]    # (2, H, W)
    
    fp_u = fp[0][valid_mask[0]]
    fp_v = fp[1][valid_mask[0]]
    fg_u = fg[0][valid_mask[0]]
    fg_v = fg[1][valid_mask[0]]
    
    print(f"\nFlow Statistics (valid pixels: {valid_mask.sum().item()}):")
    print(f"  GT flow:   u=[{fg_u.min():.3f}, {fg_u.max():.3f}] mean={fg_u.mean():.4f}  v=[{fg_v.min():.3f}, {fg_v.max():.3f}] mean={fg_v.mean():.4f}")
    print(f"  Pred flow: u=[{fp_u.min():.3f}, {fp_u.max():.3f}] mean={fp_u.mean():.4f}  v=[{fp_v.min():.3f}, {fp_v.max():.3f}] mean={fp_v.mean():.4f}")
    
    error = torch.sqrt((fp_u - fg_u)**2 + (fp_v - fg_v)**2)
    gt_mag = torch.sqrt(fg_u**2 + fg_v**2)
    pred_mag = torch.sqrt(fp_u**2 + fp_v**2)
    
    print(f"  GT magnitude:   mean={gt_mag.mean():.4f} px")
    print(f"  Pred magnitude: mean={pred_mag.mean():.4f} px")
    print(f"  Flow L2 error:  mean={error.mean():.4f} px")
    print(f"  Error/GT ratio: {error.mean()/gt_mag.mean():.2f}")
    
    # Direction correctness (cosine similarity)
    dot = fp_u * fg_u + fp_v * fg_v
    cos_sim = dot / (pred_mag * gt_mag + 1e-8)
    valid_dir = pred_mag > 0.01  # only check where prediction is non-trivial
    if valid_dir.sum() > 0:
        cos_valid = cos_sim[valid_dir]
        print(f"  Cosine similarity: mean={cos_valid.mean():.4f} ({valid_dir.sum().item()} pixels with |flow|>0.01)")
        print(f"  Correct direction (cos>0): {(cos_valid > 0).float().mean():.1%}")
    
    # Confidence
    conf = torch.exp(log_conf[0, 0])
    print(f"\nConfidence: mean={conf.mean():.4f}, min={conf.min():.4f}, max={conf.max():.4f}")
    
    # Test: what pose does GT flow give through FlowToPose?
    xi_from_gt = flow_to_pose_weighted_lstsq(
        flow_gt, torch.zeros(1, 1, 35, 46, device=device), depth, intrinsics
    )
    xi_correction_gt = -xi_from_gt
    from modules.lie_algebra import se3_exp
    corrected_pose = se3_exp(xi_correction_gt) @ initial_pose
    corrected_rot = rotation_geodesic_loss(
        corrected_pose[:, :3, :3], pose_gt[:, :3, :3]
    ).item() * 180 / np.pi
    print(f"\nGT flow → FlowToPose → correction:")
    print(f"  Corrected rot error: {corrected_rot:.4f}° (from {init_rot:.2f}°)")
    print(f"  This is the CEILING: FlowToPose CAN correct to {corrected_rot:.2f}° if flow is perfect")
