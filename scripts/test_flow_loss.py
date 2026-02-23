#!/usr/bin/env python3
"""Quick test: verify flow loss computation works"""
import torch
import torch.nn.functional as F
import numpy as np
import sys, os, cv2
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from modules.lie_algebra import compute_gt_flow
from data.dataset_v3 import perturb_pose, load_poses_c2w, c2w_to_w2c

FLOW_H, FLOW_W = 35, 46
INTRINSICS_FLOW = {
    'fx': 320.0 * FLOW_W / 640,
    'fy': 320.0 * FLOW_H / 480,
    'cx': 319.5 * FLOW_W / 640,
    'cy': 239.5 * FLOW_H / 480,
}

device = 'cpu'

# Load a depth map and resize
depth_path = 'dataset/room_0/Sequence_1/depth/depth_0.png'
depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
depth = torch.from_numpy(depth_uint16.astype(np.float32) / 1000.0)  # (480, 640)
print(f"Full depth: shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]m")

# Resize to 35×46
depth_small = F.interpolate(depth.unsqueeze(0).unsqueeze(0), (FLOW_H, FLOW_W), mode='nearest').squeeze()
print(f"Resized depth: shape={depth_small.shape}, range=[{depth_small.min():.2f}, {depth_small.max():.2f}]m")

# Load GT pose
poses_c2w = load_poses_c2w('dataset/room_0/Sequence_1/traj_w_c.txt')
pose_gt = torch.from_numpy(c2w_to_w2c(poses_c2w[0])).float()

# Test flow at different perturbation levels
print(f"\nFlow GT test (intrinsics: fx={INTRINSICS_FLOW['fx']:.1f}, fy={INTRINSICS_FLOW['fy']:.1f}):")
for noise_deg in [1.0, 2.0, 5.0, 10.0, 15.0]:
    perturbed = perturb_pose(pose_gt.clone(), noise_deg, noise_deg * 0.02)
    flow_data = compute_gt_flow(depth_small, pose_gt, perturbed, INTRINSICS_FLOW)
    flow = flow_data['flow']
    valid = flow_data['valid_mask']
    
    from losses.sequence_loss import rotation_geodesic_loss
    actual_rot = rotation_geodesic_loss(
        perturbed[:3,:3].unsqueeze(0), pose_gt[:3,:3].unsqueeze(0)
    ).item() * 180 / np.pi
    
    flow_mag = flow.norm(dim=0)  # (H, W)
    valid_mask = valid.squeeze(0) > 0.5
    valid_flow = flow_mag[valid_mask]
    
    print(f"  noise={noise_deg:5.1f}° (actual={actual_rot:5.1f}°): "
          f"flow shape={flow.shape}, "
          f"mag: mean={valid_flow.mean():.3f}, max={valid_flow.max():.3f} px "
          f"(valid: {valid_mask.sum()}/{valid_mask.numel()})")

# Test confidence-weighted flow loss
from losses.sequence_loss import confidence_weighted_flow_loss
flow_pred = torch.zeros(1, 2, FLOW_H, FLOW_W)  # Zero prediction
perturbed = perturb_pose(pose_gt.clone(), 2.0, 0.03)
flow_data = compute_gt_flow(depth_small, pose_gt, perturbed, INTRINSICS_FLOW)
flow_gt = flow_data['flow'].unsqueeze(0)
valid_mask = flow_data['valid_mask'].unsqueeze(0)
log_conf = torch.zeros(1, 1, FLOW_H, FLOW_W)

flow_loss = confidence_weighted_flow_loss(flow_pred, flow_gt, log_conf, valid_mask)
print(f"\nFlow loss (zero pred, 2° perturbation): {flow_loss.item():.4f}")
print("Expected: ~0.2-1.0 (mean |flow_gt| per pixel)")
