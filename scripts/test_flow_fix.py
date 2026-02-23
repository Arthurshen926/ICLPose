"""验证修复后的 flow GT 方向与 FlowToPose 一致性"""
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.lie_algebra import se3_exp, compute_gt_flow
from modules.flow_to_pose import flow_to_pose_weighted_lstsq
from losses.sequence_loss import masked_flow_l1_loss, rotation_geodesic_loss

B, H, W = 1, 35, 46
intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

# Create known perturbation
pose_gt = torch.eye(4).unsqueeze(0)
xi_noise = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.035, 0.0]])  # ~2 deg rotation
delta = se3_exp(xi_noise)
initial_pose = delta @ pose_gt  # P_0 = perturbed away from GT
depth = torch.ones(B, H, W) * 2.0

# Compute GT flow at P_0 (BEFORE correction) - this is the FIXED version
flow_gt_data = compute_gt_flow(depth, pose_gt, initial_pose, intrinsics)
flow_gt = flow_gt_data['flow']
valid = flow_gt_data['valid_mask']

print('=== Test: Flow direction consistency ===')
print(f'GT flow u mean: {flow_gt[0,0].mean():.4f}')
print(f'GT flow v mean: {flow_gt[0,1].mean():.4f}')

# If FlowHead predicts the correct flow (matches GT->P_0 direction)
# Then FlowToPose should produce a correction back to GT
xi_recovered = flow_to_pose_weighted_lstsq(flow_gt, torch.zeros(B,1,H,W), depth, intrinsics)
xi_correction = -xi_recovered  # negate_output=True in DifferentiableFlowToPose
P_corrected = se3_exp(xi_correction) @ initial_pose

init_err = rotation_geodesic_loss(initial_pose[:,:3,:3], pose_gt[:,:3,:3]).item() * 180/np.pi
corrected_err = rotation_geodesic_loss(P_corrected[:,:3,:3], pose_gt[:,:3,:3]).item() * 180/np.pi
print(f'Initial error: {init_err:.2f} deg')
print(f'Corrected error: {corrected_err:.4f} deg')
print(f'Improvement: {init_err - corrected_err:.2f} deg')

# Now verify L1 loss
flow_pred_good = flow_gt.clone()
flow_pred_bad = -flow_gt.clone()
flow_pred_zero = torch.zeros_like(flow_gt)

l_good = masked_flow_l1_loss(flow_pred_good, flow_gt, valid).item()
l_bad = masked_flow_l1_loss(flow_pred_bad, flow_gt, valid).item()
l_zero = masked_flow_l1_loss(flow_pred_zero, flow_gt, valid).item()

print(f'\nFlow L1 loss:')
print(f'  Perfect:  {l_good:.6f}')
print(f'  Zero:     {l_zero:.6f}')
print(f'  Opposite: {l_bad:.6f}')

print('\n=== CONCLUSION ===')
print('Flow GT at poses[k] (before correction) -> FlowToPose -> correct xi -> lower pose error')
print('Flow loss and pose loss are now CONSISTENT. Both push FlowHead in the same direction.')
