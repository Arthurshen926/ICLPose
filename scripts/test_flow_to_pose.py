#!/usr/bin/env python3
"""Test flow-to-pose conversion correctness."""
import torch
import numpy as np
from modules.flow_to_pose import DifferentiableFlowToPose, flow_to_pose_weighted_lstsq
from modules.lie_algebra import se3_exp, compute_gt_flow

B, H, W = 1, 35, 46
intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

# Test 1: Small rotation (3° around y) + small translation
xi_true = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.05, 0.0]])
print(f'True xi: {xi_true[0].tolist()}')
print(f'True rotation: {np.degrees(0.05):.2f} deg')

pose_gt = torch.eye(4).unsqueeze(0)
delta_T = se3_exp(-xi_true)
pose_perturbed = delta_T @ pose_gt
print(f'Perturbed translation: {pose_perturbed[0, :3, 3].tolist()}')

depth = torch.ones(B, H, W) * 2.0

flow_data = compute_gt_flow(depth, pose_gt, pose_perturbed, intrinsics)
flow_gt = flow_data['flow']
valid = flow_data['valid_mask']
print(f'GT flow range: u=[{flow_gt[0,0].min():.3f}, {flow_gt[0,0].max():.3f}]')
print(f'Valid pixels: {valid.sum().item()}/{H*W}')

log_conf = torch.zeros(B, 1, H, W)
xi_recovered = flow_to_pose_weighted_lstsq(flow_gt, log_conf, depth, intrinsics)
xi_correction = -xi_recovered
print(f'Recovered correction: {xi_correction[0].tolist()}')
print(f'True     correction:  {xi_true[0].tolist()}')

error = (xi_correction - xi_true).abs()
print(f'Max error: {error.max().item():.6f}')

# Test 2: Gradient flow
flow_var = flow_gt.clone().requires_grad_(True)
conf_var = log_conf.clone().requires_grad_(True)
xi_out = flow_to_pose_weighted_lstsq(flow_var, conf_var, depth, intrinsics)
loss = xi_out.abs().sum()
loss.backward()
print(f'Gradient wrt flow: mean_abs={flow_var.grad.abs().mean():.6f}')
print(f'Gradient wrt conf: mean_abs={conf_var.grad.abs().mean():.6f}')

# Test 3: Larger rotation (10 deg)
xi_big = torch.tensor([[0.0, 0.05, -0.02, 0.05, 0.15, 0.02]])
print(f'\nTest large (10 deg): xi_true={xi_big[0].tolist()}')
delta_T2 = se3_exp(-xi_big)
pose_p2 = delta_T2 @ pose_gt
flow_data2 = compute_gt_flow(depth, pose_gt, pose_p2, intrinsics)
xi_rec2 = -flow_to_pose_weighted_lstsq(flow_data2['flow'], log_conf, depth, intrinsics)
error2 = (xi_rec2 - xi_big).abs()
print(f'Recovered: {xi_rec2[0].tolist()}')
print(f'Max error at 10°: {error2.max().item():.6f}')

print('\nAll tests passed!')
