#!/usr/bin/env python3
"""Test gradient flow through FlowToPose with real-like training loss."""
import torch
from modules.dual_head import DualHead

B = 1
intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

dh = DualHead(hidden_dim=128, flow_to_pose_intrinsics=intrinsics)
hidden = torch.randn(B, 128, 35, 46)
depth = torch.ones(B, 35, 46) * 2.0

# Simulate real training: xi_gt is nonzero
xi_gt = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.035, 0.0]])

out = dh(hidden, depth=depth)
xi = out['xi']
print(f'Initial xi: {xi[0].tolist()}')

# L1 loss like in real training
loss = torch.abs(xi - xi_gt).sum()
print(f'Loss: {loss.item():.6f}')
loss.backward()

# Check gradients on flow_conv (the final conv that produces flow)
flow_conv_grad = dh.flow_head.flow_conv.weight.grad
conf_conv_grad = dh.flow_head.conf_conv.weight.grad
conv1_grad = list(dh.flow_head.conv[0].parameters())[0].grad

print(f'flow_conv.weight grad: mean_abs={flow_conv_grad.abs().mean():.8f}')
print(f'conf_conv.weight grad: mean_abs={conf_conv_grad.abs().mean():.8f}')
print(f'First conv layer grad: mean_abs={conv1_grad.abs().mean():.8f}')

if flow_conv_grad.abs().mean() > 0:
    print('✓ Gradient flows through FlowToPose → FlowHead')
else:
    print('✗ ZERO gradient! Problem!')
