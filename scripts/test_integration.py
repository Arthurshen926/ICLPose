#!/usr/bin/env python3
"""Quick integration test for FlowToPose in DualHead and ICPoseNetV3."""
import torch
from ic_models.ic_pose_net_v3 import ICPoseNetV3
from modules.dual_head import DualHead

intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
B = 1

# Test 1: DualHead with FlowToPose
print("=== Test 1: DualHead with FlowToPose ===")
dh = DualHead(hidden_dim=128, flow_to_pose_intrinsics=intrinsics)
hidden = torch.randn(B, 128, 35, 46)
depth = torch.ones(B, 35, 46) * 2.0
out = dh(hidden, depth=depth)
print(f'xi shape: {out["xi"].shape}')
print(f'xi values: {out["xi"][0][:3].tolist()}...')
print(f'flow shape: {out["flow"].shape}')

# Gradient test
out['xi'].sum().backward()
g = list(dh.flow_head.parameters())[0].grad
print(f'FlowHead grad: mean_abs={g.abs().mean():.6f}')
print()

# Test 2: ICPoseNetV3 model
print("=== Test 2: ICPoseNetV3 with FlowToPose ===")
scale_configs = [
    {'name': 'fine_sd',   'feat_dim': 640,  'resolution': (35, 46)},
    {'name': 'fine_dino', 'feat_dim': 768,  'resolution': (35, 46)},
]
model = ICPoseNetV3(
    scale_configs=scale_configs,
    hidden_dim=128, output_resolution=(35, 46),
    num_iters=1, residual_mode='concat', residual_out_dim=64,
    flow_to_pose_intrinsics=intrinsics,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {n_params:,}')

# forward_with_prerendered (no depth → PoseHead fallback)
query = {'fine_sd': torch.randn(B, 640, 35, 46), 'fine_dino': torch.randn(B, 768, 35, 46)}
rendered = {'fine_sd': torch.randn(B, 640, 35, 46), 'fine_dino': torch.randn(B, 768, 35, 46)}
pose = torch.eye(4).unsqueeze(0)
out2 = model.forward_with_prerendered(query, [rendered], pose, num_iters=1)
print(f'forward_with_prerendered (no depth): xi={out2["xi_list"][0].shape}')

print('\nAll integration tests passed!')
