#!/usr/bin/env python3
"""Diagnose floater artifacts in 2DGS model."""
import sys, torch, numpy as np
sys.path.insert(0, 'scripts')
from visualize_2dgs_recon import load_ply_2dgs

ply = sys.argv[1] if len(sys.argv) > 1 else 'output/2dgs_models/OldHospital/v3/point_cloud/iteration_30000/point_cloud.ply'
m = load_ply_2dgs(ply)
scales = torch.exp(m['scales'])
opacities = torch.sigmoid(m['opacities'])

print(f'Total Gaussians: {scales.shape[0]:,}')
print(f'Scale shape: {scales.shape}')

max_scale = scales.max(dim=1).values
print(f'\nScale stats:')
print(f'  mean: {max_scale.mean():.4f}')
print(f'  median: {max_scale.median():.4f}')
print(f'  std: {max_scale.std():.4f}')
for p in [90, 95, 99, 99.5, 99.9]:
    val = torch.quantile(max_scale, p/100).item()
    print(f'  p{p}: {val:.4f}')
print(f'  max: {max_scale.max():.4f}')

for threshold in [0.5, 1.0, 2.0, 5.0, 10.0]:
    cnt = (max_scale > threshold).sum().item()
    print(f'  scale > {threshold}: {cnt} ({cnt/len(max_scale)*100:.2f}%)')

print(f'\nOpacity stats:')
print(f'  mean: {opacities.mean():.4f}')
print(f'  < 0.01: {(opacities < 0.01).sum().item()}')
print(f'  < 0.1: {(opacities < 0.1).sum().item()}')
print(f'  > 0.5: {(opacities > 0.5).sum().item()}')
print(f'  > 0.9: {(opacities > 0.9).sum().item()}')

for st in [1.0, 2.0, 5.0]:
    mask = (max_scale > st) & (opacities.squeeze() > 0.3)
    print(f'  scale>{st} & opacity>0.3: {mask.sum().item()}')
