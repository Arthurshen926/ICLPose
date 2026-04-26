#!/usr/bin/env python3
"""Smoke test for GSFFs modules."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
from pathlib import Path

import torch


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

print('Testing imports...')
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder, SegmentationHead
from gsff.losses import info_nce_loss, prototypical_loss, segmentation_ce_loss
from gsff.clustering import compute_prototypes
from gsff.pose_refine import se3_exp

# Test triplane
print('Testing triplane...')
tp = DualScaleTriplane(256, 512, 16, 10.0).cuda()  # Use 512 not 1024 for speed
xyz = torch.randn(100, 3, device='cuda')
feat_c = tp.extract_coarse(xyz)
feat_f = tp.extract_fine(xyz)
print(f'  Coarse features: {feat_c.shape}')
print(f'  Fine features: {feat_f.shape}')
print(f'  TV loss: {tp.total_variation_loss().item():.6f}')

# Test encoder
print('Testing encoder...')
enc = DualScaleEncoder(16, freeze_backbone=True).cuda()
img = torch.randn(1, 3, 280, 504, device='cuda')
with torch.no_grad():
    fc, ff = enc(img)
print(f'  Coarse 2D features: {fc.shape}')
print(f'  Fine 2D features: {ff.shape}')

# Test losses
print('Testing losses...')
f3d = torch.randn(1, 16, 10, 10, device='cuda')
f2d = torch.randn(1, 16, 10, 10, device='cuda')
f3d = torch.nn.functional.normalize(f3d, dim=1)
f2d = torch.nn.functional.normalize(f2d, dim=1)
loss = info_nce_loss(f3d, f2d, max_samples=50)
print(f'  NCE loss: {loss.item():.4f}')

# Test prototypes
print('Testing prototypes...')
protos = torch.randn(34, 16, device='cuda')
protos = torch.nn.functional.normalize(protos, dim=1)
lp = prototypical_loss(f3d, f2d, protos, max_samples=50)
print(f'  Prototypical loss: {lp.item():.4f}')

# Test se3_exp identity
print('Testing se3_exp...')
xi = torch.zeros(6, device='cuda')
T = se3_exp(xi)
print(f'  Identity: diag={[f"{x:.4f}" for x in T.diag().tolist()]}')

# Test se3_exp with small perturbation
xi2 = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.01], device='cuda')
T2 = se3_exp(xi2)
print(f'  Small perturb: t={[f"{x:.4f}" for x in T2[:3, 3].tolist()]}')

print('\nAll smoke tests passed!')
