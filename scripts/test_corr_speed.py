#!/usr/bin/env python3
"""Benchmark local_correlation implementations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch, time
from ic_models.corr_pose_net import local_correlation, _local_correlation_unfold, _local_correlation_fast_loop

torch.manual_seed(42)
device = 'cuda'

# Test 1: Standard resolution (35×46, C=128, B=8)
fmap1 = torch.randn(8, 128, 35, 46, device=device)
fmap2 = torch.randn(8, 128, 35, 46, device=device)

for _ in range(3):
    _local_correlation_unfold(fmap1, fmap2, 4)
    _local_correlation_fast_loop(fmap1, fmap2, 4)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    c1 = _local_correlation_unfold(fmap1, fmap2, 4)
torch.cuda.synchronize()
t_unfold = (time.time() - t0) / 20

t0 = time.time()
for _ in range(20):
    c2 = _local_correlation_fast_loop(fmap1, fmap2, 4)
torch.cuda.synchronize()
t_loop = (time.time() - t0) / 20

diff = (c1 - c2).abs().max().item()
print(f'Standard 35x46 (B=8, C=128):')
print(f'  unfold: {t_unfold*1000:.1f}ms')
print(f'  loop:   {t_loop*1000:.1f}ms')
print(f'  speedup: {t_loop/t_unfold:.1f}x')
print(f'  max diff: {diff:.1e}')

# Test 2: Upsampled resolution (140×184, C=64, B=8)
fmap3 = torch.randn(8, 64, 140, 184, device=device)
fmap4 = torch.randn(8, 64, 140, 184, device=device)

for _ in range(2):
    _local_correlation_fast_loop(fmap3, fmap4, 4)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(5):
    c3 = _local_correlation_fast_loop(fmap3, fmap4, 4)
torch.cuda.synchronize()
t_loop_up = (time.time() - t0) / 5

# Auto-select for upsampled
t0 = time.time()
for _ in range(5):
    c4 = local_correlation(fmap3, fmap4, 4)
torch.cuda.synchronize()
t_auto = (time.time() - t0) / 5

print(f'\nUpsampled 140x184 (B=8, C=64):')
print(f'  fast_loop: {t_loop_up*1000:.1f}ms')
print(f'  auto: {t_auto*1000:.1f}ms')
