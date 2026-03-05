#!/usr/bin/env python3
"""Test AMP (mixed precision) training with geometry solver."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ic_models.ms_flow_pose_net import MSFlowPoseNet

device = 'cuda'
model = MSFlowPoseNet(coarse_hw=(7,10), mid_hw=(15,20), fine_hw=(35,46)).to(device)
B = 2
q = {'coarse': torch.randn(B,1280,7,10,device=device),
     'mid': torch.randn(B,1280,15,20,device=device),
     'fine_sd': torch.randn(B,640,35,46,device=device),
     'fine_dino': torch.randn(B,768,35,46,device=device)}
r = {k: torch.randn_like(v) for k,v in q.items()}
depth = torch.rand(B,35,46,device=device) * 3 + 0.1

scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
optimizer.zero_grad()
with torch.cuda.amp.autocast():
    out = model(q, r, depth)

loss = out['delta_xi'].abs().mean()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f'delta_xi dtype: {out["delta_xi"].dtype}')
print(f'flow_fine dtype: {out["flow_fine"].dtype}')
grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f'grad_norm: {grad_norm:.4f}')
print('✓ AMP test PASSED!')
