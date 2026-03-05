#!/usr/bin/env python3
"""Profile rendering and model forward/backward speed."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.backends.cudnn.benchmark = True

from modules.multiscale_renderer import MultiScaleRenderer
from ic_models.ms_flow_pose_net import MSFlowPoseNet

device = 'cuda:0'
N = 10  # iterations

# Load renderer
print("Loading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'coarse': 'output/feature_3dgs/room_0_raw/coarse/best_model.pth',
        'mid': 'output/feature_3dgs/room_0_raw/mid/best_model.pth',
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

# Load model
model = MSFlowPoseNet(
    hidden_dim=128, decode_dim=64, local_radius=4, damping=1e-3,
    coarse_hw=(7, 10), mid_hw=(15, 20), fine_hw=(35, 46),
).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {n_params/1e6:.2f}M params")

# ── Profile Rendering ──
print("\n--- Profiling Rendering ---")
poses_b1 = torch.eye(4, device=device).unsqueeze(0)
poses_b4 = poses_b1.expand(4, -1, -1).contiguous()

scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']

# Warmup
for _ in range(3):
    renderer.render_batch(poses_b1, scales=scales, return_depth=True)
torch.cuda.synchronize()

for label, poses in [("B=1", poses_b1), ("B=4", poses_b4)]:
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        renderer.render_batch(poses, scales=scales, return_depth=True)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / N * 1000
    print(f"  Render ({label}, 4 scales): {ms:.0f}ms")

# ── Profile Model Forward ──
print("\n--- Profiling Model ---")
B = 4
query = {
    'coarse': torch.randn(B, 1280, 7, 10, device=device),
    'mid': torch.randn(B, 1280, 15, 20, device=device),
    'fine_sd': torch.randn(B, 640, 35, 46, device=device),
    'fine_dino': torch.randn(B, 768, 35, 46, device=device),
}
render = {k: v.clone() for k, v in query.items()}
depth = torch.rand(B, 35, 46, device=device) * 3 + 0.5

# Warmup
for _ in range(5):
    model(query, render, depth)
torch.cuda.synchronize()

# Forward only
t0 = time.time()
for _ in range(N):
    out = model(query, render, depth)
torch.cuda.synchronize()
fwd_ms = (time.time() - t0) / N * 1000
print(f"  Forward (B=4): {fwd_ms:.0f}ms")

# Forward + Backward (AMP)
scaler = torch.cuda.amp.GradScaler()
for _ in range(3):
    model.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(query, render, depth)
    loss = out['flow_fine'].abs().mean() + out['delta_xi'].abs().mean()
    scaler.scale(loss).backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    model.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(query, render, depth)
    loss = out['flow_fine'].abs().mean() + out['delta_xi'].abs().mean()
    scaler.scale(loss).backward()
torch.cuda.synchronize()
fb_ms = (time.time() - t0) / N * 1000
print(f"  Fwd+Bwd (B=4, AMP): {fb_ms:.0f}ms")

# ── Summary ──
render_b4_ms = None
poses_b4_test = poses_b4
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    renderer.render_batch(poses_b4_test, scales=scales, return_depth=True)
torch.cuda.synchronize()
render_b4_ms = (time.time() - t0) / N * 1000

total_ms = render_b4_ms + fb_ms
steps_per_epoch = 810 // 4
print(f"\n--- Summary ---")
print(f"  Render B=4:      {render_b4_ms:.0f}ms")
print(f"  Model fwd+bwd:   {fb_ms:.0f}ms")
print(f"  Total per step:  {total_ms:.0f}ms ({total_ms/1000:.2f}s)")
print(f"  Steps/epoch:     {steps_per_epoch}")
print(f"  Per epoch:       {steps_per_epoch * total_ms / 60000:.1f} min")
print(f"  100 epochs:      {100 * steps_per_epoch * total_ms / 3600000:.1f} hours")

# Memory
mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
print(f"\n  Peak GPU mem: {mem_alloc:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
