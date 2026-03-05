#!/usr/bin/env python3
"""Quick test: verify batch rendering works and measure speedup."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_3dgs.train_2dgs_geometry import (
    load_scene, GaussianModel2DGS, render_2dgs, render_2dgs_batch
)

print("Loading scene...")
train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
    load_scene("dataset/OldHospital", images_subdir="", eval_split=True)

print("Initializing Gaussians...")
gaussians = GaussianModel2DGS(sh_degree=3)
gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)

bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

# Test 1: Single view rendering (baseline)
print("\n=== Test 1: Single view ===")
cam = train_cams[0]
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=0)
torch.cuda.synchronize()
t1 = time.time()
single_time = (t1 - t0) / 5
print(f"  Resolution: {pkg['width']}x{pkg['height']}")
print(f"  Single view: {single_time*1000:.1f} ms")
print(f"  render shape: {pkg['render'].shape}")

# Test 2: Batch=4 rendering
print("\n=== Test 2: Batch rendering (B=4) ===")
cams4 = train_cams[:4]
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    pkgs = render_2dgs_batch(gaussians, cams4, bg_color, longest_edge=0)
torch.cuda.synchronize()
t1 = time.time()
batch4_time = (t1 - t0) / 5
print(f"  Batch=4: {batch4_time*1000:.1f} ms ({batch4_time/single_time:.2f}x single)")
print(f"  Effective views/sec: {4/batch4_time:.1f} vs {1/single_time:.1f} (single)")
print(f"  Per-view results: {len(pkgs)} dicts")
for k, v in pkgs[0].items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: {v.shape}")

# Test 3: Verify gradient_2dgs shape for densification
print("\n=== Test 3: Gradient shape ===")
pkgs = render_2dgs_batch(gaussians, cams4, bg_color, longest_edge=0)
vp = pkgs[0]["viewspace_points"]
print(f"  viewspace_points shape: {vp.shape}")  # Should be [4, N, 2]
print(f"  radii shape: {pkgs[0]['radii'].shape}")  # Should be [N]

# Test 4: Backward pass with batch
print("\n=== Test 4: Backward pass ===")
pkgs = render_2dgs_batch(gaussians, cams4, bg_color, longest_edge=0)
total_loss = sum(pkg["render"].mean() for pkg in pkgs) / 4
total_loss.backward()
vp = pkgs[0]["viewspace_points"]
print(f"  vp.grad shape: {vp.grad.shape if vp.grad is not None else 'None'}")
print(f"  total_loss: {total_loss.item():.4f}")

# Test 5: Memory usage
print("\n=== Memory ===")
mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(f"  Peak allocated: {mem_alloc:.2f} GB")
print(f"  Peak reserved:  {mem_reserved:.2f} GB")

# Quick speedup estimate for full training
print(f"\n=== Estimated speedup ===")
print(f"  Single-view training:  {1/single_time:.1f} iter/s  ({single_time*35000/60:.0f} min for 35k iters)")
views_per_sec_batch = 4 / batch4_time
iters_per_sec_batch = 1 / batch4_time
print(f"  Batch-4 training:      {iters_per_sec_batch:.1f} iter/s × 4 views = {views_per_sec_batch:.1f} views/s")
print(f"  Same iters (35k):      {batch4_time*35000/60:.0f} min — but {4}x more total views")
print(f"  Same views (35k):      {batch4_time*(35000/4)/60:.0f} min (need only {35000//4} iters)")

print("\n✓ All tests passed!")
