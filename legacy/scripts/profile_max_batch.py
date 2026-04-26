#!/usr/bin/env python3
"""Profile maximum batch size for 2DGS training on current GPU.
Tests increasing batch sizes until OOM, with full forward+backward pass.
"""
import torch, time, sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.cuda.empty_cache()
gc.collect()

from feature_3dgs.train_2dgs_geometry import (
    load_scene, GaussianModel2DGS, render_2dgs_batch
)

# Load scene
source_dir = "dataset/OldHospital"
train_cams, test_cams, pcd_xyz, pcd_rgb, extent = load_scene(source_dir, images_subdir="")
print(f"Scene: {len(train_cams)} train cams, {pcd_xyz.shape[0]} points")

# Init model
gaussians = GaussianModel2DGS(sh_degree=3)
gaussians.create_from_pcd(
    torch.tensor(pcd_xyz).cuda().float(),
    torch.tensor(pcd_rgb).cuda().float(),
    spatial_lr_scale=extent,
)
# Create a minimal args object with required fields
class MinArgs:
    percent_dense = 0.01
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    iterations = 35000
gaussians.training_setup(MinArgs())
N = gaussians._xyz.shape[0]
bg = torch.zeros(4, device="cuda")

print(f"\nGaussians: {N:,}")
print(f"Resolution: {train_cams[0].width}x{train_cams[0].height}")
print(f"GPU total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else f"GPU total: {torch.cuda.mem_get_info()[1] / 1024**3:.1f} GB")
print()

# Test increasing batch sizes
import random
results = []

for bs in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    cams = [random.choice(train_cams) for _ in range(bs)]
    
    try:
        # Forward
        t0 = time.time()
        renders = render_2dgs_batch(gaussians, cams, bg, longest_edge=0)
        torch.cuda.synchronize()
        fwd_time = time.time() - t0
        
        # Compute loss (L1 on random target, simulating real training)
        total_loss = torch.tensor(0.0, device="cuda")
        for r in renders:
            total_loss = total_loss + r["render"].mean()  # dummy loss
        total_loss = total_loss / bs
        
        # Backward
        t0 = time.time()
        total_loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - t0
        
        gaussians.optimizer.zero_grad(set_to_none=True)
        
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        peak_gb = peak_mb / 1024
        
        total_time = fwd_time + bwd_time
        views_per_sec = bs / total_time
        
        results.append((bs, peak_gb, total_time, views_per_sec))
        print(f"  batch={bs:3d}  |  peak={peak_gb:6.2f} GB  |  fwd={fwd_time:.3f}s  bwd={bwd_time:.3f}s  total={total_time:.3f}s  |  {views_per_sec:.1f} views/s")
        
        if peak_gb > 22:  # stop if close to limit
            print(f"  → Approaching 24GB limit, stopping.")
            break
            
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  batch={bs:3d}  |  *** OOM ***")
        break
    except Exception as e:
        print(f"  batch={bs:3d}  |  ERROR: {e}")
        break

print(f"\n{'='*70}")
if results:
    best = max(results, key=lambda x: x[3])
    # Also find the largest batch that fits under 20GB (leave headroom for densification)
    safe = [r for r in results if r[1] < 20]
    recommended = max(safe, key=lambda x: x[3]) if safe else results[-1]
    print(f"Fastest throughput: batch={best[0]}, {best[3]:.1f} views/s, {best[1]:.1f} GB")
    print(f"Recommended (≤20 GB): batch={recommended[0]}, {recommended[3]:.1f} views/s, {recommended[1]:.1f} GB")
    print(f"\nWith batch={recommended[0]} & 35000 iters → {recommended[0]*35000:,} total views")
    print(f"ETA: {35000 / recommended[3] / 60:.0f} min")
