#!/usr/bin/env python3
"""
Benchmark: 分析 FDA 各步骤耗时 + 测试不同 chunk_size 和 PCA 降维
"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import torch
import time
import numpy as np
from modules.featuremetric import (
    FeaturemetricAligner, compute_image_jacobian,
    compute_spatial_gradient
)
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from feature_3dgs.feature_renderer import FeatureRenderer

device = torch.device('cuda')

# ==============================
# 1. 显存分析
# ==============================
print(f"=== GPU Memory Analysis ===")
print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# ==============================
# 2. 加载资源
# ==============================
print("\nLoading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

print(f"\nAfter loading models:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 高斯数量
for name, model in renderer.models.items():
    N = model.get_xyz.shape[0]
    D = model.feature_dim
    print(f"  {name}: N={N} gaussians, D={D} feature dims")
    # 计算特征嵌入显存
    feat_mem = N * D * 4 / 1024**2  # float32, MB
    print(f"    Feature embedding: {feat_mem:.1f} MB")

# ==============================
# 3. 渲染耗时 Benchmark
# ==============================
print("\n=== Rendering Benchmark ===")

dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(10)),
    scale_names=['fine_sd', 'fine_dino'],
    noise_rot_deg=2.0,
    noise_trans_m=0.04,
    is_train=True,
    depth_resize=(35, 46),
)

sample = dataset[0]
pose = sample['initial_pose'].to(device)

# Warmup
with torch.no_grad():
    renderer.render_scale('fine_dino', pose)

# Benchmark: render_scale for each scale
for scale_name in ['fine_dino', 'fine_sd']:
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            renderer.render_scale(scale_name, pose)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    D = renderer.models[scale_name].feature_dim
    n_chunks = (D + 31) // 32
    print(f"  {scale_name} (D={D}, {n_chunks} chunks): "
          f"{np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")

# Benchmark: depth rendering
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        renderer.render_scale('fine_dino', pose, return_depth=True)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
print(f"  fine_dino + depth: {np.mean(times)*1000:.1f}ms ± {np.std(times)*1000:.1f}ms")

# Benchmark: project_gaussians alone
from gsplat import project_gaussians
model = renderer.models['fine_dino']
fH, fW = 35, 46
scale_x, scale_y = fW / 640, fH / 480
rfx, rfy = 320 * scale_x, 320 * scale_y
rcx, rcy = 319.5 * scale_x, 239.5 * scale_y

times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        xys, depths, radii, conics, comp, num_tiles, cov3d = project_gaussians(
            means3d=model.get_xyz, scales=model.get_scaling,
            glob_scale=1.0, quats=model.get_rotation,
            viewmat=pose, fx=rfx, fy=rfy, cx=rcx, cy=rcy,
            img_height=fH, img_width=fW, block_width=16,
        )
    torch.cuda.synchronize()
    times.append(time.time() - t0)
print(f"  project_gaussians alone: {np.mean(times)*1000:.1f}ms")

# Benchmark: single rasterize_gaussians call with different channel counts
from gsplat import rasterize_gaussians
opacities = model.get_opacity.squeeze(-1)
loc_features = model.get_loc_feature  # (N, D)

print("\n=== Chunk Size Benchmark ===")
for chunk_size in [16, 32, 64, 128]:
    try:
        chunk_colors = loc_features[:, :min(chunk_size, loc_features.shape[1])]
        chunk_bg = torch.zeros(chunk_colors.shape[1], device=device)
        
        # Warmup
        rasterize_gaussians(
            xys=xys, depths=depths, radii=radii, conics=conics,
            num_tiles_hit=num_tiles, colors=chunk_colors,
            opacity=opacities, img_height=fH, img_width=fW,
            block_width=16, background=chunk_bg, return_alpha=False,
        )
        torch.cuda.synchronize()
        
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            result = rasterize_gaussians(
                xys=xys, depths=depths, radii=radii, conics=conics,
                num_tiles_hit=num_tiles, colors=chunk_colors,
                opacity=opacities, img_height=fH, img_width=fW,
                block_width=16, background=chunk_bg, return_alpha=False,
            )
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"  chunk_size={chunk_size:3d}: {np.mean(times)*1000:.1f}ms "
              f"({'WORKS' if True else ''})")
    except Exception as e:
        print(f"  chunk_size={chunk_size:3d}: FAILED ({e})")

# ==============================
# 4. GN 步骤内各部分耗时
# ==============================
print("\n=== GN Step Breakdown ===")
query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
depth = sample['depth'].unsqueeze(0).to(device)
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

import torch.nn.functional as F

# Rendering
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    rendered = renderer.render_scale('fine_dino', pose)
    rendered_feat = rendered['feature_map'].unsqueeze(0)
torch.cuda.synchronize()
t_render = time.time() - t0

# Spatial gradient
torch.cuda.synchronize()
t0 = time.time()
grad_u, grad_v = compute_spatial_gradient(rendered_feat)
torch.cuda.synchronize()
t_grad = time.time() - t0

# Image Jacobian
torch.cuda.synchronize()
t0 = time.time()
Ju, Jv, valid = compute_image_jacobian(depth, INTRINSICS)
torch.cuda.synchronize()
t_jac = time.time() - t0

# Normal equations
B, D, H, W = rendered_feat.shape
N = H * W
q = F.normalize(query_feats['fine_dino'], p=2, dim=1)
residual = q - rendered_feat
gu = grad_u.reshape(B, D, N)
gv = grad_v.reshape(B, D, N)
res = residual.reshape(B, D, N)

torch.cuda.synchronize()
t0 = time.time()
A = (gu * gu).sum(1) * valid.float()
B_ = (gv * gv).sum(1) * valid.float()
C = (gu * gv).sum(1) * valid.float()
Ru = (gu * res).sum(1) * valid.float()
Rv = (gv * res).sum(1) * valid.float()
JtJ = (torch.bmm((Ju * A.unsqueeze(-1)).transpose(1, 2), Ju) +
       torch.bmm((Ju * C.unsqueeze(-1)).transpose(1, 2), Jv) +
       torch.bmm((Jv * C.unsqueeze(-1)).transpose(1, 2), Ju) +
       torch.bmm((Jv * B_.unsqueeze(-1)).transpose(1, 2), Jv))
JtR = -(torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1)) +
        torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1)))
torch.cuda.synchronize()
t_normal = time.time() - t0

# Solve
torch.cuda.synchronize()
t0 = time.time()
damping_mat = 1e-2 * torch.eye(6, device=device).unsqueeze(0)
delta = torch.linalg.solve(JtJ + damping_mat, JtR)
torch.cuda.synchronize()
t_solve = time.time() - t0

total = t_render + t_grad + t_jac + t_normal + t_solve
print(f"  Render:     {t_render*1000:6.1f}ms  ({t_render/total*100:4.1f}%)")
print(f"  Gradient:   {t_grad*1000:6.1f}ms  ({t_grad/total*100:4.1f}%)")
print(f"  Image Jac:  {t_jac*1000:6.1f}ms  ({t_jac/total*100:4.1f}%)")
print(f"  Normal Eq:  {t_normal*1000:6.1f}ms  ({t_normal/total*100:4.1f}%)")
print(f"  Solve 6x6:  {t_solve*1000:6.1f}ms  ({t_solve/total*100:4.1f}%)")
print(f"  Total:      {total*1000:6.1f}ms")

# ==============================
# 5. PCA 降维评估
# ==============================
print("\n=== PCA Compression Analysis ===")
# 收集一些渲染特征来计算 PCA
feats_all = []
for i in range(0, min(50, len(dataset)), 5):
    s = dataset[i]
    p = s['initial_pose'].to(device)
    with torch.no_grad():
        r = renderer.render_scale('fine_dino', p)
    feats_all.append(r['feature_map'].reshape(768, -1).T)  # (N_pixels, 768)

feats_cat = torch.cat(feats_all, dim=0)  # (K, 768)
print(f"  PCA data: {feats_cat.shape[0]} samples × {feats_cat.shape[1]} dims")

# Center
mean = feats_cat.mean(0, keepdim=True)
feats_centered = feats_cat - mean

# SVD (compute top-K components)
U, S, Vh = torch.linalg.svd(feats_centered, full_matrices=False)
total_var = (S ** 2).sum()
for k in [32, 64, 128, 256]:
    explained_var = (S[:k] ** 2).sum() / total_var
    print(f"  PCA-{k:3d}: explained variance = {explained_var*100:.1f}%")

# Estimate speedup
for k in [64, 128]:
    n_chunks_old = (768 + 31) // 32  # 24
    n_chunks_new = (k + 63) // 64    # with chunk_size=64
    speedup = n_chunks_old / n_chunks_new
    print(f"  PCA-{k} + chunk64: {n_chunks_new} chunks/iter (was {n_chunks_old}) → {speedup:.1f}× render speedup")

print(f"\nFinal GPU memory:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Peak:      {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"  Total:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  Free:      {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
