#!/usr/bin/env python3
"""
Joint 2DGS Geometry + Feature Training v2
==========================================
YAML 配置驱动的联合训练，支持不同数据集约定 (Cambridge / Replica / COLMAP)。

改进点 (vs v1):
  - YAML 配置文件，清晰区分数据集类型和路径
  - 多 loss 组合: L1 + SSIM + normal consistency + distortion + feature(L1+cos)
  - 定期评估 PSNR + 多通道可视化 (RGB/Depth/Normal/Feature PCA vs GT)
  - 支持全分辨率训练 (longest_edge=0) 和降分辨率
  - 特征 warmup: 先稳定几何再引入特征 loss
  - 正确的 pose 加载 (区分 Cambridge/Replica/COLMAP 约定)

用法:
  CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_joint_v2 \
    --config configs/joint_oh_v1.yaml
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from random import randint, shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from gsplat import rasterization_2dgs

from feature_gaussian.legacy_3dgs.train_2dgs_geometry import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    qvec2rotmat,
    focal2fov,
    RGB2SH,
    inverse_sigmoid,
    build_rotation,
    GaussianModel2DGS,
    CameraData,
    ssim,
    load_image_tensor,
    pearson_depth_loss,
    load_mono_depth,
)

# Reuse v1 components
from feature_gaussian.legacy_3dgs.train_2dgs_joint import (
    DA3FeatureCache,
    GaussianModel2DGSJoint,
    render_rgb_2dgs,
    render_features_2dgs,
    build_da3_image_order,
)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'exp_name': 'joint_v2',

    'dataset': {
        'type': 'colmap',       # colmap | replica | cambridge
        'source_dir': '',       # COLMAP scene directory
        'images': '',           # image subdir (empty = "images")
        'feature_dir': '',      # DA3 feature dir with coarse/mid/fine
        'feature_scales': ['fine'],   # which scales to train: coarse, mid, fine
        'traj_path': '',        # optional: override pose source (for replica)
        # Cambridge/non-Replica datasets: if traj must be loaded from traj_w_c.txt
        # and the source is not a COLMAP scene, set traj_path instead.
    },

    'model': {
        'sh_degree': 3,
        'white_background': False,
        'random_background': True,
    },

    'training': {
        'iterations': 30000,
        'longest_edge': 0,      # 0 = full resolution
        'eval_interval': 5000,  # periodic eval + visualization
        'save_interval': 5000,
        'vis_frames': [0, 200, 500, 800],  # frames for visualization

        # Learning rates
        'position_lr_init': 0.00016,
        'position_lr_final': 0.0000016,
        'feature_lr': 0.0025,    # SH feature LR
        'opacity_lr': 0.05,
        'scaling_lr': 0.005,
        'rotation_lr': 0.001,
        'percent_dense': 0.01,

        # Feature embedding
        'feature_embedding_lr': 0.01,
        'feature_weight': 0.1,
        'feature_cos_weight': 0.5,   # cosine loss weight within feature loss
        'feature_start_iter': 500,   # delay feature training for geometry warmup
        'detach_feat_geometry': False,

        # RGB losses
        'lambda_dssim': 0.2,

        # 2DGS geometry regularization
        'lambda_normal': 0.05,
        'lambda_dist': 0.01,
        'lambda_scale': 0.001,
        'scale_reg_threshold': 0.3,
        'reg_start_iter': 500,

        # Monocular depth supervision
        'mono_depth_dir': '',       # path to mono_depth/ dir with seq*/frame*.npy
        'lambda_depth': 0.0,        # 0 = disabled; recommended 0.05
        'depth_start_iter': 3000,
        'depth_warmup_iters': 2000,

        # Densification
        'densify_from_iter': 500,
        'densify_until_iter': 20000,
        'densification_interval': 100,
        'densify_grad_threshold': 0.0002,
        'opacity_reset_interval': 3000,
        'opacity_reset_value': 0.01,
    },

    'output_dir': 'output/2dgs_joint',
}


def load_config(config_path):
    """Load YAML config with defaults."""
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f)
    cfg = _deep_merge(DEFAULT_CONFIG.copy(), user_cfg)
    return cfg


def _deep_merge(base, override):
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ════════════════════════════════════════════════════════════════════════════
# Dataset-specific scene loading
# ════════════════════════════════════════════════════════════════════════════

def load_scene_colmap(source_dir, images_subdir=''):
    """Load scene from COLMAP sparse reconstruction."""
    sparse_dir = os.path.join(source_dir, "sparse", "0")
    cam_intrinsics = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    cam_extrinsics = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    images_dir = os.path.join(source_dir, images_subdir if images_subdir else "images")

    # Test split
    test_names = set()
    for test_file in [os.path.join(sparse_dir, "list_test.txt"),
                      os.path.join(source_dir, "dataset_test.txt")]:
        if os.path.exists(test_file):
            with open(test_file) as f:
                for l in f:
                    l = l.strip()
                    if l and not l.startswith("#"):
                        test_names.add(l.split(" ")[0])
            break

    train_cams, test_cams = [], []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        R = qvec2rotmat(extr.qvec).T  # COLMAP convention
        T = extr.tvec
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[0], intr.height)
        elif intr.model in ("PINHOLE", "OPENCV"):
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[1], intr.height)
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        img_path = os.path.join(images_dir, extr.name)
        if not os.path.exists(img_path):
            alt = os.path.join(source_dir, extr.name)
            if os.path.exists(alt):
                img_path = alt

        cam = CameraData(
            uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
            image=img_path, image_name=extr.name,
            width=intr.width, height=intr.height,
        )
        if extr.name in test_names:
            test_cams.append(cam)
        else:
            train_cams.append(cam)

    # Point cloud
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(ply_path):
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        pcd_xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        pcd_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.float32) / 255.0
    elif os.path.exists(bin_path):
        pcd_xyz, pcd_rgb = read_points3d_binary(bin_path)
    else:
        raise FileNotFoundError(f"No point cloud in {sparse_dir}")

    cameras_extent = _compute_extent([*train_cams, *test_cams])
    if not test_cams:
        # No explicit test split: use all for training
        train_cams = [*train_cams, *test_cams]
        test_cams = []
    print(f"  COLMAP scene: {len(train_cams)} train, {len(test_cams)} test, "
          f"{pcd_xyz.shape[0]:,} points, extent={cameras_extent:.2f}")
    return train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent


def load_scene_replica(source_dir, traj_path, images_subdir=''):
    """Load Replica scene from traj_w_c.txt + depth/rgb directories."""
    traj = np.loadtxt(traj_path).reshape(-1, 4, 4)
    rgb_dir = os.path.join(source_dir, "Sequence_1", "rgb")
    if images_subdir:
        rgb_dir = os.path.join(source_dir, images_subdir)

    frames = sorted(os.listdir(rgb_dir))
    assert len(frames) == len(traj), f"Mismatch: {len(frames)} images vs {len(traj)} poses"

    # Assume PINHOLE, must be provided in config
    raise NotImplementedError(
        "Replica scene loading requires intrinsics in config. "
        "Use dataset.type='colmap' with Replica's generated COLMAP format."
    )


def _compute_extent(cams):
    """Compute scene extent from camera centers."""
    centers = []
    for c in cams:
        W2C = np.eye(4)
        W2C[:3, :3] = c.R.T
        W2C[:3, 3] = c.T
        centers.append(np.linalg.inv(W2C)[:3, 3])
    centers = np.array(centers)
    return np.max(np.linalg.norm(centers - centers.mean(0), axis=1)) * 1.1


# ════════════════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════════════════

def pca_colorize(feat_chw):
    """[C,H,W] tensor → [H,W,3] numpy RGB via PCA."""
    if isinstance(feat_chw, torch.Tensor):
        arr = feat_chw.detach().cpu().numpy()
    else:
        arr = feat_chw
    C, H, W = arr.shape
    flat = arr.reshape(C, -1).T  # [HW, C]
    # Simple PCA via SVD (no sklearn dependency)
    flat_centered = flat - flat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(flat_centered, full_matrices=False)
    rgb = U[:, :3] * S[:3]  # [HW, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(H, W, 3)


def visualize_comparison(gaussians, train_cams, cam_to_fid, feat_cache,
                         scale_infos, scales, cfg, iteration, output_dir):
    """Generate multi-channel visualization comparing rendered vs GT."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    vis_frames = cfg['training']['vis_frames']
    longest_edge = cfg['training']['longest_edge']
    bg_color = torch.tensor(
        [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )

    # Pick cameras for visualization
    vis_cams = []
    for idx in vis_frames:
        if idx < len(train_cams):
            vis_cams.append((idx, train_cams[idx]))
    if not vis_cams:
        return

    n_cols = len(vis_cams)
    # Rows: GT, RGB, Depth, Normal, + one per feature scale
    row_labels = ['GT Image', 'Rendered RGB', 'Depth', 'Normal']
    for s in scales:
        row_labels.append(f'GT Feature ({s})')
        row_labels.append(f'Rendered Feature ({s})')
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for j, (cam_idx, cam) in enumerate(vis_cams):
        fid = cam_to_fid.get(cam.uid)

        with torch.no_grad():
            render_pkg = render_rgb_2dgs(gaussians, cam, bg_color, longest_edge)
            rendered_rgb = render_pkg["render"].clamp(0, 1).cpu()  # [3,H,W]
            depth = render_pkg["depth"].cpu()  # [1,H,W]
            rw, rh = render_pkg["width"], render_pkg["height"]

            # Normal
            rend_normal = render_pkg.get("rend_normal")
            if rend_normal is not None:
                rn = rend_normal.squeeze(0).permute(2, 0, 1)  # [3,H,W]
                normal_vis = (rn * 0.5 + 0.5).clamp(0, 1).cpu()
            else:
                normal_vis = torch.zeros(3, rh, rw)

        # GT image
        gt_image = load_image_tensor(cam)
        gt_resized = F.interpolate(
            gt_image.unsqueeze(0), size=(rh, rw),
            mode="bilinear", align_corners=False
        ).squeeze(0).cpu()

        # Row 0: GT
        axes[0, j].imshow(gt_resized.permute(1, 2, 0).numpy())
        axes[0, j].set_title(f'{cam.image_name}', fontsize=9)
        axes[0, j].axis('off')

        # Row 1: Rendered RGB
        axes[1, j].imshow(rendered_rgb.permute(1, 2, 0).numpy())
        psnr_val = -10 * np.log10(((rendered_rgb - gt_resized) ** 2).mean().item() + 1e-10)
        axes[1, j].set_title(f'PSNR={psnr_val:.1f}dB', fontsize=9)
        axes[1, j].axis('off')

        # Row 2: Depth
        d = depth.squeeze().numpy()
        valid = d > 0
        if valid.any():
            vmin, vmax = np.percentile(d[valid], [2, 98])
        else:
            vmin, vmax = 0, 1
        axes[2, j].imshow(d, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[2, j].axis('off')

        # Row 3: Normal
        axes[3, j].imshow(normal_vis.permute(1, 2, 0).numpy())
        axes[3, j].axis('off')

        # Feature rows
        row_offset = 4
        viewmat = cam.get_world_view_transform()
        for si, scale in enumerate(scales):
            gt_row = row_offset + si * 2
            rend_row = row_offset + si * 2 + 1

            # GT Feature PCA
            if fid is not None:
                gt_feat = feat_cache.get(scale, fid)
                if gt_feat is not None:
                    axes[gt_row, j].imshow(pca_colorize(gt_feat.cpu()))
            axes[gt_row, j].axis('off')

            # Rendered Feature PCA
            with torch.no_grad():
                si_info = scale_infos[scale]
                feat_colors = gaussians.get_feature(scale)
                rendered_feat = render_features_2dgs(
                    gaussians, viewmat, feat_colors,
                    si_info['h'], si_info['w'], si_info['K']
                )
            axes[rend_row, j].imshow(pca_colorize(rendered_feat.cpu()))
            axes[rend_row, j].axis('off')

    # Row labels
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=15)

    fig.suptitle(f'{cfg["exp_name"]} — Iter {iteration}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f'vis_iter{iteration:06d}.png')
    fig.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {vis_path}")
    return vis_path


def evaluate_psnr(gaussians, cams, bg_color, longest_edge, max_eval=50):
    """Compute mean PSNR over a set of cameras."""
    psnrs = []
    step = max(1, len(cams) // max_eval)
    for cam in cams[::step]:
        with torch.no_grad():
            render_pkg = render_rgb_2dgs(gaussians, cam, bg_color, longest_edge)
            image = render_pkg["render"].clamp(0, 1)
            rw, rh = render_pkg["width"], render_pkg["height"]
            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(
                gt_image.unsqueeze(0), size=(rh, rw),
                mode="bilinear", align_corners=False
            ).squeeze(0)
            mse = ((image - gt_image) ** 2).mean().item()
            psnrs.append(-10 * np.log10(max(mse, 1e-10)))
    return np.mean(psnrs) if psnrs else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def train(cfg):
    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']

    print(f"\n{'='*70}")
    print(f"  Joint 2DGS Geometry + Feature Training v2")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*70}")
    print(f"  Dataset type:   {dcfg['type']}")
    print(f"  Source:          {dcfg['source_dir']}")
    print(f"  Features:        {dcfg['feature_dir']}")
    print(f"  Scales:          {dcfg['feature_scales']}")
    print(f"  Iterations:      {tcfg['iterations']}")
    le = tcfg['longest_edge']
    print(f"  Resolution:      {'FULL' if le == 0 else f'longest_edge={le}'}")
    print(f"  Feature weight:  {tcfg['feature_weight']}")
    print(f"  Losses:          L1+SSIM(λ={tcfg['lambda_dssim']}) + "
          f"Normal(λ={tcfg['lambda_normal']}) + "
          f"Dist(λ={tcfg['lambda_dist']}) + "
          f"Scale(λ={tcfg['lambda_scale']})")
    print(f"  Output:          {output_dir}")
    print(f"{'='*70}\n")

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    ds_type = dcfg['type']
    if ds_type in ('colmap', 'cambridge'):
        train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
            load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))
    elif ds_type == 'replica':
        raise NotImplementedError("Use type='colmap' with Replica COLMAP format")
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    # ── 2. Load features ──
    scales = dcfg['feature_scales']
    if isinstance(scales, str):
        scales = [s.strip() for s in scales.split(',')]
    print(f"\nLoading features ({scales})...")
    feat_cache = DA3FeatureCache(dcfg['feature_dir'], scales)

    # Map cameras to feature frame IDs
    # For Cambridge: seq dirs are directly in source_dir
    # For COLMAP/Replica: might be in images/ subdir
    images_subdir = dcfg.get('images', '')
    if images_subdir:
        images_dir = os.path.join(dcfg['source_dir'], images_subdir)
    else:
        # Auto-detect: check if source_dir has seq* (Cambridge) or images/ (COLMAP)
        import glob
        if glob.glob(os.path.join(dcfg['source_dir'], 'seq*')):
            images_dir = dcfg['source_dir']
        elif os.path.isdir(os.path.join(dcfg['source_dir'], 'images')):
            images_dir = os.path.join(dcfg['source_dir'], 'images')
        else:
            images_dir = dcfg['source_dir']
    da3_name_to_fid = build_da3_image_order(images_dir)
    available_fids = feat_cache.frame_ids(scales[0])

    cam_to_fid = {}
    for cam in train_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in available_fids:
            cam_to_fid[cam.uid] = fid
    print(f"  Matched {len(cam_to_fid)}/{len(train_cams)} cameras to features")
    if len(cam_to_fid) == 0:
        print("ERROR: No cameras matched to features!")
        return

    # ── 3. Feature scale info (intrinsics at feature resolution) ──
    scale_infos = {}
    cam0 = train_cams[0]
    tanfovx = math.tan(cam0.FovX * 0.5)
    tanfovy = math.tan(cam0.FovY * 0.5)
    img_fx = cam0.width / (2 * tanfovx)
    img_fy = cam0.height / (2 * tanfovy)

    for scale in scales:
        dim, h, w = feat_cache.info(scale)
        sx, sy = w / cam0.width, h / cam0.height
        feat_K = torch.tensor([
            [img_fx * sx, 0, cam0.width * sx / 2.0],
            [0, img_fy * sy, cam0.height * sy / 2.0],
            [0, 0, 1],
        ], device="cuda", dtype=torch.float32)
        scale_infos[scale] = {'dim': dim, 'h': h, 'w': w, 'K': feat_K}
        print(f"  [{scale}] {dim}d @ {w}×{h}")

    # ── 4. Create model ──
    print("\nInitializing model...")
    feature_dims = {s: scale_infos[s]['dim'] for s in scales}
    gaussians = GaussianModel2DGSJoint(
        sh_degree=mcfg['sh_degree'],
        feature_scales=feature_dims,
    )
    gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)

    # Build args namespace for training_setup
    train_args = argparse.Namespace(**{
        'position_lr_init': tcfg['position_lr_init'],
        'position_lr_final': tcfg['position_lr_final'],
        'feature_lr': tcfg['feature_lr'],
        'opacity_lr': tcfg['opacity_lr'],
        'scaling_lr': tcfg['scaling_lr'],
        'rotation_lr': tcfg['rotation_lr'],
        'percent_dense': tcfg['percent_dense'],
        'feature_embedding_lr': tcfg['feature_embedding_lr'],
        'iterations': tcfg['iterations'],
    })
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg['white_background'] else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )
    random_bg = mcfg.get('random_background', False)

    # ── 5. Pre-cache mono depth ──
    mono_depth_dir = tcfg.get('mono_depth_dir', '')
    mono_depth_cache = {}
    if mono_depth_dir and os.path.isdir(mono_depth_dir):
        print(f"Pre-caching monocular depth from {mono_depth_dir}...")
        for cam in train_cams:
            depth_name = os.path.splitext(cam.image_name)[0] + ".npy"
            depth_path = os.path.join(mono_depth_dir, depth_name)
            if os.path.exists(depth_path):
                d = np.load(depth_path)  # [H,W] float32, inverse disparity 0-1
                mono_depth_cache[cam.image_name] = 1.0 - d  # invert: now same dir as rendered
        print(f"  Cached {len(mono_depth_cache)} depth maps")
    else:
        if tcfg.get('lambda_depth', 0) > 0:
            print(f"  WARNING: lambda_depth={tcfg['lambda_depth']} but mono_depth_dir not found")

    # ── 6. Pre-cache images ──
    print("Pre-caching images...")
    from PIL import Image as PILImage
    for cam in [*train_cams, *test_cams]:
        if isinstance(cam.image, str) and os.path.exists(cam.image):
            cam._cached_np = np.array(
                PILImage.open(cam.image).convert("RGB"), dtype=np.uint8
            )

    # ── 6. Training loop ──
    iterations = tcfg['iterations']
    feat_start = tcfg['feature_start_iter']
    feat_weight = tcfg['feature_weight']
    feat_cos_w = tcfg['feature_cos_weight']
    feat_detach = tcfg['detach_feat_geometry']
    lambda_dssim = tcfg['lambda_dssim']
    lambda_normal = tcfg['lambda_normal']
    lambda_dist = tcfg['lambda_dist']
    lambda_scale = tcfg['lambda_scale']
    reg_start = tcfg['reg_start_iter']
    longest_edge = tcfg['longest_edge']

    viewpoint_stack = []
    ema_loss = 0.0
    ema_rgb = 0.0
    ema_depth = 0.0
    ema_feat = {s: 0.0 for s in scales}
    best_psnr = 0.0

    log_file = open(os.path.join(output_dir, 'train.log'), 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    pbar = tqdm(range(1, iterations + 1), desc="Joint v2")
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Sample camera
        if not viewpoint_stack:
            viewpoint_stack = list(train_cams)
            shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop()
        fid = cam_to_fid.get(cam.uid)

        # ── Random background (helps with alpha/boundary learning) ──
        if random_bg:
            bg = torch.rand(3, device="cuda")
        else:
            bg = bg_color

        # ── RGB render ──
        render_pkg = render_rgb_2dgs(gaussians, cam, bg, longest_edge)
        image = render_pkg["render"]
        rw, rh = render_pkg["width"], render_pkg["height"]

        gt_image = load_image_tensor(cam)
        gt_image = F.interpolate(
            gt_image.unsqueeze(0), size=(rh, rw),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # ── RGB loss: L1 + SSIM ──
        Ll1 = F.l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        rgb_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
        loss = rgb_loss

        # ── 2DGS geometry regularization ──
        if iteration > reg_start:
            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            rend_alpha = render_pkg["rend_alpha"]

            # Normal consistency
            if lambda_normal > 0 and rend_normal is not None and surf_normal is not None:
                surf_n = surf_normal * rend_alpha.squeeze(0).detach()
                rend_n = rend_normal.squeeze(0).permute(2, 0, 1)
                if len(surf_n.shape) == 4:
                    surf_n = surf_n.squeeze(0)
                surf_n = surf_n.permute(2, 0, 1)
                normal_error = (1 - (rend_n * surf_n).sum(dim=0))[None]
                loss = loss + lambda_normal * normal_error.mean()

            # Distortion
            if lambda_dist > 0 and rend_dist is not None:
                loss = loss + lambda_dist * rend_dist.squeeze(-1).mean()

            # Scale regularization
            if lambda_scale > 0:
                log_threshold = math.log(max(tcfg['scale_reg_threshold'], 1e-6))
                excess = torch.clamp(
                    gaussians._scaling.max(dim=1).values - log_threshold, min=0
                )
                loss = loss + lambda_scale * (excess ** 2).mean()

        # ── Monocular depth supervision (Pearson correlation) ──
        ema_depth = getattr(train, '_ema_depth', 0.0)  # will init below
        depth_loss_val = torch.tensor(0.0, device="cuda")
        lambda_depth_cfg = tcfg.get('lambda_depth', 0.0)
        depth_start = tcfg.get('depth_start_iter', 3000)
        depth_warmup = tcfg.get('depth_warmup_iters', 2000)
        if lambda_depth_cfg > 0 and mono_depth_cache and iteration > depth_start:
            warmup_progress = min(1.0, (iteration - depth_start) / max(1, depth_warmup))
            lambda_depth_now = lambda_depth_cfg * warmup_progress
            rendered_depth = render_pkg["depth"]  # [1, H, W]
            if cam.image_name in mono_depth_cache:
                mono_d = torch.from_numpy(mono_depth_cache[cam.image_name]).float().cuda()
                if mono_d.shape[0] != rh or mono_d.shape[1] != rw:
                    mono_d = F.interpolate(
                        mono_d[None, None], size=(rh, rw),
                        mode="bilinear", align_corners=False
                    ).squeeze()
                depth_loss_val = lambda_depth_now * pearson_depth_loss(rendered_depth, mono_d)
                loss = loss + depth_loss_val

        # ── Feature losses ──
        total_feat_loss = torch.tensor(0.0, device="cuda")
        if fid is not None and iteration >= feat_start:
            viewmat = cam.get_world_view_transform()
            for scale in scales:
                gt_feat = feat_cache.get(scale, fid)
                if gt_feat is None:
                    continue
                si = scale_infos[scale]
                feat_colors = gaussians.get_feature(scale)

                if feat_detach:
                    det_means = gaussians.get_xyz.detach()
                    det_opacity = gaussians.get_opacity.detach()
                    det_scales2d = gaussians.get_scaling.detach()
                    det_rots = gaussians.get_rotation.detach()
                    det_scales3 = torch.cat([
                        det_scales2d,
                        torch.ones(det_scales2d.shape[0], 1, device="cuda"),
                    ], dim=-1)
                    D = feat_colors.shape[1]
                    chunks = []
                    for ci in range((D + 31) // 32):
                        cs, ce = ci * 32, min((ci + 1) * 32, D)
                        rc, *_ = rasterization_2dgs(
                            means=det_means, quats=det_rots,
                            scales=det_scales3,
                            opacities=det_opacity.squeeze(-1),
                            colors=feat_colors[:, cs:ce],
                            viewmats=viewmat[None], Ks=si['K'][None],
                            width=si['w'], height=si['h'],
                            packed=False, near_plane=0.01,
                            far_plane=500, render_mode='RGB',
                        )
                        chunks.append(rc)
                    fm = torch.cat(chunks, dim=-1)[0].permute(2, 0, 1)
                    rendered_feat = F.normalize(fm, p=2, dim=0)
                else:
                    rendered_feat = render_features_2dgs(
                        gaussians, viewmat, feat_colors,
                        si['h'], si['w'], si['K'],
                    )

                # L1 + cosine loss
                feat_l1 = F.l1_loss(rendered_feat, gt_feat)
                feat_cos = 1.0 - F.cosine_similarity(
                    rendered_feat, gt_feat, dim=0
                ).mean()
                scale_loss = feat_l1 + feat_cos_w * feat_cos
                total_feat_loss = total_feat_loss + scale_loss
                ema_feat[scale] = 0.9 * ema_feat[scale] + 0.1 * scale_loss.item()

            loss = loss + feat_weight * total_feat_loss

        # ── Safety check ──
        loss_val = loss.item()
        if torch.isnan(loss) or torch.isinf(loss) or loss_val < -0.01:
            tqdm.write(f"  [Iter {iteration}] Bad loss={loss_val:.4g}, skipping")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue
        if loss_val > 10.0:
            loss = loss.clamp(max=10.0)

        # ── Backward + gradient clipping ──
        loss.backward()
        clip_params = [
            gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
            gaussians._scaling, gaussians._rotation, gaussians._opacity,
        ]
        for sn in scales:
            clip_params.append(getattr(gaussians, f'_feat_{sn}'))
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

        with torch.no_grad():
            # Densification stats
            if iteration < tcfg['densify_until_iter']:
                vp = render_pkg["viewspace_points"]
                grad_data = vp.grad if vp.grad is not None else vp
                radii = render_pkg["radii"]
                vis = render_pkg["visibility_filter"]
                gaussians.max_radii2D[vis] = torch.max(
                    gaussians.max_radii2D[vis], radii[vis]
                )
                gaussians.add_densification_stats(grad_data, vis, rw, rh)

            # EMA
            ema_loss = 0.4 * loss_val + 0.6 * ema_loss
            ema_rgb = 0.4 * rgb_loss.item() + 0.6 * ema_rgb
            depth_v = depth_loss_val.item() if torch.is_tensor(depth_loss_val) else depth_loss_val
            ema_depth = 0.4 * depth_v + 0.6 * ema_depth

            if iteration % 10 == 0:
                feat_str = " ".join(f"{s[0]}={ema_feat[s]:.3f}" for s in scales)
                pbar.set_postfix({
                    "L": f"{ema_loss:.4f}", "RGB": f"{ema_rgb:.4f}",
                    "D": f"{ema_depth:.4f}",
                    "F": feat_str, "N": f"{gaussians.num_points:,}",
                })

            # ── Periodic eval + visualization ──
            eval_interval = tcfg['eval_interval']
            if iteration % eval_interval == 0 or iteration == iterations:
                eval_cams = test_cams if test_cams else train_cams
                psnr_val = evaluate_psnr(
                    gaussians, eval_cams, bg_color, longest_edge
                )
                feat_str = " | ".join(f"{s}={ema_feat[s]:.4f}" for s in scales)
                msg = (f"  [Iter {iteration}] PSNR={psnr_val:.2f}dB | "
                       f"RGB={ema_rgb:.4f} | Depth={ema_depth:.4f} | "
                       f"Feat=[{feat_str}] | "
                       f"N={gaussians.num_points:,}")
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    msg += " ★ BEST"
                    # Save best
                    best_dir = os.path.join(output_dir, "point_cloud", "best")
                    gaussians.save_ply(os.path.join(best_dir, "point_cloud.ply"))
                    gaussians.save_features(
                        os.path.join(output_dir, "features_best")
                    )
                log(msg)

                # Visualization
                vis_dir = os.path.join(output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                visualize_comparison(
                    gaussians, train_cams, cam_to_fid, feat_cache,
                    scale_infos, scales, cfg, iteration, vis_dir
                )

            # ── Checkpoints ──
            save_interval = tcfg['save_interval']
            if iteration % save_interval == 0 or iteration == iterations:
                save_dir = os.path.join(
                    output_dir, "point_cloud", f"iteration_{iteration}"
                )
                gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
                gaussians.save_features(
                    os.path.join(output_dir, "features"), iteration
                )

            # ── Densification ──
            if iteration < tcfg['densify_until_iter']:
                if (iteration > tcfg['densify_from_iter'] and
                        iteration % tcfg['densification_interval'] == 0):
                    size_threshold = (
                        20 if iteration > tcfg['opacity_reset_interval']
                        else None
                    )
                    gaussians.densify_and_prune(
                        tcfg['densify_grad_threshold'], 0.005,
                        cameras_extent, size_threshold,
                    )
                if iteration % tcfg['opacity_reset_interval'] == 0:
                    gaussians.reset_opacity(
                        reset_value=tcfg['opacity_reset_value']
                    )

            # ── Optimizer step ──
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        if iteration % 1000 == 1:
            feat_detail = " | ".join(
                f"{s}={ema_feat[s]:.4f}" for s in scales
            )
            log(f"  [Iter {iteration}] "
                f"RGB={ema_rgb:.4f} Depth={ema_depth:.4f} Feat=[{feat_detail}] "
                f"Total={ema_loss:.4f} N={gaussians.num_points:,}")

    # ── Final ──
    log(f"\n  Training complete. Best PSNR: {best_psnr:.2f}dB")
    log(f"  Output: {output_dir}")
    log_file.close()


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='YAML configuration file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
