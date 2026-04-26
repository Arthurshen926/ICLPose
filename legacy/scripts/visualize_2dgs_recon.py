#!/usr/bin/env python3
"""
Visualize 2DGS reconstruction quality.
Load trained 2DGS model and render RGB / depth / normal images for visual inspection.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
from plyfile import PlyData
from PIL import Image


def load_ply_2dgs(ply_path):
    """Load 2DGS model from PLY file."""
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    
    xyz = np.stack([el["x"], el["y"], el["z"]], axis=1)
    opacities = el["opacity"][..., np.newaxis]
    
    # SH features
    f_dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1)  # [N, 3]
    
    rest_names = sorted(
        [p.name for p in el.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1])
    )
    if rest_names:
        f_rest = np.stack([el[n] for n in rest_names], axis=1)  # [N, F]
    else:
        f_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)
    
    # Scales
    scale_names = sorted(
        [p.name for p in el.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1])
    )
    scales = np.stack([el[n] for n in scale_names], axis=1)  # [N, 2] for 2DGS
    
    # Rotations
    rot_names = sorted(
        [p.name for p in el.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1])
    )
    rots = np.stack([el[n] for n in rot_names], axis=1)  # [N, 4]
    
    n_gaussians = xyz.shape[0]
    n_sh_coeffs = f_dc.shape[1] + (f_rest.shape[1] if len(rest_names) > 0 else 0)
    sh_degree = int(math.sqrt(n_sh_coeffs // 3)) - 1 if n_sh_coeffs > 3 else 0
    
    print(f"  Loaded {n_gaussians:,} Gaussians, scales={scales.shape[1]}D, sh_degree={sh_degree}")
    
    # Build SH features: [N, K, 3] where K = (sh_degree+1)^2
    K = (sh_degree + 1) ** 2
    features = np.zeros((n_gaussians, K, 3), dtype=np.float32)
    features[:, 0, :] = f_dc  # DC component
    if f_rest.shape[1] > 0:
        f_rest_reshaped = f_rest.reshape(n_gaussians, 3, K - 1).transpose(0, 2, 1)  # [N, K-1, 3]
        features[:, 1:, :] = f_rest_reshaped
    
    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(opacities, dtype=torch.float32, device="cuda").squeeze(-1),
        "features": torch.tensor(features, dtype=torch.float32, device="cuda"),
        "scales": torch.tensor(scales, dtype=torch.float32, device="cuda"),
        "rotations": torch.tensor(rots, dtype=torch.float32, device="cuda"),
        "sh_degree": sh_degree,
        "is_2dgs": scales.shape[1] == 2,
    }


def load_cameras_json(json_path, num_cameras=None):
    """Load camera parameters from cameras.json."""
    with open(json_path) as f:
        cams = json.load(f)
    if num_cameras:
        # Evenly sample cameras
        step = max(1, len(cams) // num_cameras)
        cams = cams[::step][:num_cameras]
    return cams


def cam_to_viewmat(cam):
    """Convert camera dict (position + rotation) to 4x4 world-to-camera matrix.
    
    cameras.json stores:
      - rotation: C2W (camera-to-world) rotation matrix [3,3]
      - position: camera center in world coordinates [3]
    We need to reconstruct the W2C (world-to-camera) matrix.
    """
    R_c2w = np.array(cam["rotation"], dtype=np.float32)  # [3, 3], camera-to-world
    pos = np.array(cam["position"], dtype=np.float32)    # [3], camera center in world
    R_w2c = R_c2w.T                # W2C rotation = transpose of C2W
    t_w2c = -R_w2c @ pos           # W2C translation
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R_w2c
    viewmat[:3, 3] = t_w2c
    return viewmat


def render_2dgs(model, viewmat, K, width, height):
    """Render 2DGS model from a given viewpoint."""
    from gsplat import rasterization_2dgs
    
    scales = model["scales"]
    # Pad 2D scales to 3D for rasterization_2dgs
    if model["is_2dgs"]:
        scales = torch.cat([
            scales,
            torch.ones(scales.shape[0], 1, device=scales.device, dtype=scales.dtype)
        ], dim=-1)
    
    opacity = torch.sigmoid(model["opacities"])
    
    bg_color = torch.zeros(4, device="cuda")  # black background for RGB+ED
    
    colors, alphas, normals, surf_normals, distort, median_depth, info = (
        rasterization_2dgs(
            means=model["xyz"],
            quats=torch.nn.functional.normalize(model["rotations"], dim=-1),
            scales=torch.exp(scales),
            opacities=opacity,
            colors=model["features"],
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(width),
            height=int(height),
            packed=False,
            sh_degree=model["sh_degree"],
            render_mode="RGB+ED",
            backgrounds=bg_color[None],
            near_plane=0.01,
            far_plane=100.0,
        )
    )
    
    # colors: [1, H, W, 4] (RGB + expected depth)
    rendered = colors[0]  # [H, W, 4]
    rgb = rendered[:, :, :3]  # [H, W, 3]
    depth = rendered[:, :, 3]  # [H, W]
    alpha = alphas[0, :, :, 0]  # [H, W]
    
    # normals: surf_normals has shape [H, W, 3] (no batch dim)
    normal = None
    if surf_normals is not None and surf_normals.numel() > 0:
        if surf_normals.dim() == 4:
            normal = surf_normals[0]  # [1, H, W, 3] -> [H, W, 3]
        elif surf_normals.dim() == 3:
            normal = surf_normals      # already [H, W, 3]
    
    return rgb, depth, alpha, normal


def colorize_depth(depth, alpha=None, vmin=None, vmax=None):
    """Convert depth map to colored image using turbo colormap."""
    depth_np = depth.cpu().numpy()
    if alpha is not None:
        valid = alpha.cpu().numpy() > 0.5
    else:
        valid = depth_np > 0
    
    if valid.sum() == 0:
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
    
    if vmin is None:
        vmin = depth_np[valid].min()
    if vmax is None:
        vmax = depth_np[valid].max()
    
    depth_norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Turbo colormap
    import matplotlib.pyplot as plt
    cmap = plt.cm.turbo
    depth_colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # Mask out invalid regions
    depth_colored[~valid] = 0
    
    return depth_colored


def colorize_normal(normal, alpha=None):
    """Convert normal map to RGB visualization."""
    normal_np = normal.cpu().numpy()  # [H, W, 3]
    normal_vis = (normal_np * 0.5 + 0.5)
    normal_vis = np.clip(normal_vis, 0, 1)
    normal_vis = (normal_vis * 255).astype(np.uint8)
    
    if alpha is not None:
        valid = alpha.cpu().numpy() > 0.5
        normal_vis[~valid] = 0
    
    return normal_vis


def main():
    parser = argparse.ArgumentParser(description="Visualize 2DGS reconstruction")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to 2dgs_model directory")
    parser.add_argument("--iteration", type=int, default=7000,
                        help="Which iteration checkpoint to load")
    parser.add_argument("--num_views", type=int, default=10,
                        help="Number of views to render")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for rendered images")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Source data directory (for GT images)")
    args = parser.parse_args()
    
    # Paths
    ply_path = os.path.join(args.model_dir, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    cam_path = os.path.join(args.model_dir, "cameras.json")
    
    if not os.path.exists(ply_path):
        print(f"ERROR: Checkpoint not found: {ply_path}")
        # List available checkpoints
        pc_dir = os.path.join(args.model_dir, "point_cloud")
        if os.path.exists(pc_dir):
            iters = os.listdir(pc_dir)
            print(f"Available: {iters}")
        else:
            print("No point_cloud directory found. Training may not have saved any checkpoints yet.")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, f"vis_iter_{args.iteration}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Infer source_dir from cfg_args
    if args.source_dir is None:
        cfg_path = os.path.join(args.model_dir, "cfg_args")
        if os.path.exists(cfg_path):
            import ast
            with open(cfg_path) as f:
                text = f.read()
            # Parse source_path from Namespace string
            idx = text.find("source_path='")
            if idx >= 0:
                start = idx + len("source_path='")
                end = text.index("'", start)
                args.source_dir = text[start:end]
    
    print(f"Loading model from: {ply_path}")
    model = load_ply_2dgs(ply_path)
    
    print(f"Loading cameras from: {cam_path}")
    cameras = load_cameras_json(cam_path, num_cameras=args.num_views)
    print(f"  Rendering {len(cameras)} views")
    
    # Render each view
    for i, cam in enumerate(cameras):
        width = cam["width"]
        height = cam["height"]
        fx = cam["fx"]
        fy = cam["fy"]
        
        viewmat = torch.tensor(cam_to_viewmat(cam), device="cuda")
        K = torch.tensor([
            [fx, 0, width / 2.0],
            [0, fy, height / 2.0],
            [0, 0, 1],
        ], device="cuda")
        
        img_name = cam.get("img_name", f"view_{i:04d}")
        safe_name = img_name.replace("/", "_").replace("\\", "_")
        base_name = os.path.splitext(safe_name)[0]
        
        print(f"  [{i+1}/{len(cameras)}] Rendering {img_name}...")
        
        with torch.no_grad():
            rgb, depth, alpha, normal = render_2dgs(model, viewmat, K, width, height)
        
        # Save RGB
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(os.path.join(args.output_dir, f"{base_name}_rgb.png"))
        
        # Save depth
        depth_colored = colorize_depth(depth, alpha)
        Image.fromarray(depth_colored).save(os.path.join(args.output_dir, f"{base_name}_depth.png"))
        
        # Save normal
        if normal is not None:
            normal_colored = colorize_normal(normal, alpha)
            Image.fromarray(normal_colored).save(os.path.join(args.output_dir, f"{base_name}_normal.png"))
        
        # Save alpha
        alpha_np = (alpha.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha_np).save(os.path.join(args.output_dir, f"{base_name}_alpha.png"))
        
        # Load and save GT image for side-by-side comparison
        if args.source_dir:
            gt_path = os.path.join(args.source_dir, img_name)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).resize((width, height))
                gt_np = np.array(gt_img)[:, :, :3]  # Remove alpha if present
                
                # Side-by-side: GT | Rendered RGB
                comparison = np.concatenate([gt_np, rgb_np], axis=1)
                Image.fromarray(comparison).save(
                    os.path.join(args.output_dir, f"{base_name}_compare.png"))
    
    print(f"\nVisualization saved to: {args.output_dir}")
    print(f"  Total views: {len(cameras)}")
    
    # Create a summary grid  
    try:
        from PIL import Image as PILImage
        compare_files = sorted([
            f for f in os.listdir(args.output_dir) if f.endswith("_compare.png")
        ])
        if compare_files:
            imgs = [PILImage.open(os.path.join(args.output_dir, f)) for f in compare_files[:6]]
            # Create 2x3 grid or 3x2 grid
            n = len(imgs)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            w, h = imgs[0].size
            grid = PILImage.new("RGB", (w * cols, h * rows))
            for idx, img in enumerate(imgs):
                r, c = idx // cols, idx % cols
                grid.paste(img, (c * w, r * h))
            grid.save(os.path.join(args.output_dir, "grid_comparison.png"))
            print(f"  Grid comparison saved: grid_comparison.png")
    except Exception as e:
        print(f"  Grid creation failed: {e}")


if __name__ == "__main__":
    main()
