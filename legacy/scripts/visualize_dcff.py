#!/usr/bin/env python3
"""
Visualize DCFF (Deferred Cascaded Feature Field) rendered features.

Generates side-by-side comparisons:
  Row 1: RGB | Depth | Alpha
  Row 2: Z_map (16d latent, PCA→3) | Fine features (64d, PCA→3) | GT fine_geo
  Row 3: Coarse features (64d, PCA→3) | GT coarse_sem | Direct RADIO embedding
  
Also prints per-image cosine similarity metrics.

Usage:
    python scripts/visualize_dcff.py --config configs/dcff_oldhospital.yaml \
        --checkpoint output/dcff_oldhospital/checkpoints/latest.pth \
        --num_views 10 --output_dir output/dcff_oldhospital/vis
"""

import argparse
import math
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcff.hybrid_gaussian import HybridGaussianModel
from dcff.hash_grid import SpatialHashGrid
from dcff.deferred_renderer import DeferredCascadedRenderer
from feature_3dgs.train_2dgs_joint_v2 import load_scene_colmap
from feature_3dgs.train_2dgs_joint import build_da3_image_order


def cam_to_viewmat(cam, device='cuda'):
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    return torch.tensor(W2C, dtype=torch.float32, device=device)


def cam_to_K(cam, width, height, device='cuda'):
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    return torch.tensor([
        [fx, 0, width / 2.0],
        [0, fy, height / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)


def pca_colorize(feat, n_components=3, mask=None):
    """PCA→RGB colorization for a single feature map, optionally masked."""
    C, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).T.float()
    mask_flat = None if mask is None else mask.reshape(-1)
    fit_pixels = feat_flat if mask_flat is None else feat_flat[mask_flat]

    if fit_pixels.numel() == 0:
        return torch.zeros(n_components, H, W, device=feat.device)

    mean = fit_pixels.mean(dim=0, keepdim=True)
    centered = fit_pixels - mean

    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        basis = Vh[:n_components]
        proj = (feat_flat - mean) @ basis.T
    except Exception:
        proj = feat_flat[:, :n_components]

    valid_proj = proj if mask_flat is None else proj[mask_flat]
    for i in range(n_components):
        vmin, vmax = valid_proj[:, i].min(), valid_proj[:, i].max()
        if vmax > vmin:
            proj[:, i] = (proj[:, i] - vmin) / (vmax - vmin)
        else:
            proj[:, i] = 0.5

    img = proj.T.reshape(n_components, H, W)
    if mask is not None:
        img = img * mask.unsqueeze(0)
    return img


def joint_pca_colorize(feats, n_components=3, mask=None):
    """Colorize multiple feature maps with the same PCA basis for fair comparison."""
    valid_feats = [feat for feat in feats if feat is not None]
    if not valid_feats:
        return [None for _ in feats]

    C, H, W = valid_feats[0].shape
    mask_flat = None if mask is None else mask.reshape(-1)
    stacked = []
    for feat in valid_feats:
        feat_flat = feat.reshape(C, -1).T.float()
        feat_flat = feat_flat if mask_flat is None else feat_flat[mask_flat]
        if feat_flat.numel() > 0:
            stacked.append(feat_flat)

    if not stacked:
        zeros = torch.zeros(n_components, H, W, device=valid_feats[0].device)
        return [zeros.clone() if feat is not None else None for feat in feats]

    stacked = torch.cat(stacked, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean

    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        basis = Vh[:n_components]
    except Exception:
        basis = None

    all_proj = []
    all_valid_proj = []
    for feat in feats:
        if feat is None:
            all_proj.append(None)
            continue
        feat_flat = feat.reshape(C, -1).T.float()
        if basis is None:
            proj = feat_flat[:, :n_components]
        else:
            proj = (feat_flat - mean) @ basis.T
        all_proj.append(proj)
        all_valid_proj.append(proj if mask_flat is None else proj[mask_flat])

    stacked_proj = torch.cat(all_valid_proj, dim=0)
    mins = stacked_proj.min(dim=0).values
    maxs = stacked_proj.max(dim=0).values

    outputs = []
    for feat, proj in zip(feats, all_proj):
        if feat is None:
            outputs.append(None)
            continue
        proj = proj.clone()
        for i in range(n_components):
            if maxs[i] > mins[i]:
                proj[:, i] = (proj[:, i] - mins[i]) / (maxs[i] - mins[i])
            else:
                proj[:, i] = 0.5
        img = proj.T.reshape(n_components, H, W)
        if mask is not None:
            img = img * mask.unsqueeze(0)
        outputs.append(img)
    return outputs


def cosine_sim_map(pred, gt):
    """Per-pixel cosine similarity.
    
    Args:
        pred, gt: [C, H, W]
    Returns:
        sim: [H, W] cosine similarity in [-1, 1]
        mean_sim: scalar
    """
    pred_norm = F.normalize(pred, dim=0, eps=1e-6)
    gt_norm = F.normalize(gt, dim=0, eps=1e-6)
    sim = (pred_norm * gt_norm).sum(dim=0)  # [H, W]
    return sim, sim.mean().item()


def to_numpy_img(tensor_chw):
    """[C, H, W] tensor → [H, W, C] numpy uint8."""
    img = tensor_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def sim_to_heatmap(sim_hw):
    """[H, W] similarity → [H, W, 3] RGB heatmap (blue=low, red=high)."""
    sim = sim_hw.detach().cpu().numpy()
    sim = np.clip(sim, 0, 1)
    # Blue(0) → Cyan(0.25) → Green(0.5) → Yellow(0.75) → Red(1.0)
    r = np.clip(2 * sim - 1, 0, 1)
    g = np.where(sim < 0.5, 2 * sim, 2 * (1 - sim))
    b = np.clip(1 - 2 * sim, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def load_dcff_model(config_path, checkpoint_path):
    """Load DCFF model from config + checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    mcfg = cfg['model']
    hcfg = cfg['hash_grid']
    fcfg = cfg.get('fine_decoder', {})
    
    # Load scene
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(cfg['dataset']['source_dir'], cfg['dataset'].get('images', ''))
    
    # Create Gaussian model + load PLY
    gaussians = HybridGaussianModel(
        sh_degree=mcfg['sh_degree'],
        latent_dim=mcfg['latent_dim'],
    )
    ply_path = checkpoint_path.replace('.pth', '.ply')
    if not os.path.exists(ply_path):
        ply_path = os.path.join(os.path.dirname(checkpoint_path), 'latest.ply')
    gaussians.load_ply(ply_path)
    # Parameters are already on CUDA from load_ply
    
    # Compute scene extent
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scene_extent = float(np.percentile(np.linalg.norm(xyz, axis=1), 99)) * 1.2
    
    # Create hash grid
    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent,
        feature_dim=mcfg['feature_dim'],
        input_mode=hcfg.get('input_mode', 'legacy'),
        latent_dim=mcfg['latent_dim'],
        scale_dim=hcfg.get('scale_dim', 2),
        scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
        n_levels=hcfg['n_levels'],
        n_features_per_level=hcfg['n_features_per_level'],
        log2_hashmap_size=hcfg['log2_hashmap_size'],
        base_resolution=hcfg['base_resolution'],
        max_resolution=hcfg['max_resolution'],
        sh_degree=hcfg.get('sh_degree', 3),
        mlp_hidden=hcfg.get('mlp_hidden', 128),
        mlp_layers=hcfg.get('mlp_layers', 2),
    ).cuda()
    
    # Create renderer
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=mcfg['latent_dim'],
        fine_feature_dim=mcfg['feature_dim'],
        coarse_feature_dim=mcfg['feature_dim'],
        fine_hidden_dim=fcfg.get('hidden_dim'),
        fine_num_layers=fcfg.get('num_layers', 3),
        fine_use_viewdirs=fcfg.get('use_viewdirs', False),
        fine_view_degree=fcfg.get('view_degree', 2),
    ).cuda()
    
    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    hash_grid.load_state_dict(ckpt['hash_grid_state'])
    renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
    
    print(f"[DCFF] Loaded from {checkpoint_path}")
    print(f"  Iteration: {ckpt['iteration']}, Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"  Scene extent: {scene_extent:.2f}")
    
    return cfg, gaussians, renderer, train_cams, test_cams


def load_reference_model(ply_path):
    """Load direct-embedding GaussianFeatureModel for comparison.
    
    The reference model has geometry in PLY + features in a separate .pth file.
    We look for features_best/fine_radio/best_model.pth relative to the model dir.
    """
    try:
        from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
        model = GaussianFeatureModel(feature_dim=64)
        
        # Load geometry from PLY
        model.load_ply(ply_path)
        
        # Find and load features from .pth
        model_dir = str(Path(ply_path).parent.parent.parent)  # up from point_cloud/best/
        feat_pth = os.path.join(model_dir, 'features_best', 'fine_radio', 'best_model.pth')
        if not os.path.exists(feat_pth):
            feat_pth = os.path.join(model_dir, 'features', 'fine_radio', 'best_model.pth')
        
        if os.path.exists(feat_pth):
            ckpt = torch.load(feat_pth, map_location='cuda')
            loc_feature = ckpt['loc_feature']  # [N, 64]
            # Assign to model
            import torch.nn as nn
            model._loc_feature = nn.Parameter(loc_feature.cuda())
            print(f"[Reference] Loaded from {ply_path} + {feat_pth}")
            print(f"  Gaussians: {model.get_xyz.shape[0]}, Features: {model.get_loc_feature.shape[1]}d")
            return model
        else:
            print(f"[Reference] No feature .pth found near {ply_path}")
            return None
    except Exception as e:
        print(f"[Reference] Failed to load: {e}")
        import traceback; traceback.print_exc()
        return None


def render_reference(ref_model, cam, feat_h=68, feat_w=120):
    """Render features from the direct-embedding reference model."""
    from feature_3dgs.feature_renderer import FeatureRenderer
    
    viewmat = cam_to_viewmat(cam)
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    
    # Render at feature resolution
    fx = feat_w / (2 * tanfovx)
    fy = feat_h / (2 * tanfovy)
    cx = feat_w / 2.0
    cy = feat_h / 2.0
    
    with torch.no_grad():
        result = FeatureRenderer.render_features(
            gaussian_model=ref_model,
            viewmat=viewmat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            img_height=feat_h, img_width=feat_w,
        )
    return result['feature_map']  # [64, fH, fW]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dcff_oldhospital.yaml')
    parser.add_argument('--checkpoint', default='output/dcff_oldhospital/checkpoints/latest.pth')
    parser.add_argument('--reference_ply', default='output/2dgs_joint/joint_oh_v8_radio/point_cloud/best/point_cloud.ply')
    parser.add_argument('--reference_gt_dir', default='output/features_radio/OldHospital_indexed/fine_radio',
                        help='GT features for the reference model (single-scale RADIO)')
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--output_dir', default='output/dcff_oldhospital/vis')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--feat_h', type=int, default=68)
    parser.add_argument('--feat_w', type=int, default=120)
    parser.add_argument('--render_h', type=int, default=544)
    parser.add_argument('--render_w', type=int, default=960)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load DCFF model
    cfg, gaussians, renderer, train_cams, test_cams = \
        load_dcff_model(args.config, args.checkpoint)
    
    # Load reference model
    ref_model = None
    if os.path.exists(args.reference_ply):
        ref_model = load_reference_model(args.reference_ply)
    
    # Build image ordering (for GT feature loading)
    image_order = build_da3_image_order(cfg['dataset']['source_dir'])
    feature_dir = cfg['dataset']['feature_dir']
    
    # Select cameras
    cams = test_cams if args.split == 'test' else train_cams
    if len(cams) > args.num_views:
        indices = np.linspace(0, len(cams) - 1, args.num_views, dtype=int)
        cams = [cams[i] for i in indices]
    
    print(f"\nRendering {len(cams)} views from {args.split} split...")
    
    # Collect metrics
    all_fine_cos = []
    all_coarse_cos = []
    all_ref_cos = []
    
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, saving raw tensors instead")
        Image = None
    
    for vi, cam in enumerate(cams):
        print(f"\n--- View {vi+1}/{len(cams)}: {cam.image_name} ---")
        
        viewmat = cam_to_viewmat(cam)
        K = cam_to_K(cam, args.render_w, args.render_h)
        
        # Render DCFF
        with torch.no_grad():
            result = renderer(
                gaussians,
                viewmat=viewmat,
                K=K,
                width=args.render_w,
                height=args.render_h,
                render_coarse=True,
                feature_height=args.feat_h,
                feature_width=args.feat_w,
            )
        
        rgb = result['rgb'][0]             # [3, H, W]
        depth = result['depth'][0]         # [1, H, W]
        alpha = result['alpha'][0]         # [1, H, W]
        z_map = result['z_map'][0]         # [16, fH, fW]
        fine_feat = result['fine_features'][0]    # [64, fH, fW]
        coarse_feat = result['coarse_features'][0] if result['coarse_features'] is not None else None  # [64, fH, fW]
        
        # Load GT features
        img_name = cam.image_name
        fid = image_order.get(img_name, -1)
        gt_fine = gt_coarse = None
        if fid >= 0:
            fine_path = os.path.join(feature_dir, 'fine_geo', f'rgb_{fid}_fine_geo_64x68x120.pt')
            coarse_path = os.path.join(feature_dir, 'coarse_sem', f'rgb_{fid}_coarse_sem_64x68x120.pt')
            if os.path.exists(fine_path):
                gt_fine = torch.load(fine_path, map_location='cuda').float()
            if os.path.exists(coarse_path):
                gt_coarse = torch.load(coarse_path, map_location='cuda').float()
        
        # Render reference model features & compare against its own GT
        ref_feat = None
        ref_cos = None
        if ref_model is not None:
            ref_feat = render_reference(ref_model, cam, args.feat_h, args.feat_w)
            # Load reference model's own GT (single-scale RADIO)
            if fid >= 0:
                ref_gt_path = os.path.join(args.reference_gt_dir, f'rgb_{fid}_fine_radio_64x68x120.pt')
                if os.path.exists(ref_gt_path):
                    ref_gt = torch.load(ref_gt_path, map_location='cuda').float()
                    _, ref_cos = cosine_sim_map(ref_feat, ref_gt)
                    all_ref_cos.append(ref_cos)
                    print(f"  Ref cos sim (vs fine_radio GT): {ref_cos:.4f}")
        
        # Compute metrics
        if gt_fine is not None:
            _, fine_cos = cosine_sim_map(fine_feat, gt_fine)
            all_fine_cos.append(fine_cos)
            print(f"  Fine cos sim:   {fine_cos:.4f}")
        
        if gt_coarse is not None and coarse_feat is not None:
            _, coarse_cos = cosine_sim_map(coarse_feat, gt_coarse)
            all_coarse_cos.append(coarse_cos)
            print(f"  Coarse cos sim: {coarse_cos:.4f}")
        
        # --- Generate visualization ---
        if Image is None:
            continue
        
        alpha_feat_mask = F.interpolate(
            alpha.unsqueeze(0).float(), (args.feat_h, args.feat_w),
            mode='bilinear', align_corners=False,
        )[0, 0] > 0.5

        # PCA colorize all feature maps. Pred/GT use joint PCA for fair comparison.
        z_pca = pca_colorize(z_map, mask=alpha_feat_mask)
        if gt_fine is not None:
            fine_pca, gt_fine_pca = joint_pca_colorize([fine_feat, gt_fine], mask=alpha_feat_mask)
        else:
            fine_pca = pca_colorize(fine_feat, mask=alpha_feat_mask)
            gt_fine_pca = torch.zeros(3, args.feat_h, args.feat_w)

        if gt_coarse is not None and coarse_feat is not None:
            coarse_pca, gt_coarse_pca = joint_pca_colorize([coarse_feat, gt_coarse], mask=alpha_feat_mask)
        else:
            coarse_pca = pca_colorize(coarse_feat, mask=alpha_feat_mask) if coarse_feat is not None else torch.zeros(3, args.feat_h, args.feat_w)
            gt_coarse_pca = torch.zeros(3, args.feat_h, args.feat_w)

        ref_pca = pca_colorize(ref_feat, mask=alpha_feat_mask) if ref_feat is not None else torch.zeros(3, args.feat_h, args.feat_w)
        
        # Cosine similarity heatmaps
        fine_sim_hm = np.zeros((args.feat_h, args.feat_w, 3), dtype=np.uint8)
        coarse_sim_hm = np.zeros((args.feat_h, args.feat_w, 3), dtype=np.uint8)
        if gt_fine is not None:
            fine_sim, _ = cosine_sim_map(fine_feat, gt_fine)
            fine_sim_hm = sim_to_heatmap(fine_sim)
        if gt_coarse is not None and coarse_feat is not None:
            coarse_sim, _ = cosine_sim_map(coarse_feat, gt_coarse)
            coarse_sim_hm = sim_to_heatmap(coarse_sim)
        
        # Normalize depth for visualization
        d = depth[0]
        d_valid = d[alpha[0] > 0.1]
        if len(d_valid) > 0:
            dmin, dmax = d_valid.min(), d_valid.quantile(0.98)
            d_norm = ((d - dmin) / (dmax - dmin + 1e-6)).clamp(0, 1)
        else:
            d_norm = torch.zeros_like(d)
        depth_vis = d_norm.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        alpha_vis = alpha.repeat(3, 1, 1)  # [3, H, W]
        
        # Resize RGB row to match feature rows for grid assembly
        # Feature maps are at feat resolution; upscale them to render resolution for display
        panel_h, panel_w = 272, 480  # Nice display size
        
        def resize_panel(t_chw, h=panel_h, w=panel_w):
            return F.interpolate(t_chw.unsqueeze(0).float().cpu(), (h, w), mode='bilinear', align_corners=False)[0]
        
        # Row 1: RGB | Depth | Alpha
        row1_panels = [
            to_numpy_img(resize_panel(rgb.cpu())),
            to_numpy_img(resize_panel(depth_vis.cpu())),
            to_numpy_img(resize_panel(alpha_vis.cpu())),
        ]
        
        # Row 2: Z_map PCA | Fine PCA | GT Fine PCA | Fine sim heatmap
        row2_panels = [
            to_numpy_img(resize_panel(z_pca.cpu())),
            to_numpy_img(resize_panel(fine_pca.cpu())),
            to_numpy_img(resize_panel(gt_fine_pca.cpu())),
            np.array(Image.fromarray(fine_sim_hm).resize((panel_w, panel_h), Image.BILINEAR)),
        ]
        
        # Row 3: Coarse PCA | GT Coarse PCA | Ref PCA | Coarse sim heatmap
        row3_panels = [
            to_numpy_img(resize_panel(coarse_pca.cpu())),
            to_numpy_img(resize_panel(gt_coarse_pca.cpu())),
            to_numpy_img(resize_panel(ref_pca.cpu())),
            np.array(Image.fromarray(coarse_sim_hm).resize((panel_w, panel_h), Image.BILINEAR)),
        ]
        
        # Add labels
        def add_label(img, text):
            """Burn text label into top-left corner."""
            from PIL import ImageDraw, ImageFont
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            # Background rectangle
            bbox = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([0, 0, bbox[2] + 8, bbox[3] + 4], fill=(0, 0, 0))
            draw.text((4, 2), text, fill=(255, 255, 255), font=font)
            return np.array(pil)
        
        labels_r1 = ['RGB', 'Depth', 'Alpha']
        sim_str_fine = f"{fine_cos:.3f}" if gt_fine is not None else "N/A"
        sim_str_coarse = f"{coarse_cos:.3f}" if (gt_coarse is not None and coarse_feat is not None) else "N/A"
        labels_r2 = [f'Z_map ({z_map.shape[0]}d latent)', f'Fine pred (cos={sim_str_fine})', 'GT fine_geo', 'Fine sim hmap']
        labels_r3 = [f'Coarse pred (cos={sim_str_coarse})', 'GT coarse_sem', 'Ref direct embed', 'Coarse sim hmap']
        
        for i, lbl in enumerate(labels_r1):
            row1_panels[i] = add_label(row1_panels[i], lbl)
        for i, lbl in enumerate(labels_r2):
            row2_panels[i] = add_label(row2_panels[i], lbl)
        for i, lbl in enumerate(labels_r3):
            row3_panels[i] = add_label(row3_panels[i], lbl)
        
        # Pad row1 to 4 panels (add black panel)
        row1_panels.append(np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
        row1_panels[-1] = add_label(row1_panels[-1], f'{cam.image_name} (fid={fid})')
        
        # Assemble grid
        row1 = np.concatenate(row1_panels, axis=1)
        row2 = np.concatenate(row2_panels, axis=1)
        row3 = np.concatenate(row3_panels, axis=1)
        grid = np.concatenate([row1, row2, row3], axis=0)
        
        # Save
        stem = cam.image_name.replace("/", "_").replace(".png", "").replace(".jpg", "")
        out_path = os.path.join(args.output_dir, f'view_{vi:03d}_{stem}.png')
        Image.fromarray(grid).save(out_path)
        print(f"  Saved: {out_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_fine_cos:
        print(f"  Fine cos sim:   mean={np.mean(all_fine_cos):.4f}, std={np.std(all_fine_cos):.4f}, "
              f"min={np.min(all_fine_cos):.4f}, max={np.max(all_fine_cos):.4f}")
    if all_coarse_cos:
        print(f"  Coarse cos sim: mean={np.mean(all_coarse_cos):.4f}, std={np.std(all_coarse_cos):.4f}, "
              f"min={np.min(all_coarse_cos):.4f}, max={np.max(all_coarse_cos):.4f}")
    if all_ref_cos:
        print(f"  Ref cos sim:    mean={np.mean(all_ref_cos):.4f}, std={np.std(all_ref_cos):.4f}, "
              f"min={np.min(all_ref_cos):.4f}, max={np.max(all_ref_cos):.4f}")
    print()
    
    if all_fine_cos and all_ref_cos:
        gap = np.mean(all_ref_cos) - np.mean(all_fine_cos)
        print(f"  Gap (ref - dcff_fine): {gap:+.4f}")
        if gap > 0.1:
            print(f"  ⚠ DCFF fine features significantly worse than direct embedding")
        elif gap > 0.03:
            print(f"  ⚠ DCFF fine features moderately worse than direct embedding")
        else:
            print(f"  ✓ DCFF fine features comparable to direct embedding")


if __name__ == '__main__':
    main()
