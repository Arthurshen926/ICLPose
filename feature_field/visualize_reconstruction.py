"""
Feature Reconstruction Visualization for DCFF.

Visualizes:
  1. Fine vs Coarse features: side-by-side cosine similarity heatmaps
  2. Channel statistics: per-channel mean/std of predicted vs target
  3. Spatial maps: RGB, depth, alpha, confidence
  4. Feature dimensionality: PCA/t-SNE of feature distributions
  5. Channel selection: which channels FSM selects for fine vs coarse

Usage:
    python -m feature_field.visualize_reconstruction \
        --checkpoint feature_field/output/dcff_oldhospital_v10c_carrier_residual/checkpoints/best.pth \
        --camera_idx 0 \
        --output_dir feature_field/output/vis/v10c_cam0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_field.dcff.radio_teacher import CachedFeatureTeacher
from feature_field.utils.feature_track_vis import feature_group_to_rgb_images
from feature_field.utils.scene_colmap import load_scene_colmap, build_da3_image_order
from feature_field.visualize_feature_comparison import (
    _apply_dcff_postprocess,
    _build_dcff_eval_components,
)


def cosine_similarity_map(pred, target, mask=None):
    """Per-pixel cosine similarity."""
    pred_n = F.normalize(pred, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    cos = (pred_n * target_n).sum(dim=1)
    if mask is not None:
        cos = cos * mask.squeeze(1)
    return cos


def channel_statistics(pred, target, mask=None):
    """Per-channel mean and std statistics."""
    B, C, H, W = pred.shape
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    if mask is not None:
        m_flat = mask.flatten(2)
        n_valid = m_flat.sum(-1, keepdim=True).clamp(min=1)
        pred_mean = (pred_flat * m_flat).sum(-1) / n_valid.squeeze(-1)
        pred_std = ((pred_flat - pred_mean.unsqueeze(-1)) ** 2 * m_flat).sum(-1).sqrt() / n_valid.squeeze(-1).sqrt()
        target_mean = (target_flat * m_flat).sum(-1) / n_valid.squeeze(-1)
        target_std = ((target_flat - target_mean.unsqueeze(-1)) ** 2 * m_flat).sum(-1).sqrt() / n_valid.squeeze(-1).sqrt()
    else:
        pred_mean = pred_flat.mean(-1)
        pred_std = pred_flat.std(-1)
        target_mean = target_flat.mean(-1)
        target_std = target_flat.std(-1)
    return pred_mean.squeeze(0), pred_std.squeeze(0), target_mean.squeeze(0), target_std.squeeze(0)


def visualize_reconstruction(
    ckpt_path,
    source_dir,
    feature_dir,
    output_dir,
    camera_idx=0,
    device='cuda',
    images_subdir='',
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    bundle = _build_dcff_eval_components(ckpt_path, device)
    ckpt = bundle['ckpt']
    cfg = bundle['cfg']
    mcfg = cfg.get('model', {})
    feat_dim = bundle['feat_dim']
    renderer = bundle['renderer']
    gaussians = bundle['gaussians']
    feat_sharp_fine = bundle['feat_sharp_fine']
    feat_select = bundle['feat_select']

    # Load scene
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(source_dir, images_subdir)
    radio_cache = CachedFeatureTeacher(feature_dir)
    feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w

    # Camera-to-feature mapping
    import glob
    if images_subdir:
        images_dir = os.path.join(source_dir, images_subdir)
    elif os.path.isdir(os.path.join(source_dir, 'images')):
        images_dir = os.path.join(source_dir, 'images')
    else:
        images_dir = source_dir
    da3_name_to_fid = build_da3_image_order(images_dir)
    cam_to_fid = {}
    for cam in train_cams + test_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in radio_cache.frame_ids:
            cam_to_fid[cam.uid] = fid

    valid_cams = [c for c in train_cams if c.uid in cam_to_fid]
    if not valid_cams:
        raise ValueError("No valid cameras found")
    cam = valid_cams[camera_idx]
    fid = cam_to_fid.get(cam.uid)
    if fid is None:
        raise ValueError(f"Camera {cam.uid} not in cam_to_fid. image_name={cam.image_name}")
    print(f"Using camera {cam.uid} (idx={camera_idx}): {cam.image_name} → fid={fid}")

    # Load teacher features for this camera
    geo_target, sem_target = radio_cache.get(fid)

    # Load GT image
    img = Image.open(cam.image).convert('RGB')
    W_img, H_img = img.size
    scale = 960 / max(W_img, H_img)
    if scale < 1.0:
        img = img.resize((int(W_img * scale), int(H_img * scale)), Image.LANCZOS)
        W_img, H_img = img.size
    img_np = np.array(img)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    gt_rgb = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    gt_rgb = gt_rgb.to(device)

    # Camera matrices
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    viewmat = torch.tensor(W2C, dtype=torch.float32, device=device).unsqueeze(0)
    tanfovx = np.tan(cam.FovX * 0.5)
    tanfovy = np.tan(cam.FovY * 0.5)
    _, C_img, H_img, W_img = gt_rgb.shape
    fx, fy = W_img / (2 * tanfovx), H_img / (2 * tanfovy)
    K = torch.tensor([[fx, 0, W_img / 2], [0, fy, H_img / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device).unsqueeze(0)

    # Render
    renderer.eval()
    renderer.zero_grad()

    with torch.no_grad():
        result = renderer(
            gaussians,
            viewmat=viewmat,
            K=K,
            width=W_img,
            height=H_img,
            render_coarse=True,
            feature_height=feat_h,
            feature_width=feat_w,
        )
        result = _apply_dcff_postprocess(
            result,
            feat_h,
            feat_w,
            feat_sharp_fine=feat_sharp_fine,
            feat_select=feat_select,
        )

    # Get features
    fine_pred = result['fine_features']
    coarse_pred = result['coarse_features']
    rgb = result['rgb']
    depth = result['depth']
    alpha = result['alpha']

    # Load teacher features
    geo_target = geo_target.unsqueeze(0).to(device)
    sem_target = sem_target.unsqueeze(0).to(device)

    # Resize to match teacher resolution
    fine_pred_up = F.interpolate(fine_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)
    coarse_pred_up = F.interpolate(coarse_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)

    # Masks at fine resolution
    alpha_fine = F.interpolate(alpha, (feat_h, feat_w), mode='bilinear', align_corners=False)
    mask_fine = (alpha_fine > 0.5).float()

    # Coarse mask at fine resolution (for visualization)
    alpha_coarse_vis = F.interpolate(alpha, (feat_h, feat_w), mode='bilinear', align_corners=False)
    mask_coarse = (alpha_coarse_vis > 0.5).float()

    # Cosine similarities
    fine_cos = cosine_similarity_map(fine_pred_up, geo_target, mask_fine)
    coarse_cos = cosine_similarity_map(coarse_pred_up, sem_target, mask_coarse)

    # Overall metrics
    fine_cos_mean = fine_cos[mask_fine.squeeze(1) > 0.5].mean().item()
    coarse_cos_mean = coarse_cos[mask_coarse.squeeze(1) > 0.5].mean().item()
    print(f"Camera {camera_idx}: fine_cos={fine_cos_mean:.4f}, coarse_cos={coarse_cos_mean:.4f}")

    # ── Figure 1: Feature Comparison ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Feature Reconstruction (cam={camera_idx})')

    # Row 1: Fine
    axes[0, 0].imshow(rgb.squeeze(0).permute(1, 2, 0).cpu())
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')

    vmin, vmax = depth.min().item(), depth.max().item()
    axes[0, 1].imshow(depth.squeeze(0).squeeze(0).cpu(), cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Depth')
    axes[0, 1].axis('off')

    fine_rgb, geo_rgb = [
        np.asarray(img)
        for img in feature_group_to_rgb_images(
            [fine_pred_up.squeeze(0), geo_target.squeeze(0)],
            [mask_fine.squeeze(0), mask_fine.squeeze(0)],
        )
    ]
    axes[0, 2].imshow(fine_rgb)
    axes[0, 2].set_title(f'Fine Pred (cos={fine_cos_mean:.3f})')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(geo_rgb)
    axes[0, 3].set_title('Fine Target (RADIO)')
    axes[0, 3].axis('off')

    # Row 2: Coarse
    axes[1, 0].imshow(alpha.squeeze(0).squeeze(0).cpu(), cmap='gray')
    axes[1, 0].set_title('Alpha')
    axes[1, 0].axis('off')

    cos_display = fine_cos.squeeze(0).cpu().numpy()
    im = axes[1, 1].imshow(cos_display, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Fine Cosine Sim (avg={fine_cos_mean:.3f})')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])

    coarse_rgb, sem_rgb = [
        np.asarray(img)
        for img in feature_group_to_rgb_images(
            [coarse_pred_up.squeeze(0), sem_target.squeeze(0)],
            [mask_coarse.squeeze(0), mask_coarse.squeeze(0)],
        )
    ]
    axes[1, 2].imshow(coarse_rgb)
    axes[1, 2].set_title(f'Coarse Pred (cos={coarse_cos_mean:.3f})')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(sem_rgb)
    axes[1, 3].set_title('Coarse Target (RADIO)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'features_cam{camera_idx}.png'), dpi=150)
    print(f"Saved: features_cam{camera_idx}.png")

    # ── Figure 2: Channel Statistics ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    pred_mean, pred_std, tgt_mean, tgt_std = channel_statistics(
        fine_pred_up, geo_target, mask_fine)

    ch = np.arange(feat_dim)
    axes2[0].bar(ch[:32], pred_mean[:32].cpu().numpy(), alpha=0.7, label='Pred mean')
    axes2[0].bar(ch[:32], tgt_mean[:32].cpu().numpy(), alpha=0.5, label='Target mean')
    axes2[0].set_xlabel('Channel')
    axes2[0].set_ylabel('Mean')
    axes2[0].set_title('Per-channel Mean (first 32 channels)')
    axes2[0].legend()

    axes2[1].bar(ch[:32], pred_std[:32].cpu().numpy(), alpha=0.7, label='Pred std')
    axes2[1].bar(ch[:32], tgt_std[:32].cpu().numpy(), alpha=0.5, label='Target std')
    axes2[1].set_xlabel('Channel')
    axes2[1].set_ylabel('Std')
    axes2[1].set_title('Per-channel Std (first 32 channels)')
    axes2[1].legend()

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'channel_stats_cam{camera_idx}.png'), dpi=150)
    print(f"Saved: channel_stats_cam{camera_idx}.png")

    # ── Figure 3: Cosine Similarity Distribution ──
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

    fine_valid = fine_cos[mask_fine.squeeze(1) > 0.5].cpu().numpy()
    coarse_valid = coarse_cos[mask_coarse.squeeze(1) > 0.5].cpu().numpy()

    axes3[0].hist(fine_valid, bins=50, alpha=0.7, color='blue')
    axes3[0].axvline(fine_cos_mean, color='red', linestyle='--', label=f'mean={fine_cos_mean:.3f}')
    axes3[0].set_xlabel('Cosine Similarity')
    axes3[0].set_ylabel('Count')
    axes3[0].set_title('Fine Cosine Similarity Distribution')
    axes3[0].legend()

    axes3[1].hist(coarse_valid, bins=50, alpha=0.7, color='green')
    axes3[1].axvline(coarse_cos_mean, color='red', linestyle='--', label=f'mean={coarse_cos_mean:.3f}')
    axes3[1].set_xlabel('Cosine Similarity')
    axes3[1].set_ylabel('Count')
    axes3[1].set_title('Coarse Cosine Similarity Distribution')
    axes3[1].legend()

    im3 = axes3[2].imshow(coarse_cos.squeeze(0).cpu().numpy(), cmap='RdYlGn', vmin=0, vmax=1)
    axes3[2].set_title(f'Coarse Cosine Sim (avg={coarse_cos_mean:.3f})')
    axes3[2].axis('off')
    plt.colorbar(im3, ax=axes3[2])

    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, f'cosine_dist_cam{camera_idx}.png'), dpi=150)
    print(f"Saved: cosine_dist_cam{camera_idx}.png")

    print(f"\nAll visualizations saved to {output_dir}")
    return {'fine_cos': fine_cos_mean, 'coarse_cos': coarse_cos_mean}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source_dir', default='/root/ICLPose-loc/dataset/OldHospital')
    parser.add_argument('--feature_dir', default='/root/ICLPose-loc/feature_extract/output/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--output_dir', default='feature_field/output/vis/reconstruction')
    parser.add_argument('--camera_idx', type=int, default=0)
    parser.add_argument('--images_subdir', default='', help='Optional image subdir, e.g. processed')
    args = parser.parse_args()

    visualize_reconstruction(
        args.checkpoint,
        args.source_dir,
        args.feature_dir,
        args.output_dir,
        args.camera_idx,
        images_subdir=args.images_subdir,
    )
