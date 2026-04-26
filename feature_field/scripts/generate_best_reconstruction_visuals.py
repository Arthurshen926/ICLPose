#!/usr/bin/env python3
"""Export richer reconstruction visuals for the current best DCFF checkpoint.

Outputs per selected camera:
  - RGB / depth / alpha overview for the implicit map render
  - Latent PCA image + top-variance latent channel grid
  - DCFF fine/coarse PCA triplets against teacher + explicit 2DGS
  - Query-student fine/coarse PCA panels against teacher features
  - DCFF / explicit 2DGS / query-student channel grids
  - Extra explicit 2DGS render channels: RGB, depth, alpha, normals, distortion
  - Per-camera metrics JSON

The script defaults to the currently best OldHospital checkpoint (`v15d_best`)
according to `output/feature_field/postprocess_metrics/v15_summary_clean.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from feature_field.dcff.radio_teacher import CachedFeatureTeacher
from feature_field.utils.scene_colmap import load_scene_colmap, build_da3_image_order
from feature_field.visualize_feature_comparison import (
    _apply_dcff_postprocess,
    _build_dcff_eval_components,
    _cam_to_K,
    _cam_to_viewmat,
    _ensure_bchw_alpha,
    _ensure_bchw_depth,
    _load_image_tensor,
    cosine_similarity_map,
    joint_pca_colorize,
    pca_colorize_torch,
)
from feature_gaussian.legacy_3dgs.train_2dgs_joint import (
    GaussianModel2DGS,
    load_scene_joint,
    render_features_2dgs,
    render_rgb_2dgs,
)


DEFAULT_SUMMARY_JSON = ROOT / 'output/feature_field/postprocess_metrics/v15_summary_clean.json'
DEFAULT_DCFF_CKPT = ROOT / 'output/feature_field/dcff_oldhospital_v15d_v14b_coarse_smooth_frozen/checkpoints/best.pth'
DEFAULT_SOURCE_DIR = ROOT / 'dataset/OldHospital'
DEFAULT_FEATURE_DIR = ROOT / 'output/feature_extract/features_radio_dual/OldHospital_pilot'
DEFAULT_QUERY_FEATURE_DIR = ROOT / 'output/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_b4/OldHospital'
DEFAULT_EXPLICIT_PLY = ROOT / 'output/feature_gaussian/joint_radio_dual_oldhospital_teacher_geom_v1/point_cloud/best/point_cloud.ply'
DEFAULT_EXPLICIT_FINE = ROOT / 'output/feature_gaussian/joint_radio_dual_oldhospital_teacher_geom_v1/features_best/fine_geo/best_model.pth'
DEFAULT_EXPLICIT_COARSE = ROOT / 'output/feature_gaussian/joint_radio_dual_oldhospital_teacher_geom_v1/features_best/coarse_sem/best_model.pth'

WINNER_TO_CKPT = {
    'v15d_best': DEFAULT_DCFF_CKPT,
}


def parse_camera_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(',') if part.strip()]


def sanitize_name(name: str) -> str:
    return name.replace('/', '__').replace(' ', '_')


def ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_best_checkpoint(summary_json: Path, ckpt_override: Path | None) -> tuple[Path, str, dict]:
    winner_name = 'v15d_best'
    winner_meta: dict = {}
    if summary_json.exists():
        with summary_json.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        winner_meta = summary.get('winner', {})
        winner_name = winner_meta.get('experiment', winner_name)

    if ckpt_override is not None:
        ckpt_path = ckpt_override
    else:
        ckpt_path = WINNER_TO_CKPT.get(winner_name, DEFAULT_DCFF_CKPT)

    return ensure_exists(Path(ckpt_path), 'DCFF checkpoint'), winner_name, winner_meta


def normalize_map_for_display(map_hw: torch.Tensor, mask_hw: torch.Tensor | None = None) -> np.ndarray:
    values = map_hw.float()
    valid = values[mask_hw] if mask_hw is not None and mask_hw.any() else values.reshape(-1)
    if valid.numel() == 0:
        norm = torch.zeros_like(values)
    else:
        if valid.numel() >= 8:
            lo = torch.quantile(valid, 0.02)
            hi = torch.quantile(valid, 0.98)
        else:
            lo = valid.min()
            hi = valid.max()
        if torch.isclose(hi, lo):
            norm = torch.zeros_like(values)
        else:
            norm = (values - lo) / (hi - lo)
            norm = norm.clamp(0, 1)
    if mask_hw is not None:
        norm = norm * mask_hw.float()
    return norm.cpu().numpy()


def colorize_scalar_map(map_hw: torch.Tensor, mask_hw: torch.Tensor | None = None, cmap_name: str = 'turbo') -> np.ndarray:
    norm = normalize_map_for_display(map_hw, mask_hw=mask_hw)
    rgb = matplotlib.colormaps.get_cmap(cmap_name)(norm)[..., :3]
    if mask_hw is not None:
        rgb[~mask_hw.cpu().numpy()] = 0
    return rgb


def colorize_unit_interval_map(
    map_hw: torch.Tensor,
    mask_hw: torch.Tensor | None = None,
    cmap_name: str = 'RdYlGn',
) -> np.ndarray:
    values = map_hw.float().clamp(0, 1).cpu().numpy()
    rgb = matplotlib.colormaps.get_cmap(cmap_name)(values)[..., :3]
    if mask_hw is not None:
        rgb[~mask_hw.cpu().numpy()] = 0
    return rgb


def chw_to_rgb_numpy(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def save_rgb_image(image_chw: torch.Tensor, path: Path) -> None:
    arr = (chw_to_rgb_numpy(image_chw) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_pca_image(image_chw: torch.Tensor, title: str, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(chw_to_rgb_numpy(image_chw))
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_triplet(images: list[torch.Tensor], titles: list[str], suptitle: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    fig.suptitle(suptitle)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(chw_to_rgb_numpy(img))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def normal_tensor_to_rgb(normal: torch.Tensor | None, mask_hw: torch.Tensor | None = None) -> np.ndarray | None:
    if normal is None:
        return None

    normal_chw = normal.detach().cpu().float()
    if normal_chw.ndim == 4:
        normal_chw = normal_chw.squeeze(0)
    if normal_chw.ndim == 3 and normal_chw.shape[-1] == 3:
        normal_chw = normal_chw.permute(2, 0, 1)
    if normal_chw.ndim != 3 or normal_chw.shape[0] != 3:
        raise ValueError(f'Unsupported normal shape: {tuple(normal.shape)}')

    rgb = (normal_chw.clamp(-1, 1) * 0.5 + 0.5)
    if mask_hw is not None:
        rgb = rgb * mask_hw.float().unsqueeze(0)
    return chw_to_rgb_numpy(rgb)


def select_top_variance_channels(feat_chw: torch.Tensor, mask_hw: torch.Tensor | None, num_channels: int) -> list[int]:
    c = feat_chw.shape[0]
    flat = feat_chw.float().reshape(c, -1)
    if mask_hw is not None and mask_hw.any():
        flat = flat[:, mask_hw.reshape(-1)]
    if flat.numel() == 0:
        return list(range(min(num_channels, c)))
    variances = flat.var(dim=1, unbiased=False)
    k = min(num_channels, c)
    topk = torch.topk(variances, k=k, largest=True).indices.tolist()
    return sorted(topk)


def save_channel_grid(
    feat_chw: torch.Tensor,
    mask_hw: torch.Tensor | None,
    channels: list[int],
    title: str,
    path: Path,
    cols: int = 4,
) -> None:
    if not channels:
        return
    rows = math.ceil(len(channels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title)

    for ax, channel_idx in zip(axes, channels):
        rgb = colorize_scalar_map(feat_chw[channel_idx], mask_hw=mask_hw, cmap_name='turbo')
        ax.imshow(rgb)
        ax.set_title(f'ch {channel_idx}')
        ax.axis('off')

    for ax in axes[len(channels):]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def masked_cosine_summary(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    cos_map = cosine_similarity_map(pred, target, mask)
    valid = mask.squeeze(1) > 0.5
    if valid.any():
        mean_value = float(cos_map[valid].mean().item())
    else:
        mean_value = float('nan')
    return cos_map, mean_value


def build_explicit_bundle(ply_path: Path, fine_path: Path, coarse_path: Path, device: torch.device) -> dict:
    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(str(ply_path))
    gaussians.active_sh_degree = 3

    fine_ckpt = torch.load(str(fine_path), map_location='cpu')
    coarse_ckpt = torch.load(str(coarse_path), map_location='cpu')
    n_gauss = gaussians.get_xyz.shape[0]

    return {
        'gaussians': gaussians,
        'fine_gaussian_features': fine_ckpt['loc_feature'].float()[:n_gauss].to(device),
        'coarse_gaussian_features': coarse_ckpt['loc_feature'].float()[:n_gauss].to(device),
    }


def make_output_dir(base_output_dir: Path | None, winner_name: str, split: str) -> Path:
    if base_output_dir is not None:
        return base_output_dir
    return ROOT / 'output/feature_field' / f'vis_{winner_name}_rich_{split}'


def load_camera_context(source_dir: Path, feature_dir: Path, split: str) -> tuple[list, dict, CachedFeatureTeacher, dict, dict]:
    train_cams, test_cams, _, _, _ = load_scene_colmap(str(source_dir), '')
    all_2dgs_cams, _, _, _ = load_scene_joint(str(source_dir))
    explicit_by_name = {cam.image_name: cam for cam in all_2dgs_cams}

    images_dir = source_dir / 'images'
    images_root = images_dir if images_dir.is_dir() else source_dir
    da3_name_to_fid = build_da3_image_order(str(images_root))
    radio_cache = CachedFeatureTeacher(str(feature_dir))

    cam_pool = test_cams if split == 'test' else train_cams if split == 'train' else train_cams + test_cams

    valid_cams = []
    cam_to_fid = {}
    for cam in cam_pool:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is None or fid not in radio_cache.frame_ids:
            continue
        if cam.image_name not in explicit_by_name:
            continue
        valid_cams.append(cam)
        cam_to_fid[cam.uid] = fid

    return valid_cams, cam_to_fid, radio_cache, explicit_by_name, {
        'train_count': len(train_cams),
        'test_count': len(test_cams),
    }


def render_explicit_features(
    bundle: dict,
    cam,
    explicit_cam,
    feat_h: int,
    feat_w: int,
    longest_edge: int,
    device: torch.device,
) -> dict:
    viewmat = _cam_to_viewmat(cam, device).squeeze(0)
    K_feat = _cam_to_K(cam, feat_w, feat_h, device).squeeze(0)
    fine_map = render_features_2dgs(
        bundle['gaussians'],
        viewmat,
        bundle['fine_gaussian_features'],
        feat_h,
        feat_w,
        K_feat,
    ).unsqueeze(0)
    coarse_map = render_features_2dgs(
        bundle['gaussians'],
        viewmat,
        bundle['coarse_gaussian_features'],
        feat_h,
        feat_w,
        K_feat,
    ).unsqueeze(0)

    render_pkg = render_rgb_2dgs(
        bundle['gaussians'],
        explicit_cam,
        bg_color=torch.zeros(3, device=device),
        longest_edge=longest_edge,
    )
    alpha = _ensure_bchw_alpha(render_pkg['rend_alpha'])
    depth = _ensure_bchw_depth(render_pkg['depth'])
    render_dist = _ensure_bchw_depth(render_pkg.get('rend_dist'))
    alpha_feat = F.interpolate(alpha, (feat_h, feat_w), mode='bilinear', align_corners=False)
    depth_feat = F.interpolate(depth, (feat_h, feat_w), mode='bilinear', align_corners=False)

    return {
        'fine_features': fine_map,
        'coarse_features': coarse_map,
        'alpha': alpha_feat,
        'depth': depth_feat,
        'render_rgb': render_pkg['render'].detach().float(),
        'render_alpha': alpha.detach().float(),
        'render_depth': depth.detach().float(),
        'render_normal': render_pkg.get('rend_normal'),
        'surf_normal': render_pkg.get('surf_normal'),
        'render_dist': render_dist.detach().float() if render_dist is not None else None,
    }


def save_overview(
    output_path: Path,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    latent_pca: torch.Tensor,
    fine_vis: list[torch.Tensor],
    coarse_vis: list[torch.Tensor],
    fine_text: str,
    coarse_text: str,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    depth_hw = depth.squeeze(0).squeeze(0).cpu()
    alpha_hw = alpha.squeeze(0).squeeze(0).cpu()

    axes[0, 0].imshow(chw_to_rgb_numpy(rgb.squeeze(0)))
    axes[0, 0].set_title('RGB')
    axes[0, 1].imshow(colorize_scalar_map(depth_hw, mask_hw=None, cmap_name='turbo'))
    axes[0, 1].set_title('Depth')
    axes[0, 2].imshow(alpha_hw.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Alpha')
    axes[0, 3].imshow(chw_to_rgb_numpy(latent_pca))
    axes[0, 3].set_title('Latent PCA')

    row2_titles = ['DCFF Fine', 'Explicit Fine', 'Teacher Fine']
    row3_titles = ['DCFF Coarse', 'Explicit Coarse', 'Teacher Coarse']
    for col in range(3):
        axes[1, col].imshow(chw_to_rgb_numpy(fine_vis[col]))
        axes[1, col].set_title(row2_titles[col])
        axes[2, col].imshow(chw_to_rgb_numpy(coarse_vis[col]))
        axes[2, col].set_title(row3_titles[col])

    axes[1, 3].axis('off')
    axes[1, 3].text(0.02, 0.98, fine_text, va='top', fontsize=11, family='monospace')
    axes[2, 3].axis('off')
    axes[2, 3].text(0.02, 0.98, coarse_text, va='top', fontsize=11, family='monospace')

    for ax in axes.reshape(-1):
        if ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_query_overview(
    output_path: Path,
    rgb: torch.Tensor,
    fine_vis: list[torch.Tensor],
    coarse_vis: list[torch.Tensor],
    fine_cos_map: torch.Tensor,
    coarse_cos_map: torch.Tensor,
    fine_text: str,
    coarse_text: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(chw_to_rgb_numpy(rgb.squeeze(0)))
    axes[0, 0].set_title('RGB')
    axes[0, 1].imshow(chw_to_rgb_numpy(fine_vis[0]))
    axes[0, 1].set_title('Query Fine')
    axes[0, 2].imshow(chw_to_rgb_numpy(fine_vis[1]))
    axes[0, 2].set_title('Teacher Fine')
    axes[0, 3].imshow(colorize_unit_interval_map(fine_cos_map.squeeze(0), cmap_name='RdYlGn'))
    axes[0, 3].set_title('Fine Cosine')

    axes[1, 0].imshow(chw_to_rgb_numpy(coarse_vis[0]))
    axes[1, 0].set_title('Query Coarse')
    axes[1, 1].imshow(chw_to_rgb_numpy(coarse_vis[1]))
    axes[1, 1].set_title('Teacher Coarse')
    axes[1, 2].imshow(colorize_unit_interval_map(coarse_cos_map.squeeze(0), cmap_name='RdYlGn'))
    axes[1, 2].set_title('Coarse Cosine')
    axes[1, 3].axis('off')
    axes[1, 3].text(
        0.02,
        0.98,
        f'{fine_text}\n\n{coarse_text}',
        va='top',
        fontsize=11,
        family='monospace',
    )

    for ax in axes.reshape(-1):
        if ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_explicit_render_channels(
    output_path: Path,
    render_rgb: torch.Tensor,
    render_depth: torch.Tensor,
    render_alpha: torch.Tensor,
    render_normal: torch.Tensor | None,
    surf_normal: torch.Tensor | None,
    render_dist: torch.Tensor | None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    alpha_hw = render_alpha.squeeze(0).squeeze(0).cpu()
    depth_hw = render_depth.squeeze(0).squeeze(0).cpu()
    mask_hw = alpha_hw > 0.5
    dist_hw = render_dist.squeeze(0).squeeze(0).cpu() if render_dist is not None else None

    axes[0, 0].imshow(chw_to_rgb_numpy(render_rgb))
    axes[0, 0].set_title('Explicit RGB')
    axes[0, 1].imshow(colorize_scalar_map(depth_hw, mask_hw=mask_hw, cmap_name='turbo'))
    axes[0, 1].set_title('Explicit Depth')
    axes[0, 2].imshow(alpha_hw.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Explicit Alpha')

    render_normal_rgb = normal_tensor_to_rgb(render_normal, mask_hw=mask_hw)
    surf_normal_rgb = normal_tensor_to_rgb(surf_normal, mask_hw=mask_hw)
    axes[1, 0].imshow(render_normal_rgb if render_normal_rgb is not None else np.zeros((*alpha_hw.shape, 3), dtype=np.float32))
    axes[1, 0].set_title('Render Normal')
    axes[1, 1].imshow(surf_normal_rgb if surf_normal_rgb is not None else np.zeros((*alpha_hw.shape, 3), dtype=np.float32))
    axes[1, 1].set_title('Surface Normal')
    if dist_hw is not None:
        axes[1, 2].imshow(colorize_scalar_map(dist_hw, mask_hw=mask_hw, cmap_name='magma'))
        axes[1, 2].set_title('Distortion')
    else:
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, 'No distortion map', ha='center', va='center')

    for ax in axes.reshape(-1):
        if ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def process_split(args: argparse.Namespace) -> Path:
    summary_json = Path(args.summary_json)
    source_dir = ensure_exists(Path(args.source_dir), 'Source directory')
    feature_dir = ensure_exists(Path(args.feature_dir), 'Feature directory')
    query_feature_dir = Path(args.query_feature_dir) if args.query_feature_dir else None
    if query_feature_dir is not None:
        query_feature_dir = ensure_exists(query_feature_dir, 'Query feature directory')
    explicit_ply = ensure_exists(Path(args.explicit_ply), 'Explicit Gaussian point cloud')
    explicit_fine = ensure_exists(Path(args.explicit_fine), 'Explicit fine feature checkpoint')
    explicit_coarse = ensure_exists(Path(args.explicit_coarse), 'Explicit coarse feature checkpoint')

    dcff_ckpt_override = Path(args.checkpoint) if args.checkpoint else None
    ckpt_path, winner_name, winner_meta = resolve_best_checkpoint(summary_json, dcff_ckpt_override)
    output_dir = make_output_dir(Path(args.output_dir) if args.output_dir else None, winner_name, args.camera_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dcff_bundle = _build_dcff_eval_components(str(ckpt_path), str(device))
    explicit_bundle = build_explicit_bundle(explicit_ply, explicit_fine, explicit_coarse, device)
    query_cache = CachedFeatureTeacher(str(query_feature_dir)) if query_feature_dir is not None else None

    valid_cams, cam_to_fid, radio_cache, explicit_by_name, split_meta = load_camera_context(
        source_dir,
        feature_dir,
        args.camera_split,
    )
    if not valid_cams:
        raise RuntimeError(f'No valid cameras found for split={args.camera_split}')
    if query_cache is not None and (query_cache.feat_h != radio_cache.feat_h or query_cache.feat_w != radio_cache.feat_w):
        raise ValueError(
            f'Query feature resolution {query_cache.feat_h}x{query_cache.feat_w} does not match '
            f'teacher resolution {radio_cache.feat_h}x{radio_cache.feat_w}'
        )

    camera_indices = parse_camera_indices(args.camera_indices)
    if camera_indices is None:
        camera_indices = list(range(min(args.max_cameras, len(valid_cams))))

    feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w
    longest_edge = int(dcff_bundle['longest_edge'])
    results = []

    for camera_idx in camera_indices:
        if camera_idx >= len(valid_cams):
            continue
        cam = valid_cams[camera_idx]
        explicit_cam = explicit_by_name[cam.image_name]
        fid = cam_to_fid[cam.uid]
        if query_cache is not None and fid not in query_cache.frame_ids:
            raise KeyError(f'Missing query-student features for fid={fid} ({cam.image_name})')
        camera_dir = output_dir / f'cam{camera_idx:03d}_{sanitize_name(cam.image_name)}'
        camera_dir.mkdir(parents=True, exist_ok=True)

        geo_target, sem_target = radio_cache.get(fid)
        geo_target = geo_target.unsqueeze(0).to(device)
        sem_target = sem_target.unsqueeze(0).to(device)
        if query_cache is not None:
            query_fine, query_coarse = query_cache.get(fid)
            query_fine = query_fine.unsqueeze(0).to(device)
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_fine = None
            query_coarse = None

        gt_rgb = _load_image_tensor(cam, longest_edge, str(device))
        _, _, render_h, render_w = gt_rgb.shape
        viewmat = _cam_to_viewmat(cam, str(device))
        K = _cam_to_K(cam, render_w, render_h, str(device))

        with torch.no_grad():
            result = dcff_bundle['renderer'](
                dcff_bundle['gaussians'],
                viewmat=viewmat,
                K=K,
                width=render_w,
                height=render_h,
                render_coarse=True,
                feature_height=feat_h,
                feature_width=feat_w,
            )
            result = _apply_dcff_postprocess(
                result,
                feat_h,
                feat_w,
                feat_sharp_fine=dcff_bundle['feat_sharp_fine'],
                feat_select=dcff_bundle['feat_select'],
            )
            explicit_result = render_explicit_features(
                explicit_bundle,
                cam,
                explicit_cam,
                feat_h,
                feat_w,
                longest_edge,
                device,
            )

        z_map = result['z_map']
        fine_pred = result['fine_features']
        coarse_pred = result['coarse_features']
        if fine_pred.shape[-2:] != (feat_h, feat_w):
            fine_pred = F.interpolate(fine_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)
        if coarse_pred.shape[-2:] != (feat_h, feat_w):
            coarse_pred = F.interpolate(coarse_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)

        alpha_dcff = F.interpolate(result['alpha'], (feat_h, feat_w), mode='bilinear', align_corners=False)
        alpha_explicit = explicit_result['alpha']
        fine_explicit = explicit_result['fine_features']
        coarse_explicit = explicit_result['coarse_features']

        dcff_fine_cos_map, dcff_fine_cos = masked_cosine_summary(fine_pred, geo_target, alpha_dcff > 0.5)
        dcff_coarse_cos_map, dcff_coarse_cos = masked_cosine_summary(coarse_pred, sem_target, alpha_dcff > 0.5)
        explicit_fine_cos_map, explicit_fine_cos = masked_cosine_summary(fine_explicit, geo_target, alpha_explicit > 0.5)
        explicit_coarse_cos_map, explicit_coarse_cos = masked_cosine_summary(coarse_explicit, sem_target, alpha_explicit > 0.5)
        full_mask = torch.ones((1, 1, feat_h, feat_w), dtype=torch.bool, device=device)
        if query_fine is not None and query_coarse is not None:
            query_fine_cos_map, query_fine_cos = masked_cosine_summary(query_fine, geo_target, full_mask)
            query_coarse_cos_map, query_coarse_cos = masked_cosine_summary(query_coarse, sem_target, full_mask)
        else:
            query_fine_cos_map = None
            query_coarse_cos_map = None
            query_fine_cos = float('nan')
            query_coarse_cos = float('nan')

        mask_dcff_hw = (alpha_dcff.squeeze(0).squeeze(0) > 0.5).detach().cpu()
        mask_explicit_hw = (alpha_explicit.squeeze(0).squeeze(0) > 0.5).detach().cpu()
        mask_union_hw = mask_dcff_hw | mask_explicit_hw
        if not mask_union_hw.any():
            mask_union_hw = torch.ones((feat_h, feat_w), dtype=torch.bool)

        latent_cpu = z_map.squeeze(0).detach().cpu().float()
        fine_dcff_cpu = fine_pred.squeeze(0).detach().cpu().float()
        coarse_dcff_cpu = coarse_pred.squeeze(0).detach().cpu().float()
        fine_explicit_cpu = fine_explicit.squeeze(0).detach().cpu().float()
        coarse_explicit_cpu = coarse_explicit.squeeze(0).detach().cpu().float()
        geo_cpu = geo_target.squeeze(0).detach().cpu().float()
        sem_cpu = sem_target.squeeze(0).detach().cpu().float()
        if query_fine is not None and query_coarse is not None:
            query_fine_cpu = query_fine.squeeze(0).detach().cpu().float()
            query_coarse_cpu = query_coarse.squeeze(0).detach().cpu().float()
            query_fine_pca = joint_pca_colorize([query_fine_cpu, geo_cpu], mask=None)
            query_coarse_pca = joint_pca_colorize([query_coarse_cpu, sem_cpu], mask=None)
            query_fine_channels = select_top_variance_channels(query_fine_cpu, None, args.num_channels)
            query_coarse_channels = select_top_variance_channels(query_coarse_cpu, None, args.num_channels)
        else:
            query_fine_cpu = None
            query_coarse_cpu = None
            query_fine_pca = None
            query_coarse_pca = None
            query_fine_channels = []
            query_coarse_channels = []

        latent_pca = pca_colorize_torch(latent_cpu, mask=mask_dcff_hw)
        fine_pca = joint_pca_colorize([fine_dcff_cpu, fine_explicit_cpu, geo_cpu], mask=mask_union_hw)
        coarse_pca = joint_pca_colorize([coarse_dcff_cpu, coarse_explicit_cpu, sem_cpu], mask=mask_union_hw)

        latent_channels = select_top_variance_channels(latent_cpu, mask_dcff_hw, args.num_channels)
        dcff_fine_channels = select_top_variance_channels(fine_dcff_cpu, mask_dcff_hw, args.num_channels)
        dcff_coarse_channels = select_top_variance_channels(coarse_dcff_cpu, mask_dcff_hw, args.num_channels)
        explicit_fine_channels = select_top_variance_channels(fine_explicit_cpu, mask_explicit_hw, args.num_channels)
        explicit_coarse_channels = select_top_variance_channels(coarse_explicit_cpu, mask_explicit_hw, args.num_channels)

        save_rgb_image(latent_pca, camera_dir / 'latent_pca.png')
        save_triplet(
            fine_pca,
            [
                f'DCFF Fine ({dcff_fine_cos:.3f})',
                f'Explicit Fine ({explicit_fine_cos:.3f})',
                'Teacher Fine',
            ],
            'Fine Features (joint PCA basis)',
            camera_dir / 'fine_pca_triplet.png',
        )
        save_triplet(
            coarse_pca,
            [
                f'DCFF Coarse ({dcff_coarse_cos:.3f})',
                f'Explicit Coarse ({explicit_coarse_cos:.3f})',
                'Teacher Coarse',
            ],
            'Coarse Features (joint PCA basis)',
            camera_dir / 'coarse_pca_triplet.png',
        )
        save_channel_grid(
            latent_cpu,
            mask_dcff_hw,
            latent_channels,
            'Latent Channels (top variance)',
            camera_dir / 'latent_channels.png',
            cols=args.channel_cols,
        )
        save_channel_grid(
            fine_dcff_cpu,
            mask_dcff_hw,
            dcff_fine_channels,
            'DCFF Fine Channels (top variance)',
            camera_dir / 'dcff_fine_channels.png',
            cols=args.channel_cols,
        )
        save_channel_grid(
            coarse_dcff_cpu,
            mask_dcff_hw,
            dcff_coarse_channels,
            'DCFF Coarse Channels (top variance)',
            camera_dir / 'dcff_coarse_channels.png',
            cols=args.channel_cols,
        )
        save_channel_grid(
            fine_explicit_cpu,
            mask_explicit_hw,
            explicit_fine_channels,
            'Explicit 2DGS Fine Channels (top variance)',
            camera_dir / 'explicit_fine_channels.png',
            cols=args.channel_cols,
        )
        save_channel_grid(
            coarse_explicit_cpu,
            mask_explicit_hw,
            explicit_coarse_channels,
            'Explicit 2DGS Coarse Channels (top variance)',
            camera_dir / 'explicit_coarse_channels.png',
            cols=args.channel_cols,
        )
        save_explicit_render_channels(
            camera_dir / 'explicit_render_channels.png',
            explicit_result['render_rgb'].cpu(),
            explicit_result['render_depth'].cpu(),
            explicit_result['render_alpha'].cpu(),
            explicit_result['render_normal'],
            explicit_result['surf_normal'],
            explicit_result['render_dist'].cpu() if explicit_result['render_dist'] is not None else None,
        )

        if query_fine_pca is not None and query_coarse_pca is not None:
            save_triplet(
                query_fine_pca,
                [
                    f'Query Fine ({query_fine_cos:.3f})',
                    'Teacher Fine',
                ],
                'Query Fine Features (joint PCA basis)',
                camera_dir / 'query_fine_pca_pair.png',
            )
            save_triplet(
                query_coarse_pca,
                [
                    f'Query Coarse ({query_coarse_cos:.3f})',
                    'Teacher Coarse',
                ],
                'Query Coarse Features (joint PCA basis)',
                camera_dir / 'query_coarse_pca_pair.png',
            )
            save_channel_grid(
                query_fine_cpu,
                None,
                query_fine_channels,
                'Query Fine Channels (top variance)',
                camera_dir / 'query_fine_channels.png',
                cols=args.channel_cols,
            )
            save_channel_grid(
                query_coarse_cpu,
                None,
                query_coarse_channels,
                'Query Coarse Channels (top variance)',
                camera_dir / 'query_coarse_channels.png',
                cols=args.channel_cols,
            )

        fine_text = (
            f'fid={fid}\n'
            f'image={cam.image_name}\n\n'
            f'DCFF fine cos:     {dcff_fine_cos:.4f}\n'
            f'Explicit fine cos: {explicit_fine_cos:.4f}\n\n'
            f'DCFF ch:     {dcff_fine_channels}\n'
            f'Explicit ch: {explicit_fine_channels}'
        )
        coarse_text = (
            f'DCFF coarse cos:     {dcff_coarse_cos:.4f}\n'
            f'Explicit coarse cos: {explicit_coarse_cos:.4f}\n\n'
            f'Latent ch:   {latent_channels}\n'
            f'DCFF ch:     {dcff_coarse_channels}\n'
            f'Explicit ch: {explicit_coarse_channels}'
        )
        save_overview(
            camera_dir / 'overview.png',
            gt_rgb.detach().cpu(),
            result['depth'].detach().cpu(),
            result['alpha'].detach().cpu(),
            latent_pca,
            fine_pca,
            coarse_pca,
            fine_text,
            coarse_text,
        )

        if query_fine_pca is not None and query_coarse_pca is not None:
            query_fine_text = (
                f'Query fine cos:   {query_fine_cos:.4f}\n'
                f'Query ch:         {query_fine_channels}'
            )
            query_coarse_text = (
                f'Query coarse cos: {query_coarse_cos:.4f}\n'
                f'Query ch:         {query_coarse_channels}'
            )
            save_query_overview(
                camera_dir / 'query_overview.png',
                gt_rgb.detach().cpu(),
                query_fine_pca,
                query_coarse_pca,
                query_fine_cos_map.detach().cpu(),
                query_coarse_cos_map.detach().cpu(),
                query_fine_text,
                query_coarse_text,
            )

        metadata = {
            'camera_index': camera_idx,
            'camera_uid': cam.uid,
            'image_name': cam.image_name,
            'fid': fid,
            'dcff_fine_cos': dcff_fine_cos,
            'dcff_coarse_cos': dcff_coarse_cos,
            'explicit_fine_cos': explicit_fine_cos,
            'explicit_coarse_cos': explicit_coarse_cos,
            'latent_channels': latent_channels,
            'dcff_fine_channels': dcff_fine_channels,
            'dcff_coarse_channels': dcff_coarse_channels,
            'explicit_fine_channels': explicit_fine_channels,
            'explicit_coarse_channels': explicit_coarse_channels,
            'query_fine_cos': query_fine_cos,
            'query_coarse_cos': query_coarse_cos,
            'query_fine_channels': query_fine_channels,
            'query_coarse_channels': query_coarse_channels,
        }
        with (camera_dir / 'metrics.json').open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        results.append(metadata)

        print(
            f"[cam {camera_idx}] {cam.image_name} | "
            f"dcff fine={dcff_fine_cos:.4f}, coarse={dcff_coarse_cos:.4f} | "
            f"explicit fine={explicit_fine_cos:.4f}, coarse={explicit_coarse_cos:.4f} | "
            f"query fine={query_fine_cos:.4f}, coarse={query_coarse_cos:.4f}"
        )
        torch.cuda.empty_cache()

    def metric_mean(key: str) -> float:
        values = [item[key] for item in results if key in item and not math.isnan(item[key])]
        return float(np.mean(values)) if values else float('nan')

    summary = {
        'winner_experiment': winner_name,
        'winner_meta': winner_meta,
        'dcff_checkpoint': str(ckpt_path),
        'camera_split': args.camera_split,
        'source_dir': str(source_dir),
        'feature_dir': str(feature_dir),
        'query_feature_dir': str(query_feature_dir) if query_feature_dir is not None else None,
        'explicit_point_cloud': str(explicit_ply),
        'explicit_fine': str(explicit_fine),
        'explicit_coarse': str(explicit_coarse),
        'feat_height': feat_h,
        'feat_width': feat_w,
        'longest_edge': longest_edge,
        'split_counts': split_meta,
        'selected_camera_indices': camera_indices,
        'num_cameras_rendered': len(results),
        'mean_metrics': {
            'dcff_fine_cos': metric_mean('dcff_fine_cos'),
            'dcff_coarse_cos': metric_mean('dcff_coarse_cos'),
            'explicit_fine_cos': metric_mean('explicit_fine_cos'),
            'explicit_coarse_cos': metric_mean('explicit_coarse_cos'),
            'query_fine_cos': metric_mean('query_fine_cos'),
            'query_coarse_cos': metric_mean('query_coarse_cos'),
        },
        'cameras': results,
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_json', default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--source_dir', default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument('--feature_dir', default=str(DEFAULT_FEATURE_DIR))
    parser.add_argument(
        '--query_feature_dir',
        default=str(DEFAULT_QUERY_FEATURE_DIR),
        help='Optional exported query-student fine_geo/coarse_sem directory (set empty to disable)',
    )
    parser.add_argument('--explicit_ply', default=str(DEFAULT_EXPLICIT_PLY))
    parser.add_argument('--explicit_fine', default=str(DEFAULT_EXPLICIT_FINE))
    parser.add_argument('--explicit_coarse', default=str(DEFAULT_EXPLICIT_COARSE))
    parser.add_argument('--camera_split', default='test', choices=['train', 'test', 'all'])
    parser.add_argument('--camera_indices', default=None, help='Comma-separated indices within the selected split')
    parser.add_argument('--max_cameras', type=int, default=6)
    parser.add_argument('--num_channels', type=int, default=8)
    parser.add_argument('--channel_cols', type=int, default=4)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available but the script was asked to use it.')

    output_dir = process_split(args)
    print(f'Rich reconstruction visuals saved to: {output_dir}')


if __name__ == '__main__':
    main()
