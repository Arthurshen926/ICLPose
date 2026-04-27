#!/usr/bin/env python3
"""
Feature Visualization: 2DGS Explicit vs DCFF Implicit Feature Fields
===================================================================
Renders and visualizes features from both explicit (2DGS) and implicit (DCFF) representations.

2DGS Explicit Features:
  - Loads Gaussian PLY + feature checkpoints from joint training
  - Renders via rasterization_2dgs with channel chunking
  - Compares against pre-extracted RADIO teacher features

DCFF Implicit Features:
  - Uses visualize_reconstruction.py approach
  - Loads hash grid + fine/coarse decoder
  - Compares against RADIO teacher features

Output:
  - PCA colorized feature maps
  - Cosine similarity heatmaps
  - Per-channel statistics
  - Summary comparison across experiments
"""

import os
import sys
import argparse
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_field.dcff.hybrid_gaussian import HybridGaussianModel
from feature_field.dcff.hash_grid import SpatialHashGrid
from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer
from feature_field.dcff.feature_selection import FeatureSelectionModule
from feature_field.dcff.radio_teacher import CachedFeatureTeacher
from feature_field.runtime import DepthGuidedRefiner, FeatSharp
from feature_field.utils.checkpoint_io import safe_torch_load
from feature_field.utils.dcff_eval_targets import (
    build_teacher_target_provider,
    infer_feature_dims,
)
from feature_field.utils.region_metrics import (
    compute_region_masks,
    empty_region_score_dict,
    extend_region_scores,
    score_regions,
    summarize_region_scores,
)
from feature_field.utils.scene_colmap import load_scene_colmap, build_da3_image_order


def subtract_position_mean(feat, position_mean):
    """Remove per-spatial-position mean feature before PCA visualization."""
    if feat is None or position_mean is None:
        return feat
    if feat.dim() == 4:
        return feat - position_mean.unsqueeze(0).to(device=feat.device, dtype=feat.dtype)
    return feat - position_mean.to(device=feat.device, dtype=feat.dtype)


def pca_colorize_torch(feat, n_components=3, mask=None, return_proj=False):
    """PCA colorization for [C, H, W] tensor."""
    C, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).T.float()
    mask_flat = None if mask is None else mask.reshape(-1)
    fit_pixels = feat_flat if mask_flat is None else feat_flat[mask_flat]

    if fit_pixels.numel() == 0:
        img = torch.zeros(3, H, W, device=feat.device)
        return (img, None, None) if return_proj else img

    mean = fit_pixels.mean(dim=0, keepdim=True)
    centered = fit_pixels - mean

    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        basis = Vh[:n_components]
    except Exception:
        basis = None

    if basis is not None:
        proj = (feat_flat - mean) @ basis.T
    else:
        proj = feat_flat[:, :n_components]

    valid_proj = proj if mask_flat is None else proj[mask_flat]
    mins = valid_proj.min(dim=0).values
    maxs = valid_proj.max(dim=0).values

    proj = proj.clone()
    for i in range(n_components):
        if maxs[i] > mins[i]:
            proj[:, i] = (proj[:, i] - mins[i]) / (maxs[i] - mins[i])
        else:
            proj[:, i] = 0.5

    img = proj.T.reshape(3, H, W)
    if mask is not None:
        img = img * mask.unsqueeze(0)
    
    if return_proj:
        return img, proj, basis
    return img


def joint_pca_colorize(feats, n_components=3, mask=None):
    """Colorize multiple feature maps with the same PCA basis."""
    valid_feats = [f for f in feats if f is not None]
    if not valid_feats:
        return [None for _ in feats]

    C = valid_feats[0].shape[0]
    mask_flat = None if mask is None else mask.reshape(-1)
    stacked = []
    for feat in valid_feats:
        feat_flat = feat.reshape(C, -1).T.float()
        feat_flat = feat_flat if mask_flat is None else feat_flat[mask_flat]
        if feat_flat.numel() > 0:
            stacked.append(feat_flat)
    
    if not stacked:
        return [torch.zeros(3, feats[0].shape[1], feats[0].shape[2], device=feats[0].device) 
                if f is not None else None for f in feats]

    stacked = torch.cat(stacked, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean

    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        basis = Vh[:n_components]
    except Exception:
        basis = None

    all_proj = []
    all_valid = []
    for feat in feats:
        if feat is None:
            all_proj.append(None)
            continue
        feat_flat = feat.reshape(C, -1).T.float()
        proj = (feat_flat - mean) @ basis.T if basis is not None else feat_flat[:, :n_components]
        all_proj.append(proj)
        all_valid.append(proj if mask_flat is None else proj[mask_flat])

    stacked_proj = torch.cat(all_valid, dim=0)
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
        img = proj.T.reshape(3, feat.shape[1], feat.shape[2])
        if mask is not None:
            img = img * mask.unsqueeze(0)
        outputs.append(img)
    return outputs


def target_basis_pca_colorize(
    pred_feats,
    target_feats,
    n_components=3,
    mask=None,
    normalize_inputs=True,
):
    """Colorize predictions and targets with a PCA basis fitted on targets only.

    This keeps predicted and teacher feature colors comparable. The target
    projections also define the normalization range; predictions are projected
    into that same range instead of getting their own contrast stretch.
    """
    if len(pred_feats) != len(target_feats):
        raise ValueError("pred_feats and target_feats must have the same length")

    all_feats = [f for pair in zip(pred_feats, target_feats) for f in pair if f is not None]
    if not all_feats:
        return [None for _ in pred_feats], [None for _ in target_feats]

    first = all_feats[0]
    C = first.shape[0]
    H = first.shape[1]
    W = first.shape[2]
    comp = min(n_components, C)
    mask_flat = None if mask is None else mask.reshape(-1).bool()

    def _prepare(feat):
        feat = feat.float()
        if normalize_inputs:
            feat = F.normalize(feat.unsqueeze(0), p=2, dim=1).squeeze(0)
        return feat

    target_pixels = []
    for feat in target_feats:
        if feat is None:
            continue
        if feat.shape[0] != C:
            raise ValueError("all features must share channel dimension")
        feat = _prepare(feat)
        flat = feat.reshape(C, -1).T.float()
        if mask_flat is not None:
            flat = flat[mask_flat]
        if flat.numel() > 0:
            target_pixels.append(flat)

    if not target_pixels:
        zero = torch.zeros(n_components, H, W, device=first.device)
        return (
            [zero.clone() if f is not None else None for f in pred_feats],
            [zero.clone() if f is not None else None for f in target_feats],
        )

    target_stack = torch.cat(target_pixels, dim=0)
    mean = target_stack.mean(dim=0, keepdim=True)
    centered = target_stack - mean

    basis = None
    if centered.shape[0] >= 2 and comp > 0:
        try:
            cov = centered.T @ centered
            cov = cov / max(1, centered.shape[0] - 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            order = torch.argsort(eigvals, descending=True)
            basis = eigvecs[:, order[:comp]].T
        except Exception:
            basis = None

    if basis is None:
        basis = torch.eye(C, device=target_stack.device, dtype=target_stack.dtype)[:comp]

    target_proj_for_range = (target_stack - mean) @ basis.T
    mins = target_proj_for_range.min(dim=0).values
    maxs = target_proj_for_range.max(dim=0).values

    def _project(feat):
        if feat is None:
            return None
        feat = _prepare(feat)
        flat = feat.reshape(C, -1).T.float()
        proj = (flat - mean.to(flat.device)) @ basis.to(flat.device).T
        proj = proj.clone()
        mins_dev = mins.to(proj.device)
        maxs_dev = maxs.to(proj.device)
        for i in range(comp):
            if maxs_dev[i] > mins_dev[i]:
                proj[:, i] = (proj[:, i] - mins_dev[i]) / (maxs_dev[i] - mins_dev[i])
            else:
                proj[:, i] = 0.5
        if comp < n_components:
            pad = torch.full(
                (proj.shape[0], n_components - comp),
                0.5,
                device=proj.device,
                dtype=proj.dtype,
            )
            proj = torch.cat([proj, pad], dim=1)
        img = proj[:, :n_components].T.reshape(n_components, feat.shape[1], feat.shape[2])
        if mask is not None:
            img = img * mask.to(device=img.device, dtype=img.dtype).unsqueeze(0)
        return img

    return [_project(f) for f in pred_feats], [_project(f) for f in target_feats]


def compute_position_mean(features):
    """Average feature per spatial location over a list of [C,H,W] tensors."""
    valid = [f.float() for f in features if f is not None]
    if not valid:
        return None
    return torch.stack(valid, dim=0).mean(dim=0)


def cosine_similarity_map(pred, target, mask=None):
    """Per-pixel cosine similarity."""
    pred_n = F.normalize(pred, p=2, dim=1)
    target_n = F.normalize(target, p=2, dim=1)
    cos = (pred_n * target_n).sum(dim=1)
    if mask is not None:
        cos = cos * mask.squeeze(1)
    return cos


def sim_to_heatmap(sim_hw):
    """[H, W] similarity → [H, W, 3] RGB heatmap."""
    sim = sim_hw.detach().cpu().numpy()
    sim = np.clip(sim, 0, 1)
    r = np.clip(2 * sim - 1, 0, 1)
    g = np.where(sim < 0.5, 2 * sim, 2 * (1 - sim))
    b = np.clip(1 - 2 * sim, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _load_image_tensor(cam, longest_edge, device):
    img = Image.open(cam.image).convert('RGB')
    width, height = img.size
    scale = longest_edge / max(width, height)
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


def _cam_to_viewmat(cam, device):
    w2c = np.eye(4)
    w2c[:3, :3] = cam.R.T
    w2c[:3, 3] = cam.T
    return torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)


def _cam_to_K(cam, width, height, device):
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    return torch.tensor(
        [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _ensure_bchw_alpha(alpha):
    if alpha is None:
        return None
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        return alpha.permute(0, 3, 1, 2)
    if alpha.ndim == 3:
        return alpha.unsqueeze(1)
    return alpha


def _ensure_bchw_depth(depth):
    if depth is None:
        return None
    if depth.ndim == 4 and depth.shape[-1] == 1:
        return depth.permute(0, 3, 1, 2)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    return depth


def _compute_scene_extent_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    return float(np.percentile(np.linalg.norm(xyz, axis=1), 99)) * 1.2


def _restore_gaussians_from_checkpoint(gaussians, ckpt):
    if 'gaussians_state' not in ckpt:
        return

    gs = ckpt['gaussians_state']
    device = gaussians._xyz.device
    with torch.no_grad():
        gaussians._xyz.data.copy_(gs['_xyz'].to(device))
        gaussians._rotation.data.copy_(gs['_rotation'].to(device))
        gaussians._scaling.data.copy_(gs['_scaling'].to(device))
        gaussians._opacity.data.copy_(gs['_opacity'].to(device))
        gaussians._features_dc.data.copy_(gs['_features_dc'].to(device))
        gaussians._features_rest.data.copy_(gs['_features_rest'].to(device))
        if 'latent' in ckpt and ckpt['latent'].shape == gaussians._latent.shape:
            gaussians._latent.data.copy_(ckpt['latent'].to(device))
    gaussians.active_sh_degree = gs.get('active_sh_degree', gaussians.active_sh_degree)


def _load_checkpoint_with_retry(ckpt_path, attempts=5, sleep_s=2.0):
    last_err = None
    for attempt in range(attempts):
        try:
            return safe_torch_load(ckpt_path, map_location='cpu')
        except RuntimeError as err:
            last_err = err
            if 'PytorchStreamReader failed reading zip archive' not in str(err) or attempt == attempts - 1:
                raise
            time.sleep(sleep_s)
    raise last_err


def _build_dcff_eval_components(ckpt_path, device):
    ckpt = _load_checkpoint_with_retry(ckpt_path)
    cfg = ckpt.get('config', {})
    mcfg = cfg.get('model', {})
    tcfg = cfg.get('training', {})
    hcfg = cfg.get('hash_grid', {})
    fcfg = cfg.get('fine_decoder', {})
    ccfg = cfg.get('coarse_decoder', {})
    refiner_cfg = cfg.get('refiner', {})
    fsm_cfg = cfg.get('fsm', {})

    fine_feat_dim, coarse_feat_dim = infer_feature_dims(cfg)
    feat_dim = fine_feat_dim
    latent_dim = int(mcfg.get('latent_dim', 32))
    sh_degree = int(mcfg.get('sh_degree', 3))
    fine_latent_dim = mcfg.get('fine_latent_dim')
    coarse_latent_dim = mcfg.get('coarse_latent_dim')

    ckpt_ply = str(Path(ckpt_path).with_suffix('.ply'))
    init_ply = tcfg.get('init_ply')
    ply_path = ckpt_ply if os.path.exists(ckpt_ply) else init_ply
    if ply_path is None or not os.path.exists(ply_path):
        raise FileNotFoundError(f"No PLY found for checkpoint: {ckpt_path}")

    gaussians = HybridGaussianModel(sh_degree=sh_degree, latent_dim=latent_dim)
    gaussians.load_ply(ply_path, freeze_geometry=True)
    gaussians.active_sh_degree = sh_degree
    _restore_gaussians_from_checkpoint(gaussians, ckpt)

    scene_extent = _compute_scene_extent_from_ply(init_ply) if init_ply and os.path.exists(init_ply) else float(
        np.percentile(np.linalg.norm(gaussians.get_xyz.detach().cpu().numpy(), axis=1), 99)
    ) * 1.2

    hash_latent_dim = int(coarse_latent_dim) if coarse_latent_dim is not None else latent_dim

    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent,
        feature_dim=coarse_feat_dim,
        input_mode=hcfg.get('input_mode', 'implicit_scale'),
        latent_dim=hash_latent_dim,
        scale_dim=hcfg.get('scale_dim', 2),
        scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
        n_levels=hcfg.get('n_levels', 16),
        n_features_per_level=hcfg.get('n_features_per_level', 2),
        log2_hashmap_size=hcfg.get('log2_hashmap_size', 19),
        base_resolution=hcfg.get('base_resolution', 16),
        max_resolution=hcfg.get('max_resolution', 2048),
        sh_degree=hcfg.get('sh_degree', sh_degree),
        mlp_hidden=hcfg.get('mlp_hidden', 256),
        mlp_layers=hcfg.get('mlp_layers', 4),
        forward_chunk_size=hcfg.get('forward_chunk_size', 0),
    ).to(device)
    hash_grid.load_state_dict(ckpt['hash_grid_state'])

    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=latent_dim,
        fine_latent_dim=int(fine_latent_dim) if fine_latent_dim is not None else None,
        coarse_latent_dim=int(coarse_latent_dim) if coarse_latent_dim is not None else None,
        fine_feature_dim=fine_feat_dim,
        coarse_feature_dim=coarse_feat_dim,
        fine_hidden_dim=fcfg.get('hidden_dim', 256),
        fine_num_layers=fcfg.get('num_layers', 5),
        fine_use_viewdirs=fcfg.get('use_viewdirs', True),
        fine_view_degree=fcfg.get('view_degree', 2),
        fine_decoder_type=fcfg.get('type', 'pointwise'),
        coarse_mode=ccfg.get('mode', 'implicit_only'),
        coarse_carrier_hidden_dim=ccfg.get('carrier_hidden_dim'),
        coarse_gate_hidden_dim=ccfg.get('gate_hidden_dim'),
        coarse_forward_batch_chunk_size=ccfg.get('forward_batch_chunk_size', 0),
        coarse_smoothing_kernel=mcfg.get('coarse_smoothing_kernel', 1),
    ).to(device)
    renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
    if renderer.coarse_carrier_fusion is not None and 'coarse_fusion_state' in ckpt:
        renderer.coarse_carrier_fusion.load_state_dict(ckpt['coarse_fusion_state'])

    feat_sharp_fine = None
    refiner_type = refiner_cfg.get('type')
    if refiner_type == 'depth_guided':
        feat_sharp_fine = DepthGuidedRefiner(
            fine_feat_dim,
            hidden_dim=int(refiner_cfg.get('hidden_dim', 128)),
        ).to(device)
    elif refiner_type == 'featsharp':
        feat_sharp_fine = FeatSharp(fine_feat_dim).to(device)
    if feat_sharp_fine is not None and 'feat_sharp_fine_state' in ckpt:
        feat_sharp_fine.load_state_dict(ckpt['feat_sharp_fine_state'])
        feat_sharp_fine.eval()

    feat_select = None
    if fsm_cfg.get('enable', False):
        if fine_feat_dim != coarse_feat_dim:
            raise ValueError("FSM eval requires equal fine/coarse feature dims")
        feat_select = FeatureSelectionModule(
            feature_dim=fine_feat_dim,
            hidden_dim=int(fsm_cfg.get('hidden_dim', 32)),
            num_heads=int(fsm_cfg.get('num_heads', 4)),
            channel_routing_mode=fsm_cfg.get('channel_routing_mode', 'categorical'),
            use_channel_select=fsm_cfg.get('use_channel_select', True),
            use_cross_attn=fsm_cfg.get('use_cross_attn', True),
            use_spatial_conf=fsm_cfg.get('use_spatial_conf', True),
        ).to(device)
        if 'fsm_state' in ckpt:
            feat_select.load_state_dict(ckpt['fsm_state'])
        feat_select.eval()

    hash_grid.eval()
    renderer.eval()

    return {
        'ckpt': ckpt,
        'cfg': cfg,
        'ply_path': ply_path,
        'gaussians': gaussians,
        'hash_grid': hash_grid,
        'renderer': renderer,
        'feat_sharp_fine': feat_sharp_fine,
        'feat_select': feat_select,
        'feat_dim': feat_dim,
        'fine_feat_dim': fine_feat_dim,
        'coarse_feat_dim': coarse_feat_dim,
        'latent_dim': latent_dim,
        'longest_edge': int(tcfg.get('longest_edge', 960)),
        'coarse_downsample': bool(tcfg.get('coarse_downsample', False)),
    }


def _apply_dcff_postprocess(result, feat_h, feat_w, feat_sharp_fine=None, feat_select=None):
    result = dict(result)
    if feat_sharp_fine is not None:
        depth_f = F.interpolate(result['depth'], (feat_h, feat_w), mode='bilinear', align_corners=False)
        alpha_f = F.interpolate(result['alpha'], (feat_h, feat_w), mode='bilinear', align_corners=False)
        result['fine_features'] = feat_sharp_fine(result['fine_features'], depth=depth_f, alpha=alpha_f)

    if feat_select is not None and result.get('coarse_features') is not None:
        depth_f = F.interpolate(result['depth'], (feat_h, feat_w), mode='bilinear', align_corners=False)
        alpha_f = F.interpolate(result['alpha'], (feat_h, feat_w), mode='bilinear', align_corners=False)
        coarse_f = result['coarse_features']
        if coarse_f.shape[-2:] != (feat_h, feat_w):
            coarse_f = F.interpolate(coarse_f, (feat_h, feat_w), mode='bilinear', align_corners=False)
        fsm_r = feat_select(
            result['fine_features'],
            coarse_f,
            alpha_f,
            depth_f,
            temperature=0.5,
            hard=False,
        )
        result['fine_features'] = fsm_r['fine_features']
        result['coarse_features'] = fsm_r['coarse_features']

    return result


def _compute_dcff_feature_metrics(result, geo_target, sem_target, coarse_downsample):
    feat_h, feat_w = geo_target.shape[-2:]
    coarse_h, coarse_w = feat_h // 2, feat_w // 2

    fine_pred = result['fine_features']
    if fine_pred.shape[-2:] != (feat_h, feat_w):
        fine_pred = F.interpolate(fine_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)
    alpha_fine = result['alpha']
    if alpha_fine.shape[-2:] != (feat_h, feat_w):
        alpha_fine = F.interpolate(alpha_fine, (feat_h, feat_w), mode='bilinear', align_corners=False)
    mask_fine = (alpha_fine > 0.5).float()
    fine_cos_map = cosine_similarity_map(fine_pred, geo_target, mask_fine)
    fine_cos_mean = fine_cos_map[mask_fine.squeeze(1) > 0.5].mean().item()
    fine_regions, fine_edge_strength = compute_region_masks(depth=result['depth'], alpha=alpha_fine)
    fine_region_scores = score_regions(fine_cos_map, fine_regions)

    coarse_target = sem_target
    if coarse_downsample:
        coarse_target = F.interpolate(sem_target, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
    coarse_pred = result['coarse_features']
    if coarse_pred.shape[-2:] != coarse_target.shape[-2:]:
        coarse_pred = F.interpolate(coarse_pred, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
    alpha_coarse = result['alpha']
    if alpha_coarse.shape[-2:] != coarse_target.shape[-2:]:
        alpha_coarse = F.interpolate(alpha_coarse, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
    mask_coarse = (alpha_coarse > 0.5).float()
    coarse_cos_map = cosine_similarity_map(coarse_pred, coarse_target, mask_coarse)
    coarse_cos_mean = coarse_cos_map[mask_coarse.squeeze(1) > 0.5].mean().item()
    coarse_depth = result['depth']
    if coarse_depth.shape[-2:] != coarse_target.shape[-2:]:
        coarse_depth = F.interpolate(coarse_depth, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
    coarse_regions, coarse_edge_strength = compute_region_masks(depth=coarse_depth, alpha=alpha_coarse)
    coarse_region_scores = score_regions(coarse_cos_map, coarse_regions)

    coarse_pred_vis = coarse_pred
    coarse_target_vis = coarse_target
    coarse_cos_vis = coarse_cos_map
    if coarse_pred_vis.shape[-2:] != (feat_h, feat_w):
        coarse_pred_vis = F.interpolate(coarse_pred_vis, (feat_h, feat_w), mode='bilinear', align_corners=False)
        coarse_target_vis = F.interpolate(coarse_target_vis, (feat_h, feat_w), mode='bilinear', align_corners=False)
        coarse_cos_vis = F.interpolate(coarse_cos_map.unsqueeze(1), (feat_h, feat_w), mode='bilinear', align_corners=False).squeeze(1)

    return {
        'fine_pred': fine_pred,
        'coarse_pred': coarse_pred,
        'fine_cos_map': fine_cos_map,
        'coarse_cos_map': coarse_cos_map,
        'coarse_cos_vis': coarse_cos_vis,
        'coarse_pred_vis': coarse_pred_vis,
        'coarse_target_vis': coarse_target_vis,
        'alpha_fine': alpha_fine,
        'mask_fine': mask_fine,
        'mask_coarse': mask_coarse,
        'fine_cos_mean': fine_cos_mean,
        'coarse_cos_mean': coarse_cos_mean,
        'fine_regions': fine_regions,
        'coarse_regions': coarse_regions,
        'fine_edge_strength': fine_edge_strength,
        'coarse_edge_strength': coarse_edge_strength,
        'fine_region_scores': fine_region_scores,
        'coarse_region_scores': coarse_region_scores,
    }


def visualize_dcff(
    ckpt_path,
    source_dir,
    feature_dir,
    output_dir,
    camera_indices=None,
    camera_split='auto',
    device='cuda',
    exp_name='dcff',
    images_subdir=None,
):
    """Visualize DCFF implicit feature field."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[DCFF Visualize] {exp_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {output_dir}")

    bundle = _build_dcff_eval_components(ckpt_path, device)
    ckpt = bundle['ckpt']
    cfg = bundle['cfg']
    gaussians = bundle['gaussians']
    hash_grid = bundle['hash_grid']
    renderer = bundle['renderer']
    feat_sharp_fine = bundle['feat_sharp_fine']
    feat_select = bundle['feat_select']
    longest_edge = bundle['longest_edge']
    coarse_downsample = bundle['coarse_downsample']

    if images_subdir is None:
        images_subdir = cfg.get('dataset', {}).get('images', '')
    train_cams, test_cams, _, _, cameras_extent = load_scene_colmap(source_dir, images_subdir)
    teacher_provider = build_teacher_target_provider(
        cfg,
        ckpt,
        source_dir=source_dir,
        feature_dir=feature_dir,
        device=device,
        images_subdir=images_subdir,
    )

    if camera_split == 'train':
        cam_pool = train_cams
    elif camera_split == 'test':
        cam_pool = test_cams
    elif camera_split == 'all':
        cam_pool = train_cams + test_cams
    else:
        cam_pool = test_cams if test_cams else train_cams

    valid_cams = teacher_provider.filter_cameras(cam_pool)
    if camera_indices is None:
        camera_indices = list(range(min(6, len(valid_cams))))

    all_fine_cos = []
    all_coarse_cos = []
    fine_region_all = empty_region_score_dict()
    coarse_region_all = empty_region_score_dict()

    fine_pred_samples = []
    fine_target_samples = []
    coarse_pred_samples = []
    coarse_target_samples = []
    cached_visuals = []

    hash_grid.eval()
    renderer.eval()
    if feat_sharp_fine is not None:
        feat_sharp_fine.eval()
    if feat_select is not None:
        feat_select.eval()

    print(f"  COLMAP scene: {len(train_cams)} train, {len(test_cams)} test, extent={cameras_extent:.2f}")
    print(f"  PLY: {bundle['ply_path']}")
    print(
        f"  Teacher: {teacher_provider.mode}, dims fine={bundle['fine_feat_dim']} "
        f"coarse={bundle['coarse_feat_dim']}, coarse_downsample={coarse_downsample}, split={camera_split}"
    )

    for idx in camera_indices:
        if idx >= len(valid_cams):
            continue
        cam = valid_cams[idx]
        geo_target, sem_target, fids = teacher_provider.get_batch([cam])
        fid = fids[0]
        feat_h, feat_w = geo_target.shape[-2:]

        gt_rgb = _load_image_tensor(cam, longest_edge, device)
        _, _, H_render, W_render = gt_rgb.shape
        viewmat = _cam_to_viewmat(cam, device)
        K = _cam_to_K(cam, W_render, H_render, device)

        with torch.no_grad():
            result = renderer(
                gaussians,
                viewmat=viewmat,
                K=K,
                width=W_render,
                height=H_render,
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

        rgb = result['rgb']
        depth = result['depth']
        alpha = result['alpha']

        metrics = _compute_dcff_feature_metrics(
            result,
            geo_target,
            sem_target,
            coarse_downsample=coarse_downsample,
        )
        fine_pred_up = metrics['fine_pred']
        coarse_pred_up = metrics['coarse_pred_vis']
        coarse_target_vis = metrics['coarse_target_vis']
        fine_cos = metrics['fine_cos_map']
        coarse_cos = metrics['coarse_cos_vis']
        mask_fine = metrics['mask_fine']
        mask_coarse = F.interpolate(
            metrics['mask_coarse'],
            (feat_h, feat_w),
            mode='nearest',
        ) if metrics['mask_coarse'].shape[-2:] != (feat_h, feat_w) else metrics['mask_coarse']

        fine_cos_mean = metrics['fine_cos_mean']
        coarse_cos_mean = metrics['coarse_cos_mean']
        all_fine_cos.append(fine_cos_mean)
        all_coarse_cos.append(coarse_cos_mean)
        extend_region_scores(fine_region_all, metrics['fine_region_scores'])
        extend_region_scores(coarse_region_all, metrics['coarse_region_scores'])

        fine_pred_chw = fine_pred_up.squeeze(0).detach().cpu()
        fine_target_chw = geo_target.squeeze(0).detach().cpu()
        coarse_pred_chw = coarse_pred_up.squeeze(0).detach().cpu()
        coarse_target_chw = coarse_target_vis.squeeze(0).detach().cpu()
        fine_pred_samples.append(fine_pred_chw)
        fine_target_samples.append(fine_target_chw)
        coarse_pred_samples.append(coarse_pred_chw)
        coarse_target_samples.append(coarse_target_chw)

        cached_visuals.append({
            'idx': idx,
            'fid': fid if fid is not None else cam.image_name,
            'rgb': rgb.detach().cpu(),
            'depth': depth.detach().cpu(),
            'alpha': alpha.detach().cpu(),
            'fine_cos': fine_cos.detach().cpu(),
            'coarse_cos': coarse_cos.detach().cpu(),
            'mask_fine': mask_fine.detach().cpu(),
            'mask_coarse': mask_coarse.detach().cpu(),
            'fine_region_scores': {k: v[0] for k, v in metrics['fine_region_scores'].items()},
            'coarse_region_scores': {k: v[0] for k, v in metrics['coarse_region_scores'].items()},
            'fine_cos_mean': fine_cos_mean,
            'coarse_cos_mean': coarse_cos_mean,
            'fine_valid': fine_cos[mask_fine.squeeze(1) > 0.5].detach().cpu().numpy(),
            'coarse_valid': coarse_cos[mask_coarse.squeeze(1) > 0.5].detach().cpu().numpy(),
        })

        print(f"  Camera {idx}: fine_cos={fine_cos_mean:.4f}, coarse_cos={coarse_cos_mean:.4f}")

    fine_pred_vis_list, fine_target_vis_list = target_basis_pca_colorize(
        fine_pred_samples,
        fine_target_samples,
        mask=None,
    )
    coarse_pred_vis_list, coarse_target_vis_list = target_basis_pca_colorize(
        coarse_pred_samples,
        coarse_target_samples,
        mask=None,
    )

    for item, fine_pred_rgb, fine_target_rgb, coarse_pred_rgb, coarse_target_rgb in zip(
        cached_visuals,
        fine_pred_vis_list,
        fine_target_vis_list,
        coarse_pred_vis_list,
        coarse_target_vis_list,
    ):
        idx = item['idx']
        fid = item['fid']
        rgb = item['rgb']
        depth = item['depth']
        alpha = item['alpha']
        fine_cos = item['fine_cos']
        coarse_cos = item['coarse_cos']
        mask_fine = item['mask_fine']
        mask_coarse = item['mask_coarse']
        fine_cos_mean = item['fine_cos_mean']
        coarse_cos_mean = item['coarse_cos_mean']
        fine_region_scores = item['fine_region_scores']
        coarse_region_scores = item['coarse_region_scores']

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'DCFF {exp_name} - cam={idx} (fid={fid})')

        axes[0, 0].imshow(rgb.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 0].set_title('RGB')
        axes[0, 0].axis('off')

        vmin, vmax = depth.min().item(), depth.max().item()
        axes[0, 1].imshow(depth.squeeze(0).squeeze(0).cpu(), cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Depth')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(fine_pred_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 2].set_title(f'Fine Pred (cos={fine_cos_mean:.3f})')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(fine_target_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 3].set_title('Fine Target (RADIO)')
        axes[0, 3].axis('off')

        axes[1, 0].imshow(alpha.squeeze(0).squeeze(0).cpu(), cmap='gray')
        axes[1, 0].set_title('Alpha')
        axes[1, 0].axis('off')

        cos_display = fine_cos.squeeze(0).cpu().numpy()
        im = axes[1, 1].imshow(cos_display, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Fine Cosine Sim (avg={fine_cos_mean:.3f})')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        axes[1, 1].text(
            0.02,
            0.02,
            f"far-edge={fine_region_scores['far_edge']:.3f}\nedge={fine_region_scores['edge']:.3f}",
            transform=axes[1, 1].transAxes,
            fontsize=9,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2},
        )

        axes[1, 2].imshow(coarse_pred_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1, 2].set_title(f'Coarse Pred (cos={coarse_cos_mean:.3f})')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(coarse_target_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1, 3].set_title('Coarse Target (RADIO)')
        axes[1, 3].axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'dcff_{exp_name}_cam{idx}.png'), dpi=150)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fine_valid = item['fine_valid']
        coarse_valid = item['coarse_valid']

        axes2[0].hist(fine_valid, bins=50, alpha=0.7, color='blue')
        axes2[0].axvline(fine_cos_mean, color='red', linestyle='--')
        axes2[0].set_title(f'Fine Cosine Dist (mean={fine_cos_mean:.3f})')
        axes2[0].set_xlabel('Cosine Similarity')

        axes2[1].hist(coarse_valid, bins=50, alpha=0.7, color='green')
        axes2[1].axvline(coarse_cos_mean, color='red', linestyle='--')
        axes2[1].set_title(f'Coarse Cosine Dist (mean={coarse_cos_mean:.3f})')
        axes2[1].set_xlabel('Cosine Similarity')

        cos_display_c = coarse_cos.squeeze(0).cpu().numpy()
        im3 = axes2[2].imshow(cos_display_c, cmap='RdYlGn', vmin=0, vmax=1)
        axes2[2].set_title(f"Coarse Cosine Map\nfar-edge={coarse_region_scores['far_edge']:.3f}")
        axes2[2].axis('off')
        plt.colorbar(im3, ax=axes2[2])

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f'dcff_{exp_name}_cam{idx}_dist.png'), dpi=150)
        plt.close(fig2)

    fine_mean = np.mean(all_fine_cos) if all_fine_cos else 0
    coarse_mean = np.mean(all_coarse_cos) if all_coarse_cos else 0
    print(f"  [{exp_name}] Summary: fine_cos={fine_mean:.4f}, coarse_cos={coarse_mean:.4f}")
    
    return {
        'fine_cos': fine_mean,
        'coarse_cos': coarse_mean,
        'fine_cos_std': np.std(all_fine_cos) if all_fine_cos else 0,
        'coarse_cos_std': np.std(all_coarse_cos) if all_coarse_cos else 0,
        'fine_region_cos': summarize_region_scores(fine_region_all),
        'coarse_region_cos': summarize_region_scores(coarse_region_all),
    }


def visualize_2dgs_explicit(
    ply_path,
    fine_feat_path,
    coarse_feat_path,
    source_dir,
    feature_dir,
    output_dir,
    camera_indices=None,
    camera_split='all',
    device='cuda',
    exp_name='2dgs',
):
    """Visualize 2DGS explicit feature field.
    
    Features are stored per-Gaussian in best_model.pth files.
    We load them and render via rasterization_2dgs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[2DGS Visualize] {exp_name}")
    print(f"  PLY: {ply_path}")
    print(f"  Fine features: {fine_feat_path}")
    print(f"  Coarse features: {coarse_feat_path}")
    print(f"  Output: {output_dir}")

    from feature_gaussian.legacy_3dgs.train_2dgs_joint import (
        GaussianModel2DGS, load_scene_joint, render_features_2dgs, render_rgb_2dgs
    )

    all_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene_joint(source_dir)
    train_cams, test_cams, _, _, _ = load_scene_colmap(source_dir, '')
    da3_name_to_fid = build_da3_image_order(source_dir)

    radio_cache = CachedFeatureTeacher(feature_dir)
    feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w

    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = 3

    fine_ckpt = torch.load(fine_feat_path, map_location='cpu')
    feat_fine_gauss = fine_ckpt['loc_feature'].float()
    N_gauss = gaussians.get_xyz.shape[0]
    feat_fine_gauss = feat_fine_gauss[:N_gauss].to(device)

    coarse_ckpt = torch.load(coarse_feat_path, map_location='cpu')
    feat_coarse_gauss = coarse_ckpt['loc_feature'].float()
    feat_coarse_gauss = feat_coarse_gauss[:N_gauss].to(device)

    cam_to_fid = {}
    for cam in all_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in radio_cache.frame_ids:
            cam_to_fid[cam.uid] = fid

    train_names = {c.image_name for c in train_cams}
    test_names = {c.image_name for c in test_cams}
    if camera_split == 'train':
        valid_cams = [c for c in all_cams if c.uid in cam_to_fid and c.image_name in train_names]
    elif camera_split == 'test':
        valid_cams = [c for c in all_cams if c.uid in cam_to_fid and c.image_name in test_names]
    else:
        valid_cams = [c for c in all_cams if c.uid in cam_to_fid]
    if camera_indices is None:
        camera_indices = list(range(min(6, len(valid_cams))))

    all_fine_cos = []
    all_coarse_cos = []
    fine_region_all = empty_region_score_dict()
    coarse_region_all = empty_region_score_dict()

    fine_pred_samples = []
    fine_target_samples = []
    coarse_pred_samples = []
    coarse_target_samples = []
    cached_visuals = []

    for idx in camera_indices:
        if idx >= len(valid_cams):
            continue
        cam = valid_cams[idx]
        fid = cam_to_fid.get(cam.uid)
        if fid is None:
            continue

        geo_target, sem_target = radio_cache.get(fid)
        geo_target = geo_target.unsqueeze(0).to(device)
        sem_target = sem_target.unsqueeze(0).to(device)

        img = Image.open(cam.image).convert('RGB')
        W_img, H_img = img.size
        scale = 960 / max(W_img, H_img)
        if scale < 1.0:
            img = img.resize((int(W_img * scale), int(H_img * scale)), Image.LANCZOS)
        img_np = np.array(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        gt_rgb = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt_rgb = gt_rgb.to(device)

        W2C = np.eye(4)
        W2C[:3, :3] = cam.R.T
        W2C[:3, 3] = cam.T
        viewmat = torch.tensor(W2C, dtype=torch.float32, device=device)
        tanfovx = np.tan(cam.FovX * 0.5)
        tanfovy = np.tan(cam.FovY * 0.5)
        _, _, H_render, W_render = gt_rgb.shape
        fx, fy = W_render / (2 * tanfovx), H_render / (2 * tanfovy)
        K = torch.tensor([[fx, 0, W_render / 2], [0, fy, H_render / 2], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        with torch.no_grad():
            fine_pred_raw = render_features_2dgs(gaussians, viewmat, feat_fine_gauss, H_render, W_render, K)
            coarse_pred_raw = render_features_2dgs(gaussians, viewmat, feat_coarse_gauss, H_render, W_render, K)
            
            fine_pred_raw = fine_pred_raw.unsqueeze(0)
            coarse_pred_raw = coarse_pred_raw.unsqueeze(0)
            render_pkg = render_rgb_2dgs(
                gaussians,
                cam,
                bg_color=torch.zeros(3, device=device),
                longest_edge=960,
            )
            depth_map = _ensure_bchw_depth(render_pkg['depth'])
            alpha_map = _ensure_bchw_alpha(render_pkg['rend_alpha'])
        
        fine_pred_up = F.interpolate(fine_pred_raw, (feat_h, feat_w), mode='bilinear', align_corners=False)
        coarse_pred_up = F.interpolate(coarse_pred_raw, (feat_h, feat_w), mode='bilinear', align_corners=False)
        
        fine_pred_up = F.normalize(fine_pred_up, p=2, dim=1)
        coarse_pred_up = F.normalize(coarse_pred_up, p=2, dim=1)

        alpha_fine = F.interpolate(alpha_map, (feat_h, feat_w), mode='bilinear', align_corners=False)
        depth_fine = F.interpolate(depth_map, (feat_h, feat_w), mode='bilinear', align_corners=False)
        mask_fine = (alpha_fine > 0.5).float()
        mask_coarse = mask_fine.clone()

        fine_cos = cosine_similarity_map(fine_pred_up, geo_target, mask_fine)
        coarse_cos = cosine_similarity_map(coarse_pred_up, sem_target, mask_coarse)
        region_masks, _ = compute_region_masks(depth=depth_fine, alpha=alpha_fine)
        fine_region_scores = score_regions(fine_cos, region_masks)
        coarse_region_scores = score_regions(coarse_cos, region_masks)

        valid_mask = mask_fine.squeeze(1) > 0.5
        if valid_mask.sum() > 0:
            fine_cos_mean = fine_cos[valid_mask].mean().item()
        else:
            fine_cos_mean = float('nan')
        
        valid_mask_c = mask_coarse.squeeze(1) > 0.5
        if valid_mask_c.sum() > 0:
            coarse_cos_mean = coarse_cos[valid_mask_c].mean().item()
        else:
            coarse_cos_mean = float('nan')
        
        all_fine_cos.append(fine_cos_mean)
        all_coarse_cos.append(coarse_cos_mean)
        extend_region_scores(fine_region_all, fine_region_scores)
        extend_region_scores(coarse_region_all, coarse_region_scores)

        print(f"  Camera {idx}: fine_cos={fine_cos_mean:.4f}, coarse_cos={coarse_cos_mean:.4f}")

        fine_pred_samples.append(fine_pred_up.squeeze(0).detach().cpu())
        fine_target_samples.append(geo_target.squeeze(0).detach().cpu())
        coarse_pred_samples.append(coarse_pred_up.squeeze(0).detach().cpu())
        coarse_target_samples.append(sem_target.squeeze(0).detach().cpu())

        cached_visuals.append({
            'idx': idx,
            'fid': fid,
            'gt_rgb': gt_rgb.detach().cpu(),
            'alpha': alpha_map.detach().cpu(),
            'depth': depth_map.detach().cpu(),
            'fine_cos': fine_cos.detach().cpu(),
            'coarse_cos': coarse_cos.detach().cpu(),
            'fine_region_scores': {k: v[0] for k, v in fine_region_scores.items()},
            'coarse_region_scores': {k: v[0] for k, v in coarse_region_scores.items()},
            'mask_fine': mask_fine.detach().cpu(),
            'mask_coarse': mask_coarse.detach().cpu(),
            'fine_cos_mean': fine_cos_mean,
            'coarse_cos_mean': coarse_cos_mean,
            'fine_valid': fine_cos[mask_fine.squeeze(1) > 0.5].detach().cpu().numpy(),
            'coarse_valid': coarse_cos[mask_coarse.squeeze(1) > 0.5].detach().cpu().numpy(),
        })

    fine_pred_vis_list, fine_target_vis_list = target_basis_pca_colorize(
        fine_pred_samples,
        fine_target_samples,
        mask=None,
    )
    coarse_pred_vis_list, coarse_target_vis_list = target_basis_pca_colorize(
        coarse_pred_samples,
        coarse_target_samples,
        mask=None,
    )

    for item, fine_pred_rgb, fine_target_rgb, coarse_pred_rgb, coarse_target_rgb in zip(
        cached_visuals,
        fine_pred_vis_list,
        fine_target_vis_list,
        coarse_pred_vis_list,
        coarse_target_vis_list,
    ):
        idx = item['idx']
        fid = item['fid']
        gt_rgb = item['gt_rgb']
        alpha_map = item['alpha']
        depth_map = item['depth']
        fine_cos = item['fine_cos']
        coarse_cos = item['coarse_cos']
        fine_cos_mean = item['fine_cos_mean']
        coarse_cos_mean = item['coarse_cos_mean']
        fine_region_scores = item['fine_region_scores']
        coarse_region_scores = item['coarse_region_scores']

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'2DGS {exp_name} - cam={idx} (fid={fid})')

        axes[0, 0].imshow(gt_rgb.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 0].set_title('RGB')
        axes[0, 0].axis('off')

        d = depth_map.squeeze(0).squeeze(0).cpu().numpy()
        valid = d > 0
        if valid.any():
            vmin, vmax = np.percentile(d[valid], [2, 98])
        else:
            vmin, vmax = 0, 1
        axes[0, 1].imshow(d, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Depth')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(fine_pred_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 2].set_title(f'Fine Pred (cos={fine_cos_mean:.3f})')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(fine_target_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0, 3].set_title('Fine Target (RADIO)')
        axes[0, 3].axis('off')

        axes[1, 0].text(0.5, 0.5, f'2DGS Explicit\n{gaussians.get_xyz.shape[0]:,} Gaussians',
                        ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')

        cos_display = fine_cos.squeeze(0).cpu().numpy()
        im = axes[1, 1].imshow(cos_display, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Fine Cosine Sim (avg={fine_cos_mean:.3f})')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        axes[1, 1].text(
            0.02,
            0.02,
            f"far-edge={fine_region_scores['far_edge']:.3f}\nedge={fine_region_scores['edge']:.3f}",
            transform=axes[1, 1].transAxes,
            fontsize=9,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 2},
        )

        axes[1, 2].imshow(coarse_pred_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1, 2].set_title(f'Coarse Pred (cos={coarse_cos_mean:.3f})')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(coarse_target_rgb.permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1, 3].set_title('Coarse Target (RADIO)')
        axes[1, 3].axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'2dgs_{exp_name}_cam{idx}.png'), dpi=150)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fine_valid = item['fine_valid']
        coarse_valid = item['coarse_valid']

        axes2[0].hist(fine_valid, bins=50, alpha=0.7, color='blue')
        axes2[0].axvline(fine_cos_mean, color='red', linestyle='--')
        axes2[0].set_title(f'Fine Cosine Dist (mean={fine_cos_mean:.3f})')
        axes2[0].set_xlabel('Cosine Similarity')

        axes2[1].hist(coarse_valid, bins=50, alpha=0.7, color='green')
        axes2[1].axvline(coarse_cos_mean, color='red', linestyle='--')
        axes2[1].set_title(f'Coarse Cosine Dist (mean={coarse_cos_mean:.3f})')
        axes2[1].set_xlabel('Cosine Similarity')

        cos_display_c = coarse_cos.squeeze(0).cpu().numpy()
        im3 = axes2[2].imshow(cos_display_c, cmap='RdYlGn', vmin=0, vmax=1)
        axes2[2].set_title(f"Coarse Cosine Map\nfar-edge={coarse_region_scores['far_edge']:.3f}")
        axes2[2].axis('off')
        plt.colorbar(im3, ax=axes2[2])

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f'2dgs_{exp_name}_cam{idx}_dist.png'), dpi=150)
        plt.close(fig2)

    fine_mean = np.mean(all_fine_cos) if all_fine_cos else 0
    coarse_mean = np.mean(all_coarse_cos) if all_coarse_cos else 0
    print(f"  [{exp_name}] Summary: fine_cos={fine_mean:.4f}, coarse_cos={coarse_mean:.4f}")
    
    return {
        'fine_cos': fine_mean,
        'coarse_cos': coarse_mean,
        'fine_cos_std': np.std(all_fine_cos) if all_fine_cos else 0,
        'coarse_cos_std': np.std(all_coarse_cos) if all_coarse_cos else 0,
        'fine_region_cos': summarize_region_scores(fine_region_all),
        'coarse_region_cos': summarize_region_scores(coarse_region_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_base', type=str, default='feature_field/output/vis_comparison')
    parser.add_argument('--source_dir', type=str, default='/root/ICLPose-loc/dataset/OldHospital')
    parser.add_argument('--feature_dir', type=str, 
                       default='/root/ICLPose-loc/feature_extract/output/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--images_subdir', type=str, default=None)
    parser.add_argument('--dcff_checkpoint', type=str, default=None)
    parser.add_argument('--dcff_exp_name', type=str, default='dcff_custom')
    parser.add_argument('--camera_indices', type=str, default=None,
                       help='Comma-separated camera indices, e.g., "0,1,2,3,4,5"')
    parser.add_argument('--camera_split', type=str, default='auto', choices=['auto', 'train', 'test', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_dcff', action='store_true')
    parser.add_argument('--skip_2dgs', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)
    
    camera_indices = None
    if args.camera_indices:
        camera_indices = [int(x) for x in args.camera_indices.split(',')]

    results = {}

    if not args.skip_dcff:
        if args.dcff_checkpoint:
            experiments = {args.dcff_exp_name: args.dcff_checkpoint}
        else:
            experiments = {
                'v10c_baseline': 'feature_field/output/dcff_oldhospital_v10c_carrier_residual/checkpoints/best.pth',
                'v12_fsm': 'feature_field/output/dcff_oldhospital_v12_fsm/checkpoints/latest.pth',
                'v13a_no_fsm': 'feature_field/output/dcff_oldhospital_v13a_no_fsm/checkpoints/latest.pth',
                'v13b_fsm_b16': 'feature_field/output/dcff_oldhospital_v13b_fsm_b16_test/checkpoints/latest.pth',
            }
        
        for exp_name, ckpt_path in experiments.items():
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {exp_name}: checkpoint not found at {ckpt_path}")
                continue
            
            output_dir = os.path.join(args.output_base, f'dcff_{exp_name}')
            try:
                result = visualize_dcff(
                    ckpt_path=ckpt_path,
                    source_dir=args.source_dir,
                    feature_dir=args.feature_dir,
                    output_dir=output_dir,
                    camera_indices=camera_indices,
                    camera_split=args.camera_split,
                    device=args.device,
                    exp_name=exp_name,
                    images_subdir=args.images_subdir,
                )
                results[exp_name] = result
            except Exception as e:
                print(f"  [ERROR] {exp_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()

    if not args.skip_2dgs:
        exp_name = 'joint_radio_dual'
        output_dir = os.path.join(args.output_base, f'2dgs_{exp_name}')
        ply_path = 'feature_gaussian/output/joint_radio_dual_oldhospital_pilot/point_cloud/best/point_cloud.ply'
        fine_feat_path = 'feature_gaussian/output/joint_radio_dual_oldhospital_pilot/features/fine_geo/best_model.pth'
        coarse_feat_path = 'feature_gaussian/output/joint_radio_dual_oldhospital_pilot/features/coarse_sem/best_model.pth'
        
        if os.path.exists(ply_path):
            try:
                result = visualize_2dgs_explicit(
                    ply_path=ply_path,
                    fine_feat_path=fine_feat_path,
                    coarse_feat_path=coarse_feat_path,
                    source_dir=args.source_dir,
                    feature_dir=args.feature_dir,
                    output_dir=output_dir,
                    camera_indices=camera_indices,
                    camera_split=args.camera_split,
                    device=args.device,
                    exp_name=exp_name,
                )
                results[exp_name] = result
            except Exception as e:
                print(f"  [ERROR] 2DGS {exp_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [SKIP] 2DGS: PLY not found at {ply_path}")

    print("\n" + "=" * 70)
    print("  FEATURE VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Fine Cos':<12} {'Coarse Cos':<12}")
    print("-" * 70)
    for name, res in sorted(results.items()):
        print(f"{name:<25} {res['fine_cos']:.4f} ± {res.get('fine_cos_std', 0):.3f}   "
              f"{res['coarse_cos']:.4f} ± {res.get('coarse_cos_std', 0):.3f}")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {args.output_base}/")
    print("Use these directories for your quantitative evaluation.")


if __name__ == '__main__':
    main()
