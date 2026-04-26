#!/usr/bin/env python3
"""Multi-seed evaluation for concat localization model.

Runs N evaluations with different noise seeds, reports mean±std statistics.
Uses the same DCFF rendering pipeline as training (including DepthGuidedRefiner).
"""
import argparse
import copy
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_field.dcff import DeferredCascadedRenderer, HybridGaussianModel, SpatialHashGrid
from data.radio_loc_dataset import RadioLocDataset, collate_fn
from pose_refine import load_concat_pose_checkpoint, load_concat_pose_model
from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.utils.geometry_solver import pnp_ransac_solve, feature_metric_solve
from pose_refine.utils.lie_algebra import se3_exp
from feature_field import (
    build_dcff as shared_build_dcff,
    intrinsics_to_K as shared_intrinsics_to_K,
    render_batch as shared_render_batch,
)
from feature_field.runtime import apply_localization_map_state
from feature_field.utils.project_config import load_mainline_config


# ═════════════════════════════════════════════════════════════════════════════
#  FeatSharp / DepthGuidedRefiner (must match train_concat_loc.py definitions)
# ═════════════════════════════════════════════════════════════════════════════

class FeatSharp(nn.Module):
    """Lightweight learnable sharpening applied after rasterization."""

    def __init__(self, feature_dim, kernel_size=3):
        super().__init__()
        self.sharpen = nn.Sequential(
            nn.Conv2d(
                feature_dim, feature_dim, kernel_size,
                padding=kernel_size // 2, groups=feature_dim,
            ),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        for m in self.sharpen:
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if m.kernel_size == (1, 1):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    m.weight.data *= 0.01

    def forward(self, x, depth=None, alpha=None):
        return x + self.sharpen(x)


class DepthGuidedRefiner(nn.Module):
    """Spatial refinement conditioned on depth and alpha."""

    def __init__(self, feature_dim=64, hidden_dim=128):
        super().__init__()
        input_dim = feature_dim + 2
        self.refiner = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, features, depth=None, alpha=None):
        if depth is None or alpha is None:
            return features
        fh, fw = features.shape[-2:]
        if depth.shape[-2:] != (fh, fw):
            depth = F.interpolate(depth, (fh, fw), mode='bilinear', align_corners=False)
        if alpha.shape[-2:] != (fh, fw):
            alpha = F.interpolate(alpha, (fh, fw), mode='bilinear', align_corners=False)
        depth_norm = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        x = torch.cat([features, depth_norm, alpha], dim=1)
        return features + self.refiner(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def intrinsics_to_K(intr, device):
    """Convert intrinsics dict {fx, fy, cx, cy} to 3×3 K matrix."""
    K = torch.tensor([
        [intr['fx'], 0, intr['cx']],
        [0, intr['fy'], intr['cy']],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    return K


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def _deep_update_dict(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_state_dict_compatible(module, state_dict, label, printer=print):
    if not state_dict:
        return False
    try:
        module.load_state_dict(state_dict, strict=True)
        printer(f'  Loaded {label}')
        return True
    except (RuntimeError, KeyError) as e:
        printer(f'  Skipping {label} (architecture changed): {e}')
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  DCFF Loading (mirrors train_concat_loc.py _build_dcff)
# ═════════════════════════════════════════════════════════════════════════════

def build_dcff(config, device):
    """Build and load the frozen DCFF rendering pipeline.

    Returns:
        gaussians: HybridGaussianModel with loaded geometry + latent
        dcff_renderer: DeferredCascadedRenderer with loaded fine decoder
        feat_sharp: DepthGuidedRefiner or FeatSharp with loaded weights
    """
    cfg_dcff = config.get('dcff', {})
    dcff_ckpt_path = cfg_dcff.get('checkpoint')
    joint_ckpt_path = cfg_dcff.get('joint_checkpoint')
    ply_path = cfg_dcff.get('ply_path')

    if not dcff_ckpt_path or not os.path.isfile(dcff_ckpt_path):
        raise FileNotFoundError(f"DCFF checkpoint not found: {dcff_ckpt_path}")
    if not ply_path or not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    latent_dim = cfg_dcff.get('latent_dim', 32)
    feature_dim = cfg_dcff.get('feature_dim', 64)

    # 1. Load 2DGS geometry + latent
    gaussians = HybridGaussianModel(sh_degree=3, latent_dim=latent_dim)
    gaussians.load_ply(ply_path, freeze_geometry=True)
    gaussians.active_sh_degree = 3
    print(f'  Loaded {gaussians.num_points:,} Gaussians from {ply_path}')

    # Scene extent for hash grid
    xyz = gaussians.get_xyz.detach()
    scene_extent = float((xyz.max(dim=0).values - xyz.min(dim=0).values).max()) * 0.6
    print(f'  Scene extent: {scene_extent:.2f}')

    # Load DCFF training config if available (for hash_grid / fine_decoder params)
    dcff_config_path = os.path.join(os.path.dirname(dcff_ckpt_path), '..', 'config.yaml')
    dcff_cfg = {}
    if os.path.isfile(dcff_config_path):
        with open(dcff_config_path) as f:
            dcff_cfg = yaml.safe_load(f) or {}
        print(f'  Loaded DCFF config from {dcff_config_path}')

    fine_decoder_override = cfg_dcff.get('fine_decoder_override')
    if fine_decoder_override:
        dcff_cfg = _deep_update_dict(dcff_cfg, {'fine_decoder': fine_decoder_override})
        print(f'  Overriding fine_decoder config: {fine_decoder_override}')

    refiner_override = cfg_dcff.get('refiner_override')
    if refiner_override:
        dcff_cfg = _deep_update_dict(dcff_cfg, {'refiner': refiner_override})
        print(f'  Overriding refiner config: {refiner_override}')

    hcfg = dcff_cfg.get('hash_grid', {})
    fcfg = dcff_cfg.get('fine_decoder', {})

    # 2. Build hash grid
    input_mode = hcfg.get('input_mode', 'implicit_scale')
    hash_grid = SpatialHashGrid(
        scene_extent=hcfg.get('scene_extent', scene_extent),
        feature_dim=feature_dim,
        input_mode=input_mode,
        latent_dim=latent_dim,
        n_levels=hcfg.get('n_levels', 16),
        n_features_per_level=hcfg.get('n_features_per_level', 2),
        log2_hashmap_size=hcfg.get('log2_hashmap_size', 19),
        base_resolution=hcfg.get('base_resolution', 16),
        max_resolution=hcfg.get('max_resolution', 2048),
        mlp_hidden=hcfg.get('mlp_hidden', 128),
        mlp_layers=hcfg.get('mlp_layers', 2),
        scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
    ).to(device)

    # 3. Build renderer
    dcff_renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=latent_dim,
        fine_feature_dim=feature_dim,
        coarse_feature_dim=feature_dim,
        fine_hidden_dim=fcfg.get('hidden_dim', 128),
        fine_num_layers=fcfg.get('num_layers', 3),
        fine_use_viewdirs=fcfg.get('use_viewdirs', False),
        fine_view_degree=fcfg.get('view_degree', 2),
        fine_decoder_type=fcfg.get('type', 'mlp'),
        coarse_smoothing_kernel=cfg_dcff.get('coarse_smoothing_kernel', 1),
    ).to(device)

    # 4. Feature refinement (auto-detect from DCFF checkpoint or config)
    ckpt = torch.load(dcff_ckpt_path, map_location=device)

    # Detect refiner type: honor explicit config overrides first, fall back to checkpoint auto-detection only
    refiner_type = dcff_cfg.get('refiner', {}).get('type', None)
    fs_state = ckpt.get('feat_sharp_fine_state', {})
    if refiner_type == 'depth_guided':
        refiner_hidden = dcff_cfg.get('refiner', {}).get('hidden_dim', 128)
        feat_sharp = DepthGuidedRefiner(feature_dim, hidden_dim=refiner_hidden).to(device)
        print(f'  Using DepthGuidedRefiner (hidden={refiner_hidden}) [from config]')
    elif refiner_type == 'featsharp':
        feat_sharp = FeatSharp(feature_dim).to(device)
        print('  Using FeatSharp [from config]')
    elif refiner_type is None and 'refiner.0.weight' in fs_state and fs_state['refiner.0.weight'].shape[1] == feature_dim + 2:
        # Auto-detect: input_dim = feature_dim + 2 (depth + alpha) → DepthGuidedRefiner
        hidden_dim = fs_state['refiner.0.weight'].shape[0]
        feat_sharp = DepthGuidedRefiner(feature_dim, hidden_dim=hidden_dim).to(device)
        print(f'  Using DepthGuidedRefiner (hidden={hidden_dim}) [auto-detected]')
    else:
        feat_sharp = FeatSharp(feature_dim).to(device)
        print('  Using FeatSharp')

    # 5. Load checkpoint weights
    print(f'  Loading DCFF checkpoint: {dcff_ckpt_path}')
    hash_grid.load_state_dict(ckpt['hash_grid_state'])
    _load_state_dict_compatible(dcff_renderer.fine_decoder, ckpt.get('fine_decoder_state'), 'fine_decoder weights')
    _load_state_dict_compatible(feat_sharp, fs_state, 'feat_sharp weights')
    if 'latent' in ckpt:
        saved_latent = ckpt['latent'].to(device)
        if saved_latent.shape == gaussians._latent.shape:
            with torch.no_grad():
                gaussians._latent.data.copy_(saved_latent)
            print('  Restored latent embeddings from checkpoint')
        else:
            print(f'  WARNING: Latent shape mismatch: ckpt={saved_latent.shape} '
                  f'vs model={gaussians._latent.shape}, skipping')

    if joint_ckpt_path:
        if not os.path.isfile(joint_ckpt_path):
            raise FileNotFoundError(f"Joint checkpoint not found: {joint_ckpt_path}")
        print(f'  Loading joint feature checkpoint: {joint_ckpt_path}')
        joint_ckpt = torch.load(joint_ckpt_path, map_location=device)
        joint_map_state = joint_ckpt.get('map_renderer_state_dict') or {}

        # Allow config to restrict which components are overridden from joint checkpoint
        allowed = cfg_dcff.get('joint_override_components', None)
        if allowed is not None:
            allowed = set(allowed)
            print(f'  joint_override_components filter: {sorted(allowed)}')

        loaded_components = []
        if 'fine_decoder' in joint_map_state and (allowed is None or 'fine_decoder' in allowed):
            if _load_state_dict_compatible(dcff_renderer.fine_decoder, joint_map_state['fine_decoder'], 'joint fine_decoder'):
                loaded_components.append('fine_decoder')
        if 'feat_sharp' in joint_map_state and (allowed is None or 'feat_sharp' in allowed):
            if _load_state_dict_compatible(feat_sharp, joint_map_state['feat_sharp'], 'joint feat_sharp'):
                loaded_components.append('feat_sharp')
        if 'hash_grid_mlp' in joint_map_state and (allowed is None or 'hash_grid_mlp' in allowed):
            if _load_state_dict_compatible(dcff_renderer.hash_grid.mlp, joint_map_state['hash_grid_mlp'], 'joint hash_grid_mlp'):
                loaded_components.append('hash_grid_mlp')
        if loaded_components:
            print(f"  Overrode DCFF modules from joint checkpoint: {', '.join(loaded_components)}")
        else:
            print('  WARNING: joint checkpoint has no map_renderer_state_dict overrides; keeping base DCFF weights')

    dcff_iter = ckpt.get('iteration', '?')
    print(f'  DCFF loaded (iter {dcff_iter})')

    # 6. Freeze everything
    gaussians._latent.requires_grad_(False)
    for p in hash_grid.parameters():
        p.requires_grad_(False)
    for p in dcff_renderer.parameters():
        p.requires_grad_(False)
    for p in feat_sharp.parameters():
        p.requires_grad_(False)
    hash_grid.eval()
    dcff_renderer.eval()
    feat_sharp.eval()

    return gaussians, dcff_renderer, feat_sharp


# ═════════════════════════════════════════════════════════════════════════════
#  Rendering (mirrors train_concat_loc.py _render_dcff_at_pose)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def render_at_pose(gaussians, dcff_renderer, feat_sharp, pose_w2c, K,
                   render_h, render_w):
    """Render DCFF fine features + depth at a single w2c pose.

    Returns:
        fine_features: (1, C, H, W)
        depth: (1, 1, H, W)
    """
    viewmat = pose_w2c.float()
    result = dcff_renderer(
        gaussians,
        viewmat=viewmat,
        K=K,
        width=render_w,
        height=render_h,
        render_coarse=False,
    )
    fine_feat = result['fine_features'].float()
    depth = result['depth']
    alpha = result.get('alpha')
    fine_feat = feat_sharp(
        fine_feat,
        depth=depth.float() if depth is not None else None,
        alpha=alpha.float() if alpha is not None else None,
    )
    return fine_feat, depth


@torch.no_grad()
def render_batch(gaussians, dcff_renderer, feat_sharp, poses_w2c, K,
                 render_h, render_w):
    """Render DCFF features + depth for a batch of w2c poses.

    Returns:
        ref_fine: (B, C, H, W)
        depth: (B, H, W)
    """
    B = poses_w2c.shape[0]
    fine_list, depth_list = [], []
    for i in range(B):
        fine_i, depth_i = render_at_pose(
            gaussians, dcff_renderer, feat_sharp,
            poses_w2c[i], K, render_h, render_w,
        )
        fine_list.append(fine_i.squeeze(0))             # (C, H, W)
        depth_list.append(depth_i.squeeze(0).squeeze(0))  # (H, W)

    return torch.stack(fine_list, dim=0), torch.stack(depth_list, dim=0)


# ═════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(config, checkpoint_path, device):
    """Load ConcatPoseNet from checkpoint.

    Returns:
        model: ConcatPoseNet in eval mode
        epoch: checkpoint epoch
    """
    return load_concat_pose_model(config, checkpoint_path, device, printer=print)


# Shared system-layer runtime is authoritative. The legacy helpers remain in
# this file for continuity, but actual execution uses the centralized versions.
build_dcff = shared_build_dcff
intrinsics_to_K = shared_intrinsics_to_K
render_batch = shared_render_batch
load_model = load_concat_pose_model


# ═════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, gaussians, dcff_renderer, feat_sharp, val_loader, device,
             outer_iters, gru_iters, render_h, render_w, use_coarse=True,
             solver='wls', irls_iters=0, robust_kernel='huber',
             direct_refine_iters=0):
    """Run one full evaluation pass.

    Args:
        solver: 'wls' (default), 'pnp' (PnP+RANSAC), 'hybrid' (WLS+PnP),
                'wls_full' (WLS for both rot+trans), 'direct' (feature-metric only)
        irls_iters: IRLS iterations for WLS solver (0 = no IRLS)
        robust_kernel: 'huber', 'gm', or 'gnc_gm'
        direct_refine_iters: Extra direct LK iterations AFTER flow-based outer iters

    Returns dict with rot/trans statistics.
    """
    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = gru_iters

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)
    all_rot_errs, all_trans_errs = [], []

    for batch in tqdm(val_loader, desc=f'Eval oi={outer_iters} gru={gru_iters} {solver}',
                      leave=False):
        query_fine = batch['query_fine'].to(device)
        query_coarse = batch.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.to(device)
        else:
            query_coarse = None
        pose_gt = batch['pose_gt'].to(device)
        pose_cur = batch['pose_init'].to(device)

        # Flow-based outer iteration refinement loop
        use_flow = not solver.startswith('direct')
        n_flow_iters = outer_iters if use_flow else 0

        for outer_i in range(n_flow_iters):
            ref_fine, depth = render_batch(
                gaussians, dcff_renderer, feat_sharp,
                pose_cur, K, render_h, render_w,
            )

            fwd_irls = irls_iters if solver == 'wls' else 0
            fwd_kernel = robust_kernel if solver == 'wls' else None

            with autocast(enabled=True):
                pred = model(
                    query_fine, ref_fine, depth,
                    intrinsics=render_intr,
                    query_coarse=query_coarse,
                    irls_iters=fwd_irls,
                    robust_kernel=fwd_kernel,
                )

            use_pnp = (solver == 'pnp') or \
                       (solver == 'hybrid' and outer_i == n_flow_iters - 1)

            if use_pnp and 'flow' in pred:
                with torch.cuda.amp.autocast(enabled=False):
                    flow_f = pred['flow'].float()
                    conf_f = pred['confidence'].float()
                    depth_for_pnp = depth.float()
                    if depth_for_pnp.dim() == 3:
                        depth_for_pnp = depth_for_pnp.unsqueeze(1)
                    pnp_xi = pnp_ransac_solve(
                        flow_f, depth_for_pnp, render_intr,
                        confidence=conf_f,
                        reprojection_threshold=4.0,
                        n_iters=2000,
                    )
                    T_delta = se3_exp(pnp_xi.float())
            elif solver == 'wls_full' and 'delta_xi_full' in pred:
                with torch.cuda.amp.autocast(enabled=False):
                    T_delta = se3_exp(pred['delta_xi_full'].float())
            elif 'delta_xi' in pred:
                with torch.cuda.amp.autocast(enabled=False):
                    T_delta = se3_exp(pred['delta_xi'].float())
            else:
                continue

            with torch.cuda.amp.autocast(enabled=False):
                pose_cur = torch.bmm(T_delta, pose_cur.float())

        # Direct feature-metric refinement iterations (LK-style)
        n_direct = direct_refine_iters if solver != 'direct' else outer_iters
        for di in range(n_direct):
            ref_fine, depth = render_batch(
                gaussians, dcff_renderer, feat_sharp,
                pose_cur, K, render_h, render_w,
            )
            with torch.cuda.amp.autocast(enabled=False):
                depth_s = depth.float() if depth.dim() == 3 else depth.squeeze(1).float()
                # Use higher damping for stability; negate if solver starts with '-'
                damp = 1.0 if 'highdamp' in solver else 1e-2
                direct_xi, residual = feature_metric_solve(
                    query_fine.float(), ref_fine.float(), depth_s,
                    render_intr, damping=damp,
                )
                if 'neg' in solver:
                    direct_xi = -direct_xi
                T_delta = se3_exp(direct_xi.float())
                pose_cur = torch.bmm(T_delta, pose_cur.float())

        # Compute pose errors
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = pose_cur.float()[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp(
                (trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = torch.acos(cos_angle) * 180.0 / math.pi
            c_pred = camera_centers_from_w2c(pose_cur.float())
            c_gt = camera_centers_from_w2c(pose_gt.float())
            trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000  # mm

        all_rot_errs.extend(rot_err.cpu().tolist())
        all_trans_errs.extend(trans_err.cpu().tolist())

    model.gru_iters = original_gru

    rot = np.array(all_rot_errs)
    trans = np.array(all_trans_errs)
    return {
        'rot_mean': float(np.nanmean(rot)),
        'rot_median': float(np.nanmedian(rot)),
        'trans_mean': float(np.nanmean(trans)),
        'trans_median': float(np.nanmedian(trans)),
        'pct_1deg': float(np.mean(rot < 1.0) * 100),
        'pct_5deg': float(np.mean(rot < 5.0) * 100),
        'joint_01deg_53mm': float(np.mean((rot < 0.1) & (trans < 5.3)) * 100),
        'joint_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
        'joint_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Multi-Start Evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_multistart(model, gaussians, dcff_renderer, feat_sharp,
                        dataset, device,
                        outer_iters, gru_iters, render_h, render_w,
                        num_starts=5, use_coarse=True,
                        solver='wls', irls_iters=0, robust_kernel='huber',
                        selection='confidence'):
    """Multi-start evaluation: run refinement from N perturbations, pick best.

    Selection strategies:
        'confidence': Mean flow confidence from last GRU iteration (higher=better)
        'similarity': Cosine similarity between query and rendered features
        'delta_mag': Negative magnitude of predicted pose update (smaller=better)
        'consensus': Pick pose closest to the median of all refined poses

    Returns dict with rot/trans statistics.
    """
    from data.radio_loc_dataset import add_pose_noise

    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = gru_iters

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    all_rot_errs, all_trans_errs = [], []
    all_best_scores = []

    for idx in tqdm(range(len(dataset)),
                    desc=f'MS({num_starts},{selection}) oi={outer_iters}'):
        sample_base = dataset[idx]
        query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
        query_coarse = sample_base.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample_base['pose_gt'].unsqueeze(0).to(device)
        pose_gt_np = sample_base['pose_gt'].numpy()

        candidate_poses = []
        candidate_scores = []

        for start_i in range(num_starts):
            pose_init_np = add_pose_noise(
                pose_gt_np,
                dataset.noise_rot_deg,
                dataset.noise_trans_m,
            )
            pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)

            use_flow = not solver.startswith('direct')
            n_flow_iters = outer_iters if use_flow else 0
            last_pred = None

            for outer_i in range(n_flow_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w,
                )
                fwd_irls = irls_iters if solver == 'wls' else 0
                fwd_kernel = robust_kernel if solver == 'wls' else None

                with autocast(enabled=True):
                    pred = model(
                        query_fine, ref_fine, depth,
                        intrinsics=render_intr,
                        query_coarse=query_coarse,
                        irls_iters=fwd_irls,
                        robust_kernel=fwd_kernel,
                    )
                last_pred = pred

                use_pnp = (solver == 'pnp') or \
                           (solver == 'hybrid' and outer_i == n_flow_iters - 1)

                if use_pnp and 'flow' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        flow_f = pred['flow'].float()
                        conf_f = pred['confidence'].float()
                        depth_for_pnp = depth.float()
                        if depth_for_pnp.dim() == 3:
                            depth_for_pnp = depth_for_pnp.unsqueeze(1)
                        pnp_xi = pnp_ransac_solve(
                            flow_f, depth_for_pnp, render_intr,
                            confidence=conf_f,
                            reprojection_threshold=4.0,
                            n_iters=2000,
                        )
                        T_delta = se3_exp(pnp_xi.float())
                elif 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                else:
                    continue

                with torch.cuda.amp.autocast(enabled=False):
                    pose_cur = torch.bmm(T_delta, pose_cur.float())

            # Compute score based on selection strategy
            if selection == 'confidence' and last_pred is not None:
                conf = last_pred.get('confidence')
                if conf is not None:
                    score = conf.float().mean().item()
                else:
                    score = 0.0
            elif selection == 'delta_mag' and last_pred is not None:
                xi = last_pred.get('delta_xi')
                if xi is not None:
                    score = -xi.float().norm().item()
                else:
                    score = 0.0
            elif selection == 'similarity':
                ref_final, _ = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w,
                )
                q_norm = F.normalize(query_fine.float(), dim=1)
                r_norm = F.normalize(ref_final.float(), dim=1)
                score = (q_norm * r_norm).sum(dim=1).mean().item()
            else:
                score = 0.0

            candidate_poses.append(pose_cur.clone())
            candidate_scores.append(score)

        # Select best pose
        if selection == 'consensus':
            # Pick pose closest to median translation
            all_t = torch.cat([p[:, :3, 3] for p in candidate_poses], dim=0)
            median_t = all_t.median(dim=0).values
            dists = [(p[:, :3, 3] - median_t.unsqueeze(0)).norm().item()
                     for p in candidate_poses]
            best_idx = int(np.argmin(dists))
        else:
            best_idx = int(np.argmax(candidate_scores))

        best_pose = candidate_poses[best_idx]
        best_score = candidate_scores[best_idx]

        with torch.cuda.amp.autocast(enabled=False):
            R_pred = best_pose.float()[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp(
                (trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = torch.acos(cos_angle) * 180.0 / math.pi
            c_pred = camera_centers_from_w2c(best_pose.float())
            c_gt = camera_centers_from_w2c(pose_gt.float())
            trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000

        all_rot_errs.append(rot_err.item())
        all_trans_errs.append(trans_err.item())
        all_best_scores.append(best_score)

    model.gru_iters = original_gru

    rot = np.array(all_rot_errs)
    trans = np.array(all_trans_errs)
    scores = np.array(all_best_scores)
    return {
        'rot_mean': float(np.nanmean(rot)),
        'rot_median': float(np.nanmedian(rot)),
        'trans_mean': float(np.nanmean(trans)),
        'trans_median': float(np.nanmedian(trans)),
        'pct_1deg': float(np.mean(rot < 1.0) * 100),
        'pct_5deg': float(np.mean(rot < 5.0) * 100),
        'joint_01deg_53mm': float(np.mean((rot < 0.1) & (trans < 5.3)) * 100),
        'joint_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
        'joint_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
        'score_mean': float(np.mean(scores)),
        'score_median': float(np.median(scores)),
    }


def evaluate_cascade(model, gaussians, dcff_renderer, feat_sharp,
                     dataset, device,
                     outer_iters, gru_iters, render_h, render_w,
                     rounds=3, starts_per_round=10, refine_noise_deg=1.0,
                     refine_noise_m=0.03, use_coarse=True,
                     solver='wls', irls_iters=0, robust_kernel='huber',
                     selection='consensus'):
    """Cascaded refinement: multiple rounds of multi-start + selection.

    Round 1: N starts from original noise → select best → initial refined poses.
    Round 2+: M starts from small perturbations around previous best pose
              → select best → further refined poses.

    Selection strategies:
      'consensus': pick pose closest to median translation (no extra render)
      'similarity': pick pose with highest render-query feature similarity
      'delta_mag': pick pose with smallest final correction magnitude
    """
    from data.radio_loc_dataset import add_pose_noise

    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = gru_iters

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    all_rot_errs, all_trans_errs = [], []
    round_stats = {r: {'rot': [], 'trans': []} for r in range(rounds)}

    for idx in tqdm(range(len(dataset)), desc=f'Cascade({rounds}r×{starts_per_round}s)'):
        sample_base = dataset[idx]
        query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
        query_coarse = sample_base.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample_base['pose_gt'].unsqueeze(0).to(device)
        pose_gt_np = sample_base['pose_gt'].numpy()

        best_pose = None

        for round_i in range(rounds):
            candidate_poses = []
            candidate_scores = []
            n_starts = starts_per_round

            for start_i in range(n_starts):
                if round_i == 0:
                    pose_init_np = add_pose_noise(
                        pose_gt_np,
                        dataset.noise_rot_deg,
                        dataset.noise_trans_m,
                    )
                    pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)
                else:
                    best_np = best_pose.detach().cpu().squeeze(0).numpy()
                    pose_init_np = add_pose_noise(
                        best_np,
                        refine_noise_deg,
                        refine_noise_m,
                    )
                    pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)

                last_delta_xi = None
                for outer_i in range(outer_iters):
                    ref_fine, depth = render_batch(
                        gaussians, dcff_renderer, feat_sharp,
                        pose_cur, K, render_h, render_w,
                    )
                    fwd_irls = irls_iters if solver == 'wls' else 0
                    fwd_kernel = robust_kernel if solver == 'wls' else None

                    with autocast(enabled=True):
                        pred = model(
                            query_fine, ref_fine, depth,
                            intrinsics=render_intr,
                            query_coarse=query_coarse,
                            irls_iters=fwd_irls,
                            robust_kernel=fwd_kernel,
                        )

                    if 'delta_xi' in pred:
                        last_delta_xi = pred['delta_xi'].detach()

                    use_pnp = (solver == 'pnp') or \
                               (solver == 'hybrid' and outer_i == outer_iters - 1)

                    if use_pnp and 'flow' in pred:
                        with torch.cuda.amp.autocast(enabled=False):
                            flow_f = pred['flow'].detach().float()
                            conf_f = pred['confidence'].detach().float()
                            depth_for_pnp = depth.detach().float()
                            if depth_for_pnp.dim() == 3:
                                depth_for_pnp = depth_for_pnp.unsqueeze(1)
                            pnp_xi = pnp_ransac_solve(
                                flow_f, depth_for_pnp, render_intr,
                                confidence=conf_f,
                                reprojection_threshold=4.0,
                                n_iters=2000,
                            )
                            T_delta = se3_exp(pnp_xi.float())
                    elif 'delta_xi' in pred:
                        with torch.cuda.amp.autocast(enabled=False):
                            T_delta = se3_exp(pred['delta_xi'].float())
                    else:
                        continue

                    with torch.cuda.amp.autocast(enabled=False):
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

                # Score each candidate
                score = 0.0
                if selection == 'similarity':
                    ref_final, _ = render_batch(
                        gaussians, dcff_renderer, feat_sharp,
                        pose_cur, K, render_h, render_w,
                    )
                    q_norm = F.normalize(query_fine.float(), dim=1)
                    r_norm = F.normalize(ref_final.float(), dim=1)
                    score = (q_norm * r_norm).sum(dim=1).mean().item()
                    del ref_final
                elif selection == 'delta_mag' and last_delta_xi is not None:
                    score = -last_delta_xi.float().norm().item()

                candidate_poses.append(pose_cur.detach().cpu().clone())
                candidate_scores.append(score)
                del ref_fine, depth, pred, pose_cur

            # Select best pose
            if selection == 'consensus':
                all_t = torch.cat([p[:, :3, 3] for p in candidate_poses], dim=0)
                median_t = all_t.median(dim=0).values
                dists = [(p[:, :3, 3] - median_t.unsqueeze(0)).norm().item()
                         for p in candidate_poses]
                best_idx = int(np.argmin(dists))
            else:
                best_idx = int(np.argmax(candidate_scores))
            best_pose = candidate_poses[best_idx].to(device)

            # Track per-round error
            with torch.cuda.amp.autocast(enabled=False):
                R_pred = best_pose.float()[:, :3, :3]
                R_gt = pose_gt.float()[:, :3, :3]
                R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
                trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                cos_angle = torch.clamp(
                    (trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                rot_err = torch.acos(cos_angle) * 180.0 / math.pi
                c_pred = camera_centers_from_w2c(best_pose.float())
                c_gt = camera_centers_from_w2c(pose_gt.float())
                trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000

            round_stats[round_i]['rot'].append(rot_err.item())
            round_stats[round_i]['trans'].append(trans_err.item())

        all_rot_errs.append(rot_err.item())
        all_trans_errs.append(trans_err.item())

        # Free GPU memory between samples
        del candidate_poses, best_pose, query_fine, query_coarse, pose_gt
        torch.cuda.empty_cache()

    model.gru_iters = original_gru

    rot = np.array(all_rot_errs)
    trans = np.array(all_trans_errs)

    result = {
        'rot_mean': float(np.nanmean(rot)),
        'rot_median': float(np.nanmedian(rot)),
        'trans_mean': float(np.nanmean(trans)),
        'trans_median': float(np.nanmedian(trans)),
        'pct_1deg': float(np.mean(rot < 1.0) * 100),
        'pct_5deg': float(np.mean(rot < 5.0) * 100),
        'joint_01deg_53mm': float(np.mean((rot < 0.1) & (trans < 5.3)) * 100),
        'joint_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
        'joint_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
    }

    # Print per-round progression
    print(f"\n  ── Cascade per-round progression ──")
    for r in range(rounds):
        r_rot = np.array(round_stats[r]['rot'])
        r_trans = np.array(round_stats[r]['trans'])
        noise_str = (f"noise={dataset.noise_rot_deg}°/{dataset.noise_trans_m}m"
                     if r == 0 else f"noise={refine_noise_deg}°/{refine_noise_m}m")
        print(f"  Round {r}: rot_med={np.median(r_rot):.3f}°  "
              f"trans_med={np.median(r_trans):.1f}mm  "
              f"<1°={np.mean(r_rot < 1.0)*100:.1f}%  ({noise_str})")

    return result


@torch.no_grad()
def evaluate_twostage(model_s1, model_s2, gaussians, dcff_renderer, feat_sharp,
                      dataset, device,
                      s1_outer_iters, s1_gru_iters,
                      s2_outer_iters, s2_gru_iters,
                      render_h, render_w,
                      s1_starts=10, use_coarse=True,
                      solver='wls', irls_iters=0, robust_kernel='huber',
                      stage1_map_state=None, stage2_map_state=None):
    """Two-stage evaluation pipeline.

    Stage 1: model_s1 + multi-start consensus → coarse refined poses.
    Stage 2: model_s2 iterative refinement from Stage 1 output → fine poses.
    """
    from data.radio_loc_dataset import add_pose_noise

    model_s1.eval()
    model_s2.eval()

    original_gru_s1 = model_s1.gru_iters
    original_gru_s2 = model_s2.gru_iters
    model_s1.gru_iters = s1_gru_iters
    model_s2.gru_iters = s2_gru_iters

    render_intr = model_s1._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    all_rot_errs, all_trans_errs = [], []
    s1_rot_errs, s1_trans_errs = [], []

    for idx in tqdm(range(len(dataset)), desc=f'TwoStage(S1:{s1_starts}ms→S2:{s2_outer_iters}oi)'):
        sample_base = dataset[idx]
        query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
        query_coarse = sample_base.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample_base['pose_gt'].unsqueeze(0).to(device)
        pose_gt_np = sample_base['pose_gt'].numpy()

        # ── Stage 1: Multi-start consensus with model_s1 ──
        if stage1_map_state is not None:
            apply_localization_map_state(dcff_renderer, feat_sharp, stage1_map_state, printer=None)
        candidate_poses = []
        for start_i in range(s1_starts):
            pose_init_np = add_pose_noise(
                pose_gt_np, dataset.noise_rot_deg, dataset.noise_trans_m)
            pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)

            for oi in range(s1_outer_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w)
                with autocast(enabled=True):
                    pred = model_s1(
                        query_fine, ref_fine, depth,
                        intrinsics=render_intr,
                        query_coarse=query_coarse)
                if 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

            candidate_poses.append(pose_cur.detach().cpu().clone())
            del ref_fine, depth, pred, pose_cur

        # Consensus selection
        all_t = torch.cat([p[:, :3, 3] for p in candidate_poses], dim=0)
        median_t = all_t.median(dim=0).values
        dists = [(p[:, :3, 3] - median_t.unsqueeze(0)).norm().item()
                 for p in candidate_poses]
        best_idx = int(np.argmin(dists))
        s1_pose = candidate_poses[best_idx].to(device)

        # Track Stage 1 error
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = s1_pose.float()[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace_val = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            s1_rot = torch.acos(cos_angle) * 180.0 / math.pi
            s1_trans = torch.norm(s1_pose.float()[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000
        s1_rot_errs.append(s1_rot.item())
        s1_trans_errs.append(s1_trans.item())

        # ── Stage 2: Iterative refinement with model_s2 ──
        if stage2_map_state is not None:
            apply_localization_map_state(dcff_renderer, feat_sharp, stage2_map_state, printer=None)
        pose_cur = s1_pose.clone()
        for oi in range(s2_outer_iters):
            ref_fine, depth = render_batch(
                gaussians, dcff_renderer, feat_sharp,
                pose_cur, K, render_h, render_w)
            fwd_irls = irls_iters if solver == 'wls' else 0
            fwd_kernel = robust_kernel if solver == 'wls' else None
            with autocast(enabled=True):
                pred = model_s2(
                    query_fine, ref_fine, depth,
                    intrinsics=render_intr,
                    query_coarse=query_coarse,
                    irls_iters=fwd_irls,
                    robust_kernel=fwd_kernel)
            if 'delta_xi' in pred:
                with torch.cuda.amp.autocast(enabled=False):
                    T_delta = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T_delta, pose_cur.float())

        # Track Stage 2 (final) error
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = pose_cur.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace_val = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = torch.acos(cos_angle) * 180.0 / math.pi
            c_pred = camera_centers_from_w2c(pose_cur.float())
            c_gt = camera_centers_from_w2c(pose_gt.float())
            trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000

        all_rot_errs.append(rot_err.item())
        all_trans_errs.append(trans_err.item())

        del candidate_poses, s1_pose, pose_cur, query_fine, query_coarse, pose_gt
        torch.cuda.empty_cache()

    model_s1.gru_iters = original_gru_s1
    model_s2.gru_iters = original_gru_s2

    rot = np.array(all_rot_errs)
    trans = np.array(all_trans_errs)
    s1_rot = np.array(s1_rot_errs)
    s1_trans = np.array(s1_trans_errs)

    print(f"\n  ── Two-Stage Results ──")
    print(f"  Stage 1 ({s1_starts}ms consensus): "
          f"rot_med={np.median(s1_rot):.3f}°  trans_med={np.median(s1_trans):.1f}mm  "
          f"<1°={np.mean(s1_rot < 1.0)*100:.1f}%")
    print(f"  Stage 2 ({s2_outer_iters}oi refine):  "
          f"rot_med={np.median(rot):.3f}°  trans_med={np.median(trans):.1f}mm  "
          f"<1°={np.mean(rot < 1.0)*100:.1f}%")

    return {
        'rot_mean': float(np.nanmean(rot)),
        'rot_median': float(np.nanmedian(rot)),
        'trans_mean': float(np.nanmean(trans)),
        'trans_median': float(np.nanmedian(trans)),
        'pct_1deg': float(np.mean(rot < 1.0) * 100),
        'pct_5deg': float(np.mean(rot < 5.0) * 100),
        'joint_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
        'joint_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
        's1_rot_median': float(np.median(s1_rot)),
        's1_trans_median': float(np.median(s1_trans)),
    }


def evaluate_twostage_sweep(model_s1, model_s2, gaussians, dcff_renderer, feat_sharp,
                            dataset, device,
                            s1_outer_iters, s1_gru_iters, s2_gru_iters,
                            s2_oi_list, render_h, render_w,
                            s1_starts=10, use_coarse=True,
                            solver='wls', irls_iters=0, robust_kernel='huber',
                            stage1_map_state=None, stage2_map_state=None):
    """Two-stage evaluation with Stage 1 pose caching for efficient Stage 2 sweeping.

    Runs Stage 1 once (MS consensus), caches all poses, then evaluates Stage 2
    for each value in s2_oi_list without re-running Stage 1.
    """
    from data.radio_loc_dataset import add_pose_noise

    model_s1.eval()
    model_s2.eval()

    original_gru_s1 = model_s1.gru_iters
    original_gru_s2 = model_s2.gru_iters
    model_s1.gru_iters = s1_gru_iters
    model_s2.gru_iters = s2_gru_iters

    render_intr = model_s1._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    # ── Stage 1: Run once, cache consensus poses ──
    print(f'\n  Stage 1: {s1_starts} starts × {s1_outer_iters} oi → consensus ...')
    cached_s1_poses = []  # List of (3,4) or (4,4) tensors on CPU
    s1_rot_errs, s1_trans_errs = [], []
    gt_poses = []

    for idx in tqdm(range(len(dataset)), desc=f'S1({s1_starts}ms)'):
        sample_base = dataset[idx]
        query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
        query_coarse = sample_base.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample_base['pose_gt'].unsqueeze(0).to(device)
        pose_gt_np = sample_base['pose_gt'].numpy()
        gt_poses.append(pose_gt.cpu().clone())

        if stage1_map_state is not None:
            apply_localization_map_state(dcff_renderer, feat_sharp, stage1_map_state, printer=None)
        candidate_poses = []
        for start_i in range(s1_starts):
            pose_init_np = add_pose_noise(
                pose_gt_np, dataset.noise_rot_deg, dataset.noise_trans_m)
            pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)

            for oi in range(s1_outer_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w)
                with autocast(enabled=True):
                    pred = model_s1(
                        query_fine, ref_fine, depth,
                        intrinsics=render_intr,
                        query_coarse=query_coarse)
                if 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

            candidate_poses.append(pose_cur.detach().cpu().clone())
            del ref_fine, depth, pred, pose_cur

        # Consensus selection
        all_t = torch.cat([p[:, :3, 3] for p in candidate_poses], dim=0)
        median_t = all_t.median(dim=0).values
        dists = [(p[:, :3, 3] - median_t.unsqueeze(0)).norm().item()
                 for p in candidate_poses]
        best_idx = int(np.argmin(dists))
        s1_pose = candidate_poses[best_idx]
        cached_s1_poses.append(s1_pose)

        # Track Stage 1 error
        with torch.cuda.amp.autocast(enabled=False):
            s1_on_dev = s1_pose.to(device).float()
            R_pred = s1_on_dev[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace_val = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            s1_rot = torch.acos(cos_angle) * 180.0 / math.pi
            s1_trans = torch.norm(s1_on_dev[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000
        s1_rot_errs.append(s1_rot.item())
        s1_trans_errs.append(s1_trans.item())

        del candidate_poses, query_fine, query_coarse, pose_gt
        torch.cuda.empty_cache()

    s1_rot = np.array(s1_rot_errs)
    s1_trans = np.array(s1_trans_errs)
    print(f'  Stage 1 done: rot_med={np.median(s1_rot):.3f}° trans_med={np.median(s1_trans):.1f}mm '
          f'<1°={np.mean(s1_rot < 1.0)*100:.1f}%')

    # ── Stage 2: Sweep over s2_outer_iters values ──
    all_results = {}
    for s2_oi in s2_oi_list:
        print(f'\n  Stage 2: {s2_oi} outer iters from cached S1 poses ...')
        s2_rot_errs, s2_trans_errs = [], []

        for idx in tqdm(range(len(dataset)), desc=f'S2(oi={s2_oi})'):
            if stage2_map_state is not None:
                apply_localization_map_state(dcff_renderer, feat_sharp, stage2_map_state, printer=None)
            sample_base = dataset[idx]
            query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
            query_coarse = sample_base.get('query_coarse')
            if query_coarse is not None and use_coarse:
                query_coarse = query_coarse.unsqueeze(0).to(device)
            else:
                query_coarse = None
            pose_gt = gt_poses[idx].to(device)

            pose_cur = cached_s1_poses[idx].to(device).clone()
            for oi in range(s2_oi):
                ref_fine, depth_r = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w)
                fwd_irls = irls_iters if solver == 'wls' else 0
                fwd_kernel = robust_kernel if solver == 'wls' else None
                with autocast(enabled=True):
                    pred = model_s2(
                        query_fine, ref_fine, depth_r,
                        intrinsics=render_intr,
                        query_coarse=query_coarse,
                        irls_iters=fwd_irls,
                        robust_kernel=fwd_kernel)
                if 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())
                del ref_fine, depth_r, pred

            with torch.cuda.amp.autocast(enabled=False):
                R_pred = pose_cur.float()[:, :3, :3]
                R_gt = pose_gt.float()[:, :3, :3]
                R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
                trace_val = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                rot_err = torch.acos(cos_angle) * 180.0 / math.pi
                c_pred = camera_centers_from_w2c(pose_cur.float())
                c_gt = camera_centers_from_w2c(pose_gt.float())
                trans_err = torch.norm(c_pred - c_gt, dim=1) * 1000

            s2_rot_errs.append(rot_err.item())
            s2_trans_errs.append(trans_err.item())
            del pose_cur, query_fine, query_coarse, pose_gt
            torch.cuda.empty_cache()

        rot = np.array(s2_rot_errs)
        trans = np.array(s2_trans_errs)
        print(f'  S2(oi={s2_oi}): rot_med={np.median(rot):.3f}° trans_med={np.median(trans):.1f}mm '
              f'<1°={np.mean(rot < 1.0)*100:.1f}% joint@1°/50mm={np.mean((rot < 1) & (trans < 50))*100:.1f}%')

        all_results[s2_oi] = {
            'rot_mean': float(np.nanmean(rot)),
            'rot_median': float(np.nanmedian(rot)),
            'trans_mean': float(np.nanmean(trans)),
            'trans_median': float(np.nanmedian(trans)),
            'pct_1deg': float(np.mean(rot < 1.0) * 100),
            'pct_5deg': float(np.mean(rot < 5.0) * 100),
            'joint_1deg_50mm': float(np.mean((rot < 1.0) & (trans < 50.0)) * 100),
            'joint_5deg_100mm': float(np.mean((rot < 5.0) & (trans < 100.0)) * 100),
            's1_rot_median': float(np.median(s1_rot)),
            's1_trans_median': float(np.median(s1_trans)),
        }

    model_s1.gru_iters = original_gru_s1
    model_s2.gru_iters = original_gru_s2
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  NN + Featuremetric Refinement
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_nn_then_fm(model, gaussians, dcff_renderer, feat_sharp,
                        dataset, device,
                        outer_iters, gru_iters, render_h, render_w,
                        num_starts=10, use_coarse=True,
                        solver='wls', irls_iters=0, robust_kernel='huber',
                        selection='consensus',
                        fm_iters=30, fm_damping=1e-3, fm_step_limit=0.01):
    """NN multi-start → consensus → featuremetric Gauss-Newton refinement.

    Pipeline:
        1. Run NN multi-start (same as evaluate_multistart) → consensus pose
        2. Iteratively refine with featuremetric GN steps:
           - Render features + depth at current pose
           - Compute Gauss-Newton update from feature residual
           - Apply pose update via se3_exp
           - Track convergence via residual norm

    Args:
        fm_iters: Max featuremetric GN iterations (default: 30)
        fm_damping: LM damping factor (default: 1e-3)
        fm_step_limit: Max step norm for convergence check (default: 0.01)

    Returns dict with rot/trans statistics for both NN-only and NN+FM.
    """
    from data.radio_loc_dataset import add_pose_noise
    from modules.featuremetric import featuremetric_gauss_newton_step

    model.eval()
    original_gru = model.gru_iters
    model.gru_iters = gru_iters

    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)

    nn_rot_errs, nn_trans_errs = [], []
    fm_rot_errs, fm_trans_errs = [], []
    fm_iterations_used = []

    for idx in tqdm(range(len(dataset)),
                    desc=f'NN+FM(MS={num_starts},fm={fm_iters})'):
        sample_base = dataset[idx]
        query_fine = sample_base['query_fine'].unsqueeze(0).to(device)
        query_coarse = sample_base.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        else:
            query_coarse = None
        pose_gt = sample_base['pose_gt'].unsqueeze(0).to(device)

        # ── Stage 1: NN multi-start → consensus ──────────────────────────
        candidate_poses = []
        for start_i in range(num_starts):
            pose_init_np = add_pose_noise(
                sample_base['pose_gt'].numpy(),
                dataset.noise_rot_deg,
                dataset.noise_trans_m,
            )
            pose_cur = torch.from_numpy(pose_init_np).float().unsqueeze(0).to(device)

            for outer_i in range(outer_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, render_h, render_w,
                )
                fwd_irls = irls_iters if solver == 'wls' else 0
                fwd_kernel = robust_kernel if solver == 'wls' else None

                with autocast(enabled=True):
                    pred = model(
                        query_fine, ref_fine, depth,
                        intrinsics=render_intr,
                        query_coarse=query_coarse,
                        irls_iters=fwd_irls,
                        robust_kernel=fwd_kernel,
                    )

                if 'delta_xi' in pred:
                    with torch.cuda.amp.autocast(enabled=False):
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

            candidate_poses.append(pose_cur.clone())

        # Consensus: pick pose closest to median translation
        all_t = torch.cat([p[:, :3, 3] for p in candidate_poses], dim=0)
        median_t = all_t.median(dim=0).values
        dists = [(p[:, :3, 3] - median_t.unsqueeze(0)).norm().item()
                 for p in candidate_poses]
        best_idx = int(np.argmin(dists))
        nn_pose = candidate_poses[best_idx]

        # NN-only error
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = nn_pose.float()[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            nn_rot_err = (torch.acos(cos_angle) * 180.0 / math.pi).item()
            nn_trans_err = (torch.norm(nn_pose.float()[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000).item()

        nn_rot_errs.append(nn_rot_err)
        nn_trans_errs.append(nn_trans_err)

        # ── Stage 2: Featuremetric GN refinement ─────────────────────────
        pose_fm = nn_pose.clone().float()
        prev_residual_norm = float('inf')
        actual_iters = 0

        for fm_i in range(fm_iters):
            # Render features + depth at current pose
            ref_fine, depth = render_batch(
                gaussians, dcff_renderer, feat_sharp,
                pose_fm, K, render_h, render_w,
            )

            # L2-normalize features for stable gradients
            q_norm = F.normalize(query_fine.float(), dim=1)
            r_norm = F.normalize(ref_fine.float(), dim=1)

            # Compute residual norm for convergence check
            residual_norm = (q_norm - r_norm).pow(2).sum().item()

            # GN step
            delta_xi = featuremetric_gauss_newton_step(
                q_norm, r_norm, depth.float(),
                intrinsics=render_intr,
                damping=fm_damping,
            )

            step_norm = delta_xi.norm().item()
            actual_iters += 1

            # Apply pose update
            T_delta = se3_exp(delta_xi)
            pose_fm = torch.bmm(T_delta, pose_fm)

            # Convergence check
            if step_norm < fm_step_limit:
                break
            if residual_norm > prev_residual_norm * 1.05:
                # Diverging — revert and stop
                T_inv = se3_exp(-delta_xi)
                pose_fm = torch.bmm(T_inv, pose_fm)
                actual_iters -= 1
                break
            prev_residual_norm = residual_norm

        fm_iterations_used.append(actual_iters)

        # FM error
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = pose_fm.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            fm_rot_err = (torch.acos(cos_angle) * 180.0 / math.pi).item()
            fm_trans_err = (torch.norm(pose_fm.float()[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000).item()

        fm_rot_errs.append(fm_rot_err)
        fm_trans_errs.append(fm_trans_err)

    model.gru_iters = original_gru

    nn_rot = np.array(nn_rot_errs)
    nn_trans = np.array(nn_trans_errs)
    fm_rot = np.array(fm_rot_errs)
    fm_trans = np.array(fm_trans_errs)
    iters_arr = np.array(fm_iterations_used)

    return {
        # NN-only metrics
        'nn_rot_mean': float(np.nanmean(nn_rot)),
        'nn_rot_median': float(np.nanmedian(nn_rot)),
        'nn_trans_mean': float(np.nanmean(nn_trans)),
        'nn_trans_median': float(np.nanmedian(nn_trans)),
        'nn_pct_1deg': float(np.mean(nn_rot < 1.0) * 100),
        'nn_joint_1deg_50mm': float(np.mean((nn_rot < 1.0) & (nn_trans < 50.0)) * 100),
        # NN+FM metrics
        'fm_rot_mean': float(np.nanmean(fm_rot)),
        'fm_rot_median': float(np.nanmedian(fm_rot)),
        'fm_trans_mean': float(np.nanmean(fm_trans)),
        'fm_trans_median': float(np.nanmedian(fm_trans)),
        'fm_pct_1deg': float(np.mean(fm_rot < 1.0) * 100),
        'fm_pct_5deg': float(np.mean(fm_rot < 5.0) * 100),
        'fm_joint_01deg_53mm': float(np.mean((fm_rot < 0.1) & (fm_trans < 5.3)) * 100),
        'fm_joint_1deg_50mm': float(np.mean((fm_rot < 1.0) & (fm_trans < 50.0)) * 100),
        'fm_joint_5deg_100mm': float(np.mean((fm_rot < 5.0) & (fm_trans < 100.0)) * 100),
        # FM statistics
        'fm_iters_mean': float(np.mean(iters_arr)),
        'fm_iters_median': float(np.median(iters_arr)),
        # Improvement
        'rot_improvement': float(np.nanmedian(nn_rot) - np.nanmedian(fm_rot)),
        'trans_improvement': float(np.nanmedian(nn_trans) - np.nanmedian(fm_trans)),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed evaluation for concat localization model')
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--checkpoint', required=True, help='Localization model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of evaluation seeds (default: 5)')
    parser.add_argument('--outer_iters', type=int, nargs='+', default=[10],
                        help='Outer iteration counts to test (default: [10])')
    parser.add_argument('--gru_iters', type=int, nargs='+', default=[6],
                        help='GRU iteration counts to test (default: [6])')
    parser.add_argument('--noise_deg', type=float, default=3.0,
                        help='Rotation noise in degrees (default: 3.0)')
    parser.add_argument('--noise_m', type=float, default=0.10,
                        help='Translation noise in meters (default: 0.10)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--solver', type=str, nargs='+',
                        default=['default'],
                        choices=['default', 'wls_full', 'pnp', 'hybrid',
                                 'irls2', 'irls3_gnc', 'direct', 'flow+direct5',
                                 'flow+direct10', 'flow+direct20',
                                 'direct_neg', 'flow+direct5_neg',
                                 'flow+direct5_highdamp', 'flow+direct5_neg_highdamp'],
                        help='Pose solvers to test (default: use model delta_xi)')
    parser.add_argument('--multi_start', type=int, default=0,
                        help='Multi-start count (0=disabled). Runs N perturbations '
                             'per image, picks best by selection metric.')
    parser.add_argument('--ms_selection', type=str, nargs='+',
                        default=['confidence'],
                        choices=['confidence', 'similarity', 'delta_mag', 'consensus'],
                        help='Multi-start selection strategies to test')
    parser.add_argument('--cascade', type=int, default=0,
                        help='Cascade rounds (0=disabled). Multi-start consensus '
                             'followed by refinement rounds with smaller noise.')
    parser.add_argument('--cascade_starts', type=int, default=10,
                        help='Number of starts per cascade round (default: 10)')
    parser.add_argument('--cascade_noise_deg', type=float, default=1.0,
                        help='Rotation noise for cascade refinement rounds (default: 1.0)')
    parser.add_argument('--cascade_noise_m', type=float, default=0.03,
                        help='Translation noise for cascade refinement rounds (default: 0.03)')
    parser.add_argument('--cascade_selection', type=str, default='consensus',
                        choices=['consensus', 'similarity', 'delta_mag'],
                        help='Selection strategy for cascade rounds')
    # Two-stage evaluation
    parser.add_argument('--stage2_config', type=str, default=None,
                        help='Config for Stage 2 model (enables two-stage pipeline)')
    parser.add_argument('--stage2_checkpoint', type=str, default=None,
                        help='Checkpoint for Stage 2 model')
    parser.add_argument('--s1_starts', type=int, default=10,
                        help='Multi-start count for Stage 1 consensus (default: 10)')
    parser.add_argument('--s2_outer_iters', type=int, default=5,
                        help='Outer iterations for Stage 2 refinement (default: 5)')
    parser.add_argument('--s2_oi_sweep', type=int, nargs='+', default=None,
                        help='Sweep Stage 2 outer iters (caches S1 poses). Overrides --s2_outer_iters.')
    parser.add_argument('--s2_gru_iters', type=int, default=6,
                        help='GRU iterations for Stage 2 (default: 6)')
    # Featuremetric refinement
    parser.add_argument('--fm_refine', action='store_true',
                        help='Enable featuremetric GN refinement after NN multi-start')
    parser.add_argument('--fm_iters', type=int, default=30,
                        help='Max featuremetric GN iterations (default: 30)')
    parser.add_argument('--fm_damping', type=float, default=1e-3,
                        help='LM damping factor for featuremetric (default: 1e-3)')
    parser.add_argument('--fm_starts', type=int, default=10,
                        help='Multi-start count for NN stage in FM mode (default: 10)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)

    config = load_mainline_config(args.config)

    # ── Build DCFF rendering pipeline ─────────────────────────────────────
    print('Building DCFF rendering pipeline...')
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)

    # ── Load localization model ───────────────────────────────────────────
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    pose_ckpt = load_concat_pose_checkpoint(args.checkpoint, device)
    restored_map = apply_localization_map_state(
        dcff_renderer,
        feat_sharp,
        pose_ckpt,
        printer=print,
    )
    if restored_map:
        print(f"Restored localization map state: {', '.join(restored_map)}")

    dcff_cfg = config['dcff']
    render_h = dcff_cfg.get('render_height', 68)
    render_w = dcff_cfg.get('render_width', 120)
    use_coarse = config.get('model', {}).get('use_coarse', True)
    ds_cfg = config['dataset']

    # ── Set intrinsics on model (needed for _scale_intrinsics) ────────────
    from data.radio_loc_dataset import read_colmap_cameras, camera_params_to_intrinsics
    colmap_dir = ds_cfg['colmap_dir']
    colmap_cameras = read_colmap_cameras(os.path.join(colmap_dir, 'cameras.bin'))
    first_cam = next(iter(colmap_cameras.values()))
    from data.radio_loc_dataset import camera_params_to_intrinsics
    model.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    model.IMG_HW = (int(first_cam.height), int(first_cam.width))
    print(f'Intrinsics: fx={model.BASE_INTRINSICS["fx"]:.1f} '
          f'fy={model.BASE_INTRINSICS["fy"]:.1f} '
          f'cx={model.BASE_INTRINSICS["cx"]:.1f} '
          f'cy={model.BASE_INTRINSICS["cy"]:.1f}  '
          f'img={first_cam.width}×{first_cam.height}')

    # ── Header ────────────────────────────────────────────────────────────
    print(f'\n{"=" * 72}')
    print(f'Eval: {args.checkpoint}  (epoch {ckpt_epoch})')
    print(f'Config: {args.config}')
    print(f'Noise: {args.noise_deg}° / {args.noise_m}m')
    print(f'Render: {render_w}×{render_h}  Seeds: {args.num_seeds}')
    print(f'Solvers: {args.solver}')
    print(f'{"=" * 72}\n')

    # Map solver names to (solver_type, irls_iters, robust_kernel, direct_refine_iters)
    solver_configs = {
        'default': ('wls', 0, 'huber', 0),
        'wls_full': ('wls_full', 0, 'huber', 0),
        'pnp': ('pnp', 0, 'huber', 0),
        'hybrid': ('hybrid', 0, 'huber', 0),
        'irls2': ('wls', 2, 'huber', 0),
        'irls3_gnc': ('wls', 3, 'gnc_gm', 0),
        'direct': ('direct', 0, 'huber', 0),
        'direct_neg': ('direct_neg', 0, 'huber', 0),
        'flow+direct5': ('wls', 0, 'huber', 5),
        'flow+direct10': ('wls', 0, 'huber', 10),
        'flow+direct20': ('wls', 0, 'huber', 20),
        'flow+direct5_neg': ('flow_neg', 0, 'huber', 5),
        'flow+direct5_highdamp': ('flow_highdamp', 0, 'huber', 5),
        'flow+direct5_neg_highdamp': ('flow_neg_highdamp', 0, 'huber', 5),
    }

    # ── Cascade evaluation (if requested) — checked FIRST ───────────────
    if args.cascade > 0:
        print(f'\n{"═" * 72}')
        print(f'  CASCADE EVALUATION: {args.cascade} rounds × {args.cascade_starts} starts (selection={args.cascade_selection})')
        print(f'  Round 0 noise: {args.noise_deg}° / {args.noise_m}m')
        print(f'  Refine noise: {args.cascade_noise_deg}° / {args.cascade_noise_m}m')
        print(f'{"═" * 72}\n')

        for solver_name in args.solver:
            solver_type, irls_n, robust_k, direct_n = solver_configs[solver_name]
            for outer in args.outer_iters:
                for gru in args.gru_iters:
                    seed_results = []
                    for seed in range(args.num_seeds):
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
                        coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
                        val_ds = RadioLocDataset(
                            feature_dir=ds_cfg['feature_dir'],
                            colmap_dir=ds_cfg['colmap_dir'],
                            split_file=ds_cfg['test_split'],
                            fine_hw=fine_hw,
                            coarse_hw=coarse_hw,
                            noise_rot_deg=args.noise_deg,
                            noise_trans_m=args.noise_m,
                            cache_in_memory=(seed == 0),
                        )

                        metrics = evaluate_cascade(
                            model, gaussians, dcff_renderer, feat_sharp,
                            val_ds, device,
                            outer_iters=outer, gru_iters=gru,
                            render_h=render_h, render_w=render_w,
                            rounds=args.cascade,
                            starts_per_round=args.cascade_starts,
                            refine_noise_deg=args.cascade_noise_deg,
                            refine_noise_m=args.cascade_noise_m,
                            use_coarse=use_coarse,
                            solver=solver_type,
                            irls_iters=irls_n,
                            robust_kernel=robust_k,
                            selection=args.cascade_selection,
                        )
                        seed_results.append(metrics)
                        print(f'  [seed {seed}] cascade({args.cascade}r×{args.cascade_starts}s) '
                              f'oi={outer} gru={gru}  '
                              f'rot={metrics["rot_median"]:.3f}°  '
                              f'trans={metrics["trans_median"]:.1f}mm  '
                              f'<1°={metrics["pct_1deg"]:.1f}%')

                    # Aggregate seeds
                    print(f'\n  ── cascade({args.cascade}r×{args.cascade_starts}s) '
                          f'oi={outer} gru={gru}: {args.num_seeds}-seed summary ──')
                    header = f'  {"metric":<20} {"mean":>10} {"± std":>10} {"min":>10} {"max":>10}'
                    print(header)
                    print(f'  {"-" * 60}')
                    for key in ['rot_mean', 'rot_median', 'trans_mean', 'trans_median',
                                'pct_1deg', 'pct_5deg', 'joint_1deg_50mm',
                                'joint_5deg_100mm']:
                        vals = np.array([r[key] for r in seed_results])
                        if 'rot' in key:
                            unit, fmt = '°', '.3f'
                        elif 'trans' in key:
                            unit, fmt = 'mm', '.1f'
                        else:
                            unit, fmt = '%', '.1f'
                        print(f'  {key:<20} {np.mean(vals):>9{fmt}}{unit} '
                              f'±{np.std(vals):>8{fmt}}{unit} '
                              f'{np.min(vals):>9{fmt}}{unit} '
                              f'{np.max(vals):>9{fmt}}{unit}')
                    print()
        return

    # ── Featuremetric refinement evaluation (if requested) ────────────────
    if args.fm_refine:
        print(f'\n{"═" * 72}')
        print(f'  NN + FEATUREMETRIC REFINEMENT')
        print(f'  NN stage: MS={args.fm_starts} starts, consensus selection')
        print(f'  FM stage: {args.fm_iters} GN iters, damping={args.fm_damping}')
        print(f'  Noise: {args.noise_deg}° / {args.noise_m}m')
        print(f'{"═" * 72}\n')

        for solver_name in args.solver:
            solver_type, irls_n, robust_k, direct_n = solver_configs[solver_name]
            for outer in args.outer_iters:
                for gru in args.gru_iters:
                    seed_results = []
                    for seed in range(args.num_seeds):
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
                        coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
                        val_ds = RadioLocDataset(
                            feature_dir=ds_cfg['feature_dir'],
                            colmap_dir=ds_cfg['colmap_dir'],
                            split_file=ds_cfg['test_split'],
                            fine_hw=fine_hw,
                            coarse_hw=coarse_hw,
                            noise_rot_deg=args.noise_deg,
                            noise_trans_m=args.noise_m,
                            cache_in_memory=(seed == 0),
                        )

                        metrics = evaluate_nn_then_fm(
                            model, gaussians, dcff_renderer, feat_sharp,
                            val_ds, device,
                            outer_iters=outer, gru_iters=gru,
                            render_h=render_h, render_w=render_w,
                            num_starts=args.fm_starts,
                            use_coarse=use_coarse,
                            solver=solver_type,
                            irls_iters=irls_n,
                            robust_kernel=robust_k,
                            selection='consensus',
                            fm_iters=args.fm_iters,
                            fm_damping=args.fm_damping,
                        )
                        seed_results.append(metrics)

                        # Print per-seed results
                        print(f'  [seed {seed}] oi={outer} gru={gru}')
                        print(f'    NN-only:  rot={metrics["nn_rot_median"]:.3f}°  '
                              f'trans={metrics["nn_trans_median"]:.1f}mm  '
                              f'<1°={metrics["nn_pct_1deg"]:.1f}%')
                        print(f'    NN+FM:    rot={metrics["fm_rot_median"]:.3f}°  '
                              f'trans={metrics["fm_trans_median"]:.1f}mm  '
                              f'<1°={metrics["fm_pct_1deg"]:.1f}%  '
                              f'fm_iters={metrics["fm_iters_mean"]:.1f}')
                        print(f'    Δ:        rot={metrics["rot_improvement"]:+.3f}°  '
                              f'trans={metrics["trans_improvement"]:+.1f}mm')

                    # Aggregate across seeds
                    print(f'\n  ── NN+FM oi={outer} gru={gru}: '
                          f'{args.num_seeds}-seed summary ──')
                    header = f'  {"metric":<25} {"mean":>10} {"± std":>10} {"min":>10} {"max":>10}'
                    print(header)
                    print(f'  {"-" * 65}')
                    for key in ['nn_rot_median', 'nn_trans_median',
                                'fm_rot_median', 'fm_trans_median',
                                'fm_pct_1deg', 'fm_pct_5deg',
                                'fm_joint_1deg_50mm', 'fm_joint_5deg_100mm',
                                'fm_iters_mean',
                                'rot_improvement', 'trans_improvement']:
                        vals = np.array([r[key] for r in seed_results])
                        if 'rot' in key or 'improvement' in key and 'trans' not in key:
                            unit, fmt = '°', '.3f'
                        elif 'trans' in key:
                            unit, fmt = 'mm', '.1f'
                        elif 'iters' in key:
                            unit, fmt = '', '.1f'
                        else:
                            unit, fmt = '%', '.1f'
                        print(f'  {key:<25} {np.mean(vals):>9{fmt}}{unit} '
                              f'±{np.std(vals):>8{fmt}}{unit} '
                              f'{np.min(vals):>9{fmt}}{unit} '
                              f'{np.max(vals):>9{fmt}}{unit}')
                    print()
        return

    # ── Multi-start evaluation (if requested) ────────────────────────────
    if args.multi_start > 0:
        print(f'\n{"═" * 72}')
        print(f'  MULTI-START EVALUATION: {args.multi_start} starts per image')
        print(f'  Selection strategies: {args.ms_selection}')
        print(f'{"═" * 72}\n')

        for sel_strategy in args.ms_selection:
            for solver_name in args.solver:
                solver_type, irls_n, robust_k, direct_n = solver_configs[solver_name]
                for outer in args.outer_iters:
                    for gru in args.gru_iters:
                        seed_results = []
                        for seed in range(args.num_seeds):
                            torch.manual_seed(seed)
                            np.random.seed(seed)

                            fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
                            coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
                            val_ds = RadioLocDataset(
                                feature_dir=ds_cfg['feature_dir'],
                                colmap_dir=ds_cfg['colmap_dir'],
                                split_file=ds_cfg['test_split'],
                                fine_hw=fine_hw,
                                coarse_hw=coarse_hw,
                                noise_rot_deg=args.noise_deg,
                                noise_trans_m=args.noise_m,
                                cache_in_memory=(seed == 0),
                            )

                            metrics = evaluate_multistart(
                                model, gaussians, dcff_renderer, feat_sharp,
                                val_ds, device,
                                outer_iters=outer, gru_iters=gru,
                                render_h=render_h, render_w=render_w,
                                num_starts=args.multi_start,
                                use_coarse=use_coarse,
                                solver=solver_type,
                                irls_iters=irls_n,
                                robust_kernel=robust_k,
                                selection=sel_strategy,
                            )
                            seed_results.append(metrics)
                            print(f'  [seed {seed}] sel={sel_strategy} MS={args.multi_start} '
                                  f'oi={outer} gru={gru}  '
                                  f'rot={metrics["rot_median"]:.3f}°  '
                                  f'trans={metrics["trans_median"]:.1f}mm  '
                                  f'score={metrics["score_mean"]:.4f}  '
                                  f'<1°={metrics["pct_1deg"]:.1f}%')

                        # Aggregate
                        print(f'\n  ── sel={sel_strategy} MS={args.multi_start} '
                              f'oi={outer} gru={gru}: {args.num_seeds}-seed summary ──')
                        header = f'  {"metric":<20} {"mean":>10} {"± std":>10} {"min":>10} {"max":>10}'
                        print(header)
                        print(f'  {"-" * 60}')
                        for key in ['rot_mean', 'rot_median', 'trans_mean', 'trans_median',
                                    'pct_1deg', 'pct_5deg', 'joint_1deg_50mm',
                                    'joint_5deg_100mm', 'score_mean', 'score_median']:
                            vals = np.array([r[key] for r in seed_results])
                            if 'rot' in key:
                                unit, fmt = '°', '.3f'
                            elif 'trans' in key:
                                unit, fmt = 'mm', '.1f'
                            elif 'score' in key:
                                unit, fmt = '', '.4f'
                            else:
                                unit, fmt = '%', '.1f'
                            print(f'  {key:<20} {np.mean(vals):>9{fmt}}{unit} '
                                  f'±{np.std(vals):>8{fmt}}{unit} '
                                  f'{np.min(vals):>9{fmt}}{unit} '
                                  f'{np.max(vals):>9{fmt}}{unit}')
                        print()
        return

    # ── Two-stage evaluation (if requested) ────────────────────────────────
    if args.stage2_config and args.stage2_checkpoint:
        s2_oi_list = args.s2_oi_sweep if args.s2_oi_sweep else [args.s2_outer_iters]
        use_sweep = len(s2_oi_list) > 1

        print(f'\n{"═" * 72}')
        print(f'  TWO-STAGE EVALUATION' + (' (sweep mode)' if use_sweep else ''))
        print(f'  Stage 1: {args.checkpoint} + MS={args.s1_starts} consensus')
        print(f'  Stage 2: {args.stage2_checkpoint} × oi={s2_oi_list}')
        print(f'  Noise: {args.noise_deg}° / {args.noise_m}m')
        print(f'{"═" * 72}\n')

        # Load Stage 2 model
        s2_config = load_mainline_config(args.stage2_config)
        s2_model, s2_epoch = load_model(s2_config, args.stage2_checkpoint, device)
        print(f'Stage 2 model loaded from epoch {s2_epoch}')

        for outer in args.outer_iters:
            for gru in args.gru_iters:
                seed_results = {oi: [] for oi in s2_oi_list}
                for seed in range(args.num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
                    coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
                    val_ds = RadioLocDataset(
                        feature_dir=ds_cfg['feature_dir'],
                        colmap_dir=ds_cfg['colmap_dir'],
                        split_file=ds_cfg['test_split'],
                        fine_hw=fine_hw,
                        coarse_hw=coarse_hw,
                        noise_rot_deg=args.noise_deg,
                        noise_trans_m=args.noise_m,
                        cache_in_memory=(seed == 0),
                    )

                    apply_localization_map_state(
                        dcff_renderer,
                        feat_sharp,
                        pose_ckpt,
                        printer=None,
                    )

                    if use_sweep:
                        s2_ckpt = load_concat_pose_checkpoint(args.stage2_checkpoint, device)
                        restored_s2_map = apply_localization_map_state(
                            dcff_renderer,
                            feat_sharp,
                            s2_ckpt,
                            printer=print if seed == 0 else None,
                        )
                        if restored_s2_map and seed == 0:
                            print(f"Restored stage-2 map state: {', '.join(restored_s2_map)}")
                        sweep_res = evaluate_twostage_sweep(
                            model, s2_model, gaussians, dcff_renderer, feat_sharp,
                            val_ds, device,
                            s1_outer_iters=outer, s1_gru_iters=gru,
                            s2_gru_iters=args.s2_gru_iters,
                            s2_oi_list=s2_oi_list,
                            render_h=render_h, render_w=render_w,
                            s1_starts=args.s1_starts,
                            use_coarse=use_coarse,
                            solver='wls', irls_iters=0, robust_kernel='huber',
                        )
                        for oi, m in sweep_res.items():
                            seed_results[oi].append(m)
                            print(f'  [seed {seed}] S1→S2(oi={oi})  '
                                  f'S1: rot={m["s1_rot_median"]:.3f}° trans={m["s1_trans_median"]:.1f}mm  →  '
                                  f'S2: rot={m["rot_median"]:.3f}° trans={m["trans_median"]:.1f}mm  '
                                  f'<1°={m["pct_1deg"]:.1f}%')
                    else:
                        s2_ckpt = load_concat_pose_checkpoint(args.stage2_checkpoint, device)
                        restored_s2_map = apply_localization_map_state(
                            dcff_renderer,
                            feat_sharp,
                            s2_ckpt,
                            printer=print if seed == 0 else None,
                        )
                        if restored_s2_map and seed == 0:
                            print(f"Restored stage-2 map state: {', '.join(restored_s2_map)}")
                        metrics = evaluate_twostage(
                            model, s2_model, gaussians, dcff_renderer, feat_sharp,
                            val_ds, device,
                            s1_outer_iters=outer, s1_gru_iters=gru,
                            s2_outer_iters=s2_oi_list[0], s2_gru_iters=args.s2_gru_iters,
                            render_h=render_h, render_w=render_w,
                            s1_starts=args.s1_starts,
                            use_coarse=use_coarse,
                            solver='wls', irls_iters=0, robust_kernel='huber',
                        )
                        seed_results[s2_oi_list[0]].append(metrics)
                        print(f'  [seed {seed}] S1→S2  oi={outer} gru={gru}  '
                              f'S1: rot={metrics["s1_rot_median"]:.3f}° trans={metrics["s1_trans_median"]:.1f}mm  →  '
                              f'S2: rot={metrics["rot_median"]:.3f}° trans={metrics["trans_median"]:.1f}mm  '
                              f'<1°={metrics["pct_1deg"]:.1f}%')

                # Aggregate seeds per s2_oi value
                for s2_oi in s2_oi_list:
                    results_list = seed_results[s2_oi]
                    if not results_list:
                        continue
                    print(f'\n  ── TwoStage S1(ms={args.s1_starts},oi={outer}) → S2(oi={s2_oi}): '
                          f'{len(results_list)}-seed summary ──')
                    header = f'  {"metric":<20} {"mean":>10} {"± std":>10} {"min":>10} {"max":>10}'
                    print(header)
                    print(f'  {"-" * 60}')
                    for key in ['rot_mean', 'rot_median', 'trans_mean', 'trans_median',
                                'pct_1deg', 'pct_5deg', 'joint_1deg_50mm',
                                'joint_5deg_100mm', 's1_rot_median', 's1_trans_median']:
                        vals = np.array([r[key] for r in results_list])
                        if 'rot' in key:
                            unit, fmt = '°', '.3f'
                        elif 'trans' in key:
                            unit, fmt = 'mm', '.1f'
                        else:
                            unit, fmt = '%', '.1f'
                        print(f'  {key:<20} {np.mean(vals):>9{fmt}}{unit} '
                              f'±{np.std(vals):>8{fmt}}{unit} '
                              f'{np.min(vals):>9{fmt}}{unit} '
                              f'{np.max(vals):>9{fmt}}{unit}')
                print()
        return

    # ── Multi-seed evaluation ─────────────────────────────────────────────
    for solver_name in args.solver:
        solver_type, irls_n, robust_k, direct_n = solver_configs[solver_name]
        print(f'\n{"─" * 72}')
        print(f'  Solver: {solver_name} (type={solver_type}, irls={irls_n}, kernel={robust_k}, direct={direct_n})')
        print(f'{"─" * 72}')

        for outer in args.outer_iters:
            for gru in args.gru_iters:
                seed_results = []

                for seed in range(args.num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
                    coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
                    val_ds = RadioLocDataset(
                        feature_dir=ds_cfg['feature_dir'],
                        colmap_dir=ds_cfg['colmap_dir'],
                        split_file=ds_cfg['test_split'],
                        fine_hw=fine_hw,
                        coarse_hw=coarse_hw,
                        noise_rot_deg=args.noise_deg,
                        noise_trans_m=args.noise_m,
                        cache_in_memory=(seed == 0),
                    )
                    val_loader = DataLoader(
                        val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn,
                    )

                    metrics = evaluate(
                        model, gaussians, dcff_renderer, feat_sharp,
                        val_loader, device,
                        outer_iters=outer, gru_iters=gru,
                        render_h=render_h, render_w=render_w,
                        use_coarse=use_coarse,
                        solver=solver_type,
                        irls_iters=irls_n,
                        robust_kernel=robust_k,
                        direct_refine_iters=direct_n,
                    )
                    seed_results.append(metrics)

                    print(f'  [seed {seed}] solver={solver_name} oi={outer} gru={gru}  '
                          f'rot={metrics["rot_median"]:.3f}°  '
                          f'trans={metrics["trans_median"]:.1f}mm  '
                          f'<1°={metrics["pct_1deg"]:.1f}%  '
                          f'j@1/50={metrics["joint_1deg_50mm"]:.1f}%')

                # ── Aggregate across seeds ────────────────────────────────
                print(f'\n  ── solver={solver_name} oi={outer} gru={gru}: '
                      f'{args.num_seeds}-seed summary ──')
                header = f'  {"metric":<20} {"mean":>10} {"± std":>10} {"min":>10} {"max":>10}'
                print(header)
                print(f'  {"-" * 60}')

                for key in ['rot_mean', 'rot_median', 'trans_mean', 'trans_median',
                            'pct_1deg', 'pct_5deg', 'joint_01deg_53mm',
                            'joint_1deg_50mm', 'joint_5deg_100mm']:
                    vals = np.array([r[key] for r in seed_results])
                    unit = '°' if 'rot' in key else ('mm' if 'trans' in key else '%')
                    fmt = '.3f' if 'rot' in key else ('.1f' if 'trans' in key else '.1f')
                    print(f'  {key:<20} {np.mean(vals):>9{fmt}}{unit} '
                          f'±{np.std(vals):>8{fmt}}{unit} '
                          f'{np.min(vals):>9{fmt}}{unit} '
                          f'{np.max(vals):>9{fmt}}{unit}')
                print()


if __name__ == '__main__':
    main()
