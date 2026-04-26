#!/usr/bin/env python3
"""
Lucas-Kanade Feature-Matching Pose Refinement (no learned components).

Pure Gauss-Newton optimization on feature matching cost:
  min_ξ  Σ_i ||feat_query(p_i) - feat_rendered(π(T·P_i))||²

Uses the Image Jacobian + feature spatial gradient to compute the
total Jacobian of the feature residual w.r.t. SE(3) pose parameters.

This approach has NO learned flow predictor — the "flow" is implicitly
defined by the feature gradient, and the pose update is a direct
least-squares solution.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_lk_refinement.py \
        --config configs/refiner_oldhospital.yaml \
        --noise_deg 8 5 3 1 --outer_iters 1 5 10 20 \
        --lk_damping 0.01 --feat_hw 68 120
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.lie_algebra import se3_exp
from modules.geometry_solver import compute_image_jacobian
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


class RadioRenderer:
    """Render RADIO features + depth from 2DGS model."""
    def __init__(self, ply_path, feature_model_path, device,
                 img_hw=(1080, 1920), fx=1663.12, fy=1663.12,
                 cx=960.0, cy=540.0):
        self.device = device
        self.img_hw = img_hw
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        ckpt = torch.load(feature_model_path, map_location='cpu', weights_only=True)
        feat_dim = ckpt['feature_dim']
        self.gs_model = GaussianFeatureModel(feature_dim=feat_dim)
        self.gs_model.load_ply(ply_path)
        self.gs_model._loc_feature = nn.Parameter(ckpt['loc_feature'].to(device))
        self.gs_model = self.gs_model.to(device)
        self.gs_model.eval()

    def render(self, viewmat, feat_hw=(68, 120)):
        """Render features and depth in one call."""
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        fH, fW = feat_hw
        s_fx = self.fx * fW / self.img_hw[1]
        s_fy = self.fy * fH / self.img_hw[0]
        s_cx = self.cx * fW / self.img_hw[1]
        s_cy = self.cy * fH / self.img_hw[0]

        result = FeatureRenderer.render_features_batch(
            self.gs_model, viewmat,
            fx=s_fx, fy=s_fy, cx=s_cx, cy=s_cy,
            img_height=fH, img_width=fW,
            norm_feat_before_render=True, norm_feat_after_render=True)
        feat_map = result['feature_map']

        depths = []
        for i in range(viewmat.shape[0]):
            d = FeatureRenderer.render_depth(
                self.gs_model, viewmat[i],
                fx=s_fx, fy=s_fy, cx=s_cx, cy=s_cy,
                img_height=fH, img_width=fW)
            depths.append(d)
        depth = torch.stack(depths)

        return feat_map, depth


def add_noise_deterministic(pose_w2c, rot_deg, trans_m, seed):
    device = pose_w2c.device
    rng = torch.Generator()
    rng.manual_seed(seed)
    axis = torch.randn(3, generator=rng)
    axis = axis / (axis.norm() + 1e-8)
    angle = rot_deg * math.pi / 180.0
    omega = axis * angle
    direction = torch.randn(3, generator=rng)
    direction = direction / (direction.norm() + 1e-8)
    trans = direction * trans_m
    xi = torch.cat([trans, omega]).to(device)
    delta_T = se3_exp(xi)
    return delta_T @ pose_w2c


def pose_error(pred_w2c, gt_w2c):
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)
    pos_err = (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel.diagonal().sum()
    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rot_err = torch.acos(cos_a).item() * 180.0 / math.pi
    return pos_err, rot_err


def compute_feature_gradient(feat_map, smooth_sigma=0.0):
    """Compute spatial gradient of feature map.

    Args:
        feat_map: (B, C, H, W) feature map
        smooth_sigma: if > 0, apply Gaussian smoothing before gradient

    Returns:
        grad_x: (B, C, H, W) gradient in x direction
        grad_y: (B, C, H, W) gradient in y direction
    """
    if smooth_sigma > 0:
        # Gaussian smoothing per channel
        k = max(int(smooth_sigma * 3) * 2 + 1, 3)
        B, C, H, W = feat_map.shape
        # Apply Gaussian blur
        feat_map = F.avg_pool2d(
            F.pad(feat_map, [k//2]*4, mode='replicate'),
            kernel_size=k, stride=1, padding=0)

    # Central differences with replicate padding
    pad = F.pad(feat_map, [1, 1, 1, 1], mode='replicate')
    grad_x = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]) / 2.0
    grad_y = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]) / 2.0
    return grad_x, grad_y


def lk_pose_step(query_feat, rendered_feat, depth, intrinsics, feat_hw,
                 img_hw, damping=0.01, sequential=True, trans_scale=1.0,
                 smooth_sigma=0.0):
    """Compute one Lucas-Kanade pose update step.

    The LK feature residual is:
        r_i = feat_query(p_i) - feat_rendered(p_i)  for each pixel p_i

    The total Jacobian per pixel is:
        J_total_i = [grad_x_feat(p_i), grad_y_feat(p_i)] @ J_image(p_i)
                   = (C, 2) @ (2, 6) = (C, 6)

    The GN update solves:
        (Σ J_i^T J_i + λI) δξ = Σ J_i^T r_i

    Args:
        query_feat: (B, C, H, W) query feature map
        rendered_feat: (B, C, H, W) rendered feature map at current pose
        depth: (B, H, W) depth map at current pose
        intrinsics: dict with fx, fy, cx, cy at ORIGINAL image resolution
        feat_hw: (H, W) feature resolution
        img_hw: (H, W) original image resolution
        damping: LM damping factor
        sequential: if True, solve rotation first, then translation
        trans_scale: scale factor for translation component

    Returns:
        delta_xi: (B, 6) pose update in se(3) [tx,ty,tz,wx,wy,wz]
    """
    B, C, H, W = query_feat.shape

    # Scale intrinsics to feature resolution
    sx = W / img_hw[1]
    sy = H / img_hw[0]
    scaled_intrinsics = {
        'fx': intrinsics['fx'] * sx,
        'fy': intrinsics['fy'] * sy,
        'cx': intrinsics['cx'] * sx,
        'cy': intrinsics['cy'] * sy,
    }

    # Feature residual: r = query - rendered
    residual = query_feat - rendered_feat  # (B, C, H, W)

    # Spatial gradient of rendered features
    grad_x, grad_y = compute_feature_gradient(rendered_feat, smooth_sigma=smooth_sigma)  # (B, C, H, W)

    # Image Jacobian: how pixel positions change with pose
    Ju, Jv, valid = compute_image_jacobian(depth, scaled_intrinsics)
    # Ju, Jv: (B, N, 6), valid: (B, N)  where N = H*W

    # Reshape for computation
    N = H * W
    residual_flat = residual.reshape(B, C, N).permute(0, 2, 1)  # (B, N, C)
    grad_x_flat = grad_x.reshape(B, C, N).permute(0, 2, 1)      # (B, N, C)
    grad_y_flat = grad_y.reshape(B, C, N).permute(0, 2, 1)      # (B, N, C)

    # Total Jacobian per pixel: (B, N, C, 6)
    # J_total = grad_x * Ju + grad_y * Jv  (outer products over C and 6)
    # For each pixel i: J_total[i] = grad_x[i].unsqueeze(-1) @ Ju[i].unsqueeze(-2) + ...
    # But this is (B, N, C, 6) which is large. Instead, compute JtJ and Jtr directly.

    # JtJ = Σ_i J_i^T J_i where J_i is (C, 6)
    # Jtr = Σ_i J_i^T r_i where r_i is (C,)

    # J_i = grad_x_i (C,1) @ Ju_i (1,6) + grad_y_i (C,1) @ Jv_i (1,6)
    # J_i^T r_i = Ju_i * (grad_x_i · r_i) + Jv_i * (grad_y_i · r_i)

    # Compute dot products: grad · residual per pixel
    gx_dot_r = (grad_x_flat * residual_flat).sum(dim=2)  # (B, N)
    gy_dot_r = (grad_y_flat * residual_flat).sum(dim=2)  # (B, N)

    # Apply valid mask
    valid_f = valid.float()
    gx_dot_r = gx_dot_r * valid_f
    gy_dot_r = gy_dot_r * valid_f

    if sequential:
        # ── Solve rotation first (DOFs 3,4,5), then translation (DOFs 0,1,2) ──

        # Rotation-only Jacobian (last 3 columns)
        Ju_rot = Ju[:, :, 3:]  # (B, N, 3)
        Jv_rot = Jv[:, :, 3:]

        Jtr_rot = torch.bmm((gx_dot_r * valid_f).unsqueeze(1), Ju_rot).squeeze(1) + \
                  torch.bmm((gy_dot_r * valid_f).unsqueeze(1), Jv_rot).squeeze(1)
        # (B, 3)

        # JtJ for rotation
        # We need: Σ J_rot_i^T J_rot_i
        # J_rot_i = grad_x_i ⊗ Ju_rot_i + grad_y_i ⊗ Jv_rot_i  (C×3 matrix)
        # J_rot_i^T J_rot_i = (||grad_x||² Ju^T Ju + ||grad_y||² Jv^T Jv
        #                      + (grad_x·grad_y)(Ju^T Jv + Jv^T Ju))

        gx_sq = (grad_x_flat * grad_x_flat).sum(dim=2) * valid_f  # (B, N)
        gy_sq = (grad_y_flat * grad_y_flat).sum(dim=2) * valid_f
        gx_gy = (grad_x_flat * grad_y_flat).sum(dim=2) * valid_f

        # JuJu, JvJv, JuJv as (B, N, 3, 3)
        JuJu_rot = Ju_rot.unsqueeze(-1) * Ju_rot.unsqueeze(-2)  # (B, N, 3, 3)
        JvJv_rot = Jv_rot.unsqueeze(-1) * Jv_rot.unsqueeze(-2)
        JuJv_rot = Ju_rot.unsqueeze(-1) * Jv_rot.unsqueeze(-2)

        JtJ_rot = (gx_sq.unsqueeze(-1).unsqueeze(-1) * JuJu_rot +
                   gy_sq.unsqueeze(-1).unsqueeze(-1) * JvJv_rot +
                   gx_gy.unsqueeze(-1).unsqueeze(-1) * (JuJv_rot + JuJv_rot.transpose(-1, -2)))
        JtJ_rot = JtJ_rot.sum(dim=1)  # (B, 3, 3)

        # LM damping
        diag = torch.diagonal(JtJ_rot, dim1=-2, dim2=-1)
        JtJ_rot = JtJ_rot + damping * torch.diag_embed(diag.clamp(min=1e-6))

        # Solve
        delta_rot = torch.linalg.solve(JtJ_rot, Jtr_rot.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # Translation solve (using full Jacobian but fixing rotation)
        Ju_trans = Ju[:, :, :3]
        Jv_trans = Jv[:, :, :3]

        # Subtract rotation contribution from residual
        # effective_residual = residual - J_rot @ delta_rot
        # In flow space: effective_flow_u -= Ju_rot @ delta_rot, same for v
        flow_correction_u = torch.bmm(Ju_rot, delta_rot.unsqueeze(-1)).squeeze(-1)  # (B, N)
        flow_correction_v = torch.bmm(Jv_rot, delta_rot.unsqueeze(-1)).squeeze(-1)

        # Updated dot products accounting for rotation
        # r_new = r_old - (grad_x * flow_corr_u + grad_y * flow_corr_v)
        gx_dot_r_new = gx_dot_r - (grad_x_flat * grad_x_flat).sum(2) * flow_correction_u \
                       - (grad_x_flat * grad_y_flat).sum(2) * flow_correction_v
        gy_dot_r_new = gy_dot_r - (grad_y_flat * grad_x_flat).sum(2) * flow_correction_u \
                       - (grad_y_flat * grad_y_flat).sum(2) * flow_correction_v
        gx_dot_r_new = gx_dot_r_new * valid_f
        gy_dot_r_new = gy_dot_r_new * valid_f

        Jtr_trans = torch.bmm(gx_dot_r_new.unsqueeze(1), Ju_trans).squeeze(1) + \
                    torch.bmm(gy_dot_r_new.unsqueeze(1), Jv_trans).squeeze(1)

        JuJu_trans = Ju_trans.unsqueeze(-1) * Ju_trans.unsqueeze(-2)
        JvJv_trans = Jv_trans.unsqueeze(-1) * Jv_trans.unsqueeze(-2)
        JuJv_trans = Ju_trans.unsqueeze(-1) * Jv_trans.unsqueeze(-2)

        JtJ_trans = (gx_sq.unsqueeze(-1).unsqueeze(-1) * JuJu_trans +
                     gy_sq.unsqueeze(-1).unsqueeze(-1) * JvJv_trans +
                     gx_gy.unsqueeze(-1).unsqueeze(-1) * (JuJv_trans + JuJv_trans.transpose(-1, -2)))
        JtJ_trans = JtJ_trans.sum(dim=1)
        diag_t = torch.diagonal(JtJ_trans, dim1=-2, dim2=-1)
        JtJ_trans = JtJ_trans + damping * torch.diag_embed(diag_t.clamp(min=1e-6))

        delta_trans = torch.linalg.solve(JtJ_trans, Jtr_trans.unsqueeze(-1)).squeeze(-1)
        delta_trans = delta_trans * trans_scale

        delta_xi = torch.cat([delta_trans, delta_rot], dim=1)  # (B, 6)

    else:
        # ── Joint solve for all 6 DOFs ──
        Jtr = torch.bmm(gx_dot_r.unsqueeze(1), Ju).squeeze(1) + \
              torch.bmm(gy_dot_r.unsqueeze(1), Jv).squeeze(1)  # (B, 6)

        gx_sq = (grad_x_flat * grad_x_flat).sum(dim=2) * valid_f
        gy_sq = (grad_y_flat * grad_y_flat).sum(dim=2) * valid_f
        gx_gy = (grad_x_flat * grad_y_flat).sum(dim=2) * valid_f

        JuJu = Ju.unsqueeze(-1) * Ju.unsqueeze(-2)
        JvJv = Jv.unsqueeze(-1) * Jv.unsqueeze(-2)
        JuJv = Ju.unsqueeze(-1) * Jv.unsqueeze(-2)

        JtJ = (gx_sq.unsqueeze(-1).unsqueeze(-1) * JuJu +
               gy_sq.unsqueeze(-1).unsqueeze(-1) * JvJv +
               gx_gy.unsqueeze(-1).unsqueeze(-1) * (JuJv + JuJv.transpose(-1, -2)))
        JtJ = JtJ.sum(dim=1)
        diag = torch.diagonal(JtJ, dim1=-2, dim2=-1)
        JtJ = JtJ + damping * torch.diag_embed(diag.clamp(min=1e-6))

        # Scale translation
        if trans_scale != 1.0:
            Jtr[:, :3] *= trans_scale

        delta_xi = torch.linalg.solve(JtJ, Jtr.unsqueeze(-1)).squeeze(-1)

    return delta_xi


@torch.no_grad()
def evaluate(config, noise_degs, outer_iters_list, feat_hw=(68, 120),
             lk_damping=0.01, sequential=True, trans_scale=0.0,
             smooth_sigma=0.0, device='cuda'):
    rc = config['renderer']
    intrinsics = {'fx': rc['fx'], 'fy': rc['fy'], 'cx': rc['cx'], 'cy': rc['cy']}
    img_hw = tuple(rc.get('img_hw', [1080, 1920]))

    # Load renderer
    renderer = RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        img_hw, rc['fx'], rc['fy'], rc['cx'], rc['cy'])
    print(f"Renderer loaded. Feature HW: {feat_hw}, Damping: {lk_damping}, Smooth σ: {smooth_sigma}")

    # Load data
    dc = config['data']
    feature_dir = Path(dc['feature_dir']) / 'fine_radio'
    traj_path = dc['traj_path']

    pattern = re.compile(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt')
    features = {}
    for f in sorted(feature_dir.iterdir()):
        m = pattern.match(f.name)
        if m:
            features[int(m.group(1))] = f

    poses_c2w = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses_c2w.append(np.array(vals).reshape(4, 4))

    test_idx_path = os.path.join(dc['feature_dir'], 'test_indices.npy')
    if os.path.exists(test_idx_path):
        test_indices = np.load(test_idx_path).tolist()
    else:
        test_indices = sorted(features.keys())
    test_indices = [i for i in test_indices if i in features and i < len(poses_c2w)]
    print(f"Test set: {len(test_indices)} frames")

    max_outer = max(outer_iters_list)

    for noise_deg in noise_degs:
        noise_trans = noise_deg / 8.0 * 0.25
        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg}° / {noise_trans:.2f}m")
        print(f"{'='*60}")

        per_iter_rot = {n: [] for n in range(max_outer + 1)}
        per_iter_trans = {n: [] for n in range(max_outer + 1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            feat = F.normalize(feat, p=2, dim=1)

            # Resize to eval resolution
            feat = F.interpolate(feat, feat_hw, mode='bilinear', align_corners=False)
            feat = F.normalize(feat, p=2, dim=1)

            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
            per_iter_rot[0].append(rot_err)
            per_iter_trans[0].append(pos_err)

            for it in range(1, max_outer + 1):
                rendered_feat, depth = renderer.render(pose_cur, feat_hw)
                rendered_feat = F.normalize(rendered_feat, p=2, dim=1)

                delta_xi = lk_pose_step(
                    feat, rendered_feat, depth.squeeze(1),
                    intrinsics, feat_hw, img_hw,
                    damping=lk_damping, sequential=sequential,
                    trans_scale=trans_scale, smooth_sigma=smooth_sigma)

                pose_cur = se3_exp(delta_xi) @ pose_cur

                pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
                per_iter_rot[it].append(rot_err)
                per_iter_trans[it].append(pos_err)

        # Report
        for n_iters in outer_iters_list:
            init_rot = np.median(per_iter_rot[0])
            init_trans = np.median(per_iter_trans[0])
            final_rot = np.median(per_iter_rot[n_iters])
            final_trans = np.median(per_iter_trans[n_iters])

            iter_detail = ' → '.join(
                f'I{i}={np.median(per_iter_rot[i]):.2f}°'
                for i in range(n_iters + 1))

            print(f"\n  N={n_iters}: {init_rot:.2f}° → {final_rot:.2f}° "
                  f"(Δ{final_rot - init_rot:+.2f}°) | "
                  f"{init_trans:.1f}cm → {final_trans:.1f}cm "
                  f"(Δ{final_trans - init_trans:+.1f}cm)")
            print(f"    {iter_detail}")
            print(f"    Mean rot: {np.mean(per_iter_rot[n_iters]):.2f}°, "
                  f"P90 rot: {np.percentile(per_iter_rot[n_iters], 90):.2f}°")


def main():
    parser = argparse.ArgumentParser(description='LK Feature-Matching Pose Refinement')
    parser.add_argument('--config', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8, 5, 3, 1])
    parser.add_argument('--outer_iters', nargs='+', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--feat_hw', nargs=2, type=int, default=[68, 120])
    parser.add_argument('--lk_damping', type=float, default=0.01)
    parser.add_argument('--sequential', action='store_true', default=True)
    parser.add_argument('--joint', action='store_true',
                        help='Use joint 6-DOF solve instead of sequential')
    parser.add_argument('--trans_scale', type=float, default=0.0,
                        help='Translation scale (0=disable)')
    parser.add_argument('--smooth_sigma', type=float, default=0.0,
                        help='Gaussian smoothing sigma for feature gradient')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sequential = not args.joint

    evaluate(config, args.noise_deg, args.outer_iters,
             feat_hw=tuple(args.feat_hw), lk_damping=args.lk_damping,
             sequential=sequential, trans_scale=args.trans_scale,
             smooth_sigma=args.smooth_sigma)


if __name__ == '__main__':
    main()
