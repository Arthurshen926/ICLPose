#!/usr/bin/env python3
"""
Line-search evaluation for PoseRefiner on OldHospital.

At each outer iteration, instead of blindly applying the full delta_xi,
tests multiple step sizes α and selects the one that minimizes feature
matching cost (MSE between query and rendered features).

This prevents overshoot without needing to retrain the model.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_line_search.py \
        --config configs/refiner_oldhospital.yaml \
        --checkpoint output/refiner_oh_v1/checkpoints/latest.pth \
        --noise_deg 8 5 3 --outer_iters 5 \
        --alphas 0.3 0.5 0.7 1.0 1.3
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

from ic_models.pose_refiner import PoseRefiner
from modules.lie_algebra import se3_exp
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


class RadioRenderer:
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

    def render_features(self, viewmat, feat_hw=(68, 120)):
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        fH, fW = feat_hw
        result = FeatureRenderer.render_features_batch(
            self.gs_model, viewmat,
            fx=self.fx * fW / self.img_hw[1],
            fy=self.fy * fH / self.img_hw[0],
            cx=self.cx * fW / self.img_hw[1],
            cy=self.cy * fH / self.img_hw[0],
            img_height=fH, img_width=fW,
            norm_feat_before_render=True, norm_feat_after_render=True)
        return result['feature_map']

    def render_depth(self, viewmat, depth_hw=(68, 120)):
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        dH, dW = depth_hw
        depths = []
        for i in range(viewmat.shape[0]):
            d = FeatureRenderer.render_depth(
                self.gs_model, viewmat[i],
                fx=self.fx * dW / self.img_hw[1],
                fy=self.fy * dH / self.img_hw[0],
                cx=self.cx * dW / self.img_hw[1],
                cy=self.cy * dH / self.img_hw[0],
                img_height=dH, img_width=dW)
            depths.append(d)
        return torch.stack(depths)


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


def feature_cost(query_feat, rendered_feat):
    """MSE between L2-normalized query and rendered features. Lower = better match."""
    q = F.normalize(query_feat, dim=1)
    r = F.normalize(rendered_feat, dim=1)
    return (q - r).pow(2).mean().item()


@torch.no_grad()
def evaluate(config, checkpoint, noise_degs, max_outer_iters, alphas, device='cuda'):
    rc = config['renderer']
    feat_hw = tuple(rc.get('feat_hw', [68, 120]))

    # Load model
    mc = config.get('model', {})
    intrinsics = {'fx': rc['fx'], 'fy': rc['fy'], 'cx': rc['cx'], 'cy': rc['cy']}
    model = PoseRefiner(
        in_dim=mc.get('in_dim', 64),
        match_dim=mc.get('match_dim', 64),
        hidden_dim=mc.get('hidden_dim', 128),
        n_heads=mc.get('n_heads', 4),
        n_attn_layers=mc.get('n_attn_layers', 2),
        ffn_dim=mc.get('ffn_dim', 128),
        local_radius=mc.get('local_radius', 4),
        fine_iters=mc.get('fine_iters', 8),
        damping=mc.get('damping', 1e-3),
        coarse_hw=tuple(mc.get('coarse_hw', [17, 30])),
        fine_hw=tuple(mc.get('fine_hw', [34, 60])),
        solver_upsample=mc.get('solver_upsample', 4),
        solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
        intrinsics=intrinsics,
        img_hw=tuple(rc.get('img_hw', [1080, 1920])),
        depth_normalize=mc.get('depth_normalize', True),
        sequential_solve=mc.get('sequential_solve', True),
        detach_conf=mc.get('detach_conf', True),
        conf_floor=mc.get('conf_floor', 0.1),
        use_trans_head=mc.get('use_trans_head', False),
        trans_head_mode=mc.get('trans_head_mode', 'replace'),
        solver_trans_scale=mc.get('solver_trans_scale', 0.0),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")
    print(f"Alpha candidates: {alphas}")

    # Load renderer
    renderer = RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        tuple(rc.get('img_hw', [1080, 1920])),
        rc['fx'], rc['fy'], rc['cx'], rc['cy'])

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

    for noise_deg in noise_degs:
        noise_trans = noise_deg / 8.0 * 0.25
        print(f"\n{'='*70}")
        print(f"  Noise: {noise_deg}° / {noise_trans:.2f}m  |  max_iters={max_outer_iters}")
        print(f"{'='*70}")

        # Track results for: no-LS baseline (α=1.0 always) and LS (best α)
        per_iter_rot_nols = {n: [] for n in range(max_outer_iters + 1)}
        per_iter_rot_ls = {n: [] for n in range(max_outer_iters + 1)}
        per_iter_trans_nols = {n: [] for n in range(max_outer_iters + 1)}
        per_iter_trans_ls = {n: [] for n in range(max_outer_iters + 1)}
        alpha_chosen = {n: [] for n in range(1, max_outer_iters + 1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)

            # Two parallel pose tracks
            pose_nols = noisy_w2c.unsqueeze(0)
            pose_ls = noisy_w2c.unsqueeze(0)

            pos_err, rot_err = pose_error(noisy_w2c, gt_w2c)
            per_iter_rot_nols[0].append(rot_err)
            per_iter_rot_ls[0].append(rot_err)
            per_iter_trans_nols[0].append(pos_err)
            per_iter_trans_ls[0].append(pos_err)

            for it in range(1, max_outer_iters + 1):
                # === No Line Search (baseline α=1.0) ===
                ref_feat_nols = renderer.render_features(pose_nols, feat_hw)
                depth_nols = renderer.render_depth(pose_nols, feat_hw)
                result_nols = model(feat.to(device), ref_feat_nols, depth_nols)
                if 'delta_xi' in result_nols:
                    pose_nols = se3_exp(result_nols['delta_xi']) @ pose_nols

                pos_err, rot_err = pose_error(pose_nols[0], gt_w2c)
                per_iter_rot_nols[it].append(rot_err)
                per_iter_trans_nols[it].append(pos_err)

                # === Line Search (best α) ===
                ref_feat_ls = renderer.render_features(pose_ls, feat_hw)
                depth_ls = renderer.render_depth(pose_ls, feat_hw)
                result_ls = model(feat.to(device), ref_feat_ls, depth_ls)

                if 'delta_xi' in result_ls:
                    delta_xi = result_ls['delta_xi']
                    best_alpha = 1.0
                    best_cost = float('inf')
                    best_pose = None

                    for alpha in alphas:
                        scaled_xi = delta_xi * alpha
                        candidate_pose = se3_exp(scaled_xi) @ pose_ls
                        # Render features at candidate pose to evaluate cost
                        candidate_feat = renderer.render_features(candidate_pose, feat_hw)
                        cost = feature_cost(feat.to(device), candidate_feat)
                        if cost < best_cost:
                            best_cost = cost
                            best_alpha = alpha
                            best_pose = candidate_pose

                    pose_ls = best_pose
                    alpha_chosen[it].append(best_alpha)

                pos_err, rot_err = pose_error(pose_ls[0], gt_w2c)
                per_iter_rot_ls[it].append(rot_err)
                per_iter_trans_ls[it].append(pos_err)

            if (idx_i + 1) % 20 == 0:
                print(f"  [{idx_i+1}/{len(test_indices)}] "
                      f"NoLS: {np.median(per_iter_rot_nols[max_outer_iters]):.2f}° "
                      f"LS: {np.median(per_iter_rot_ls[max_outer_iters]):.2f}°")

        # Report
        print(f"\n  --- Results at {noise_deg}° noise ---")
        print(f"  {'Iter':<6} {'NoLS rot':>10} {'LS rot':>10} {'NoLS trans':>12} {'LS trans':>12} {'αmean':>8} {'αmed':>8}")
        print(f"  {'-'*68}")
        for i in range(max_outer_iters + 1):
            nols_r = np.median(per_iter_rot_nols[i])
            ls_r = np.median(per_iter_rot_ls[i])
            nols_t = np.median(per_iter_trans_nols[i])
            ls_t = np.median(per_iter_trans_ls[i])
            if i > 0:
                am = np.mean(alpha_chosen[i])
                amd = np.median(alpha_chosen[i])
                print(f"  I{i:<5} {nols_r:>9.2f}° {ls_r:>9.2f}° {nols_t:>10.1f}cm {ls_t:>10.1f}cm {am:>8.2f} {amd:>8.2f}")
            else:
                print(f"  I{i:<5} {nols_r:>9.2f}° {ls_r:>9.2f}° {nols_t:>10.1f}cm {ls_t:>10.1f}cm {'init':>8} {'init':>8}")

        # Summary stats
        nols_final_r = np.median(per_iter_rot_nols[max_outer_iters])
        ls_final_r = np.median(per_iter_rot_ls[max_outer_iters])
        nols_final_t = np.median(per_iter_trans_nols[max_outer_iters])
        ls_final_t = np.median(per_iter_trans_ls[max_outer_iters])
        print(f"\n  NoLS final: {nols_final_r:.2f}° / {nols_final_t:.1f}cm  "
              f"(P90: {np.percentile(per_iter_rot_nols[max_outer_iters], 90):.2f}°)")
        print(f"  LS   final: {ls_final_r:.2f}° / {ls_final_t:.1f}cm  "
              f"(P90: {np.percentile(per_iter_rot_ls[max_outer_iters], 90):.2f}°)")
        improvement = (nols_final_r - ls_final_r) / nols_final_r * 100
        print(f"  LS improvement: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8, 5, 3])
    parser.add_argument('--outer_iters', type=int, default=5)
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.3, 0.5, 0.7, 1.0, 1.3])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint, args.noise_deg, args.outer_iters, args.alphas)


if __name__ == '__main__':
    main()
