#!/usr/bin/env python3
"""
Fixed-seed evaluation for PoseRefiner on OldHospital.

Evaluates at multiple noise levels and outer iteration counts
with deterministic noise for reproducible metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_refiner.py \
        --config configs/refiner_oldhospital.yaml \
        --checkpoint output/refiner_oh_v1/checkpoints/best.pth \
        --noise_deg 8 5 3 --outer_iters 1 3 5
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
    """Add deterministic noise using a specific seed."""
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


@torch.no_grad()
def evaluate(config, checkpoint, noise_degs, outer_iters_list, device='cuda',
             flow_scale=1.0, flow_scale_schedule=None):
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
        use_flow_scale_head=mc.get('use_flow_scale_head', False),
        raw_coarse_corr=mc.get('raw_coarse_corr', False),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    if flow_scale != 1.0:
        model.flow_scale = flow_scale
        print(f"Applied flow_scale = {flow_scale}")
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

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

    # Load test indices
    test_idx_path = os.path.join(dc['feature_dir'], 'test_indices.npy')
    if os.path.exists(test_idx_path):
        test_indices = np.load(test_idx_path).tolist()
    else:
        test_indices = sorted(features.keys())
    test_indices = [i for i in test_indices if i in features and i < len(poses_c2w)]
    print(f"Test set: {len(test_indices)} frames")

    max_outer = max(outer_iters_list)

    # Evaluate
    for noise_deg in noise_degs:
        noise_trans = noise_deg / 8.0 * 0.25  # Scale trans proportionally
        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg}° / {noise_trans:.2f}m")
        print(f"{'='*60}")

        per_iter_rot = {n: [] for n in range(max_outer + 1)}
        per_iter_trans = {n: [] for n in range(max_outer + 1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            # Record init error
            pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
            per_iter_rot[0].append(rot_err)
            per_iter_trans[0].append(pos_err)

            for it in range(1, max_outer + 1):
                # Per-iteration flow scaling
                if flow_scale_schedule:
                    fs = flow_scale_schedule[min(it-1, len(flow_scale_schedule)-1)]
                    model.flow_scale = fs
                ref_feat = renderer.render_features(pose_cur, feat_hw)
                depth = renderer.render_depth(pose_cur, feat_hw)
                result = model(feat.to(device), ref_feat, depth)
                if 'delta_xi' in result:
                    pose_cur = se3_exp(result['delta_xi']) @ pose_cur

                pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
                per_iter_rot[it].append(rot_err)
                per_iter_trans[it].append(pos_err)

        # Report results
        for n_iters in outer_iters_list:
            init_rot = np.median(per_iter_rot[0])
            init_trans = np.median(per_iter_trans[0])
            final_rot = np.median(per_iter_rot[n_iters])
            final_trans = np.median(per_iter_trans[n_iters])
            delta_rot = final_rot - init_rot
            delta_trans = final_trans - init_trans

            iter_detail = ' → '.join(
                f'I{i}={np.median(per_iter_rot[i]):.2f}°'
                for i in range(n_iters + 1))

            print(f"\n  N={n_iters}: {init_rot:.2f}° → {final_rot:.2f}° "
                  f"(Δ{delta_rot:+.2f}°) | "
                  f"{init_trans:.1f}cm → {final_trans:.1f}cm "
                  f"(Δ{delta_trans:+.1f}cm)")
            print(f"    {iter_detail}")
            print(f"    Mean rot: {np.mean(per_iter_rot[n_iters]):.2f}°, "
                  f"P90 rot: {np.percentile(per_iter_rot[n_iters], 90):.2f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8, 5, 3])
    parser.add_argument('--outer_iters', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--flow_scale', type=float, default=1.0,
                        help='Multiplicative scale for predicted flow before solver')
    parser.add_argument('--flow_scale_schedule', nargs='+', type=float, default=None,
                        help='Per-iteration flow scales, e.g. 1.0 0.6 0.6')
    parser.add_argument('--fine_iters', type=int, default=None,
                        help='Override GRU iterations (default: use model config)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.fine_iters is not None:
        config.setdefault('model', {})['fine_iters'] = args.fine_iters

    evaluate(config, args.checkpoint, args.noise_deg, args.outer_iters,
             flow_scale=args.flow_scale, flow_scale_schedule=args.flow_scale_schedule)


if __name__ == '__main__':
    main()
