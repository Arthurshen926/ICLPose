#!/usr/bin/env python3
"""
Adaptive step-size evaluation for PoseRefiner.

At each outer iteration, scale the predicted flow by a factor that
decreases with iteration count: alpha_k = alpha_init * decay^k.

This prevents the later iterations (at smaller residual errors) from
overshooting while keeping the early iterations aggressive.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_adaptive.py \
        --config configs/refiner_oh_v8c.yaml \
        --checkpoint output/refiner_oh_v8c/checkpoints/latest.pth \
        --noise_deg 8 5 3 \
        --outer_iters 5 \
        --alpha_init 1.0 --decay 0.7
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.pose_refiner import PoseRefiner
from modules.lie_algebra import se3_exp
from scripts.eval_refiner import RadioRenderer, add_noise_deterministic, pose_error


def load_model(config, checkpoint, device):
    mc = config.get('model', {})
    rc = config['renderer']
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
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, ckpt.get('epoch', '?')


@torch.no_grad()
def evaluate_adaptive(config, checkpoint, noise_degs, n_iters,
                      alpha_init, decay, device='cuda'):
    rc = config['renderer']
    feat_hw = tuple(rc.get('feat_hw', [68, 120]))

    model, epoch = load_model(config, checkpoint, device)
    print(f"Loaded epoch={epoch}")

    renderer = RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        tuple(rc.get('img_hw', [1080, 1920])),
        rc['fx'], rc['fy'], rc['cx'], rc['cy'])

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

    # Print step sizes
    alphas = [alpha_init * (decay ** k) for k in range(n_iters)]
    print(f"Step sizes: {' → '.join(f'{a:.2f}' for a in alphas)}")

    for noise_deg in noise_degs:
        noise_trans = noise_deg / 8.0 * 0.25
        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg}° / {noise_trans:.2f}m  |  {n_iters} iters")
        print(f"{'='*60}")

        per_iter_rot = {n: [] for n in range(n_iters + 1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            _, rot_err = pose_error(pose_cur[0], gt_w2c)
            per_iter_rot[0].append(rot_err)

            for it in range(n_iters):
                ref_feat = renderer.render_features(pose_cur, feat_hw)
                depth = renderer.render_depth(pose_cur, feat_hw)
                result = model(feat.to(device), ref_feat, depth)
                if 'delta_xi' in result:
                    alpha = alphas[it]
                    scaled_xi = result['delta_xi'] * alpha
                    pose_cur = se3_exp(scaled_xi) @ pose_cur

                _, rot_err = pose_error(pose_cur[0], gt_w2c)
                per_iter_rot[it + 1].append(rot_err)

        # Report
        init_rot = np.median(per_iter_rot[0])
        final_rot = np.median(per_iter_rot[n_iters])
        iter_detail = ' → '.join(
            f'I{i}={np.median(per_iter_rot[i]):.2f}°'
            for i in range(n_iters + 1))

        print(f"\n  {init_rot:.2f}° → {final_rot:.2f}° "
              f"(Δ{final_rot - init_rot:+.2f}°, "
              f"{(init_rot - final_rot)/init_rot*100:.1f}%↓)")
        print(f"  {iter_detail}")
        print(f"  Mean: {np.mean(per_iter_rot[n_iters]):.2f}°, "
              f"P90: {np.percentile(per_iter_rot[n_iters], 90):.2f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8, 5, 3])
    parser.add_argument('--outer_iters', type=int, default=5)
    parser.add_argument('--alpha_init', type=float, default=1.0)
    parser.add_argument('--decay', type=float, default=0.7)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_adaptive(config, args.checkpoint, args.noise_deg, args.outer_iters,
                      args.alpha_init, args.decay)


if __name__ == '__main__':
    main()
