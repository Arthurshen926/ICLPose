#!/usr/bin/env python3
"""
Cascade evaluation: Run two PoseRefiner models in sequence.

Stage 1 (fast, aggressive): V5 for N1 steps on large errors
Stage 2 (stable, precise):  V6 for N2 steps on reduced errors

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_cascade.py \
        --config1 configs/refiner_oh_v5.yaml \
        --ckpt1 output/refiner_oh_v5/checkpoints/latest.pth \
        --n1 3 \
        --config2 configs/refiner_oh_v6.yaml \
        --ckpt2 output/refiner_oh_v6/checkpoints/best.pth \
        --n2 5 \
        --noise_deg 8 5 3
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
        use_flow_scale_head=mc.get('use_flow_scale_head', False),
        raw_coarse_corr=mc.get('raw_coarse_corr', False),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    # Filter out size-mismatched keys
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  Loaded {checkpoint} (epoch={epoch}, {len(filtered)}/{len(model_state)} keys)")
    return model


def load_renderer(config, device):
    rc = config['renderer']
    return RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        tuple(rc.get('img_hw', [1080, 1920])),
        rc['fx'], rc['fy'], rc['cx'], rc['cy'])


@torch.no_grad()
def evaluate_cascade(cfg1, ckpt1, n1, cfg2, ckpt2, n2,
                     noise_degs, device='cuda', flow_scale1=1.0, flow_scale2=1.0,
                     flow_scale_schedule2=None):
    feat_hw1 = tuple(cfg1['renderer'].get('feat_hw', [68, 120]))
    feat_hw2 = tuple(cfg2['renderer'].get('feat_hw', [68, 120]))

    print("=== Stage 1 ===")
    model1 = load_model(cfg1, ckpt1, device)
    renderer1 = load_renderer(cfg1, device)
    if flow_scale1 != 1.0:
        model1.flow_scale = flow_scale1
        print(f"  flow_scale1 = {flow_scale1}")

    print("\n=== Stage 2 ===")
    model2 = load_model(cfg2, ckpt2, device)
    # Stage 2 shares same renderer if same ply/features
    rc1, rc2 = cfg1['renderer'], cfg2['renderer']
    if rc1['ply_path'] == rc2['ply_path'] and rc1['feature_model_path'] == rc2['feature_model_path']:
        renderer2 = renderer1
        print("  (sharing renderer with Stage 1)")
    else:
        renderer2 = load_renderer(cfg2, device)
    if flow_scale2 != 1.0:
        model2.flow_scale = flow_scale2
        print(f"  flow_scale2 = {flow_scale2}")
    if flow_scale_schedule2 is not None:
        print(f"  flow_scale_schedule2 = {flow_scale_schedule2}")

    # Load data (use config1's data paths)
    dc = cfg1['data']
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
    print(f"\nTest set: {len(test_indices)} frames")
    total_iters = n1 + n2

    for noise_deg in noise_degs:
        noise_trans = noise_deg / 8.0 * 0.25
        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg}° / {noise_trans:.2f}m  |  Cascade: {n1}+{n2} steps")
        print(f"{'='*60}")

        per_iter_rot = {n: [] for n in range(total_iters + 1)}
        per_iter_trans = {n: [] for n in range(total_iters + 1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
            per_iter_rot[0].append(rot_err)
            per_iter_trans[0].append(pos_err)

            # Stage 1
            for it in range(1, n1 + 1):
                ref_feat = renderer1.render_features(pose_cur, feat_hw1)
                depth = renderer1.render_depth(pose_cur, feat_hw1)
                result = model1(feat.to(device), ref_feat, depth)
                if 'delta_xi' in result:
                    pose_cur = se3_exp(result['delta_xi']) @ pose_cur
                pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
                per_iter_rot[it].append(rot_err)
                per_iter_trans[it].append(pos_err)

            # Stage 2
            for it in range(n1 + 1, total_iters + 1):
                # Apply per-iteration flow_scale schedule if provided
                if flow_scale_schedule2 is not None:
                    s2_iter = it - n1 - 1  # 0-based stage2 iteration index
                    fs = flow_scale_schedule2[min(s2_iter, len(flow_scale_schedule2) - 1)]
                    model2.flow_scale = fs
                ref_feat = renderer2.render_features(pose_cur, feat_hw2)
                depth = renderer2.render_depth(pose_cur, feat_hw2)
                result = model2(feat.to(device), ref_feat, depth)
                if 'delta_xi' in result:
                    pose_cur = se3_exp(result['delta_xi']) @ pose_cur
                pos_err, rot_err = pose_error(pose_cur[0], gt_w2c)
                per_iter_rot[it].append(rot_err)
                per_iter_trans[it].append(pos_err)

        # Report
        init_rot = np.median(per_iter_rot[0])
        init_trans = np.median(per_iter_trans[0])
        final_rot = np.median(per_iter_rot[total_iters])
        final_trans = np.median(per_iter_trans[total_iters])

        iter_parts = []
        for i in range(total_iters + 1):
            marker = ' |' if i == n1 else ''
            iter_parts.append(f'I{i}={np.median(per_iter_rot[i]):.2f}°{marker}')
        iter_detail = ' → '.join(iter_parts)

        print(f"\n  {init_rot:.2f}° → {final_rot:.2f}° "
              f"(Δ{final_rot - init_rot:+.2f}°) | "
              f"{init_trans:.1f}cm → {final_trans:.1f}cm "
              f"(Δ{final_trans - init_trans:+.1f}cm)")
        print(f"  {iter_detail}")

        # Also report stage boundaries
        s1_rot = np.median(per_iter_rot[n1])
        s1_trans = np.median(per_iter_trans[n1])
        print(f"\n  Stage 1 ({n1} steps): {init_rot:.2f}° → {s1_rot:.2f}°")
        print(f"  Stage 2 ({n2} steps): {s1_rot:.2f}° → {final_rot:.2f}°")
        print(f"  Mean rot: {np.mean(per_iter_rot[total_iters]):.2f}°, "
              f"P90 rot: {np.percentile(per_iter_rot[total_iters], 90):.2f}°")

        # Also report single-model baselines for comparison
        print(f"\n  Reference (V5-only {n1+n2} steps): see separate eval")


def main():
    parser = argparse.ArgumentParser(description='Cascade PoseRefiner evaluation')
    parser.add_argument('--config1', required=True, help='Stage 1 config')
    parser.add_argument('--ckpt1', required=True, help='Stage 1 checkpoint')
    parser.add_argument('--n1', type=int, default=3, help='Stage 1 outer iterations')
    parser.add_argument('--config2', required=True, help='Stage 2 config')
    parser.add_argument('--ckpt2', required=True, help='Stage 2 checkpoint')
    parser.add_argument('--n2', type=int, default=5, help='Stage 2 outer iterations')
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8, 5, 3])
    parser.add_argument('--flow_scale1', type=float, default=1.0)
    parser.add_argument('--flow_scale2', type=float, default=1.0)
    parser.add_argument('--flow_scale_schedule2', nargs='+', type=float, default=None,
                        help='Per-iteration flow_scale schedule for stage 2 (overrides --flow_scale2)')
    args = parser.parse_args()

    with open(args.config1) as f:
        cfg1 = yaml.safe_load(f)
    with open(args.config2) as f:
        cfg2 = yaml.safe_load(f)

    evaluate_cascade(cfg1, args.ckpt1, args.n1,
                     cfg2, args.ckpt2, args.n2,
                     args.noise_deg,
                     flow_scale1=args.flow_scale1,
                     flow_scale2=args.flow_scale2,
                     flow_scale_schedule2=args.flow_scale_schedule2)


if __name__ == '__main__':
    main()
