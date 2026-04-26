"""
eval_xi_stop.py - Evaluation with delta_xi-based early stopping

Stops iterating when the predicted correction magnitude exceeds a 
threshold or when the ratio between consecutive corrections indicates
the model is no longer converging.

Stop criteria:
1. xi_threshold: Stop if |delta_xi_rot| > threshold (prevent overshoot)
2. xi_ratio: Stop if |delta_xi_rot[k]| / |delta_xi_rot[k-1]| > ratio (diverging)
3. xi_floor: Stop if |delta_xi_rot| < floor (already converged)
"""
import sys, math, re, os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
from pathlib import Path

from modules.lie_algebra import se3_exp
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


class RadioRenderer:
    def __init__(self, ply_path, feature_model_path, device,
                 img_hw=(1080, 1920), fx=1663.12, fy=1663.12, cx=960.0, cy=540.0):
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
        if viewmat.dim() == 2: viewmat = viewmat.unsqueeze(0)
        fH, fW = feat_hw
        result = FeatureRenderer.render_features_batch(
            self.gs_model, viewmat,
            fx=self.fx * fW / self.img_hw[1], fy=self.fy * fH / self.img_hw[0],
            cx=self.cx * fW / self.img_hw[1], cy=self.cy * fH / self.img_hw[0],
            img_height=fH, img_width=fW,
            norm_feat_before_render=True, norm_feat_after_render=True)
        return result['feature_map']

    def render_depth(self, viewmat, depth_hw=(68, 120)):
        if viewmat.dim() == 2: viewmat = viewmat.unsqueeze(0)
        dH, dW = depth_hw
        depths = []
        for i in range(viewmat.shape[0]):
            d = FeatureRenderer.render_depth(
                self.gs_model, viewmat[i],
                fx=self.fx * dW / self.img_hw[1], fy=self.fy * dH / self.img_hw[0],
                cx=self.cx * dW / self.img_hw[1], cy=self.cy * dH / self.img_hw[0],
                img_height=dH, img_width=dW)
            depths.append(d)
        return torch.stack(depths)


def add_noise_deterministic(pose_w2c, noise_deg, noise_trans, seed):
    device = pose_w2c.device
    rng = torch.Generator()
    rng.manual_seed(seed)
    axis = torch.randn(3, generator=rng)
    axis = axis / (axis.norm() + 1e-8)
    angle = noise_deg * math.pi / 180.0
    omega = axis * angle
    direction = torch.randn(3, generator=rng)
    direction = direction / (direction.norm() + 1e-8)
    trans = direction * noise_trans
    xi = torch.cat([trans, omega]).to(device)
    delta_T = se3_exp(xi)
    return delta_T @ pose_w2c


def pose_error(T_pred, T_gt):
    R_pred = T_pred[:3, :3]
    R_gt = T_gt[:3, :3]
    t_pred = T_pred[:3, 3]
    t_gt = T_gt[:3, 3]
    pos_err = (t_pred - t_gt).norm().item() * 100
    cos_val = ((R_pred @ R_gt.T).trace() - 1) / 2
    cos_val = cos_val.clamp(-1, 1)
    rot_err = torch.acos(cos_val).item() * 180 / np.pi
    return pos_err, rot_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8.0, 5.0, 3.0])
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--xi_threshold', type=float, default=None,
                        help='Stop if xi_rot_mag exceeds this (degrees)')
    parser.add_argument('--xi_ratio', type=float, default=None,
                        help='Stop if xi_rot_mag[k]/xi_rot_mag[k-1] > ratio')
    parser.add_argument('--xi_floor', type=float, default=None,
                        help='Stop if xi_rot_mag < floor (already converged)')
    parser.add_argument('--min_iters', type=int, default=1,
                        help='Always run at least this many iterations')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda')

    # Load model
    from ic_models.pose_refiner import PoseRefiner
    mc = cfg['model']
    model = PoseRefiner(
        in_dim=mc['in_dim'], match_dim=mc['match_dim'],
        hidden_dim=mc['hidden_dim'], n_heads=mc['n_heads'],
        n_attn_layers=mc['n_attn_layers'], ffn_dim=mc['ffn_dim'],
        local_radius=mc.get('local_radius', 4),
        fine_iters=mc.get('fine_iters', 8),
        damping=mc.get('damping', 0.001),
        coarse_hw=mc.get('coarse_hw', [17, 30]),
        fine_hw=mc.get('fine_hw', [68, 120]),
        solver_hw=mc.get('solver_hw', [68, 120]),
        solver_upsample=mc.get('solver_upsample', 4),
        depth_normalize=mc.get('depth_normalize', False),
        sequential_solve=mc.get('sequential_solve', True),
        detach_conf=mc.get('detach_conf', True),
        conf_floor=mc.get('conf_floor', 0.1),
        solver_trans_scale=mc.get('solver_trans_scale', 0.0),
        use_trans_head=mc.get('use_trans_head', False),
        trans_head_mode=mc.get('trans_head_mode', 'replace'),
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}")

    # Load renderer
    rc = cfg['renderer']
    renderer = RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        tuple(rc.get('img_hw', [1080, 1920])),
        rc['fx'], rc['fy'], rc['cx'], rc['cy'])
    feat_hw = tuple(rc['feat_hw'])

    # Load data
    dc = cfg['data']
    feature_dir = Path(dc['feature_dir']) / 'fine_radio'
    traj_path = dc['traj_path']
    pattern_re = re.compile(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt')
    features = {}
    for f in sorted(feature_dir.iterdir()):
        m = pattern_re.match(f.name)
        if m:
            features[int(m.group(1))] = f

    poses_c2w = []
    with open(traj_path) as f_traj:
        for line in f_traj:
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
    print(f"Stop criteria: threshold={args.xi_threshold}, ratio={args.xi_ratio}, "
          f"floor={args.xi_floor}, min_iters={args.min_iters}")

    for noise_deg in args.noise_deg:
        noise_trans = noise_deg / 8.0 * 0.25

        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg:.1f}° / {noise_trans:.2f}m")
        print(f"{'='*60}")

        # Results with stopping criterion
        stopped_rot_errors = []
        stopped_iters = []
        # Also track fixed-N for comparison
        fixed_n_rot = {n: [] for n in range(args.max_iters + 1)}

        for frame_idx in test_indices:
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            _, init_rot = pose_error(pose_cur[0], gt_w2c)
            fixed_n_rot[0].append(init_rot)

            stopped = False
            stop_pose = pose_cur.clone()
            stop_iter = 0
            prev_xi_mag = None

            with torch.no_grad():
                for it in range(1, args.max_iters + 1):
                    ref_feat = renderer.render_features(pose_cur, feat_hw)
                    depth = renderer.render_depth(pose_cur, feat_hw)
                    result = model(feat.to(device), ref_feat, depth)

                    if 'delta_xi' in result:
                        xi = result['delta_xi'][0]
                        xi_rot_mag = xi[3:].norm().item() * 180 / np.pi

                        # Check stop criteria (only after min_iters)
                        if it >= args.min_iters and not stopped:
                            should_stop = False

                            if args.xi_threshold and xi_rot_mag > args.xi_threshold:
                                should_stop = True

                            if args.xi_ratio and prev_xi_mag is not None:
                                if xi_rot_mag / (prev_xi_mag + 1e-8) > args.xi_ratio:
                                    should_stop = True

                            if args.xi_floor and xi_rot_mag < args.xi_floor:
                                should_stop = True

                            if should_stop:
                                stopped = True
                                # Don't apply this update — keep previous pose

                        if not stopped:
                            pose_cur = se3_exp(result['delta_xi']) @ pose_cur
                            stop_pose = pose_cur.clone()
                            stop_iter = it
                        else:
                            # Still update pose_cur for fixed-N tracking
                            pose_cur = se3_exp(result['delta_xi']) @ pose_cur

                        prev_xi_mag = xi_rot_mag

                    _, rot_err = pose_error(pose_cur[0], gt_w2c)
                    fixed_n_rot[it].append(rot_err)

            _, stopped_rot = pose_error(stop_pose[0], gt_w2c)
            stopped_rot_errors.append(stopped_rot)
            stopped_iters.append(stop_iter)

        init_med = np.median(fixed_n_rot[0])
        stopped_med = np.median(stopped_rot_errors)

        print(f"\n  With stop criterion:")
        print(f"    Init: {init_med:.2f}° → Stopped: {stopped_med:.2f}° "
              f"({(1 - stopped_med/init_med)*100:.1f}% improvement)")
        counts = np.bincount(stopped_iters, minlength=args.max_iters+1)
        print(f"    Stop distribution: " +
              " | ".join(f"I{i}:{counts[i]}" for i in range(args.max_iters+1) if counts[i] > 0))

        print(f"\n  Fixed-N comparison:")
        for n in [1, 2, 3, 5, 7]:
            if n <= args.max_iters and fixed_n_rot[n]:
                med = np.median(fixed_n_rot[n])
                print(f"    N={n}: {med:.2f}°")


if __name__ == '__main__':
    main()
