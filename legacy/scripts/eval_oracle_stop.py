"""
eval_oracle_stop.py - Oracle stopping evaluation

For each test frame, run N iterations and pick the iteration that gives
the lowest rotation error. This tells us the theoretical best achievable
with a perfect stopping criterion.

Also reports per-iteration improvement distribution stats.
"""
import sys, math
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


def pose_error(T_pred, T_gt):
    """Compute position and rotation error."""
    R_pred = T_pred[:3, :3]
    R_gt = T_gt[:3, :3]
    t_pred = T_pred[:3, 3]
    t_gt = T_gt[:3, 3]

    pos_err = (t_pred - t_gt).norm().item() * 100  # cm
    cos_val = ((R_pred @ R_gt.T).trace() - 1) / 2
    cos_val = cos_val.clamp(-1, 1)
    rot_err = torch.acos(cos_val).item() * 180 / np.pi
    return pos_err, rot_err


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', nargs='+', type=float, default=[8.0, 5.0, 3.0])
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--flow_scale', type=float, default=None)
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

    if args.flow_scale is not None:
        model.flow_scale = args.flow_scale

    # Load renderer (inline, same as eval_refiner.py)
    rc = cfg['renderer']
    
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
    
    renderer = RadioRenderer(
        ply_path=rc['ply_path'],
        feature_model_path=rc['feature_model_path'],
        device=device,
        img_hw=tuple(rc['img_hw']),
        fx=rc['fx'], fy=rc['fy'], cx=rc['cx'], cy=rc['cy']
    )
    feat_hw = tuple(rc['feat_hw'])

    # Load data (same logic as eval_refiner.py)
    import re, os
    dc = cfg['data']
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

    for noise_deg in args.noise_deg:
        noise_trans = noise_deg / 8.0 * 0.25

        print(f"\n{'='*60}")
        print(f"  Noise: {noise_deg:.1f}° / {noise_trans:.2f}m")
        print(f"{'='*60}")

        oracle_rot_errors = []
        oracle_stop_iters = []
        per_iter_improvements = {i: [] for i in range(1, args.max_iters + 1)}
        per_iter_rot = {i: [] for i in range(args.max_iters + 1)}
        # Track delta_xi magnitude per iteration for stop criterion analysis
        per_iter_xi_rot_mag = {i: [] for i in range(1, args.max_iters + 1)}
        per_iter_improved = {i: [] for i in range(1, args.max_iters + 1)}
        per_iter_conf_mean = {i: [] for i in range(1, args.max_iters + 1)}

        for frame_idx in test_indices:
            feat = torch.load(str(features[frame_idx]),
                              map_location=device, weights_only=True).float().unsqueeze(0)
            c2w = torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c = torch.linalg.inv(c2w).to(device)

            seed = frame_idx * 1000 + int(noise_deg * 100)
            noisy_w2c = add_noise_deterministic(gt_w2c, noise_deg, noise_trans, seed)
            pose_cur = noisy_w2c.unsqueeze(0)

            _, init_rot = pose_error(pose_cur[0], gt_w2c)
            
            best_rot = init_rot
            best_iter = 0
            all_rots = [init_rot]

            with torch.no_grad():
                for it in range(1, args.max_iters + 1):
                    ref_feat = renderer.render_features(pose_cur, feat_hw)
                    depth = renderer.render_depth(pose_cur, feat_hw)
                    result = model(feat.to(device), ref_feat, depth)
                    
                    # Track delta_xi rotation magnitude (last 3 = rotation part)
                    if 'delta_xi' in result:
                        xi = result['delta_xi'][0]  # [6]
                        rot_mag = xi[3:].norm().item() * 180 / np.pi
                        per_iter_xi_rot_mag[it].append(rot_mag)
                        pose_cur = se3_exp(result['delta_xi']) @ pose_cur
                    
                    # Track confidence
                    if 'conf_fine' in result:
                        conf = result['conf_fine'].mean().item()
                        per_iter_conf_mean[it].append(conf)

                    _, rot_err = pose_error(pose_cur[0], gt_w2c)
                    all_rots.append(rot_err)
                    improved = rot_err < all_rots[-2]
                    per_iter_improved[it].append(improved)
                    
                    if rot_err < best_rot:
                        best_rot = rot_err
                        best_iter = it

                    # Track per-iter improvement
                    improvement = all_rots[it-1] - rot_err
                    per_iter_improvements[it].append(improvement)

            oracle_rot_errors.append(best_rot)
            oracle_stop_iters.append(best_iter)
            
            for i in range(args.max_iters + 1):
                if i < len(all_rots):
                    per_iter_rot[i].append(all_rots[i])

        oracle_med = np.median(oracle_rot_errors)
        init_med = np.median(per_iter_rot[0])
        
        print(f"\n  Oracle (best stop per frame):")
        print(f"    Init: {init_med:.2f}° → Oracle: {oracle_med:.2f}° "
              f"({(1 - oracle_med/init_med)*100:.1f}% improvement)")
        
        # Stop iteration distribution
        counts = np.bincount(oracle_stop_iters, minlength=args.max_iters+1)
        print(f"    Stop distribution: " + 
              " | ".join(f"I{i}:{counts[i]}" for i in range(args.max_iters+1) if counts[i] > 0))
        
        # Standard (no oracle) results for comparison
        print(f"\n  Standard (all frames same N iters):")
        for n in [1, 2, 3, 5, 7]:
            if n <= args.max_iters:
                med = np.median(per_iter_rot[n])
                print(f"    N={n}: {med:.2f}°")
        
        # Per-iteration improvement stats
        print(f"\n  Per-iter improvement (mean ± std):")
        for it in range(1, min(args.max_iters + 1, 8)):
            imp = per_iter_improvements[it]
            if imp:
                mean_imp = np.mean(imp)
                std_imp = np.std(imp)
                pct_improve = np.mean([x > 0 for x in imp]) * 100
                print(f"    Iter {it}: {mean_imp:+.3f}° ± {std_imp:.3f}° "
                      f"({pct_improve:.0f}% of frames improve)")

        # Delta_xi rotation magnitude analysis (key for stop criterion)
        print(f"\n  Delta_xi rot magnitude vs improvement:")
        for it in range(1, min(args.max_iters + 1, 6)):
            xi_mags = per_iter_xi_rot_mag.get(it, [])
            improved = per_iter_improved.get(it, [])
            if xi_mags and improved:
                xi_mags = np.array(xi_mags)
                improved = np.array(improved)
                good_mag = xi_mags[improved].mean() if improved.any() else 0
                bad_mag = xi_mags[~improved].mean() if (~improved).any() else 0
                print(f"    Iter {it}: mean_xi_rot={xi_mags.mean():.3f}° | "
                      f"improved={good_mag:.3f}° | worsened={bad_mag:.3f}°")
        
        # Confidence analysis
        print(f"\n  Confidence per iteration:")
        for it in range(1, min(args.max_iters + 1, 6)):
            confs = per_iter_conf_mean.get(it, [])
            improved = per_iter_improved.get(it, [])
            if confs and improved:
                confs = np.array(confs)
                improved = np.array(improved)
                good_conf = confs[improved].mean() if improved.any() else 0
                bad_conf = confs[~improved].mean() if (~improved).any() else 0
                print(f"    Iter {it}: mean_conf={confs.mean():.3f} | "
                      f"improved={good_conf:.3f} | worsened={bad_conf:.3f}")


if __name__ == '__main__':
    main()
