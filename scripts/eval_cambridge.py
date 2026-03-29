#!/usr/bin/env python3
"""Evaluate MSFlowPoseNet on Cambridge Landmarks with official train/test split.

Supports three inference modes:
  1) render:   3DGS feature rendering (standard GSFF pipeline)
  2) warp:     retrieval-based depth-warp from nearest training frame
  3) both:     run both for comparison

Reports median rotation/translation errors per Cambridge Landmarks convention.

Usage:
    # Standard 3DGS rendering evaluation
    python scripts/eval_cambridge.py \
        --config configs/gsff_oldhospital.yaml \
        --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
        --mode render --iters 1 3 5 10

    # Retrieval-warp evaluation
    python scripts/eval_cambridge.py \
        --config configs/gsff_oldhospital.yaml \
        --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
        --mode warp --iters 5 10

    # Both modes for comparison
    python scripts/eval_cambridge.py \
        --config configs/gsff_oldhospital.yaml \
        --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
        --mode both --iters 1 5 10
"""

import sys
import yaml
import torch
import math
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.depth_warp import backward_warp_features
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4, load_poses_c2w, c2w_to_w2c
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_model(cfg, device):
    mc = cfg['model']
    rc = cfg.get('renderer', {})
    intrinsics = {
        'fx': rc.get('fx', 320.0),
        'fy': rc.get('fy', 320.0),
        'cx': rc.get('cx', 319.5),
        'cy': rc.get('cy', 239.5),
    }
    model = MSFlowPoseNet(
        hidden_dim=mc.get('hidden_dim', 128),
        decode_dim=mc.get('decode_dim', 64),
        local_radius=mc.get('local_radius', 4),
        damping=mc.get('damping', 0.001),
        coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
        mid_hw=tuple(mc.get('mid_hw', [15, 20])),
        fine_hw=tuple(mc.get('fine_hw', [35, 46])),
        fine_iters=mc.get('fine_iters', 4),
        mid_iters=mc.get('mid_iters', 1),
        corr_temperature=mc.get('corr_temperature', 1.0),
        intrinsics=intrinsics,
        img_hw=(rc.get('img_height', 480), rc.get('img_width', 640)),
        coarse_in_dim=mc.get('coarse_in_dim', 512),
        mid_in_dim=mc.get('mid_in_dim', 512),
        fine_sd_in_dim=mc.get('fine_sd_in_dim', 512),
        fine_dino_in_dim=mc.get('fine_dino_in_dim', 768),
        irls_iters=mc.get('irls_iters', 0),
        irls_huber_k=mc.get('irls_huber_k', 1.345),
        deep_flow_head=mc.get('deep_flow_head', False),
        cross_scale_context=mc.get('cross_scale_context', False),
        cross_scale_dim=mc.get('cross_scale_dim', 32),
        pose_refinement=mc.get('pose_refinement', False),
        corr_dilations=tuple(mc['corr_dilations']) if mc.get('corr_dilations') else None,
        geometry_upsample=mc.get('geometry_upsample', 1),
        multiscale_consistency=mc.get('multiscale_consistency', False),
        ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
        pixel_stride=mc.get('pixel_stride', 1),
        adaptive_damping=mc.get('adaptive_damping', False),
        adaptive_damping_max=mc.get('adaptive_damping_max', 0.1),
        adaptive_damping_cond_thresh=mc.get('adaptive_damping_cond_thresh', 1e4),
        positional_encoding=mc.get('positional_encoding', False),
        pe_mode=mc.get('pe_mode', 'concat'),
        pe_dim=mc.get('pe_dim', 32),
        depth_pe_dim=mc.get('depth_pe_dim', 0),
        skip_coarse_flow=mc.get('skip_coarse_flow', False),
        learnable_temperature=mc.get('learnable_temperature', False),
        directional_confidence=mc.get('directional_confidence', False),
        dino_all_scales=mc.get('dino_all_scales', False),
        dino_replace_sd=mc.get('dino_replace_sd', False),
        localizability_prior=mc.get('localizability_prior', False),
        standardize_before_corr=mc.get('standardize_before_corr', False),
    ).to(device)
    return model


def build_renderer(cfg, device):
    rc = cfg['renderer']
    return MultiScaleRenderer(
        ply_path=rc['ply_path'],
        scale_model_paths=rc['scale_model_paths'],
        device=device,
        img_height=rc.get('img_height', 480),
        img_width=rc.get('img_width', 640),
        fx=rc.get('fx', 320.0),
        fy=rc.get('fy', 320.0),
        cx=rc.get('cx', 319.5),
        cy=rc.get('cy', 239.5),
        sharpen_strength=rc.get('sharpen_strength', 0.0),
        sharpen_kernel_size=rc.get('sharpen_kernel_size', 3),
    )


def build_test_loader(cfg):
    """Build test DataLoader using official Cambridge split."""
    dc = cfg['data']
    feature_dir = dc['train_feature_dir']
    traj_path = dc['train_traj_path']

    # Load official test indices
    test_idx_path = Path(feature_dir) / 'test_indices.npy'
    if test_idx_path.exists():
        test_indices = np.load(str(test_idx_path)).tolist()
        print(f"[Cambridge] Loaded {len(test_indices)} official test indices")
    else:
        raise FileNotFoundError(
            f"Official test indices not found: {test_idx_path}\n"
            f"Generate them from dataset_test.txt first.")

    test_ds = PoseDatasetV4(
        feature_base_dir=feature_dir,
        traj_path=traj_path,
        frame_indices=test_indices,
        noise_rot_deg=dc.get('val_noise_rot_deg', dc.get('noise_rot_deg', 10.0)),
        noise_trans_m=dc.get('val_noise_trans_m', dc.get('noise_trans_m', 1.0)),
        is_train=False,  # use perturbed GT (no NetVLAD)
    )

    return DataLoader(
        test_ds,
        batch_size=dc.get('val_batch_size', 2),
        shuffle=False,
        num_workers=dc.get('num_workers', 4),
        collate_fn=collate_v4,
        pin_memory=True,
    )


class TrainingFeatureIndex:
    """Nearest-neighbor retrieval from training features for depth-warp inference."""

    def __init__(self, feature_dir, traj_path, device='cpu'):
        train_idx_path = Path(feature_dir) / 'train_indices.npy'
        if train_idx_path.exists():
            train_indices = np.load(str(train_idx_path)).tolist()
        else:
            raise FileNotFoundError(f"Training indices not found: {train_idx_path}")

        # Load training features
        ds = PoseDatasetV4(
            feature_base_dir=feature_dir,
            traj_path=traj_path,
            frame_indices=train_indices,
            noise_rot_deg=0.0,
            noise_trans_m=0.0,
            is_train=False,
        )

        # Cache w2c poses and world positions
        self.poses_w2c = torch.from_numpy(ds.poses_w2c).float()
        poses_c2w = load_poses_c2w(traj_path)
        poses_c2w = poses_c2w[train_indices]
        self.positions_world = torch.from_numpy(
            poses_c2w[:, :3, 3].astype(np.float32))

        # Cache all training features in CPU memory
        print(f"[RetrievalIndex] Caching {len(ds)} training features...")
        self.features = []
        for i in tqdm(range(len(ds)), desc="Loading train features", leave=False):
            item = ds[i]
            self.features.append({k: v for k, v in item['query_feats'].items()})
        print(f"[RetrievalIndex] Cached {len(self.features)} frames")

    def find_nearest(self, pose_w2c, k=1):
        """Find k nearest training frames by camera position distance."""
        R = pose_w2c[:, :3, :3]
        t = pose_w2c[:, :3, 3]
        cam_pos = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
        dists = torch.cdist(cam_pos.cpu(), self.positions_world)
        _, indices = dists.topk(k, largest=False)
        return indices

    def get_nearest_batch(self, pose_w2c, device):
        """Get features + poses for nearest training frame per batch element."""
        indices = self.find_nearest(pose_w2c, k=1)
        B = pose_w2c.shape[0]
        ref_feats_list = {scale: [] for scale in self.features[0].keys()}
        ref_poses = []
        for b in range(B):
            idx = indices[b, 0].item()
            feats = self.features[idx]
            for scale in feats:
                ref_feats_list[scale].append(feats[scale])
            ref_poses.append(self.poses_w2c[idx])
        ref_feats = {
            scale: torch.stack(ref_feats_list[scale]).to(device)
            for scale in ref_feats_list
        }
        return ref_feats, torch.stack(ref_poses).to(device)


def compute_pose_errors(pose_pred, pose_gt):
    """Compute rotation (deg) and translation (m) errors."""
    R_rel = torch.bmm(
        pose_pred[:, :3, :3].transpose(1, 2),
        pose_gt.float()[:, :3, :3])
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    rot_err_deg = torch.acos(cos_angle) * 180.0 / math.pi
    trans_err_m = torch.norm(
        pose_pred[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1)
    return rot_err_deg, trans_err_m


def evaluate_render(model, renderer, test_loader, num_iters, device):
    """Evaluate with 3DGS feature rendering (standard GSFF pipeline)."""
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"render iters={num_iters}", leave=False):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)

            for oi in range(num_iters):
                res = renderer.render_batch(
                    pose_cur,
                    scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                    return_depth=True,
                )
                rf = {
                    'coarse': res['coarse_feat'],
                    'mid': res['mid_feat'],
                    'fine_sd': res['fine_sd_feat'],
                    'fine_dino': res['fine_dino_feat'],
                }
                depth = res.get('depth_map')
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, rf, depth)
                if 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            rot_err, trans_err = compute_pose_errors(pose_cur, pose_gt)
            all_rot.extend(rot_err.cpu().tolist())
            all_trans.extend(trans_err.cpu().tolist())

    return _compute_metrics(all_rot, all_trans)


def evaluate_warp(model, renderer, test_loader, feat_index,
                  scale_intrinsics, num_iters, device):
    """Evaluate with retrieval-based depth-warp (test-time valid)."""
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"warp iters={num_iters}", leave=False):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)

            for oi in range(num_iters):
                ref_feats, ref_poses = feat_index.get_nearest_batch(
                    pose_cur, device)
                depth = renderer.render_depth_batch(pose_cur.float())
                warped = backward_warp_features(
                    ref_feats, depth,
                    ref_poses.float(), pose_cur.float(),
                    scale_intrinsics)
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, warped, depth)
                if 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            rot_err, trans_err = compute_pose_errors(pose_cur, pose_gt)
            all_rot.extend(rot_err.cpu().tolist())
            all_trans.extend(trans_err.cpu().tolist())

    return _compute_metrics(all_rot, all_trans)


def _compute_metrics(all_rot, all_trans):
    """Compute Cambridge Landmarks standard metrics."""
    r = np.array(all_rot)
    t = np.array(all_trans)  # already in meters
    return {
        'rot_median_deg': float(np.median(r)),
        'rot_mean_deg': float(np.mean(r)),
        'trans_median_m': float(np.median(t)),
        'trans_mean_m': float(np.mean(t)),
        'pct_lt_1deg': float(np.mean(r < 1.0) * 100),
        'pct_lt_5deg': float(np.mean(r < 5.0) * 100),
        'pct_lt_025m': float(np.mean(t < 0.25) * 100),
        'pct_lt_050m': float(np.mean(t < 0.50) * 100),
        'joint_1deg_025m': float(np.mean((r < 1.0) & (t < 0.25)) * 100),
        'joint_5deg_050m': float(np.mean((r < 5.0) & (t < 0.50)) * 100),
        'num_samples': len(r),
    }


def print_metrics(metrics, label):
    """Print metrics in Cambridge Landmarks format."""
    print(f"  {label}:")
    print(f"    Median: {metrics['trans_median_m']:.3f}m / "
          f"{metrics['rot_median_deg']:.2f}°")
    print(f"    Mean:   {metrics['trans_mean_m']:.3f}m / "
          f"{metrics['rot_mean_deg']:.2f}°")
    print(f"    <1°: {metrics['pct_lt_1deg']:.1f}%  "
          f"<5°: {metrics['pct_lt_5deg']:.1f}%  "
          f"<0.25m: {metrics['pct_lt_025m']:.1f}%  "
          f"<0.50m: {metrics['pct_lt_050m']:.1f}%")
    print(f"    Joint 1°/0.25m: {metrics['joint_1deg_025m']:.1f}%  "
          f"Joint 5°/0.50m: {metrics['joint_5deg_050m']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate on Cambridge Landmarks (official test split)')
    parser.add_argument('--config', required=True, help='Config YAML')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--mode', choices=['render', 'warp', 'both'],
                        default='both', help='Inference mode')
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='Outer iteration counts to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Save JSON results to this path')
    parser.add_argument('--noise-rot', type=float, default=None,
                        help='Override test noise rotation (degrees)')
    parser.add_argument('--noise-trans', type=float, default=None,
                        help='Override test noise translation (meters)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override test noise if specified
    if args.noise_rot is not None:
        cfg['data']['val_noise_rot_deg'] = args.noise_rot
        cfg['data']['noise_rot_deg'] = args.noise_rot
    if args.noise_trans is not None:
        cfg['data']['val_noise_trans_m'] = args.noise_trans
        cfg['data']['noise_trans_m'] = args.noise_trans

    device = torch.device('cuda')
    print(f"[Config] {args.config}")
    print(f"[Noise]  rot={cfg['data'].get('noise_rot_deg', 10.0)}°, "
          f"trans={cfg['data'].get('noise_trans_m', 1.0)}m")

    # Build components
    renderer = build_renderer(cfg, device)
    model = build_model(cfg, device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    epoch = ckpt.get('epoch', '?')
    print(f"[Checkpoint] {args.checkpoint} (epoch={epoch})")

    # Build test loader with official Cambridge split
    test_loader = build_test_loader(cfg)

    # Build retrieval index if needed
    feat_index = None
    scale_intrinsics = None
    if args.mode in ('warp', 'both'):
        dc = cfg['data']
        feat_index = TrainingFeatureIndex(
            dc['train_feature_dir'], dc['train_traj_path'], device='cpu')
        scale_intrinsics = renderer.get_scale_intrinsics()

    # Evaluate
    all_results = {}
    print(f"\n{'='*70}")
    print(f"  Cambridge Landmarks Evaluation — OldHospital")
    print(f"  Checkpoint: {Path(args.checkpoint).stem} (epoch={epoch})")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"{'='*70}")

    for num_iters in args.iters:
        print(f"\n--- Outer iterations: {num_iters} ---")
        iter_results = {}

        if args.mode in ('render', 'both'):
            metrics = evaluate_render(
                model, renderer, test_loader, num_iters, device)
            print_metrics(metrics, f"3DGS Render (iters={num_iters})")
            iter_results['render'] = metrics

        if args.mode in ('warp', 'both'):
            metrics = evaluate_warp(
                model, renderer, test_loader, feat_index,
                scale_intrinsics, num_iters, device)
            print_metrics(metrics, f"Retrieval Warp (iters={num_iters})")
            iter_results['warp'] = metrics

        all_results[f"iters_{num_iters}"] = iter_results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY (Median translation / rotation)")
    print(f"{'='*70}")
    print(f"  {'Iters':<8} ", end='')
    if args.mode in ('render', 'both'):
        print(f"{'3DGS Render':<25} ", end='')
    if args.mode in ('warp', 'both'):
        print(f"{'Retrieval Warp':<25}", end='')
    print()
    for num_iters in args.iters:
        key = f"iters_{num_iters}"
        print(f"  {num_iters:<8} ", end='')
        if args.mode in ('render', 'both') and 'render' in all_results[key]:
            m = all_results[key]['render']
            print(f"{m['trans_median_m']:.3f}m / {m['rot_median_deg']:.2f}°      ", end='')
        if args.mode in ('warp', 'both') and 'warp' in all_results[key]:
            m = all_results[key]['warp']
            print(f"{m['trans_median_m']:.3f}m / {m['rot_median_deg']:.2f}°", end='')
        print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
