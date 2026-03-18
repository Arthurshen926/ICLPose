#!/usr/bin/env python3
"""
Quick eval: sweep step_scale for pose update.

Tests whether scaling down delta_xi before se3_exp improves accuracy.
If step_scale < 1 helps, the model is over-correcting.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_step_scale.py \
        --config configs/exp142_oh_dino_all_scales.yaml \
        --checkpoint output/exp142_oh_dino_all_scales/checkpoints/best.pth \
        --scales 0.1 0.2 0.3 0.5 0.7 1.0 \
        --noise_deg 3.0 --noise_m 0.3 --outer_iters 5
"""
import argparse, yaml, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scales', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    p.add_argument('--noise_deg', type=float, default=None)
    p.add_argument('--noise_m', type=float, default=None)
    p.add_argument('--outer_iters', type=int, default=5)
    p.add_argument('--damping_sweep', type=float, nargs='+', default=None,
                   help='Sweep damping values instead of step_scale')
    return p.parse_args()


def build_model_and_data(cfg, ckpt_path, device, noise_deg=None, noise_m=None):
    mc = cfg['model']
    rc = cfg['renderer']
    dc = cfg['data']

    # Build renderer
    renderer = MultiScaleRenderer(
        ply_path=rc['ply_path'],
        scale_model_paths=rc['scale_model_paths'],
        device=device,
        img_height=rc.get('img_height', 480),
        img_width=rc.get('img_width', 640),
        fx=rc.get('fx', 320.0),
        fy=rc.get('fy', 320.0),
        cx=rc.get('cx', 319.5),
        cy=rc.get('cy', 239.5),
    )

    # Build model
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
        damping=mc.get('damping', 1e-3),
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
        dino_all_scales=mc.get('dino_all_scales', False),
        attention_coarse=mc.get('attention_coarse', False),
        attention_coarse_heads=mc.get('attention_coarse_heads', 4),
        attention_coarse_layers=mc.get('attention_coarse_layers', 2),
    )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Dataset
    nd = noise_deg if noise_deg is not None else dc.get('noise_rot_deg', 10.0)
    nm = noise_m if noise_m is not None else dc.get('noise_trans_m', 1.0)
    
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc.get('train_traj_path'),
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=nd,
        noise_trans_m=nm,
        is_train=True,  # Use is_train=True so noise_rot_deg is respected
    )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * dc.get('val_split_ratio', 0.1)))
    n_train = n_total - n_val
    _, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=dc.get('val_batch_size', 4),
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=collate_v4)

    return model, renderer, loader


@torch.no_grad()
def cache_batches(loader, device):
    """Cache all batches with deterministic noise."""
    batches = []
    for batch in loader:
        cached = {
            'query_feats': {k: v.to(device) for k, v in batch['query_feats'].items()},
            'pose_gt': batch['pose_gt'].to(device).float(),
            'initial_pose': batch['initial_pose'].to(device).float(),
        }
        batches.append(cached)
    return batches


@torch.no_grad()
def evaluate(model, renderer, batches, device, outer_iters, step_scale=1.0, use_amp=True):
    all_rot, all_trans = [], []
    
    for batch in tqdm(batches, desc=f"scale={step_scale:.2f}", leave=False):
        query_feats = batch['query_feats']
        pose_gt = batch['pose_gt']
        pose_cur = batch['initial_pose'].clone()

        for oi in range(outer_iters):
            with torch.cuda.amp.autocast(enabled=use_amp):
                scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
                render_out = renderer.render_batch(pose_cur, scales=scales, return_depth=True)
                render_feats = {
                    'coarse': render_out['coarse_feat'],
                    'mid': render_out['mid_feat'],
                    'fine_sd': render_out['fine_sd_feat'],
                    'fine_dino': render_out['fine_dino_feat'],
                }
                depth = render_out.get('depth_map')
                pred = model(query_feats, render_feats, depth)

            if 'delta_xi' in pred:
                # Apply step_scale
                xi = pred['delta_xi'].float() * step_scale
                T_delta = se3_exp(xi)
                pose_cur = torch.bmm(T_delta, pose_cur)

        # Final metrics
        R_pred = pose_cur[:, :3, :3]
        R_gt = pose_gt[:, :3, :3]
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        rot_err = torch.acos(cos_angle) * 180.0 / math.pi
        
        t_pred = pose_cur[:, :3, 3]
        t_gt = pose_gt[:, :3, 3]
        trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000

        all_rot.extend(rot_err.cpu().tolist())
        all_trans.extend(trans_err.cpu().tolist())

    rot = np.array(all_rot)
    trans = np.array(all_trans)
    return {
        'rot_mean': float(np.mean(rot)),
        'rot_median': float(np.median(rot)),
        'trans_mean': float(np.mean(trans)),
        'trans_median': float(np.median(trans)),
        'pct_lt1': float(np.mean(rot < 1.0) * 100),
        'pct_lt5': float(np.mean(rot < 5.0) * 100),
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, renderer, loader = build_model_and_data(
        cfg, args.checkpoint, device, args.noise_deg, args.noise_m)
    
    # Cache all batches so every step_scale sees the same initial poses
    print("Caching val batches...")
    torch.manual_seed(12345)
    np.random.seed(12345)
    batches = cache_batches(loader, device)
    
    print(f"\nConfig: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Noise: {args.noise_deg}° / {args.noise_m}m")
    print(f"Outer iters: {args.outer_iters}")
    n_samples = sum(b['pose_gt'].shape[0] for b in batches)
    print(f"Val samples: {n_samples}")
    print(f"{'='*70}")
    print(f"{'step_scale':>10} {'rot_mean':>10} {'rot_med':>10} {'trans_mean':>12} {'<1°':>6} {'<5°':>6}")
    print(f"{'='*70}")

    for scale in args.scales:
        m = evaluate(model, renderer, batches, device,
                     args.outer_iters, step_scale=scale)
        print(f"{scale:>10.2f} {m['rot_mean']:>10.2f}° {m['rot_median']:>10.2f}° "
              f"{m['trans_mean']:>12.1f}mm {m['pct_lt1']:>5.1f}% {m['pct_lt5']:>5.1f}%")

    # Also test step_scale=0 (no update = raw noise)
    m0 = evaluate(model, renderer, batches, device,
                  args.outer_iters, step_scale=0.0)
    print(f"{'0.00':>10} {m0['rot_mean']:>10.2f}° {m0['rot_median']:>10.2f}° "
          f"{m0['trans_mean']:>12.1f}mm {m0['pct_lt1']:>5.1f}% {m0['pct_lt5']:>5.1f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
