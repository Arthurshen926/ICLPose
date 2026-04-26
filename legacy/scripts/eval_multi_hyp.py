#!/usr/bin/env python3
"""Multi-hypothesis evaluation: run N seeds, pick best per sample by residual.

For each sample, picks the seed whose final flow residual is lowest,
effectively simulating a multi-hypothesis strategy at test time.
"""
import os, sys, math, argparse, yaml
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_refine import load_concat_pose_model as load_model
from scene_feature_field import build_dcff, intrinsics_to_K, render_batch
from data.radio_loc_dataset import (
    RadioLocDataset, read_colmap_cameras, camera_params_to_intrinsics
)
from modules.lie_algebra import se3_exp
from utils.project_config import load_mainline_config


def eval_with_residuals(model, gaussians, dcff_renderer, feat_sharp, val_loader, device,
                        outer_iters, render_h, render_w, use_coarse=True):
    """Run eval, return per-sample rot/trans errors AND flow residuals."""
    model.eval()
    render_intr = model._scale_intrinsics(render_h, render_w)
    K = intrinsics_to_K(render_intr, device)
    
    all_rot, all_trans, all_residuals = [], [], []
    
    for batch in tqdm(val_loader, desc=f'Eval oi={outer_iters}', leave=False):
        query_fine = batch['query_fine'].to(device)
        query_coarse = batch.get('query_coarse')
        if query_coarse is not None and use_coarse:
            query_coarse = query_coarse.to(device)
        else:
            query_coarse = None
        pose_gt = batch['pose_gt'].to(device)
        pose_cur = batch['pose_init'].to(device)
        
        for outer_i in range(outer_iters):
            ref_fine, depth = render_batch(
                gaussians, dcff_renderer, feat_sharp,
                pose_cur, K, render_h, render_w,
            )
            with autocast(enabled=True):
                pred = model(query_fine, ref_fine, depth,
                           intrinsics=render_intr,
                           query_coarse=query_coarse)
            
            with torch.cuda.amp.autocast(enabled=False):
                T_delta = se3_exp(pred['delta_xi'].float())
                pose_cur = torch.bmm(T_delta, pose_cur.float())
        
        # Compute flow residual at final pose (render at final pose, check alignment)
        ref_final, depth_final = render_batch(
            gaussians, dcff_renderer, feat_sharp,
            pose_cur, K, render_h, render_w,
        )
        with torch.no_grad():
            # Feature residual: L2 distance between query and rendered features
            residual = (query_fine.float() - ref_final.float()).pow(2).mean(dim=1)  # (B, H, W)
            if depth_final.dim() == 4:
                depth_final = depth_final.squeeze(1)
            valid = depth_final > 0
            # Mean residual over valid pixels
            batch_residuals = []
            for b in range(residual.shape[0]):
                if valid[b].any():
                    batch_residuals.append(residual[b][valid[b]].mean().item())
                else:
                    batch_residuals.append(float('inf'))
        
        # Pose errors
        with torch.cuda.amp.autocast(enabled=False):
            R_pred = pose_cur.float()[:, :3, :3]
            R_gt = pose_gt.float()[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = torch.acos(cos_angle) * 180.0 / math.pi
            t_pred = pose_cur.float()[:, :3, 3]
            t_gt = pose_gt.float()[:, :3, 3]
            trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000
        
        all_rot.extend(rot_err.cpu().tolist())
        all_trans.extend(trans_err.cpu().tolist())
        all_residuals.extend(batch_residuals)
    
    return np.array(all_rot), np.array(all_trans), np.array(all_residuals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--noise_deg', type=float, default=3.0)
    parser.add_argument('--noise_m', type=float, default=0.10)
    parser.add_argument('--num_hyp', type=int, default=5, help='Number of hypotheses')
    parser.add_argument('--outer_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda')
    config = load_mainline_config(args.config)
    
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)
    model, ckpt_epoch = load_model(config, args.checkpoint, device)
    
    dcff_cfg = config['dcff']
    render_h = dcff_cfg.get('render_height', 68)
    render_w = dcff_cfg.get('render_width', 120)
    use_coarse = config.get('model', {}).get('use_coarse', True)
    ds_cfg = config['dataset']
    
    colmap_cameras = read_colmap_cameras(os.path.join(ds_cfg['colmap_dir'], 'cameras.bin'))
    first_cam = next(iter(colmap_cameras.values()))
    model.BASE_INTRINSICS = camera_params_to_intrinsics(first_cam)
    model.IMG_HW = (int(first_cam.height), int(first_cam.width))
    
    fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))
    coarse_hw = tuple(ds_cfg.get('coarse_hw', fine_hw))
    
    print(f'\nMulti-hypothesis eval: {args.num_hyp} hypotheses, {args.noise_deg}°/{args.noise_m}m')
    print(f'Checkpoint: {args.checkpoint} (epoch {ckpt_epoch})')
    print(f'Outer iters: {args.outer_iters}\n')
    
    # Run N seeds, collect per-sample results
    n_samples = None
    all_rots = []      # (num_hyp, n_samples)
    all_trans = []
    all_resids = []
    
    for seed in range(args.num_hyp):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        val_ds = RadioLocDataset(
            feature_dir=ds_cfg['feature_dir'],
            colmap_dir=ds_cfg['colmap_dir'],
            split_file=ds_cfg['test_split'],
            fine_hw=fine_hw, coarse_hw=coarse_hw,
            noise_rot_deg=args.noise_deg,
            noise_trans_m=args.noise_m,
            cache_in_memory=(seed == 0),
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        
        rot, trans, resid = eval_with_residuals(
            model, gaussians, dcff_renderer, feat_sharp, val_loader, device,
            args.outer_iters, render_h, render_w, use_coarse,
        )
        
        if n_samples is None:
            n_samples = len(rot)
        assert len(rot) == n_samples
        
        all_rots.append(rot)
        all_trans.append(trans)
        all_resids.append(resid)
        
        print(f'  [hyp {seed}] rot={np.median(rot):.2f}°  trans={np.median(trans):.1f}mm  '
              f'resid={np.mean(resid):.4f}')
    
    all_rots = np.stack(all_rots)     # (num_hyp, n_samples)
    all_trans = np.stack(all_trans)
    all_resids = np.stack(all_resids)
    
    # Strategy 1: Oracle best (pick best by GT trans error — upper bound)
    oracle_best_idx = np.argmin(all_trans, axis=0)  # per sample
    oracle_rot = all_rots[oracle_best_idx, np.arange(n_samples)]
    oracle_trans = all_trans[oracle_best_idx, np.arange(n_samples)]
    
    # Strategy 2: Pick by lowest residual (no GT needed — practical strategy)
    resid_best_idx = np.argmin(all_resids, axis=0)
    resid_rot = all_rots[resid_best_idx, np.arange(n_samples)]
    resid_trans = all_trans[resid_best_idx, np.arange(n_samples)]
    
    # Strategy 3: Average across all seeds (per-sample)
    avg_rot = np.mean(all_rots, axis=0)
    avg_trans = np.mean(all_trans, axis=0)
    
    # Strategy 4: Median seed (single best seed by median trans)
    seed_medians = [np.median(all_trans[i]) for i in range(args.num_hyp)]
    best_seed = np.argmin(seed_medians)
    
    print(f'\n{"=" * 60}')
    print(f'Results ({args.num_hyp} hypotheses, {n_samples} samples):')
    print(f'{"=" * 60}')
    
    def report(name, rot, trans):
        pct1 = np.mean(rot < 1.0) * 100
        j1_50 = np.mean((rot < 1.0) & (trans < 50.0)) * 100
        j5_100 = np.mean((rot < 5.0) & (trans < 100.0)) * 100
        print(f'  {name:30s}  rot={np.median(rot):6.2f}°  trans={np.median(trans):7.1f}mm  '
              f'<1°={pct1:5.1f}%  j@1/50={j1_50:5.1f}%  j@5/100={j5_100:5.1f}%')
    
    report(f'Best single seed (#{best_seed})', all_rots[best_seed], all_trans[best_seed])
    report('Pick by residual (practical)', resid_rot, resid_trans)
    report('Oracle best (GT-based)', oracle_rot, oracle_trans)
    
    # Show improvement from single seed to multi-hyp
    single_med = np.median(all_trans[best_seed])
    resid_med = np.median(resid_trans)
    oracle_med = np.median(oracle_trans)
    print(f'\n  Improvement over best single seed ({single_med:.1f}mm):')
    print(f'    Residual selection: {single_med - resid_med:+.1f}mm ({(single_med-resid_med)/single_med*100:+.1f}%)')
    print(f'    Oracle selection:   {single_med - oracle_med:+.1f}mm ({(single_med-oracle_med)/single_med*100:+.1f}%)')


if __name__ == '__main__':
    main()
