#!/usr/bin/env python3
"""Diagnostic: GT-flow upper bound for the geometry solver.

Replaces predicted flow with ground-truth flow computed from GT depth + poses,
then feeds it to the WLS solver to determine the theoretical best performance
achievable with perfect flow at the current rendering resolution.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/diag_gt_flow.py \
      --config configs/concat_loc_oh_v19d.yaml --gpu 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve
from modules.lie_algebra import se3_exp
from ic_models.concat_pose_net import ConcatPoseNet

from scene_feature_field import build_dcff, intrinsics_to_K, render_batch
from data.radio_loc_dataset import (
    RadioLocDataset, collate_fn, read_colmap_cameras, camera_params_to_intrinsics,
)
from utils.project_config import load_mainline_config


def rotation_error_deg(T_pred, T_gt):
    R_diff = torch.bmm(T_pred[:, :3, :3], T_gt[:, :3, :3].transpose(1, 2))
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)) * 180.0 / np.pi


def translation_error_mm(T_pred, T_gt):
    return torch.norm(T_pred[:, :3, 3] - T_gt[:, :3, 3], dim=1) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--noise_deg', type=float, default=3.0)
    parser.add_argument('--noise_m', type=float, default=0.10)
    parser.add_argument('--outer_iters', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--flow_noise_px', type=float, nargs='+', default=[0.0],
                        help='Add Gaussian noise (in pixels) to GT flow')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)

    config = load_mainline_config(args.config)

    # Load DCFF pipeline
    print('Building DCFF rendering pipeline...')
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)

    dcff_cfg = config['dcff']
    render_h = dcff_cfg.get('render_height', 68)
    render_w = dcff_cfg.get('render_width', 120)
    ds_cfg = config['dataset']

    # Get intrinsics
    colmap_dir = ds_cfg['colmap_dir']
    colmap_cameras = read_colmap_cameras(os.path.join(colmap_dir, 'cameras.bin'))
    first_cam = next(iter(colmap_cameras.values()))
    base_intr = camera_params_to_intrinsics(first_cam)
    orig_h, orig_w = int(first_cam.height), int(first_cam.width)

    sx = render_w / orig_w
    sy = render_h / orig_h
    render_intr = {
        'fx': base_intr['fx'] * sx, 'fy': base_intr['fy'] * sy,
        'cx': base_intr['cx'] * sx, 'cy': base_intr['cy'] * sy,
    }
    K = intrinsics_to_K(render_intr, device)
    print(f'Render: {render_w}×{render_h}  fx={render_intr["fx"]:.2f}  fy={render_intr["fy"]:.2f}')

    fine_hw = tuple(ds_cfg.get('fine_hw', [render_h, render_w]))

    print(f'\n{"="*72}')
    print(f'GT Flow Diagnostic')
    print(f'Noise: {args.noise_deg}°/{args.noise_m}m  Resolution: {render_h}×{render_w}')
    print(f'Seeds: {args.num_seeds}  Outer iters: {args.outer_iters}')
    print(f'Flow noise (px): {args.flow_noise_px}')
    print(f'{"="*72}\n')

    for flow_noise in args.flow_noise_px:
        print(f'\n--- Flow noise: {flow_noise:.2f} px ---')
        for max_oi in args.outer_iters:
            seed_medians_t, seed_medians_r = [], []

            for seed in range(args.num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                val_ds = RadioLocDataset(
                    feature_dir=ds_cfg['feature_dir'],
                    colmap_dir=ds_cfg['colmap_dir'],
                    split_file=ds_cfg['test_split'],
                    fine_hw=fine_hw,
                    coarse_hw=fine_hw,
                    noise_rot_deg=args.noise_deg,
                    noise_trans_m=args.noise_m,
                    cache_in_memory=(seed == 0),
                )
                val_loader = DataLoader(
                    val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, collate_fn=collate_fn,
                )

                all_t_err, all_r_err = [], []

                for batch in tqdm(val_loader, desc=f'noise={flow_noise} oi={max_oi} s={seed}', leave=False):
                    pose_gt = batch['pose_gt'].to(device).float()
                    pose_cur = batch['pose_init'].to(device).float()
                    B = pose_gt.shape[0]

                    for oi in range(max_oi):
                        _, depth_cur = render_batch(
                            gaussians, dcff_renderer, feat_sharp, pose_cur, K, render_h, render_w,
                        )

                        gt_flow, gt_valid = ConcatPoseNet.compute_gt_flow(
                            pose_cur, pose_gt, depth_cur,
                            target_hw=(render_h, render_w),
                            intrinsics=render_intr,
                        )

                        if flow_noise > 0:
                            noise = torch.randn_like(gt_flow) * flow_noise
                            gt_flow = gt_flow + noise * gt_valid

                        Ju, Jv, valid = compute_image_jacobian(depth_cur.float(), render_intr)
                        valid_flow_flat = gt_valid.squeeze(1).reshape(B, -1) > 0.5
                        valid_combined = valid & valid_flow_flat
                        conf = gt_valid.float()

                        delta_xi = diff_pose_solve(
                            gt_flow.float(), conf, Ju, Jv, valid_combined, damping=1e-3,
                        )
                        T_delta = se3_exp(delta_xi.float())
                        pose_cur = torch.bmm(T_delta, pose_cur)

                    t_err = translation_error_mm(pose_cur, pose_gt)
                    r_err = rotation_error_deg(pose_cur, pose_gt)
                    all_t_err.extend(t_err.cpu().tolist())
                    all_r_err.extend(r_err.cpu().tolist())

                t_med = np.median(all_t_err)
                r_med = np.median(all_r_err)
                seed_medians_t.append(t_med)
                seed_medians_r.append(r_med)

            t_mean = np.mean(seed_medians_t)
            t_std = np.std(seed_medians_t)
            r_mean = np.mean(seed_medians_r)
            r_std = np.std(seed_medians_r)
            print(f'  oi={max_oi:2d}:  trans={t_mean:6.1f}mm ±{t_std:4.1f}  '
                  f'rot={r_mean:5.3f}° ±{r_std:5.3f}  '
                  f'[{", ".join(f"{m:.1f}" for m in seed_medians_t)}]')

    print(f'\nFor reference: v19d model achieves ~148mm / 2.13° at same noise')


if __name__ == '__main__':
    main()
