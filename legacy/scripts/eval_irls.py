#!/usr/bin/env python3
"""
Quick eval script to test IRLS effect on an existing checkpoint.
Tests different irls_iters and outer_iters values without retraining.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_irls.py \
        --config configs/exp039_room0_optimized.yaml \
        --checkpoint output/exp039_room0_optimized/checkpoints/best.pth \
        --irls_iters 0 1 3 5 \
        --val_outer_iters 5 10 15
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4


def rotation_error(T_pred, T_gt):
    R_pred = T_pred[:, :3, :3]
    R_gt = T_gt[:, :3, :3]
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_angle) * 180 / np.pi


def translation_error(T_pred, T_gt):
    return torch.norm(T_pred[:, :3, 3] - T_gt[:, :3, 3], dim=1) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--irls_iters', type=int, nargs='+', default=[0, 1, 3, 5])
    parser.add_argument('--val_outer_iters', type=int, nargs='+', default=[5])
    parser.add_argument('--huber_k', type=float, default=1.345)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mc = config['model']
    rc = config['renderer']
    tc = config['training']
    dc = config['data']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Build val dataset (same seed=42 split as training)
    # Use is_train=True so it applies 8° noise (matching training), not 25° default eval noise
    full_dataset = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc['train_traj_path'],
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=dc.get('noise_rot_deg', 15.0),
        noise_trans_m=dc.get('noise_trans_m', 0.5),
        is_train=True,
    )
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * dc.get('val_split_ratio', 0.1)))
    n_train = n_total - n_val
    _, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=2, collate_fn=collate_v4)

    # Full-image intrinsics (model internally scales to fine resolution)
    fine_hw = mc.get('fine_hw', [35, 46])
    intrinsics = {
        'fx': rc.get('fx', 320.0),
        'fy': rc.get('fy', 320.0),
        'cx': rc.get('cx', 319.5),
        'cy': rc.get('cy', 239.5),
    }

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"\n{'='*70}")
    print(f"  IRLS Evaluation: {args.config}")
    print(f"  Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    print(f"  Val samples: {len(val_ds)}")
    print(f"  Testing irls_iters: {args.irls_iters}")
    print(f"  Testing val_outer_iters: {args.val_outer_iters}")
    print(f"{'='*70}\n")

    for n_outer in args.val_outer_iters:
        for n_irls in args.irls_iters:
            model = MSFlowPoseNet(
                hidden_dim=mc.get('hidden_dim', 128),
                decode_dim=mc.get('decode_dim', 64),
                local_radius=mc.get('local_radius', 4),
                damping=mc.get('damping', 1e-3),
                coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
                mid_hw=tuple(mc.get('mid_hw', [15, 20])),
                fine_hw=tuple(fine_hw),
                fine_iters=mc.get('fine_iters', 4),
                mid_iters=mc.get('mid_iters', 1),
                corr_temperature=mc.get('corr_temperature', 1.0),
                intrinsics=intrinsics,
                img_hw=(rc.get('img_height', 480), rc.get('img_width', 640)),
                coarse_in_dim=mc.get('coarse_in_dim', 512),
                mid_in_dim=mc.get('mid_in_dim', 512),
                fine_sd_in_dim=mc.get('fine_sd_in_dim', 512),
                fine_dino_in_dim=mc.get('fine_dino_in_dim', 768),
                irls_iters=n_irls,
                irls_huber_k=args.huber_k,
                deep_flow_head=mc.get('deep_flow_head', False),
                cross_scale_context=mc.get('cross_scale_context', False),
                cross_scale_dim=mc.get('cross_scale_dim', 32),
                pose_refinement=mc.get('pose_refinement', False),
                corr_dilations=tuple(mc['corr_dilations']) if mc.get('corr_dilations') else None,
                geometry_upsample=mc.get('geometry_upsample', 1),
        multiscale_consistency=mc.get('multiscale_consistency', False),
        ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
        pixel_stride=mc.get('pixel_stride', 1),
            positional_encoding=mc.get('positional_encoding', False),
            pe_mode=mc.get('pe_mode', 'concat'),
            pe_dim=mc.get('pe_dim', 32),
            ).to(device)

            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()

            all_rot, all_trans = [], []
            for batch in tqdm(val_loader, desc=f"oi={n_outer},irls={n_irls}", leave=False):
                query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
                pose_gt = batch['pose_gt'].to(device)
                pose_cur = batch['initial_pose'].to(device)

                for outer_i in range(n_outer):
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
                        result = renderer.render_batch(
                            pose_cur, scales=scales, return_depth=True)
                        render_feats = {
                            'coarse': result['coarse_feat'],
                            'mid': result['mid_feat'],
                            'fine_sd': result['fine_sd_feat'],
                            'fine_dino': result['fine_dino_feat'],
                        }
                        depth = result.get('depth_map')
                        pred = model(query_feats, render_feats, depth)

                    if 'delta_xi' in pred:
                        T_delta = se3_exp(pred['delta_xi'].float())
                        pose_cur = torch.bmm(T_delta, pose_cur.float())

                rot_err = rotation_error(pose_cur, pose_gt)
                trans_err = translation_error(pose_cur, pose_gt)
                all_rot.extend(rot_err.cpu().tolist())
                all_trans.extend(trans_err.cpu().tolist())

            rot_arr = np.array(all_rot)
            trans_arr = np.array(all_trans)
            pct_lt1 = (rot_arr < 1.0).mean() * 100

            print(f"  oi={n_outer:2d} irls={n_irls} | "
                  f"rot={rot_arr.mean():.3f}° (med {np.median(rot_arr):.3f}°) | "
                  f"trans={trans_arr.mean():.1f}mm (med {np.median(trans_arr):.1f}mm) | "
                  f"<1°={pct_lt1:.1f}%")

            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
