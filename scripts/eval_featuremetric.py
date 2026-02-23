#!/usr/bin/env python3
"""
FDA (Featuremetric Direct Alignment) 全量评估
在多个噪声级别下评估 900 帧的定位精度
"""
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

# 使用文件日志避免 stdout 缓冲问题
LOG_FILE = '/tmp/fda_eval.log'
_log_fh = open(LOG_FILE, 'w')
import os as _os

def log(msg=''):
    """同时输出到 stdout 和日志文件, 立即 flush + fsync"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()
    _log_fh.write(msg + '\n')
    _log_fh.flush()
    _os.fsync(_log_fh.fileno())


import torch
import numpy as np
import time
import argparse
from modules.featuremetric import FeaturemetricAligner
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from losses.sequence_loss import rotation_geodesic_loss

def evaluate(args):
    device = torch.device('cuda')
    INTRINSICS_FLOW = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

    log("Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
        scale_model_paths={
            'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
            'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
        },
        device=device,
    )

    aligner = FeaturemetricAligner(
        renderer=renderer,
        intrinsics=INTRINSICS_FLOW,
        scale_names=args.scales.split(','),
        damping=args.damping,
        max_iters=args.max_iters,
        use_rendered_depth=args.use_rendered_depth,
        rel_convergence_thresh=args.rel_convergence_thresh,
        rel_convergence_patience=args.rel_convergence_patience,
    )

    for noise_deg in args.noise_levels:
        noise_trans = noise_deg / 50.0
        num_frames = min(args.num_frames, 900) if args.num_frames > 0 else 900

        dataset = PoseDatasetV3(
            feature_base_dir='output/features_multiscale/room_0',
            traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
            depth_dir='dataset/room_0/Sequence_1/depth',
            frame_indices=list(range(900)),
            scale_names=['fine_sd', 'fine_dino'],
            noise_rot_deg=noise_deg,
            noise_trans_m=noise_trans,
            is_train=True,
            depth_resize=(35, 46),
        )

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        init_rots, final_rots = [], []
        init_trans, final_trans = [], []
        improvements = []
        timings = []

        log(f"\n{'='*70}")
        log(f"Noise: {noise_deg}° / {noise_trans:.3f}m  |  {num_frames} frames  |  mode={'fast' if args.fast else 'lm'}")
        log(f"{'='*70}")

        for idx in range(num_frames):
          try:
            sample = dataset[idx]
            query_feats = {k: v.unsqueeze(0).to(device) for k, v in sample['query_feats'].items()}
            pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
            initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)

            init_rot = rotation_geodesic_loss(
                initial_pose[:, :3, :3], pose_gt[:, :3, :3]
            ).item() * 180 / np.pi
            init_t = torch.norm(initial_pose[:, :3, 3] - pose_gt[:, :3, 3]).item()

            t0 = time.time()
            align_fn = aligner.align_fast if args.fast else aligner.align
            result = align_fn(
                query_feats=query_feats,
                initial_pose=initial_pose,
                depth_for_jac=depth,
                verbose=False,
            )
            elapsed = time.time() - t0

            eval_pose = result.get('best_pose', result['final_pose'])
            final_rot = rotation_geodesic_loss(
                eval_pose[:, :3, :3], pose_gt[:, :3, :3]
            ).item() * 180 / np.pi
            final_t = torch.norm(eval_pose[:, :3, 3] - pose_gt[:, :3, 3]).item()

            init_rots.append(init_rot)
            final_rots.append(final_rot)
            init_trans.append(init_t)
            final_trans.append(final_t)
            improvements.append(init_rot - final_rot)
            timings.append(elapsed)

            if (idx + 1) % 10 == 0 or idx < 5:
                avg_imp = np.mean(improvements[-100:])
                avg_fin = np.mean(final_rots[-100:])
                log(f"  [{idx+1:4d}/{num_frames}] "
                    f"Δ={avg_imp:+.2f}° final_rot={avg_fin:.2f}° "
                    f"({elapsed:.2f}s/frame)")
          except Exception as e:
            log(f"  [ERROR at idx={idx}] {e}")
            import traceback
            traceback.print_exc()

        # 汇总统计
        init_rots = np.array(init_rots)
        final_rots = np.array(final_rots)
        init_trans_arr = np.array(init_trans)
        final_trans_arr = np.array(final_trans)
        improvements = np.array(improvements)

        log(f"\n--- Results: Noise {noise_deg}° ---")
        log(f"  Rotation (°):  init={np.median(init_rots):.2f} → final={np.median(final_rots):.2f} "
              f"(median Δ={np.median(improvements):+.2f}°)")
        log(f"  Rotation mean: init={np.mean(init_rots):.2f} → final={np.mean(final_rots):.2f}")
        log(f"  Translation (m): init={np.median(init_trans_arr):.4f} → "
              f"final={np.median(final_trans_arr):.4f}")
        log(f"  <1° rate:  {(final_rots < 1.0).mean()*100:.1f}%  (init: {(init_rots < 1.0).mean()*100:.1f}%)")
        log(f"  <2° rate:  {(final_rots < 2.0).mean()*100:.1f}%  (init: {(init_rots < 2.0).mean()*100:.1f}%)")
        log(f"  <5° rate:  {(final_rots < 5.0).mean()*100:.1f}%  (init: {(init_rots < 5.0).mean()*100:.1f}%)")
        log(f"  Positive rate: {(improvements > 0).mean()*100:.1f}%")
        log(f"  Time: {np.mean(timings):.2f}s/frame")

        # Train/Val 分离
        train_end = min(810, num_frames)
        val_start = 810 if num_frames > 810 else num_frames
        splits = [("Train", list(range(train_end)))]
        if num_frames > 810:
            splits.append(("Val", list(range(810, num_frames))))
        for split, indices in splits:
            if not indices:
                continue
            fr = final_rots[indices]
            ft = final_trans_arr[indices]
            imp = improvements[indices]
            log(f"  [{split:5s}] rot={np.median(fr):.2f}° (mean={np.mean(fr):.2f}°) "
                  f"trans={np.median(ft):.4f}m <1°={100*(fr<1).mean():.1f}% "
                  f"<5°={100*(fr<5).mean():.1f}% pos={100*(imp>0).mean():.1f}%")

    log("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_levels', nargs='+', type=float, default=[2.0, 5.0])
    parser.add_argument('--damping', type=float, default=1e-2)
    parser.add_argument('--max_iters', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=0, help='0 = all 900')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scales', type=str, default='fine_dino',
                       help='Comma-separated scale names, e.g. fine_dino or fine_sd,fine_dino')
    parser.add_argument('--fast', action='store_true', default=True, help='Use fast GN mode')
    parser.add_argument('--lm', dest='fast', action='store_false', help='Use LM mode (slower, more robust)')
    parser.add_argument('--use_rendered_depth', action='store_true', default=False)
    parser.add_argument('--rendered_depth', dest='use_rendered_depth', action='store_true')
    parser.add_argument('--rel_convergence_thresh', type=float, default=0.0,
                       help='Relative convergence threshold (0=disabled, 0.005=0.5%%)')
    parser.add_argument('--rel_convergence_patience', type=int, default=3,
                       help='Patience for relative convergence')
    args = parser.parse_args()
    evaluate(args)
