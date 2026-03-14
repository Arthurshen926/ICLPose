#!/usr/bin/env python3
"""Evaluate MSFlowPoseNet checkpoints with iterative refinement at val.

Usage:
    python scripts/eval_iterative.py --config configs/exp032_cosine_fiters8.yaml \
        --checkpoint output/exp032_cosine_fiters8/checkpoints/best.pth

    # Test multiple iteration counts
    python scripts/eval_iterative.py --config configs/exp032_cosine_fiters8.yaml \
        --checkpoint output/exp032_cosine_fiters8/checkpoints/best.pth \
        --iters 1 3 5 10

    # Also test 'latest' checkpoint
    python scripts/eval_iterative.py --config configs/exp032_cosine_fiters8.yaml \
        --checkpoint output/exp032_cosine_fiters8/checkpoints/best.pth \
                     output/exp032_cosine_fiters8/checkpoints/latest.pth
"""
import sys, yaml, torch, math, argparse, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from modules.featuremetric import FeaturemetricAligner
from data.dataset_v4 import PoseDatasetV4, collate_v4
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
    )


def build_val_loader(cfg):
    dc = cfg['data']
    # If explicit val paths provided, use them; otherwise split from train
    val_feat_dir = dc.get('val_feature_dir') or dc['train_feature_dir']
    val_traj = dc.get('val_traj_path') or dc['train_traj_path']
    val_depth = dc.get('val_depth_dir') or dc.get('train_depth_dir')

    if dc.get('val_traj_path'):
        # Dedicated val set
        val_ds = PoseDatasetV4(
            feature_base_dir=val_feat_dir,
            traj_path=val_traj,
            depth_dir=val_depth,
            noise_rot_deg=dc.get('val_noise_rot_deg', dc.get('noise_rot_deg', 8.0)),
            noise_trans_m=dc.get('val_noise_trans_m', dc.get('noise_trans_m', 0.25)),
            is_train=False,
        )
    else:
        # Split from train
        full_ds = PoseDatasetV4(
            feature_base_dir=dc['train_feature_dir'],
            traj_path=dc['train_traj_path'],
            depth_dir=dc.get('train_depth_dir'),
            noise_rot_deg=dc.get('noise_rot_deg', 8.0),
            noise_trans_m=dc.get('noise_trans_m', 0.25),
            is_train=True,
        )
        n_val = max(1, int(len(full_ds) * dc.get('val_split_ratio', 0.1)))
        _, val_ds = torch.utils.data.random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    return DataLoader(
        val_ds,
        batch_size=dc.get('val_batch_size', 4),
        shuffle=False,
        num_workers=dc.get('num_workers', 2),
        collate_fn=collate_v4,
        pin_memory=True,
    )


def evaluate(model, renderer, val_loader, num_iters, device, fda_aligner=None):
    """Run evaluation with a given number of outer iterations.

    Args:
        fda_aligner: optional FeaturemetricAligner for post-refinement
    """
    model.eval()
    all_rot, all_trans = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"iters={num_iters}", leave=False):
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
                if oi < num_iters - 1 and 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            if 'delta_xi' in pred:
                T = se3_exp(pred['delta_xi'].float())
                pp = torch.bmm(T, pose_cur.float())

                # ── FDA post-refinement (optional) ──
                if fda_aligner is not None and depth is not None:
                    fda_result = fda_aligner.align(
                        query_feats=qf,
                        initial_pose=pp,
                        depth_for_jac=depth,
                    )
                    pp = fda_result['best_pose'].float()

                Rr = torch.bmm(pp[:, :3, :3].transpose(1, 2), pose_gt.float()[:, :3, :3])
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(pp[:, :3, 3] - pose_gt.float()[:, :3, 3], dim=1) * 1000
                all_rot.extend(re.cpu().tolist())
                all_trans.extend(te.cpu().tolist())

    r = np.array(all_rot)
    t = np.array(all_trans)
    metrics = {
        'rot_mean': float(np.mean(r)),
        'rot_median': float(np.median(r)),
        'trans_mean': float(np.mean(t)),
        'trans_median': float(np.median(t)),
        'pct_lt1deg': float(np.mean(r < 1.0) * 100),
        'pct_lt5deg': float(np.mean(r < 5.0) * 100),
        'joint_01deg_53mm': float(np.mean((r < 0.1) & (t < 5.3)) * 100),
        'joint_1deg_50mm': float(np.mean((r < 1.0) & (t < 50.0)) * 100),
        'joint_5deg_100mm': float(np.mean((r < 5.0) & (t < 100.0)) * 100),
        'num_samples': len(r),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSFlowPoseNet with iterative refinement')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--checkpoint', type=str, nargs='+', required=True,
                        help='Checkpoint path(s) to evaluate')
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5],
                        help='Number of outer iterations to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON to this path')
    parser.add_argument('--fda_refine', action='store_true',
                        help='Apply FDA post-refinement after model prediction')
    parser.add_argument('--fda_iters', type=int, default=20,
                        help='Max FDA Gauss-Newton iterations')
    parser.add_argument('--fda_scales', type=str, nargs='+',
                        default=['fine_sd', 'fine_dino'],
                        help='Feature scales to use for FDA refinement')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda'
    print(f"[Config] {args.config}")

    print("[Loading renderer...]")
    renderer = build_renderer(cfg, device)
    model = build_model(cfg, device)
    val_loader = build_val_loader(cfg)

    # ── FDA aligner (optional) ──
    fda_aligner = None
    if args.fda_refine:
        rc = cfg.get('renderer', {})
        intrinsics = {
            'fx': rc.get('fx', 320.0),
            'fy': rc.get('fy', 320.0),
            'cx': rc.get('cx', 319.5),
            'cy': rc.get('cy', 239.5),
        }
        fda_aligner = FeaturemetricAligner(
            renderer=renderer,
            intrinsics=intrinsics,
            scale_names=args.fda_scales,
            max_iters=args.fda_iters,
            damping=1e-2,
        )
        print(f"[FDA] Post-refinement enabled: scales={args.fda_scales}, "
              f"max_iters={args.fda_iters}")

    all_results = {}
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        ckpt_name = Path(ckpt_path).stem
        epoch = ckpt.get('epoch', '?')
        print(f"\n{'=' * 60}")
        print(f"  Checkpoint: {ckpt_name} (epoch={epoch})")
        print(f"{'=' * 60}")

        ckpt_results = {}
        for num_iters in args.iters:
            metrics = evaluate(model, renderer, val_loader, num_iters, device,
                               fda_aligner=fda_aligner)
            r = metrics
            print(f"  iters={num_iters}: rot={r['rot_mean']:.2f}° "
                  f"(med {r['rot_median']:.2f}°)  "
                  f"trans={r['trans_mean']:.1f}mm  "
                  f"<1°={r['pct_lt1deg']:.1f}%  "
                  f"joint@0.1°/5.3mm={r['joint_01deg_53mm']:.1f}%  "
                  f"joint@1°/50mm={r['joint_1deg_50mm']:.1f}%")
            ckpt_results[f"iters_{num_iters}"] = metrics

        all_results[ckpt_name] = ckpt_results

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
