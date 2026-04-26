#!/usr/bin/env python3
"""Evaluate MSFlowPoseNet with variable noise AND outer iterations.

Usage:
    python scripts/eval_noise_sweep.py --config configs/exp039_room0_optimized.yaml \
        --checkpoint output/exp039_room0_optimized/checkpoints/best.pth \
        --noise-rot 1 2 4 8 --iters 5 10 15 --gpu 5
"""
import sys, yaml, torch, math, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_with_settings(model, renderer, val_loader, device, num_iters):
    """Evaluate with given outer iterations."""
    all_rot, all_trans = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"oi={num_iters:2d}", leave=False):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)

            for oi in range(num_iters):
                res = renderer.render_batch(pose_cur,
                    scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                    return_depth=True)
                rf = {'coarse': res['coarse_feat'], 'mid': res['mid_feat'],
                      'fine_sd': res['fine_sd_feat'], 'fine_dino': res['fine_dino_feat']}
                depth = res.get('depth_map')
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, rf, depth)
                if 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())

            Rr = torch.bmm(pose_cur[:, :3, :3].float().transpose(1, 2),
                           pose_gt.float()[:, :3, :3])
            tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
            ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
            re = torch.acos(ca) * 180 / math.pi
            te = torch.norm(pose_cur[:, :3, 3].float() - pose_gt.float()[:, :3, 3],
                           dim=1) * 1000
            all_rot.extend(re.cpu().tolist())
            all_trans.extend(te.cpu().tolist())
    return np.array(all_rot), np.array(all_trans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--noise-rot', type=float, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--iters', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Renderer
    print("[Loading renderer...]")
    rc = cfg['renderer']
    renderer = MultiScaleRenderer(
        ply_path=rc['ply_path'], scale_model_paths=rc['scale_model_paths'],
        device=device, img_height=rc.get('img_height', 480),
        img_width=rc.get('img_width', 640), fx=rc.get('fx', 320.0),
        fy=rc.get('fy', 320.0), cx=rc.get('cx', 319.5), cy=rc.get('cy', 239.5))

    # Model
    mc = cfg['model']
    model = MSFlowPoseNet(
        hidden_dim=mc.get('hidden_dim', 128), decode_dim=mc.get('decode_dim', 64),
        local_radius=mc.get('local_radius', 4), damping=mc.get('damping', 0.001),
        coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
        mid_hw=tuple(mc.get('mid_hw', [15, 20])),
        fine_hw=tuple(mc.get('fine_hw', [35, 46])),
        fine_iters=mc.get('fine_iters', 4),
        mid_iters=mc.get('mid_iters', 1),
        coarse_in_dim=mc.get('coarse_in_dim', 32),
        mid_in_dim=mc.get('mid_in_dim', 64),
        fine_sd_in_dim=mc.get('fine_sd_in_dim', 64),
        fine_dino_in_dim=mc.get('fine_dino_in_dim', 64),
        irls_iters=mc.get('irls_iters', 0),
        geometry_upsample=mc.get('geometry_upsample', 1),
        multiscale_consistency=mc.get('multiscale_consistency', False),
        ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
        pixel_stride=mc.get('pixel_stride', 1),
            positional_encoding=mc.get('positional_encoding', False),
            pe_mode=mc.get('pe_mode', 'concat'),
            pe_dim=mc.get('pe_dim', 32),
    ).to(device)

    print(f"[Loading checkpoint: {args.checkpoint}]")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')}")

    dc = cfg['data']
    # Determine noise ratio for translation based on config
    base_rot = dc.get('val_noise_rot_deg', dc.get('noise_rot_deg', 8.0))
    base_trans = dc.get('val_noise_trans_m', dc.get('noise_trans_m', 0.25))
    trans_per_rot = base_trans / base_rot  # Keep rot/trans ratio constant

    print(f"\n{'='*80}")
    print(f"  Noise sweep × Outer iterations")
    print(f"  Noise levels: {args.noise_rot}° | Outer iters: {args.iters}")
    print(f"{'='*80}\n")

    for noise_rot in args.noise_rot:
        noise_trans = noise_rot * trans_per_rot
        # Create dataset with this noise level
        full_ds = PoseDatasetV4(
            feature_base_dir=dc['train_feature_dir'],
            traj_path=dc['train_traj_path'],
            depth_dir=dc.get('train_depth_dir'),
            noise_rot_deg=noise_rot,
            noise_trans_m=noise_trans,
            is_train=True)
        n_val = max(1, int(len(full_ds) * dc.get('val_split_ratio', 0.1)))
        _, val_ds = torch.utils.data.random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                num_workers=2, collate_fn=collate_v4, pin_memory=True)

        print(f"--- Noise: {noise_rot:.1f}° / {noise_trans:.3f}m ({len(val_ds)} samples) ---")
        for num_iters in args.iters:
            r, t = eval_with_settings(model, renderer, val_loader, device, num_iters)
            print(f"  oi={num_iters:2d}: rot={np.mean(r):.3f}° (med {np.median(r):.3f}°)  "
                  f"trans={np.mean(t):.1f}mm (med {np.median(t):.1f}mm)  "
                  f"<1°={np.mean(r < 1) * 100:.1f}%  <0.5°={np.mean(r < 0.5) * 100:.1f}%  "
                  f"<0.1°={np.mean(r < 0.1) * 100:.1f}%")
        print()


if __name__ == '__main__':
    main()
