#!/usr/bin/env python3
"""Evaluate MSFlowPoseNet checkpoint with variable outer iterations.

Usage:
    python scripts/eval_outer_iters.py --config configs/exp039_room0_optimized.yaml \
        --checkpoint output/exp039_room0_optimized/checkpoints/best.pth \
        --iters 1 3 5 10 15 20 --gpu 5
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
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
        irls_iters=mc.get('irls_iters', 3),
        corr_temperature=mc.get('corr_temperature', 1.0),
        skip_coarse_flow=mc.get('skip_coarse_flow', False),
    ).to(device)

    # Load checkpoint
    print(f"[Loading checkpoint: {args.checkpoint}]")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  Epoch: {epoch}")

    # Val dataset (same split as training)
    dc = cfg['data']
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc['train_traj_path'],
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=dc.get('val_noise_rot_deg', dc.get('noise_rot_deg', 8.0)),
        noise_trans_m=dc.get('val_noise_trans_m', dc.get('noise_trans_m', 0.25)),
        is_train=True)
    n_val = max(1, int(len(full_ds) * dc.get('val_split_ratio', 0.1)))
    _, val_ds = torch.utils.data.random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_v4, pin_memory=True)
    print(f"  Val samples: {len(val_ds)}")

    print(f"\n{'='*70}")
    print(f"  Testing outer iterations: {args.iters}")
    print(f"{'='*70}\n")

    for num_iters in args.iters:
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

                # Final pose = pose_cur (already updated by all iterations)
                Rr = torch.bmm(pose_cur[:, :3, :3].float().transpose(1, 2),
                               pose_gt.float()[:, :3, :3])
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(pose_cur[:, :3, 3].float() - pose_gt.float()[:, :3, 3],
                               dim=1) * 1000
                all_rot.extend(re.cpu().tolist())
                all_trans.extend(te.cpu().tolist())

        r = np.array(all_rot)
        t = np.array(all_trans)
        print(f"  oi={num_iters:2d}: rot={np.mean(r):.3f}° (med {np.median(r):.3f}°)  "
              f"trans={np.mean(t):.1f}mm (med {np.median(t):.1f}mm)  "
              f"<1°={np.mean(r < 1) * 100:.1f}%  <0.5°={np.mean(r < 0.5) * 100:.1f}%")


if __name__ == '__main__':
    main()
