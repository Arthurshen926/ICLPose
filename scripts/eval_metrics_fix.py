#!/usr/bin/env python3
"""
Quick eval with FIXED rotation metrics (clamp 1e-7 instead of 1e-4).
Tests exp031 best checkpoint to reveal true rotation accuracy hidden by the 0.81° floor.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda')
    config_path = 'configs/exp031_iterative.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Renderer
    print("[Loading renderer...]")
    rc = config['renderer']
    renderer = MultiScaleRenderer(
        ply_path=rc['ply_path'],
        scale_model_paths=rc['scale_model_paths'],
        img_height=rc['img_height'], img_width=rc['img_width'],
        fx=rc['fx'], fy=rc['fy'], cx=rc['cx'], cy=rc['cy'],
    )
    renderer.to(device)

    # Model
    mc = config['model']
    model = MSFlowPoseNet(
        hidden_dim=mc.get('hidden_dim', 128),
        decode_dim=mc.get('decode_dim', 64),
        local_radius=mc.get('local_radius', 4),
        damping=mc.get('damping', 0.001),
        fine_iters=mc.get('fine_iters', 4),
        mid_iters=mc.get('mid_iters', 1),
        coarse_hw=tuple(mc.get('coarse_hw', [7, 10])),
        mid_hw=tuple(mc.get('mid_hw', [15, 20])),
        fine_hw=tuple(mc.get('fine_hw', [35, 46])),
        coarse_in_dim=mc.get('coarse_in_dim', 32),
        mid_in_dim=mc.get('mid_in_dim', 64),
        fine_sd_in_dim=mc.get('fine_sd_in_dim', 64),
        fine_dino_in_dim=mc.get('fine_dino_in_dim', 64),
        irls_iters=mc.get('irls_iters', 3),
        corr_temperature=mc.get('corr_temperature', 1.0),
        skip_coarse_flow=mc.get('skip_coarse_flow', False),
    ).to(device)

    # Dataset (val split)
    dc = config['data']
    dataset = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc['val_traj_path'],
        depth_dir=dc.get('val_depth_dir'),
        noise_rot_deg=dc['val_noise_rot_deg'],
        noise_trans_m=dc['val_noise_trans_m'],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_v4)

    # Test checkpoints
    ckpt_dir = Path('output/exp031_iterative/checkpoints')
    for ckpt_name in ['best', 'latest']:
        ckpt_path = ckpt_dir / f'{ckpt_name}.pth'
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        epoch = ckpt.get('epoch', '?')

        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_name} (epoch={epoch})")
        print(f"{'='*60}")

        for N in [1, 3, 5, 7]:
            all_rot, all_trans = [], []
            for batch in loader:
                query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
                pose_gt = batch['pose_gt'].to(device)
                pose_cur = batch['initial_pose'].to(device)

                with torch.no_grad():
                    for i in range(N):
                        scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
                        result = renderer.render_batch(pose_cur, scales=scales, return_depth=True)
                        render_feats = {
                            'coarse': result['coarse_feat'],
                            'mid': result['mid_feat'],
                            'fine_sd': result['fine_sd_feat'],
                            'fine_dino': result['fine_dino_feat'],
                        }
                        depth = result.get('depth_map')

                        with torch.cuda.amp.autocast(enabled=True):
                            pred = model(query_feats, render_feats, depth)

                        if 'delta_xi' in pred:
                            T_delta = se3_exp(pred['delta_xi'].float())
                            pose_cur = torch.bmm(T_delta, pose_cur.float())

                    # Final error with FIXED clamp (1e-7)
                    R_pred = pose_cur[:, :3, :3].float()
                    R_gt = pose_gt[:, :3, :3].float()
                    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
                    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
                    # KEY FIX: 1e-7 instead of 1e-4 (0.81° floor removed!)
                    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
                    rot_err = torch.acos(cos_angle) * 180.0 / math.pi

                    t_pred = pose_cur[:, :3, 3].float()
                    t_gt = pose_gt[:, :3, 3].float()
                    trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000

                all_rot.extend(rot_err.cpu().tolist())
                all_trans.extend(trans_err.cpu().tolist())

            rot = np.array(all_rot)
            trans = np.array(all_trans)
            pct1 = np.mean(rot < 1.0) * 100
            pct5 = np.mean(rot < 5.0) * 100
            pct05 = np.mean(rot < 0.5) * 100
            print(f"  iters={N}: rot={np.mean(rot):.3f}° (med {np.median(rot):.3f}°) "
                  f" trans={np.mean(trans):.1f}mm"
                  f"  <0.5°={pct05:.1f}%  <1°={pct1:.1f}%  <5°={pct5:.1f}%")

            # Show outlier distribution
            high = rot[rot > 2.0]
            if len(high) > 0:
                print(f"         outliers(>2°): {len(high)} frames, "
                      f"max={rot.max():.1f}°, "
                      f"mean_of_outliers={high.mean():.1f}°")

if __name__ == '__main__':
    main()
