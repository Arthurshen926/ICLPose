#!/usr/bin/env python3
"""Quick test: evaluate exp030 checkpoints with iterative refinement at val."""
import sys, yaml, torch, math, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    with open('configs/exp030_ms_flow_v2.yaml') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda'

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
    ).to(device)

    # Val dataset (auto-split, same seed as training)
    dc = cfg['data']
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc['train_traj_path'],
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=dc.get('noise_rot_deg', 8.0),
        noise_trans_m=dc.get('noise_trans_m', 0.25), is_train=True)
    n_val = max(1, int(len(full_ds) * 0.1))
    _, val_ds = torch.utils.data.random_split(
        full_ds, [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=2, collate_fn=collate_v4, pin_memory=True)

    # Test checkpoints
    for ckpt_name in ['best', 'latest']:
        path = f'output/exp030_ms_flow_v2/checkpoints/{ckpt_name}.pth'
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_name} (epoch={ckpt['epoch']})")
        print(f"{'='*60}")

        for num_iters in [1, 3, 5]:
            all_rot, all_trans = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"iters={num_iters}", leave=False):
                    qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
                    pose_gt = batch['pose_gt'].to(device)
                    pose_cur = batch['initial_pose'].to(device)

                    for oi in range(num_iters):
                        res = renderer.render_batch(pose_cur,
                            scales=['coarse','mid','fine_sd','fine_dino'],
                            return_depth=True)
                        rf = {'coarse': res['coarse_feat'], 'mid': res['mid_feat'],
                              'fine_sd': res['fine_sd_feat'], 'fine_dino': res['fine_dino_feat']}
                        depth = res.get('depth_map')
                        with torch.cuda.amp.autocast(enabled=True):
                            pred = model(qf, rf, depth)
                        if oi < num_iters - 1 and 'delta_xi' in pred:
                            T = se3_exp(pred['delta_xi'].float())
                            pose_cur = torch.bmm(T, pose_cur.float())

                    if 'delta_xi' in pred:
                        T = se3_exp(pred['delta_xi'].float())
                        pp = torch.bmm(T, pose_cur.float())
                        Rr = torch.bmm(pp[:,:3,:3].transpose(1,2), pose_gt.float()[:,:3,:3])
                        tr = Rr[:,0,0] + Rr[:,1,1] + Rr[:,2,2]
                        ca = torch.clamp((tr-1)/2, -1+1e-4, 1-1e-4)
                        re = torch.acos(ca) * 180 / math.pi
                        te = torch.norm(pp[:,:3,3] - pose_gt.float()[:,:3,3], dim=1) * 1000
                        all_rot.extend(re.cpu().tolist())
                        all_trans.extend(te.cpu().tolist())

            r = np.array(all_rot); t = np.array(all_trans)
            print(f"  iters={num_iters}: rot={np.mean(r):.2f}° (med {np.median(r):.2f}°)  "
                  f"trans={np.mean(t):.1f}mm  <1°={np.mean(r<1)*100:.1f}%  "
                  f"<5°={np.mean(r<5)*100:.1f}%")

if __name__ == '__main__':
    main()
