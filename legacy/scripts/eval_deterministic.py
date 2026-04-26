#!/usr/bin/env python3
"""General deterministic eval: fixed seed + multiple outer_iters."""
import sys, yaml, torch, math, numpy as np, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader

def eval_model(model, renderer, val_loader, device, outer_iters):
    all_rot, all_trans = [], []
    with torch.no_grad():
        for batch in val_loader:
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_cur = batch['initial_pose'].to(device)
            for oi in range(outer_iters):
                res = renderer.render_batch(pose_cur,
                    scales=['coarse', 'mid', 'fine_sd', 'fine_dino'], return_depth=True)
                rf = {'coarse': res['coarse_feat'], 'mid': res['mid_feat'],
                      'fine_sd': res['fine_sd_feat'], 'fine_dino': res['fine_dino_feat']}
                depth = res.get('depth_map')
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, rf, depth)
                if 'delta_xi' in pred:
                    T = se3_exp(pred['delta_xi'].float())
                    pose_cur = torch.bmm(T, pose_cur.float())
            Rr = torch.bmm(pose_cur[:,:3,:3].float().transpose(1,2), pose_gt.float()[:,:3,:3])
            tr = Rr[:,0,0]+Rr[:,1,1]+Rr[:,2,2]
            ca = torch.clamp((tr-1)/2, -1+1e-7, 1-1e-7)
            re = torch.acos(ca)*180/math.pi
            te = torch.norm(pose_cur[:,:3,3].float()-pose_gt.float()[:,:3,3], dim=1)*1000
            all_rot.extend(re.cpu().tolist())
            all_trans.extend(te.cpu().tolist())
    r, t = np.array(all_rot), np.array(all_trans)
    return r, t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outer_iters', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--noise_deg', type=float, default=8.0)
    parser.add_argument('--noise_m', type=float, default=0.25)
    args = parser.parse_args()

    device = 'cuda'
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rc = cfg['renderer']
    intrinsics = {
        'img_height': rc.get('img_height', 480), 'img_width': rc.get('img_width', 640),
        'fx': rc.get('fx', 320.0), 'fy': rc.get('fy', 320.0),
        'cx': rc.get('cx', 319.5), 'cy': rc.get('cy', 239.5)
    }
    print(f"[Renderer] fx={intrinsics['fx']}, fy={intrinsics['fy']}")
    renderer = MultiScaleRenderer(
        ply_path=rc['ply_path'], scale_model_paths=rc['scale_model_paths'],
        device=device, **intrinsics)

    mc = cfg.get('model', {})
    model_intrinsics = {
        'fx': intrinsics['fx'], 'fy': intrinsics['fy'],
        'cx': intrinsics['cx'], 'cy': intrinsics['cy'],
    }
    model = MSFlowPoseNet(
        hidden_dim=mc.get('hidden_dim', 128), decode_dim=mc.get('decode_dim', 64),
        local_radius=mc.get('local_radius', 4), damping=mc.get('damping', 0.001),
        coarse_hw=mc.get('coarse_hw', [15,20]), mid_hw=mc.get('mid_hw', [30,40]),
        fine_hw=mc.get('fine_hw', [69,91]),
        coarse_in_dim=mc.get('coarse_in_dim', 32), mid_in_dim=mc.get('mid_in_dim', 64),
        fine_sd_in_dim=mc.get('fine_sd_in_dim', 64), fine_dino_in_dim=mc.get('fine_dino_in_dim', 64),
        fine_iters=mc.get('fine_iters', 8), mid_iters=mc.get('mid_iters', 1),
        irls_iters=mc.get('irls_iters', 0),
        geometry_upsample=mc.get('geometry_upsample', 1),
        multiscale_consistency=mc.get('multiscale_consistency', False),
        ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
        pixel_stride=mc.get('pixel_stride', 1),
            positional_encoding=mc.get('positional_encoding', False),
            pe_mode=mc.get('pe_mode', 'concat'),
            pe_dim=mc.get('pe_dim', 32),
        deep_flow_head=mc.get('deep_flow_head', False),
        cross_scale_context=mc.get('cross_scale_context', False),
        cross_scale_dim=mc.get('cross_scale_dim', 32),
        corr_dilations=mc.get('corr_dilations', None),
        intrinsics=model_intrinsics,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[Model] Loaded {args.checkpoint}")

    # Fixed-seed dataset
    dc = cfg['data']
    torch.manual_seed(12345)
    np.random.seed(12345)
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'], traj_path=dc['train_traj_path'],
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=args.noise_deg, noise_trans_m=args.noise_m, is_train=True)
    n_val = max(1, int(len(full_ds) * 0.1))
    _, val_ds = torch.utils.data.random_split(
        full_ds, [len(full_ds)-n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=0, collate_fn=collate_v4, pin_memory=True)
    print(f"[Data] Val: {len(val_ds)} samples, noise: {args.noise_deg}°/{args.noise_m}m")

    print(f"\n{'oi':>4} {'mean°':>8} {'med°':>8} {'mean_mm':>10} {'med_mm':>10} {'<1°':>6} {'<0.5°':>6} {'<0.25°':>6}")
    print("-" * 70)
    for oi in args.outer_iters:
        r, t = eval_model(model, renderer, val_loader, device, oi)
        pct1 = (r < 1.0).mean()*100
        pct05 = (r < 0.5).mean()*100
        pct025 = (r < 0.25).mean()*100
        print(f"{oi:4d} {r.mean():8.3f} {np.median(r):8.3f} {t.mean():10.1f} {np.median(t):10.1f} {pct1:5.1f}% {pct05:5.1f}% {pct025:5.1f}%")

if __name__ == '__main__':
    main()
