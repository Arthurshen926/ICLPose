#!/usr/bin/env python3
"""Diagnostic: analyze trained model's flow prediction vs GT and solver output."""
import sys
sys.path.insert(0, '.')

import torch
import yaml
import math
import numpy as np
import torch.nn.functional as F
from modules.lie_algebra import se3_exp

def main():
    device = torch.device('cuda')
    
    # Load config and setup
    with open('configs/refiner_oldhospital.yaml') as f:
        config = yaml.safe_load(f)
    
    # Build model
    from ic_models.pose_refiner import PoseRefiner
    mc = config['model']
    rc = config['renderer']
    model = PoseRefiner(
        in_dim=mc.get('in_dim', 64),
        match_dim=mc.get('match_dim', 64),
        hidden_dim=mc.get('hidden_dim', 128),
        n_heads=mc.get('n_heads', 4),
        n_attn_layers=mc.get('n_attn_layers', 2),
        ffn_dim=mc.get('ffn_dim', 128),
        local_radius=mc.get('local_radius', 4),
        fine_iters=mc.get('fine_iters', 4),
        damping=mc.get('damping', 0.001),
        coarse_hw=mc.get('coarse_hw', [17, 30]),
        fine_hw=mc.get('fine_hw', [34, 60]),
        solver_hw=mc.get('solver_hw', None),
        solver_upsample=mc.get('solver_upsample', 4),
        intrinsics={'fx': rc['fx'], 'fy': rc['fy'], 'cx': rc['cx'], 'cy': rc['cy']},
        img_hw=rc.get('img_hw', [1080, 1920]),
        depth_normalize=mc.get('depth_normalize', False),
        sequential_solve=mc.get('sequential_solve', True),
        detach_conf=mc.get('detach_conf', True),
        conf_floor=mc.get('conf_floor', 0.1),
        solver_trans_scale=mc.get('solver_trans_scale', 0.0),
        use_trans_head=mc.get('use_trans_head', False),
        trans_head_mode=mc.get('trans_head_mode', 'replace'),
    ).to(device)
    
    # Load checkpoint
    ckpt_path = 'output/refiner_oh_v1/checkpoints/best.pth'
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {ckpt_path}, epoch={ckpt.get('epoch', '?')}")
    except:
        print(f"Could not load {ckpt_path}, using random weights")
    
    model.eval()
    
    # Load dataset
    from scripts.train_refiner import RadioPoseDataset, RadioRenderer
    dc = config['data']
    val_ds = RadioPoseDataset(
        dc['feature_dir'], dc['traj_path'],
        noise_rot_deg=dc['val_noise_rot_deg'],
        noise_trans_m=dc['val_noise_trans_m'],
        is_train=False,
    )
    
    # Setup renderer (already imported above)
    renderer = RadioRenderer(
        rc['ply_path'], rc['feature_model_path'], device,
        rc['img_hw'], rc['fx'], rc['fy'], rc['cx'], rc['cy'],
    )
    
    feat_hw = tuple(rc['feat_hw'])
    
    # Analyze first 10 samples
    print(f"\n{'='*80}")
    print(f"Flow analysis (fine_hw={model.FINE_HW}, solver_hw={model.SOLVER_HW})")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx in range(min(10, len(val_ds))):
            sample = val_ds[idx]
            query_feat = sample['query_feat'].unsqueeze(0).to(device)
            pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
            pose_cur = sample['initial_pose'].unsqueeze(0).to(device)
            
            # Compute initial error
            pred_c2w = torch.inverse(pose_cur[0])
            gt_c2w = torch.inverse(pose_gt[0])
            R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
            trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
            cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
            init_rot = torch.acos(cos_a).item() * 180.0 / math.pi
            init_trans = (pred_c2w[:3, 3] - gt_c2w[:3, 3]).norm().item() * 100
            
            print(f"--- Sample {idx}: init error = {init_trans:.1f}cm / {init_rot:.2f}° ---")
            
            for outer_iter in range(3):
                # Render at current pose
                render_feat = renderer.render_features(pose_cur, feat_hw)
                depth = renderer.render_depth(pose_cur, feat_hw)
                
                # Model forward
                result = model(query_feat, render_feat, depth)
                
                pred_flow = result['flow_fine']
                pred_conf = result['conf_fine']
                
                # GT flow at fine resolution
                gt_flow, gt_mask = model.compute_gt_flow(
                    pose_cur, pose_gt, depth, model.FINE_HW)
                
                # Flow error
                flow_err = (pred_flow - gt_flow).abs() * gt_mask
                n_valid = gt_mask.sum().clamp(min=1)
                mae = flow_err.sum() / (n_valid * 2)
                
                # Flow magnitude
                pred_mag = pred_flow.abs().mean()
                gt_mag = gt_flow.abs().mean()
                
                # Delta xi from solver
                delta_xi = result.get('delta_xi', None)
                if delta_xi is not None:
                    rot_update = delta_xi[0, 3:].norm().item() * 180.0 / math.pi
                    trans_update = delta_xi[0, :3].norm().item() * 100
                else:
                    rot_update = 0
                    trans_update = 0
                
                # Apply update
                if delta_xi is not None:
                    pose_cur = se3_exp(delta_xi) @ pose_cur
                
                # New error
                pred_c2w = torch.inverse(pose_cur[0])
                gt_c2w = torch.inverse(pose_gt[0])
                R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
                trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
                cos_a = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
                rot_err = torch.acos(cos_a).item() * 180.0 / math.pi
                
                conf_mean = pred_conf.mean().item()
                conf_std = pred_conf.std().item()
                
                print(f"  Iter {outer_iter}: flow MAE={mae:.3f}px, "
                      f"|pred|={pred_mag:.3f} |gt|={gt_mag:.3f}, "
                      f"conf={conf_mean:.3f}±{conf_std:.3f}, "
                      f"Δrot={rot_update:.3f}°, after={rot_err:.2f}°")
            print()

if __name__ == '__main__':
    main()
