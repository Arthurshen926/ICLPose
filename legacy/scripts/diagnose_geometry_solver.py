#!/usr/bin/env python3
"""
Geometry Solver Diagnostic for Stairs
======================================
Analyzes WHY flow EPE=0.3px → expected 0.23° but actual 1.25° (5× gap).

Tests:
1. GT flow → geometry solver accuracy (isolates depth/solver quality)
2. Per-iteration flow EPE and pose error (not averaged)
3. Jacobian condition number analysis
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.geometry_solver import compute_image_jacobian, diff_pose_solve
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader


def pose_error(pose_pred, pose_gt):
    """Compute rotation (degrees) and translation (mm) errors."""
    Rr = torch.bmm(pose_pred[:,:3,:3].float().transpose(1,2), pose_gt.float()[:,:3,:3])
    tr = Rr[:,0,0]+Rr[:,1,1]+Rr[:,2,2]
    ca = torch.clamp((tr-1)/2, -1+1e-7, 1-1e-7)
    rot_err = torch.acos(ca)*180/math.pi
    trans_err = torch.norm(pose_pred[:,:3,3].float()-pose_gt.float()[:,:3,3], dim=1)*1000
    return rot_err, trans_err


def analyze_gt_flow_vs_pred(model, renderer, val_loader, device, outer_iters=5):
    """CORE TEST: Feed GT flow to geometry solver and compare with pred flow."""
    print("\n" + "="*70)
    print("ANALYSIS: GT Flow → Geometry Solver vs Predicted Flow")
    print("If GT flow still gives bad pose → depth/solver is the bottleneck")
    print("="*70)
    
    per_iter_pred_rot = [[] for _ in range(outer_iters)]
    per_iter_gt_rot = [[] for _ in range(outer_iters)]
    per_iter_epe = [[] for _ in range(outer_iters)]
    per_iter_cond = [[] for _ in range(outer_iters)]
    
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
            pose_gt = batch['pose_gt'].to(device)
            pose_pred = batch['initial_pose'].to(device).clone()
            pose_gt_path = pose_gt.clone()  # For GT flow reference
            B = pose_gt.shape[0]
            
            for oi in range(outer_iters):
                # Render features + depth at current predicted pose
                res = renderer.render_batch(
                    pose_pred, scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                    return_depth=True)
                rf = {'coarse': res['coarse_feat'], 'mid': res['mid_feat'],
                      'fine_sd': res['fine_sd_feat'], 'fine_dino': res['fine_dino_feat']}
                depth = res.get('depth_map')  # (B, H, W)
                
                # Model forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(qf, rf, depth)
                
                flow_pred = pred.get('flow_fine')
                conf = pred.get('conf_fine')
                
                if flow_pred is None or depth is None:
                    continue
                
                fH, fW = flow_pred.shape[-2:]
                
                # depth_fine is the depth passed to model, resized to fine_hw if needed
                depth_fine = depth.float()
                if depth_fine.shape[-2:] != (fH, fW):
                    depth_fine = F.interpolate(
                        depth_fine.unsqueeze(1), size=(fH, fW),
                        mode='bilinear', align_corners=False).squeeze(1)
                
                # Compute GT flow using model's method (same as training)
                gt_result = model.compute_gt_flow(
                    pose_pred, pose_gt, depth_fine, (fH, fW))
                gt_flow = gt_result[0]  # (B, 2, fH, fW)
                valid_mask = gt_result[1]  # (B, 1, fH, fW)
                
                # Flow EPE
                mask_2d = valid_mask.squeeze(1) > 0  # (B, fH, fW)
                if mask_2d.any():
                    epe = (flow_pred.float() - gt_flow).norm(dim=1)[mask_2d].mean().item()
                    per_iter_epe[oi].append(epe)
                
                # --- Solve with PREDICTED flow (normal path) ---
                delta_xi_pred = pred['delta_xi']
                delta_T_pred = se3_exp(delta_xi_pred.float())
                pose_after_pred = torch.bmm(delta_T_pred, pose_pred.float())
                re_pred, te_pred = pose_error(pose_after_pred, pose_gt)
                per_iter_pred_rot[oi].extend(re_pred.cpu().tolist())
                
                # --- Solve with GT flow through geometry solver ---
                depth_f32 = depth_fine.float()
                flow_gt_f32 = gt_flow.float()
                conf_f32 = conf.float() if conf is not None else torch.ones(B, 1, fH, fW, device=device) * 0.5
                
                s = model.geometry_upsample
                if s > 1:
                    tgt = (fH * s, fW * s)
                    flow_gt_f32 = F.interpolate(flow_gt_f32, size=tgt, mode='bilinear', align_corners=False) * s
                    conf_f32 = F.interpolate(conf_f32, size=tgt, mode='bilinear', align_corners=False)
                    need_sq = depth_f32.ndim == 3
                    if need_sq:
                        depth_f32 = depth_f32.unsqueeze(1)
                    depth_f32 = F.interpolate(depth_f32, size=tgt, mode='bilinear', align_corners=False)
                    if need_sq:
                        depth_f32 = depth_f32.squeeze(1)
                
                Ju, Jv, valid = compute_image_jacobian(depth_f32, model.geo_intrinsics)
                
                delta_xi_gt = diff_pose_solve(
                    flow_gt_f32, conf_f32, Ju, Jv, valid,
                    damping=model.damping,
                    irls_iters=model.irls_iters,
                    irls_huber_k=model.irls_huber_k,
                    pixel_stride=model.pixel_stride,
                )
                
                delta_T_gt = se3_exp(delta_xi_gt.float())
                pose_after_gt = torch.bmm(delta_T_gt, pose_pred.float())
                re_gt, te_gt = pose_error(pose_after_gt, pose_gt)
                per_iter_gt_rot[oi].extend(re_gt.cpu().tolist())
                
                # Jacobian condition number
                J_full = torch.cat([Ju, Jv], dim=1)
                valid2 = valid.repeat(1, 2)
                for b in range(B):
                    J_b = J_full[b][valid2[b]]
                    if J_b.shape[0] >= 6:
                        _, S, _ = torch.linalg.svd(J_b.unsqueeze(0), full_matrices=False)
                        cond = (S[0, 0] / S[0, -1]).item()
                        per_iter_cond[oi].append(cond)
                
                # Update pose for next iteration
                pose_pred = pose_after_pred
    
    # Print results
    print(f"\n{'Iter':>4} | {'PredRot°':>9} | {'GTflowRot°':>11} | {'FlowEPE':>8} | {'JacCond':>10} | {'Gap':>5}")
    print("-" * 72)
    for oi in range(outer_iters):
        pred_med = np.median(per_iter_pred_rot[oi]) if per_iter_pred_rot[oi] else float('nan')
        gt_med = np.median(per_iter_gt_rot[oi]) if per_iter_gt_rot[oi] else float('nan')
        epe_med = np.median(per_iter_epe[oi]) if per_iter_epe[oi] else float('nan')
        cond_med = np.median(per_iter_cond[oi]) if per_iter_cond[oi] else float('nan')
        gap = pred_med / max(gt_med, 1e-6) if not np.isnan(gt_med) else float('nan')
        print(f"  {oi:>2}  | {pred_med:>8.3f}° | {gt_med:>10.3f}° | {epe_med:>7.3f}px | {cond_med:>10.1f} | {gap:>4.1f}×")
    
    # Summary
    final_pred = np.median(per_iter_pred_rot[-1]) if per_iter_pred_rot[-1] else float('nan')
    final_gt = np.median(per_iter_gt_rot[-1]) if per_iter_gt_rot[-1] else float('nan')
    print(f"\n  SUMMARY (final iteration):")
    print(f"    Predicted flow → solver: {final_pred:.3f}°")
    print(f"    GT flow → solver:        {final_gt:.3f}°")
    if final_gt > 0.3:
        print(f"    ⚠ GT flow still gives {final_gt:.3f}° → DEPTH/SOLVER is limiting!")
        print(f"    → Improving flow quality alone won't break through this floor")
    elif final_gt < 0.1:
        print(f"    ✓ GT flow → solver gives {final_gt:.3f}° → solver OK, FLOW is limiting")
    else:
        print(f"    ~ GT flow → solver gives {final_gt:.3f}° → both flow and solver matter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outer_iters', type=int, default=5)
    parser.add_argument('--noise_deg', type=float, default=5.0)
    parser.add_argument('--noise_m', type=float, default=0.15)
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
        irls_iters=mc.get('irls_iters', 0), irls_huber_k=mc.get('irls_huber_k', 1.345),
        geometry_upsample=mc.get('geometry_upsample', 1),
        multiscale_consistency=mc.get('multiscale_consistency', False),
        ms_consistency_sigma=mc.get('ms_consistency_sigma', 1.0),
        pixel_stride=mc.get('pixel_stride', 1),
        positional_encoding=mc.get('positional_encoding', False),
        pe_mode=mc.get('pe_mode', 'concat'), pe_dim=mc.get('pe_dim', 32),
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
    print(f"[Model] Fine HW: {model.FINE_HW}, geo_upsample: {model.geometry_upsample}")
    print(f"[Model] Damping: {model.damping}, IRLS: {model.irls_iters}")

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
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=collate_v4, pin_memory=True)
    print(f"[Data] Val: {len(val_ds)} samples, noise: {args.noise_deg}°/{args.noise_m}m")

    analyze_gt_flow_vs_pred(model, renderer, val_loader, device, args.outer_iters)

    print("\nDIAGNOSTIC COMPLETE")


if __name__ == '__main__':
    main()
