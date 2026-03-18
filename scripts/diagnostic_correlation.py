#!/usr/bin/env python3
"""
Diagnostic: Test raw correlation quality for OldHospital.

Skip all learned flow heads — just compute dot-product correlation between
query and rendered features, take argmax, and convert to pixel flow.
Then use Image Jacobian + WLS to compute pose update.

This tells us the theoretical upper bound of correlation-based matching.
If argmax correlation gives good flow → the flow heads are the bottleneck.
If argmax correlation gives bad flow → the features/rendering are the bottleneck.
"""
import argparse, yaml, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from modules.geometry_solver import diff_pose_solve, compute_image_jacobian
from data.dataset_v4 import PoseDatasetV4, collate_v4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--noise_deg', type=float, default=5.0)
    p.add_argument('--noise_m', type=float, default=0.5)
    p.add_argument('--outer_iters', type=int, default=5)
    p.add_argument('--scale', type=str, default='fine_dino',
                   choices=['coarse', 'mid', 'fine_sd', 'fine_dino'])
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--damping', type=float, default=0.01)
    return p.parse_args()


def correlation_to_flow(corr, H, W):
    """Convert a correlation volume to flow by soft argmax.
    
    corr: (B, H*W, H*W) — softmaxed correlation
    Returns flow: (B, 2, H, W)
    """
    B = corr.shape[0]
    device = corr.device
    
    # Create coordinate grids
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    coords = torch.stack([u_coords.reshape(-1), v_coords.reshape(-1)], dim=-1)  # (H*W, 2)
    
    # Soft argmax: expected coordinates
    # corr is (B, H*W_q, H*W_r), softmax over last dim (reference)
    expected_coords = torch.bmm(corr, coords.unsqueeze(0).expand(B, -1, -1))  # (B, H*W, 2)
    
    # Flow = expected_ref_coord - query_coord
    query_coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
    flow = expected_coords - query_coords  # (B, H*W, 2)
    
    # Reshape to (B, 2, H, W)
    flow = flow.permute(0, 2, 1).reshape(B, 2, H, W)
    return flow


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda')
    rc = cfg['renderer']
    dc = cfg['data']
    mc = cfg['model']

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

    # Dataset
    full_ds = PoseDatasetV4(
        feature_base_dir=dc['train_feature_dir'],
        traj_path=dc.get('train_traj_path'),
        depth_dir=dc.get('train_depth_dir'),
        noise_rot_deg=args.noise_deg,
        noise_trans_m=args.noise_m,
        is_train=True,
    )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    _, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=collate_v4)

    # Cache batches for determinism
    torch.manual_seed(12345)
    np.random.seed(12345)
    batches = []
    for batch in loader:
        batches.append({
            'query_feats': {k: v.to(device) for k, v in batch['query_feats'].items()},
            'pose_gt': batch['pose_gt'].to(device).float(),
            'initial_pose': batch['initial_pose'].to(device).float(),
        })

    # Intrinsics
    fx, fy = rc.get('fx', 320.0), rc.get('fy', 320.0)
    cx, cy = rc.get('cx', 319.5), rc.get('cy', 239.5)
    img_h, img_w = rc.get('img_height', 480), rc.get('img_width', 640)

    print(f"Config: {args.config}")
    print(f"Scale: {args.scale}, Temperature: {args.temperature}")
    print(f"Noise: {args.noise_deg}° / {args.noise_m}m")
    print(f"Outer iters: {args.outer_iters}, Damping: {args.damping}")
    n_samples = sum(b['pose_gt'].shape[0] for b in batches)
    print(f"Val samples: {n_samples}")
    print("=" * 60)

    # First: measure initial error (no update)
    all_rot_init = []
    for batch in batches:
        pose_init = batch['initial_pose']
        pose_gt = batch['pose_gt']
        R_rel = torch.bmm(pose_init[:, :3, :3].transpose(1, 2), pose_gt[:, :3, :3])
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        all_rot_init.extend((torch.acos(cos_angle) * 180.0 / math.pi).cpu().tolist())
    
    print(f"Initial noise: {np.mean(all_rot_init):.2f}° (med {np.median(all_rot_init):.2f}°)")

    # Now: iterative refinement using raw correlation flow  
    for oi_max in [1, 3, 5]:
        all_rot = []
        all_trans = []

        for batch in tqdm(batches, desc=f"oi={oi_max}", leave=False):
            query_feats = batch['query_feats']
            pose_gt = batch['pose_gt']
            pose_cur = batch['initial_pose'].clone()

            for oi in range(oi_max):
                with torch.cuda.amp.autocast(enabled=True):
                    # Render features at current pose
                    scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
                    render_out = renderer.render_batch(pose_cur, scales=scales, return_depth=True)
                    
                    # Get query and reference features for chosen scale
                    q_feat = query_feats[args.scale].float()   # (B, C, H, W)
                    r_feat = render_out[f'{args.scale}_feat'].float()  # (B, C, H, W)
                    depth = render_out.get('depth_map')  # (B, Hd, Wd)
                    
                    B, C, H, W = q_feat.shape
                    
                    # L2 normalize
                    q_norm = F.normalize(q_feat, dim=1)
                    r_norm = F.normalize(r_feat, dim=1)
                    
                    # Resize to matching resolution if needed
                    if q_norm.shape[-2:] != r_norm.shape[-2:]:
                        r_norm = F.interpolate(r_norm, size=q_norm.shape[-2:], mode='bilinear', align_corners=False)
                    
                    B, C, H, W = q_norm.shape
                    
                    # Compute correlation: (B, H*W, H*W)
                    q_flat = q_norm.reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
                    r_flat = r_norm.reshape(B, C, -1)                    # (B, C, HW)
                    corr = torch.bmm(q_flat, r_flat)  # (B, HW, HW)
                    corr = corr / args.temperature
                    corr = F.softmax(corr, dim=-1)
                    
                    # Correlation → flow via soft argmax
                    flow = correlation_to_flow(corr, H, W)
                    
                    # Resize depth to match flow resolution
                    if depth is not None and depth.shape[-2:] != (H, W):
                        depth_resized = F.interpolate(
                            depth.unsqueeze(1), size=(H, W), mode='nearest'
                        ).squeeze(1)
                    else:
                        depth_resized = depth
                    
                    # Scale intrinsics for this resolution
                    sx = W / img_w
                    sy = H / img_h
                    fx_s = fx * sx
                    fy_s = fy * sy
                    cx_s = cx * sx
                    cy_s = cy * sy

                # Compute Image Jacobian
                with torch.no_grad():
                    intr = {'fx': fx_s, 'fy': fy_s, 'cx': cx_s, 'cy': cy_s}
                    Ju, Jv, valid = compute_image_jacobian(depth_resized, intr)
                    
                    conf = torch.ones(B, 1, H, W, device=device)
                    
                    delta_xi = diff_pose_solve(
                        flow.float(), conf, Ju, Jv, valid,
                        damping=args.damping
                    )
                    
                    T_delta = se3_exp(delta_xi)
                    pose_cur = torch.bmm(T_delta, pose_cur)

            # Final metrics
            R_pred = pose_cur[:, :3, :3]
            R_gt = pose_gt[:, :3, :3]
            R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = torch.acos(cos_angle) * 180.0 / math.pi
            
            t_pred = pose_cur[:, :3, 3]
            t_gt = pose_gt[:, :3, 3]
            trans_err = torch.norm(t_pred - t_gt, dim=1) * 1000

            all_rot.extend(rot_err.cpu().tolist())
            all_trans.extend(trans_err.cpu().tolist())

        rot = np.array(all_rot)
        trans = np.array(all_trans)
        pct1 = float(np.mean(rot < 1.0) * 100)
        pct5 = float(np.mean(rot < 5.0) * 100)
        print(f"  oi={oi_max}: rot={np.mean(rot):.2f}° (med {np.median(rot):.2f}°) "
              f"trans={np.mean(trans):.0f}mm  <1°={pct1:.1f}%  <5°={pct5:.1f}%")
    
    # Also measure correlation quality
    print("\n--- Correlation Quality Analysis ---")
    with torch.no_grad():
        batch = batches[0]
        q_feat = batch['query_feats'][args.scale].float()
        pose_gt = batch['pose_gt']
        
        # Render at GT pose
        render_out_gt = renderer.render_batch(pose_gt, scales=[args.scale], return_depth=False)
        r_feat_gt = render_out_gt[f'{args.scale}_feat'].float()
        
        # Render at perturbed pose
        pose_noisy = batch['initial_pose']
        render_out_noisy = renderer.render_batch(pose_noisy, scales=[args.scale], return_depth=False)
        r_feat_noisy = render_out_noisy[f'{args.scale}_feat'].float()
        
        if r_feat_gt.shape[-2:] != q_feat.shape[-2:]:
            r_feat_gt = F.interpolate(r_feat_gt, size=q_feat.shape[-2:], mode='bilinear', align_corners=False)
            r_feat_noisy = F.interpolate(r_feat_noisy, size=q_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        B, C, H, W = q_feat.shape
        
        # Cosine similarity at GT pose (should be high)
        q_n = F.normalize(q_feat, dim=1)
        r_gt_n = F.normalize(r_feat_gt, dim=1)
        r_noisy_n = F.normalize(r_feat_noisy, dim=1)
        
        cos_gt = (q_n * r_gt_n).sum(dim=1).mean()
        cos_noisy = (q_n * r_noisy_n).sum(dim=1).mean()
        
        print(f"  Query vs Rendered@GT:    cosine={cos_gt.item():.4f}")
        print(f"  Query vs Rendered@Noisy: cosine={cos_noisy.item():.4f}")
        print(f"  Difference:              {(cos_gt - cos_noisy).item():.4f}")
        
        # Correlation peak quality at GT pose
        q_flat = q_n.reshape(B, C, -1).permute(0, 2, 1)
        r_gt_flat = r_gt_n.reshape(B, C, -1)
        corr_gt = torch.bmm(q_flat, r_gt_flat)  # (B, HW, HW)
        
        # Check if diagonal is the peak (should be if features are distinctive)
        diag = torch.diagonal(corr_gt, dim1=1, dim2=2)  # (B, HW) — identity correspondence
        max_vals, max_ids = corr_gt.max(dim=-1)  # max over reference
        identity_ids = torch.arange(H*W, device=device).unsqueeze(0).expand(B, -1)
        pct_correct = (max_ids == identity_ids).float().mean() * 100
        print(f"  Correct matches @GT:     {pct_correct.item():.1f}%")
        print(f"  Mean diagonal corr:      {diag.mean().item():.4f}")
        print(f"  Mean max corr:           {max_vals.mean().item():.4f}")


if __name__ == '__main__':
    main()
