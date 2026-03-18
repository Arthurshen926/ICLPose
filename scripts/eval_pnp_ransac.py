#!/usr/bin/env python3
"""Evaluate MSFlowPoseNet with PnP+RANSAC pose solver instead of WLS.

Instead of using the Image Jacobian weighted-least-squares solver, this script:
1. Runs the model to get flow + confidence predictions
2. Back-projects rendered pixels to 3D world coordinates using depth + current pose
3. Computes 2D target positions from flow predictions
4. Solves for the new pose using cv2.solvePnPRansac

This should be more robust to systematic flow errors in repetitive textures (OldHospital).

Usage:
    python scripts/eval_pnp_ransac.py --config configs/exp138_oh_aggressive.yaml \
        --checkpoint output/exp138_oh_aggressive/checkpoints/best.pth \
        --gpu 2 --iters 1 3 5 10
"""
import sys, yaml, torch, math, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def pnp_ransac_solve(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    depth: torch.Tensor,
    pose_current: torch.Tensor,
    intrinsics_feat: dict,
    intrinsics_full: dict,
    img_hw: tuple,
    conf_threshold: float = 0.3,
    ransac_iters: int = 1000,
    reproj_threshold: float = 2.0,
) -> torch.Tensor:
    """
    PnP+RANSAC pose solver from flow predictions.

    Args:
        flow: (B, 2, H, W) predicted flow at feature resolution
        confidence: (B, 1, H, W) per-pixel confidence
        depth: (B, H, W) rendered depth at feature resolution
        pose_current: (B, 4, 4) current w2c pose
        intrinsics_feat: {'fx','fy','cx','cy'} at feature resolution
        intrinsics_full: {'fx','fy','cx','cy'} at full image resolution
        img_hw: (H_full, W_full) full image resolution
        conf_threshold: minimum confidence to use a pixel
        ransac_iters: RANSAC iterations
        reproj_threshold: reprojection error threshold for inliers (pixels at feature res)

    Returns:
        pose_new: (B, 4, 4) solved w2c pose
    """
    B = flow.shape[0]
    H, W = flow.shape[2], flow.shape[3]
    device = flow.device

    fx_f = intrinsics_feat['fx']
    fy_f = intrinsics_feat['fy']
    cx_f = intrinsics_feat['cx']
    cy_f = intrinsics_feat['cy']

    # Full-resolution camera matrix for PnP
    fx_full = intrinsics_full['fx']
    fy_full = intrinsics_full['fy']
    cx_full = intrinsics_full['cx']
    cy_full = intrinsics_full['cy']
    K_full = np.array([
        [fx_full, 0, cx_full],
        [0, fy_full, cy_full],
        [0, 0, 1]
    ], dtype=np.float64)

    H_full, W_full = img_hw
    scale_w = W_full / W
    scale_h = H_full / H

    results = []

    for b in range(B):
        # Get per-pixel data
        flow_b = flow[b].cpu().numpy()         # (2, H, W)
        conf_b = confidence[b, 0].cpu().numpy()  # (H, W)
        depth_b = depth[b].cpu().numpy()       # (H, W)
        T_cur = pose_current[b].cpu().numpy()  # (4, 4) w2c

        # Create pixel grid
        v_coords, u_coords = np.meshgrid(
            np.arange(H, dtype=np.float64),
            np.arange(W, dtype=np.float64),
            indexing='ij')

        # Valid pixels: good depth and confidence
        valid = (depth_b > 0.05) & (conf_b > conf_threshold)
        valid_idx = np.where(valid.ravel())[0]

        if len(valid_idx) < 10:
            # Not enough points, keep current pose
            results.append(pose_current[b])
            continue

        u_flat = u_coords.ravel()[valid_idx]
        v_flat = v_coords.ravel()[valid_idx]
        Z_flat = depth_b.ravel()[valid_idx]
        du_flat = flow_b[0].ravel()[valid_idx]
        dv_flat = flow_b[1].ravel()[valid_idx]

        # Back-project to 3D camera coordinates at current pose
        X_cam = Z_flat * (u_flat - cx_f) / fx_f
        Y_cam = Z_flat * (v_flat - cy_f) / fy_f
        pts_cam = np.stack([X_cam, Y_cam, Z_flat, np.ones_like(Z_flat)], axis=1)  # (N, 4)

        # Transform to world coordinates
        T_cur_inv = np.linalg.inv(T_cur)  # c2w
        pts_world = (T_cur_inv @ pts_cam.T).T[:, :3]  # (N, 3)

        # 2D target positions in query image (at feature resolution → scale to full)
        u_target = (u_flat + du_flat) * scale_w
        v_target = (v_flat + dv_flat) * scale_h
        pts_2d = np.stack([u_target, v_target], axis=1)  # (N, 2)

        # If too many points, randomly subsample for speed
        max_pts = 5000
        if len(pts_world) > max_pts:
            # Weight by confidence for better sampling
            conf_vals = conf_b.ravel()[valid_idx]
            probs = conf_vals / conf_vals.sum()
            idx = np.random.choice(len(pts_world), max_pts, replace=False, p=probs)
            pts_world = pts_world[idx]
            pts_2d = pts_2d[idx]

        # PnP+RANSAC
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_world.astype(np.float64),
                pts_2d.astype(np.float64),
                K_full,
                distCoeffs=None,
                iterationsCount=ransac_iters,
                reprojectionError=reproj_threshold * max(scale_w, scale_h),
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception as e:
            results.append(pose_current[b])
            continue

        if not success or inliers is None or len(inliers) < 6:
            results.append(pose_current[b])
            continue

        # Refine with inliers only using iterative PnP
        try:
            success2, rvec, tvec = cv2.solvePnP(
                pts_world[inliers.ravel()].astype(np.float64),
                pts_2d[inliers.ravel()].astype(np.float64),
                K_full, None,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except:
            pass

        # Convert to 4x4 w2c matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        T_new = np.eye(4, dtype=np.float64)
        T_new[:3, :3] = R_mat
        T_new[:3, 3] = tvec.ravel()

        results.append(torch.tensor(T_new, dtype=torch.float32, device=device))

    return torch.stack(results, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--iters', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--conf-threshold', type=float, default=0.3)
    parser.add_argument('--ransac-iters', type=int, default=1000)
    parser.add_argument('--reproj-threshold', type=float, default=2.0)
    parser.add_argument('--compare-wls', action='store_true',
                        help='Also run WLS solver for comparison')
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

    # Get intrinsics at different resolutions
    fine_hw = tuple(mc.get('fine_hw', [35, 46]))
    img_h = rc.get('img_height', 480)
    img_w = rc.get('img_width', 640)
    base_fx = rc.get('fx', 320.0)
    base_fy = rc.get('fy', 320.0)
    base_cx = rc.get('cx', 319.5)
    base_cy = rc.get('cy', 239.5)

    intrinsics_feat = {
        'fx': base_fx * fine_hw[1] / img_w,
        'fy': base_fy * fine_hw[0] / img_h,
        'cx': base_cx * fine_hw[1] / img_w,
        'cy': base_cy * fine_hw[0] / img_h,
    }
    intrinsics_full = {'fx': base_fx, 'fy': base_fy, 'cx': base_cx, 'cy': base_cy}

    # Check for geometry_upsample
    geo_up = mc.get('geometry_upsample', 1)
    if geo_up > 1:
        up_hw = (fine_hw[0] * geo_up, fine_hw[1] * geo_up)
        intrinsics_feat = {
            'fx': base_fx * up_hw[1] / img_w,
            'fy': base_fy * up_hw[0] / img_h,
            'cx': base_cx * up_hw[1] / img_w,
            'cy': base_cy * up_hw[0] / img_h,
        }
        fine_hw = up_hw

    print(f"  Feature resolution: {fine_hw}")
    print(f"  Feature intrinsics: fx={intrinsics_feat['fx']:.2f} fy={intrinsics_feat['fy']:.2f}")
    print(f"  Full intrinsics: fx={intrinsics_full['fx']:.2f} fy={intrinsics_full['fy']:.2f}")

    # Val dataset
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
    print(f"  PnP+RANSAC Evaluation")
    print(f"  conf_threshold={args.conf_threshold}, ransac_iters={args.ransac_iters}, "
          f"reproj_threshold={args.reproj_threshold}")
    print(f"  Testing outer iterations: {args.iters}")
    if args.compare_wls:
        print(f"  Also comparing with WLS solver")
    print(f"{'='*70}\n")

    for num_iters in args.iters:
        all_rot_pnp, all_trans_pnp = [], []
        all_rot_wls, all_trans_wls = [], []
        all_inlier_ratios = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"oi={num_iters:2d}", leave=False):
                qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
                pose_gt = batch['pose_gt'].to(device)
                pose_pnp = batch['initial_pose'].to(device).float()
                pose_wls = pose_pnp.clone() if args.compare_wls else None

                for oi in range(num_iters):
                    # Render at current PnP pose
                    res = renderer.render_batch(pose_pnp,
                        scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                        return_depth=True)
                    rf = {'coarse': res['coarse_feat'], 'mid': res['mid_feat'],
                          'fine_sd': res['fine_sd_feat'], 'fine_dino': res['fine_dino_feat']}
                    depth = res.get('depth_map')

                    with torch.cuda.amp.autocast(enabled=True):
                        pred = model(qf, rf, depth)

                    # Get flow and confidence at fine resolution
                    flow_fine = pred['flow_fine'].float()
                    conf_fine = pred['conf_fine'].float()

                    # Resize depth to match flow if needed
                    if depth is not None:
                        depth_f = depth.float()
                        if depth_f.shape[-2:] != flow_fine.shape[-2:]:
                            need_sq = (depth_f.ndim == 3)
                            if need_sq:
                                depth_f = depth_f.unsqueeze(1)
                            depth_f = F.interpolate(
                                depth_f, size=flow_fine.shape[-2:],
                                mode='bilinear', align_corners=False)
                            if need_sq:
                                depth_f = depth_f.squeeze(1)

                        # Apply geometry_upsample if needed
                        if geo_up > 1:
                            tgt = (flow_fine.shape[-2] * geo_up, flow_fine.shape[-1] * geo_up)
                            flow_fine = F.interpolate(flow_fine, size=tgt,
                                mode='bilinear', align_corners=False) * geo_up
                            conf_fine = F.interpolate(conf_fine, size=tgt,
                                mode='bilinear', align_corners=False)
                            if depth_f.ndim == 3:
                                depth_f = depth_f.unsqueeze(1)
                            depth_f = F.interpolate(depth_f, size=tgt,
                                mode='bilinear', align_corners=False)
                            if depth_f.ndim == 4:
                                depth_f = depth_f.squeeze(1)

                    # PnP+RANSAC solve
                    pose_pnp = pnp_ransac_solve(
                        flow_fine, conf_fine, depth_f, pose_pnp,
                        intrinsics_feat, intrinsics_full, (img_h, img_w),
                        conf_threshold=args.conf_threshold,
                        ransac_iters=args.ransac_iters,
                        reproj_threshold=args.reproj_threshold,
                    )

                    # WLS solve for comparison
                    if args.compare_wls:
                        res_wls = renderer.render_batch(pose_wls,
                            scales=['coarse', 'mid', 'fine_sd', 'fine_dino'],
                            return_depth=True)
                        rf_wls = {'coarse': res_wls['coarse_feat'], 'mid': res_wls['mid_feat'],
                                  'fine_sd': res_wls['fine_sd_feat'], 'fine_dino': res_wls['fine_dino_feat']}
                        depth_wls = res_wls.get('depth_map')
                        with torch.cuda.amp.autocast(enabled=True):
                            pred_wls = model(qf, rf_wls, depth_wls)
                        if 'delta_xi' in pred_wls:
                            T = se3_exp(pred_wls['delta_xi'].float())
                            pose_wls = torch.bmm(T, pose_wls.float())

                # Compute PnP errors
                Rr = torch.bmm(pose_pnp[:, :3, :3].float().transpose(1, 2),
                               pose_gt.float()[:, :3, :3])
                tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                re = torch.acos(ca) * 180 / math.pi
                te = torch.norm(pose_pnp[:, :3, 3].float() - pose_gt.float()[:, :3, 3],
                               dim=1) * 1000
                all_rot_pnp.extend(re.cpu().tolist())
                all_trans_pnp.extend(te.cpu().tolist())

                # Compute WLS errors
                if args.compare_wls:
                    Rr = torch.bmm(pose_wls[:, :3, :3].float().transpose(1, 2),
                                   pose_gt.float()[:, :3, :3])
                    tr = Rr[:, 0, 0] + Rr[:, 1, 1] + Rr[:, 2, 2]
                    ca = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
                    re = torch.acos(ca) * 180 / math.pi
                    te = torch.norm(pose_wls[:, :3, 3].float() - pose_gt.float()[:, :3, 3],
                                   dim=1) * 1000
                    all_rot_wls.extend(re.cpu().tolist())
                    all_trans_wls.extend(te.cpu().tolist())

        r_pnp = np.array(all_rot_pnp)
        t_pnp = np.array(all_trans_pnp)
        print(f"  [PnP]  oi={num_iters:2d}: rot={np.mean(r_pnp):.3f}° (med {np.median(r_pnp):.3f}°)  "
              f"trans={np.mean(t_pnp):.1f}mm (med {np.median(t_pnp):.1f}mm)  "
              f"<1°={np.mean(r_pnp < 1) * 100:.1f}%  <0.5°={np.mean(r_pnp < 0.5) * 100:.1f}%")

        if args.compare_wls:
            r_wls = np.array(all_rot_wls)
            t_wls = np.array(all_trans_wls)
            print(f"  [WLS]  oi={num_iters:2d}: rot={np.mean(r_wls):.3f}° (med {np.median(r_wls):.3f}°)  "
                  f"trans={np.mean(t_wls):.1f}mm (med {np.median(t_wls):.1f}mm)  "
                  f"<1°={np.mean(r_wls < 1) * 100:.1f}%  <0.5°={np.mean(r_wls < 0.5) * 100:.1f}%")
            # Show improvement
            improve_rot = np.median(r_wls) - np.median(r_pnp)
            print(f"         PnP improvement: {improve_rot:+.3f}° median rotation")
        print()


if __name__ == '__main__':
    main()
