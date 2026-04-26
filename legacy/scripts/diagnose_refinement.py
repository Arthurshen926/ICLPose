#!/usr/bin/env python3
"""
Diagnostic script: analyze pose refinement convergence for a single test image.
Logs per-iteration loss, gradient norm, and pose error to identify bottlenecks.
"""

import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import se3_exp, render_features_transformed


def pose_error(pred_w2c, gt_w2c):
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)
    pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100.0
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_err = torch.acos(cos_angle).item() * 180.0 / math.pi
    return pos_err, rot_err


def refine_with_logging(
    feat_2d, means3d, quats, scales, opacities, colors,
    init_viewmat, K_mat, width, height, gt_w2c,
    n_iters=200, lr=0.01, stage_name="coarse", loss_type='mse',
):
    """Refine pose with per-step logging."""
    device = init_viewmat.device
    delta_xi = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta_xi], lr=lr)

    means3d = means3d.detach()
    quats = quats.detach()
    scales = scales.detach()
    opacities = opacities.detach()
    colors = colors.detach()
    feat_2d = feat_2d.detach()
    init_viewmat = init_viewmat.detach()
    K_unsq = K_mat.unsqueeze(0).detach()

    best_loss = float('inf')
    best_viewmat = init_viewmat.clone()
    log = []

    for i in range(n_iters):
        optimizer.zero_grad()
        delta_T = se3_exp(delta_xi)
        feat_3d = render_features_transformed(
            means3d, quats, scales, opacities, colors,
            delta_T, init_viewmat, K_unsq, width, height, 16,
        )
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        
        # Also compute cosine similarity
        cos_sim = (feat_3d_norm * feat_2d).sum(dim=1).mean().item()
        
        if loss_type == 'huber':
            loss = F.smooth_l1_loss(feat_3d_norm, feat_2d)
        else:
            loss = F.mse_loss(feat_3d_norm, feat_2d)
        loss.backward()
        grad_norm = delta_xi.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
        optimizer.step()

        cur_viewmat = (se3_exp(delta_xi.detach()) @ init_viewmat)
        pos_err, rot_err = pose_error(cur_viewmat, gt_w2c)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_viewmat = cur_viewmat.clone()

        entry = {
            'iter': i, 'loss': loss.item(), 'cos_sim': cos_sim,
            'grad_norm': grad_norm, 'pos_cm': pos_err, 'rot_deg': rot_err,
            'delta_xi': delta_xi.detach().cpu().tolist(),
        }
        log.append(entry)
        if i % 20 == 0 or i == n_iters - 1:
            print(f"  [{stage_name} {i:3d}/{n_iters}] loss={loss.item():.6f} cos={cos_sim:.4f} "
                  f"|∇|={grad_norm:.6f} pos={pos_err:.1f}cm rot={rot_err:.3f}° "
                  f"ξ=[{delta_xi[0].item():.4f},{delta_xi[1].item():.4f},{delta_xi[2].item():.4f},"
                  f"{delta_xi[3].item():.4f},{delta_xi[4].item():.4f},{delta_xi[5].item():.4f}]")

    best_pos, best_rot = pose_error(best_viewmat, gt_w2c)
    return best_viewmat, log, best_pos, best_rot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cameras_json', type=str, required=True)
    parser.add_argument('--test_idx', type=int, default=0, help='Index of test image')
    parser.add_argument('--render_height', type=int, default=540)
    parser.add_argument('--render_width', type=int, default=960)
    parser.add_argument('--fine_render_height', type=int, default=270)
    parser.add_argument('--fine_render_width', type=int, default=480)
    parser.add_argument('--coarse_iters', type=int, default=200)
    parser.add_argument('--fine_iters', type=int, default=200)
    parser.add_argument('--coarse_lr', type=float, default=0.01)
    parser.add_argument('--fine_lr', type=float, default=0.005)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'huber'])
    parser.add_argument('--rounds', type=int, default=1, help='Number of coarse-fine rounds')
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = argparse.Namespace(**ckpt['args'])

    # Gaussian model
    gs_model = GaussianFeatureModel(feature_dim=ckpt_args.feature_dim)
    gs_model.load_ply(args.model_path)
    gs_model = gs_model.to(device)
    gs_model.eval()
    means3d = gs_model.get_xyz.detach()
    quats = gs_model.get_rotation.detach()
    scales_raw = gs_model.get_scaling
    scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
    opacities = gs_model.get_opacity.squeeze(-1).detach()
    scene_extent = ckpt['scene_extent']

    # Triplane
    triplane = DualScaleTriplane(
        coarse_resolution=ckpt_args.coarse_resolution,
        fine_resolution=ckpt_args.fine_resolution,
        feature_dim=ckpt_args.feature_dim,
        scene_extent=scene_extent,
    ).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()

    # Encoder (handle old vs new architecture)
    encoder_state = ckpt['encoder']
    use_old = any('fine_encoder.encoder.' in k for k in encoder_state.keys())
    encoder = DualScaleEncoder(
        feature_dim=ckpt_args.feature_dim,
        freeze_backbone=True,
        use_old_fine_encoder=use_old,
    ).to(device)
    encoder.load_state_dict(encoder_state)
    if use_old:
        print("  Using old FineEncoder (v1 checkpoint)")
    encoder.eval()

    # Cameras
    with open(args.cameras_json) as f:
        all_cams = json.load(f)
    cam_by_name = {c['img_name']: c for c in all_cams}

    source_dir = Path(args.source_dir)
    test_file = source_dir / 'dataset_test.txt'
    test_samples = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            parts = line.split()
            test_samples.append({'img_name': parts[0]})

    train_file = source_dir / 'dataset_train.txt'
    train_names = set()
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            train_names.add(line.split()[0])
    train_cams = [c for c in all_cams if c['img_name'] in train_names]
    train_positions = np.array([c['position'] for c in train_cams])

    # Camera intrinsics
    first_cam = all_cams[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']
    render_h, render_w = args.render_height, args.render_width
    fx = first_cam['fx'] * render_w / orig_w
    fy = first_cam['fy'] * render_h / orig_h

    K_mat = torch.zeros(3, 3, device=device)
    K_mat[0, 0] = fx; K_mat[1, 1] = fy
    K_mat[0, 2] = render_w / 2.0; K_mat[1, 2] = render_h / 2.0
    K_mat[2, 2] = 1.0

    coarse_h = render_h // 14
    coarse_w = render_w // 14
    K_coarse = K_mat.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h

    fine_h = min(render_h, args.fine_render_height)
    fine_w = min(render_w, args.fine_render_width)
    K_fine = K_mat.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Pre-extract triplane features
    with torch.no_grad():
        coarse_colors = triplane.extract_coarse(means3d)
        coarse_colors = F.normalize(coarse_colors, p=2, dim=1)
        fine_colors = triplane.extract_fine(means3d)
        fine_colors = F.normalize(fine_colors, p=2, dim=1)

    # Pick test sample
    sample = test_samples[args.test_idx]
    img_name = sample['img_name']
    print(f"\n=== Diagnosing: {img_name} (idx={args.test_idx}) ===")

    img_path = source_dir / img_name
    if not img_path.exists():
        img_path = source_dir / 'processed' / img_name
    img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_norm = (img_tensor.unsqueeze(0).to(device) - MEAN) / STD

    # GT pose
    cam = cam_by_name[img_name]
    gt_c2w = np.eye(4, dtype=np.float32)
    gt_c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
    gt_c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
    gt_w2c = torch.from_numpy(np.linalg.inv(gt_c2w).astype(np.float32)).to(device)

    # Init pose (position-NN)
    test_pos = np.array(cam['position'])
    dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
    nn_idx = np.argmin(dists)
    nn_cam = train_cams[nn_idx]
    init_c2w = np.eye(4, dtype=np.float32)
    init_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
    init_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
    init_w2c = torch.from_numpy(np.linalg.inv(init_c2w).astype(np.float32)).to(device)

    init_pos, init_rot = pose_error(init_w2c, gt_w2c)
    print(f"Init error: pos={init_pos:.1f}cm rot={init_rot:.2f}°")

    # 2D features
    with torch.no_grad():
        coarse_feat_2d, fine_feat_2d = encoder(img_norm)
        coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
        fine_feat_2d = F.normalize(fine_feat_2d, p=2, dim=1)
    coarse_feat_2d = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
    fine_feat_2d = F.interpolate(fine_feat_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)

    # Also compute feature match quality at GT pose
    with torch.no_grad():
        from gsff.pose_refine import render_features_for_pose
        gt_vm = gt_w2c.unsqueeze(0)
        gt_coarse = render_features_for_pose(means3d, quats, scales, opacities, coarse_colors,
                                              gt_vm, K_coarse.unsqueeze(0), coarse_w, coarse_h)
        gt_coarse_n = F.normalize(gt_coarse, p=2, dim=1)
        gt_cos_coarse = (gt_coarse_n * coarse_feat_2d).sum(dim=1).mean().item()
        gt_loss_coarse = F.mse_loss(gt_coarse_n, coarse_feat_2d).item()

        gt_fine = render_features_for_pose(means3d, quats, scales, opacities, fine_colors,
                                            gt_vm, K_fine.unsqueeze(0), fine_w, fine_h)
        gt_fine_n = F.normalize(gt_fine, p=2, dim=1)
        gt_cos_fine = (gt_fine_n * fine_feat_2d).sum(dim=1).mean().item()
        gt_loss_fine = F.mse_loss(gt_fine_n, fine_feat_2d).item()

    print(f"Feature quality @ GT:  coarse cos={gt_cos_coarse:.4f} loss={gt_loss_coarse:.6f}")
    print(f"                       fine   cos={gt_cos_fine:.4f} loss={gt_loss_fine:.6f}")

    # Stage 1: Coarse
    print(f"\n--- Stage 1: Coarse ({coarse_w}×{coarse_h}, lr={args.coarse_lr}, iters={args.coarse_iters}) ---")
    refined_w2c, coarse_log, c_pos, c_rot = refine_with_logging(
        coarse_feat_2d, means3d, quats, scales, opacities, coarse_colors,
        init_w2c, K_coarse, coarse_w, coarse_h, gt_w2c,
        n_iters=args.coarse_iters, lr=args.coarse_lr, stage_name="coarse",
        loss_type=args.loss_type,
    )
    print(f"  → Best coarse: pos={c_pos:.1f}cm rot={c_rot:.3f}°")

    # Stage 2: Fine
    print(f"\n--- Stage 2: Fine ({fine_w}×{fine_h}, lr={args.fine_lr}, iters={args.fine_iters}) ---")
    final_w2c, fine_log, f_pos, f_rot = refine_with_logging(
        fine_feat_2d, means3d, quats, scales, opacities, fine_colors,
        refined_w2c, K_fine, fine_w, fine_h, gt_w2c,
        n_iters=args.fine_iters, lr=args.fine_lr, stage_name="fine",
        loss_type=args.loss_type,
    )
    print(f"  → Best fine: pos={f_pos:.1f}cm rot={f_rot:.3f}°")

    # Stage 3-4: Second round (if requested)
    if args.rounds > 1:
        print(f"\n--- Stage 3: Coarse Round 2 (from fine result) ---")
        r2_coarse, _, r2c_pos, r2c_rot = refine_with_logging(
            coarse_feat_2d, means3d, quats, scales, opacities, coarse_colors,
            final_w2c, K_coarse, coarse_w, coarse_h, gt_w2c,
            n_iters=args.coarse_iters, lr=args.coarse_lr * 0.5, stage_name="coarse2",
            loss_type=args.loss_type,
        )
        print(f"  → Best coarse R2: pos={r2c_pos:.1f}cm rot={r2c_rot:.3f}°")

        print(f"\n--- Stage 4: Fine Round 2 ---")
        r2_fine, _, r2f_pos, r2f_rot = refine_with_logging(
            fine_feat_2d, means3d, quats, scales, opacities, fine_colors,
            r2_coarse, K_fine, fine_w, fine_h, gt_w2c,
            n_iters=args.fine_iters, lr=args.fine_lr * 0.5, stage_name="fine2",
            loss_type=args.loss_type,
        )
        print(f"  → Best fine R2: pos={r2f_pos:.1f}cm rot={r2f_rot:.3f}°")
        f_pos, f_rot = r2f_pos, r2f_rot

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for {img_name}:")
    print(f"  Init:    {init_pos:.1f}cm / {init_rot:.2f}°")
    print(f"  Coarse:  {c_pos:.1f}cm / {c_rot:.3f}°")
    print(f"  Fine:    {f_pos:.1f}cm / {f_rot:.3f}°")
    print(f"  GT feat: coarse cos={gt_cos_coarse:.3f}, fine cos={gt_cos_fine:.3f}")
    print(f"{'='*60}")

    # Check: what's the loss at GT vs best?
    print(f"\n  Loss landscape check:")
    print(f"    Coarse loss @ GT:       {gt_loss_coarse:.6f}")
    print(f"    Coarse loss @ best:     {coarse_log[-1]['loss']:.6f}")
    print(f"    Fine loss @ GT:         {gt_loss_fine:.6f}")
    print(f"    Fine loss @ best:       {fine_log[-1]['loss']:.6f}")
    is_gt_lower = gt_loss_coarse < min(e['loss'] for e in coarse_log)
    print(f"    GT loss IS global min?  {'YES' if is_gt_lower else 'NO (GT loss > optimized loss!)'}")


if __name__ == '__main__':
    main()
