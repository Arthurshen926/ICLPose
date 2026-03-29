#!/usr/bin/env python3
"""
Analyze failure cases in GSFFs evaluation.
For each test sample, compute init error, final error, and categorize failures.
Uses 3R no-lr-decay config (current best single-start).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import refine_pose
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def cam_to_w2c(cam):
    R = torch.tensor(cam['rotation'], dtype=torch.float32)
    pos = torch.tensor(cam['position'], dtype=torch.float32)
    w2c = torch.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ pos
    return w2c


def pose_error(pred_w2c, gt_w2c):
    R_rel = gt_w2c[:3, :3] @ pred_w2c[:3, :3].T
    trace = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
    rot = torch.acos(trace).item() * 180 / np.pi
    pos = torch.norm(
        torch.inverse(gt_w2c)[:3, 3] - torch.inverse(pred_w2c)[:3, 3]
    ).item() * 100
    return pos, rot


def main():
    device = torch.device('cuda')
    
    ckpt_path = 'output/gsff/OldHospital/checkpoints/final.pth'
    ply_path = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
    cameras_json = 'output/2dgs_models/OldHospital/v7_depth/cameras.json'
    source_dir = 'dataset/OldHospital'
    
    print("Loading model...", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    scene_extent = ckpt.get('scene_extent', 70.0)
    enc_keys = list(ckpt['encoder'].keys())
    use_old = any('fine_encoder.encoder.' in k for k in enc_keys)
    
    triplane = DualScaleTriplane(256, 1024, 16, scene_extent).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()
    
    encoder = DualScaleEncoder(16, freeze_backbone=True, use_old_fine_encoder=use_old).to(device)
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder.eval()
    
    gs = GaussianFeatureModel(feature_dim=16)
    gs.load_ply(ply_path)
    gs = gs.cuda()
    
    means3d = gs._xyz.detach().to(device)
    quats = gs.get_rotation.detach().to(device)
    scales_raw = gs.get_scaling.detach().to(device)
    if scales_raw.shape[1] == 2:
        scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1)
    else:
        scales = scales_raw
    opacities = gs.get_opacity.squeeze(-1).detach().to(device)
    
    with open(cameras_json) as f:
        cams = json.load(f)
    test_file = os.path.join(source_dir, 'dataset_test.txt')
    test_names = set()
    with open(test_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                test_names.add(parts[0])
    train_cams = [c for c in cams if c['img_name'] not in test_names]
    test_cams = [c for c in cams if c['img_name'] in test_names]
    train_positions = np.array([c['position'] for c in train_cams])
    
    cam0 = cams[0]
    coarse_h, coarse_w = 38, 68
    fine_h, fine_w = 270, 480
    render_h, render_w = 540, 960
    
    scale_x = render_w / cam0['width']
    scale_y = render_h / cam0['height']
    K_render = torch.tensor([
        [cam0['fx']*scale_x, 0, cam0['width']/2*scale_x],
        [0, cam0['fy']*scale_y, cam0['height']/2*scale_y],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    K_coarse = K_render.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h
    K_fine = K_render.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h
    
    normalize = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    
    with torch.no_grad():
        coarse_colors = F.normalize(triplane.extract_coarse(means3d), p=2, dim=1)
        fine_colors = F.normalize(triplane.extract_fine(means3d), p=2, dim=1)
    
    # Use ALL test samples (or subset) for detailed analysis
    n_samples = min(50, len(test_cams))  # 50 samples for good coverage
    sample_indices = np.linspace(0, len(test_cams)-1, n_samples, dtype=int)
    
    print(f"\nAnalyzing {n_samples} test samples with 3R no-lr-decay...", flush=True)
    print(f"{'idx':>4} {'img_name':>30} {'init_pos':>9} {'init_rot':>9} {'final_pos':>10} {'final_rot':>10} {'status':>8}", flush=True)
    print("-" * 85, flush=True)
    
    results = []
    for si, idx in enumerate(tqdm(sample_indices, desc="Analyzing")):
        cam = test_cams[idx]
        gt_w2c = cam_to_w2c(cam).to(device)
        
        # NN init
        dists = np.linalg.norm(train_positions - np.array(cam['position']), axis=1)
        nn_idx = np.argmin(dists)
        nn_w2c = cam_to_w2c(train_cams[nn_idx]).to(device)
        init_pos, init_rot = pose_error(nn_w2c, gt_w2c)
        
        # Encode
        img = Image.open(os.path.join(source_dir, cam['img_name'])).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        img_norm = normalize(img_tensor)
        
        with torch.no_grad():
            coarse_2d, fine_2d = encoder(img_norm)
            coarse_2d = F.interpolate(coarse_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
            coarse_2d = F.normalize(coarse_2d, p=2, dim=1)
            fine_2d = F.interpolate(fine_2d, (fine_h, fine_w), mode='bilinear', align_corners=False)
            fine_2d = F.normalize(fine_2d, p=2, dim=1)
        
        # 3R no-lr-decay
        current = nn_w2c.clone()
        for rnd in range(3):
            current = refine_pose(
                coarse_2d, means3d, quats, scales, opacities, coarse_colors,
                current, K_coarse, coarse_w, coarse_h,
                n_iters=300, lr=0.01)
            current = refine_pose(
                fine_2d, means3d, quats, scales, opacities, fine_colors,
                current, K_fine, fine_w, fine_h,
                n_iters=300, lr=0.005)
        
        final_pos, final_rot = pose_error(current, gt_w2c)
        
        status = "GOOD" if final_pos < 50 else ("MEDIUM" if final_pos < 200 else "FAIL")
        
        r = {
            'idx': int(idx), 'img_name': cam['img_name'],
            'init_pos': init_pos, 'init_rot': init_rot,
            'final_pos': final_pos, 'final_rot': final_rot,
            'status': status,
        }
        results.append(r)
        
        print(f"{idx:4d} {cam['img_name']:>30s} {init_pos:8.1f}cm {init_rot:8.2f}° {final_pos:9.1f}cm {final_rot:9.2f}° {status:>8s}", flush=True)
    
    # Analysis
    results_arr = np.array([(r['init_pos'], r['init_rot'], r['final_pos'], r['final_rot']) for r in results])
    init_pos_arr = results_arr[:, 0]
    final_pos_arr = results_arr[:, 2]
    final_rot_arr = results_arr[:, 3]
    
    good = [r for r in results if r['status'] == 'GOOD']
    medium = [r for r in results if r['status'] == 'MEDIUM']
    fail = [r for r in results if r['status'] == 'FAIL']
    
    print(f"\n{'='*70}", flush=True)
    print(f"FAILURE ANALYSIS ({n_samples} samples)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  GOOD (<50cm):   {len(good):3d} ({100*len(good)/n_samples:.0f}%)", flush=True)
    print(f"  MEDIUM (50-200cm): {len(medium):3d} ({100*len(medium)/n_samples:.0f}%)", flush=True)
    print(f"  FAIL (>200cm):  {len(fail):3d} ({100*len(fail)/n_samples:.0f}%)", flush=True)
    
    print(f"\n  Median pos: {np.median(final_pos_arr):.1f} cm", flush=True)
    print(f"  Median rot: {np.median(final_rot_arr):.2f}°", flush=True)
    print(f"  P75 pos:    {np.percentile(final_pos_arr, 75):.1f} cm", flush=True)
    print(f"  P90 pos:    {np.percentile(final_pos_arr, 90):.1f} cm", flush=True)
    
    # Correlation: init error vs final error
    corr = np.corrcoef(init_pos_arr, final_pos_arr)[0, 1]
    print(f"\n  Correlation(init_pos, final_pos): {corr:.3f}", flush=True)
    
    # Stats per bucket
    print(f"\n  Init distance → final error breakdown:", flush=True)
    for low, high in [(0, 100), (100, 200), (200, 300), (300, 500)]:
        mask = (init_pos_arr >= low) & (init_pos_arr < high)
        if mask.sum() > 0:
            fp = final_pos_arr[mask]
            print(f"    Init {low:3d}-{high:3d}cm ({mask.sum():2d} samples): "
                  f"median_final={np.median(fp):.1f}cm, "
                  f"fail_rate={100*np.sum(fp > 200)/mask.sum():.0f}%", flush=True)
    
    # Worst cases
    print(f"\n  Top 10 worst cases:", flush=True)
    sorted_by_final = sorted(results, key=lambda r: r['final_pos'], reverse=True)
    for r in sorted_by_final[:10]:
        print(f"    {r['img_name']:>30s}: init={r['init_pos']:.0f}cm/{r['init_rot']:.1f}° → "
              f"final={r['final_pos']:.0f}cm/{r['final_rot']:.2f}°", flush=True)
    
    # Save results
    out_path = 'output/gsff_failure_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
