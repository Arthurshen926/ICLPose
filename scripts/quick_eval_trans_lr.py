#!/usr/bin/env python3
"""
Quick eval: test different trans_lr_scale values on 30 samples.
Tests whether giving translation a higher learning rate improves position error.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder, OldFineEncoder
from gsff.pose_refine import refine_pose, render_features_for_pose
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
    
    print("Loading model...")
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
    
    # Uniformly sample test images
    n_samples = 30
    sample_indices = np.linspace(0, len(test_cams)-1, n_samples, dtype=int)
    
    # Test configs: different trans_lr_scale values, all using 3R no-lr-decay
    configs = [
        {'name': 'baseline(1x)', 'trans_lr_scale': 1.0},
        {'name': 'trans_2x', 'trans_lr_scale': 2.0},
        {'name': 'trans_3x', 'trans_lr_scale': 3.0},
        {'name': 'trans_5x', 'trans_lr_scale': 5.0},
        {'name': 'trans_10x', 'trans_lr_scale': 10.0},
    ]
    
    rounds = 3
    c_iters, f_iters = 300, 300
    c_lr, f_lr = 0.01, 0.005
    
    results = {c['name']: {'pos': [], 'rot': []} for c in configs}
    
    for si, idx in enumerate(tqdm(sample_indices, desc="Samples")):
        cam = test_cams[idx]
        gt_w2c = cam_to_w2c(cam).to(device)
        
        # NN init
        dists = np.linalg.norm(train_positions - np.array(cam['position']), axis=1)
        nn_idx = np.argmin(dists)
        nn_w2c = cam_to_w2c(train_cams[nn_idx]).to(device)
        
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
        
        for cfg in configs:
            current = nn_w2c.clone()
            tls = cfg['trans_lr_scale']
            for rnd in range(rounds):
                current = refine_pose(
                    coarse_2d, means3d, quats, scales, opacities, coarse_colors,
                    current, K_coarse, coarse_w, coarse_h,
                    n_iters=c_iters, lr=c_lr,
                    trans_lr_scale=tls)
                
                current = refine_pose(
                    fine_2d, means3d, quats, scales, opacities, fine_colors,
                    current, K_fine, fine_w, fine_h,
                    n_iters=f_iters, lr=f_lr,
                    trans_lr_scale=tls)
            
            pos, rot = pose_error(current, gt_w2c)
            results[cfg['name']]['pos'].append(pos)
            results[cfg['name']]['rot'].append(rot)
        
        # Print progress every 5 samples
        if (si + 1) % 5 == 0 or si == 0:
            print(f"\n--- After {si+1} samples ---")
            for cfg in configs:
                p = results[cfg['name']]['pos']
                r = results[cfg['name']]['rot']
                print(f"  {cfg['name']:15s}: med_pos={np.median(p):6.1f}cm  med_rot={np.median(r):.2f}°  "
                      f"mean_pos={np.mean(p):6.1f}cm")
    
    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({n_samples} samples, {rounds}R no-lr-decay)")
    print(f"{'='*70}")
    for cfg in configs:
        p = np.array(results[cfg['name']]['pos'])
        r = np.array(results[cfg['name']]['rot'])
        print(f"  {cfg['name']:15s}: med_pos={np.median(p):6.1f}cm  med_rot={np.median(r):.2f}°  "
              f"mean_pos={np.mean(p):6.1f}cm  p90_pos={np.percentile(p,90):6.1f}cm")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
