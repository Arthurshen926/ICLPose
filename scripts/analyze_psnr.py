#!/usr/bin/env python3
"""Quick per-view PSNR analysis to identify failure modes."""
import torch, sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
from feature_3dgs.train_2dgs_geometry import (
    GaussianModel2DGS, load_scene, load_image_tensor, render_2dgs
)
from plyfile import PlyData
from torch import nn

def load_gaussians_from_ply(ply_path, sh_degree=3):
    gaussians = GaussianModel2DGS(sh_degree=sh_degree)
    gaussians.active_sh_degree = sh_degree
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    N = len(v)
    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device='cuda'))
    f_dc = np.stack([v[f'f_dc_{i}'] for i in range(3)], axis=1)
    gaussians._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float32, device='cuda').reshape(N,1,3))
    n_rest = (sh_degree + 1)**2 - 1
    f_rest = np.stack([v[f'f_rest_{i}'] for i in range(n_rest*3)], axis=1)
    gaussians._features_rest = nn.Parameter(torch.tensor(f_rest, dtype=torch.float32, device='cuda').reshape(N,n_rest,3))
    gaussians._opacity = nn.Parameter(torch.tensor(v['opacity'][:, None], dtype=torch.float32, device='cuda'))
    scales = np.stack([v['scale_0'], v['scale_1']], axis=1)
    gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device='cuda'))
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)
    gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device='cuda'))
    gaussians.max_radii2D = torch.zeros(N, device='cuda')
    return gaussians, N

def main():
    ply_path = 'output/2dgs_models/OldHospital/v3_retrain10/point_cloud/iteration_20000/point_cloud.ply'
    print(f"Loading model from {ply_path}")
    gaussians, N = load_gaussians_from_ply(ply_path)
    print(f"Loaded {N:,} Gaussians")

    # Opacity stats
    opacities = gaussians.get_opacity.detach().squeeze()
    print(f"\nOpacity stats: mean={opacities.mean():.3f} median={opacities.median():.3f} "
          f"<0.1={((opacities<0.1).sum()/N*100):.1f}% >0.9={((opacities>0.9).sum()/N*100):.1f}%")
    
    # Scale stats
    scales = gaussians.get_scaling.detach()
    max_scale = scales.max(dim=1).values
    print(f"Scale stats: mean={max_scale.mean():.4f} median={max_scale.median():.4f} "
          f">1.0={((max_scale>1).sum()/N*100):.1f}% >5.0={((max_scale>5).sum()/N*100):.1f}%")

    _, test_cams, _, _, _ = load_scene('dataset/OldHospital', eval_split=True)
    print(f"Test views: {len(test_cams)}")

    bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
    bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')

    # Evaluate sample views with BOTH black and white background
    psnrs_black = []
    psnrs_white = []
    sample = test_cams[::9]  # every 9th view (~20 views)
    print(f"\nEvaluating {len(sample)} sample views...")
    
    with torch.no_grad():
        for cam in sample:
            rp_b = render_2dgs(gaussians, cam, bg_black, longest_edge=0)
            rp_w = render_2dgs(gaussians, cam, bg_white, longest_edge=0)
            rw, rh = rp_b['width'], rp_b['height']

            gt = load_image_tensor(cam)
            gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode='bilinear', align_corners=False).squeeze(0)

            img_b = rp_b['render'].clamp(0, 1)
            img_w = rp_w['render'].clamp(0, 1)

            mse_b = F.mse_loss(img_b, gt).item()
            mse_w = F.mse_loss(img_w, gt).item()
            psnr_b = -10 * math.log10(mse_b) if mse_b > 0 else 99
            psnr_w = -10 * math.log10(mse_w) if mse_w > 0 else 99

            alpha = rp_b['rend_alpha']
            low_alpha_pct = (alpha.squeeze() < 0.5).float().mean().item() * 100

            psnrs_black.append(psnr_b)
            psnrs_white.append(psnr_w)
            
            print(f"  {cam.image_name}: black={psnr_b:.2f} white={psnr_w:.2f} delta={psnr_w-psnr_b:+.2f} low_alpha={low_alpha_pct:.1f}%")

    print(f"\n{'='*60}")
    print(f"Summary ({len(sample)} views):")
    print(f"  Black BG: mean={np.mean(psnrs_black):.2f} dB")
    print(f"  White BG: mean={np.mean(psnrs_white):.2f} dB")
    print(f"  Delta:    {np.mean(psnrs_white)-np.mean(psnrs_black):+.2f} dB")
    print(f"  White better: {sum(1 for b,w in zip(psnrs_black, psnrs_white) if w>b)}/{len(sample)}")

if __name__ == "__main__":
    main()
