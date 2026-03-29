"""
Diagnostic: measure feature rendering quality at various perturbation levels.
Tests if cos_sim degrades with small pose perturbations (which would explain training divergence).
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from modules.multiscale_renderer import MultiScaleRenderer
from modules.lie_algebra import se3_exp

def load_query_features(feature_dir, frame_idx, scale_name, scale_config):
    """Load query feature for a given frame."""
    cfg = scale_config[scale_name]
    subdir = cfg['subdir']
    
    # Try to find the file
    base = Path(feature_dir) / subdir
    # v1 format: rgb_{idx}_{scale}_{dim}x{H}x{W}.pt
    for f in sorted(base.glob(f'rgb_{frame_idx}_*.pt')):
        return torch.load(str(f), map_location='cpu')
    raise FileNotFoundError(f"No feature file for frame {frame_idx} in {base}")


def main():
    device = 'cuda:0'
    
    # Config matching exp186
    ply_path = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
    scale_model_paths = {
        'coarse': 'output/feature_3dgs/oldhospital_ae_perscale/coarse/best_model.pth',
        'mid': 'output/feature_3dgs/oldhospital_ae_perscale/mid/best_model.pth',
        'fine_sd': 'output/feature_3dgs/oldhospital_ae_perscale/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/oldhospital_ae_perscale/fine_dino/best_model.pth',
    }
    feature_dir = 'output/features_multiscale_compressed/OldHospital_indexed'
    traj_path = f'{feature_dir}/traj_w_c.txt'
    
    fx, fy = 1673.5, 1673.5
    cx, cy = 960.0, 540.0
    img_h, img_w = 1080, 1920
    
    # V1 scale config
    scale_config = {
        'coarse': {'subdir': 'coarse', 'dim': 32},
        'mid': {'subdir': 'mid', 'dim': 64},
        'fine_sd': {'subdir': 'fine_sd', 'dim': 64},
        'fine_dino': {'subdir': 'fine_dino', 'dim': 64},
    }
    
    # Load poses (one 4x4 matrix per line, 16 values)
    poses_c2w_flat = np.loadtxt(traj_path)
    n_frames = poses_c2w_flat.shape[0]
    poses_c2w = poses_c2w_flat.reshape(n_frames, 4, 4)
    poses_w2c = np.linalg.inv(poses_c2w)
    
    print(f"Loaded {n_frames} poses")
    
    # Load renderer
    renderer = MultiScaleRenderer(
        ply_path=ply_path,
        scale_model_paths=scale_model_paths,
        device=device,
        img_height=img_h, img_width=img_w,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )
    
    # Test frames
    test_frames = [0, 100, 500]
    # Perturbation levels (degrees rotation, meters translation)
    perturbation_levels = [
        (0, 0),      # GT pose
        (1, 0.02),   # Very small
        (2, 0.05),   # Start of curriculum
        (5, 0.1),    # Moderate
        (10, 0.3),   # Training max
    ]
    
    scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']
    
    for frame_idx in test_frames:
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx}")
        print(f"{'='*60}")
        
        pose_w2c_gt = torch.tensor(poses_w2c[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        
        for rot_deg, trans_m in perturbation_levels:
            # Create perturbation
            if rot_deg == 0 and trans_m == 0:
                pose_w2c = pose_w2c_gt
            else:
                # Random SE3 perturbation
                np.random.seed(42)  # Reproducible
                rot_rad = rot_deg * np.pi / 180.0
                xi = np.zeros(6)
                # Random rotation axis
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis) * rot_rad
                xi[:3] = axis
                # Random translation
                trans_dir = np.random.randn(3)
                trans_dir = trans_dir / np.linalg.norm(trans_dir) * trans_m
                xi[3:] = trans_dir
                
                xi_tensor = torch.tensor(xi, dtype=torch.float32, device=device).unsqueeze(0)
                delta_T = se3_exp(xi_tensor)  # (1, 4, 4)
                pose_w2c = delta_T @ pose_w2c_gt
            
            # Render at this pose
            result = renderer.render_batch(pose_w2c, scales=scales, return_depth=False)
            
            cos_sims = {}
            for scale in scales:
                rendered = result[f'{scale}_feat']  # (1, C, H, W)
                query = load_query_features(feature_dir, frame_idx, scale, scale_config)
                query = query.to(device).unsqueeze(0)  # (1, C, H, W)
                
                # Resize if needed
                if query.shape != rendered.shape:
                    query = torch.nn.functional.interpolate(
                        query, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
                
                # L2 normalize
                rendered_n = torch.nn.functional.normalize(rendered, p=2, dim=1)
                query_n = torch.nn.functional.normalize(query, p=2, dim=1)
                
                # Per-pixel cos_sim
                cos_sim = (rendered_n * query_n).sum(dim=1)  # (1, H, W)
                cos_sims[scale] = cos_sim.mean().item()
            
            avg = np.mean(list(cos_sims.values()))
            print(f"  rot={rot_deg:2d}° trans={trans_m:.2f}m → "
                  f"coarse={cos_sims['coarse']:.3f} mid={cos_sims['mid']:.3f} "
                  f"fine_sd={cos_sims['fine_sd']:.3f} fine_dino={cos_sims['fine_dino']:.3f} "
                  f"avg={avg:.3f}")


if __name__ == '__main__':
    main()
