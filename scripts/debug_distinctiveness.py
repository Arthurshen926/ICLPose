"""
Diagnostic: Check rendered feature distinctiveness AND depth accuracy.
If rendered features are less distinctive than query features, 
correlation matching can't produce meaningful flow.
If depth is wrong, GT flow targets are wrong → divergence.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from modules.multiscale_renderer import MultiScaleRenderer

def compute_distinctiveness(feat_map):
    """
    Measure how distinctive features are. 
    For each pixel, compute max cosine similarity with all OTHER pixels.
    Distinctiveness = 1 - mean(max_sim).
    High distinctiveness = good for matching.
    """
    B, C, H, W = feat_map.shape
    feat_flat = feat_map.reshape(B, C, -1)  # (B, C, N)
    feat_flat = F.normalize(feat_flat, p=2, dim=1)  # L2 normalize
    
    # Cosine similarity matrix (N x N)
    sim = torch.bmm(feat_flat.transpose(1, 2), feat_flat)  # (B, N, N)
    
    # Mask diagonal
    N = sim.shape[-1]
    mask = torch.eye(N, device=sim.device).unsqueeze(0).bool()
    sim.masked_fill_(mask, -1.0)
    
    # Max similarity for each pixel (excluding self)
    max_sim = sim.max(dim=-1).values  # (B, N)
    
    return {
        'distinctiveness': (1 - max_sim.mean()).item(),
        'max_sim_mean': max_sim.mean().item(),
        'max_sim_std': max_sim.std().item(),
        'max_sim_min': max_sim.min().item(),  # Most distinctive pixel
        'max_sim_max': max_sim.max().item(),  # Least distinctive pixel
    }

def main():
    device = 'cuda:0'
    
    ply_path = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
    scale_model_paths = {
        'coarse': 'output/feature_3dgs/oldhospital_ae_perscale/coarse/best_model.pth',
        'mid': 'output/feature_3dgs/oldhospital_ae_perscale/mid/best_model.pth',
        'fine_sd': 'output/feature_3dgs/oldhospital_ae_perscale/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/oldhospital_ae_perscale/fine_dino/best_model.pth',
    }
    feature_dir = 'output/features_multiscale_compressed/OldHospital_indexed'
    traj_path = f'{feature_dir}/traj_w_c.txt'
    depth_dir = 'dataset/OldHospital/Sequence_1/depth'
    
    fx, fy = 1673.5, 1673.5
    cx, cy = 960.0, 540.0
    img_h, img_w = 1080, 1920
    
    scale_config = {
        'coarse': {'subdir': 'coarse', 'dim': 32},
        'mid': {'subdir': 'mid', 'dim': 64},
        'fine_sd': {'subdir': 'fine_sd', 'dim': 64},
        'fine_dino': {'subdir': 'fine_dino', 'dim': 64},
    }
    
    # Load poses
    poses_c2w_flat = np.loadtxt(traj_path)
    n_frames = poses_c2w_flat.shape[0]
    poses_c2w = poses_c2w_flat.reshape(n_frames, 4, 4)
    poses_w2c = np.linalg.inv(poses_c2w)
    
    renderer = MultiScaleRenderer(
        ply_path=ply_path,
        scale_model_paths=scale_model_paths,
        device=device,
        img_height=img_h, img_width=img_w,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )
    
    print("\n" + "="*70)
    print("FEATURE DISTINCTIVENESS: Rendered vs Query")
    print("="*70)
    
    for frame_idx in [0, 100, 500]:
        pose_w2c = torch.tensor(poses_w2c[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        result = renderer.render_batch(pose_w2c, scales=['coarse', 'mid', 'fine_sd', 'fine_dino'], return_depth=True)
        
        print(f"\nFrame {frame_idx}:")
        for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
            rendered = result[f'{scale}_feat']
            
            # Load query
            base = Path(feature_dir) / scale_config[scale]['subdir']
            query_file = list(sorted(base.glob(f'rgb_{frame_idx}_*.pt')))[0]
            query = torch.load(str(query_file), map_location=device).unsqueeze(0)
            
            if query.shape != rendered.shape:
                query = F.interpolate(query, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
            
            r_dist = compute_distinctiveness(rendered)
            q_dist = compute_distinctiveness(query)
            
            print(f"  {scale:10s}: rendered dist={r_dist['distinctiveness']:.4f} "
                  f"(max_sim={r_dist['max_sim_mean']:.4f}±{r_dist['max_sim_std']:.4f}) | "
                  f"query dist={q_dist['distinctiveness']:.4f} "
                  f"(max_sim={q_dist['max_sim_mean']:.4f}±{q_dist['max_sim_std']:.4f})")
    
    # ========================
    # DEPTH CHECK
    # ========================
    print("\n" + "="*70)
    print("DEPTH RENDERING CHECK")
    print("="*70)
    
    depth_files_exist = Path(depth_dir).exists()
    print(f"  Depth directory exists: {depth_files_exist}")
    
    if depth_files_exist:
        import cv2
        depth_files = sorted(Path(depth_dir).glob('*.png'))
        print(f"  Found {len(depth_files)} depth files")
        
        for frame_idx in [0, 100]:
            if frame_idx >= len(depth_files):
                continue
            # Load GT depth
            depth_gt_raw = cv2.imread(str(depth_files[frame_idx]), cv2.IMREAD_UNCHANGED)
            if depth_gt_raw is None:
                continue
            depth_gt = depth_gt_raw.astype(np.float32) / 1000.0  # mm -> meters
            
            # Render depth
            pose_w2c = torch.tensor(poses_w2c[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
            depth_rendered = renderer.render_depth_batch(pose_w2c)  # (1, H, W)
            depth_rendered = depth_rendered[0].cpu().numpy()
            
            # Resize GT to match rendered
            rH, rW = depth_rendered.shape
            depth_gt_resized = cv2.resize(depth_gt, (rW, rH), interpolation=cv2.INTER_NEAREST)
            
            # Compare
            valid = (depth_gt_resized > 0.1) & (depth_gt_resized < 100) & (depth_rendered > 0.1)
            if valid.sum() > 0:
                diff = np.abs(depth_gt_resized[valid] - depth_rendered[valid])
                rel_err = diff / depth_gt_resized[valid]
                print(f"\n  Frame {frame_idx}: rendered shape={depth_rendered.shape}")
                print(f"    GT depth range: [{depth_gt_resized[valid].min():.2f}, {depth_gt_resized[valid].max():.2f}]m")
                print(f"    Rendered depth range: [{depth_rendered[valid].min():.2f}, {depth_rendered[valid].max():.2f}]m")
                print(f"    Mean abs error: {diff.mean():.3f}m")
                print(f"    Median rel error: {np.median(rel_err):.4f}")
                print(f"    Valid coverage: {valid.sum()}/{valid.size} ({valid.sum()/valid.size*100:.1f}%)")
            else:
                print(f"\n  Frame {frame_idx}: No valid depth pixels for comparison")
    else:
        print("  No GT depth available for comparison")

    # ========================
    # RENDERED DEPTH STATS
    # ========================
    print("\n" + "="*70)
    print("RENDERED DEPTH STATS (no GT comparison)")
    print("="*70)
    for frame_idx in [0, 100, 500]:
        pose_w2c = torch.tensor(poses_w2c[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        depth = renderer.render_depth_batch(pose_w2c)[0].cpu().numpy()
        valid = depth > 0
        if valid.sum() > 0:
            print(f"  Frame {frame_idx}: min={depth[valid].min():.2f}m, max={depth[valid].max():.2f}m, "
                  f"mean={depth[valid].mean():.2f}m, std={depth[valid].std():.2f}m, "
                  f"coverage={valid.sum()}/{valid.size} ({valid.sum()/valid.size*100:.1f}%)")
        else:
            print(f"  Frame {frame_idx}: No valid depth!")


if __name__ == '__main__':
    main()
