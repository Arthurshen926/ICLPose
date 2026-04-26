#!/usr/bin/env python3
"""
Visualize depth-warp vs 3DGS-rendered features for OldHospital.

Shows WHY depth-warp works: it preserves spatial distinctiveness of stored
features, while 3DGS rendering severely smooths features → all pixels look alike.

Output: output/depth_warp_comparison.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from modules.multiscale_renderer import MultiScaleRenderer
from modules.depth_warp import backward_warp_features
from modules.lie_algebra import se3_exp


def load_features_and_poses(feature_dir, traj_path, idx=50):
    """Load a single frame's features and poses."""
    feature_dir = Path(feature_dir)
    
    # Load trajectory
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(torch.tensor(vals).reshape(4, 4))
    poses = torch.stack(poses)
    
    # Load features for the specific frame
    fine_dir = feature_dir / 'fine'
    files = sorted(fine_dir.glob('*.pt'))
    
    feat = torch.load(files[idx], map_location='cpu')
    pose_c2w = poses[idx]
    pose_w2c = torch.linalg.inv(pose_c2w)
    
    return feat, pose_w2c, pose_c2w


def features_to_rgb(feat, method='pca'):
    """Convert feature map to RGB via PCA for visualization."""
    C, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).T.numpy()  # (H*W, C)
    
    # Remove NaN/Inf
    valid = np.isfinite(feat_flat).all(axis=1)
    if valid.sum() < 10:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    pca = PCA(n_components=3)
    rgb_flat = np.zeros((H * W, 3))
    rgb_flat[valid] = pca.fit_transform(feat_flat[valid])
    
    # Normalize to [0, 1]
    for c in range(3):
        vmin, vmax = np.percentile(rgb_flat[valid, c], [2, 98])
        if vmax - vmin > 1e-6:
            rgb_flat[:, c] = np.clip((rgb_flat[:, c] - vmin) / (vmax - vmin), 0, 1)
        else:
            rgb_flat[:, c] = 0.5
    
    return (rgb_flat.reshape(H, W, 3) * 255).astype(np.uint8)


def compute_self_similarity(feat):
    """Compute mean pairwise cosine similarity (discriminativeness metric)."""
    C, H, W = feat.shape
    feat_flat = feat.reshape(C, -1).float()  # (C, N)
    feat_norm = F.normalize(feat_flat, dim=0)  # L2 normalize each pixel
    
    # Sample 500 random pairs for efficiency
    N = feat_flat.shape[1]
    n_samples = min(500, N)
    idx = torch.randperm(N)[:n_samples]
    sampled = feat_norm[:, idx]  # (C, n_samples)
    
    sim_matrix = sampled.T @ sampled  # (n_samples, n_samples)
    # Get off-diagonal elements
    mask = ~torch.eye(n_samples, dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()
    
    return mean_sim


def main():
    import os
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '5'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    feature_dir = 'output/features_da3_unified/OldHospital'
    traj_path = 'output/features_da3_unified/OldHospital/traj_w_c.txt'
    
    print("[1/5] Loading features and poses...")
    query_feat, pose_w2c_gt, pose_c2w_gt = load_features_and_poses(
        feature_dir, traj_path, idx=50)
    
    # Add perturbation to simulate initial pose error (4° rotation, 0.2m translation)
    torch.manual_seed(42)
    noise = torch.zeros(6)
    noise[:3] = torch.randn(3) * 0.2       # translation noise (m)
    noise[3:] = torch.randn(3) * (4.0 * np.pi / 180)  # rotation noise (rad)
    T_noise = se3_exp(noise.unsqueeze(0)).squeeze(0)  # (4, 4)
    pose_w2c_est = T_noise @ pose_w2c_gt   # perturbed w2c
    
    print(f"  Query features: {query_feat.shape}")
    print(f"  Perturbation: ~4° rotation, ~0.2m translation")
    
    print("[2/5] Loading renderer (3DGS)...")
    renderer = MultiScaleRenderer(
        ply_path='output/2dgs_joint/joint_oh_v3/point_cloud/best/point_cloud.ply',
        scale_model_paths={
            'fine': 'output/2dgs_joint/joint_oh_v3/features_best/fine/best_model.pth',
        },
        device=device,
        img_height=1080, img_width=1920,
        fx=1673.5, fy=1673.5, cx=960.0, cy=540.0,
        scale_resolutions={'fine': [69, 121]},
    )
    
    print("[3/5] Rendering 3DGS features at estimated pose...")
    with torch.no_grad():
        # Render features from 3DGS
        render_result = renderer.render_batch(
            pose_w2c_est.unsqueeze(0).to(device),
            scales=['fine'], return_depth=True)
        rendered_feat = render_result['fine_feat'][0].cpu()  # (C, H, W)
        depth_est = render_result['depth_map'][0].cpu()      # (H, W)
    
    print(f"  Rendered features: {rendered_feat.shape}")
    print(f"  Depth: {depth_est.shape}, range [{depth_est[depth_est>0].min():.1f}, {depth_est.max():.1f}]m")
    
    # Upsample rendered features to match query resolution if needed
    qH, qW = query_feat.shape[1], query_feat.shape[2]
    if rendered_feat.shape[1:] != (qH, qW):
        print(f"  Upsampling rendered {rendered_feat.shape[1:]} -> ({qH}, {qW})")
        rendered_feat = F.interpolate(rendered_feat.unsqueeze(0), size=(qH, qW),
                                       mode='bilinear', align_corners=False)[0]
    
    print("[4/5] Computing depth-warped features...")
    with torch.no_grad():
        # Also render at GT pose for comparison
        render_gt = renderer.render_batch(
            pose_w2c_gt.unsqueeze(0).to(device),
            scales=['fine'], return_depth=True)
        rendered_feat_gt = render_gt['fine_feat'][0].cpu()
        depth_gt = render_gt['depth_map'][0].cpu()
        
        # Upsample rendered GT features to match query resolution
        if rendered_feat_gt.shape[1:] != (qH, qW):
            rendered_feat_gt = F.interpolate(rendered_feat_gt.unsqueeze(0), size=(qH, qW),
                                              mode='bilinear', align_corners=False)[0]
        
        # Depth warp: use depth at estimated pose to warp query features
        depth_for_warp = renderer.render_depth_batch(
            pose_w2c_est.unsqueeze(0).to(device))[0].cpu()
        
        scale_intrinsics = renderer.get_scale_intrinsics()
        warped = backward_warp_features(
            {'fine': query_feat.unsqueeze(0).float()},
            depth_for_warp.unsqueeze(0).float(),
            pose_w2c_gt.unsqueeze(0).float(),
            pose_w2c_est.unsqueeze(0).float(),
            scale_intrinsics,
        )
        warped_feat = warped['fine'][0].cpu()  # (C, H, W)
    
    print(f"  Warped features: {warped_feat.shape}")
    
    print("[5/5] Computing metrics and generating visualization...")
    
    # Self-similarity (lower = more discriminative)
    sim_query = compute_self_similarity(query_feat)
    sim_rendered = compute_self_similarity(rendered_feat)
    sim_warped = compute_self_similarity(warped_feat)
    sim_rendered_gt = compute_self_similarity(rendered_feat_gt)
    
    # Cosine similarity between query@GT and reference features
    def cosine_map(a, b):
        """Per-pixel cosine similarity between two feature maps."""
        a_norm = F.normalize(a, dim=0)
        b_norm = F.normalize(b, dim=0)
        return (a_norm * b_norm).sum(dim=0)  # (H, W)
    
    cos_rendered = cosine_map(query_feat.float(), rendered_feat.float())
    cos_warped = cosine_map(query_feat.float(), warped_feat.float())
    cos_rendered_gt = cosine_map(query_feat.float(), rendered_feat_gt.float())
    
    # PCA visualization
    query_rgb = features_to_rgb(query_feat)
    rendered_rgb = features_to_rgb(rendered_feat)
    warped_rgb = features_to_rgb(warped_feat)
    rendered_gt_rgb = features_to_rgb(rendered_feat_gt)
    
    # ═══════════════════════════════════════════
    #  Plot
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle('Depth-Warp vs 3DGS Rendering: Why Domain Gap Kills Pose Estimation',
                 fontsize=16, fontweight='bold')
    
    # Row 1: PCA feature visualizations
    axes[0, 0].imshow(query_rgb)
    axes[0, 0].set_title(f'Query Features (stored)\nself-sim={sim_query:.3f}', fontsize=11)
    
    axes[0, 1].imshow(rendered_gt_rgb)
    axes[0, 1].set_title(f'3DGS Rendered @ GT pose\nself-sim={sim_rendered_gt:.3f}', fontsize=11)
    
    axes[0, 2].imshow(rendered_rgb)
    axes[0, 2].set_title(f'3DGS Rendered @ estimated pose\nself-sim={sim_rendered:.3f}', fontsize=11)
    
    axes[0, 3].imshow(warped_rgb)
    axes[0, 3].set_title(f'Depth-Warped @ estimated pose\nself-sim={sim_warped:.3f}', fontsize=11)
    
    # Row 2: Cosine similarity maps
    axes[1, 0].text(0.5, 0.5, 
                     f'Self-Similarity\n(lower = better)\n\n'
                     f'Query:     {sim_query:.3f}\n'
                     f'3DGS@GT:   {sim_rendered_gt:.3f}\n'
                     f'3DGS@Est:  {sim_rendered:.3f}\n'
                     f'Warp@Est:  {sim_warped:.3f}\n\n'
                     f'---\n'
                     f'Ideal: <0.2\n'
                     f'Broken: >0.8',
                     transform=axes[1, 0].transAxes,
                     fontsize=13, ha='center', va='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 0].set_axis_off()
    axes[1, 0].set_title('Discriminativeness Summary', fontsize=11)
    
    im1 = axes[1, 1].imshow(cos_rendered_gt.numpy(), vmin=0.5, vmax=1.0, cmap='RdYlGn')
    axes[1, 1].set_title(f'Cos(Query, 3DGS@GT)\nmean={cos_rendered_gt.mean():.3f}', fontsize=11)
    plt.colorbar(im1, ax=axes[1, 1], shrink=0.7)
    
    im2 = axes[1, 2].imshow(cos_rendered.numpy(), vmin=0.5, vmax=1.0, cmap='RdYlGn')
    axes[1, 2].set_title(f'Cos(Query, 3DGS@Est)\nmean={cos_rendered.mean():.3f}', fontsize=11)
    plt.colorbar(im2, ax=axes[1, 2], shrink=0.7)
    
    im3 = axes[1, 3].imshow(cos_warped.numpy(), vmin=0.5, vmax=1.0, cmap='RdYlGn')
    axes[1, 3].set_title(f'Cos(Query, Warp@Est)\nmean={cos_warped.mean():.3f}', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 3], shrink=0.7)
    
    # Row 3: Depth & flow analysis
    depth_vis = depth_est.numpy()
    depth_vis[depth_vis <= 0] = np.nan
    im_d = axes[2, 0].imshow(depth_vis, cmap='magma')
    axes[2, 0].set_title('Rendered Depth @ Est Pose', fontsize=11)
    plt.colorbar(im_d, ax=axes[2, 0], shrink=0.7)
    
    # Difference in cosine: warp - rendered (how much better warp is)
    cos_diff = cos_warped.numpy() - cos_rendered.numpy()
    im_diff = axes[2, 1].imshow(cos_diff, vmin=-0.3, vmax=0.3, cmap='RdBu_r')
    axes[2, 1].set_title(f'Cos(Warp) - Cos(3DGS)\nmean diff={cos_diff.mean():.3f}', fontsize=11)
    plt.colorbar(im_diff, ax=axes[2, 1], shrink=0.7)
    
    # Histogram of cosine similarities
    axes[2, 2].hist(cos_rendered_gt.flatten().numpy(), bins=50, alpha=0.6, 
                     label=f'3DGS@GT μ={cos_rendered_gt.mean():.3f}', color='orange')
    axes[2, 2].hist(cos_rendered.flatten().numpy(), bins=50, alpha=0.6,
                     label=f'3DGS@Est μ={cos_rendered.mean():.3f}', color='red')
    axes[2, 2].hist(cos_warped.flatten().numpy(), bins=50, alpha=0.6,
                     label=f'Warp@Est μ={cos_warped.mean():.3f}', color='green')
    axes[2, 2].set_xlabel('Cosine Similarity')
    axes[2, 2].set_ylabel('Pixel Count')
    axes[2, 2].legend(fontsize=9)
    axes[2, 2].set_title('Distribution of Per-Pixel Similarity', fontsize=11)
    
    # Matching margin analysis
    # For each pixel, compute: cos(correct) - max(cos(other pixels))
    # This measures if the CORRECT pixel has highest similarity
    q_norm = F.normalize(query_feat.float(), dim=0)  # (C, H, W)
    r_norm_rendered = F.normalize(rendered_feat.float(), dim=0)
    r_norm_warped = F.normalize(warped_feat.float(), dim=0)
    
    # Sample some pixels and check matching accuracy
    H, W = query_feat.shape[1], query_feat.shape[2]
    n_test = min(200, H * W)
    test_idx = torch.randperm(H * W)[:n_test]
    
    q_flat = q_norm.reshape(query_feat.shape[0], -1)[:, test_idx]  # (C, n_test)
    
    # For rendered: find best match for each test pixel
    r_flat_rendered = r_norm_rendered.reshape(query_feat.shape[0], -1)  # (C, N)
    r_flat_warped = r_norm_warped.reshape(query_feat.shape[0], -1)
    
    sim_mat_r = q_flat.T @ r_flat_rendered  # (n_test, N)
    sim_mat_w = q_flat.T @ r_flat_warped
    
    # Matching accuracy: predicted match vs correct match
    pred_r = sim_mat_r.argmax(dim=1)
    pred_w = sim_mat_w.argmax(dim=1)
    correct = test_idx
    
    match_acc_r = (pred_r == correct).float().mean().item() * 100
    match_acc_w = (pred_w == correct).float().mean().item() * 100
    
    axes[2, 3].bar(['3DGS@Est', 'Warp@Est'], [match_acc_r, match_acc_w],
                    color=['red', 'green'], alpha=0.7)
    axes[2, 3].set_ylabel('Matching Accuracy (%)')
    axes[2, 3].set_title(f'Pixel Matching Accuracy\n3DGS={match_acc_r:.1f}%, Warp={match_acc_w:.1f}%',
                         fontsize=11)
    axes[2, 3].set_ylim(0, 100)
    for i, v in enumerate([match_acc_r, match_acc_w]):
        axes[2, 3].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    for ax in axes.flat:
        if ax.images or ax.get_title():
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    out_path = 'output/depth_warp_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {out_path}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Why Depth-Warp Works")
    print(f"{'='*60}")
    print(f"  Self-similarity (lower=better discriminativeness):")
    print(f"    Query features:        {sim_query:.4f}")
    print(f"    3DGS rendered @GT:     {sim_rendered_gt:.4f}  ← domain gap")
    print(f"    3DGS rendered @Est:    {sim_rendered:.4f}  ← even worse")
    print(f"    Depth-warped @Est:     {sim_warped:.4f}  ← preserves quality")
    print(f"")
    print(f"  Cosine similarity to query:")
    print(f"    3DGS @GT:   mean={cos_rendered_gt.mean():.4f}")
    print(f"    3DGS @Est:  mean={cos_rendered.mean():.4f}")
    print(f"    Warp @Est:  mean={cos_warped.mean():.4f}")
    print(f"")
    print(f"  Pixel matching accuracy:")
    print(f"    3DGS @Est:  {match_acc_r:.1f}%")
    print(f"    Warp @Est:  {match_acc_w:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
