#!/usr/bin/env python3
"""
CorrPoseNet 可视化脚本

可视化网络内部每次迭代的:
  1. Rendered vs Query 特征图 (PCA 3D → RGB)
  2. Correlation volume argmax → 匹配偏移方向
  3. Predicted flow field (color-coded)
  4. Confidence map (像素权重 / 特征选择)
  5. GT vs Predicted flow 对比
  
Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_corrposenet.py \
        --model_path output/corr_pose/exp005_bs32/best_model.pth \
        --frame_idx 42 \
        --noise_rot_deg 15 \
        --output_dir output/vis_corrposenet
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
import argparse

from ic_models.corr_pose_net import CorrPoseNet, local_correlation
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3, perturb_pose
from modules.featuremetric import compute_image_jacobian
from modules.lie_algebra import se3_exp, se3_log, pose_inverse


# ============================================================================
# Visualization helpers
# ============================================================================

def feat_to_rgb(feat_map, method='pca'):
    """
    将高维特征图转为 RGB (H,W,3) 用于显示.
    feat_map: (C, H, W) tensor
    """
    C, H, W = feat_map.shape
    feat_np = feat_map.detach().cpu().numpy()
    flat = feat_np.reshape(C, -1).T  # (N, C)
    
    if method == 'pca':
        pca = PCA(n_components=3)
        rgb = pca.fit_transform(flat)  # (N, 3)
    else:  # first 3 channels
        rgb = flat[:, :3]
    
    # Normalize to [0, 1]
    for c in range(3):
        vmin, vmax = rgb[:, c].min(), rgb[:, c].max()
        if vmax > vmin:
            rgb[:, c] = (rgb[:, c] - vmin) / (vmax - vmin)
        else:
            rgb[:, c] = 0.5
    
    return rgb.reshape(H, W, 3)


def flow_to_rgb(flow, max_magnitude=None):
    """
    Flow (2, H, W) → RGB (H, W, 3) using HSV encoding.
    Hue = direction, Saturation = 1, Value = magnitude.
    """
    u = flow[0].detach().cpu().numpy()
    v = flow[1].detach().cpu().numpy()
    
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # [-pi, pi]
    
    if max_magnitude is None:
        max_magnitude = mag.max() + 1e-8
    
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    sat = np.ones_like(hue)
    val = np.clip(mag / max_magnitude, 0, 1)
    
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


def corr_argmax_to_rgb(corr, radius):
    """
    Correlation volume (81, H, W) → 匹配偏移方向可视化 (H, W, 3).
    Argmax → (dx, dy) 偏移 → HSV color.
    """
    side = 2 * radius + 1
    peak_idx = corr.argmax(dim=0).cpu().numpy()  # (H, W) ∈ [0, side²-1]
    peak_dy = (peak_idx // side) - radius  # [-r, r]
    peak_dx = (peak_idx % side) - radius   # [-r, r]
    
    # HSV: direction + magnitude
    mag = np.sqrt(peak_dx**2 + peak_dy**2 + 1e-8)
    angle = np.arctan2(peak_dy.astype(float), peak_dx.astype(float))
    
    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.clip(mag / radius, 0, 1)
    
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


def corr_peak_value(corr):
    """Correlation peak value (confidence of match) per pixel."""
    return corr.max(dim=0)[0].detach().cpu().numpy()  # (H, W)


# ============================================================================
# Main visualization
# ============================================================================

def visualize_single_frame(
    net, renderer, dataset, frame_idx_in_dataset, intrinsics, 
    device, output_dir, num_iters=3, noise_seed=None,
):
    """对单帧运行网络并可视化每个迭代的内部信号."""
    
    os.makedirs(output_dir, exist_ok=True)
    net.eval()
    
    # --- Load sample ---
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
        np.random.seed(noise_seed)
    
    sample = dataset[frame_idx_in_dataset]
    query_feats = sample['query_feats']['fine_dino'].unsqueeze(0).to(device)
    pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
    initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
    depth = sample['depth'].unsqueeze(0).to(device) if 'depth' in sample else None
    
    B, D, H, W = query_feats.shape
    
    print(f"Frame {sample['frame_idx']}: feat ({D},{H},{W})")
    
    # Compute initial error
    from scripts.train_corr_pose import compute_pose_error
    init_rot, init_trans = compute_pose_error(initial_pose, pose_gt)
    print(f"  Initial error: rot={init_rot.item():.2f}°, trans={init_trans.item():.3f}m")
    
    # --- Run network step by step (manually, to capture intermediates) ---
    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
    
    fmap_q = net.encode(query_feats)                            # (1, 128, H, W)
    hidden = torch.tanh(net.context_encoder(query_feats))       # (1, 128, H, W)
    
    pose = initial_pose
    
    # Collect per-iteration data
    iter_data = []
    
    for k in range(num_iters):
        with torch.no_grad():
            rendered_feats = net._render_batch(
                renderer, pose.detach(), 'fine_dino', device
            )
        
        fmap_r = net.encode(rendered_feats)
        corr = local_correlation(fmap_r, fmap_q, net.corr_radius)
        corr_feat = net.corr_encoder(corr)
        hidden = net.gru(hidden, corr_feat)
        
        flow = net.flow_head(hidden)
        conf = torch.sigmoid(net.conf_head(hidden))
        delta_xi = torch.zeros(1, 6, device=device)  # We'll compute it properly
        
        # Compute GT flow for comparison
        T_rel = pose_gt @ pose_inverse(pose.detach())
        xi_gt = se3_log(T_rel)
        gt_flow_u = torch.bmm(Ju, xi_gt.unsqueeze(-1)).squeeze(-1)  # (1, N)
        gt_flow_v = torch.bmm(Jv, xi_gt.unsqueeze(-1)).squeeze(-1)
        gt_flow = torch.stack([
            gt_flow_u.reshape(1, H, W),
            gt_flow_v.reshape(1, H, W),
        ], dim=1)  # (1, 2, H, W)
        
        # Geometric solve
        from ic_models.corr_pose_net import diff_pose_solve
        delta_xi = diff_pose_solve(flow, conf, Ju, Jv, valid, net.damping)
        delta_T = se3_exp(delta_xi)
        pose_new = delta_T @ pose
        
        rot_err, trans_err = compute_pose_error(pose_new, pose_gt)
        
        iter_data.append({
            'k': k,
            'rendered_feats': rendered_feats[0].detach(),  # (D, H, W)
            'fmap_r': fmap_r[0].detach(),                  # (128, H, W)
            'fmap_q': fmap_q[0].detach(),                  # (128, H, W)
            'corr': corr[0].detach(),                      # (81, H, W)
            'flow': flow[0].detach(),                      # (2, H, W)
            'gt_flow': gt_flow[0].detach(),                # (2, H, W)
            'conf': conf[0, 0].detach(),                   # (H, W)
            'delta_xi': delta_xi[0].detach(),              # (6,)
            'rot_err': rot_err.item(),
            'trans_err': trans_err.item(),
        })
        
        pose = pose_new.detach()
        
        print(f"  Iter {k+1}: rot={rot_err.item():.2f}°, trans={trans_err.item():.3f}m, "
              f"|flow|_max={flow.abs().max().item():.2f}px, "
              f"conf_mean={conf.mean().item():.3f}")
    
    # --- Plot ---
    n_iters = len(iter_data)
    fig, axes = plt.subplots(n_iters, 6, figsize=(30, 5 * n_iters))
    if n_iters == 1:
        axes = axes[np.newaxis, :]
    
    # Column titles
    col_titles = [
        'Rendered Feature (PCA)',
        'Corr Argmax Direction',
        'Corr Peak Value',
        'Predicted Flow',
        'GT Flow',
        'Confidence (Feature Selection)',
    ]
    
    # Compute max flow magnitude across all iters for consistent coloring
    max_flow_mag = max(
        d['flow'].abs().max().item() for d in iter_data
    )
    max_gt_flow_mag = max(
        d['gt_flow'].abs().max().item() for d in iter_data
    )
    max_mag = max(max_flow_mag, max_gt_flow_mag, 1.0)
    
    for i, data in enumerate(iter_data):
        # Col 0: Rendered features (PCA → RGB)
        axes[i, 0].imshow(feat_to_rgb(data['rendered_feats']))
        axes[i, 0].set_ylabel(f"Iter {i+1}\nrot={data['rot_err']:.1f}°", fontsize=12)
        
        # Col 1: Correlation argmax direction
        axes[i, 1].imshow(corr_argmax_to_rgb(data['corr'], net.corr_radius))
        
        # Col 2: Correlation peak value
        peak_val = corr_peak_value(data['corr'])
        im2 = axes[i, 2].imshow(peak_val, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Col 3: Predicted flow
        axes[i, 3].imshow(flow_to_rgb(data['flow'], max_magnitude=max_mag))
        
        # Col 4: GT flow
        axes[i, 4].imshow(flow_to_rgb(data['gt_flow'], max_magnitude=max_mag))
        
        # Col 5: Confidence map
        im5 = axes[i, 5].imshow(data['conf'].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im5, ax=axes[i, 5], fraction=0.046)
    
    # Set column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11)
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add query feature as reference
    fig.suptitle(
        f"Frame {sample['frame_idx']} — "
        f"Init: {init_rot.item():.1f}° → Final: {iter_data[-1]['rot_err']:.1f}°  |  "
        f"K={num_iters} iterations, corr_radius={net.corr_radius}",
        fontsize=14, fontweight='bold',
    )
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"frame_{sample['frame_idx']}_iters.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # --- Additional: Query feature + confidence overlay ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Query features PCA
    axes2[0].imshow(feat_to_rgb(sample['query_feats']['fine_dino']))
    axes2[0].set_title('Query Feature (PCA)')
    
    # Final confidence as heatmap overlay
    conf_final = iter_data[-1]['conf'].cpu().numpy()
    axes2[1].imshow(feat_to_rgb(sample['query_feats']['fine_dino']), alpha=0.5)
    im_conf = axes2[1].imshow(conf_final, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes2[1].set_title(f'Confidence Overlay (iter {n_iters})')
    plt.colorbar(im_conf, ax=axes2[1], fraction=0.046)
    
    # Confidence evolution across iterations
    for i, data in enumerate(iter_data):
        c = data['conf'].cpu().numpy()
        axes2[2].plot(
            sorted(c.flatten()), 
            np.linspace(0, 1, c.size),
            label=f'Iter {i+1} (mean={c.mean():.3f})',
        )
    axes2[2].set_xlabel('Confidence value')
    axes2[2].set_ylabel('CDF')
    axes2[2].set_title('Confidence Distribution per Iteration')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    
    for ax in axes2[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f"frame_{sample['frame_idx']}_confidence.png")
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path2}")
    
    # --- Additional: Flow error map ---
    fig3, axes3 = plt.subplots(1, n_iters, figsize=(5 * n_iters, 5))
    if n_iters == 1:
        axes3 = [axes3]
    
    for i, data in enumerate(iter_data):
        flow_err = (data['flow'] - data['gt_flow']).norm(dim=0).cpu().numpy()  # (H, W)
        im = axes3[i].imshow(flow_err, cmap='magma', vmin=0, vmax=max_mag * 0.5)
        axes3[i].set_title(f'Iter {i+1}: |flow_pred - flow_gt|\n'
                          f'mean={flow_err.mean():.2f}px, rot={data["rot_err"]:.1f}°')
        axes3[i].set_xticks([])
        axes3[i].set_yticks([])
        plt.colorbar(im, ax=axes3[i], fraction=0.046)
    
    plt.tight_layout()
    save_path3 = os.path.join(output_dir, f"frame_{sample['frame_idx']}_flow_error.png")
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path3}")


# ============================================================================
# Entry
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='output/corr_pose/exp005_bs32/best_model.pth')
    parser.add_argument('--ply_path', default='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply')
    parser.add_argument('--feature_model', default='output/feature_3dgs/room_0_raw/fine_dino/best_model.pth')
    parser.add_argument('--feature_dir', default='output/features_multiscale/room_0')
    parser.add_argument('--traj_path', default='dataset/room_0/Sequence_1/traj_w_c.txt')
    parser.add_argument('--depth_dir', default='dataset/room_0/Sequence_1/depth')
    parser.add_argument('--frame_idx', type=int, nargs='+', default=[42, 100, 300, 500, 850],
                        help='Frame indices to visualize')
    parser.add_argument('--noise_rot_deg', type=float, default=15.0)
    parser.add_argument('--noise_seed', type=int, default=42)
    parser.add_argument('--num_iters', type=int, default=3)
    parser.add_argument('--output_dir', default='output/vis_corrposenet')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load model ---
    print("Loading model...")
    ckpt = torch.load(args.model_path, map_location=device)
    model_args = ckpt.get('args', {})
    
    net = CorrPoseNet(
        feat_dim=model_args.get('feat_dim', 768),
        enc_dim=model_args.get('enc_dim', 128),
        hidden_dim=model_args.get('hidden_dim', 128),
        corr_radius=model_args.get('corr_radius', 4),
        num_iters=model_args.get('num_iters', 3),
        damping=model_args.get('damping', 0.001),
    ).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"  Loaded: {args.model_path} (val_rot={ckpt.get('val_rot_median', '?')}°)")
    
    # --- Load renderer ---
    print("Loading renderer...")
    renderer = MultiScaleRenderer(
        ply_path=args.ply_path,
        scale_model_paths={'fine_dino': args.feature_model},
        device=device,
        img_height=480, img_width=640,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5,
    )
    
    # --- Load dataset ---
    dataset = PoseDatasetV3(
        feature_base_dir=args.feature_dir,
        traj_path=args.traj_path,
        depth_dir=args.depth_dir,
        frame_indices=list(range(900)),
        scale_names=['fine_dino'],
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_m=args.noise_rot_deg / 50.0,
        is_train=True,
        depth_resize=(35, 46),
    )
    
    INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
    
    # --- Visualize each frame ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    for fidx in args.frame_idx:
        if fidx >= len(dataset):
            print(f"Skipping frame {fidx} (dataset has {len(dataset)} frames)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Visualizing frame {fidx}")
        print(f"{'='*60}")
        
        visualize_single_frame(
            net=net,
            renderer=renderer,
            dataset=dataset,
            frame_idx_in_dataset=fidx,
            intrinsics=INTRINSICS,
            device=device,
            output_dir=args.output_dir,
            num_iters=args.num_iters,
            noise_seed=args.noise_seed + fidx,
        )


if __name__ == '__main__':
    main()
