"""
Visualize RADIO-GS rendered features via PCA decomposition.

Usage:
    python radio_gs/scripts/visualize_features.py \
        --config radio_gs/configs/replica_explicit.yaml \
        --checkpoint output/radio_gs/replica_explicit/checkpoints/best.pth \
        --num_views 10 \
        --output_dir output/radio_gs/vis/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def features_to_pca_rgb(features: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """Convert high-dim feature map to RGB via PCA.

    Args:
        features: [C, H, W] feature tensor.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    C, H, W = features.shape
    feat_flat = features.reshape(C, -1).T.cpu().numpy()  # [HW, C]

    mean = feat_flat.mean(axis=0)
    centered = feat_flat - mean

    # Fast PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_proj = U[:, :n_components] * S[:n_components]  # [HW, 3]

    # Normalize to [0, 1]
    for i in range(n_components):
        vmin, vmax = pca_proj[:, i].min(), pca_proj[:, i].max()
        if vmax - vmin > 1e-8:
            pca_proj[:, i] = (pca_proj[:, i] - vmin) / (vmax - vmin)
        else:
            pca_proj[:, i] = 0.5

    rgb = (pca_proj.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb


def save_comparison(gt_feat, rendered_feat, decoded_feat, save_path):
    """Save side-by-side PCA visualization: GT | Rendered-compact | Decoded."""
    try:
        from PIL import Image
    except ImportError:
        print('  PIL not available, skipping visualization')
        return

    panels = []
    if gt_feat is not None:
        panels.append(features_to_pca_rgb(gt_feat))
    panels.append(features_to_pca_rgb(rendered_feat))
    if decoded_feat is not None:
        panels.append(features_to_pca_rgb(decoded_feat))

    combined = np.concatenate(panels, axis=1)
    Image.fromarray(combined).save(save_path)


def main():
    parser = argparse.ArgumentParser(description='RADIO-GS Feature Visualization')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    from radio_gs.config import load_config
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.join(config.output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    from radio_gs.scripts.eval_downstream import build_components, render_decoded
    model, codec, sharpener, renderer = build_components(config, args.checkpoint, device)

    print(f'Rendering {args.num_views} views for visualization...')
    print(f'Output: {output_dir}')

    # Load poses
    pose_file = os.path.join('dataset', config.scene, config.val_split, 'traj_w_c.txt')
    if os.path.exists(pose_file):
        raw_poses = np.loadtxt(pose_file).reshape(-1, 4, 4).astype(np.float32)

        # Load GT features for comparison
        feature_dir = config.feature_dir.replace(config.train_split, config.val_split)
        backbone_dir = os.path.join(feature_dir, 'backbone')
        if not os.path.isdir(backbone_dir):
            backbone_dir = feature_dir

        for view_idx in range(min(args.num_views, len(raw_poses))):
            c2w = raw_poses[view_idx]
            w2c = torch.tensor(np.linalg.inv(c2w).astype(np.float32))

            decoded, result = render_decoded(
                model, codec, sharpener, renderer, w2c, device
            )

            # Load GT feature if available
            gt_path = os.path.join(backbone_dir, f'rgb_{view_idx}.pt')
            gt_feat = None
            if os.path.exists(gt_path):
                gt_feat = torch.load(gt_path, map_location='cpu')
                if gt_feat.dim() == 4:
                    gt_feat = gt_feat.squeeze(0)

            try:
                from PIL import Image

                # PCA of decoded
                decoded_pca = features_to_pca_rgb(decoded.squeeze(0).cpu())
                Image.fromarray(decoded_pca).save(
                    os.path.join(output_dir, f'view_{view_idx:04d}_decoded_pca.png')
                )

                # PCA of GT
                if gt_feat is not None:
                    gt_pca = features_to_pca_rgb(gt_feat)
                    Image.fromarray(gt_pca).save(
                        os.path.join(output_dir, f'view_{view_idx:04d}_gt_pca.png')
                    )
                    # Side-by-side: GT | Decoded
                    combined = np.concatenate([gt_pca, decoded_pca], axis=1)
                    Image.fromarray(combined).save(
                        os.path.join(output_dir, f'view_{view_idx:04d}_comparison.png')
                    )

                # Depth visualization
                depth = result.get('depth_map', None) if isinstance(result, dict) else None
                if depth is not None:
                    depth_np = depth.cpu().numpy()
                    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                    depth_img = (depth_norm * 255).astype(np.uint8)
                    Image.fromarray(depth_img).save(
                        os.path.join(output_dir, f'view_{view_idx:04d}_depth.png')
                    )

            except ImportError:
                np.save(os.path.join(output_dir, f'view_{view_idx:04d}_decoded.npy'),
                        decoded.squeeze(0).cpu().numpy())

            # Per-pixel cosine similarity
            if gt_feat is not None:
                gt_dev = gt_feat.unsqueeze(0).float().to(device)
                if decoded.shape[-2:] != gt_dev.shape[-2:]:
                    decoded_rs = F.interpolate(decoded, gt_dev.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    decoded_rs = decoded
                cos = F.cosine_similarity(decoded_rs, gt_dev, dim=1).mean().item()
                print(f'  View {view_idx}: cosine={cos:.4f}')
            else:
                print(f'  View {view_idx}: saved')
    else:
        print(f'  Pose file not found: {pose_file}')
        print('  Provide poses via dataset or --pose_file flag')

    print(f'\nDone! Visualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
