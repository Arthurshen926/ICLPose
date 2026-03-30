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
    from radio_gs.scripts.eval_downstream import load_model_and_codec, render_and_decode
    model, codec, renderer = load_model_and_codec(config, args.checkpoint, device)

    print(f'Rendering {args.num_views} views for visualization...')
    print(f'Output: {output_dir}')

    # Load poses
    pose_file = os.path.join('dataset', config.scene, config.val_split, 'traj_w_c.txt')
    if os.path.exists(pose_file):
        poses_c2w = []
        with open(pose_file) as f:
            lines = f.read().strip().split('\n')
        for i in range(0, len(lines), 4):
            if i + 4 > len(lines):
                break
            rows = [list(map(float, lines[i + j].split())) for j in range(4)]
            poses_c2w.append(torch.tensor(rows, dtype=torch.float32))

        for view_idx in range(min(args.num_views, len(poses_c2w))):
            c2w = poses_c2w[view_idx]
            w2c = torch.inverse(c2w)

            decoded, depth = render_and_decode(
                model, codec, renderer, w2c, device, config
            )

            # PCA visualization of decoded features
            decoded_np = features_to_pca_rgb(decoded.squeeze(0).cpu())

            try:
                from PIL import Image
                save_path = os.path.join(output_dir, f'view_{view_idx:04d}_decoded_pca.png')
                Image.fromarray(decoded_np).save(save_path)

                if depth is not None:
                    depth_np = depth.cpu().numpy()
                    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                    depth_img = (depth_norm * 255).astype(np.uint8)
                    depth_path = os.path.join(output_dir, f'view_{view_idx:04d}_depth.png')
                    Image.fromarray(depth_img).save(depth_path)
            except ImportError:
                np.save(os.path.join(output_dir, f'view_{view_idx:04d}_decoded_pca.npy'), decoded_np)

            print(f'  View {view_idx}: saved')
    else:
        print(f'  Pose file not found: {pose_file}')
        print('  Provide poses via dataset or --pose_file flag')

    print(f'\nDone! Visualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
