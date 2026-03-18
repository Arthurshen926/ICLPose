#!/usr/bin/env python3
"""
Split a joint multi-scale 3DGS embedding model into per-scale model files.

The MultiScaleRenderer expects separate per-scale models:
  renderer.scale_model_paths:
    coarse: .../coarse.pth
    mid: .../mid.pth
    fine_sd: .../fine_sd.pth
    fine_dino: .../fine_dino.pth

But train_multiscale_embedding_v2 saves a single joint model.
This script splits it.

Usage:
    python scripts/split_joint_model.py \
        --joint_model output/feature_3dgs/oldhospital_pca_v1/best_model.pth \
        --output_dir output/feature_3dgs/oldhospital_pca_v1/per_scale
"""
import argparse, os, torch

# Scale dimension layout (from MultiScaleGaussianModel)
FINE_SD_DIM = 64
FINE_DINO_DIM = 64
MID_DIM = 64
COARSE_DIM = 32

FINE_SD_START = 0
FINE_SD_END = FINE_SD_DIM           # 64
FINE_DINO_START = FINE_SD_END       # 64
FINE_DINO_END = FINE_DINO_START + FINE_DINO_DIM  # 128
MID_START = FINE_DINO_END           # 128
MID_END = MID_START + MID_DIM      # 192
COARSE_START = MID_END              # 192
COARSE_END = COARSE_START + COARSE_DIM  # 224


SCALES = {
    'coarse':    (COARSE_START, COARSE_END, COARSE_DIM),
    'mid':       (MID_START, MID_END, MID_DIM),
    'fine_sd':   (FINE_SD_START, FINE_SD_END, FINE_SD_DIM),
    'fine_dino': (FINE_DINO_START, FINE_DINO_END, FINE_DINO_DIM),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_model', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.joint_model, map_location='cpu')
    loc_feature = ckpt['loc_feature']  # (N, 224)
    N = loc_feature.shape[0]
    print(f"Joint model: {N} Gaussians, {loc_feature.shape[1]} dims")
    print(f"  Iteration: {ckpt.get('iteration', '?')}")
    print(f"  Loss: {ckpt.get('loss', '?')}")

    # Get resolution from scale_dims if available
    scale_dims = ckpt.get('scale_dims', {})

    for name, (start, end, dim) in SCALES.items():
        feat = loc_feature[:, start:end]  # (N, dim)
        assert feat.shape == (N, dim), f"Shape mismatch for {name}: {feat.shape}"

        # Resolution from joint model or defaults
        res_map = {
            'coarse': [15, 26],
            'mid': [30, 53],
            'fine_sd': [69, 121],
            'fine_dino': [69, 121],
        }

        out_path = os.path.join(args.output_dir, f'{name}.pth')
        torch.save({
            'iteration': ckpt.get('iteration', 0),
            'loc_feature': feat,
            'loss': ckpt.get('loss', 0),
            'feature_dim': dim,
            'resolution': res_map[name],
        }, out_path)
        print(f"  {name}: dim={dim}, shape={feat.shape} → {out_path}")

    print("Done!")


if __name__ == '__main__':
    main()
