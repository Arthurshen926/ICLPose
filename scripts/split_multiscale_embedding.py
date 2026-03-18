#!/usr/bin/env python3
"""
 train_multiscale_embedding_v2.py 保存的联合 checkpoint 拆分为
MultiScaleRenderer 所需的各尺度独立 checkpoint。

:
    python scripts/split_multiscale_embedding.py \
        --ckpt output/feature_3dgs/stairs_v3/best_model.pth \
        --output_dir output/feature_3dgs/stairs_v3/per_scale
"""
import argparse
import torch
from pathlib import Path

SCALE_RESOLUTIONS = {
    'coarse':    (15, 20),   # room_0 default (480×640 → 4:3 aspect)
    'mid':       (30, 40),
    'fine_sd':   (35, 46),
    'fine_dino': (35, 46),
}

# Stride-7 resolutions (DINOv2 stride 14→7, 2× fine resolution)
STRIDE7_RESOLUTIONS = {
    'coarse':    (15, 20),
    'mid':       (30, 40),
    'fine_sd':   (69, 91),
    'fine_dino': (69, 91),
}

# Stairs-specific resolutions (480x640)
STAIRS_RESOLUTIONS = {
    'coarse':    (15, 20),
    'mid':       (30, 40),
    'fine_sd':   (35, 46),
    'fine_dino': (35, 46),
}

# OldHospital resolutions (1080x1920 → 16:9 aspect)
OLDHOSPITAL_RESOLUTIONS = {
    'coarse':    (15, 26),
    'mid':       (30, 53),
    'fine_sd':   (35, 61),
    'fine_dino': (35, 61),
}

# OldHospital stride-7 resolutions (DINOv2 stride 14→7)
OLDHOSPITAL_STRIDE7_RESOLUTIONS = {
    'coarse':    (15, 26),
    'mid':       (30, 53),
    'fine_sd':   (69, 121),
    'fine_dino': (69, 121),
}

# Stairs stride-7 resolutions (DINOv2 stride 14→7, 480×640)
STAIRS_STRIDE7_RESOLUTIONS = {
    'coarse':    (15, 20),
    'mid':       (30, 40),
    'fine_sd':   (69, 91),
    'fine_dino': (69, 91),
}


def split_checkpoint(ckpt_path: str, output_dir: str, resolutions: dict = None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    loc_feature = ckpt['loc_feature']   # (N, 224)
    scale_dims = ckpt.get('scale_dims', {
        'fine_sd': 64, 'fine_dino': 64, 'mid': 64, 'coarse': 32
    })
    
    print(f"Loaded: {ckpt_path}")
    print(f"  loc_feature: {loc_feature.shape}")
    print(f"  scale_dims: {scale_dims}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split along feature dimension
    # Order must match training order: fine_sd, fine_dino, mid, coarse
    scale_order = ['fine_sd', 'fine_dino', 'mid', 'coarse']
    offset = 0
    
    for scale_name in scale_order:
        dim = scale_dims.get(scale_name, 64)
        feat_slice = loc_feature[:, offset:offset + dim]
        offset += dim
        
        res = (resolutions or SCALE_RESOLUTIONS).get(scale_name, (35, 46))
        
        out_ckpt = {
            'iteration': ckpt.get('iteration', 0),
            'loc_feature': feat_slice,       # (N, D_scale)
            'feature_dim': dim,
            'resolution': list(res),
            'loss': ckpt.get('loss', 0.0),
        }
        
        out_path = output_dir / f'{scale_name}.pth'
        torch.save(out_ckpt, str(out_path))
        print(f"  Saved {scale_name}: {feat_slice.shape} → {out_path}")
    
    print(f"Done. Per-scale checkpoints in: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='Path to best_model.pth or final_model.pth')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for per-scale checkpoints')
    parser.add_argument('--scene', default=None,
                        choices=['stairs', 'oldhospital', 'room_0', 'stride7', 'oldhospital_stride7', 'stairs_stride7'],
                        help='Use preset resolutions for known scenes')
    args = parser.parse_args()
    
    if args.scene == 'stairs':
        resolutions = STAIRS_RESOLUTIONS
    elif args.scene == 'oldhospital':
        resolutions = OLDHOSPITAL_RESOLUTIONS
    elif args.scene == 'stride7':
        resolutions = STRIDE7_RESOLUTIONS
    elif args.scene == 'oldhospital_stride7':
        resolutions = OLDHOSPITAL_STRIDE7_RESOLUTIONS
    elif args.scene == 'stairs_stride7':
        resolutions = STAIRS_STRIDE7_RESOLUTIONS
    else:
        resolutions = SCALE_RESOLUTIONS  # default room_0
    
    split_checkpoint(args.ckpt, args.output_dir, resolutions)


if __name__ == '__main__':
    main()
