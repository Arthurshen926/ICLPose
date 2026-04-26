#!/usr/bin/env python3
"""
Post-hoc floater pruning for 2DGS models.
Removes large-scale, high-opacity Gaussians that appear as ghost artifacts.

Usage:
    python scripts/prune_floaters.py --ply_path <path> [--max_scale 1.0] [--min_opacity 0.3]
"""
import sys, os, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.visualize_2dgs_recon import load_ply_2dgs


def prune_floaters_from_ply(ply_path, output_path, max_scale=1.0, min_opacity_for_prune=0.3):
    """
    Remove floater Gaussians from a PLY file.
    Prunes Gaussians where max(scale) > max_scale AND opacity > min_opacity_for_prune.
    Also prunes near-zero opacity Gaussians (< 0.005).
    """
    from plyfile import PlyData, PlyElement

    print(f"Loading: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    n_orig = len(vertex.data)

    # Extract scale and opacity
    scale_names = [p.name for p in vertex.properties if p.name.startswith('scale_')]
    n_scales = len(scale_names)
    scales_raw = np.stack([vertex[f'scale_{i}'] for i in range(n_scales)], axis=1)
    scales = np.exp(scales_raw)  # actual scale values
    max_scales = scales.max(axis=1)

    opacity_raw = vertex['opacity']
    opacities = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

    # Floater mask: large scale AND high opacity
    floater_mask = (max_scales > max_scale) & (opacities > min_opacity_for_prune)
    n_floaters = floater_mask.sum()

    # Also prune near-transparent Gaussians
    transparent_mask = opacities < 0.005
    n_transparent = transparent_mask.sum()

    prune_mask = floater_mask | transparent_mask
    keep_mask = ~prune_mask
    n_keep = keep_mask.sum()

    print(f"  Original:    {n_orig:,} Gaussians")
    print(f"  Floaters:    {n_floaters:,} (scale>{max_scale} & opacity>{min_opacity_for_prune})")
    print(f"  Transparent: {n_transparent:,} (opacity<0.005)")
    print(f"  Pruned:      {prune_mask.sum():,}")
    print(f"  Kept:        {n_keep:,}")

    # Show stats of pruned floaters
    if n_floaters > 0:
        f_scales = max_scales[floater_mask]
        f_opacities = opacities[floater_mask]
        print(f"\n  Floater stats:")
        print(f"    Scale:   mean={f_scales.mean():.3f}, max={f_scales.max():.3f}")
        print(f"    Opacity: mean={f_opacities.mean():.3f}, max={f_opacities.max():.3f}")

    # Filter and save
    new_data = vertex.data[keep_mask]
    new_element = PlyElement.describe(new_data, 'vertex')
    PlyData([new_element]).write(output_path)
    print(f"\n  Saved: {output_path} ({n_keep:,} Gaussians)")
    return n_keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path',
                        default='output/2dgs_models/OldHospital/v3/point_cloud/iteration_30000/point_cloud.ply')
    parser.add_argument('--output_path', default=None,
                        help='Output path (default: same dir with _pruned suffix)')
    parser.add_argument('--max_scale', type=float, default=1.0,
                        help='Max allowed scale for Gaussians (larger ones with high opacity are pruned)')
    parser.add_argument('--min_opacity', type=float, default=0.3,
                        help='Min opacity to consider a large Gaussian as floater')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite original PLY')
    args = parser.parse_args()

    if args.output_path is None:
        if args.inplace:
            args.output_path = args.ply_path
        else:
            base = args.ply_path.replace('.ply', '_pruned.ply')
            args.output_path = base

    prune_floaters_from_ply(args.ply_path, args.output_path,
                            max_scale=args.max_scale,
                            min_opacity_for_prune=args.min_opacity)


if __name__ == '__main__':
    main()
