#!/usr/bin/env python3
"""Standalone evaluation script for saved checkpoints."""

import sys
import os
import json
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, evaluate, precompute_pooled_features
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to experiment output directory (contains model_best.pt)')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / 'config.json'
    ckpt_path = exp_dir / 'model_best.pt'

    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    device = f'cuda:{args.gpu}'

    # Parse feature mode
    feat_mode = config['feat']
    use_fine = 'fine' in feat_mode or 'both' in feat_mode
    use_coarse = 'coarse' in feat_mode or 'both' in feat_mode
    use_summary = '+sum' in feat_mode

    print(f"Loading {exp_dir.name}...")
    print(f"  pool={config['pool']}, feat={feat_mode}, patch_dim={config['patch_dim']}")
    if 'attn_heads' in config:
        print(f"  attn_heads={config['attn_heads']}")

    # Load data
    train_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'train', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'test', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)

    trans_mean, trans_std = train_data.compute_normalization()
    test_data.normalize_translations(trans_mean, trans_std)

    # Build model with same config
    hidden_dims = config.get('hidden_dims', [1024, 512, 256])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.split(',')]

    model = PatchPoseRegressor(
        pool_type=config['pool'],
        feat_mode=config['feat'],
        patch_dim=config['patch_dim'],
        hidden_dims=hidden_dims,
        dropout=config.get('dropout', 0.15),
        attn_heads=config.get('attn_heads', 4),
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"  Loaded best model from epoch {ckpt.get('epoch', '?')}")
    print(f"  Best val score: {ckpt.get('best_val', '?')}")

    # For attention pooling, need to compute pooled features with learned attention weights
    if config['pool'] in ('attn', 'conv'):
        test_data_device = PatchPoseDataset(
            args.feature_dir, args.dataset_dir, 'test', 'cpu',
            use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
        test_pooled = precompute_pooled_features(model, test_data_device, device, batch_size=16)
        del test_data_device
    else:
        test_pooled = None

    # Move norm params
    trans_mean = trans_mean.to(device)
    trans_std = trans_std.to(device)
    test_data = test_data.to(device)

    # Evaluate
    results, trans_errors, rot_errors, trans_pred = evaluate(
        model, test_data, train_data, trans_mean, trans_std, str(exp_dir),
        test_pooled=test_pooled)

    # Save results
    results_path = exp_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
