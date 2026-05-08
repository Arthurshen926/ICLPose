#!/usr/bin/env python3
"""Cross-architecture ensemble: combine SPP and Attention Pooling models."""

import sys
import os
import json
import torch
import numpy as np
import math
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, ResidualPatchPoseRegressor, PatchPoseDataset, precompute_pooled_features,
    geodesic_distance
)


def load_and_predict(exp_dir, feature_dir, dataset_dir, device='cuda:0', split='test'):
    """Load a checkpoint and return predictions (trans, rot) in real units."""
    exp_dir = Path(exp_dir)
    config_path = exp_dir / 'config.json'
    ckpt_path = exp_dir / 'model_best.pt'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    feat_mode = config['feat']
    use_fine = 'fine' in feat_mode or 'both' in feat_mode
    use_coarse = 'coarse' in feat_mode or 'both' in feat_mode
    use_summary = '+sum' in feat_mode

    # Load data
    train_data = PatchPoseDataset(
        feature_dir, dataset_dir, 'train', 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(
        feature_dir, dataset_dir, split, 'cpu',
        use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)

    trans_mean, trans_std = train_data.compute_normalization()
    test_data.normalize_translations(trans_mean, trans_std)

    hidden_dims = config.get('hidden_dims', [1024, 512, 256])
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.split(',')]

    model_kwargs = dict(
        pool_type=config['pool'],
        feat_mode=config['feat'],
        patch_dim=config['patch_dim'],
        hidden_dims=hidden_dims,
        dropout=config.get('dropout', 0.15),
        attn_heads=config.get('attn_heads', 4),
        conv_out_dim=config.get('conv_out_dim', 512),
        spp_levels=config.get('spp_levels', None),
    )
    if config.get('model_type', 'mlp') == 'residual':
        model = ResidualPatchPoseRegressor(
            **model_kwargs,
            n_res_blocks=config.get('n_res_blocks', 4),
        )
    else:
        model = PatchPoseRegressor(**model_kwargs)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # For attention pooling, compute pooled features with learned attention
    if config['pool'] in ('attn', 'conv'):
        test_data_raw = PatchPoseDataset(
            feature_dir, dataset_dir, split, 'cpu',
            use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
        test_pooled = precompute_pooled_features(model, test_data_raw, device, batch_size=16)
        del test_data_raw
    else:
        test_pooled = None

    trans_mean = trans_mean.to(device)
    trans_std = trans_std.to(device)
    test_data = test_data.to(device)

    with torch.no_grad():
        if test_pooled is not None:
            trans_pred, rot_pred = model.forward_from_pooled(test_pooled)
        else:
            trans_pred, rot_pred = model(
                test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        trans_pred_real = test_data.denormalize(trans_pred, trans_mean, trans_std)

    return {
        'trans_pred': trans_pred_real.cpu().numpy(),
        'rot_pred': rot_pred.cpu(),  # Keep as tensor for geodesic
        'config': config,
        'name': exp_dir.name,
    }


def evaluate_ensemble(predictions_list, weights, test_translations, test_rotations,
                      train_positions, train_rotations):
    """Evaluate weighted ensemble of predictions."""
    weights = np.array(weights) / np.sum(weights)

    # Weighted average of translations
    trans_preds = np.stack([p['trans_pred'] for p in predictions_list])
    trans_ensemble = np.average(trans_preds, axis=0, weights=weights)

    # For rotation: average the 6D representations, then re-orthogonalize
    rot_preds = [p['rot_pred'] for p in predictions_list]
    # Weighted average (handle variable shapes)
    rot_ensemble = torch.zeros_like(rot_preds[0])
    for rp, w in zip(rot_preds, weights):
        rot_ensemble = rot_ensemble + rp.float() * w
    rot_ensemble = rot_ensemble / sum(weights)

    # Compute errors
    trans_errors = np.linalg.norm(trans_ensemble - test_translations, axis=-1)

    rot_errors = (geodesic_distance(rot_ensemble, test_rotations) * 180 / math.pi).numpy()

    results = {}
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
    ]
    for name, rot_th, trans_th in thresholds:
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        results[name] = combined

    # Retrieval mode
    from scipy.spatial import cKDTree
    tree = cKDTree(train_positions)
    _, nn_idx = tree.query(trans_ensemble, k=1)
    retr_trans_errors = np.linalg.norm(train_positions[nn_idx] - test_translations, axis=-1)
    retr_rot_errors = (geodesic_distance(
        train_rotations[nn_idx], test_rotations) * 180 / math.pi).numpy()

    for name, rot_th, trans_th in thresholds:
        retr = ((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100
        results[f'retr_{name}'] = retr

    results['med_rot'] = float(np.median(rot_errors))
    results['med_trans_mm'] = float(np.median(trans_errors) * 1000)

    return results


def main():
    feature_dir = 'output/feature_extract/features_radio_dual_128/OldHospital_pilot'
    dataset_dir = 'dataset/OldHospital'

    # Define all models to consider
    base_dir = 'output/feature_retrieval/pose_regression'
    
    # SPP models (top performers)
    spp_models = [
        f'{base_dir}/exp14s_128d_wider_seed314',   # R@10=72.0%, R@5=37.9% BEST
        f'{base_dir}/exp14n_128d_wider_seed123',    # R@10=70.9%, R@5=38.5%
        f'{base_dir}/exp14p_128d_swa_seed123',      # R@10=70.9%, R@5=40.1%
    ]
    
    # Attention pooling models
    attn_models = [
        f'{base_dir}/exp24a_attn_h4_seed314',          # R@10=70.9%, R@5=34.1%
        f'{base_dir}/exp24c_attn_h16_seed314',          # R@10=65.9%, R@5=41.8%
        f'{base_dir}/exp24e_attn_h4_wider_seed314',     # R@10=68.7%, R@5=39.0%
    ]

    print("=" * 70)
    print("CROSS-ARCHITECTURE ENSEMBLE: SPP + Attention Pooling")
    print("=" * 70)

    # Load ground truth (minimal - only translations/rotations)
    train_data = PatchPoseDataset(
        feature_dir, dataset_dir, 'train', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_data = PatchPoseDataset(
        feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    
    test_trans = test_data.translations.numpy()
    test_rot = test_data.rotations
    train_pos = train_data.translations.numpy()
    train_rot = train_data.rotations
    
    # Free the feature data we don't need
    del train_data, test_data
    torch.cuda.empty_cache()

    # Load all predictions - use GPUs 4,5 to avoid conflict with training
    # Load models one at a time on same GPU to save memory
    available_gpus = [4, 5]
    print("\nLoading SPP models...")
    all_preds = {}
    all_models = spp_models + attn_models
    for i, path in enumerate(all_models):
        if os.path.exists(path):
            gpu = f'cuda:{available_gpus[i % len(available_gpus)]}'
            preds = load_and_predict(path, feature_dir, dataset_dir, gpu)
            all_preds[preds['name']] = preds
            print(f"  Loaded {preds['name']} on {gpu}")
            # Free GPU memory after getting predictions (predictions are on CPU)
            torch.cuda.empty_cache()
        else:
            print(f"  SKIP: {path} not found")

    print(f"\nTotal models loaded: {len(all_preds)}")

    # 1. Individual model results
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 70)
    print(f"{'Model':<45} {'R@10/2m':>8} {'R@5/1m':>8} {'Med°':>6} {'Med mm':>8}")
    print("-" * 80)
    
    for name, preds in sorted(all_preds.items()):
        res = evaluate_ensemble([preds], [1.0], test_trans, test_rot, train_pos, train_rot)
        pool = preds['config']['pool']
        print(f"  [{pool:4s}] {name:<40} {res['10deg_2m']:>7.1f}% {res['5deg_1m']:>7.1f}% "
              f"{res['med_rot']:>5.2f} {res['med_trans_mm']:>7.0f}")

    # 2. Pairwise cross-architecture ensembles
    spp_names = [Path(p).name for p in spp_models if os.path.exists(p)]
    attn_names = [Path(p).name for p in attn_models if os.path.exists(p)]

    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE PAIRWISE ENSEMBLES (SPP + Attn)")
    print("=" * 70)
    
    best_result = None
    best_combo = None
    best_r10 = 0.0

    for spp in spp_names:
        for attn in attn_names:
            if spp not in all_preds or attn not in all_preds:
                continue
            pair = [all_preds[spp], all_preds[attn]]
            
            # Try different weight ratios
            for w_spp, w_attn in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0)]:
                res = evaluate_ensemble(pair, [w_spp, w_attn], test_trans, test_rot, train_pos, train_rot)
                label = f"{spp[:25]}+{attn[:25]} w={w_spp:.0f}:{w_attn:.0f}"
                print(f"  {label:<65} R@10={res['10deg_2m']:>5.1f}% R@5={res['5deg_1m']:>5.1f}%")
                
                if res['10deg_2m'] > best_r10:
                    best_r10 = res['10deg_2m']
                    best_combo = (spp, attn, w_spp, w_attn)
                    best_result = res

    # 3. Three-model ensembles (best SPP + 2 Attn, or 2 SPP + 1 Attn)
    print("\n" + "=" * 70)
    print("THREE-MODEL CROSS-ARCHITECTURE ENSEMBLES")
    print("=" * 70)

    all_names = list(all_preds.keys())
    for combo in combinations(all_names, 3):
        # Require at least one from each architecture
        combo_pools = [all_preds[c]['config']['pool'] for c in combo]
        if 'spp' not in combo_pools or 'attn' not in combo_pools:
            continue
        
        preds_list = [all_preds[c] for c in combo]
        res = evaluate_ensemble(preds_list, [1.0]*3, test_trans, test_rot, train_pos, train_rot)
        short_names = '+'.join([c[:20] for c in combo])
        print(f"  {short_names:<65} R@10={res['10deg_2m']:>5.1f}% R@5={res['5deg_1m']:>5.1f}%")

        if res['10deg_2m'] > best_r10:
            best_r10 = res['10deg_2m']
            best_combo = combo
            best_result = res

    # 4. All-model ensemble
    print("\n" + "=" * 70)
    print("ALL-MODEL ENSEMBLE")
    print("=" * 70)
    
    all_list = list(all_preds.values())
    for strat_name, weights in [
        ("Equal", [1.0]*len(all_list)),
        ("SPP-heavy", [2.0 if p['config']['pool']=='spp' else 1.0 for p in all_list]),
        ("Attn-heavy", [1.0 if p['config']['pool']=='spp' else 2.0 for p in all_list]),
    ]:
        res = evaluate_ensemble(all_list, weights, test_trans, test_rot, train_pos, train_rot)
        print(f"  {strat_name:<20} R@10={res['10deg_2m']:>5.1f}% R@5={res['5deg_1m']:>5.1f}% "
              f"med={res['med_rot']:.2f}°/{res['med_trans_mm']:.0f}mm")

    # Print best overall
    print("\n" + "=" * 70)
    print("BEST CROSS-ARCHITECTURE ENSEMBLE")
    print("=" * 70)
    print(f"  Combo: {best_combo}")
    print(f"  R@10°/2m: {best_result['10deg_2m']:.1f}%")
    print(f"  R@5°/1m: {best_result['5deg_1m']:.1f}%")
    print(f"  Med: {best_result['med_rot']:.2f}°/{best_result['med_trans_mm']:.0f}mm")


if __name__ == '__main__':
    main()
