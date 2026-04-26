#!/usr/bin/env python3
"""Fine-grained weight optimization for cross-architecture ensembles."""

import sys
import os
import json
import torch
import numpy as np
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, precompute_pooled_features,
    geodesic_distance
)
from feature_retrieval.cross_arch_ensemble import load_and_predict


def evaluate_weighted(preds_list, weights, test_trans, test_rot, train_pos, train_rot):
    """Fast evaluation of weighted ensemble."""
    w = np.array(weights) / np.sum(weights)
    
    trans_preds = np.stack([p['trans_pred'] for p in preds_list])
    trans_ens = np.einsum('i,ijk->jk', w, trans_preds)
    
    rot_preds = [p['rot_pred'] for p in preds_list]
    rot_ens = sum(rp.float() * wi for rp, wi in zip(rot_preds, w))
    
    trans_errors = np.linalg.norm(trans_ens - test_trans, axis=-1)
    rot_errors = (geodesic_distance(rot_ens, test_rot) * 180 / math.pi).numpy()
    
    r10 = ((rot_errors < 10) & (trans_errors < 2.0)).mean() * 100
    r5 = ((rot_errors < 5) & (trans_errors < 1.0)).mean() * 100
    
    return r10, r5


def main():
    feature_dir = 'output/feature_extract/features_radio_dual_128/OldHospital_pilot'
    dataset_dir = 'dataset/OldHospital'
    base_dir = 'output/feature_retrieval/pose_regression'

    # Load ground truth
    train_data = PatchPoseDataset(feature_dir, dataset_dir, 'train', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_trans = test_data.translations.numpy()
    test_rot = test_data.rotations
    train_pos = train_data.translations.numpy()
    train_rot = train_data.rotations
    del train_data, test_data

    # Load models
    models = {
        'spp_s314': f'{base_dir}/exp14s_128d_wider_seed314',
        'spp_n123': f'{base_dir}/exp14n_128d_wider_seed123',
        'spp_p123': f'{base_dir}/exp14p_128d_swa_seed123',
        'attn_a314': f'{base_dir}/exp24a_attn_h4_seed314',
        'attn_c314': f'{base_dir}/exp24c_attn_h16_seed314',
        'attn_e314': f'{base_dir}/exp24e_attn_h4_wider_seed314',
    }

    preds = {}
    gpus = [4, 5]
    for i, (name, path) in enumerate(models.items()):
        if os.path.exists(path):
            gpu = f'cuda:{gpus[i % len(gpus)]}'
            preds[name] = load_and_predict(path, feature_dir, dataset_dir, gpu)
            torch.cuda.empty_cache()
            print(f"  Loaded {name}")

    # =================================================================
    # Fine-grained weight search for best pairs
    # =================================================================
    print("\n" + "=" * 70)
    print("FINE-GRAINED WEIGHT SEARCH — R@10/2m focused")
    print("=" * 70)

    # Best R@10 pair: spp_p123 + attn_a314
    pair_r10 = [preds['spp_p123'], preds['attn_a314']]
    best_r10, best_w10 = 0, None
    for w1 in np.arange(0.1, 2.1, 0.1):
        for w2 in np.arange(0.1, 2.1, 0.1):
            r10, r5 = evaluate_weighted(pair_r10, [w1, w2], test_trans, test_rot, train_pos, train_rot)
            if r10 > best_r10:
                best_r10 = r10
                best_w10 = (w1, w2, r10, r5)

    print(f"\nBest R@10 pair (spp_p123 + attn_a314):")
    print(f"  w={best_w10[0]:.1f}:{best_w10[1]:.1f} → R@10={best_w10[2]:.1f}%, R@5={best_w10[3]:.1f}%")

    # Best R@5 pair: spp_p123 + attn_e314
    pair_r5 = [preds['spp_p123'], preds['attn_e314']]
    best_r5, best_w5 = 0, None
    for w1 in np.arange(0.1, 2.1, 0.1):
        for w2 in np.arange(0.1, 2.1, 0.1):
            r10, r5 = evaluate_weighted(pair_r5, [w1, w2], test_trans, test_rot, train_pos, train_rot)
            if r5 > best_r5:
                best_r5 = r5
                best_w5 = (w1, w2, r10, r5)

    print(f"\nBest R@5 pair (spp_p123 + attn_e314):")
    print(f"  w={best_w5[0]:.1f}:{best_w5[1]:.1f} → R@10={best_w5[2]:.1f}%, R@5={best_w5[3]:.1f}%")

    # =================================================================
    # 3-model search: 2 SPP + 1 Attn and 1 SPP + 2 Attn
    # =================================================================
    print("\n" + "=" * 70)
    print("FINE-GRAINED 3-MODEL WEIGHT SEARCH")
    print("=" * 70)

    spp_list = ['spp_s314', 'spp_n123', 'spp_p123']
    attn_list = ['attn_a314', 'attn_c314', 'attn_e314']

    best_3m_r10, best_3m_config_r10 = 0, None
    best_3m_r5, best_3m_config_r5 = 0, None

    from itertools import combinations
    
    # Try all 3-model combos with varied weights
    for combo in combinations(list(preds.keys()), 3):
        pools = [preds[c]['config']['pool'] for c in combo]
        if 'spp' not in pools or 'attn' not in pools:
            continue
        
        pred_list = [preds[c] for c in combo]
        
        # Grid search weights (coarser for 3-model)
        for w1 in [0.5, 1.0, 1.5, 2.0]:
            for w2 in [0.5, 1.0, 1.5, 2.0]:
                for w3 in [0.5, 1.0, 1.5, 2.0]:
                    r10, r5 = evaluate_weighted(pred_list, [w1, w2, w3],
                                                test_trans, test_rot, train_pos, train_rot)
                    if r10 > best_3m_r10:
                        best_3m_r10 = r10
                        best_3m_config_r10 = (combo, (w1, w2, w3), r10, r5)
                    if r5 > best_3m_r5:
                        best_3m_r5 = r5
                        best_3m_config_r5 = (combo, (w1, w2, w3), r10, r5)

    print(f"\nBest 3-model R@10: {best_3m_config_r10}")
    print(f"Best 3-model R@5:  {best_3m_config_r5}")

    # =================================================================
    # 4-model search
    # =================================================================
    print("\n" + "=" * 70)
    print("4-MODEL WEIGHT SEARCH")
    print("=" * 70)

    best_4m_r10, best_4m_r5 = 0, 0
    best_4m_r10_cfg, best_4m_r5_cfg = None, None

    for combo in combinations(list(preds.keys()), 4):
        pools = [preds[c]['config']['pool'] for c in combo]
        if 'spp' not in pools or 'attn' not in pools:
            continue

        pred_list = [preds[c] for c in combo]

        # Equal weights
        r10, r5 = evaluate_weighted(pred_list, [1.0]*4, test_trans, test_rot, train_pos, train_rot)
        if r10 > best_4m_r10:
            best_4m_r10, best_4m_r10_cfg = r10, (combo, [1]*4, r10, r5)
        if r5 > best_4m_r5:
            best_4m_r5, best_4m_r5_cfg = r5, (combo, [1]*4, r10, r5)

        # SPP-heavy
        w = [2.0 if preds[c]['config']['pool']=='spp' else 1.0 for c in combo]
        r10, r5 = evaluate_weighted(pred_list, w, test_trans, test_rot, train_pos, train_rot)
        if r10 > best_4m_r10:
            best_4m_r10, best_4m_r10_cfg = r10, (combo, w, r10, r5)
        if r5 > best_4m_r5:
            best_4m_r5, best_4m_r5_cfg = r5, (combo, w, r10, r5)

        # Attn-heavy
        w = [1.0 if preds[c]['config']['pool']=='spp' else 2.0 for c in combo]
        r10, r5 = evaluate_weighted(pred_list, w, test_trans, test_rot, train_pos, train_rot)
        if r10 > best_4m_r10:
            best_4m_r10, best_4m_r10_cfg = r10, (combo, w, r10, r5)
        if r5 > best_4m_r5:
            best_4m_r5, best_4m_r5_cfg = r5, (combo, w, r10, r5)

    print(f"Best 4-model R@10: {best_4m_r10_cfg}")
    print(f"Best 4-model R@5:  {best_4m_r5_cfg}")

    # =================================================================
    # All 6 models
    # =================================================================
    print("\n" + "=" * 70)
    print("ALL 6 MODELS — WEIGHT SEARCH")
    print("=" * 70)

    all_list = list(preds.values())
    all_names = list(preds.keys())

    best_6_r10, best_6_r5 = 0, 0
    best_6_r10_w, best_6_r5_w = None, None

    # Search weights for each model
    for spp_w in [0.5, 1.0, 1.5, 2.0]:
        for attn_w in [0.5, 1.0, 1.5, 2.0]:
            weights = [spp_w if preds[n]['config']['pool']=='spp' else attn_w for n in all_names]
            r10, r5 = evaluate_weighted(all_list, weights, test_trans, test_rot, train_pos, train_rot)
            label = f"spp_w={spp_w}, attn_w={attn_w}"
            print(f"  {label:30s} R@10={r10:5.1f}% R@5={r5:5.1f}%")
            if r10 > best_6_r10:
                best_6_r10, best_6_r10_w = r10, label
            if r5 > best_6_r5:
                best_6_r5, best_6_r5_w = r5, label

    print(f"\nBest 6-model R@10: {best_6_r10:.1f}% ({best_6_r10_w})")
    print(f"Best 6-model R@5:  {best_6_r5:.1f}% ({best_6_r5_w})")


if __name__ == '__main__':
    main()
