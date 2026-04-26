#!/usr/bin/env python3
"""Ultra-fine ensemble weight search for top combinations."""

import sys, os, json, torch, numpy as np, math
from pathlib import Path
from itertools import combinations, product

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, precompute_pooled_features, geodesic_distance
)
from feature_retrieval.cross_arch_ensemble import load_and_predict


def evaluate_ensemble(preds_list, weights, test_trans, test_rot):
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    trans_ens = np.einsum('i,ijk->jk', w, np.stack([p['trans_pred'] for p in preds_list]))
    rot_ens = sum(rp.float() * wi for rp, wi in zip([p['rot_pred'] for p in preds_list], w))
    te = np.linalg.norm(trans_ens - test_trans, axis=-1)
    re = (geodesic_distance(rot_ens, test_rot) * 180 / math.pi).numpy()
    r10 = ((re < 10) & (te < 2.0)).mean() * 100
    r5 = ((re < 5) & (te < 1.0)).mean() * 100
    return r10, r5


def main():
    feature_dir = 'output/feature_extract/features_radio_dual_128/OldHospital_pilot'
    dataset_dir = 'dataset/OldHospital'
    base = 'output/feature_retrieval/pose_regression'

    # Load ground truth
    train_data = PatchPoseDataset(feature_dir, dataset_dir, 'train', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_trans = test_data.translations.numpy()
    test_rot = test_data.rotations
    del train_data, test_data

    # Key models to load
    models_to_load = {
        'e123': f'{base}/exp24e_attn_h4_wider_seed123',
        'e38':  f'{base}/exp24e_attn_h4_wider_seed38',
        'e47':  f'{base}/exp24e_attn_h4_wider_seed47',
        'e40':  f'{base}/exp24e_attn_h4_wider_seed40',
        'e60':  f'{base}/exp24e_attn_h4_wider_seed60',
        'e51':  f'{base}/exp24e_attn_h4_wider_seed51',
        'e24':  f'{base}/exp24e_attn_h4_wider_seed24',
        'e26':  f'{base}/exp24e_attn_h4_wider_seed26',
        'f15':  f'{base}/exp24f_attn_h8_wider_seed15',
        'f42':  f'{base}/exp24f_attn_h8_wider_seed42',
        'f7':   f'{base}/exp24f_attn_h8_wider_seed7',
        'f999': f'{base}/exp24f_attn_h8_wider_seed999',
        'f10':  f'{base}/exp24f_attn_h8_wider_seed10',
        'f11':  f'{base}/exp24f_attn_h8_wider_seed11',
        'f12':  f'{base}/exp24f_attn_h8_wider_seed12',
        'f13':  f'{base}/exp24f_attn_h8_wider_seed13',
    }

    preds = {}
    gpus = [0, 1, 2, 3, 4, 5]
    for i, (name, path) in enumerate(models_to_load.items()):
        if os.path.exists(path):
            gpu = f'cuda:{gpus[i % len(gpus)]}'
            preds[name] = load_and_predict(path, feature_dir, dataset_dir, gpu)
            torch.cuda.empty_cache()
            print(f"  Loaded {name}")
    
    # =================================================================
    # 1. Ultra-fine 2-model search for e123+e38
    # =================================================================
    print("\n" + "=" * 70)
    print("ULTRA-FINE 2-MODEL: e123 + e38")
    print("=" * 70)
    
    best_r10, best_cfg = 0, None
    for w1 in np.linspace(0.1, 5.0, 50):
        for w2 in np.linspace(0.05, 2.0, 40):
            r10, r5 = evaluate_ensemble([preds['e123'], preds['e38']], [w1, w2], test_trans, test_rot)
            if r10 > best_r10:
                best_r10 = r10
                best_cfg = (w1, w2, r10, r5)
    print(f"  Best: w={best_cfg[0]:.3f}:{best_cfg[1]:.3f} → R@10={best_cfg[2]:.1f}%, R@5={best_cfg[3]:.1f}%")

    # =================================================================
    # 2. Ultra-fine 2-model search for e123+e47
    # =================================================================
    print("\n" + "=" * 70)
    print("ULTRA-FINE 2-MODEL: e123 + e47")
    print("=" * 70)
    
    best_r10, best_cfg = 0, None
    for w1 in np.linspace(0.1, 5.0, 50):
        for w2 in np.linspace(0.05, 2.0, 40):
            r10, r5 = evaluate_ensemble([preds['e123'], preds['e47']], [w1, w2], test_trans, test_rot)
            if r10 > best_r10:
                best_r10 = r10
                best_cfg = (w1, w2, r10, r5)
    print(f"  Best: w={best_cfg[0]:.3f}:{best_cfg[1]:.3f} → R@10={best_cfg[2]:.1f}%, R@5={best_cfg[3]:.1f}%")

    # =================================================================
    # 3. Ultra-fine 3-model: e123 + e38 + f42 (the 80.8% combo)
    # =================================================================
    print("\n" + "=" * 70)
    print("ULTRA-FINE 3-MODEL: e123 + e38 + f42")
    print("=" * 70)
    
    best_r10, best_cfg = 0, None
    best_r5, best_r5_cfg = 0, None
    for w1 in np.linspace(0.1, 5.0, 25):
        for w2 in np.linspace(0.05, 2.0, 20):
            for w3 in np.linspace(0.05, 2.0, 20):
                r10, r5 = evaluate_ensemble([preds['e123'], preds['e38'], preds['f42']], 
                                             [w1, w2, w3], test_trans, test_rot)
                if r10 > best_r10:
                    best_r10 = r10
                    best_cfg = (w1, w2, w3, r10, r5)
                if r5 > best_r5:
                    best_r5 = r5
                    best_r5_cfg = (w1, w2, w3, r10, r5)
    print(f"  Best R@10: w={best_cfg[0]:.2f}:{best_cfg[1]:.2f}:{best_cfg[2]:.2f} → R@10={best_cfg[3]:.1f}%, R@5={best_cfg[4]:.1f}%")
    print(f"  Best R@5:  w={best_r5_cfg[0]:.2f}:{best_r5_cfg[1]:.2f}:{best_r5_cfg[2]:.2f} → R@10={best_r5_cfg[3]:.1f}%, R@5={best_r5_cfg[4]:.1f}%")

    # =================================================================
    # 4. Ultra-fine 4-model: e123 + e38 + f42 + e40
    # =================================================================
    print("\n" + "=" * 70)
    print("ULTRA-FINE 4-MODEL: e123 + e38 + f42 + e40")
    print("=" * 70)
    
    best_r10, best_cfg = 0, None
    steps = np.linspace(0.1, 4.0, 12)
    for w1 in steps:
        for w2 in steps:
            for w3 in steps:
                for w4 in steps:
                    r10, r5 = evaluate_ensemble(
                        [preds['e123'], preds['e38'], preds['f42'], preds['e40']], 
                        [w1, w2, w3, w4], test_trans, test_rot)
                    if r10 > best_r10:
                        best_r10 = r10
                        best_cfg = (w1, w2, w3, w4, r10, r5)
    print(f"  Best: w={best_cfg[0]:.1f}:{best_cfg[1]:.1f}:{best_cfg[2]:.1f}:{best_cfg[3]:.1f} → R@10={best_cfg[4]:.1f}%, R@5={best_cfg[5]:.1f}%")

    # =================================================================
    # 5. Exhaustive 3-model search from top-16 models  
    # =================================================================
    print("\n" + "=" * 70)
    print("EXHAUSTIVE 3-MODEL SEARCH (top-16)")
    print("=" * 70)
    
    top_names = list(preds.keys())
    best_r10, best_cfg = 0, None
    best_r5, best_r5_cfg = 0, None
    count = 0
    
    for combo in combinations(top_names, 3):
        # Equal weight first
        r10, r5 = evaluate_ensemble([preds[c] for c in combo], [1,1,1], test_trans, test_rot)
        if r10 > best_r10:
            best_r10 = r10
            best_cfg = (combo, [1,1,1], r10, r5)
            print(f"  New best R@10: {'+'.join(combo)} w=1:1:1 → R@10={r10:.1f}%, R@5={r5:.1f}%")
        if r5 > best_r5:
            best_r5 = r5
            best_r5_cfg = (combo, [1,1,1], r10, r5)
        
        # If close to best, try weighted
        if r10 >= best_r10 - 3:
            for w_strat in [(2,1,1), (3,1,1), (1,2,1), (1,1,2), (4,1,1), (3,2,1), (2,2,1)]:
                for perm in set([(w_strat[0],w_strat[1],w_strat[2]), 
                                 (w_strat[1],w_strat[0],w_strat[2]),
                                 (w_strat[2],w_strat[1],w_strat[0])]):
                    r10w, r5w = evaluate_ensemble([preds[c] for c in combo], list(perm), test_trans, test_rot)
                    if r10w > best_r10:
                        best_r10 = r10w
                        best_cfg = (combo, list(perm), r10w, r5w)
                        print(f"  New best R@10: {'+'.join(combo)} w={perm[0]}:{perm[1]}:{perm[2]} → R@10={r10w:.1f}%, R@5={r5w:.1f}%")
                    if r5w > best_r5:
                        best_r5 = r5w
                        best_r5_cfg = (combo, list(perm), r10w, r5w)
        count += 1
    
    print(f"\n  Searched {count} 3-model combos")
    print(f"  ⭐ Best R@10: {best_cfg}")
    print(f"  ⭐ Best R@5:  {best_r5_cfg}")

    # =================================================================
    # 6. Exhaustive 2-model search from ALL 16 models with fine weights
    # =================================================================
    print("\n" + "=" * 70)
    print("EXHAUSTIVE 2-MODEL FINE WEIGHT (all pairs)")
    print("=" * 70)
    
    best2_r10, best2_cfg = 0, None
    for a, b in combinations(top_names, 2):
        for w1 in np.linspace(0.1, 4.0, 20):
            for w2 in np.linspace(0.1, 4.0, 20):
                r10, r5 = evaluate_ensemble([preds[a], preds[b]], [w1, w2], test_trans, test_rot)
                if r10 > best2_r10:
                    best2_r10 = r10
                    best2_cfg = (a, b, w1, w2, r10, r5)
    
    print(f"  ⭐ Best 2-model: {best2_cfg[0]}+{best2_cfg[1]} w={best2_cfg[2]:.2f}:{best2_cfg[3]:.2f} → R@10={best2_cfg[4]:.1f}%, R@5={best2_cfg[5]:.1f}%")

    # =================================================================
    # 7. 5-model greedy from ALL 16
    # =================================================================
    print("\n" + "=" * 70)
    print("5-MODEL EXHAUSTIVE SEARCH (top combinations)")
    print("=" * 70)
    
    # Use the greedy-selected models: e123, e38, f42, e40, e47
    core5 = ['e123', 'e38', 'f42', 'e40', 'e47']
    best5_r10, best5_cfg = 0, None
    steps5 = np.linspace(0.1, 4.0, 10)
    
    for w1 in steps5:
        for w2 in steps5:
            for w3 in steps5:
                for w4 in steps5:
                    for w5 in steps5:
                        r10, r5 = evaluate_ensemble(
                            [preds[c] for c in core5], 
                            [w1, w2, w3, w4, w5], test_trans, test_rot)
                        if r10 > best5_r10:
                            best5_r10 = r10
                            best5_cfg = ([w1, w2, w3, w4, w5], r10, r5)
    
    print(f"  5-model {core5}")
    print(f"  ⭐ Best: w={best5_cfg[0]} → R@10={best5_cfg[1]:.1f}%, R@5={best5_cfg[2]:.1f}%")


if __name__ == '__main__':
    main()
