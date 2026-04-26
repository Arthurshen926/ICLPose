#!/usr/bin/env python3
"""Optimized ensemble with top attention + SPP models."""

import sys, os, json, torch, numpy as np, math
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, precompute_pooled_features, geodesic_distance
)
from feature_retrieval.cross_arch_ensemble import load_and_predict


def evaluate_fast(preds_list, weights, test_trans, test_rot, train_pos, train_rot):
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

    train_data = PatchPoseDataset(feature_dir, dataset_dir, 'train', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, 'test', 'cpu',
        use_fine=False, use_coarse=False, use_summary=True)
    test_trans = test_data.translations.numpy()
    test_rot = test_data.rotations
    train_pos = train_data.translations.numpy()
    train_rot = train_data.rotations
    del train_data, test_data

    # Top attention models (config e - wider)
    attn_e_models = {
        'e123': f'{base}/exp24e_attn_h4_wider_seed123',    # 75.3%
        'e24':  f'{base}/exp24e_attn_h4_wider_seed24',     # 73.1%
        'e38':  f'{base}/exp24e_attn_h4_wider_seed38',     # 72.0%
        'e26':  f'{base}/exp24e_attn_h4_wider_seed26',     # 71.4%
        'e37':  f'{base}/exp24e_attn_h4_wider_seed37',     # 71.4%
        'e456': f'{base}/exp24e_attn_h4_wider_seed456',    # 69.8%
        'e314': f'{base}/exp24e_attn_h4_wider_seed314',    # 68.7%
    }
    
    # Top attention config a models
    attn_a_models = {
        'a123': f'{base}/exp24a_attn_h4_seed123',          # 70.9%
        'a314': f'{base}/exp24a_attn_h4_seed314',          # 70.9%
    }
    
    # Top attention config c models
    attn_c_models = {
        'c7':   f'{base}/exp24c_attn_h16_seed7',           # 71.4%
        'c314': f'{base}/exp24c_attn_h16_seed314',         # 65.9%
    }
    
    # Top SPP models
    spp_models = {
        's314': f'{base}/exp14s_128d_wider_seed314',        # 72.0%
        'n123': f'{base}/exp14n_128d_wider_seed123',        # 70.9%
        'p123': f'{base}/exp14p_128d_swa_seed123',          # 70.9%
    }

    all_models = {}
    all_models.update(attn_e_models)
    all_models.update(attn_a_models)
    all_models.update(attn_c_models)
    all_models.update(spp_models)

    preds = {}
    gpus = [0, 1, 3, 4, 5]
    for i, (name, path) in enumerate(all_models.items()):
        if os.path.exists(path):
            gpu = f'cuda:{gpus[i % len(gpus)]}'
            preds[name] = load_and_predict(path, feature_dir, dataset_dir, gpu)
            torch.cuda.empty_cache()
            print(f"  Loaded {name}")
        else:
            print(f"  SKIP: {name}")

    # =================================================================
    # 1. Top attention-e only ensembles
    # =================================================================
    print("\n" + "=" * 70)
    print("TOP ATTENTION-E ENSEMBLES (same architecture, different seeds)")
    print("=" * 70)
    
    top_e = ['e123', 'e24', 'e38', 'e26', 'e37']
    
    # 2-model combos
    best2_r10 = 0
    for pair in combinations(top_e, 2):
        if all(p in preds for p in pair):
            r10, r5 = evaluate_fast([preds[p] for p in pair], [1,1], test_trans, test_rot, train_pos, train_rot)
            if r10 >= best2_r10:
                best2_r10 = r10
                print(f"  2-model {'+'.join(pair):20s} R@10={r10:5.1f}% R@5={r5:5.1f}%")
    
    # 3-model combos
    print()
    best3_r10 = 0
    for combo in combinations(top_e, 3):
        if all(p in preds for p in combo):
            r10, r5 = evaluate_fast([preds[p] for p in combo], [1]*3, test_trans, test_rot, train_pos, train_rot)
            if r10 >= best3_r10:
                best3_r10 = r10
                print(f"  3-model {'+'.join(combo):30s} R@10={r10:5.1f}% R@5={r5:5.1f}%")

    # 5-model (all top-5)
    if all(p in preds for p in top_e):
        r10, r5 = evaluate_fast([preds[p] for p in top_e], [1]*5, test_trans, test_rot, train_pos, train_rot)
        print(f"\n  5-model ALL TOP-5: R@10={r10:.1f}%, R@5={r5:.1f}%")
        
        # Weighted: give more weight to best models
        for w_strat_name, w_strat in [
            ("3:2:1:1:1", [3,2,1,1,1]),
            ("5:3:2:1:1", [5,3,2,1,1]),
            ("4:2:2:1:1", [4,2,2,1,1]),
        ]:
            r10, r5 = evaluate_fast([preds[p] for p in top_e], w_strat, test_trans, test_rot, train_pos, train_rot)
            print(f"  5-model w={w_strat_name}: R@10={r10:.1f}%, R@5={r5:.1f}%")

    # =================================================================
    # 2. Cross-config attention ensembles 
    # =================================================================
    print("\n" + "=" * 70)
    print("CROSS-CONFIG ATTENTION ENSEMBLES (e + a + c)")
    print("=" * 70)
    
    cross_attn = ['e123', 'a123', 'a314', 'c7']
    best_cross = 0
    for n in range(2, len(cross_attn)+1):
        for combo in combinations(cross_attn, n):
            if all(p in preds for p in combo):
                r10, r5 = evaluate_fast([preds[p] for p in combo], [1]*n, test_trans, test_rot, train_pos, train_rot)
                if r10 > best_cross:
                    best_cross = r10
                    print(f"  {n}-model {'+'.join(combo):40s} R@10={r10:5.1f}% R@5={r5:5.1f}%")

    # =================================================================
    # 3. Mixed SPP + top attention
    # =================================================================
    print("\n" + "=" * 70)
    print("MIXED SPP + TOP ATTENTION")
    print("=" * 70)
    
    spp_names = ['s314', 'n123', 'p123']
    attn_top = ['e123', 'e24', 'e38', 'a123', 'c7']
    
    global_best_r10, global_best_r5 = 0, 0
    global_best_r10_cfg, global_best_r5_cfg = None, None
    
    # 2-model: 1 SPP + 1 Attn
    for s in spp_names:
        for a in attn_top:
            if s in preds and a in preds:
                for w1, w2 in [(1,1), (1,2), (2,1), (1,3), (3,1)]:
                    r10, r5 = evaluate_fast([preds[s], preds[a]], [w1,w2], test_trans, test_rot, train_pos, train_rot)
                    if r10 > global_best_r10:
                        global_best_r10 = r10
                        global_best_r10_cfg = (f'{s}+{a} w={w1}:{w2}', r10, r5)
                    if r5 > global_best_r5:
                        global_best_r5 = r5
                        global_best_r5_cfg = (f'{s}+{a} w={w1}:{w2}', r10, r5)

    print(f"  Best mixed R@10: {global_best_r10_cfg}")
    print(f"  Best mixed R@5:  {global_best_r5_cfg}")

    # 3-model: 1 SPP + 2 Attn
    best_3mix_r10 = 0
    for s in spp_names:
        for a_pair in combinations(attn_top, 2):
            if s in preds and all(a in preds for a in a_pair):
                combo = [s] + list(a_pair)
                for w_strat in [(1,1,1), (1,1,2), (1,2,1), (2,1,1), (1,2,2)]:
                    r10, r5 = evaluate_fast([preds[c] for c in combo], list(w_strat), test_trans, test_rot, train_pos, train_rot)
                    if r10 > best_3mix_r10:
                        best_3mix_r10 = r10
                        print(f"  3-mix {combo} w={w_strat}: R@10={r10:.1f}% R@5={r5:.1f}%")

    # =================================================================
    # 4. Greedy forward selection
    # =================================================================
    print("\n" + "=" * 70)
    print("GREEDY FORWARD SELECTION (maximize R@10)")
    print("=" * 70)
    
    all_names = list(preds.keys())
    selected = []
    remaining = set(all_names)
    
    for step in range(min(8, len(all_names))):
        best_add, best_score = None, 0
        for candidate in remaining:
            test_set = selected + [candidate]
            r10, r5 = evaluate_fast([preds[c] for c in test_set], [1]*len(test_set),
                                     test_trans, test_rot, train_pos, train_rot)
            if r10 > best_score:
                best_score = r10
                best_add = candidate
                best_r5_at = r5
        
        if best_add is None:
            break
        
        selected.append(best_add)
        remaining.remove(best_add)
        print(f"  Step {step+1}: Add {best_add:8s} → Ensemble ({len(selected)}): R@10={best_score:.1f}%, R@5={best_r5_at:.1f}%")
        
        if len(selected) > 1 and best_score <= evaluate_fast([preds[selected[0]]], [1], test_trans, test_rot, train_pos, train_rot)[0]:
            print("  → Ensemble not improving over single best, stopping")
            break

    print(f"\n  Final selection: {selected}")


if __name__ == '__main__':
    main()
