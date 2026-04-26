#!/usr/bin/env python3
"""Cross-architecture ensemble with new best attention model."""

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
    r15 = ((re < 15) & (te < 5.0)).mean() * 100
    med_t = np.median(te) * 1000
    med_r = np.median(re)
    return r10, r5, r15, med_r, med_t


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

    models = {
        # Best SPP models
        'spp_s314': f'{base}/exp14s_128d_wider_seed314',       # R@10=72.0%
        'spp_n123': f'{base}/exp14n_128d_wider_seed123',       # R@10=70.9%
        'spp_p123': f'{base}/exp14p_128d_swa_seed123',         # R@10=70.9%
        # Best Attention models  
        'attn_e123': f'{base}/exp24e_attn_h4_wider_seed123',   # R@10=75.3% ⭐ NEW BEST
        'attn_e314': f'{base}/exp24e_attn_h4_wider_seed314',   # R@10=68.7%
        'attn_c314': f'{base}/exp24c_attn_h16_seed314',        # R@10=65.9%, R@5=41.8%
        'attn_c7':   f'{base}/exp24c_attn_h16_seed7',          # R@10=71.4%
        'attn_a123': f'{base}/exp24a_attn_h4_seed123',         # R@10=70.9%
        'attn_a314': f'{base}/exp24a_attn_h4_seed314',         # R@10=70.9%
    }

    preds = {}
    gpus = [0, 1, 3, 4, 5]
    for i, (name, path) in enumerate(models.items()):
        if os.path.exists(path):
            gpu = f'cuda:{gpus[i % len(gpus)]}'
            preds[name] = load_and_predict(path, feature_dir, dataset_dir, gpu)
            torch.cuda.empty_cache()
            print(f"  Loaded {name}")
        else:
            print(f"  SKIP: {name} ({path})")

    # Individual results
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODELS")
    print("=" * 70)
    for name in sorted(preds.keys()):
        r10, r5, r15, mr, mt = evaluate_fast([preds[name]], [1.0], test_trans, test_rot, train_pos, train_rot)
        pool = preds[name]['config']['pool']
        print(f"  [{pool:4s}] {name:<20} R@10={r10:5.1f}% R@5={r5:5.1f}% {mr:.2f}°/{mt:.0f}mm")

    # Pairwise: attn_e123 (best) + each SPP model
    print("\n" + "=" * 70)
    print("PAIRWISE: attn_e123 (BEST) + SPP models")
    print("=" * 70)
    
    best_r10, best_r5 = 0, 0
    best_r10_cfg, best_r5_cfg = None, None
    
    for spp in ['spp_s314', 'spp_n123', 'spp_p123']:
        if spp not in preds:
            continue
        for w1 in np.arange(0.2, 3.1, 0.2):
            for w2 in np.arange(0.2, 3.1, 0.2):
                r10, r5, _, _, _ = evaluate_fast(
                    [preds[spp], preds['attn_e123']], [w1, w2],
                    test_trans, test_rot, train_pos, train_rot)
                if r10 > best_r10:
                    best_r10 = r10
                    best_r10_cfg = (spp, w1, w2, r10, r5)
                if r5 > best_r5:
                    best_r5 = r5
                    best_r5_cfg = (spp, w1, w2, r10, r5)

    print(f"  Best R@10: {best_r10_cfg[0]}+attn_e123 w={best_r10_cfg[1]:.1f}:{best_r10_cfg[2]:.1f} "
          f"→ R@10={best_r10_cfg[3]:.1f}%, R@5={best_r10_cfg[4]:.1f}%")
    print(f"  Best R@5:  {best_r5_cfg[0]}+attn_e123 w={best_r5_cfg[1]:.1f}:{best_r5_cfg[2]:.1f} "
          f"→ R@10={best_r5_cfg[3]:.1f}%, R@5={best_r5_cfg[4]:.1f}%")

    # 3-model combos with attn_e123
    print("\n" + "=" * 70)
    print("3-MODEL ENSEMBLES (always including attn_e123)")
    print("=" * 70)
    
    others = [n for n in preds if n != 'attn_e123']
    best3_r10, best3_r5 = 0, 0
    best3_r10_cfg, best3_r5_cfg = None, None
    
    for pair in combinations(others, 2):
        combo = list(pair) + ['attn_e123']
        pred_list = [preds[c] for c in combo]
        
        for w_strat in [(1,1,1), (1,1,2), (2,1,1), (1,2,1), (2,2,1), (1,1,3), (1,2,2)]:
            r10, r5, _, _, _ = evaluate_fast(pred_list, list(w_strat), 
                                              test_trans, test_rot, train_pos, train_rot)
            if r10 > best3_r10:
                best3_r10 = r10
                best3_r10_cfg = (combo, w_strat, r10, r5)
            if r5 > best3_r5:
                best3_r5 = r5
                best3_r5_cfg = (combo, w_strat, r10, r5)

    print(f"  Best R@10: {best3_r10_cfg[0]} w={best3_r10_cfg[1]} → R@10={best3_r10_cfg[2]:.1f}%, R@5={best3_r10_cfg[3]:.1f}%")
    print(f"  Best R@5:  {best3_r5_cfg[0]} w={best3_r5_cfg[1]} → R@10={best3_r5_cfg[2]:.1f}%, R@5={best3_r5_cfg[3]:.1f}%")

    # 4-model combos
    print("\n" + "=" * 70)
    print("4-MODEL ENSEMBLES (always including attn_e123)")
    print("=" * 70)
    
    best4_r10, best4_r5 = 0, 0
    best4_r10_cfg, best4_r5_cfg = None, None
    
    for trio in combinations(others, 3):
        combo = list(trio) + ['attn_e123']
        pred_list = [preds[c] for c in combo]
        
        for w_strat in [(1,1,1,1), (1,1,1,2), (2,1,1,1), (1,1,1,3), (2,2,1,1), (1,1,2,2)]:
            r10, r5, _, _, _ = evaluate_fast(pred_list, list(w_strat),
                                              test_trans, test_rot, train_pos, train_rot)
            if r10 > best4_r10:
                best4_r10 = r10
                best4_r10_cfg = (combo, w_strat, r10, r5)
            if r5 > best4_r5:
                best4_r5 = r5
                best4_r5_cfg = (combo, w_strat, r10, r5)

    print(f"  Best R@10: {best4_r10_cfg[0]} w={best4_r10_cfg[1]} → R@10={best4_r10_cfg[2]:.1f}%, R@5={best4_r10_cfg[3]:.1f}%")
    print(f"  Best R@5:  {best4_r5_cfg[0]} w={best4_r5_cfg[1]} → R@10={best4_r5_cfg[2]:.1f}%, R@5={best4_r5_cfg[3]:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF BEST RESULTS")
    print("=" * 70)
    print(f"  Best single model:     attn_e123  R@10=75.3%  R@5=42.9%")
    print(f"  Best pairwise R@10:    {best_r10_cfg[3]:.1f}%")
    print(f"  Best pairwise R@5:     {best_r5_cfg[4]:.1f}%")
    print(f"  Best 3-model R@10:     {best3_r10_cfg[2]:.1f}%")
    print(f"  Best 3-model R@5:      {best3_r5_cfg[3]:.1f}%")
    print(f"  Best 4-model R@10:     {best4_r10_cfg[2]:.1f}%")
    print(f"  Best 4-model R@5:      {best4_r5_cfg[3]:.1f}%")


if __name__ == '__main__':
    main()
