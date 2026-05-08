#!/usr/bin/env python3
"""Comprehensive evaluation of ALL models + expanded ensemble search.

Evaluates all new config f, g, and config e new seeds,
then does exhaustive ensemble search including fine weight tuning.
"""

import sys, os, json, torch, numpy as np, math
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_retrieval.patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, precompute_pooled_features, geodesic_distance
)
from feature_retrieval.cross_arch_ensemble import load_and_predict


def evaluate_single(preds, test_trans, test_rot, train_pos, train_rot):
    """Evaluate single model predictions."""
    te = np.linalg.norm(preds['trans_pred'] - test_trans, axis=-1)
    re = (geodesic_distance(preds['rot_pred'], test_rot) * 180 / math.pi).numpy()
    r10 = ((re < 10) & (te < 2.0)).mean() * 100
    r5 = ((re < 5) & (te < 1.0)).mean() * 100
    return r10, r5, te, re


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


def fine_weight_search(preds_list, names, test_trans, test_rot, n_models=2, n_steps=21):
    """Fine weight search for n_models ensemble."""
    best_r10, best_r5 = 0, 0
    best_r10_cfg, best_r5_cfg = None, None
    
    if n_models == 2:
        for w1 in np.linspace(0.1, 3.0, n_steps):
            for w2 in np.linspace(0.1, 3.0, n_steps):
                r10, r5 = evaluate_ensemble(preds_list, [w1, w2], test_trans, test_rot)
                if r10 > best_r10:
                    best_r10 = r10
                    best_r10_cfg = (f'{names[0]}+{names[1]} w={w1:.2f}:{w2:.2f}', r10, r5)
                if r5 > best_r5:
                    best_r5 = r5
                    best_r5_cfg = (f'{names[0]}+{names[1]} w={w1:.2f}:{w2:.2f}', r10, r5)
    elif n_models == 3:
        steps = np.linspace(0.1, 3.0, 11)
        for w1 in steps:
            for w2 in steps:
                for w3 in steps:
                    r10, r5 = evaluate_ensemble(preds_list, [w1, w2, w3], test_trans, test_rot)
                    if r10 > best_r10:
                        best_r10 = r10
                        best_r10_cfg = (f'{"+".join(names)} w={w1:.1f}:{w2:.1f}:{w3:.1f}', r10, r5)
                    if r5 > best_r5:
                        best_r5 = r5
                        best_r5_cfg = (f'{"+".join(names)} w={w1:.1f}:{w2:.1f}:{w3:.1f}', r10, r5)
    
    return best_r10, best_r10_cfg, best_r5, best_r5_cfg


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
    train_pos = train_data.translations.numpy()
    train_rot = train_data.rotations
    del train_data, test_data

    # Discover all models
    all_models = {}
    
    # Config e models (original 36 seeds + 24 new)
    for d in sorted(Path(base).glob('exp24e_attn_h4_wider_seed*')):
        if (d / 'model_best.pt').exists() and '_full' not in d.name:
            seed = d.name.split('seed')[-1]
            all_models[f'e{seed}'] = str(d)
    
    # Config a models
    for d in sorted(Path(base).glob('exp24a_attn_h4_seed*')):
        if (d / 'model_best.pt').exists():
            seed = d.name.split('seed')[-1]
            all_models[f'a{seed}'] = str(d)
    
    # Config c models
    for d in sorted(Path(base).glob('exp24c_attn_h16_seed*')):
        if (d / 'model_best.pt').exists():
            seed = d.name.split('seed')[-1]
            all_models[f'c{seed}'] = str(d)
    
    # Config f models (NEW)
    for d in sorted(Path(base).glob('exp24f_attn_h8_wider_seed*')):
        if (d / 'model_best.pt').exists():
            seed = d.name.split('seed')[-1]
            all_models[f'f{seed}'] = str(d)
    
    # Config g models (NEW)
    for d in sorted(Path(base).glob('exp24g_attn_h4_xwide_seed*')):
        if (d / 'model_best.pt').exists():
            seed = d.name.split('seed')[-1]
            all_models[f'g{seed}'] = str(d)
    
    # Top SPP models
    spp_models = {
        's314': f'{base}/exp14s_128d_wider_seed314',
        'n123': f'{base}/exp14n_128d_wider_seed123',
        'p123': f'{base}/exp14p_128d_swa_seed123',
    }
    all_models.update({k: v for k, v in spp_models.items() if Path(v, 'model_best.pt').exists()})

    print(f"Found {len(all_models)} models total")
    print(f"  Config e: {sum(1 for k in all_models if k.startswith('e'))}")
    print(f"  Config a: {sum(1 for k in all_models if k.startswith('a'))}")
    print(f"  Config c: {sum(1 for k in all_models if k.startswith('c'))}")
    print(f"  Config f: {sum(1 for k in all_models if k.startswith('f'))}")
    print(f"  Config g: {sum(1 for k in all_models if k.startswith('g'))}")
    print(f"  SPP:      {sum(1 for k in all_models if k.startswith(('s','n','p')))}")

    # =================================================================
    # 1. Evaluate all models individually
    # =================================================================
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL EVALUATION")
    print("=" * 70)
    
    gpus = [0, 1, 2, 3, 4, 5]
    preds = {}
    results = []
    
    for i, (name, path) in enumerate(sorted(all_models.items())):
        gpu = f'cuda:{gpus[i % len(gpus)]}'
        try:
            preds[name] = load_and_predict(path, feature_dir, dataset_dir, gpu)
            r10, r5, te, re = evaluate_single(preds[name], test_trans, test_rot, train_pos, train_rot)
            results.append((name, r10, r5, np.median(te), np.median(re)))
            torch.cuda.empty_cache()
        except Exception as ex:
            print(f"  SKIP {name}: {ex}")
    
    # Sort by R@10
    results.sort(key=lambda x: -x[1])
    
    print(f"\n{'Model':>12s}  R@10°/2m  R@5°/1m  MedTE(m)  MedRE(°)")
    print("-" * 60)
    for name, r10, r5, mte, mre in results[:30]:
        marker = " ⭐" if r10 >= 70 else ""
        print(f"  {name:>10s}  {r10:6.1f}%  {r5:6.1f}%  {mte:7.3f}   {mre:6.2f}{marker}")
    
    # Per-config statistics
    for prefix, label in [('e', 'Config e'), ('f', 'Config f'), ('g', 'Config g'), ('a', 'Config a'), ('c', 'Config c')]:
        cfg_results = [r for r in results if r[0].startswith(prefix)]
        if cfg_results:
            r10s = [r[1] for r in cfg_results]
            print(f"\n  {label} ({len(cfg_results)} models): mean={np.mean(r10s):.1f}%, std={np.std(r10s):.1f}%, "
                  f"min={np.min(r10s):.1f}%, max={np.max(r10s):.1f}%")

    # =================================================================
    # 2. Best 2-model ensembles (focus on top models)
    # =================================================================
    print("\n" + "=" * 70)
    print("TOP 2-MODEL ENSEMBLES")
    print("=" * 70)
    
    # Only use top-15 models for 2-model search
    top_names = [r[0] for r in results[:15]]
    
    best2 = []
    for a, b in combinations(top_names, 2):
        if a in preds and b in preds:
            r10, r5 = evaluate_ensemble([preds[a], preds[b]], [1, 1], test_trans, test_rot)
            best2.append((f'{a}+{b}', r10, r5))
    
    best2.sort(key=lambda x: -x[1])
    print("\nTop 2-model (equal weight):")
    for cfg, r10, r5 in best2[:15]:
        print(f"  {cfg:25s}  R@10={r10:5.1f}%  R@5={r5:5.1f}%")
    
    # =================================================================
    # 3. Fine weight search for top 2-model ensembles  
    # =================================================================
    print("\n" + "=" * 70)
    print("FINE WEIGHT SEARCH (top 2-model)")
    print("=" * 70)
    
    global_best_r10 = 0
    global_best_cfg = None
    
    for cfg, r10, r5 in best2[:8]:  # Top 8 pairs
        names_pair = cfg.split('+')
        a, b = names_pair
        br10, br10_cfg, br5, br5_cfg = fine_weight_search(
            [preds[a], preds[b]], [a, b], test_trans, test_rot, n_models=2, n_steps=31)
        if br10 > global_best_r10:
            global_best_r10 = br10
            global_best_cfg = br10_cfg
        print(f"  {cfg:25s}  Best R@10: {br10_cfg}")
        if br5_cfg != br10_cfg:
            print(f"  {'':25s}  Best R@5:  {br5_cfg}")
    
    print(f"\n  ⭐ GLOBAL BEST 2-MODEL: {global_best_cfg}")

    # =================================================================
    # 4. Greedy forward selection (from ALL models)
    # =================================================================
    print("\n" + "=" * 70)
    print("GREEDY FORWARD SELECTION (all models)")
    print("=" * 70)
    
    available = list(preds.keys())
    selected = []
    remaining = set(available)
    
    for step in range(min(10, len(available))):
        best_add, best_score, best_r5 = None, 0, 0
        for candidate in remaining:
            test_set = selected + [candidate]
            r10, r5 = evaluate_ensemble([preds[c] for c in test_set], [1]*len(test_set),
                                         test_trans, test_rot)
            if r10 > best_score:
                best_score = r10
                best_add = candidate
                best_r5 = r5
        
        if best_add is None:
            break
        
        selected.append(best_add)
        remaining.remove(best_add)
        print(f"  Step {step+1}: Add {best_add:10s} → Ensemble ({len(selected)}): R@10={best_score:.1f}%, R@5={best_r5:.1f}%")
        
        # Stop if no improvement for 3 steps
        if len(selected) >= 4:
            recent = [best_score]
            prev_best = max(
                evaluate_ensemble([preds[c] for c in selected[:-1]], [1]*len(selected[:-1]), test_trans, test_rot)[0],
                evaluate_ensemble([preds[c] for c in selected[:-2]], [1]*len(selected[:-2]), test_trans, test_rot)[0] if len(selected) >= 3 else 0,
            )
            if best_score <= prev_best:
                print(f"  → No improvement. Best was with {len(selected)-1} models.")
                break
    
    print(f"\n  Final greedy selection: {selected}")
    
    # Fine-weight the greedy top-3
    if len(selected) >= 3:
        top3 = selected[:3]
        print(f"\n  Fine-weight search for greedy top-3: {top3}")
        steps = np.linspace(0.1, 4.0, 16)
        best_g3_r10 = 0
        for w1 in steps:
            for w2 in steps:
                for w3 in steps:
                    r10, r5 = evaluate_ensemble([preds[c] for c in top3], [w1, w2, w3], test_trans, test_rot)
                    if r10 > best_g3_r10:
                        best_g3_r10 = r10
                        print(f"    w={w1:.1f}:{w2:.1f}:{w3:.1f} → R@10={r10:.1f}%, R@5={r5:.1f}%")
    
    # =================================================================
    # 5. Cross-architecture ensembles (best from each config)
    # =================================================================
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE ENSEMBLES")
    print("=" * 70)
    
    # Pick best from each config
    best_per_config = {}
    for prefix in ['e', 'f', 'g', 'a', 'c', 's', 'n', 'p']:
        cfg_results = [(r[0], r[1]) for r in results if r[0].startswith(prefix)]
        if cfg_results:
            cfg_results.sort(key=lambda x: -x[1])
            best_per_config[prefix] = cfg_results[0][0]  # name of best model
            print(f"  Best {prefix}: {cfg_results[0][0]} ({cfg_results[0][1]:.1f}%)")
    
    # Try all combos of best-per-config
    config_keys = list(best_per_config.keys())
    for n in range(2, min(len(config_keys)+1, 6)):
        best_combo_r10 = 0
        for combo in combinations(config_keys, n):
            names_combo = [best_per_config[c] for c in combo]
            if all(n in preds for n in names_combo):
                r10, r5 = evaluate_ensemble([preds[n] for n in names_combo], [1]*n, test_trans, test_rot)
                if r10 > best_combo_r10:
                    best_combo_r10 = r10
                    print(f"  {n}-config {'+'.join(combo):15s} ({'+'.join(names_combo):30s}): R@10={r10:.1f}%, R@5={r5:.1f}%")


    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best individual:  {results[0][0]:>10s} = {results[0][1]:.1f}% R@10, {results[0][2]:.1f}% R@5")
    if best2:
        print(f"  Best 2-model eq:  {best2[0][0]:>25s} = {best2[0][1]:.1f}% R@10")
    if global_best_cfg:
        print(f"  Best 2-model opt: {global_best_cfg}")
    print(f"  Greedy selection: {selected}")


if __name__ == '__main__':
    main()
