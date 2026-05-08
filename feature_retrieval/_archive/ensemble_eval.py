#!/usr/bin/env python3
"""
Ensemble evaluation: average predictions from multiple trained models.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from patch_regressor_v7 import (
    PatchPoseRegressor, PatchPoseDataset, 
    geodesic_distance, gram_schmidt_6d_to_matrix, quaternion_to_matrix
)

def load_model_and_predict(ckpt_path, test_data, device):
    """Load a model checkpoint and generate predictions."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    
    # Reconstruct model
    model = PatchPoseRegressor(
        pool_type=config['pool'],
        feat_mode=config['feat'],
        patch_dim=config.get('patch_dim', 64),
        hidden_dims=tuple(config['hidden_dims']),
        dropout=config.get('dropout', 0.1),
        attn_heads=config.get('attn_heads', 4),
        conv_out_dim=config.get('conv_out_dim', 512),
        spp_levels=config.get('spp_levels', None),
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Get normalization params
    norm = ckpt['norm_params']
    trans_mean = norm['mean'].to(device)
    trans_std = norm['std'].to(device)
    
    with torch.no_grad():
        trans_pred, rot_pred = model(
            test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        trans_pred_real = trans_pred * trans_std + trans_mean
    
    return trans_pred_real, rot_pred


def evaluate_predictions(trans_pred, rot_pred, test_data, train_data):
    """Evaluate predictions against ground truth."""
    trans_errors = (trans_pred - test_data.translations).norm(dim=-1).cpu().numpy()
    rot_errors = (geodesic_distance(rot_pred, test_data.rotations) * 180 / math.pi).cpu().numpy()
    
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
    ]
    
    results = {
        'rot_median': float(np.median(rot_errors)),
        'trans_median_mm': float(np.median(trans_errors) * 1000),
        'recall': {},
        'pass': {},
    }
    
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th).mean() * 100
        trans_pass = (trans_errors < trans_th).mean() * 100
        combined = ((rot_errors < rot_th) & (trans_errors < trans_th)).mean() * 100
        results['recall'][name] = float(combined)
        results['pass'][name] = {'rot': float(rot_pass), 'trans': float(trans_pass)}
    
    # Retrieval mode
    if train_data is not None:
        from scipy.spatial import cKDTree
        train_positions = train_data.translations.cpu().numpy()
        pred_positions = trans_pred.cpu().numpy()
        
        tree = cKDTree(train_positions)
        dists, nn_indices = tree.query(pred_positions, k=1)
        
        train_rot = train_data.rotations.cpu()
        test_rot = test_data.rotations.cpu()
        test_trans_np = test_data.translations.cpu().numpy()
        
        retr_trans_errors = np.linalg.norm(train_positions[nn_indices] - test_trans_np, axis=-1)
        retr_rot_pred = train_rot[nn_indices]
        retr_rot_errors = (geodesic_distance(retr_rot_pred, test_rot) * 180 / math.pi).numpy()
        
        results['retrieval'] = {}
        for name, rot_th, trans_th in thresholds:
            combined = ((retr_rot_errors < rot_th) & (retr_trans_errors < trans_th)).mean() * 100
            results['retrieval'][name] = float(combined)
    
    return results


def average_rotations(rot_list):
    """Average rotation matrices via SVD projection to SO(3)."""
    # Simple average then project to nearest rotation via SVD
    avg = torch.stack(rot_list, dim=0).mean(dim=0)  # (B, 3, 3)
    U, S, Vt = torch.linalg.svd(avg)
    # Ensure proper rotation (det = +1)
    det = torch.det(U @ Vt)
    sign = torch.ones_like(det)
    sign[det < 0] = -1
    # Correct the last column of U
    U_corrected = U.clone()
    U_corrected[:, :, -1] *= sign.unsqueeze(-1)
    return U_corrected @ Vt


def main():
    parser = argparse.ArgumentParser(description='Ensemble Evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Paths to model checkpoint directories')
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--feature_dir', type=str,
                        default='output/feature_extract/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load data (load all features to be safe)
    print("Loading data...")
    train_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'train', 'cpu',
        use_fine=True, use_coarse=True, use_summary=True)
    test_data = PatchPoseDataset(
        args.feature_dir, args.dataset_dir, 'test', 'cpu',
        use_fine=True, use_coarse=True, use_summary=True)
    train_data.to(device)
    test_data.to(device)
    
    # Collect predictions from each model
    all_trans = []
    all_rot = []
    
    for model_dir in args.models:
        ckpt_path = Path(model_dir) / 'model_best.pt'
        if not ckpt_path.exists():
            print(f"  SKIP (not found): {ckpt_path}")
            continue
        print(f"  Loading: {model_dir}")
        trans_pred, rot_pred = load_model_and_predict(ckpt_path, test_data, device)
        all_trans.append(trans_pred)
        all_rot.append(rot_pred)
        
        # Also evaluate individual model
        res = evaluate_predictions(trans_pred, rot_pred, test_data, train_data)
        print(f"    Individual: {res['rot_median']:.2f}deg / {res['trans_median_mm']:.0f}mm | "
              f"R@10/2={res['recall']['10deg_2m']:.1f}% | R@5/1={res['recall']['5deg_1m']:.1f}%")
    
    if len(all_trans) < 2:
        print("Need at least 2 models for ensemble")
        return
    
    # Ensemble: average translations, SVD-project average rotations
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE ({len(all_trans)} models)")
    print(f"{'=' * 70}")
    
    # Average translations
    ens_trans = torch.stack(all_trans, dim=0).mean(dim=0)
    
    # Average rotations
    ens_rot = average_rotations(all_rot)
    
    # Evaluate ensemble
    results = evaluate_predictions(ens_trans, ens_rot, test_data, train_data)
    
    print(f"Rotation  (median): {results['rot_median']:.2f}deg")
    print(f"Translation (median): {results['trans_median_mm']:.0f}mm")
    for name in ['5deg_1m', '10deg_2m', '15deg_5m']:
        p = results['pass'][name]
        print(f"  R@{name}: {results['recall'][name]:.1f}%  "
              f"(rot_pass={p['rot']:.1f}%, trans_pass={p['trans']:.1f}%)")
    
    if 'retrieval' in results:
        print(f"\nRetrieval mode:")
        for name in ['5deg_1m', '10deg_2m', '15deg_5m']:
            print(f"  R@{name}: {results['retrieval'][name]:.1f}%")
    
    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_dir}/ensemble_results.json")


if __name__ == '__main__':
    main()
