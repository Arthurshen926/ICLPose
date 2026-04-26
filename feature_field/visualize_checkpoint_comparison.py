#!/usr/bin/env python3
"""
Feature Visualization with Multiple Checkpoint Sources
==================================================
Compare features from different checkpoint iterations to find the best one.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_field.visualize_feature_comparison import visualize_dcff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_base', type=str, default='feature_field/output/vis_comparison_latest')
    parser.add_argument('--source_dir', type=str, default='/root/ICLPose-loc/dataset/OldHospital')
    parser.add_argument('--feature_dir', type=str, 
                       default='/root/ICLPose-loc/feature_extract/output/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--camera_indices', type=str, default=None)
    parser.add_argument('--camera_split', type=str, default='auto', choices=['auto', 'train', 'test', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)
    
    camera_indices = None
    if args.camera_indices:
        camera_indices = [int(x) for x in args.camera_indices.split(',')]

    # Define experiments with multiple checkpoint sources
    experiments = {
        # v12_fsm: best vs latest
        'v12_fsm_best': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v12_fsm/checkpoints/best.pth',
            'note': 'iter 500 (Phase 3 start)',
        },
        'v12_fsm_latest': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v12_fsm/checkpoints/latest.pth',
            'note': 'iter 26000+ (Phase 3 stable)',
        },
        # v13a: best vs latest
        'v13a_best': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v13a_no_fsm/checkpoints/best.pth',
            'note': 'iter 500 (Phase 3 start)',
        },
        'v13a_latest': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v13a_no_fsm/checkpoints/latest.pth',
            'note': 'iter 9000+ (Phase 3 stable)',
        },
        # v10c: best vs latest (for comparison - geometry frozen)
        'v10c_best': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v10c_carrier_residual/checkpoints/best.pth',
            'note': 'iter 22550 (frozen geometry)',
        },
        'v10c_latest': {
            'ckpt': 'feature_field/output/dcff_oldhospital_v10c_carrier_residual/checkpoints/latest.pth',
            'note': 'iter 40000 (frozen geometry)',
        },
    }
    
    results = {}
    
    for exp_name, exp_info in experiments.items():
        ckpt_path = exp_info['ckpt']
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {exp_name}: checkpoint not found at {ckpt_path}")
            continue
        
        output_dir = os.path.join(args.output_base, f'dcff_{exp_name}')
        try:
            result = visualize_dcff(
                ckpt_path=ckpt_path,
                source_dir=args.source_dir,
                feature_dir=args.feature_dir,
                output_dir=output_dir,
                camera_indices=camera_indices,
                camera_split=args.camera_split,
                device=args.device,
                exp_name=exp_name,
            )
            results[exp_name] = result
            print(f"  [{exp_name}] {exp_info['note']}")
        except Exception as e:
            print(f"  [ERROR] {exp_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import torch
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("  FEATURE VISUALIZATION COMPARISON")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Checkpoint':<20} {'Fine Cos':<12} {'Coarse Cos':<12}")
    print("-" * 80)
    
    for exp_name in ['v10c_best', 'v10c_latest', 'v12_fsm_best', 'v12_fsm_latest', 'v13a_best', 'v13a_latest']:
        if exp_name in results:
            r = results[exp_name]
            note = experiments[exp_name]['note'][:18]
            print(f"{exp_name:<25} {note:<20} {r['fine_cos']:.4f} ± {r.get('fine_cos_std', 0):.3f}   "
                  f"{r['coarse_cos']:.4f} ± {r.get('coarse_cos_std', 0):.3f}")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {args.output_base}/")


if __name__ == '__main__':
    main()
