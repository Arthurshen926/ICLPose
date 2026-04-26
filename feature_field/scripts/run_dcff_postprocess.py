#!/usr/bin/env python3
"""Run both quantitative metrics and saved visualizations for a DCFF checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_field.eval_dcff_metrics import main as eval_main
from feature_field.visualize_feature_comparison import visualize_dcff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--output_root', default='feature_field/output')
    parser.add_argument('--source_dir', default='dataset/OldHospital')
    parser.add_argument('--feature_dir', default='feature_extract/output/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_visual_cameras', type=int, default=6)
    return parser.parse_args()


def run_eval(
    checkpoint: str,
    split: str,
    output_json: Path,
    batch_size: int,
    source_dir: str,
    feature_dir: str,
    device: str,
) -> None:
    import sys
    argv_backup = sys.argv
    try:
        sys.argv = [
            'eval_dcff_metrics',
            '--checkpoint', checkpoint,
            '--source_dir', source_dir,
            '--feature_dir', feature_dir,
            '--device', device,
            '--camera_split', split,
            '--batch_size', str(batch_size),
            '--output_json', str(output_json),
        ]
        eval_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    metrics_dir = output_root / 'postprocess_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_json = metrics_dir / f'{args.exp_name}_train.json'
    test_json = metrics_dir / f'{args.exp_name}_test.json'
    run_eval(args.checkpoint, 'train', train_json, args.batch_size, args.source_dir, args.feature_dir, args.device)
    run_eval(args.checkpoint, 'test', test_json, args.batch_size, args.source_dir, args.feature_dir, args.device)

    vis_train_dir = output_root / f'vis_{args.exp_name}_train'
    vis_test_dir = output_root / f'vis_{args.exp_name}_test'
    camera_indices = list(range(args.num_visual_cameras))

    train_res = visualize_dcff(
        ckpt_path=args.checkpoint,
        source_dir=args.source_dir,
        feature_dir=args.feature_dir,
        output_dir=str(vis_train_dir),
        camera_indices=camera_indices,
        camera_split='train',
        device=args.device,
        exp_name=args.exp_name,
    )
    test_res = visualize_dcff(
        ckpt_path=args.checkpoint,
        source_dir=args.source_dir,
        feature_dir=args.feature_dir,
        output_dir=str(vis_test_dir),
        camera_indices=camera_indices,
        camera_split='test',
        device=args.device,
        exp_name=args.exp_name,
    )

    summary = {
        'exp_name': args.exp_name,
        'checkpoint': args.checkpoint,
        'train_metrics_json': str(train_json),
        'test_metrics_json': str(test_json),
        'train_visual_dir': str(vis_train_dir),
        'test_visual_dir': str(vis_test_dir),
        'train_visual_summary': train_res,
        'test_visual_summary': test_res,
    }
    summary_path = metrics_dir / f'{args.exp_name}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
