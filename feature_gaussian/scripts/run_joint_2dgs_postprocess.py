#!/usr/bin/env python3
"""Run quantitative evaluation and saved visualizations for a joint 2DGS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_gaussian.evaluation.eval_joint_reconstruction import main as eval_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--output_root', default='feature_gaussian/output')
    parser.add_argument('--num_visual_cameras', type=int, default=6)
    return parser.parse_args()


def run_eval(config: str, split: str, output_dir: Path, num_samples: int) -> None:
    import sys

    argv_backup = sys.argv
    try:
        sys.argv = [
            'eval_joint_reconstruction',
            '--config', config,
            '--camera_split', split,
            '--num_samples', str(num_samples),
            '--max_metrics_cameras', '999999',
            '--output_dir', str(output_dir),
        ]
        eval_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    metrics_dir = output_root / 'postprocess_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dir = metrics_dir / f'{args.exp_name}_train_eval'
    test_dir = metrics_dir / f'{args.exp_name}_test_eval'
    run_eval(args.config, 'train', train_dir, args.num_visual_cameras)
    run_eval(args.config, 'test', test_dir, args.num_visual_cameras)

    summary = {
        'exp_name': args.exp_name,
        'config': args.config,
        'train_eval_dir': str(train_dir),
        'test_eval_dir': str(test_dir),
        'train_metrics_json': str(train_dir / 'metrics.json'),
        'test_metrics_json': str(test_dir / 'metrics.json'),
        'train_summary_json': str(train_dir / 'summary.json'),
        'test_summary_json': str(test_dir / 'summary.json'),
        'train_visual_dir': str(train_dir),
        'test_visual_dir': str(test_dir),
    }
    summary_path = metrics_dir / f'{args.exp_name}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
