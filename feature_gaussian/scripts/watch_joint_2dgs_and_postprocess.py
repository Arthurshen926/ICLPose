#!/usr/bin/env python3
"""Wait for a joint 2DGS run to finish, then trigger postprocess with visualizations."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--output_root', default='feature_gaussian/output')
    parser.add_argument('--poll_seconds', type=int, default=60)
    parser.add_argument('--num_visual_cameras', type=int, default=6)
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root) / args.exp_name
    train_log = run_dir / 'train.log'

    print(f'[watch] Waiting for {train_log}')
    while not train_log.exists():
        time.sleep(args.poll_seconds)

    print(f'[watch] Monitoring {train_log}')
    while True:
        try:
            text = train_log.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            time.sleep(args.poll_seconds)
            continue

        if 'Training complete.' in text:
            break
        time.sleep(args.poll_seconds)

    print(f'[watch] Training finished for {args.exp_name}, starting postprocess')
    subprocess.run(
        [
            args.python,
            '-m',
            'feature_gaussian.scripts.run_joint_2dgs_postprocess',
            '--config', args.config,
            '--exp_name', args.exp_name,
            '--output_root', args.output_root,
            '--num_visual_cameras', str(args.num_visual_cameras),
        ],
        check=True,
    )
    print(f'[watch] Postprocess finished for {args.exp_name}')


if __name__ == '__main__':
    main()
