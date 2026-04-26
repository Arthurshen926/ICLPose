#!/usr/bin/env python3
"""Wait for a DCFF run to finish, then trigger postprocess with visualizations."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--output_root', default='feature_field/output')
    parser.add_argument('--checkpoint_name', default='best.pth')
    parser.add_argument('--poll_seconds', type=int, default=60)
    parser.add_argument('--source_dir', default='dataset/OldHospital')
    parser.add_argument('--feature_dir', default='feature_extract/output/features_radio_dual/OldHospital_pilot')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_visual_cameras', type=int, default=6)
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    run_dir = output_root / args.exp_name
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
        if 'Training complete in' in text:
            break
        time.sleep(args.poll_seconds)

    checkpoint = run_dir / 'checkpoints' / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing checkpoint for postprocess: {checkpoint}')

    print(f'[watch] Training finished for {args.exp_name}, starting postprocess')
    subprocess.run(
        [
            args.python,
            '-m',
            'feature_field.scripts.run_dcff_postprocess',
            '--checkpoint', str(checkpoint),
            '--exp_name', args.exp_name,
            '--output_root', str(output_root),
            '--source_dir', args.source_dir,
            '--feature_dir', args.feature_dir,
            '--device', args.device,
            '--batch_size', str(args.batch_size),
            '--num_visual_cameras', str(args.num_visual_cameras),
        ],
        check=True,
    )
    print(f'[watch] Postprocess finished for {args.exp_name}')


if __name__ == '__main__':
    main()
