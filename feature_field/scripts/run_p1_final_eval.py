#!/usr/bin/env python3
"""Run the final P1 DCFF evaluations in parallel across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


JOBS = [
    ('v14b_best_train', 'feature_field/output/dcff_oldhospital_v14b_spatial_only_frozen/checkpoints/best.pth', 'train'),
    ('v14b_best_test', 'feature_field/output/dcff_oldhospital_v14b_spatial_only_frozen/checkpoints/best.pth', 'test'),
    ('v14b_latest_train', 'feature_field/output/dcff_oldhospital_v14b_spatial_only_frozen/checkpoints/latest.pth', 'train'),
    ('v14b_latest_test', 'feature_field/output/dcff_oldhospital_v14b_spatial_only_frozen/checkpoints/latest.pth', 'test'),
    ('v14c_best_train', 'feature_field/output/dcff_oldhospital_v14c_no_cross_attn_frozen/checkpoints/best.pth', 'train'),
    ('v14c_best_test', 'feature_field/output/dcff_oldhospital_v14c_no_cross_attn_frozen/checkpoints/best.pth', 'test'),
    ('v14c_latest_train', 'feature_field/output/dcff_oldhospital_v14c_no_cross_attn_frozen/checkpoints/latest.pth', 'train'),
    ('v14c_latest_test', 'feature_field/output/dcff_oldhospital_v14c_no_cross_attn_frozen/checkpoints/latest.pth', 'test'),
    ('v14d_best_train', 'feature_field/output/dcff_oldhospital_v14d_binary_no_cross_attn_frozen/checkpoints/best.pth', 'train'),
    ('v14d_best_test', 'feature_field/output/dcff_oldhospital_v14d_binary_no_cross_attn_frozen/checkpoints/best.pth', 'test'),
    ('v14d_latest_train', 'feature_field/output/dcff_oldhospital_v14d_binary_no_cross_attn_frozen/checkpoints/latest.pth', 'train'),
    ('v14d_latest_test', 'feature_field/output/dcff_oldhospital_v14d_binary_no_cross_attn_frozen/checkpoints/latest.pth', 'test'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='feature_field/output/eval_p1_final')
    parser.add_argument('--gpus', default='3,4')
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def run_job(gpu: str, job: tuple[str, str, str], output_dir: Path, batch_size: int) -> dict:
    name, ckpt, split = job
    out_json = output_dir / f'{name}.json'
    cmd = [
        sys.executable,
        '-m', 'feature_field.eval_dcff_metrics',
        '--checkpoint', ckpt,
        '--camera_split', split,
        '--batch_size', str(batch_size),
        '--output_json', str(out_json),
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    print(f"[{gpu}] RUN {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    with open(out_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if not gpus:
        raise ValueError('At least one GPU must be provided')

    buckets = [[] for _ in gpus]
    for idx, job in enumerate(JOBS):
        buckets[idx % len(gpus)].append(job)

    children = []
    for gpu, bucket in zip(gpus, buckets):
        child_code = (
            'import json\n'
            'from feature_field.scripts.run_p1_final_eval import run_job\n'
            f'gpu = {gpu!r}\n'
            f'bucket = {bucket!r}\n'
            f'output_dir = {str(output_dir)!r}\n'
            f'batch_size = {args.batch_size!r}\n'
            'results = {}\n'
            'from pathlib import Path\n'
            'for job in bucket:\n'
            '    results[job[0]] = run_job(gpu, job, Path(output_dir), batch_size)\n'
            'with open(Path(output_dir) / f"partial_gpu{gpu}.json", "w", encoding="utf-8") as f:\n'
            '    json.dump(results, f, indent=2)\n'
        )
        log_path = output_dir / f'gpu{gpu}.log'
        with open(log_path, 'w', encoding='utf-8') as log_file:
            proc = subprocess.Popen([sys.executable, '-c', child_code], stdout=log_file, stderr=subprocess.STDOUT)
        children.append(proc)

    for proc in children:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'Child evaluator failed with code {proc.returncode}')

    merged = {}
    for gpu in gpus:
        partial_path = output_dir / f'partial_gpu{gpu}.json'
        with open(partial_path, 'r', encoding='utf-8') as f:
            merged.update(json.load(f))
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
    print(f'Wrote {output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
