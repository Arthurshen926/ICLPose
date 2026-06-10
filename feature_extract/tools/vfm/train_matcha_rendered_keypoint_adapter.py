"""Train a MATCHA-style dual-softmax RADIO selector adapter."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.matcha_adapter_training import MatchaAdapterTrainingConfig, train_matcha_dual_softmax_selector
from feature_extract.vfm.patch_selector_training import (
    load_patch_selector_training_set_npz,
    load_safe_patch_selector_checkpoint,
    save_safe_patch_selector_checkpoint,
)


def _sha(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_cache", required=True)
    parser.add_argument("--init_safe_checkpoint", default="")
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--projection_lr", type=float, default=None)
    parser.add_argument("--residual_lr", type=float, default=None)
    parser.add_argument("--pairwise_lr", type=float, default=None)
    parser.add_argument("--gate_lr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--hard_negative_weight", type=float, default=0.2)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--inlier_loss_weight", type=float, default=0.05)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.2)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples, metadata = load_patch_selector_training_set_npz(Path(args.sample_cache))
    init_run = load_safe_patch_selector_checkpoint(Path(args.init_safe_checkpoint), device=args.device) if args.init_safe_checkpoint else None
    config = MatchaAdapterTrainingConfig(
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        projection_lr=None if args.projection_lr is None else float(args.projection_lr),
        residual_lr=None if args.residual_lr is None else float(args.residual_lr),
        pairwise_lr=None if args.pairwise_lr is None else float(args.pairwise_lr),
        gate_lr=None if args.gate_lr is None else float(args.gate_lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        inlier_loss_weight=float(args.inlier_loss_weight),
        anchor_loss_weight=float(args.anchor_loss_weight),
        eval_split_fraction=float(args.eval_split_fraction),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        device=str(args.device),
        seed=int(args.seed),
    )
    run = train_matcha_dual_softmax_selector(samples, config, init_run=init_run)
    save_safe_patch_selector_checkpoint(run, Path(args.output_model))
    summary = {
        "stage": "stage_m2_matcha_dual_softmax_adapter",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_cache": {
            "path": str(args.sample_cache),
            "sha256": _sha(args.sample_cache),
            "metadata": metadata,
            "sample_count": int(samples.sample_count),
        },
        "init_safe_checkpoint": {
            "path": str(args.init_safe_checkpoint),
            "sha256": _sha(args.init_safe_checkpoint) if args.init_safe_checkpoint else "",
        },
        "config": asdict(config),
        "training": asdict(run.summary),
        "outputs": {"model": str(args.output_model)},
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
