"""Train a MATCHA-style coarse descriptor plus 8x8 offset-bin adapter."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineTrainingConfig,
    load_matcha_coarse_fine_training_set_npz,
    save_matcha_coarse_fine_adapter,
    train_matcha_coarse_fine_adapter,
)


def _sha(path: str) -> str:
    value = Path(path)
    return file_sha256_short(value) if value.exists() and value.is_file() else ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_cache", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--residual_hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dual_softmax_weight", type=float, default=1.0)
    parser.add_argument("--offset_loss_weight", type=float, default=0.25)
    parser.add_argument("--pair_fine_loss_weight", type=float, default=0.25)
    parser.add_argument("--confidence_loss_weight", type=float, default=0.1)
    parser.add_argument("--detector_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--detector_target_mode",
        choices=("matcha_confidence", "hard_negative_bce"),
        default="matcha_confidence",
    )
    parser.add_argument("--keypoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_weight", type=float, default=0.2)
    parser.add_argument("--hard_negative_margin", type=float, default=0.2)
    parser.add_argument("--eval_split_fraction", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--input_norm_mode", choices=("identity", "layernorm"), default="identity")
    parser.add_argument("--gate_mode", choices=("residual", "sigmoid"), default="residual")
    parser.add_argument("--residual_gate_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    samples, metadata = load_matcha_coarse_fine_training_set_npz(Path(args.sample_cache))
    config = MatchaCoarseFineTrainingConfig(
        output_dim=int(args.output_dim),
        residual_hidden_dim=int(args.residual_hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        temperature=float(args.temperature),
        dual_softmax_weight=float(args.dual_softmax_weight),
        offset_loss_weight=float(args.offset_loss_weight),
        pair_fine_loss_weight=float(args.pair_fine_loss_weight),
        confidence_loss_weight=float(args.confidence_loss_weight),
        detector_loss_weight=float(args.detector_loss_weight),
        detector_target_mode=str(args.detector_target_mode),
        keypoint_loss_weight=float(args.keypoint_loss_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
        eval_split_fraction=float(args.eval_split_fraction),
        group_size=int(args.group_size),
        input_norm_mode=str(args.input_norm_mode),
        gate_mode=str(args.gate_mode),
        residual_gate_scale=float(args.residual_gate_scale),
        device=str(args.device),
        seed=int(args.seed),
    )
    run = train_matcha_coarse_fine_adapter(samples, config)
    save_matcha_coarse_fine_adapter(run, Path(args.output_model))
    summary = {
        "stage": "matcha_coarse_fine_adapter_training",
        "elapsed_sec": float(time.perf_counter() - started),
        "sample_cache": {
            "path": str(args.sample_cache),
            "sha256": _sha(args.sample_cache),
            "metadata": metadata,
            "sample_count": int(samples.sample_count),
        },
        "config": asdict(config),
        "training": run.summary,
        "outputs": {"model": str(args.output_model)},
    }
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
