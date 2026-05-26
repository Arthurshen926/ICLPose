"""Train a selector with explicit COLMAP track/geometry utility supervision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.track_feature_sampling import (
    load_colmap_track_observations_jsonl,
    sample_token_track_observations,
)
from feature_extract.vfm.track_utility_training import TrackUtilityTrainingConfig, run_track_utility_training
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Track-supervised selector pretraining")
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--utility_weight", type=float, default=0.5)
    parser.add_argument("--sparsity_weight", type=float, default=0.005)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args(argv)

    track_path = Path(args.track_observations)
    manifest_path = Path(args.token_manifest)
    colmap_observations = load_colmap_track_observations_jsonl(track_path)
    sampled = sample_token_track_observations(
        colmap_observations,
        TokenBankManifest.from_json(manifest_path),
        layer_name=args.layer_name,
        missing=args.missing,
    )
    config = TrackUtilityTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        output_dim=args.output_dim,
        group_size=args.group_size,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        min_observations=args.min_observations,
        temperature=args.temperature,
        consistency_weight=args.consistency_weight,
        contrastive_weight=args.contrastive_weight,
        utility_weight=args.utility_weight,
        sparsity_weight=args.sparsity_weight,
        init_checkpoint=args.init_checkpoint,
    )
    run = run_track_utility_training(sampled, config)
    payload = {
        "config": asdict(config),
        "inputs": {
            "input_files": {
                "track_observations": {
                    "path": str(track_path),
                    "sha256": file_sha256_short(track_path),
                },
                "token_manifest": {
                    "path": str(manifest_path),
                    "sha256": file_sha256_short(manifest_path),
                },
            },
            "colmap_observation_count": len(colmap_observations),
            "sampled_observation_count": len(sampled),
        },
        "result": asdict(run.summary),
    }
    if args.init_checkpoint:
        payload["inputs"]["input_files"]["init_checkpoint"] = {
            "path": args.init_checkpoint,
            "sha256": file_sha256_short(Path(args.init_checkpoint)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run.selector.state_dict(), checkpoint)


if __name__ == "__main__":
    main()
