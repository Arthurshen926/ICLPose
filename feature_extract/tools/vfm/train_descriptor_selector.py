"""Train a descriptor-level VFM selector on fixed candidate banks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch

from feature_extract.vfm.artifacts import candidate_bank_artifact_metadata
from feature_extract.vfm.descriptor_selector_training import (
    DescriptorSelectorTrainingConfig,
    run_descriptor_selector_training,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a VFM descriptor selector smoke model")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_descriptors", required=True)
    parser.add_argument("--map_descriptors", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--rank_temperature", type=float, default=0.1)
    parser.add_argument("--sparsity_weight", type=float, default=0.005)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    config = DescriptorSelectorTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        output_dim=args.output_dim,
        group_size=args.group_size,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eval_split_fraction=args.eval_fraction,
        rank_temperature=args.rank_temperature,
        sparsity_weight=args.sparsity_weight,
    )
    candidate_bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    run = run_descriptor_selector_training(
        bank=candidate_bank,
        query_descriptors=TokenDescriptorBank.from_npz(Path(args.query_descriptors)),
        map_descriptors=TokenDescriptorBank.from_npz(Path(args.map_descriptors)),
        config=config,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "inputs": candidate_bank_artifact_metadata(
                    candidate_bank,
                    Path(args.bank),
                    {
                        "query_descriptors": args.query_descriptors,
                        "map_descriptors": args.map_descriptors,
                    },
                ),
                "result": asdict(run.summary),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run.selector.state_dict(), checkpoint)


if __name__ == "__main__":
    main()
