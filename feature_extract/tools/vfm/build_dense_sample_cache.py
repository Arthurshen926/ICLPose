"""Build persistent sampled dense-token caches for selector training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import candidate_bank_artifact_metadata
from feature_extract.vfm.dense_selector_training import (
    DenseSelectorTrainingConfig,
    build_dense_selector_sample_cache,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a persistent sampled dense-token cache")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--map_manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--spatial_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling_seed", type=int, default=None)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--diagnostic_query_limit", type=int, default=0)
    parser.add_argument("--sample_cache", default="", help="Optional existing cache to reuse and extend")
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)

    config = DenseSelectorTrainingConfig(
        seed=args.seed,
        eval_split_fraction=args.eval_fraction,
        layer_name=args.layer_name,
        spatial_samples=args.spatial_samples,
        sampling_seed=args.sampling_seed,
        diagnostic_query_limit=args.diagnostic_query_limit,
        sample_cache=args.sample_cache,
        sample_cache_dtype=args.cache_dtype,
    )
    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    summary = build_dense_selector_sample_cache(
        bank=bank,
        query_manifest=TokenBankManifest.from_json(Path(args.query_manifest)),
        map_manifest=TokenBankManifest.from_json(Path(args.map_manifest)),
        config=config,
        output=Path(args.output),
    )
    payload = {
        "config": asdict(config),
        "inputs": candidate_bank_artifact_metadata(
            bank,
            Path(args.bank),
            {
                "query_manifest": args.query_manifest,
                "map_manifest": args.map_manifest,
                "input_sample_cache": args.sample_cache,
            },
        ),
        **summary,
    }
    output_summary = Path(args.summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
