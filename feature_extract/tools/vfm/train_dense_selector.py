"""Train a selector directly on dense token maps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch

from feature_extract.vfm.artifacts import candidate_bank_artifact_metadata
from feature_extract.vfm.dense_selector_training import (
    DenseSelectorTrainingConfig,
    run_dense_selector_training,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.selector_descriptor_scoring import _extract_state_dict, infer_selector_dims_from_state_dict
from feature_extract.vfm.track_feature_sampling import (
    load_colmap_track_observations_jsonl,
    sample_token_track_observations,
)
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a dense VFM selector smoke model")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--map_manifest", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dim", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--rank_temperature", type=float, default=0.1)
    parser.add_argument("--sparsity_weight", type=float, default=0.005)
    parser.add_argument("--spatial_samples", type=int, default=256)
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=None,
        help="Optional seed for spatial token sampling; use a fixed value to share sampled caches across training seeds",
    )
    parser.add_argument(
        "--diagnostic_query_limit",
        type=int,
        default=0,
        help="Limit initial/final loss and top1 diagnostics without limiting the training query pool",
    )
    parser.add_argument(
        "--preload_sampled_features",
        action="store_true",
        help="Preload sampled dense features used by the training pool to avoid repeated NPZ scans",
    )
    parser.add_argument("--sample_cache", default="", help="Load sampled dense features from a persistent NPZ cache")
    parser.add_argument("--write_sample_cache", default="", help="Write the sampled dense features used by this run")
    parser.add_argument("--sample_cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--basin_bce_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--basin_translation_threshold_m", type=float, default=5.0)
    parser.add_argument("--basin_rotation_threshold_deg", type=float, default=10.0)
    parser.add_argument("--hard_negative_margin", type=float, default=0.1)
    parser.add_argument(
        "--utility_weighted_pooling",
        action="store_true",
        help="Mean-pool selected descriptors with the selector utility map so utility receives ranking gradients",
    )
    parser.add_argument(
        "--init_anchor_weight",
        type=float,
        default=0.0,
        help="L2 regularization weight that keeps a warm-started selector near its initial parameters",
    )
    parser.add_argument("--track_observations", default="", help="Optional COLMAP track observation JSONL")
    parser.add_argument(
        "--track_token_manifest",
        default="",
        help="Token manifest used to sample track observations; defaults to --map_manifest",
    )
    parser.add_argument("--track_missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--track_supervision_weight", type=float, default=0.0)
    parser.add_argument("--track_batch_size", type=int, default=64)
    parser.add_argument("--track_min_observations", type=int, default=2)
    parser.add_argument("--track_temperature", type=float, default=0.1)
    parser.add_argument("--track_consistency_weight", type=float, default=1.0)
    parser.add_argument("--track_contrastive_weight", type=float, default=1.0)
    parser.add_argument("--track_utility_weight", type=float, default=0.5)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    output_dim = args.output_dim
    group_size = args.group_size
    if args.init_checkpoint:
        state_dict = _extract_state_dict(torch.load(Path(args.init_checkpoint), map_location="cpu"))
        _input_dim, inferred_output_dim, inferred_group_size = infer_selector_dims_from_state_dict(state_dict)
        if output_dim is None:
            output_dim = inferred_output_dim
        if group_size is None:
            group_size = inferred_group_size
    else:
        if output_dim is None:
            output_dim = 64
        if group_size is None:
            group_size = 16

    config = DenseSelectorTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        output_dim=output_dim,
        group_size=group_size,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        eval_split_fraction=args.eval_fraction,
        rank_temperature=args.rank_temperature,
        sparsity_weight=args.sparsity_weight,
        layer_name=args.layer_name,
        spatial_samples=args.spatial_samples,
        sampling_seed=args.sampling_seed,
        init_checkpoint=args.init_checkpoint,
        diagnostic_query_limit=args.diagnostic_query_limit,
        preload_sampled_features=args.preload_sampled_features,
        sample_cache=args.sample_cache,
        write_sample_cache=args.write_sample_cache,
        sample_cache_dtype=args.sample_cache_dtype,
        basin_bce_weight=args.basin_bce_weight,
        hard_negative_weight=args.hard_negative_weight,
        basin_translation_threshold_m=args.basin_translation_threshold_m,
        basin_rotation_threshold_deg=args.basin_rotation_threshold_deg,
        hard_negative_margin=args.hard_negative_margin,
        utility_weighted_pooling=args.utility_weighted_pooling,
        init_anchor_weight=args.init_anchor_weight,
        track_supervision_weight=args.track_supervision_weight,
        track_batch_size=args.track_batch_size,
        track_min_observations=args.track_min_observations,
        track_temperature=args.track_temperature,
        track_consistency_weight=args.track_consistency_weight,
        track_contrastive_weight=args.track_contrastive_weight,
        track_utility_weight=args.track_utility_weight,
    )
    candidate_bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    map_manifest = TokenBankManifest.from_json(Path(args.map_manifest))
    sampled_track_observations = None
    colmap_track_count = 0
    if args.track_observations:
        track_path = Path(args.track_observations)
        track_manifest_path = Path(args.track_token_manifest) if args.track_token_manifest else Path(args.map_manifest)
        colmap_track_observations = load_colmap_track_observations_jsonl(track_path)
        colmap_track_count = len(colmap_track_observations)
        sampled_track_observations = sample_token_track_observations(
            colmap_track_observations,
            TokenBankManifest.from_json(track_manifest_path),
            layer_name=args.layer_name,
            missing=args.track_missing,
        )
    run = run_dense_selector_training(
        bank=candidate_bank,
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        config=config,
        track_observations=sampled_track_observations,
    )
    input_paths = {
        "query_manifest": args.query_manifest,
        "map_manifest": args.map_manifest,
        "init_checkpoint": args.init_checkpoint,
    }
    if args.track_observations:
        input_paths["track_observations"] = args.track_observations
        input_paths["track_token_manifest"] = args.track_token_manifest or args.map_manifest
    inputs = candidate_bank_artifact_metadata(candidate_bank, Path(args.bank), input_paths)
    if args.track_observations:
        inputs["track_supervision"] = {
            "colmap_observation_count": colmap_track_count,
            "sampled_observation_count": len(sampled_track_observations or ()),
            "missing": args.track_missing,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "inputs": inputs,
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
