"""Build raw high-dimensional VFM 3D landmark feature banks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.landmark_feature_aggregation import (
    LandmarkAggregationConfig,
    aggregate_landmark_features,
    aggregate_landmark_features_torch,
    evaluate_landmark_observation_retrieval,
    evaluate_landmark_split_stability,
)
from feature_extract.vfm.map_lifting import mapability_summary, save_selected_track_bank_npz
from feature_extract.vfm.track_feature_sampling import (
    load_sampled_track_observations_npz,
    load_colmap_track_observations_jsonl,
    sample_token_track_observations,
    save_sampled_track_observations_npz,
)
from feature_extract.vfm.tokens import TokenBankManifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build raw VFM 3D landmark features from COLMAP tracks")
    parser.add_argument("--track_observations", default="")
    parser.add_argument("--token_manifest", default="")
    parser.add_argument("--sampled_observation_cache", default="")
    parser.add_argument("--write_sampled_observation_cache", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument(
        "--method",
        default="mean",
        choices=(
            "mean",
            "cosine_weighted_mean",
            "random_observation",
            "geometry_weighted",
            "robust_trimmed_mean",
            "view_consistent",
            "geometric_median",
            "medoid",
        ),
    )
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--missing", default="skip", choices=("skip", "error"))
    parser.add_argument(
        "--utility_mode",
        default="inverse_reprojection",
        choices=(
            "inverse_reprojection",
            "center",
            "inverse_reprojection_center",
            "view_consistency",
            "inverse_reprojection_center_view",
        ),
    )
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim_fraction", type=float, default=0.2)
    parser.add_argument("--view_consistent_keep", type=int, default=4)
    parser.add_argument("--geometric_median_iterations", type=int, default=32)
    parser.add_argument("--l2_normalize_observations", action="store_true")
    parser.add_argument("--weight_floor", type=float, default=1e-6)
    parser.add_argument("--torch_aggregate_device", default="")
    parser.add_argument("--skip_split_stability", action="store_true")
    parser.add_argument("--skip_retrieval_diagnostics", action="store_true")
    parser.add_argument("--retrieval_top_k", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    if args.sampled_observation_cache:
        sampled, cache_metadata = load_sampled_track_observations_npz(Path(args.sampled_observation_cache))
        colmap_observations = []
    else:
        if not args.track_observations or not args.token_manifest:
            raise ValueError(
                "provide --sampled_observation_cache or both --track_observations and --token_manifest"
            )
        track_path = Path(args.track_observations)
        manifest_path = Path(args.token_manifest)
        colmap_observations = load_colmap_track_observations_jsonl(track_path)
        sampled = sample_token_track_observations(
            colmap_observations,
            TokenBankManifest.from_json(manifest_path),
            layer_name=args.layer_name,
            missing=args.missing,
            utility_mode=args.utility_mode,
            weight_floor=args.weight_floor,
            sample_mode=args.sample_mode,
        )
        cache_metadata = {}
        if args.write_sampled_observation_cache:
            save_sampled_track_observations_npz(
                sampled,
                Path(args.write_sampled_observation_cache),
                metadata={
                    "track_observations": args.track_observations,
                    "token_manifest": args.token_manifest,
                    "layer_name": args.layer_name,
                    "sample_mode": args.sample_mode,
                    "utility_mode": args.utility_mode,
                    "missing": args.missing,
                },
            )
    config = LandmarkAggregationConfig(
        method=args.method,
        min_observations=args.min_observations,
        seed=args.seed,
        trim_fraction=args.trim_fraction,
        view_consistent_keep=args.view_consistent_keep,
        geometric_median_iterations=args.geometric_median_iterations,
        l2_normalize_observations=args.l2_normalize_observations,
        weight_floor=args.weight_floor,
    )
    if args.torch_aggregate_device:
        bank = aggregate_landmark_features_torch(sampled, config, device=args.torch_aggregate_device)
    else:
        bank = aggregate_landmark_features(sampled, config)
    output_bank = Path(args.output_bank)
    save_selected_track_bank_npz(bank, output_bank)

    stability = None
    if not args.skip_split_stability:
        stability = evaluate_landmark_split_stability(sampled, config, seed=args.seed)
    retrieval = None
    if not args.skip_retrieval_diagnostics:
        retrieval = evaluate_landmark_observation_retrieval(
            sampled,
            config,
            top_k=tuple(args.retrieval_top_k),
            seed=args.seed,
        )
    input_files = {}
    if args.track_observations:
        input_files["track_observations"] = {
            "path": args.track_observations,
            "sha256": file_sha256_short(Path(args.track_observations)),
        }
    if args.token_manifest:
        input_files["token_manifest"] = {
            "path": args.token_manifest,
            "sha256": file_sha256_short(Path(args.token_manifest)),
        }
    if args.sampled_observation_cache:
        input_files["sampled_observation_cache"] = {
            "path": args.sampled_observation_cache,
            "sha256": file_sha256_short(Path(args.sampled_observation_cache)),
        }
    if args.write_sampled_observation_cache:
        input_files["written_sampled_observation_cache"] = {
            "path": args.write_sampled_observation_cache,
        }

    summary = {
        "stage": "raw_vfm_3d_landmark_feature_aggregation",
        "aggregation": config.to_dict(),
        "input_observation_count": len(colmap_observations),
        "sampled_observation_count": len(sampled),
        "sampled_observation_cache_metadata": cache_metadata,
        "utility_mode": args.utility_mode,
        "sample_mode": args.sample_mode,
        "torch_aggregate_device": args.torch_aggregate_device,
        "mapability": asdict(mapability_summary(bank)),
        "diagnostics": {
            "split_stability": None if stability is None else stability.to_dict(),
            "retrieval": None if retrieval is None else retrieval.to_dict(),
        },
        "input_files": input_files,
        "output_files": {
            "bank": str(output_bank),
        },
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
