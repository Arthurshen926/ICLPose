"""Build Stage H2 raw Gaussian anchor maps for a min-votes sweep.

This tool avoids recomputing all-view owner visibility for every min-votes
setting. It votes once, samples each requested min-votes setting, aggregates the
union once, then subsets the union map into per-setting maps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import (
    _load_camera_by_image,
    _load_feature,
    _parse_default_camera,
    _select_records,
    _vote_stats,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_raw_landmarks import (
    RawGaussianFeatureAggregationConfig,
    VfmGaussianAnchorVoteConfig,
    aggregate_raw_vfm_features_to_gaussian_anchors,
    sample_gaussian_indices_from_votes,
    subset_gaussian_anchor_map_by_source_indices,
    vote_gaussians_from_vfm_token_saliency,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, load_gaussian_vfm_source_from_ply
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_min_votes_list(text: str) -> list[int]:
    values = [int(item) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("--min_votes_list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("--min_votes_list values must be positive")
    return sorted(set(values))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage H2 raw VFM Gaussian anchor map min-votes sweep")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_selection", default="uniform", choices=("prefix", "uniform"))
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--vote_top_token_fraction", type=float, default=0.05)
    parser.add_argument("--vote_min_saliency", type=float, default=0.0)
    parser.add_argument("--vote_saliency_mode", default="norm", choices=("norm", "local_contrast"))
    parser.add_argument("--vote_min_owner_opacity", type=float, default=0.0)
    parser.add_argument("--max_anchors", type=int, default=10000)
    parser.add_argument("--min_votes_list", default="1,2,3")
    parser.add_argument("--sample_opacity_power", type=float, default=0.0)
    parser.add_argument("--sample_min_opacity", type=float, default=0.0)
    parser.add_argument("--nms_voxel_size", type=float, default=0.03)
    parser.add_argument("--min_observations", type=int, default=2)
    parser.add_argument("--disable_token_owner_visibility", action="store_true")
    parser.add_argument("--aggregation_owner_min_opacity", type=float, default=None)
    parser.add_argument("--no_l2_normalize_observations", action="store_true")
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--default_camera", default="2,1024,576,883,512,288,0")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args(argv)

    min_votes_values = _parse_min_votes_list(args.min_votes_list)
    manifest = TokenBankManifest.from_json(Path(args.reference_manifest))
    manifest.validate(verify_checksums=False)
    pose_by_image = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.reference_pose_file))}
    camera_by_image = _load_camera_by_image(args.camera_model_dir)
    fallback_camera = _parse_default_camera(args.default_camera)
    records = [record for record in manifest.records if record.image_id in pose_by_image]
    records = _select_records(records, int(args.max_views), args.view_selection)
    views = [
        GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=_load_feature(Path(record.token_path), args.layer_name),
            pose_w2c=pose_by_image[record.image_id].pose_w2c,
            camera=camera_by_image.get(record.image_id, fallback_camera),
        )
        for record in records
    ]
    if not views:
        raise ValueError("no reference views with both tokens and poses")

    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=int(args.max_gaussians))
    vote_config = VfmGaussianAnchorVoteConfig(
        top_token_fraction=float(args.vote_top_token_fraction),
        min_saliency=float(args.vote_min_saliency),
        saliency_mode=args.vote_saliency_mode,
        min_owner_opacity=float(args.vote_min_owner_opacity),
    )
    votes, vote_summary = vote_gaussians_from_vfm_token_saliency(source, views, vote_config)
    sampled_by_min_votes: dict[int, np.ndarray] = {}
    for min_votes in min_votes_values:
        sampled_by_min_votes[int(min_votes)] = sample_gaussian_indices_from_votes(
            source,
            votes,
            max_anchors=int(args.max_anchors),
            min_votes=int(min_votes),
            nms_voxel_size=float(args.nms_voxel_size),
            opacity_power=float(args.sample_opacity_power),
            min_opacity=float(args.sample_min_opacity),
        )
    non_empty = [sampled for sampled in sampled_by_min_votes.values() if sampled.size > 0]
    if not non_empty:
        raise ValueError("no sampled Gaussians for requested min-votes settings")
    union_sampled = np.unique(np.concatenate(non_empty).astype(np.int64, copy=False))

    aggregation_config = RawGaussianFeatureAggregationConfig(
        min_observations=int(args.min_observations),
        l2_normalize_observations=not bool(args.no_l2_normalize_observations),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
        require_token_owner_visibility=not bool(args.disable_token_owner_visibility),
        owner_min_opacity=(
            float(args.aggregation_owner_min_opacity)
            if args.aggregation_owner_min_opacity is not None
            else float(args.vote_min_owner_opacity)
        ),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    union_anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        source,
        union_sampled,
        views,
        aggregation_config,
        metadata={
            "vote_summary": vote_summary,
            "sampled_gaussian_count": int(union_sampled.size),
            "sweep_union": True,
            "min_votes_values": [int(value) for value in min_votes_values],
        },
    )
    union_anchor_map.save_npz(output_root / "union_raw_anchor_map.npz")
    np.savez_compressed(
        output_root / "votes.npz",
        vote_counts=votes.astype(np.int64, copy=False),
        union_sampled_indices=union_sampled.astype(np.int64, copy=False),
        **{
            f"sampled_min{int(min_votes)}": sampled.astype(np.int64, copy=False)
            for min_votes, sampled in sampled_by_min_votes.items()
        },
    )

    per_setting = []
    for min_votes, sampled in sampled_by_min_votes.items():
        setting_dir = output_root / f"min_votes_{int(min_votes)}"
        setting_dir.mkdir(parents=True, exist_ok=True)
        anchor_map = subset_gaussian_anchor_map_by_source_indices(
            union_anchor_map,
            source,
            sampled,
            metadata={
                "stage": "stage_h2_raw_gaussian_anchor_map",
                "vote_summary": vote_summary,
                "sampled_gaussian_count": int(sampled.size),
                "sweep_union_sampled_gaussian_count": int(union_sampled.size),
                "min_votes": int(min_votes),
            },
        )
        map_path = setting_dir / "raw_anchor_map.npz"
        summary_path = setting_dir / "summary.json"
        anchor_map.save_npz(map_path)
        setting_summary = {
            "stage": "stage_h2_raw_gaussian_anchor_map_sweep_setting",
            "source_gaussian_count": int(source.xyz.shape[0]),
            "sampled_gaussian_count": int(sampled.size),
            "union_sampled_gaussian_count": int(union_sampled.size),
            "anchor_count": int(len(anchor_map)),
            "feature_dim": int(anchor_map.feature_dim),
            "view_count": int(len(views)),
            "vote_config": vote_config.to_dict(),
            "sampling_config": {
                "max_anchors": int(args.max_anchors),
                "min_votes": int(min_votes),
                "nms_voxel_size": float(args.nms_voxel_size),
                "opacity_power": float(args.sample_opacity_power),
                "min_opacity": float(args.sample_min_opacity),
            },
            "selected_vote_stats": _vote_stats(votes[sampled]),
            "aggregation_config": aggregation_config.to_dict(),
            "vote_summary": vote_summary,
            "inputs": {
                "gaussian_ply": args.gaussian_ply,
                "reference_manifest": args.reference_manifest,
                "reference_pose_file": args.reference_pose_file,
                "camera_model_dir": args.camera_model_dir,
                "layer_name": args.layer_name,
                "view_selection": args.view_selection,
            },
            "outputs": {
                "anchor_map": str(map_path),
                "summary": str(summary_path),
                "votes": str(output_root / "votes.npz"),
            },
            "missing_after_union_aggregation": int(anchor_map.metadata.get("missing_after_union_aggregation", 0)),
        }
        summary_path.write_text(json.dumps(setting_summary, indent=2, sort_keys=True) + "\n")
        per_setting.append(setting_summary)

    sweep_summary = {
        "stage": "stage_h2_raw_gaussian_anchor_map_sweep",
        "source_gaussian_count": int(source.xyz.shape[0]),
        "union_sampled_gaussian_count": int(union_sampled.size),
        "union_anchor_count": int(len(union_anchor_map)),
        "feature_dim": int(union_anchor_map.feature_dim),
        "view_count": int(len(views)),
        "min_votes_values": [int(value) for value in min_votes_values],
        "vote_config": vote_config.to_dict(),
        "aggregation_config": aggregation_config.to_dict(),
        "vote_summary": vote_summary,
        "settings": per_setting,
        "outputs": {
            "union_anchor_map": str(output_root / "union_raw_anchor_map.npz"),
            "votes": str(output_root / "votes.npz"),
        },
    }
    (output_root / "sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
