"""Build ULF-Loc-style sparse Gaussian VFM landmark anchors.

This side-path samples reliable feature-bearing Gaussians from an already fused
Gaussian VFM field. It does not replace the canonical SfM landmark pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.semidense_anchor_map import (
    GaussianConsensusAnchorConfig,
    build_gaussian_consensus_anchor_map,
)


def _array_stats(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summary(field: GaussianVFMField, config: GaussianConsensusAnchorConfig, anchor_map) -> dict[str, object]:
    return {
        "stage": "stage_h_gaussian_consensus_anchor_map",
        "source_gaussian_count": int(len(field)),
        "candidate_gaussian_count": int(anchor_map.metadata.get("candidate_gaussian_count", len(anchor_map))),
        "anchor_count": int(len(anchor_map)),
        "feature_dim": int(anchor_map.feature_dim),
        "keep_fraction": float(len(anchor_map) / max(len(field), 1)),
        "config": config.to_dict(),
        "source_field_metadata": dict(field.metadata or {}),
        "quality": _array_stats(anchor_map.quality_scores),
        "support_count": _array_stats(anchor_map.support_counts),
        "opacity": _array_stats(anchor_map.opacity),
        "scale": _array_stats(anchor_map.scale),
        "mean_distance": _array_stats(anchor_map.mean_distances),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage H Gaussian consensus VFM landmark anchors")
    parser.add_argument("--gaussian_field", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--source_landmark_bank",
        default="",
        help="Optional selected track bank used only to inherit nearest-track reference visibility.",
    )
    parser.add_argument("--max_anchors", type=int, default=20000)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--min_opacity", type=float, default=0.02)
    parser.add_argument("--max_gaussian_scale", type=float, default=None)
    parser.add_argument("--max_mean_distance", type=float, default=None)
    parser.add_argument("--nms_voxel_size", type=float, default=0.03)
    parser.add_argument("--support_weight", type=float, default=1.0)
    parser.add_argument("--opacity_weight", type=float, default=1.0)
    parser.add_argument("--distance_weight", type=float, default=0.5)
    parser.add_argument("--scale_weight", type=float, default=0.5)
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    args = parser.parse_args(argv)

    field = GaussianVFMField.load_npz(Path(args.gaussian_field))
    config = GaussianConsensusAnchorConfig(
        max_anchors=int(args.max_anchors),
        min_support=int(args.min_support),
        min_opacity=float(args.min_opacity),
        max_gaussian_scale=args.max_gaussian_scale,
        max_mean_distance=args.max_mean_distance,
        nms_voxel_size=float(args.nms_voxel_size),
        support_weight=float(args.support_weight),
        opacity_weight=float(args.opacity_weight),
        distance_weight=float(args.distance_weight),
        scale_weight=float(args.scale_weight),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
    )
    observation_ids = None
    if args.source_landmark_bank:
        bank = load_selected_track_bank_npz(Path(args.source_landmark_bank))
        observation_ids = {
            int(track_id): tuple(track.observation_image_ids)
            for track_id, track in bank.tracks.items()
        }
    anchor_map = build_gaussian_consensus_anchor_map(
        field,
        config,
        nearest_track_observation_image_ids=observation_ids,
    )
    output_npz = Path(args.output_npz)
    summary_json = Path(args.summary_json)
    anchor_map.save_npz(output_npz)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_summary(field, config, anchor_map), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
