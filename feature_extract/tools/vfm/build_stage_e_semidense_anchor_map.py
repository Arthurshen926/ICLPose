"""Build Stage E semi-dense reliable anchor maps from sparse SfM landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.semidense_anchor_map import (
    SemiDenseAnchorConfig,
    build_sfm_guided_semidense_anchor_map,
    semidense_anchor_map_stats,
    write_semidense_anchor_visualizations,
)
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _track_stats(track_observations: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    reproj_by_track: dict[int, list[float]] = {}
    for obs in load_colmap_track_observations_jsonl(Path(track_observations)):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        reproj_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return xyz_by_track, {
        int(track_id): float(np.mean(values)) for track_id, values in reproj_by_track.items() if values
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build Stage E SfM-guided semi-dense reliable anchor map")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--max_distance", type=float, default=0.05)
    parser.add_argument("--k_neighbors", type=int, default=2)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--max_gaussian_scale", type=float, default=0.0)
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--min_landmark_observations", type=int, default=2)
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--no_sparse", action="store_true")
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--visual_max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--visualization_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args(argv)

    xyz_by_track, reprojection_by_track = _track_stats(Path(args.track_observations))
    sparse = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.landmark_bank)),
        xyz_by_track,
        reprojection_by_track,
    )
    gaussians = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=int(args.max_gaussians))
    config = SemiDenseAnchorConfig(
        max_distance=float(args.max_distance),
        k_neighbors=int(args.k_neighbors),
        min_support=int(args.min_support),
        min_opacity=float(args.min_opacity),
        max_gaussian_scale=None if float(args.max_gaussian_scale) <= 0.0 else float(args.max_gaussian_scale),
        max_landmark_variance=args.max_landmark_variance,
        min_landmark_observations=int(args.min_landmark_observations),
        include_sparse=not bool(args.no_sparse),
        l2_normalize_features=not bool(args.no_l2_normalize_features),
    )
    semidense = build_sfm_guided_semidense_anchor_map(sparse, gaussians, config)
    semidense.save_npz(Path(args.output_npz))
    visual_outputs = write_semidense_anchor_visualizations(
        Path(args.visualization_dir),
        sparse,
        semidense,
        max_points=int(args.visual_max_points),
        seed=int(args.seed),
    )
    summary = {
        "stage": "stage_e_semidense_reliable_anchor_map",
        "method": "sfm_guided_gaussian_anchor_expansion",
        "config": config.to_dict(),
        "stats": semidense_anchor_map_stats(
            semidense,
            sparse_landmark_count=len(sparse),
            source_gaussian_count=int(gaussians.xyz.shape[0]),
        ),
        "inputs": {
            "gaussian_ply": args.gaussian_ply,
            "landmark_bank": args.landmark_bank,
            "track_observations": args.track_observations,
            "max_gaussians": int(args.max_gaussians),
        },
        "outputs": {
            "semidense_npz": args.output_npz,
            **{key: str(value) for key, value in visual_outputs.items()},
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
