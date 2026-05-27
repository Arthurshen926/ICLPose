"""Build a landmark-associated Gaussian VFM feature field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFieldConfig,
    associate_landmarks_to_gaussians,
    load_gaussian_vfm_source_from_ply,
)
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def _load_xyz_by_track(track_observations: Path) -> dict[int, np.ndarray]:
    xyz_by_track: dict[int, np.ndarray] = {}
    for obs in load_colmap_track_observations_jsonl(track_observations):
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
    return xyz_by_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Associate 3D VFM landmarks to nearby Gaussians")
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--max_distance", type=float, default=0.05)
    parser.add_argument("--k_neighbors", type=int, default=4)
    parser.add_argument("--min_support", type=int, default=1)
    parser.add_argument("--max_landmark_variance", type=float, default=None)
    parser.add_argument("--min_landmark_observations", type=int, default=2)
    parser.add_argument("--max_gaussians", type=int, default=0)
    parser.add_argument("--no_l2_normalize_features", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    config = GaussianVFMFieldConfig(
        max_distance=args.max_distance,
        k_neighbors=args.k_neighbors,
        min_support=args.min_support,
        max_landmark_variance=args.max_landmark_variance,
        min_landmark_observations=args.min_landmark_observations,
        l2_normalize_features=not args.no_l2_normalize_features,
    )
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply), max_gaussians=args.max_gaussians)
    landmarks = LandmarkMapIndex.from_track_bank(
        load_selected_track_bank_npz(Path(args.landmark_bank)),
        _load_xyz_by_track(Path(args.track_observations)),
    )
    field = associate_landmarks_to_gaussians(source, landmarks, config)
    metadata = {
        **dict(field.metadata or {}),
        "gaussian_ply": str(args.gaussian_ply),
        "landmark_bank": str(args.landmark_bank),
        "track_observations": str(args.track_observations),
        "source_gaussian_count": int(source.xyz.shape[0]),
        "source_landmark_count": int(len(landmarks)),
    }
    field = type(field)(
        xyz=field.xyz,
        features=field.features,
        opacity=field.opacity,
        scale=field.scale,
        gaussian_indices=field.gaussian_indices,
        nearest_track_ids=field.nearest_track_ids,
        support_counts=field.support_counts,
        mean_distances=field.mean_distances,
        metadata=metadata,
    )
    field.save_npz(Path(args.output))
    summary = {
        "stage": "gaussian_vfm_field",
        "feature_dim": int(field.feature_dim),
        "feature_bearing_gaussian_count": int(len(field)),
        "source_gaussian_count": int(source.xyz.shape[0]),
        "source_landmark_count": int(len(landmarks)),
        "coverage_fraction": 0.0 if source.xyz.shape[0] == 0 else float(len(field) / source.xyz.shape[0]),
        "mean_support_count": 0.0 if len(field) == 0 else float(np.mean(field.support_counts)),
        "mean_association_distance": 0.0 if len(field) == 0 else float(np.mean(field.mean_distances)),
        "config": config.to_dict(),
        "outputs": {"field": str(args.output)},
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
