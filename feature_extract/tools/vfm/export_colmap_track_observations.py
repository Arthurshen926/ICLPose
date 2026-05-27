"""Export COLMAP track observations for VFM mapability experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.colmap_tracks import load_colmap_track_observations
from feature_extract.vfm.landmark_visibility import (
    LandmarkVisibilityIndex,
    coverage_balanced_track_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export COLMAP track observations")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--max_observations", type=int, default=0)
    parser.add_argument("--sampling_strategy", default="prefix", choices=("prefix", "coverage_balanced"))
    parser.add_argument("--observations_per_track", type=int, default=3)
    parser.add_argument("--min_sampled_track_observations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility_npz", default="")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    observations = load_colmap_track_observations(
        Path(args.model_dir),
        min_track_length=args.min_track_length,
    )
    available_observation_count = len(observations)
    available_track_count = len({obs.track_id for obs in observations})
    if args.visibility_npz:
        LandmarkVisibilityIndex.from_observations(observations).save_npz(
            Path(args.visibility_npz),
            metadata={
                "model_dir": str(Path(args.model_dir)),
                "min_track_length": args.min_track_length,
                "source_observation_count": available_observation_count,
                "source_track_count": available_track_count,
            },
        )
    if args.max_observations > 0:
        if args.sampling_strategy == "prefix":
            observations = observations[: args.max_observations]
        elif args.sampling_strategy == "coverage_balanced":
            observations = coverage_balanced_track_observations(
                observations,
                max_observations=args.max_observations,
                observations_per_track=args.observations_per_track,
                min_track_observations=args.min_sampled_track_observations,
                seed=args.seed,
            )
        else:
            raise ValueError(f"unsupported sampling_strategy: {args.sampling_strategy}")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for obs in observations:
        lines.append(
            json.dumps(
                {
                    "track_id": obs.track_id,
                    "image_id": obs.image_id,
                    "point2d_idx": obs.point2d_idx,
                    "xy": list(obs.xy),
                    "xyz": obs.xyz.astype(float).tolist(),
                    "track_length": obs.track_length,
                    "reprojection_error": obs.reprojection_error,
                    "camera_id": obs.camera_id,
                    "image_width": obs.image_width,
                    "image_height": obs.image_height,
                    "camera_center": None if obs.camera_center is None else obs.camera_center.astype(float).tolist(),
                    "viewing_ray": None if obs.viewing_ray is None else obs.viewing_ray.astype(float).tolist(),
                },
                sort_keys=True,
            )
        )
    output_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""))

    track_ids = {obs.track_id for obs in observations}
    image_ids = {obs.image_id for obs in observations}
    summary = {
        "model_dir": str(Path(args.model_dir)),
        "min_track_length": args.min_track_length,
        "available_observation_count": available_observation_count,
        "available_track_count": available_track_count,
        "observation_count": len(observations),
        "track_count": len(track_ids),
        "image_count": len(image_ids),
        "sampling_strategy": args.sampling_strategy,
        "observations_per_track": args.observations_per_track,
        "min_sampled_track_observations": args.min_sampled_track_observations,
        "seed": args.seed,
        "visibility_npz": args.visibility_npz,
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
