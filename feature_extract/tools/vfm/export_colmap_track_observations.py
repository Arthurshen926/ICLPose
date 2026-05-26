"""Export COLMAP track observations for VFM mapability experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.colmap_tracks import load_colmap_track_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Export COLMAP track observations")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--min_track_length", type=int, default=2)
    parser.add_argument("--max_observations", type=int, default=0)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    observations = load_colmap_track_observations(
        Path(args.model_dir),
        min_track_length=args.min_track_length,
    )
    available_observation_count = len(observations)
    available_track_count = len({obs.track_id for obs in observations})
    if args.max_observations > 0:
        observations = observations[: args.max_observations]

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
    }
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
