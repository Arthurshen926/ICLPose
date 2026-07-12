"""Build a lightweight real-image geometry index for ALIKE support observations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    build_support_observation_geometry_index,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_track_observations_jsonl", required=True)
    parser.add_argument("--support_feature_cache", required=True)
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    observation_path = Path(args.support_track_observations_jsonl)
    feature_cache_path = Path(args.support_feature_cache)
    observations = load_colmap_track_observations_jsonl(observation_path)
    observation_tracks = np.asarray([row.track_id for row in observations], dtype=np.int64)
    with np.load(feature_cache_path, allow_pickle=False) as data:
        feature_metadata = json.loads(str(data["metadata_json"].item()))
        feature_tracks = np.asarray(data["track_ids"], dtype=np.int64)
        descriptor_shape = tuple(int(value) for value in data["descriptors"].shape)
        detector_score_count = int(data["detector_scores"].shape[0])
    if feature_metadata.get("format") != "full_support_alike_observation_features_v1":
        raise ValueError("support feature cache is not a full ALIKE observation cache")
    observation_hash = file_sha256_short(observation_path)
    if str(feature_metadata.get("support_track_observations_sha256", "")) != observation_hash:
        raise ValueError("support feature cache was built from a different observation JSONL")
    if not np.array_equal(feature_tracks, observation_tracks):
        raise ValueError("support feature cache rows differ from the observation JSONL")
    if descriptor_shape[0] != len(observations) or detector_score_count != len(observations):
        raise ValueError("support feature cache arrays have incompatible lengths")

    index = build_support_observation_geometry_index(observations)
    metadata = {
        "support_track_observations_jsonl": str(observation_path),
        "support_track_observations_sha256": observation_hash,
        "support_feature_cache": str(feature_cache_path),
        "support_feature_cache_sha256": file_sha256_short(feature_cache_path),
        "observation_count": int(len(index)),
        "track_count": int(np.unique(index.track_ids).size),
        "image_count": int(len(index.image_ids)),
        "descriptor_dimension": int(descriptor_shape[1]),
        "alike_model_name": str(feature_metadata.get("alike_model_name", "")),
        "alike_checkpoint_sha256": str(feature_metadata.get("model_checkpoint_sha256", "")),
        "coordinate_source": "sfm_observation_xy",
        "feature_storage": "external_source_row_reference",
    }
    output_path = Path(args.output_index)
    save_support_observation_geometry_index_npz(index, output_path, metadata=metadata)
    summary = {
        "stage": "local_maplet_support_observation_geometry",
        "metadata": metadata,
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "geometry_index": str(output_path),
            "summary": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
