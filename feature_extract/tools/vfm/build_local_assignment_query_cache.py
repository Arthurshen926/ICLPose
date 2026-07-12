"""Build frozen query-side global/heatmap features for local assignment episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_local_assignment_probe_pose import _validate_probe_binding
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import UniqueTrackCandidateSet
from feature_extract.vfm.localization.pipeline import _load_feature_map
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import _sample_feature_vectors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_arrays", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=576)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    probe_path = Path(args.probe_arrays)
    with np.load(probe_path, allow_pickle=False) as data:
        query_ids = tuple(str(item) for item in data["query_ids"].tolist())
        query_xy = np.asarray(data["query_xy"], dtype=np.float64)
        candidates = UniqueTrackCandidateSet(
            bank_row_indices=np.asarray(data["bank_row_indices"], dtype=np.int64),
            track_ids=np.asarray(data["candidate_track_ids"], dtype=np.int64),
            prototype_ids=np.asarray(data["candidate_prototype_ids"], dtype=np.int64),
            coarse_scores=np.asarray(data["coarse_scores"], dtype=np.float32),
        )
    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    binding = _validate_probe_binding(
        probe_path=probe_path,
        landmark_metadata=landmark_metadata,
        candidates=candidates,
        landmark_index=landmark_index,
    )
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = {str(record.image_id): record for record in manifest.records}
    if set(query_ids) - set(records):
        raise ValueError(f"query manifest is missing probe images: {sorted(set(query_ids) - set(records))[:5]}")
    run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=str(args.device))
    mapper = JointFeatureMapper(run.model, device=str(args.device))
    output_dim = int(landmark_index.feature_dim)
    descriptors = np.zeros((len(query_ids), output_dim), dtype=np.float32)
    heatmap_scores = np.zeros((len(query_ids),), dtype=np.float32)
    rows_by_image: dict[str, list[int]] = {}
    for row, query_id in enumerate(query_ids):
        rows_by_image.setdefault(str(query_id), []).append(int(row))
    for query_id, rows in rows_by_image.items():
        record = records[query_id]
        raw_map = _load_feature_map(Path(record.token_path), key=str(args.feature_key))
        mapped = mapper.project(raw_map)
        image_width = int(args.image_width)
        image_height = int(args.image_height)
        xy = query_xy[rows]
        descriptors[rows] = _sample_feature_vectors(
            mapped.coarse_descriptors,
            xy,
            np.full((len(rows),), image_width, dtype=np.int64),
            np.full((len(rows),), image_height, dtype=np.int64),
            str(args.sample_mode),
        )
        if mapped.heatmap is None:
            raise ValueError("joint mapper did not produce a heatmap")
        sampled_heatmap = _sample_feature_vectors(
            np.asarray(mapped.heatmap, dtype=np.float32)[None],
            xy,
            np.full((len(rows),), image_width, dtype=np.int64),
            np.full((len(rows),), image_height, dtype=np.int64),
            str(args.sample_mode),
        )
        heatmap_scores[rows] = sampled_heatmap[:, 0]
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(norms, 1e-8)
    metadata = {
        "stage": "local_assignment_query_feature_cache",
        "binding": binding,
        "probe_arrays": str(probe_path),
        "probe_arrays_sha256": file_sha256_short(probe_path),
        "query_manifest": str(args.query_manifest),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "matcha_joint_checkpoint_sha256": file_sha256_short(Path(args.matcha_joint_checkpoint)),
        "descriptor_space_id": str(landmark_metadata.get("descriptor_space_id", "")),
        "projection_space_id": str(
            dict(landmark_metadata.get("descriptor_space_manifest", {})).get("projection_space_id", "")
        ),
        "feature_key": str(args.feature_key),
        "sample_mode": str(args.sample_mode),
        "sampling_convention": "sfm_pixel_endpoint_to_token_endpoint_v1",
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "query_count": int(len(rows_by_image)),
        "observation_count": int(len(query_ids)),
        "feature_dimension": int(output_dim),
    }
    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        query_global_features=np.asarray(descriptors, dtype=np.float16),
        query_heatmap_scores=np.asarray(heatmap_scores, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        **metadata,
        "heatmap_score_mean": float(np.mean(heatmap_scores)),
        "heatmap_score_std": float(np.std(heatmap_scores)),
        "output_cache": str(output_path),
        "output_cache_sha256": file_sha256_short(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
