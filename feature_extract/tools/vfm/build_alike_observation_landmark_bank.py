"""Build real-image ALIKE observation landmark banks for detector proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
    save_landmark_index_npz,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    extract_real_image_observation_features,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_track_observations_jsonl", required=True)
    parser.add_argument("--source_landmark_bank", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    return parser.parse_args(argv)


def aggregate_alike_track_descriptors(
    track_ids: np.ndarray,
    descriptors: np.ndarray,
    detector_scores: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized normalized or detector-weighted track descriptor aggregation."""

    tracks = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    features, valid = normalize_rows(np.asarray(descriptors, dtype=np.float32))
    scores = np.asarray(detector_scores, dtype=np.float32).reshape(-1)
    if features.shape[0] != len(tracks) or len(scores) != len(tracks):
        raise ValueError("track ids, descriptors and detector scores must have the same length")
    if str(method) not in {"normalized_mean", "detector_weighted_mean"}:
        raise ValueError("unsupported ALIKE aggregation method")
    if not np.all(valid):
        raise ValueError("ALIKE observation cache contains invalid descriptors")
    order = np.argsort(tracks, kind="stable")
    sorted_tracks = tracks[order]
    sorted_features = features[order]
    sorted_scores = np.nan_to_num(scores[order], nan=0.0, neginf=0.0, posinf=1.0)
    unique_tracks, starts, counts = np.unique(
        sorted_tracks,
        return_index=True,
        return_counts=True,
    )
    if str(method) == "normalized_mean":
        weights = np.ones((len(sorted_tracks),), dtype=np.float32)
    else:
        weights = np.clip(sorted_scores, 1e-3, 1.0).astype(np.float32)
    weighted_sums = np.add.reduceat(sorted_features * weights[:, None], starts, axis=0)
    weight_sums = np.add.reduceat(weights, starts)
    means, valid_means = normalize_rows(weighted_sums / np.maximum(weight_sums[:, None], 1e-8))
    if not np.all(valid_means):
        raise ValueError("ALIKE aggregation produced invalid track descriptors")
    repeated_means = np.repeat(means, counts, axis=0)
    cosine_residuals = 1.0 - np.sum(sorted_features * repeated_means, axis=1)
    variances = (
        np.add.reduceat(np.maximum(cosine_residuals, 0.0) * weights, starts)
        / np.maximum(weight_sums, 1e-8)
    ).astype(np.float32)
    return unique_tracks, means.astype(np.float32), counts.astype(np.int64), variances


def _descriptor_space_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf8")).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = Path(args.support_track_observations_jsonl)
    observations = load_colmap_track_observations_jsonl(observation_path)
    track_ids = np.asarray([observation.track_id for observation in observations], dtype=np.int64)
    feature_cache = Path(args.feature_cache)
    expected_cache = {
        "format": "full_support_alike_observation_features_v1",
        "support_track_observations_jsonl": str(observation_path),
        "support_track_observations_sha256": file_sha256_short(observation_path),
        "observation_count": int(len(observations)),
        "track_count": int(np.unique(track_ids).size),
        "alike_model_name": str(args.alike_model_name),
    }
    cache_hit = feature_cache.exists()
    if cache_hit:
        with np.load(feature_cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            mismatches = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in expected_cache.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"stale full ALIKE cache: {json.dumps(mismatches, sort_keys=True)}")
            cached_tracks = np.asarray(data["track_ids"], dtype=np.int64)
            descriptors = np.asarray(data["descriptors"], dtype=np.float32)
            detector_scores = np.asarray(data["detector_scores"], dtype=np.float32)
        if not np.array_equal(cached_tracks, track_ids):
            raise ValueError("full ALIKE cache observation rows differ from source JSONL")
    else:
        devices = tuple(value.strip() for value in str(args.devices).split(",") if value.strip())
        extracted = extract_real_image_observation_features(
            observations,
            image_root=Path(args.image_root),
            devices=devices,
            feature_type="alike",
            alike_model_name=str(args.alike_model_name),
            matcha_repo=Path(args.matcha_repo),
            image_batch_size=1,
        )
        descriptors = extracted.descriptors
        detector_scores = extracted.detector_scores
        metadata = {**expected_cache, **dict(extracted.metadata), "cache_dtype": str(args.cache_dtype)}
        cache_dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            feature_cache,
            track_ids=track_ids,
            descriptors=np.asarray(descriptors, dtype=cache_dtype),
            detector_scores=np.asarray(detector_scores, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )

    source_index, source_metadata = load_landmark_index_npz(Path(args.source_landmark_bank))
    if np.unique(source_index.track_ids).size != len(source_index):
        raise ValueError("source landmark bank must contain exactly one row per physical track")
    source_order = np.argsort(source_index.track_ids, kind="stable")
    source_tracks = source_index.track_ids[source_order]
    observed_tracks = np.unique(track_ids)
    missing_source_tracks = np.setdiff1d(source_tracks, observed_tracks)
    if missing_source_tracks.size:
        raise ValueError(f"ALIKE observation cache is missing {len(missing_source_tracks)} source tracks")
    source_observation_mask = np.isin(track_ids, source_tracks)
    excluded_track_ids, excluded_track_counts = np.unique(
        track_ids[~source_observation_mask],
        return_counts=True,
    )
    if excluded_track_ids.size and np.any(excluded_track_counts != 1):
        raise ValueError("non-source ALIKE tracks are not all singleton observations")
    banks = {}
    for method in ("normalized_mean", "detector_weighted_mean"):
        unique_tracks, features, counts, variances = aggregate_alike_track_descriptors(
            track_ids[source_observation_mask],
            descriptors[source_observation_mask],
            detector_scores[source_observation_mask],
            method=method,
        )
        if not np.array_equal(unique_tracks, source_tracks):
            raise ValueError("filtered ALIKE observations do not match the source track universe")
        manifest = {
            "projection_source": "real_image_alike_observation_full_map",
            "alike_model_name": str(args.alike_model_name),
            "alike_checkpoint_sha256": str(metadata.get("model_checkpoint_sha256", "")),
            "support_track_observations_sha256": file_sha256_short(observation_path),
            "source_image_manifest_sha256": str(metadata.get("source_image_manifest_sha256", "")),
            "aggregation_method": str(method),
            "descriptor_dimension": int(features.shape[1]),
            "normalization_mode": "observation_l2_then_track_l2",
            "coordinate_source": "sfm_observation_xy",
        }
        manifest["descriptor_space_id"] = _descriptor_space_id(manifest)
        index = LandmarkMapIndex(
            track_ids=unique_tracks,
            xyz=source_index.xyz[source_order],
            features=features,
            mean_variances=variances,
            observation_counts=counts,
            observation_image_ids=tuple(() for _ in unique_tracks),
            reprojection_errors=source_index.reprojection_errors[source_order],
            feature_ambiguities=source_index.feature_ambiguities[source_order],
            prototype_ids=np.zeros_like(unique_tracks),
        )
        output_path = output_dir / f"alike_observations_{method}.npz"
        bank_metadata = {
            "stage": "alike_projected_observation_landmark_bank",
            "descriptor_space_id": manifest["descriptor_space_id"],
            "descriptor_space_manifest": manifest,
            "feature_cache": str(feature_cache),
            "feature_cache_sha256": file_sha256_short(feature_cache),
            "source_landmark_bank": str(args.source_landmark_bank),
            "source_landmark_bank_sha256": file_sha256_short(Path(args.source_landmark_bank)),
            "source_descriptor_space_id": source_metadata.get("descriptor_space_id"),
            "feature_dim": int(index.feature_dim),
            "landmark_count": int(len(index)),
            "observation_image_ids": "external_support_track_observations_jsonl",
        }
        save_landmark_index_npz(index, output_path, metadata=bank_metadata)
        banks[method] = {
            "path": str(output_path),
            "sha256": file_sha256_short(output_path),
            "descriptor_space_id": manifest["descriptor_space_id"],
            "mean_track_variance": float(np.mean(variances)),
        }
    summary = {
        "stage": "alike_projected_observation_landmark_bank_sweep",
        "feature_cache": {
            "path": str(feature_cache),
            "sha256": file_sha256_short(feature_cache),
            "cache_hit": bool(cache_hit),
            "metadata": metadata,
        },
        "banks": banks,
        "track_universe": {
            "source_track_count": int(len(source_tracks)),
            "excluded_singleton_track_count": int(len(excluded_track_ids)),
            "excluded_singleton_observation_count": int(np.sum(excluded_track_counts)),
        },
        "runtime_seconds": float(time.time() - start_time),
        "outputs": {"summary": str(output_dir / "summary.json")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
