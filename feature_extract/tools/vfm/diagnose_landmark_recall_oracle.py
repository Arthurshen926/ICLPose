"""Diagnose whether GT SfM landmarks are retrievable from a projected landmark bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.eval_real_radio_landmark_hybrid import (
    projected_cache_expected_metadata,
    validate_projected_cache_metadata,
)
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
    project_landmark_features_with_joint_model,
    project_landmark_index_features,
)
from feature_extract.vfm.localization.landmark_recall_oracle import (
    LandmarkRecallRecord,
    rank_correct_landmark,
    summarize_landmark_recall_records,
)
from feature_extract.vfm.localization.pipeline import _load_feature_map
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import _sample_feature_vector, load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--track_observations_jsonl", required=True)
    parser.add_argument("--projected_landmark_cache", default="")
    parser.add_argument("--landmark_bank", default="")
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument(
        "--projection_preset",
        default="joint_query_to_projected_observation_landmark",
        choices=(
            "raw_query_to_raw_landmark",
            "joint_query_to_projected_observation_landmark",
            "post_aggregate_1x1_projection_baseline",
        ),
    )
    parser.add_argument("--allow_diagnostic_projection", action="store_true")
    parser.add_argument("--projection_batch_size", type=int, default=8192)
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--top_ks", default="1,5,10,20,50,100")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_observations_per_query", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _parse_top_ks(value: str) -> tuple[int, ...]:
    top_ks = tuple(sorted({int(item) for item in str(value).split(",") if item.strip()}))
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("--top_ks must contain positive comma-separated integers")
    return top_ks


def _resolve_device(device: str) -> str:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def _track_stats(observations) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    errors_by_track: dict[int, list[float]] = {}
    for obs in observations:
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        errors_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    return (
        xyz_by_track,
        {int(track_id): float(np.mean(values)) for track_id, values in errors_by_track.items() if values},
    )


def _write_csv(path: Path, records: Sequence[LandmarkRecallRecord]) -> None:
    rows = [record.to_dict() for record in records]
    fieldnames = [
        "query_id",
        "correct_track_id",
        "correct_rank",
        "correct_score",
        "rank1_track_id",
        "rank1_score",
        "score_gap_to_rank1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, records: Sequence[LandmarkRecallRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _sample_observation_descriptor(mapped_feature_map: np.ndarray, observation, *, sample_mode: str) -> np.ndarray:
    return _sample_feature_vector(
        mapped_feature_map,
        np.asarray(observation.xy, dtype=np.float32),
        image_width=int(observation.image_width),
        image_height=int(observation.image_height),
        sample_mode=str(sample_mode),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if str(args.projection_preset) == "post_aggregate_1x1_projection_baseline" and not bool(
        args.allow_diagnostic_projection
    ):
        raise ValueError(
            "projection_preset=post_aggregate_1x1_projection_baseline is diagnostic-only; "
            "rerun with --allow_diagnostic_projection only for ablation/debugging"
        )
    device = _resolve_device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = _parse_top_ks(str(args.top_ks))

    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    observations = load_colmap_track_observations_jsonl(Path(args.track_observations_jsonl))
    observations_by_query: dict[str, list[object]] = {}
    for observation in observations:
        observations_by_query.setdefault(str(observation.image_id), []).append(observation)

    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device)
    mapper = None if str(args.projection_preset) == "raw_query_to_raw_landmark" else JointFeatureMapper(joint_run.model, device=device)
    cache_metadata: dict[str, object] = {}
    if str(args.projected_landmark_cache):
        landmark_index, cache_metadata = load_landmark_index_npz(Path(args.projected_landmark_cache))
        expected_modes = {
            "raw_query_to_raw_landmark": "raw_landmark_bank",
            "joint_query_to_projected_observation_landmark": "full_map_projected_observations",
            "post_aggregate_1x1_projection_baseline": "post_aggregate_1x1_projection_baseline",
        }
        validate_projected_cache_metadata(
            cache_metadata,
            projected_cache_expected_metadata(
                projection_mode=expected_modes[str(args.projection_preset)],
                feature_key=str(args.feature_key),
                matcha_joint_checkpoint=Path(args.matcha_joint_checkpoint),
                track_observations=Path(args.track_observations_jsonl),
                feature_dim=int(landmark_index.feature_dim),
            ),
        )
    else:
        if not str(args.landmark_bank):
            raise ValueError("--landmark_bank is required when --projected_landmark_cache is not provided")
        xyz_by_track, reprojection_error_by_track = _track_stats(observations)
        raw_bank = load_selected_track_bank_npz(Path(args.landmark_bank))
        landmark_index = LandmarkMapIndex.from_track_bank(raw_bank, xyz_by_track, reprojection_error_by_track)
        if str(args.projection_preset) == "post_aggregate_1x1_projection_baseline":
            landmark_index = project_landmark_index_features(
                landmark_index,
                lambda values: project_landmark_features_with_joint_model(
                    joint_run.model,
                    values,
                    device=device,
                    batch_size=int(args.projection_batch_size),
                ),
            )
    normalized_landmark_features, valid_landmark_mask = normalize_rows(landmark_index.features)

    recall_records: list[LandmarkRecallRecord] = []
    query_count_with_observations = 0
    for record in records:
        query_id = str(record.image_id)
        query_observations = list(observations_by_query.get(query_id, []))
        if not query_observations:
            continue
        query_count_with_observations += 1
        if int(args.max_observations_per_query) > 0:
            query_observations = query_observations[: int(args.max_observations_per_query)]
        feature_map = _load_feature_map(Path(record.token_path), key=str(args.feature_key))
        mapped = np.asarray(feature_map, dtype=np.float32) if mapper is None else mapper.project(feature_map).coarse_descriptors
        for observation in query_observations:
            descriptor = _sample_observation_descriptor(mapped, observation, sample_mode=str(args.sample_mode))
            recall_records.append(
                rank_correct_landmark(
                    query_descriptor=descriptor,
                    landmark_features=normalized_landmark_features,
                    landmark_track_ids=landmark_index.track_ids,
                    correct_track_id=int(observation.track_id),
                    query_id=query_id,
                    landmark_features_are_normalized=True,
                    valid_landmark_mask=valid_landmark_mask,
                )
            )

    rows_csv = output_dir / "landmark_recall_oracle_rows.csv"
    rows_jsonl = output_dir / "landmark_recall_oracle_rows.jsonl"
    _write_csv(rows_csv, recall_records)
    _write_jsonl(rows_jsonl, recall_records)
    summary = {
        "stage": "landmark_recall_oracle",
        "query_manifest": str(args.query_manifest),
        "track_observations_jsonl": str(args.track_observations_jsonl),
        "projected_landmark_cache": str(args.projected_landmark_cache),
        "landmark_bank": str(args.landmark_bank),
        "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
        "projection_preset": str(args.projection_preset),
        "feature_key": str(args.feature_key),
        "sample_mode": str(args.sample_mode),
        "query_count": int(len(records)),
        "query_count_with_observations": int(query_count_with_observations),
        "landmark_count": int(len(landmark_index)),
        "projected_landmark_cache_metadata": cache_metadata,
        "metrics": summarize_landmark_recall_records(recall_records, top_ks=top_ks),
        "outputs": {
            "rows_csv": str(rows_csv),
            "rows_jsonl": str(rows_jsonl),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
