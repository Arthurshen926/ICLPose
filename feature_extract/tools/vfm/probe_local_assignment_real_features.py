"""Probe real-image local features on a fixed top-L landmark candidate artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_local_assignment_probe_pose import _validate_probe_binding
from feature_extract.tools.vfm.probe_local_assignment_support_views import (
    _evaluate_pose_strategy,
    _resolve_devices,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
    score_candidate_support_feature_pooling,
    summarize_assignment_strategy,
)
from feature_extract.vfm.localization.real_image_observation_features import (
    extract_real_image_observation_features,
)
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_probe_arrays", required=True)
    parser.add_argument("--query_track_observations_jsonl", required=True)
    parser.add_argument("--support_track_observations_jsonl", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_type", default="alike", choices=("alike", "radio_intermediate"))
    parser.add_argument("--alike_model_name", default="alike-t")
    parser.add_argument("--matcha_repo", default="/root/matcha")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--radio_intermediate_index", type=int, default=-6)
    parser.add_argument("--score_prefix", default="")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--image_batch_size", type=int, default=1)
    parser.add_argument("--colmap_model_dir", default="")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument(
        "--pose_strategies",
        default=(
            "coarse_prototype,all_support_top2_mean,alike_support_top2_mean,alike_support_top4_mean,"
            "radiofinal_top2_alike_top2_50_50,coarse_alike_top2_50_50,proposal_oracle"
        ),
    )
    return parser.parse_args(argv)


def _load_base_probe(path: Path):
    with np.load(Path(path), allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    candidates = UniqueTrackCandidateSet(
        bank_row_indices=np.asarray(payload["bank_row_indices"], dtype=np.int64),
        track_ids=np.asarray(payload["candidate_track_ids"], dtype=np.int64),
        prototype_ids=np.asarray(payload["candidate_prototype_ids"], dtype=np.int64),
        coarse_scores=np.asarray(payload["coarse_scores"], dtype=np.float32),
    )
    return payload, candidates


def _resolve_query_observations(
    *,
    query_ids: Sequence[str],
    point2d_indices: np.ndarray,
    correct_track_ids: np.ndarray,
    path: Path,
) -> list[ColmapTrackObservation]:
    observations = load_colmap_track_observations_jsonl(Path(path), image_ids=set(query_ids))
    lookup = {
        (str(observation.image_id), int(observation.point2d_idx), int(observation.track_id)): observation
        for observation in observations
    }
    selected: list[ColmapTrackObservation] = []
    missing: list[tuple[str, int, int]] = []
    for query_id, point2d_index, track_id in zip(query_ids, point2d_indices, correct_track_ids):
        key = (str(query_id), int(point2d_index), int(track_id))
        observation = lookup.get(key)
        if observation is None:
            missing.append(key)
        else:
            selected.append(observation)
    if missing:
        raise ValueError(f"failed to resolve {len(missing)} probe query observations; examples: {missing[:5]}")
    return selected


def _blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ValueError("score matrices must have matching shapes")
    output = np.full(lhs.shape, -np.inf, dtype=np.float32)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    output[valid] = float(left_weight) * lhs[valid] + (1.0 - float(left_weight)) * rhs[valid]
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_probe_path = Path(args.base_probe_arrays)
    payload, candidates = _load_base_probe(base_probe_path)
    query_ids = tuple(str(item) for item in payload["query_ids"].tolist())
    point2d_indices = np.asarray(payload["query_point2d_indices"], dtype=np.int64)
    correct_track_ids = np.asarray(payload["correct_track_ids"], dtype=np.int64)
    query_observations = _resolve_query_observations(
        query_ids=query_ids,
        point2d_indices=point2d_indices,
        correct_track_ids=correct_track_ids,
        path=Path(args.query_track_observations_jsonl),
    )
    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    binding = _validate_probe_binding(
        probe_path=base_probe_path,
        landmark_metadata=landmark_metadata,
        candidates=candidates,
        landmark_index=landmark_index,
    )

    candidate_track_ids = {
        int(track_id) for track_id in candidates.track_ids[candidates.valid_mask].tolist()
    }
    support_observations = load_colmap_track_observations_jsonl(
        Path(args.support_track_observations_jsonl),
        track_ids=candidate_track_ids,
    )
    all_observations = [*query_observations, *support_observations]
    devices = _resolve_devices(str(args.devices))
    extracted = extract_real_image_observation_features(
        all_observations,
        image_root=Path(args.image_root),
        devices=devices,
        feature_type=str(args.feature_type),
        alike_model_name=str(args.alike_model_name),
        matcha_repo=Path(args.matcha_repo),
        radio_version=str(args.radio_version),
        radio_repo=Path(args.radio_repo),
        radio_intermediate_index=int(args.radio_intermediate_index),
        image_batch_size=int(args.image_batch_size),
    )
    query_count = len(query_observations)
    query_features = extracted.descriptors[:query_count]
    support_features = extracted.descriptors[query_count:]
    query_detector_scores = extracted.detector_scores[:query_count]
    support_detector_scores = extracted.detector_scores[query_count:]
    support_track_ids = np.asarray([observation.track_id for observation in support_observations], dtype=np.int64)

    score_prefix = str(args.score_prefix).strip()
    if not score_prefix:
        score_prefix = (
            "alike"
            if str(args.feature_type) == "alike"
            else f"radio_intermediate_{int(args.radio_intermediate_index)}".replace("-", "m")
        )
    local_scores = score_candidate_support_feature_pooling(
        query_features=query_features,
        candidates=candidates,
        support_track_ids=support_track_ids,
        support_features=support_features,
        prefix=score_prefix,
    )
    base_scores = {
        key[len("strategy__") :]: np.asarray(value, dtype=np.float32)
        for key, value in payload.items()
        if key.startswith("strategy__")
    }
    coarse = base_scores.get("coarse_prototype", candidates.coarse_scores)
    radio_top2 = base_scores.get("all_support_top2_mean")
    local_top2 = local_scores[f"{score_prefix}_support_top2_mean"]
    if radio_top2 is None:
        raise ValueError("base probe must contain all_support_top2_mean")
    fusion_scores = {
        f"coarse_{score_prefix}_top2_75_25": _blend(coarse, local_top2, 0.75),
        f"coarse_{score_prefix}_top2_50_50": _blend(coarse, local_top2, 0.50),
        f"coarse_{score_prefix}_top2_25_75": _blend(coarse, local_top2, 0.25),
        f"radiofinal_top2_{score_prefix}_top2_75_25": _blend(radio_top2, local_top2, 0.75),
        f"radiofinal_top2_{score_prefix}_top2_50_50": _blend(radio_top2, local_top2, 0.50),
        f"radiofinal_top2_{score_prefix}_top2_25_75": _blend(radio_top2, local_top2, 0.25),
    }
    new_scores = {**local_scores, **fusion_scores}
    assignment = {
        name: summarize_assignment_strategy(
            candidates=candidates,
            correct_track_ids=correct_track_ids,
            query_ids=query_ids,
            scores=scores,
        )
        for name, scores in new_scores.items()
    }

    image_vocab = tuple(sorted({str(observation.image_id) for observation in support_observations}))
    image_position = {image_id: index for index, image_id in enumerate(image_vocab)}
    cache_dtype = np.float16 if str(args.cache_dtype) == "float16" else np.float32
    feature_cache_path = output_dir / "real_observation_features.npz"
    feature_cache_metadata = {
        **extracted.metadata,
        "query_observation_count": int(query_count),
        "support_observation_count": int(len(support_observations)),
        "base_probe_arrays": str(base_probe_path),
        "base_probe_arrays_sha256": file_sha256_short(base_probe_path),
        "candidate_track_count": int(len(candidate_track_ids)),
        "cache_dtype": str(args.cache_dtype),
    }
    np.savez(
        feature_cache_path,
        query_features=np.asarray(query_features, dtype=cache_dtype),
        query_detector_scores=np.asarray(query_detector_scores, dtype=np.float32),
        support_features=np.asarray(support_features, dtype=cache_dtype),
        support_detector_scores=np.asarray(support_detector_scores, dtype=np.float32),
        support_track_ids=support_track_ids,
        support_image_ids=np.asarray(image_vocab, dtype=np.str_),
        support_image_indices=np.asarray(
            [image_position[str(observation.image_id)] for observation in support_observations],
            dtype=np.int32,
        ),
        support_point2d_indices=np.asarray(
            [int(observation.point2d_idx) for observation in support_observations],
            dtype=np.int64,
        ),
        metadata_json=np.asarray(json.dumps(feature_cache_metadata, sort_keys=True), dtype=np.str_),
    )

    extended_payload = dict(payload)
    for name, scores in new_scores.items():
        extended_payload[f"strategy__{name}"] = np.asarray(scores, dtype=np.float32)
    extended_probe_path = output_dir / "assignment_probe_arrays_with_real_features.npz"
    np.savez(extended_probe_path, **extended_payload)

    pose_summaries: dict[str, object] = {}
    pose_rows: dict[str, object] = {}
    if str(args.colmap_model_dir):
        model_dir = Path(args.colmap_model_dir)
        cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
        images = read_colmap_images_binary(model_dir / "images.bin")
        images_by_name = {str(image.image_name): image for image in images.values()}
        all_scores = {**base_scores, **new_scores}
        pose_strategies = tuple(
            item.strip() for item in str(args.pose_strategies).split(",") if item.strip()
        )
        for strategy in pose_strategies:
            if strategy != "proposal_oracle" and strategy not in all_scores:
                raise ValueError(f"unknown pose strategy: {strategy}")
            pose_summary, rows = _evaluate_pose_strategy(
                strategy=strategy,
                scores=None if strategy == "proposal_oracle" else all_scores[strategy],
                candidates=candidates,
                query_observations=query_observations,
                query_ids=query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )
            pose_summaries[strategy] = pose_summary
            pose_rows[strategy] = rows
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")

    base_summary = json.loads((base_probe_path.parent / "summary.json").read_text())
    summary = {
        "stage": "s4_l0_real_image_local_feature_probe",
        "binding": binding,
        "descriptor_space": dict(base_summary.get("descriptor_space", {})),
        "feature_source": feature_cache_metadata,
        "score_prefix": score_prefix,
        "assignment": assignment,
        "pose": pose_summaries,
        "reference_metrics": {
            name: base_summary.get("assignment", {}).get(name)
            for name in ("coarse_prototype", "all_support_top2_mean", "all_support_top4_mean")
        },
        "limitations": [
            "query nodes remain held-out GT SfM observations for this L0 feature probe",
            "ALIKE is a fixed pretrained dense descriptor and has not been specialized to landmark assignment",
            "score fusion weights are fixed diagnostics, not calibrated probabilities",
            "all source pixels come from real query/support images; no render is used",
        ],
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "feature_cache": str(feature_cache_path),
            "feature_cache_sha256": file_sha256_short(feature_cache_path),
            "extended_probe_arrays": str(extended_probe_path),
            "pose_rows": str(pose_rows_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
