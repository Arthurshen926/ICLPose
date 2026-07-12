"""Probe support-view and maplet-context reranking on held-out landmark proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.eval_real_radio_landmark_hybrid import (
    projected_cache_expected_metadata,
    validate_projected_cache_metadata,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    ColmapTrackObservation,
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.local_maplet_matching import load_local_maplet_support_index_npz
from feature_extract.vfm.localization.descriptor_space import token_feature_source_config
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
    sample_projected_track_observations,
)
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
    projected_support_descriptors_by_track,
    proposal_recall_summary,
    retrieve_unique_track_candidates_exact,
    score_support_assignment_strategies,
    summarize_assignment_strategy,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    resolve_pose_match_conflicts,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.localization.pipeline import _load_feature_map
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
)
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.track_feature_sampling import (
    _sample_feature_vectors,
    load_colmap_track_observations_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_track_observations_jsonl", required=True)
    parser.add_argument("--support_manifest", required=True)
    parser.add_argument("--support_track_observations_jsonl", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--top_l", type=int, default=20)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_observations_per_query", type=int, default=20)
    parser.add_argument("--observation_sampling", default="uniform", choices=("uniform", "first"))
    parser.add_argument("--sample_mode", default="bilinear", choices=("nearest", "bilinear"))
    parser.add_argument("--retrieval_device", default="cuda:0")
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--projection_devices", default="cuda:0,cuda:1")
    parser.add_argument("--projection_image_batch_size", type=int, default=8)
    parser.add_argument("--projection_load_workers", type=int, default=2)
    parser.add_argument("--colmap_model_dir", default="")
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=5000)
    parser.add_argument(
        "--pose_strategies",
        default=(
            "coarse_prototype,all_support_best,all_support_logmeanexp_tau0p05,"
            "coarse_all_support_best_75_25,coarse_all_support_best_50_50,"
            "coarse_all_support_logmeanexp_50_50,maplet_support_best,"
            "coarse_maplet_context_75_25,oracle_view_angle_best,proposal_oracle"
        ),
    )
    return parser.parse_args(argv)


def _resolve_devices(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not requested:
        requested = ("cpu",)
    resolved: list[str] = []
    for item in requested:
        device = torch.device(item)
        if device.type == "cuda":
            if not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count():
                continue
        if str(device) not in resolved:
            resolved.append(str(device))
    return tuple(resolved or ["cpu"])


def _select_query_observations(
    records,
    observations_by_image: dict[str, list[ColmapTrackObservation]],
    *,
    max_observations_per_query: int,
    sampling: str,
) -> tuple[list[ColmapTrackObservation], list[str]]:
    selected: list[ColmapTrackObservation] = []
    query_ids: list[str] = []
    for record in records:
        query_id = str(record.image_id)
        observations = sorted(
            observations_by_image.get(query_id, ()),
            key=lambda item: (int(item.point2d_idx), int(item.track_id)),
        )
        limit = int(max_observations_per_query)
        if limit > 0 and len(observations) > limit:
            if str(sampling) == "uniform":
                positions = np.linspace(0, len(observations) - 1, num=limit, dtype=np.int64)
                observations = [observations[int(position)] for position in positions.tolist()]
            else:
                observations = observations[:limit]
        selected.extend(observations)
        query_ids.extend([query_id] * len(observations))
    return selected, query_ids


def _local_context_descriptors(
    mapped_feature_map: np.ndarray,
    observations: Sequence[ColmapTrackObservation],
    *,
    radius: int = 1,
) -> np.ndarray:
    feature_map = np.asarray(mapped_feature_map, dtype=np.float32)
    channels, token_height, token_width = feature_map.shape
    output = np.zeros((len(observations), channels), dtype=np.float32)
    for row, observation in enumerate(observations):
        if observation.image_width is None or observation.image_height is None:
            continue
        x = float(observation.xy[0]) / max(float(observation.image_width - 1), 1.0) * max(token_width - 1, 0)
        y = float(observation.xy[1]) / max(float(observation.image_height - 1), 1.0) * max(token_height - 1, 0)
        x_index = int(np.rint(np.clip(x, 0.0, max(token_width - 1, 0))))
        y_index = int(np.rint(np.clip(y, 0.0, max(token_height - 1, 0))))
        x0 = max(0, x_index - int(radius))
        x1 = min(token_width, x_index + int(radius) + 1)
        y0 = max(0, y_index - int(radius))
        y1 = min(token_height, y_index + int(radius) + 1)
        patch = feature_map[:, y0:y1, x0:x1].reshape(channels, -1).T
        norms = np.linalg.norm(patch, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        if not np.any(valid):
            continue
        mean = np.mean(patch[valid] / np.maximum(norms[valid], 1e-8), axis=0)
        output[row] = mean / max(float(np.linalg.norm(mean)), 1e-8)
    return output


def _project_query_observations(
    records,
    observations_by_image: dict[str, list[ColmapTrackObservation]],
    mapper: JointFeatureMapper,
    *,
    feature_key: str,
    sample_mode: str,
    max_observations_per_query: int,
    sampling: str,
) -> tuple[list[ColmapTrackObservation], list[str], np.ndarray, np.ndarray, np.ndarray]:
    selected, query_ids = _select_query_observations(
        records,
        observations_by_image,
        max_observations_per_query=int(max_observations_per_query),
        sampling=str(sampling),
    )
    selected_by_image: dict[str, list[tuple[int, ColmapTrackObservation]]] = {}
    for row, (query_id, observation) in enumerate(zip(query_ids, selected)):
        selected_by_image.setdefault(str(query_id), []).append((int(row), observation))
    descriptors: np.ndarray | None = None
    contexts: np.ndarray | None = None
    for record in records:
        image_rows = selected_by_image.get(str(record.image_id), ())
        if not image_rows:
            continue
        raw_map = _load_feature_map(Path(record.token_path), key=str(feature_key))
        mapped = mapper.project(raw_map).coarse_descriptors
        observations = [item[1] for item in image_rows]
        sampled = _sample_feature_vectors(
            mapped,
            np.asarray([observation.xy for observation in observations], dtype=np.float64),
            np.asarray([observation.image_width for observation in observations], dtype=np.int64),
            np.asarray([observation.image_height for observation in observations], dtype=np.int64),
            str(sample_mode),
        )
        local_context = _local_context_descriptors(mapped, observations)
        if descriptors is None:
            descriptors = np.zeros((len(selected), int(mapped.shape[0])), dtype=np.float32)
            contexts = np.zeros_like(descriptors)
        for local_row, (global_row, _observation) in enumerate(image_rows):
            descriptors[int(global_row)] = sampled[local_row]
            contexts[int(global_row)] = local_context[local_row]
    if descriptors is None or contexts is None:
        raise ValueError("no held-out query observations could be projected")
    viewing_rays = np.stack(
        [
            np.zeros((3,), dtype=np.float64)
            if observation.viewing_ray is None
            else np.asarray(observation.viewing_ray, dtype=np.float64).reshape(3)
            for observation in selected
        ],
        axis=0,
    )
    return selected, query_ids, descriptors, contexts, viewing_rays


def _validate_maplet_compatibility(
    maplet_index,
    maplet_metadata: dict[str, object],
    candidate_index,
    candidate_metadata: dict[str, object],
) -> dict[str, object]:
    source_path_value = str(maplet_metadata.get("source_landmark_index", ""))
    if not source_path_value:
        raise ValueError("maplet support index is missing source_landmark_index")
    source_path = Path(source_path_value)
    if not source_path.exists():
        raise ValueError(f"maplet source landmark bank does not exist: {source_path}")
    expected_hash = str(maplet_metadata.get("source_landmark_index_sha256", ""))
    actual_hash = file_sha256_short(source_path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            f"maplet source bank hash mismatch: expected {expected_hash!r}, got {actual_hash!r}"
        )
    source_index, source_metadata = load_landmark_index_npz(source_path)
    if str(maplet_metadata.get("source_descriptor_space_id", "")) != str(
        source_metadata.get("descriptor_space_id", "")
    ):
        raise ValueError("maplet source descriptor_space_id does not match its source bank")
    source_manifest = source_metadata.get("descriptor_space_manifest")
    candidate_manifest = candidate_metadata.get("descriptor_space_manifest")
    if not isinstance(source_manifest, dict) or not isinstance(candidate_manifest, dict):
        raise ValueError("landmark banks must contain descriptor_space_manifest")
    source_projection = str(source_manifest.get("projection_space_id", ""))
    candidate_projection = str(candidate_manifest.get("projection_space_id", ""))
    if not source_projection or source_projection != candidate_projection:
        raise ValueError(
            "maplet/candidate projection spaces differ: "
            f"{source_projection!r} vs {candidate_projection!r}"
        )
    if not np.array_equal(maplet_index.anchor_track_ids, source_index.track_ids):
        raise ValueError("maplet anchor rows are not aligned with the recorded source bank")
    if int(maplet_index.maplets.context_features.shape[1]) != int(candidate_index.feature_dim):
        raise ValueError("maplet context and candidate descriptor dimensions differ")
    candidate_unique_tracks = np.unique(candidate_index.track_ids)
    if not np.array_equal(candidate_unique_tracks, maplet_index.anchor_track_ids):
        raise ValueError("candidate and maplet banks do not contain the same physical tracks")
    return {
        "source_landmark_index": str(source_path),
        "source_landmark_index_sha256": actual_hash,
        "source_descriptor_space_id": str(source_metadata.get("descriptor_space_id", "")),
        "projection_space_id": source_projection,
        "anchor_track_count": int(len(maplet_index.anchor_track_ids)),
    }


def _project_support_observations_multi_device(
    observations: Sequence[ColmapTrackObservation],
    manifest: TokenBankManifest,
    checkpoint: Path,
    primary_mapper: JointFeatureMapper,
    *,
    devices: Sequence[str],
    feature_key: str,
    sample_mode: str,
    image_batch_size: int,
    load_workers: int,
) -> tuple[list, dict[str, object]]:
    device_values = tuple(str(item) for item in devices)
    image_ids = sorted({str(observation.image_id) for observation in observations})
    device_by_image = {image_id: index % len(device_values) for index, image_id in enumerate(image_ids)}
    partitions = [
        [observation for observation in observations if device_by_image[str(observation.image_id)] == index]
        for index in range(len(device_values))
    ]

    def worker(index: int):
        device = device_values[index]
        mapper = primary_mapper
        if index > 0:
            run = load_matcha_joint_model(Path(checkpoint), device=device)
            mapper = JointFeatureMapper(run.model, device=device)
        return sample_projected_track_observations(
            partitions[index],
            manifest,
            mapper,
            feature_key=str(feature_key),
            missing="error",
            sample_mode=str(sample_mode),
            projection_image_batch_size=int(image_batch_size),
            projection_load_workers=int(load_workers),
        )

    if len(device_values) == 1:
        results = [worker(0)]
    else:
        with ThreadPoolExecutor(max_workers=len(device_values)) as executor:
            results = list(executor.map(worker, range(len(device_values))))
    sampled = [row for partition, _metadata in results for row in partition]
    metadata = {
        "devices": list(device_values),
        "partition_image_counts": [
            int(metadata.get("projected_image_count", 0)) for _partition, metadata in results
        ],
        "input_observation_count": int(sum(int(metadata.get("input_observation_count", 0)) for _rows, metadata in results)),
        "effective_observation_count": int(
            sum(int(metadata.get("effective_observation_count", 0)) for _rows, metadata in results)
        ),
        "sampled_observation_count": int(len(sampled)),
        "projected_image_count": int(sum(int(metadata.get("projected_image_count", 0)) for _rows, metadata in results)),
        "missing_image_count": int(sum(int(metadata.get("missing_image_count", 0)) for _rows, metadata in results)),
    }
    return sampled, metadata


def _top_candidate_column(candidates: UniqueTrackCandidateSet, scores: np.ndarray, row: int) -> int | None:
    valid = np.flatnonzero(candidates.valid_mask[int(row)] & np.isfinite(scores[int(row)]))
    if valid.size == 0:
        return None
    order = np.argsort(-scores[int(row), valid], kind="mergesort")
    return int(valid[int(order[0])])


def _evaluate_pose_strategy(
    *,
    strategy: str,
    scores: np.ndarray | None,
    selection_scores: np.ndarray | None = None,
    candidates: UniqueTrackCandidateSet,
    query_observations: Sequence[ColmapTrackObservation],
    query_ids: Sequence[str],
    landmark_index,
    cameras,
    images_by_name,
    reprojection_error_px: float,
    iterations: int,
    max_matches: int | None = None,
    pose_selection_mode: str = "score_topk",
    min_matches: int | None = None,
    min_selection_score: float | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assignment_scores = None if scores is None else np.asarray(scores, dtype=np.float32)
    confidence_scores = (
        assignment_scores
        if selection_scores is None
        else np.asarray(selection_scores, dtype=np.float32)
    )
    expected_shape = candidates.track_ids.shape
    if assignment_scores is not None and assignment_scores.shape != expected_shape:
        raise ValueError("assignment scores do not align with candidate tracks")
    if confidence_scores is not None and confidence_scores.shape != expected_shape:
        raise ValueError("selection scores do not align with candidate tracks")
    if max_matches is None and (min_matches is not None or min_selection_score is not None):
        raise ValueError("adaptive pose selection requires max_matches")
    rows_by_image: dict[str, list[int]] = {}
    for row, query_id in enumerate(query_ids):
        rows_by_image.setdefault(str(query_id), []).append(int(row))
    output_rows: list[dict[str, object]] = []
    for query_id, query_rows in rows_by_image.items():
        image = images_by_name.get(query_id)
        if image is None:
            output_rows.append({"query_id": query_id, "success": False, "reason": "missing_colmap_image"})
            continue
        matches: list[QueryTo3DMatch] = []
        for query_row in query_rows:
            observation = query_observations[query_row]
            if str(strategy) == "proposal_oracle":
                correct_columns = np.flatnonzero(
                    candidates.valid_mask[query_row]
                    & (candidates.track_ids[query_row] == int(observation.track_id))
                )
                if correct_columns.size == 0:
                    continue
                column = int(correct_columns[0])
                score = 1.0
            else:
                if assignment_scores is None:
                    raise ValueError(f"scores are required for strategy {strategy}")
                selected = _top_candidate_column(candidates, assignment_scores, query_row)
                if selected is None:
                    continue
                column = int(selected)
                if confidence_scores is None:
                    raise ValueError(f"selection scores are required for strategy {strategy}")
                score = float(confidence_scores[query_row, column])
            bank_row = int(candidates.bank_row_indices[query_row, column])
            matches.append(
                QueryTo3DMatch(
                    token_index=int(observation.point2d_idx),
                    xy=np.asarray(observation.xy, dtype=np.float64),
                    track_id=int(candidates.track_ids[query_row, column]),
                    xyz=np.asarray(landmark_index.xyz[bank_row], dtype=np.float64),
                    similarity=float(score),
                    ratio=0.0,
                    landmark_variance=float(landmark_index.mean_variances[bank_row]),
                    source=f"assignment_probe:{strategy}",
                    prototype_id=int(candidates.prototype_ids[query_row, column]),
                )
            )
        camera = cameras[int(image.camera_id)]
        if max_matches is None:
            matches = resolve_pose_match_conflicts(matches)
        else:
            matches = select_pose_safe_matches(
                matches,
                max_matches=int(max_matches),
                image_width=int(camera.width),
                image_height=int(camera.height),
                mode=str(pose_selection_mode),
                min_matches=min_matches,
                min_confidence=min_selection_score,
            )
        finite_confidences = np.asarray(
            [float(match.similarity) for match in matches if np.isfinite(float(match.similarity))],
            dtype=np.float64,
        )
        # The current backend is uniform RANSAC, not PROSAC. Keep solver input
        # order independent of uncalibrated network scores after any explicit
        # score/coverage subset selection.
        matches = stable_uniform_ransac_order(matches)
        try:
            import cv2

            seed = int.from_bytes(hashlib.sha256(query_id.encode("utf8")).digest()[:4], "little")
            cv2.setRNGSeed(int(seed % (2**31 - 1)))
        except ImportError:
            pass
        result = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(reprojection_error_px),
            iterations=int(iterations),
            refine_method="LM",
        )
        gt_pose = np.eye(4, dtype=np.float64)
        gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
        gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
        error = pnp_pose_error(result.pose_w2c, gt_pose)
        finite_pose = bool(
            result.success
            and np.isfinite(error.translation_m)
            and np.isfinite(error.rotation_deg)
        )
        spatial_all = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            pose_w2c=result.pose_w2c,
        )
        spatial_inliers = match_spatial_distribution_stats(
            matches,
            int(camera.width),
            int(camera.height),
            result.inlier_mask,
            pose_w2c=result.pose_w2c,
        )
        residuals = pnp_reprojection_residual_stats(
            matches,
            result.pose_w2c,
            camera,
            inlier_mask=result.inlier_mask,
        )
        output_rows.append(
            {
                "query_id": query_id,
                "success": finite_pose,
                "solver_success": bool(result.success),
                "failure_reason": (
                    None
                    if finite_pose
                    else (
                        "non_finite_pose_error"
                        if result.success
                        else "pnp_solver_failure"
                    )
                ),
                "match_count": int(result.match_count),
                "inlier_count": int(result.inlier_count),
                "inlier_ratio": float(result.inlier_ratio),
                "all_grid_4x4_occupancy_frac": spatial_all.get(
                    "grid_4x4_occupancy_frac"
                ),
                "inlier_grid_4x4_occupancy_frac": spatial_inliers.get(
                    "grid_4x4_occupancy_frac"
                ),
                **residuals,
                "selection_confidence_min": (
                    None if finite_confidences.size == 0 else float(np.min(finite_confidences))
                ),
                "selection_confidence_median": (
                    None if finite_confidences.size == 0 else float(np.median(finite_confidences))
                ),
                "selection_confidence_max": (
                    None if finite_confidences.size == 0 else float(np.max(finite_confidences))
                ),
                "translation_m": None if not np.isfinite(error.translation_m) else float(error.translation_m),
                "rotation_deg": None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg),
                "pose_w2c": (
                    None
                    if result.pose_w2c is None
                    else np.asarray(result.pose_w2c, dtype=np.float64).reshape(4, 4).tolist()
                ),
            }
        )
    success = [row for row in output_rows if bool(row.get("success"))]
    translations = np.asarray([row["translation_m"] for row in success], dtype=np.float64)
    rotations = np.asarray([row["rotation_deg"] for row in success], dtype=np.float64)
    summary: dict[str, object] = {
        "query_count": int(len(output_rows)),
        "success_count": int(len(success)),
        "success_rate": 0.0 if not output_rows else float(len(success) / len(output_rows)),
        "median_translation_m_success": None if translations.size == 0 else float(np.median(translations)),
        "p90_translation_m_success": None if translations.size == 0 else float(np.percentile(translations, 90.0)),
        "median_rotation_deg_success": None if rotations.size == 0 else float(np.median(rotations)),
        "median_matches": (
            None
            if not output_rows
            else float(np.median([row.get("match_count", 0) for row in output_rows]))
        ),
        "median_inliers_success": None if not success else float(np.median([row["inlier_count"] for row in success])),
    }
    for distance, angle, name in ((0.25, 2.0, "25cm_2deg"), (0.10, 5.0, "10cm_5deg"), (0.05, 5.0, "5cm_5deg")):
        summary[f"recall_{name}"] = (
            0.0
            if not output_rows
            else float(
                np.mean(
                    [
                        bool(row.get("success"))
                        and row.get("translation_m") is not None
                        and row.get("rotation_deg") is not None
                        and float(row["translation_m"]) <= distance
                        and float(row["rotation_deg"]) <= angle
                        for row in output_rows
                    ]
                )
            )
        )
    return summary, output_rows


def _write_candidate_rows(
    path: Path,
    *,
    candidates: UniqueTrackCandidateSet,
    query_observations: Sequence[ColmapTrackObservation],
    query_ids: Sequence[str],
    strategy_scores: dict[str, np.ndarray],
    maplet_support_counts: np.ndarray,
    all_support_counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for query_row, (query_id, observation) in enumerate(zip(query_ids, query_observations)):
            for column in np.flatnonzero(candidates.valid_mask[query_row]).tolist():
                track_id = int(candidates.track_ids[query_row, column])
                row = {
                    "query_row": int(query_row),
                    "query_id": str(query_id),
                    "query_point2d_idx": int(observation.point2d_idx),
                    "query_x": float(observation.xy[0]),
                    "query_y": float(observation.xy[1]),
                    "correct_track_id": int(observation.track_id),
                    "candidate_track_id": track_id,
                    "is_correct_track": bool(track_id == int(observation.track_id)),
                    "coarse_rank": int(column + 1),
                    "prototype_id": int(candidates.prototype_ids[query_row, column]),
                    "coarse_score": float(candidates.coarse_scores[query_row, column]),
                    "maplet_support_count": int(maplet_support_counts[query_row, column]),
                    "all_support_count": int(all_support_counts[query_row, column]),
                    "strategy_scores": {
                        name: float(values[query_row, column])
                        for name, values in strategy_scores.items()
                        if np.isfinite(values[query_row, column])
                    },
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.top_l) <= 0:
        raise ValueError("--top_l must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    support_manifest = TokenBankManifest.from_json(Path(args.support_manifest))
    query_manifest.validate(verify_checksums=False)
    support_manifest.validate(verify_checksums=False)
    query_source = token_feature_source_config(query_manifest, str(args.feature_key))
    support_source = token_feature_source_config(support_manifest, str(args.feature_key))
    if query_source != support_source:
        raise ValueError("query and support token manifests use different descriptor source configurations")
    records = list(query_manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    query_image_ids = {str(record.image_id) for record in records}
    query_observations = load_colmap_track_observations_jsonl(
        Path(args.query_track_observations_jsonl),
        image_ids=query_image_ids,
    )
    observations_by_image: dict[str, list[ColmapTrackObservation]] = {}
    for observation in query_observations:
        observations_by_image.setdefault(str(observation.image_id), []).append(observation)

    landmark_index, landmark_metadata = load_landmark_index_npz(Path(args.projected_landmark_bank))
    validate_projected_cache_metadata(
        landmark_metadata,
        projected_cache_expected_metadata(
            projection_mode="full_map_projected_observations",
            feature_key=str(args.feature_key),
            matcha_joint_checkpoint=Path(args.matcha_joint_checkpoint),
            track_observations=Path(args.support_track_observations_jsonl),
            feature_dim=int(landmark_index.feature_dim),
            descriptor_source_config=query_source,
        ),
    )
    maplet_index, maplet_metadata = load_local_maplet_support_index_npz(Path(args.maplet_support_index))
    maplet_audit = _validate_maplet_compatibility(
        maplet_index,
        maplet_metadata,
        landmark_index,
        landmark_metadata,
    )

    projection_devices = _resolve_devices(str(args.projection_devices))
    run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=projection_devices[0])
    mapper = JointFeatureMapper(run.model, device=projection_devices[0])
    selected_observations, selected_query_ids, query_descriptors, query_context, query_rays = (
        _project_query_observations(
            records,
            observations_by_image,
            mapper,
            feature_key=str(args.feature_key),
            sample_mode=str(args.sample_mode),
            max_observations_per_query=int(args.max_observations_per_query),
            sampling=str(args.observation_sampling),
        )
    )
    candidates = retrieve_unique_track_candidates_exact(
        query_descriptors,
        landmark_index,
        top_l=int(args.top_l),
        device=str(args.retrieval_device),
        batch_size=int(args.retrieval_batch_size),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_track_ids = {int(track_id) for track_id in candidates.track_ids[candidates.valid_mask].tolist()}
    support_observations = load_colmap_track_observations_jsonl(
        Path(args.support_track_observations_jsonl),
        track_ids=candidate_track_ids,
    )
    sampled_support, support_projection_metadata = _project_support_observations_multi_device(
        support_observations,
        support_manifest,
        Path(args.matcha_joint_checkpoint),
        mapper,
        devices=projection_devices,
        feature_key=str(args.feature_key),
        sample_mode=str(args.sample_mode),
        image_batch_size=int(args.projection_image_batch_size),
        load_workers=int(args.projection_load_workers),
    )
    support_by_track = projected_support_descriptors_by_track(sampled_support, support_observations)
    probe = score_support_assignment_strategies(
        query_descriptors=query_descriptors,
        query_context_descriptors=query_context,
        query_viewing_rays=query_rays,
        candidates=candidates,
        landmark_index=landmark_index,
        maplet_index=maplet_index,
        support_by_track=support_by_track,
    )

    correct_track_ids = [int(observation.track_id) for observation in selected_observations]
    assignment = {
        name: summarize_assignment_strategy(
            candidates=candidates,
            correct_track_ids=correct_track_ids,
            query_ids=selected_query_ids,
            scores=values,
        )
        for name, values in probe.strategy_scores.items()
    }
    proposal = proposal_recall_summary(
        candidates,
        correct_track_ids,
        top_ks=tuple(sorted({1, 5, 10, int(args.top_l)})),
    )

    pose_summaries: dict[str, object] = {}
    pose_rows: dict[str, list[dict[str, object]]] = {}
    if str(args.colmap_model_dir):
        model_dir = Path(args.colmap_model_dir)
        cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
        images = read_colmap_images_binary(model_dir / "images.bin")
        images_by_name = {str(image.image_name): image for image in images.values()}
        requested_pose_strategies = tuple(
            item.strip() for item in str(args.pose_strategies).split(",") if item.strip()
        )
        for strategy in requested_pose_strategies:
            if strategy != "proposal_oracle" and strategy not in probe.strategy_scores:
                raise ValueError(f"unknown pose strategy: {strategy}")
            strategy_summary, strategy_rows = _evaluate_pose_strategy(
                strategy=strategy,
                scores=None if strategy == "proposal_oracle" else probe.strategy_scores[strategy],
                candidates=candidates,
                query_observations=selected_observations,
                query_ids=selected_query_ids,
                landmark_index=landmark_index,
                cameras=cameras,
                images_by_name=images_by_name,
                reprojection_error_px=float(args.pnp_reprojection_error_px),
                iterations=int(args.pnp_iterations),
            )
            pose_summaries[strategy] = strategy_summary
            pose_rows[strategy] = strategy_rows

    candidate_rows_path = output_dir / "assignment_candidate_rows.jsonl"
    _write_candidate_rows(
        candidate_rows_path,
        candidates=candidates,
        query_observations=selected_observations,
        query_ids=selected_query_ids,
        strategy_scores=dict(probe.strategy_scores),
        maplet_support_counts=probe.maplet_support_counts,
        all_support_counts=probe.all_support_counts,
    )
    pose_rows_path = output_dir / "pose_rows.json"
    pose_rows_path.write_text(json.dumps(pose_rows, indent=2, sort_keys=True) + "\n")
    compact_probe_path = output_dir / "assignment_probe_arrays.npz"
    np.savez(
        compact_probe_path,
        query_ids=np.asarray(selected_query_ids, dtype=np.str_),
        query_point2d_indices=np.asarray(
            [int(observation.point2d_idx) for observation in selected_observations],
            dtype=np.int64,
        ),
        query_xy=np.asarray([observation.xy for observation in selected_observations], dtype=np.float64),
        correct_track_ids=np.asarray(correct_track_ids, dtype=np.int64),
        bank_row_indices=candidates.bank_row_indices,
        candidate_track_ids=candidates.track_ids,
        candidate_prototype_ids=candidates.prototype_ids,
        coarse_scores=candidates.coarse_scores,
        maplet_support_counts=probe.maplet_support_counts,
        all_support_counts=probe.all_support_counts,
        **{
            f"strategy__{name}": np.asarray(values, dtype=np.float32)
            for name, values in probe.strategy_scores.items()
        },
    )
    summary = {
        "stage": "s4_l0_support_view_assignment_probe",
        "protocol": {
            "query_manifest": str(args.query_manifest),
            "query_track_observations_jsonl": str(args.query_track_observations_jsonl),
            "support_manifest": str(args.support_manifest),
            "support_track_observations_jsonl": str(args.support_track_observations_jsonl),
            "query_support_disjoint": bool(query_image_ids.isdisjoint({str(record.image_id) for record in support_manifest.records})),
            "max_queries": int(args.max_queries),
            "max_observations_per_query": int(args.max_observations_per_query),
            "observation_sampling": str(args.observation_sampling),
        },
        "descriptor_space": {
            "descriptor_space_id": str(landmark_metadata.get("descriptor_space_id", "")),
            "projection_space_id": str(
                dict(landmark_metadata.get("descriptor_space_manifest", {})).get("projection_space_id", "")
            ),
            "checkpoint_sha256": file_sha256_short(Path(args.matcha_joint_checkpoint)),
            "feature_key": str(args.feature_key),
            "projected_landmark_bank": str(args.projected_landmark_bank),
            "projected_landmark_bank_sha256": file_sha256_short(Path(args.projected_landmark_bank)),
            "maplet_support_index": str(args.maplet_support_index),
            "maplet_support_index_sha256": file_sha256_short(Path(args.maplet_support_index)),
        },
        "maplet_audit": maplet_audit,
        "retrieval": {
            "backend": "exact_cosine_unique_track",
            "device": str(args.retrieval_device),
            "top_l": int(args.top_l),
            **proposal,
        },
        "support_projection": support_projection_metadata,
        "support_coverage": {
            "candidate_unique_track_count": int(len(candidate_track_ids)),
            "candidate_tracks_with_support": int(len(support_by_track)),
            "candidate_track_support_rate": (
                0.0 if not candidate_track_ids else float(len(support_by_track) / len(candidate_track_ids))
            ),
            "mean_maplet_support_count": float(
                np.mean(probe.maplet_support_counts[candidates.valid_mask])
            ),
            "mean_all_support_count": float(np.mean(probe.all_support_counts[candidates.valid_mask])),
        },
        "assignment": assignment,
        "pose": pose_summaries,
        "limitations": [
            "query descriptors are sampled at held-out GT SfM observation coordinates (L0 oracle-node probe)",
            "RADIO final mapped descriptors are the only learned feature in this run",
            "oracle_view_angle_best uses the GT query viewing ray and is not deployable before an initial pose",
            "proposal_oracle uses GT track identity only to measure the top-L/PnP upper bound",
            "no candidate is sent to PnP before each query observation group is resolved to one track",
        ],
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "candidate_rows": str(candidate_rows_path),
            "pose_rows": str(pose_rows_path),
            "probe_arrays": str(compact_probe_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
