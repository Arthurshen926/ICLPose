"""Evaluate real RADIO query-to-landmark hybrid localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--landmark_bank", required=True)
    parser.add_argument("--track_observations_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--measurement_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="radio_final")
    parser.add_argument("--query_projection", default="joint", choices=("joint", "raw"))
    parser.add_argument("--landmark_projection", default="joint", choices=("joint", "raw"))
    parser.add_argument("--projection_batch_size", type=int, default=8192)
    parser.add_argument("--projected_landmark_cache", default="")
    parser.add_argument("--landmark_search_backend", default="auto", choices=("auto", "exact", "faiss"))
    parser.add_argument("--landmark_index_cache_size", type=int, default=64)
    parser.add_argument("--match_block_size", type=int, default=512)
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--submap_mode", default="none", choices=("none", "reference_visibility", "hybrid"))
    parser.add_argument("--submap_top_n", type=int, default=5)
    parser.add_argument("--max_submap_landmarks", type=int, default=0)
    parser.add_argument("--submap_spatial_grid_rows", type=int, default=8)
    parser.add_argument("--submap_spatial_grid_cols", type=int, default=8)
    parser.add_argument("--submap_min_landmarks", type=int, default=0)
    parser.add_argument("--submap_min_spatial_cells", type=int, default=16)
    parser.add_argument("--submap_fallback_fraction", type=float, default=0.25)
    parser.add_argument("--measurement_mode", default="none", choices=("none", "owner_rgb"))
    parser.add_argument("--prediction_head", default="gated", choices=("likelihood_mean", "mean", "direct", "gated"))
    parser.add_argument("--measurement_batch_size", type=int, default=128)
    parser.add_argument("--measurement_query_batch_size", type=int, default=1)
    parser.add_argument("--measurement_max_matches", type=int, default=0)
    parser.add_argument(
        "--measurement_selection_strategy",
        default="score_spatial",
        choices=("score_spatial", "coarse_pnp_inliers"),
    )
    parser.add_argument("--measurement_grid_rows", type=int, default=4)
    parser.add_argument("--measurement_grid_cols", type=int, default=4)
    parser.add_argument("--measurement_confidence_temperature", type=float, default=4.0)
    parser.add_argument("--measurement_confidence_bias", type=float, default=0.0)
    parser.add_argument("--measurement_uncertainty_scale", type=float, default=1.0)
    parser.add_argument("--measurement_uncertainty_floor_px", type=float, default=0.0)
    parser.add_argument("--measurement_amp", action="store_true")
    parser.add_argument("--measurement_amp_dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--measurement_tensor_cache_size", type=int, default=64)
    parser.add_argument("--reference_rgb_cache_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--min_similarity_margin", type=float, default=None)
    parser.add_argument("--query_token_step", type=int, default=4)
    parser.add_argument("--query_token_selection", default="uniform", choices=("uniform", "heatmap"))
    parser.add_argument("--query_heatmap_top_k", type=int, default=0)
    parser.add_argument("--query_heatmap_nms_radius", type=int, default=0)
    parser.add_argument("--query_heatmap_grid_rows", type=int, default=4)
    parser.add_argument("--query_heatmap_grid_cols", type=int, default=4)
    parser.add_argument("--query_heatmap_min_score", type=float, default=None)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--max_landmarks", type=int, default=0)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_min_inliers", type=int, default=4)
    parser.add_argument("--enable_quality_rescore", action="store_true")
    parser.add_argument("--measurement_score_mode", default="match", choices=("match", "measurement_quality"))
    parser.add_argument("--min_measurement_confidence", type=float, default=None)
    parser.add_argument("--max_measurement_uncertainty_px", type=float, default=None)
    parser.add_argument("--pnp_min_soft_score", type=float, default=None)
    parser.add_argument("--pnp_weighted_refine", action="store_true")
    parser.add_argument("--pnp_weighted_loss", default="huber")
    parser.add_argument("--pnp_weighted_f_scale_px", type=float, default=4.0)
    parser.add_argument("--pnp_weighted_max_nfev", type=int, default=50)
    parser.add_argument("--progress_interval_queries", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def resolve_runtime_device(device: str) -> torch.device:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


class _RawFeatureMapper:
    def project(self, feature_map: np.ndarray):
        from feature_extract.vfm.localization.schemas import MappedFeatureMap

        arr = np.asarray(feature_map, dtype=np.float32)
        return MappedFeatureMap(coarse_descriptors=arr, measurement_context=arr, offset_logits=None, heatmap=None)


def _track_stats(observations) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    xyz_by_track: dict[int, np.ndarray] = {}
    errors_by_track: dict[int, list[float]] = {}
    for obs in observations:
        xyz_by_track.setdefault(int(obs.track_id), np.asarray(obs.xyz, dtype=np.float64).reshape(3))
        errors_by_track.setdefault(int(obs.track_id), []).append(float(obs.reprojection_error))
    reprojection_error_by_track = {
        int(track_id): float(np.mean(values)) for track_id, values in errors_by_track.items() if values
    }
    return xyz_by_track, reprojection_error_by_track


def _limit_landmarks(index, max_landmarks: int):
    if int(max_landmarks) <= 0 or len(index) <= int(max_landmarks):
        return index
    order = np.lexsort((-index.observation_counts, index.mean_variances))
    return index.subset(order[: int(max_landmarks)])


def _load_reference_submaps(candidate_bank: str, submap_top_n: int) -> dict[str, list[str]]:
    if not candidate_bank:
        return {}
    if int(submap_top_n) <= 0:
        raise ValueError("submap_top_n must be positive")
    by_query: dict[str, list[tuple[int, int, str]]] = {}
    order = 0
    for line in Path(candidate_bank).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("record_type") == "header":
            continue
        if item.get("record_type", "candidate") != "candidate":
            continue
        query_id = item.get("query_id")
        reference_image = item.get("reference_image")
        if query_id is None or reference_image is None:
            continue
        metadata = dict(item.get("metadata") or {})
        rank = int(metadata.get("retrieval_rank", len(by_query.get(str(query_id), [])) + 1))
        by_query.setdefault(str(query_id), []).append((rank, order, str(reference_image)))
        order += 1
    result: dict[str, list[str]] = {}
    for query_id, rows in by_query.items():
        references: list[str] = []
        for _rank, _order, reference in sorted(rows)[: int(submap_top_n)]:
            if reference not in references:
                references.append(reference)
        result[query_id] = references
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
    from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary, read_colmap_images_binary
    from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
    from feature_extract.vfm.localization.landmark_hybrid import (
        LandmarkOwnerObservationIndex,
        LandmarkRetrievalConfig,
        cameras_by_image_name,
        load_landmark_index_npz,
        project_landmark_features_with_joint_model,
        project_landmark_index_features,
        run_real_radio_landmark_hybrid_eval,
        save_landmark_index_npz,
        target_image_sizes_from_observations,
    )
    from feature_extract.vfm.localization.measurement import RGBPatchMeasurementAdapter
    from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
    from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
    from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import load_rgb_patch_measurement_branch
    from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatchingConfig
    from feature_extract.vfm.tokens import TokenBankManifest
    from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl

    runtime_device = resolve_runtime_device(str(args.device))
    device_text = str(runtime_device)
    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    query_manifest.validate(verify_checksums=False)
    observations = load_colmap_track_observations_jsonl(Path(args.track_observations_jsonl))
    xyz_by_track, reprojection_error_by_track = _track_stats(observations)
    landmark_bank = load_selected_track_bank_npz(Path(args.landmark_bank))
    landmark_index = LandmarkMapIndex.from_track_bank(landmark_bank, xyz_by_track, reprojection_error_by_track)
    landmark_index = _limit_landmarks(landmark_index, int(args.max_landmarks))

    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device_text)
    if str(args.query_projection) == "joint":
        feature_mapper = JointFeatureMapper(joint_run.model, device=device_text)
    else:
        feature_mapper = _RawFeatureMapper()

    projected_cache_hit = False
    projected_cache_metadata: dict[str, object] = {}
    projected_cache_path = Path(args.projected_landmark_cache) if str(args.projected_landmark_cache) else None
    if projected_cache_path is not None and projected_cache_path.exists():
        landmark_index, projected_cache_metadata = load_landmark_index_npz(projected_cache_path)
        projected_cache_hit = True
    elif str(args.landmark_projection) == "joint":
        landmark_index = project_landmark_index_features(
            landmark_index,
            lambda values: project_landmark_features_with_joint_model(
                joint_run.model,
                values,
                device=device_text,
                batch_size=int(args.projection_batch_size),
            ),
        )
        if projected_cache_path is not None:
            projected_cache_metadata = {
                "landmark_bank": str(args.landmark_bank),
                "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
                "landmark_projection": str(args.landmark_projection),
                "projection_batch_size": int(args.projection_batch_size),
                "feature_dim": int(landmark_index.feature_dim),
            }
            save_landmark_index_npz(landmark_index, projected_cache_path, metadata=projected_cache_metadata)
    elif projected_cache_path is not None:
        projected_cache_metadata = {
            "landmark_bank": str(args.landmark_bank),
            "landmark_projection": str(args.landmark_projection),
            "feature_dim": int(landmark_index.feature_dim),
        }
        save_landmark_index_npz(landmark_index, projected_cache_path, metadata=projected_cache_metadata)
    landmark_index = _limit_landmarks(landmark_index, int(args.max_landmarks))
    if int(landmark_index.feature_dim) != int(joint_run.model.output_dim if str(args.query_projection) == "joint" else landmark_bank.feature_dim):
        raise ValueError(
            "query and landmark descriptor dimensions do not match; use joint/joint or raw/raw projection modes"
        )

    measurement_adapter = None
    owner_index = None
    measurement_checkpoint = Path(args.measurement_checkpoint) if str(args.measurement_checkpoint) else Path(args.matcha_joint_checkpoint)
    if str(args.measurement_mode) == "owner_rgb":
        branch = load_rgb_patch_measurement_branch(measurement_checkpoint, device=runtime_device)
        measurement_adapter = RGBPatchMeasurementAdapter(
            branch=branch,
            device=device_text,
            prediction_head=str(args.prediction_head),
            batch_size=int(args.measurement_batch_size),
            confidence_temperature=float(args.measurement_confidence_temperature),
            confidence_bias=float(args.measurement_confidence_bias),
            uncertainty_scale=float(args.measurement_uncertainty_scale),
            uncertainty_floor_px=float(args.measurement_uncertainty_floor_px),
            use_amp=bool(args.measurement_amp),
            amp_dtype=str(args.measurement_amp_dtype),
            tensor_cache_size=int(args.measurement_tensor_cache_size),
        )
        target_sizes = target_image_sizes_from_observations(observations, image_root=Path(args.image_root))
        owner_index = LandmarkOwnerObservationIndex(observations, target_image_sizes=target_sizes)

    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    cameras_by_query = cameras_by_image_name(cameras=cameras, colmap_images=images, image_root=Path(args.image_root))
    gt_poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    config = QueryTo3DMatchingConfig(
        top_k=int(args.top_k),
        ratio_threshold=None if bool(args.disable_ratio_test) else float(args.ratio_threshold),
        min_similarity=float(args.min_similarity),
        min_similarity_margin=args.min_similarity_margin,
        mutual=bool(args.mutual),
        min_observation_count=int(args.min_observation_count),
        query_token_step=int(args.query_token_step),
        max_matches=int(args.max_matches) if int(args.max_matches) > 0 else None,
        block_size=int(args.match_block_size),
    )
    retrieval_config = LandmarkRetrievalConfig(
        backend=str(args.landmark_search_backend),
        top_k=int(args.top_k),
        ratio_threshold=None if bool(args.disable_ratio_test) else float(args.ratio_threshold),
        min_similarity=float(args.min_similarity),
        min_similarity_margin=args.min_similarity_margin,
        min_observation_count=int(args.min_observation_count),
        query_token_step=int(args.query_token_step),
        max_matches=int(args.max_matches) if int(args.max_matches) > 0 else None,
        block_size=int(args.match_block_size),
        deduplicate_tracks=True,
        query_token_selection=str(args.query_token_selection),
        query_heatmap_top_k=int(args.query_heatmap_top_k),
        query_heatmap_nms_radius=int(args.query_heatmap_nms_radius),
        query_heatmap_grid_rows=int(args.query_heatmap_grid_rows),
        query_heatmap_grid_cols=int(args.query_heatmap_grid_cols),
        query_heatmap_min_score=args.query_heatmap_min_score,
    )
    reference_submaps = None
    if str(args.submap_mode) in {"reference_visibility", "hybrid"}:
        if not str(args.candidate_bank):
            raise ValueError("--candidate_bank is required when --submap_mode is reference_visibility or hybrid")
        reference_submaps = _load_reference_submaps(str(args.candidate_bank), int(args.submap_top_n))
    submap_selection_mode = "quality_spatial" if str(args.submap_mode) == "hybrid" else "lexsort"
    summary = run_real_radio_landmark_hybrid_eval(
        list(query_manifest.records),
        landmark_index=landmark_index,
        output_dir=Path(args.output_dir),
        feature_mapper=feature_mapper,
        cameras_by_query=cameras_by_query,
        gt_poses_by_query=gt_poses,
        image_root=Path(args.image_root),
        feature_key=str(args.feature_key),
        matching_config=config,
        retrieval_config=retrieval_config,
        retrieval_index_cache_size=int(args.landmark_index_cache_size),
        reference_submaps_by_query=reference_submaps,
        max_submap_landmarks=int(args.max_submap_landmarks) if int(args.max_submap_landmarks) > 0 else None,
        submap_selection_mode=submap_selection_mode,
        submap_spatial_grid_rows=int(args.submap_spatial_grid_rows),
        submap_spatial_grid_cols=int(args.submap_spatial_grid_cols),
        submap_min_landmarks=int(args.submap_min_landmarks),
        submap_min_spatial_cells=int(args.submap_min_spatial_cells),
        submap_fallback_fraction=float(args.submap_fallback_fraction),
        measurement_adapter=measurement_adapter,
        owner_index=owner_index,
        measurement_max_matches=int(args.measurement_max_matches) if int(args.measurement_max_matches) > 0 else None,
        measurement_selection_strategy=str(args.measurement_selection_strategy),
        measurement_query_batch_size=int(args.measurement_query_batch_size),
        measurement_grid_rows=int(args.measurement_grid_rows),
        measurement_grid_cols=int(args.measurement_grid_cols),
        measurement_score_mode=str(args.measurement_score_mode),
        min_measurement_confidence=args.min_measurement_confidence,
        max_measurement_uncertainty_px=args.max_measurement_uncertainty_px,
        enable_quality_rescore=bool(args.enable_quality_rescore),
        pnp_min_soft_score=args.pnp_min_soft_score,
        reference_rgb_cache_size=int(args.reference_rgb_cache_size),
        max_queries=int(args.max_queries) if int(args.max_queries) > 0 else None,
        pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
        pnp_confidence=float(args.pnp_confidence),
        pnp_min_inliers=int(args.pnp_min_inliers),
        pnp_weighted_refine=bool(args.pnp_weighted_refine),
        pnp_weighted_loss=str(args.pnp_weighted_loss),
        pnp_weighted_f_scale_px=float(args.pnp_weighted_f_scale_px),
        pnp_weighted_max_nfev=int(args.pnp_weighted_max_nfev),
        progress_interval_queries=int(args.progress_interval_queries),
    )
    summary.update(
        {
            "query_manifest": str(args.query_manifest),
            "landmark_bank": str(args.landmark_bank),
            "track_observations_jsonl": str(args.track_observations_jsonl),
            "image_root": str(args.image_root),
            "colmap_model_dir": str(args.colmap_model_dir),
            "query_pose_file": str(args.query_pose_file),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "measurement_checkpoint": str(measurement_checkpoint),
            "query_projection": str(args.query_projection),
            "landmark_projection": str(args.landmark_projection),
            "projected_landmark_cache": "" if projected_cache_path is None else str(projected_cache_path),
            "projected_landmark_cache_hit": bool(projected_cache_hit),
            "projected_landmark_cache_metadata": projected_cache_metadata,
            "candidate_bank": str(args.candidate_bank),
            "submap_mode": str(args.submap_mode),
            "submap_top_n": int(args.submap_top_n),
            "max_submap_landmarks": int(args.max_submap_landmarks),
            "submap_selection_mode": submap_selection_mode,
            "submap_spatial_grid_rows": int(args.submap_spatial_grid_rows),
            "submap_spatial_grid_cols": int(args.submap_spatial_grid_cols),
            "submap_min_landmarks": int(args.submap_min_landmarks),
            "submap_min_spatial_cells": int(args.submap_min_spatial_cells),
            "submap_fallback_fraction": float(args.submap_fallback_fraction),
            "measurement_query_batch_size": int(args.measurement_query_batch_size),
            "measurement_selection_strategy": str(args.measurement_selection_strategy),
            "measurement_confidence_temperature": float(args.measurement_confidence_temperature),
            "measurement_confidence_bias": float(args.measurement_confidence_bias),
            "measurement_uncertainty_scale": float(args.measurement_uncertainty_scale),
            "measurement_uncertainty_floor_px": float(args.measurement_uncertainty_floor_px),
            "measurement_amp": bool(args.measurement_amp),
            "measurement_amp_dtype": str(args.measurement_amp_dtype),
            "measurement_tensor_cache_size": int(args.measurement_tensor_cache_size),
            "landmark_index_cache_size": int(args.landmark_index_cache_size),
            "device": device_text,
        }
    )
    Path(summary["outputs"]["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
