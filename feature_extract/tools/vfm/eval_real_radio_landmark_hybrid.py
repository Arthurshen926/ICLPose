"""Evaluate real RADIO query-to-landmark hybrid localization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.descriptor_space import (
    canonical_descriptor_space_id,
    canonical_projection_space_id,
    descriptor_space_manifest,
    post_aggregate_1x1_descriptor_space_manifest,
    raw_descriptor_space_manifest,
    token_feature_source_config,
    validate_projected_observation_descriptor_manifest,
)


@dataclass(frozen=True)
class ProjectionPreset:
    name: str
    query_projection: str
    landmark_projection: str
    expected_projection_mode: str
    requires_projected_cache: bool = False
    diagnostic_only: bool = False


PROJECTION_PRESETS: dict[str, ProjectionPreset] = {
    "raw_query_to_raw_landmark": ProjectionPreset(
        name="raw_query_to_raw_landmark",
        query_projection="raw",
        landmark_projection="raw",
        expected_projection_mode="raw_landmark_bank",
    ),
    "joint_query_to_projected_observation_landmark": ProjectionPreset(
        name="joint_query_to_projected_observation_landmark",
        query_projection="joint",
        landmark_projection="joint",
        expected_projection_mode="full_map_projected_observations",
        requires_projected_cache=True,
    ),
    "post_aggregate_1x1_projection_baseline": ProjectionPreset(
        name="post_aggregate_1x1_projection_baseline",
        query_projection="joint",
        landmark_projection="joint",
        expected_projection_mode="post_aggregate_1x1_projection_baseline",
        diagnostic_only=True,
    ),
}


def resolve_projection_preset(args: argparse.Namespace) -> ProjectionPreset:
    preset_name = str(getattr(args, "projection_preset", "joint_query_to_projected_observation_landmark"))
    try:
        return PROJECTION_PRESETS[preset_name]
    except KeyError as exc:
        raise ValueError(f"unsupported projection_preset: {preset_name}") from exc


def projected_cache_expected_metadata(
    *,
    projection_mode: str,
    feature_key: str,
    matcha_joint_checkpoint: Path,
    track_observations: Path,
    feature_dim: int,
    descriptor_source_config: dict[str, object] | None = None,
) -> dict[str, object]:
    expected = {
        "projection_mode": str(projection_mode),
        "feature_key": str(feature_key),
        "track_observations_sha256": file_sha256_short(Path(track_observations)),
        "feature_dim": int(feature_dim),
    }
    if str(projection_mode) != "raw_landmark_bank":
        expected["matcha_joint_checkpoint_sha256"] = file_sha256_short(Path(matcha_joint_checkpoint))
    if descriptor_source_config is not None:
        expected["descriptor_source_config"] = dict(descriptor_source_config)
    return expected


def validate_projected_cache_metadata(metadata: dict[str, object], expected: dict[str, object]) -> None:
    mismatches: list[str] = []
    required_keys: tuple[str, ...] = (
        "projection_mode",
        "feature_key",
        "track_observations_sha256",
        "feature_dim",
        "descriptor_dimension",
        "normalization_mode",
        "descriptor_space_manifest",
        "descriptor_space_id",
    )
    if expected.get("projection_mode") == "full_map_projected_observations":
        required_keys = required_keys + (
            "matcha_joint_checkpoint_sha256",
            "mapper_class",
            "mapper_config_hash",
            "aggregation",
            "prototype_builder",
            "source_image_list_hash",
            "descriptor_source_config",
            "descriptor_source_config_audit",
            "sample_mode",
            "observation_selection",
        )
    for key in required_keys:
        if key not in metadata or metadata.get(key) in ("", None):
            mismatches.append(f"{key}: missing")
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual!r}")
    actual_manifest = metadata.get("descriptor_space_manifest")
    if not isinstance(actual_manifest, dict):
        mismatches.append("descriptor_space_manifest: expected object")
        actual_manifest = {}
    else:
        actual_id = str(actual_manifest.get("descriptor_space_id", ""))
        recomputed_id = canonical_descriptor_space_id(actual_manifest)
        if actual_id != recomputed_id:
            mismatches.append(
                f"descriptor_space_manifest.descriptor_space_id: expected recomputed {recomputed_id!r}, got {actual_id!r}"
            )
        if metadata.get("descriptor_space_id") != actual_id:
            mismatches.append(
                "descriptor_space_id: expected to match descriptor_space_manifest.descriptor_space_id, "
                f"got {metadata.get('descriptor_space_id')!r} vs {actual_id!r}"
            )
    if expected.get("projection_mode") == "full_map_projected_observations":
        if isinstance(actual_manifest, dict):
            try:
                validate_projected_observation_descriptor_manifest(actual_manifest)
            except ValueError as exc:
                mismatches.append(str(exc))
        aggregation = metadata.get("aggregation")
        if not isinstance(aggregation, dict):
            mismatches.append("aggregation: expected object")
            aggregation = {}
        prototype_builder = metadata.get("prototype_builder")
        if not isinstance(prototype_builder, dict):
            mismatches.append("prototype_builder: expected object")
            prototype_builder = {}
        source_config = expected.get("descriptor_source_config", metadata.get("descriptor_source_config"))
        if not isinstance(source_config, dict):
            mismatches.append("descriptor_source_config: expected object")
            source_config = {}
        expected_manifest = descriptor_space_manifest(
            checkpoint_sha256=str(expected.get("matcha_joint_checkpoint_sha256", "")),
            mapper_mode="joint_full_map",
            feature_key=str(expected.get("feature_key", "")),
            projection_source="projected_observation_full_map",
            aggregation_method=str(aggregation.get("method", "")),
            l2_normalize_observations=bool(aggregation.get("l2_normalize_observations", False)),
            image_manifest_hash=str(metadata.get("source_image_list_hash", "")),
            sfm_track_hash=str(expected.get("track_observations_sha256", "")),
            descriptor_dimension=int(expected.get("feature_dim", 0)),
            normalization_mode=str(metadata.get("normalization_mode", "")),
            output_branch="coarse_descriptors",
            radio_model=str(source_config.get("radio_model", "")),
            radio_model_version=str(source_config.get("radio_model_version", "")),
            radio_intermediate_layer_index=source_config.get("radio_intermediate_layer_index"),
            preprocessing=dict(source_config.get("preprocessing", {})),
            feature_grid_stride=int(source_config.get("feature_grid_stride", 0)),
            input_channels=int(source_config.get("input_channels", 0)),
            sampling_mode=str(metadata.get("sample_mode", "")),
            sampling_convention="pixel_endpoint_to_token_endpoint_v1",
            sampling_align_corners=True,
            prototype_builder_version=str(actual_manifest.get("prototype_builder_version", "")),
            prototype_builder_config=prototype_builder,
            view_clustering=dict(actual_manifest.get("view_clustering", {})),
            maplet_bank_version=str(actual_manifest.get("maplet_bank_version", "")),
            observation_selection=dict(metadata.get("observation_selection", {})),
        )
        for key, expected_value in expected_manifest.items():
            actual_value = actual_manifest.get(key) if isinstance(actual_manifest, dict) else None
            if actual_value != expected_value:
                mismatches.append(f"descriptor_space_manifest.{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        raise ValueError("projected landmark cache metadata mismatch: " + "; ".join(mismatches))


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
    parser.add_argument(
        "--projection_preset",
        default="joint_query_to_projected_observation_landmark",
        choices=tuple(PROJECTION_PRESETS),
    )
    parser.add_argument(
        "--allow_diagnostic_projection",
        action="store_true",
        help="Allow diagnostic-only descriptor paths such as post-aggregate 1x1 projection baselines.",
    )
    parser.add_argument("--query_projection", default="joint", choices=("joint", "raw"))
    parser.add_argument("--landmark_projection", default="joint", choices=("joint", "raw"))
    parser.add_argument("--projection_batch_size", type=int, default=8192)
    parser.add_argument("--projected_landmark_cache", default="")
    parser.add_argument(
        "--landmark_search_backend",
        default="auto",
        choices=("auto", "exact", "faiss", "torch_cuda"),
    )
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
    parser.add_argument("--nn_search_k_for_ratio", type=int, default=0)
    parser.add_argument("--proposal_top_l", type=int, default=1)
    parser.add_argument(
        "--proposal_track_deduplication",
        default="auto",
        choices=("auto", "across_tokens", "none"),
    )
    parser.add_argument(
        "--proposal_only",
        action="store_true",
        help="Export grouped proposals without running PnP; required for unresolved top-L candidates.",
    )
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
    parser.add_argument(
        "--query_xy_coordinate_mode",
        default="edge_legacy",
        choices=("edge_legacy", "cell_center"),
    )
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
    parser.add_argument("--measurement_final_match_policy", default="keep_all", choices=("keep_all", "measured_only"))
    parser.add_argument("--min_measurement_confidence", type=float, default=None)
    parser.add_argument("--max_measurement_uncertainty_px", type=float, default=None)
    parser.add_argument("--measurement_geometry_probability_model", default="")
    parser.add_argument("--min_measurement_geometry_probability", type=float, default=None)
    parser.add_argument("--drop_rejected_measurements", action="store_true")
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
    if int(args.proposal_top_l) > 1 and not bool(args.proposal_only):
        raise ValueError(
            "proposal_top_l > 1 contains mutually exclusive tracks per query token; "
            "use --proposal_only until a local assignment/conflict resolver has selected at most one track per token"
        )

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
    from feature_extract.vfm.localization.measurement_calibration import load_geometry_probability_model
    from feature_extract.vfm.localization.measurement import RGBPatchMeasurementAdapter
    from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
    from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
    from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import load_rgb_patch_measurement_branch
    from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatchingConfig
    from feature_extract.vfm.tokens import TokenBankManifest
    from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl

    runtime_device = resolve_runtime_device(str(args.device))
    device_text = str(runtime_device)
    projection_preset = resolve_projection_preset(args)
    if projection_preset.diagnostic_only and not bool(args.allow_diagnostic_projection):
        raise ValueError(
            f"projection_preset={projection_preset.name} is diagnostic-only; "
            "rerun with --allow_diagnostic_projection only for ablation/debugging"
        )
    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    query_manifest.validate(verify_checksums=False)
    query_descriptor_source_config = token_feature_source_config(query_manifest, str(args.feature_key))
    projected_cache_path = Path(args.projected_landmark_cache) if str(args.projected_landmark_cache) else None
    projected_cache_exists = bool(projected_cache_path is not None and projected_cache_path.exists())
    observations = (
        load_colmap_track_observations_jsonl(Path(args.track_observations_jsonl))
        if not projected_cache_exists or str(args.measurement_mode) == "owner_rgb"
        else []
    )
    landmark_bank = None
    landmark_index = None
    if not projected_cache_exists:
        xyz_by_track, reprojection_error_by_track = _track_stats(observations)
        landmark_bank = load_selected_track_bank_npz(Path(args.landmark_bank))
        landmark_index = LandmarkMapIndex.from_track_bank(landmark_bank, xyz_by_track, reprojection_error_by_track)
        landmark_index = _limit_landmarks(landmark_index, int(args.max_landmarks))

    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device_text)
    if int(query_descriptor_source_config["input_channels"]) != int(joint_run.model.input_dim):
        raise ValueError(
            "query token feature source does not match mapper input dimension: "
            f"tokens={query_descriptor_source_config['input_channels']!r}, model={joint_run.model.input_dim!r}"
        )
    query_projection = projection_preset.query_projection
    landmark_projection = projection_preset.landmark_projection
    if str(query_projection) == "joint":
        feature_mapper = JointFeatureMapper(joint_run.model, device=device_text)
    else:
        feature_mapper = _RawFeatureMapper()

    projected_cache_hit = False
    projected_cache_metadata: dict[str, object] = {}
    query_descriptor_space_manifest: dict[str, object] = {}
    landmark_descriptor_space_manifest: dict[str, object] = {}
    if projection_preset.requires_projected_cache and projected_cache_path is None:
        raise ValueError(f"--projected_landmark_cache is required for projection_preset={projection_preset.name}")
    if projected_cache_path is not None and projected_cache_path.exists():
        landmark_index, projected_cache_metadata = load_landmark_index_npz(projected_cache_path)
        expected_cache_metadata = projected_cache_expected_metadata(
            projection_mode=projection_preset.expected_projection_mode,
            feature_key=str(args.feature_key),
            matcha_joint_checkpoint=Path(args.matcha_joint_checkpoint),
            track_observations=Path(args.track_observations_jsonl),
            feature_dim=int(landmark_index.feature_dim),
            descriptor_source_config=query_descriptor_source_config,
        )
        validate_projected_cache_metadata(projected_cache_metadata, expected_cache_metadata)
        landmark_descriptor_space_manifest = dict(projected_cache_metadata["descriptor_space_manifest"])
        query_descriptor_space_manifest = dict(landmark_descriptor_space_manifest)
        if query_descriptor_space_manifest.get("projection_space_id") != canonical_projection_space_id(
            query_descriptor_space_manifest
        ):
            raise ValueError("query descriptor projection-space id is internally inconsistent")
        projected_cache_hit = True
    elif str(landmark_projection) == "joint":
        if landmark_index is None:
            raise RuntimeError("raw landmark index was not initialized for on-demand projection")
        if projection_preset.expected_projection_mode == "full_map_projected_observations":
            raise ValueError(
                "full-map projected-observation landmark preset requires an existing projected_landmark_cache"
            )
        landmark_index = project_landmark_index_features(
            landmark_index,
            lambda values: project_landmark_features_with_joint_model(
                joint_run.model,
                values,
                device=device_text,
                batch_size=int(args.projection_batch_size),
            ),
        )
        space_manifest = post_aggregate_1x1_descriptor_space_manifest(
            checkpoint_sha256=file_sha256_short(Path(args.matcha_joint_checkpoint)),
            feature_key=str(args.feature_key),
            raw_landmark_bank_sha256=file_sha256_short(Path(args.landmark_bank)),
            sfm_track_hash=file_sha256_short(Path(args.track_observations_jsonl)),
            descriptor_dimension=int(landmark_index.feature_dim),
            normalization_mode="row_l2_normalized_search",
        )
        landmark_descriptor_space_manifest = dict(space_manifest)
        query_descriptor_space_manifest = dict(space_manifest)
        if projected_cache_path is not None:
            projected_cache_metadata = {
                "landmark_bank": str(args.landmark_bank),
                "landmark_bank_sha256": file_sha256_short(Path(args.landmark_bank)),
                "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
                "matcha_joint_checkpoint_sha256": file_sha256_short(Path(args.matcha_joint_checkpoint)),
                "track_observations": str(args.track_observations_jsonl),
                "track_observations_sha256": file_sha256_short(Path(args.track_observations_jsonl)),
                "feature_key": str(args.feature_key),
                "projection_preset": str(projection_preset.name),
                "projection_mode": str(projection_preset.expected_projection_mode),
                "mapper_class": "JointFeatureMapper",
                "normalization_mode": "row_l2_normalized_search",
                "landmark_projection": str(landmark_projection),
                "diagnostic_only": True,
                "projection_batch_size": int(args.projection_batch_size),
                "descriptor_space_manifest": space_manifest,
                "descriptor_space_id": str(space_manifest["descriptor_space_id"]),
                "feature_dim": int(landmark_index.feature_dim),
                "descriptor_dimension": int(landmark_index.feature_dim),
            }
            save_landmark_index_npz(landmark_index, projected_cache_path, metadata=projected_cache_metadata)
    elif projected_cache_path is not None:
        if landmark_index is None:
            raise RuntimeError("raw landmark index was not initialized for cache creation")
        space_manifest = raw_descriptor_space_manifest(
            feature_key=str(args.feature_key),
            descriptor_dimension=int(landmark_index.feature_dim),
            normalization_mode="row_l2_normalized_search",
        )
        landmark_descriptor_space_manifest = dict(space_manifest)
        query_descriptor_space_manifest = dict(space_manifest)
        projected_cache_metadata = {
            "landmark_bank": str(args.landmark_bank),
            "landmark_bank_sha256": file_sha256_short(Path(args.landmark_bank)),
            "track_observations": str(args.track_observations_jsonl),
            "track_observations_sha256": file_sha256_short(Path(args.track_observations_jsonl)),
            "feature_key": str(args.feature_key),
            "projection_preset": str(projection_preset.name),
            "projection_mode": str(projection_preset.expected_projection_mode),
            "normalization_mode": "row_l2_normalized_search",
            "landmark_projection": str(landmark_projection),
            "descriptor_space_manifest": space_manifest,
            "descriptor_space_id": str(space_manifest["descriptor_space_id"]),
            "feature_dim": int(landmark_index.feature_dim),
            "descriptor_dimension": int(landmark_index.feature_dim),
        }
        save_landmark_index_npz(landmark_index, projected_cache_path, metadata=projected_cache_metadata)
    else:
        if landmark_index is None:
            raise RuntimeError("raw landmark index was not initialized")
        space_manifest = raw_descriptor_space_manifest(
            feature_key=str(args.feature_key),
            descriptor_dimension=int(landmark_index.feature_dim),
            normalization_mode="row_l2_normalized_search",
        )
        landmark_descriptor_space_manifest = dict(space_manifest)
        query_descriptor_space_manifest = dict(space_manifest)
    if query_descriptor_space_manifest.get("descriptor_space_id") != landmark_descriptor_space_manifest.get("descriptor_space_id"):
        raise ValueError(
            "query and landmark descriptor space mismatch: "
            f"query={query_descriptor_space_manifest.get('descriptor_space_id')!r}, "
            f"landmark={landmark_descriptor_space_manifest.get('descriptor_space_id')!r}"
        )
    landmark_index = _limit_landmarks(landmark_index, int(args.max_landmarks))
    query_descriptor_dimension = (
        int(joint_run.model.output_dim)
        if str(query_projection) == "joint"
        else int(query_descriptor_source_config["input_channels"])
    )
    if int(landmark_index.feature_dim) != query_descriptor_dimension:
        raise ValueError(
            "query and landmark descriptor dimensions do not match for projection_preset="
            f"{projection_preset.name}"
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
    measurement_geometry_model = None
    if str(args.measurement_geometry_probability_model):
        measurement_geometry_model = load_geometry_probability_model(Path(args.measurement_geometry_probability_model))

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
    proposal_track_deduplication = str(args.proposal_track_deduplication)
    deduplicate_proposal_tracks = (
        int(args.proposal_top_l) == 1
        if proposal_track_deduplication == "auto"
        else proposal_track_deduplication == "across_tokens"
    )
    retrieval_config = LandmarkRetrievalConfig(
        backend=str(args.landmark_search_backend),
        top_k=int(args.top_k),
        nn_search_k_for_ratio=int(args.nn_search_k_for_ratio) if int(args.nn_search_k_for_ratio) > 0 else None,
        proposal_top_l=int(args.proposal_top_l),
        ratio_threshold=None if bool(args.disable_ratio_test) else float(args.ratio_threshold),
        min_similarity=float(args.min_similarity),
        min_similarity_margin=args.min_similarity_margin,
        min_observation_count=int(args.min_observation_count),
        query_token_step=int(args.query_token_step),
        max_matches=int(args.max_matches) if int(args.max_matches) > 0 else None,
        block_size=int(args.match_block_size),
        deduplicate_tracks=bool(deduplicate_proposal_tracks),
        query_token_selection=str(args.query_token_selection),
        query_heatmap_top_k=int(args.query_heatmap_top_k),
        query_heatmap_nms_radius=int(args.query_heatmap_nms_radius),
        query_heatmap_grid_rows=int(args.query_heatmap_grid_rows),
        query_heatmap_grid_cols=int(args.query_heatmap_grid_cols),
        query_heatmap_min_score=args.query_heatmap_min_score,
        query_xy_coordinate_mode=str(args.query_xy_coordinate_mode),
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
        measurement_final_match_policy=str(args.measurement_final_match_policy),
        min_measurement_confidence=args.min_measurement_confidence,
        max_measurement_uncertainty_px=args.max_measurement_uncertainty_px,
        measurement_geometry_model=measurement_geometry_model,
        min_measurement_geometry_probability=args.min_measurement_geometry_probability,
        drop_rejected_measurements=bool(args.drop_rejected_measurements),
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
        evaluate_pose=not bool(args.proposal_only),
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
            "resolved_query_projection": str(query_projection),
            "resolved_landmark_projection": str(landmark_projection),
            "projection_preset": str(projection_preset.name),
            "projection_preset_diagnostic_only": bool(projection_preset.diagnostic_only),
            "allow_diagnostic_projection": bool(args.allow_diagnostic_projection),
            "query_descriptor_space_id": query_descriptor_space_manifest.get("descriptor_space_id"),
            "landmark_descriptor_space_id": landmark_descriptor_space_manifest.get("descriptor_space_id"),
            "query_descriptor_space_manifest": query_descriptor_space_manifest,
            "landmark_descriptor_space_manifest": landmark_descriptor_space_manifest,
            "query_descriptor_source_config": query_descriptor_source_config,
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
            "measurement_final_match_policy": str(args.measurement_final_match_policy),
            "measurement_confidence_temperature": float(args.measurement_confidence_temperature),
            "measurement_confidence_bias": float(args.measurement_confidence_bias),
            "measurement_uncertainty_scale": float(args.measurement_uncertainty_scale),
            "measurement_uncertainty_floor_px": float(args.measurement_uncertainty_floor_px),
            "measurement_geometry_probability_model": str(args.measurement_geometry_probability_model),
            "min_measurement_geometry_probability": args.min_measurement_geometry_probability,
            "drop_rejected_measurements": bool(args.drop_rejected_measurements),
            "measurement_amp": bool(args.measurement_amp),
            "measurement_amp_dtype": str(args.measurement_amp_dtype),
            "measurement_tensor_cache_size": int(args.measurement_tensor_cache_size),
            "landmark_index_cache_size": int(args.landmark_index_cache_size),
            "device": device_text,
            "proposal_only": bool(args.proposal_only),
            "proposal_track_deduplication": proposal_track_deduplication,
        }
    )
    Path(summary["outputs"]["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
