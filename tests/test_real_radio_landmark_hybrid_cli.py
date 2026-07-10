from __future__ import annotations

from feature_extract.tools.vfm.build_projected_observation_landmark_bank import (
    parse_args as parse_projected_bank_args,
)
import pytest

from feature_extract.tools.vfm.eval_real_radio_landmark_hybrid import (
    PROJECTION_PRESETS,
    main,
    parse_args,
    projected_cache_expected_metadata,
    resolve_projection_preset,
    resolve_runtime_device,
    validate_projected_cache_metadata,
)
from feature_extract.vfm.localization.descriptor_space import descriptor_space_manifest


def _full_map_descriptor_space_metadata(
    *,
    checkpoint: str = "abc123",
    tracks: str = "tracks123",
    feature_dim: int = 128,
    aggregation_method: str = "mean",
    source_image_hash: str = "images123",
) -> dict[str, object]:
    aggregation = {"method": aggregation_method, "l2_normalize_observations": False}
    space = descriptor_space_manifest(
        checkpoint_sha256=checkpoint,
        mapper_mode="joint_full_map",
        feature_key="radio_final",
        projection_source="projected_observation_full_map",
        aggregation_method=aggregation_method,
        l2_normalize_observations=False,
        image_manifest_hash=source_image_hash,
        sfm_track_hash=tracks,
        descriptor_dimension=feature_dim,
        normalization_mode="row_l2_normalized_search",
    )
    return {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": checkpoint,
        "track_observations_sha256": tracks,
        "feature_dim": feature_dim,
        "mapper_class": "JointFeatureMapper",
        "mapper_config_hash": checkpoint,
        "aggregation": aggregation,
        "source_image_list_hash": source_image_hash,
        "descriptor_dimension": feature_dim,
        "normalization_mode": "row_l2_normalized_search",
        "descriptor_space_manifest": space,
        "descriptor_space_id": space["descriptor_space_id"],
    }


def test_landmark_hybrid_cli_parser_defaults_to_no_measurement() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--landmark_bank",
            "bank.npz",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--image_root",
            "images",
            "--colmap_model_dir",
            "sparse/0",
            "--query_pose_file",
            "poses.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
        ]
    )

    assert args.measurement_mode == "none"
    assert args.landmark_projection == "joint"
    assert args.projection_preset == "joint_query_to_projected_observation_landmark"
    assert args.feature_key == "radio_final"
    assert args.query_token_step == 4
    assert args.query_token_selection == "uniform"
    assert args.query_heatmap_top_k == 0
    assert args.query_heatmap_nms_radius == 0
    assert args.top_k == 2
    assert args.nn_search_k_for_ratio == 0
    assert args.proposal_top_l == 1
    assert args.max_queries == 0
    assert args.projected_landmark_cache == ""
    assert args.landmark_search_backend == "auto"
    assert args.landmark_index_cache_size == 64
    assert args.submap_mode == "none"
    assert args.submap_top_n == 5
    assert args.submap_spatial_grid_rows == 8
    assert args.submap_spatial_grid_cols == 8
    assert args.submap_min_landmarks == 0
    assert args.submap_min_spatial_cells == 16
    assert args.submap_fallback_fraction == 0.25
    assert args.measurement_max_matches == 0
    assert args.measurement_selection_strategy == "score_spatial"
    assert args.measurement_query_batch_size == 1
    assert args.measurement_grid_rows == 4
    assert args.measurement_grid_cols == 4
    assert args.measurement_confidence_temperature == 4.0
    assert args.measurement_uncertainty_scale == 1.0
    assert args.measurement_tensor_cache_size == 64
    assert args.measurement_score_mode == "match"
    assert args.measurement_final_match_policy == "keep_all"
    assert args.drop_rejected_measurements is False
    assert args.measurement_geometry_probability_model == ""
    assert args.min_measurement_geometry_probability is None
    assert args.enable_quality_rescore is False
    assert args.pnp_weighted_refine is False
    assert args.allow_diagnostic_projection is False


def test_projection_preset_resolves_safe_modes() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--landmark_bank",
            "bank.npz",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--image_root",
            "images",
            "--colmap_model_dir",
            "sparse/0",
            "--query_pose_file",
            "poses.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
            "--projection_preset",
            "raw_query_to_raw_landmark",
        ]
    )

    resolved = resolve_projection_preset(args)

    assert resolved.query_projection == "raw"
    assert resolved.landmark_projection == "raw"
    assert resolved.expected_projection_mode == "raw_landmark_bank"


def test_post_aggregate_projection_baseline_is_diagnostic_only() -> None:
    preset = PROJECTION_PRESETS["post_aggregate_1x1_projection_baseline"]

    assert preset.diagnostic_only is True
    assert preset.expected_projection_mode == "post_aggregate_1x1_projection_baseline"


def test_eval_rejects_diagnostic_projection_without_explicit_allow() -> None:
    with pytest.raises(ValueError, match="diagnostic-only"):
        main(
            [
                "--query_manifest",
                "queries.json",
                "--landmark_bank",
                "bank.npz",
                "--track_observations_jsonl",
                "tracks.jsonl",
                "--image_root",
                "images",
                "--colmap_model_dir",
                "sparse/0",
                "--query_pose_file",
                "poses.txt",
                "--matcha_joint_checkpoint",
                "joint.pt",
                "--output_dir",
                "out",
                "--projection_preset",
                "post_aggregate_1x1_projection_baseline",
            ]
        )


def test_projected_cache_metadata_validation_rejects_stale_checkpoint() -> None:
    expected = {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": "abc123",
        "track_observations_sha256": "tracks123",
        "feature_dim": 128,
    }
    metadata = _full_map_descriptor_space_metadata(checkpoint="different")

    with pytest.raises(ValueError, match="projected landmark cache metadata mismatch"):
        validate_projected_cache_metadata(metadata, expected)


def test_projected_cache_metadata_validation_accepts_matching_descriptor_space() -> None:
    expected = {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": "abc123",
        "track_observations_sha256": "tracks123",
        "feature_dim": 128,
    }

    validate_projected_cache_metadata(_full_map_descriptor_space_metadata(), expected)


def test_projected_cache_metadata_validation_requires_source_image_hash() -> None:
    expected = {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": "abc123",
        "track_observations_sha256": "tracks123",
        "feature_dim": 128,
    }
    metadata = _full_map_descriptor_space_metadata()
    metadata.pop("source_image_list_hash")

    with pytest.raises(ValueError, match="source_image_list_hash"):
        validate_projected_cache_metadata(metadata, expected)


def test_projected_cache_metadata_validation_requires_descriptor_space_manifest() -> None:
    expected = {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": "abc123",
        "track_observations_sha256": "tracks123",
        "feature_dim": 128,
    }
    metadata = _full_map_descriptor_space_metadata()
    metadata.pop("descriptor_space_manifest")

    with pytest.raises(ValueError, match="descriptor_space_manifest"):
        validate_projected_cache_metadata(metadata, expected)


def test_projected_cache_metadata_validation_rejects_descriptor_space_mismatch() -> None:
    expected = {
        "projection_mode": "full_map_projected_observations",
        "feature_key": "radio_final",
        "matcha_joint_checkpoint_sha256": "abc123",
        "track_observations_sha256": "tracks123",
        "feature_dim": 128,
    }
    metadata = _full_map_descriptor_space_metadata()
    metadata["descriptor_space_id"] = "stale"

    with pytest.raises(ValueError, match="descriptor_space_id"):
        validate_projected_cache_metadata(metadata, expected)


def test_projected_cache_expected_metadata_hashes_inputs(tmp_path) -> None:
    checkpoint = tmp_path / "joint.pt"
    tracks = tmp_path / "tracks.jsonl"
    checkpoint.write_text("checkpoint")
    tracks.write_text("tracks")

    expected = projected_cache_expected_metadata(
        projection_mode="full_map_projected_observations",
        feature_key="radio_final",
        matcha_joint_checkpoint=checkpoint,
        track_observations=tracks,
        feature_dim=128,
    )

    assert expected["matcha_joint_checkpoint_sha256"] != ""
    assert expected["track_observations_sha256"] != ""
    assert expected["projection_mode"] == "full_map_projected_observations"


def test_landmark_hybrid_cli_parser_accepts_measurement_verification_args() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--landmark_bank",
            "bank.npz",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--image_root",
            "images",
            "--colmap_model_dir",
            "sparse/0",
            "--query_pose_file",
            "poses.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
            "--measurement_final_match_policy",
            "measured_only",
            "--drop_rejected_measurements",
            "--measurement_geometry_probability_model",
            "calibration.json",
            "--min_measurement_geometry_probability",
            "0.35",
        ]
    )

    assert args.measurement_final_match_policy == "measured_only"
    assert args.drop_rejected_measurements is True
    assert args.measurement_geometry_probability_model == "calibration.json"
    assert args.min_measurement_geometry_probability == 0.35


def test_resolve_runtime_device_falls_back_to_cpu_for_unavailable_cuda() -> None:
    device = resolve_runtime_device("cuda")

    assert str(device) in {"cuda", "cpu"}


def test_projected_observation_landmark_bank_cli_parser_defaults_to_full_map_projection() -> None:
    args = parse_projected_bank_args(
        [
            "--track_observations",
            "tracks.jsonl",
            "--token_manifest",
            "manifest.json",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_index",
            "projected.npz",
            "--summary_json",
            "summary.json",
        ]
    )

    assert args.feature_key == "radio_final"
    assert args.method == "mean"
    assert args.sample_mode == "bilinear"
    assert args.min_observations == 2
    assert args.device == "cuda"
