from __future__ import annotations

from pathlib import Path

from feature_extract.tools.vfm.run_real_radio_fullbank_sweep import (
    DEFAULT_AGGREGATION_METHODS,
    build_eval_command,
    parse_csv_ints,
    parse_csv_strings,
)


def test_fullbank_sweep_defaults_include_requested_aggregation_methods() -> None:
    assert DEFAULT_AGGREGATION_METHODS == (
        "mean",
        "cosine_weighted_mean",
        "geometry_weighted",
        "robust_trimmed_mean",
        "view_consistent",
        "geometric_median",
        "medoid",
        "ulf_geometry_weighted",
    )


def test_sweep_csv_parsers_ignore_empty_items() -> None:
    assert parse_csv_strings("mean, geometry_weighted,,medoid") == ("mean", "geometry_weighted", "medoid")
    assert parse_csv_ints("0,20,,40,80") == (0, 20, 40, 80)


def test_build_eval_command_uses_fullbank_no_submap_and_optional_measurement(tmp_path: Path) -> None:
    common = {
        "query_manifest": "query.json",
        "landmark_bank": "raw_bank.npz",
        "track_observations_jsonl": "tracks.jsonl",
        "image_root": "images",
        "colmap_model_dir": "sparse/0",
        "query_pose_file": "poses.txt",
        "matcha_joint_checkpoint": "joint.pt",
        "projected_landmark_cache": "projected.npz",
        "output_dir": str(tmp_path / "eval"),
        "device": "cuda",
        "max_queries": 0,
        "measurement_selection_strategy": "coarse_pnp_inliers",
        "proposal_top_l": 10,
        "nn_search_k_for_ratio": 20,
        "disable_ratio_test": True,
    }

    no_measurement = build_eval_command(common, measurement_k=0)
    measured = build_eval_command({**common, "measurement_checkpoint": "measurement.pt"}, measurement_k=40)

    assert "--submap_mode" in no_measurement
    assert no_measurement[no_measurement.index("--submap_mode") + 1] == "none"
    assert no_measurement[no_measurement.index("--landmark_search_backend") + 1] == "faiss"
    assert no_measurement[no_measurement.index("--query_token_selection") + 1] == "heatmap"
    assert no_measurement[no_measurement.index("--proposal_top_l") + 1] == "10"
    assert no_measurement[no_measurement.index("--nn_search_k_for_ratio") + 1] == "20"
    assert "--disable_ratio_test" in no_measurement
    assert "--measurement_mode" not in no_measurement
    assert "--measurement_max_matches" not in no_measurement

    assert measured[measured.index("--measurement_mode") + 1] == "owner_rgb"
    assert measured[measured.index("--measurement_max_matches") + 1] == "40"
    assert measured[measured.index("--measurement_selection_strategy") + 1] == "coarse_pnp_inliers"
    assert measured[measured.index("--measurement_query_batch_size") + 1] == "8"
