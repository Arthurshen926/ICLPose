from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.diagnose_landmark_recall_oracle import _sample_observation_descriptor, main, parse_args


def test_diagnose_landmark_recall_oracle_cli_parse_args() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--projected_landmark_cache",
            "landmarks.npz",
            "--landmark_bank",
            "raw_bank.npz",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--top_ks",
            "1,5,10,50",
            "--max_queries",
            "20",
        ]
    )

    assert args.query_manifest == "queries.json"
    assert args.projection_preset == "joint_query_to_projected_observation_landmark"
    assert args.top_ks == "1,5,10,50"
    assert args.max_queries == 20


def test_diagnose_landmark_recall_oracle_cli_accepts_raw_projection_preset() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--landmark_bank",
            "raw_bank.npz",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--projection_preset",
            "raw_query_to_raw_landmark",
        ]
    )

    assert args.projected_landmark_cache == ""
    assert args.landmark_bank == "raw_bank.npz"
    assert args.projection_preset == "raw_query_to_raw_landmark"


def test_diagnose_landmark_recall_oracle_cli_accepts_named_diagnostic_baseline() -> None:
    args = parse_args(
        [
            "--query_manifest",
            "queries.json",
            "--track_observations_jsonl",
            "tracks.jsonl",
            "--landmark_bank",
            "raw_bank.npz",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--image_root",
            "images",
            "--output_dir",
            "out",
            "--projection_preset",
            "post_aggregate_1x1_projection_baseline",
            "--allow_diagnostic_projection",
        ]
    )

    assert args.projection_preset == "post_aggregate_1x1_projection_baseline"
    assert args.allow_diagnostic_projection is True


def test_diagnose_landmark_recall_oracle_rejects_diagnostic_without_allow() -> None:
    with pytest.raises(ValueError, match="diagnostic-only"):
        main(
            [
                "--query_manifest",
                "queries.json",
                "--track_observations_jsonl",
                "tracks.jsonl",
                "--landmark_bank",
                "raw_bank.npz",
                "--matcha_joint_checkpoint",
                "joint.pt",
                "--image_root",
                "images",
                "--output_dir",
                "out",
                "--projection_preset",
                "post_aggregate_1x1_projection_baseline",
            ]
        )


def test_sample_observation_descriptor_passes_sample_mode_keyword() -> None:
    feature_map = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    observation = SimpleNamespace(xy=(3.0, 2.0), image_width=4, image_height=3)

    descriptor = _sample_observation_descriptor(feature_map, observation, sample_mode="nearest")

    np.testing.assert_allclose(descriptor, feature_map[:, 2, 3])
