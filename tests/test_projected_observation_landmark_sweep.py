from __future__ import annotations

import pytest

from feature_extract.tools.vfm.build_projected_observation_landmark_sweep import (
    _representation_config,
    parse_args,
    parse_csv,
)


def test_projected_landmark_sweep_defaults_cover_s3_representations() -> None:
    args = parse_args(
        [
            "--track_observations",
            "tracks.jsonl",
            "--token_manifest",
            "tokens.json",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
        ]
    )

    assert parse_csv(args.representations) == (
        "mean",
        "normalized_mean",
        "medoid",
        "descriptor_kmeans_2",
    )


def test_projected_landmark_sweep_representation_contracts() -> None:
    normalized, no_clustering = _representation_config(
        "normalized_mean",
        min_observations=2,
        weight_floor=1e-6,
        cluster_iterations=8,
    )
    clustered, clustering = _representation_config(
        "descriptor_kmeans_2",
        min_observations=2,
        weight_floor=1e-6,
        cluster_iterations=8,
    )

    assert normalized.method == "mean"
    assert normalized.l2_normalize_observations is True
    assert no_clustering is None
    assert clustered.l2_normalize_observations is True
    assert clustering is not None
    assert clustering.max_prototypes_per_track == 2


def test_projected_landmark_sweep_rejects_unknown_representation() -> None:
    with pytest.raises(ValueError, match="unsupported landmark representations"):
        parse_csv("mean,unknown")
