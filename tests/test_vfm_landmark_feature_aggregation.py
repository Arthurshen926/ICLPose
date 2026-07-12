import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_raw_vfm_landmark_bank import main as build_raw_landmark_bank_cli_main
from feature_extract.tools.vfm.build_projected_observation_landmark_bank import parse_args as parse_projected_bank_args
from feature_extract.tools.vfm.visualize_raw_vfm_landmark_bank import main as visualize_raw_landmark_bank_cli_main
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.landmark_feature_aggregation import (
    LandmarkAggregationConfig,
    LandmarkViewClusteringConfig,
    TrackPrototypeBuilder,
    aggregate_landmark_features,
    aggregate_landmark_features_torch,
    build_multi_prototype_track_bank,
    evaluate_landmark_observation_retrieval,
    evaluate_landmark_split_stability,
)
from feature_extract.vfm.map_lifting import TrackObservation, load_selected_track_bank_npz
from feature_extract.vfm.track_feature_sampling import sample_token_track_observations
from feature_extract.vfm.track_feature_sampling import (
    load_sampled_track_observations_npz,
    save_sampled_track_observations_npz,
)
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _obs(track_id, feature, utility=1.0, image_id=None, visible=True, geometry_valid=True):
    return TrackObservation(
        track_id=track_id,
        image_id=image_id or f"img_{track_id}",
        feature=np.asarray(feature, dtype=np.float32),
        visible=visible,
        geometry_valid=geometry_valid,
        utility=float(utility),
    )


def test_geometry_weighted_aggregation_uses_observation_utility():
    observations = [
        _obs(1, [0.0, 0.0], utility=1.0, image_id="a"),
        _obs(1, [10.0, 0.0], utility=9.0, image_id="b"),
    ]

    bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="geometry_weighted", min_observations=2),
    )

    assert bank.feature_dim == 2
    np.testing.assert_allclose(bank.tracks[1].mean_feature, np.asarray([9.0, 0.0], dtype=np.float32), atol=1e-5)
    assert bank.tracks[1].observation_count == 2


def test_torch_aggregate_matches_numpy_mean_when_available():
    pytest.importorskip("torch")
    observations = [
        _obs(1, [1.0, 0.0], utility=1.0, image_id="a"),
        _obs(1, [0.0, 1.0], utility=3.0, image_id="b"),
        _obs(2, [2.0, 0.0], utility=1.0, image_id="a"),
        _obs(2, [4.0, 2.0], utility=1.0, image_id="b"),
    ]
    config = LandmarkAggregationConfig(method="mean", min_observations=2, l2_normalize_observations=False)

    numpy_bank = aggregate_landmark_features(observations, config)
    torch_bank = aggregate_landmark_features_torch(observations, config, device="cpu")

    assert sorted(torch_bank.tracks) == sorted(numpy_bank.tracks)
    for track_id in sorted(numpy_bank.tracks):
        np.testing.assert_allclose(torch_bank.tracks[track_id].mean_feature, numpy_bank.tracks[track_id].mean_feature)
        np.testing.assert_allclose(torch_bank.tracks[track_id].variance, numpy_bank.tracks[track_id].variance)


def test_shared_track_prototype_builder_matches_numpy_and_differentiable_paths():
    torch = pytest.importorskip("torch")
    observations = [
        _obs(1, [2.0, 0.0], image_id="a"),
        _obs(1, [0.0, 1.0], image_id="b"),
        _obs(2, [0.0, 3.0], image_id="a"),
        _obs(2, [1.0, 1.0], image_id="b"),
    ]
    builder = TrackPrototypeBuilder(
        LandmarkAggregationConfig(method="mean", min_observations=2, l2_normalize_observations=False),
        normalize_final_prototypes=True,
    )

    bank = builder.build_bank(observations)
    fast_bank = builder.build_bank_torch(observations, device="cpu")
    rows = torch.tensor(np.stack([obs.feature for obs in observations]), dtype=torch.float32, requires_grad=True)
    prototypes, counts = builder.aggregate_torch(rows, torch.tensor([0, 0, 1, 1]), 2)

    np.testing.assert_allclose(
        prototypes.detach().numpy(),
        np.stack([bank.tracks[1].mean_feature, bank.tracks[2].mean_feature]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.stack([fast_bank.tracks[1].mean_feature, fast_bank.tracks[2].mean_feature]),
        np.stack([bank.tracks[1].mean_feature, bank.tracks[2].mean_feature]),
        atol=1e-6,
    )
    np.testing.assert_array_equal(counts.detach().numpy(), np.asarray([2.0, 2.0]))
    prototypes.sum().backward()
    assert rows.grad is not None


def test_robust_trimmed_mean_rejects_feature_outlier():
    observations = [
        _obs(3, [1.0, 0.0], image_id="a"),
        _obs(3, [1.1, 0.0], image_id="b"),
        _obs(3, [9.0, 0.0], image_id="outlier"),
    ]

    mean_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="mean", min_observations=2),
    )
    robust_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="robust_trimmed_mean", min_observations=2, trim_fraction=0.34),
    )

    assert mean_bank.tracks[3].mean_feature[0] > 3.0
    np.testing.assert_allclose(robust_bank.tracks[3].mean_feature, np.asarray([1.05, 0.0], dtype=np.float32), atol=1e-4)


def test_view_consistent_aggregation_keeps_mutually_consistent_observations():
    observations = [
        _obs(7, [1.0, 0.0], image_id="a"),
        _obs(7, [0.9, 0.1], image_id="b"),
        _obs(7, [-1.0, 0.0], image_id="bad"),
    ]

    bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="view_consistent", min_observations=2, view_consistent_keep=2),
    )

    feature = bank.tracks[7].mean_feature
    assert feature[0] > 0.8
    assert abs(float(feature[1])) < 0.1
    assert bank.tracks[7].observation_count == 2


def test_ulf_geometry_weighted_aggregation_combines_geometry_and_consensus():
    observations = [
        _obs(11, [1.0, 0.0], utility=1.0, image_id="a"),
        _obs(11, [0.95, 0.05], utility=1.0, image_id="b"),
        _obs(11, [-1.0, 0.0], utility=8.0, image_id="bad_geometry_outlier"),
    ]

    geometry_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="geometry_weighted", min_observations=2, l2_normalize_observations=True),
    )
    ulf_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="ulf_geometry_weighted", min_observations=2),
    )

    assert geometry_bank.tracks[11].mean_feature[0] < 0.0
    assert ulf_bank.tracks[11].mean_feature[0] > 0.8
    assert abs(float(ulf_bank.tracks[11].mean_feature[1])) < 0.1
    assert ulf_bank.tracks[11].observation_count == 2


def test_projected_observation_bank_cli_accepts_ulf_geometry_weighted_method(tmp_path):
    args = parse_projected_bank_args(
        [
            "--track_observations",
            "tracks.jsonl",
            "--token_manifest",
            "manifest.json",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--method",
            "ulf_geometry_weighted",
            "--output_index",
            str(tmp_path / "bank.npz"),
            "--summary_json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert args.method == "ulf_geometry_weighted"


def test_geometric_median_and_medoid_are_robust_to_outlier():
    observations = [
        _obs(8, [1.0, 0.0], image_id="a"),
        _obs(8, [1.2, 0.0], image_id="b"),
        _obs(8, [9.0, 0.0], image_id="outlier"),
    ]

    median_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="geometric_median", min_observations=2),
    )
    medoid_bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="medoid", min_observations=2),
    )

    assert median_bank.tracks[8].mean_feature[0] == pytest.approx(1.2, abs=0.25)
    assert min(abs(float(medoid_bank.tracks[8].mean_feature[0]) - 1.0), abs(float(medoid_bank.tracks[8].mean_feature[0]) - 1.2)) < 1e-6
    assert medoid_bank.tracks[8].observation_count == 3


def test_multi_prototype_builder_separates_descriptor_modes_deterministically():
    observations = [
        _obs(8, [1.0, 0.0], image_id="a"),
        _obs(8, [0.98, 0.02], image_id="b"),
        _obs(8, [0.0, 1.0], image_id="c"),
        _obs(8, [0.02, 0.98], image_id="d"),
        _obs(9, [1.0, 0.0], image_id="e"),
        _obs(9, [0.9, 0.1], image_id="f"),
        _obs(9, [0.8, 0.2], image_id="g"),
    ]
    config = LandmarkViewClusteringConfig(
        max_prototypes_per_track=2,
        min_observations_per_prototype=2,
        iterations=8,
    )

    bank_a = build_multi_prototype_track_bank(
        observations,
        aggregation=LandmarkAggregationConfig(method="mean", min_observations=2, l2_normalize_observations=True),
        clustering=config,
    )
    bank_b = build_multi_prototype_track_bank(
        observations,
        aggregation=LandmarkAggregationConfig(method="mean", min_observations=2, l2_normalize_observations=True),
        clustering=config,
    )

    assert [(item.track_id, item.prototype_id) for item in bank_a.prototypes] == [(8, 0), (8, 1), (9, 0)]
    assert [item.observation_count for item in bank_a.prototypes] == [2, 2, 3]
    for left, right in zip(bank_a.prototypes, bank_b.prototypes):
        np.testing.assert_allclose(left.feature, right.feature)
    track8 = [item.feature for item in bank_a.prototypes if item.track_id == 8]
    assert abs(float(np.dot(track8[0], track8[1]))) < 0.1


def test_random_observation_aggregation_is_seeded_and_deterministic():
    observations = [
        _obs(9, [1.0, 0.0], image_id="a"),
        _obs(9, [2.0, 0.0], image_id="b"),
        _obs(9, [3.0, 0.0], image_id="c"),
    ]
    config = LandmarkAggregationConfig(method="random_observation", min_observations=2, seed=11)

    bank_a = aggregate_landmark_features(observations, config)
    bank_b = aggregate_landmark_features(observations, config)

    np.testing.assert_array_equal(bank_a.tracks[9].mean_feature, bank_b.tracks[9].mean_feature)
    assert bank_a.tracks[9].observation_count == 3


def test_landmark_feature_diagnostics_report_split_stability_and_retrieval():
    observations = []
    for track_id, center in [(1, np.asarray([1.0, 0.0])), (2, np.asarray([0.0, 1.0]))]:
        for obs_idx in range(4):
            observations.append(
                _obs(
                    track_id,
                    center + np.asarray([0.01 * obs_idx, -0.01 * obs_idx], dtype=np.float32),
                    image_id=f"t{track_id}_{obs_idx}",
                )
            )

    config = LandmarkAggregationConfig(method="mean", min_observations=2, l2_normalize_observations=True)
    stability = evaluate_landmark_split_stability(observations, config, seed=5)
    retrieval = evaluate_landmark_observation_retrieval(observations, config, top_k=(1, 2), seed=5)

    assert stability.common_track_count == 2
    assert stability.mean_cosine_similarity > 0.99
    assert retrieval.query_count == 2
    assert retrieval.recall_at_k[1] == pytest.approx(1.0)
    assert retrieval.recall_at_k[2] == pytest.approx(1.0)


def test_track_sampling_can_use_center_weighted_observation_utility(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _write_token(tmp_path, "img_a", np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)),
        )
    )
    observations = [
        ColmapTrackObservation(
            track_id=1,
            image_id="img_a",
            point2d_idx=0,
            xy=(0.0, 0.0),
            xyz=np.zeros(3),
            track_length=2,
            reprojection_error=0.1,
            camera_id=1,
            image_width=3,
            image_height=3,
        ),
        ColmapTrackObservation(
            track_id=1,
            image_id="img_a",
            point2d_idx=1,
            xy=(1.0, 1.0),
            xyz=np.zeros(3),
            track_length=2,
            reprojection_error=0.1,
            camera_id=1,
            image_width=3,
            image_height=3,
        ),
    ]

    sampled = sample_token_track_observations(
        observations,
        manifest,
        layer_name="radio_final",
        utility_mode="inverse_reprojection_center",
    )

    assert sampled[1].utility > sampled[0].utility


def test_track_sampling_bilinear_interpolates_token_map(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _write_token(
                tmp_path,
                "img_a",
                np.asarray([[[1.0, 3.0], [5.0, 7.0]], [[10.0, 30.0], [50.0, 70.0]]], dtype=np.float32),
            ),
        )
    )
    observations = [
        ColmapTrackObservation(
            track_id=2,
            image_id="img_a",
            point2d_idx=0,
            xy=(1.0, 1.0),
            xyz=np.zeros(3),
            track_length=2,
            reprojection_error=0.1,
            camera_id=1,
            image_width=3,
            image_height=3,
        )
    ]

    sampled = sample_token_track_observations(
        observations,
        manifest,
        layer_name="radio_final",
        sample_mode="bilinear",
    )

    np.testing.assert_allclose(sampled[0].feature, np.asarray([4.0, 40.0], dtype=np.float32), atol=1e-6)


def test_track_sampling_can_use_view_consistency_utility(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _write_token(tmp_path, "img_a", np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)),
            _write_token(tmp_path, "img_b", np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)),
            _write_token(tmp_path, "img_c", np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)),
        )
    )
    observations = [
        ColmapTrackObservation(1, "img_a", 0, (0.0, 0.0), np.zeros(3), 3, 0.1, 1, 1, 1, viewing_ray=np.asarray([1.0, 0.0, 0.0])),
        ColmapTrackObservation(1, "img_b", 0, (0.0, 0.0), np.zeros(3), 3, 0.1, 1, 1, 1, viewing_ray=np.asarray([1.0, 0.0, 0.0])),
        ColmapTrackObservation(1, "img_c", 0, (0.0, 0.0), np.zeros(3), 3, 0.1, 1, 1, 1, viewing_ray=np.asarray([-1.0, 0.0, 0.0])),
    ]

    sampled = sample_token_track_observations(
        observations,
        manifest,
        layer_name="radio_final",
        utility_mode="view_consistency",
    )

    assert sampled[0].utility == pytest.approx(sampled[1].utility)
    assert sampled[0].utility > sampled[2].utility


def _write_token(tmp_path, image_id, feature):
    path = tmp_path / f"{image_id}.npz"
    write_npz_token_record(path, {"radio_final": np.asarray(feature, dtype=np.float16)})
    return TokenBankRecord(
        image_id=image_id,
        token_path=path,
        layers=(TokenLayerSpec("radio_final", "synthetic", "final", feature.shape[0], 1),),
        split="train",
        scene="Synthetic",
        checksum=compute_file_sha256(path),
    )


def test_build_raw_vfm_landmark_bank_cli_writes_bank_and_summary(tmp_path):
    manifest = TokenBankManifest(
        records=(
            _write_token(tmp_path, "img_a", np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)),
            _write_token(tmp_path, "img_b", np.asarray([[[0.0]], [[1.0]]], dtype=np.float32)),
            _write_token(tmp_path, "img_c", np.asarray([[[0.8]], [[0.2]]], dtype=np.float32)),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    track_path = tmp_path / "tracks.jsonl"
    rows = [
        ColmapTrackObservation(
            track_id=10,
            image_id="img_a",
            point2d_idx=0,
            xy=(0.0, 0.0),
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track_length=2,
            reprojection_error=0.1,
            camera_id=1,
            image_width=1,
            image_height=1,
        ),
        ColmapTrackObservation(
            track_id=10,
            image_id="img_b",
            point2d_idx=0,
            xy=(0.0, 0.0),
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track_length=2,
            reprojection_error=0.2,
            camera_id=1,
            image_width=1,
            image_height=1,
        ),
        ColmapTrackObservation(
            track_id=10,
            image_id="img_c",
            point2d_idx=0,
            xy=(0.0, 0.0),
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track_length=3,
            reprojection_error=0.1,
            camera_id=1,
            image_width=1,
            image_height=1,
        ),
    ]
    track_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "track_id": row.track_id,
                    "image_id": row.image_id,
                    "point2d_idx": row.point2d_idx,
                    "xy": list(row.xy),
                    "xyz": row.xyz.tolist(),
                    "track_length": row.track_length,
                    "reprojection_error": row.reprojection_error,
                    "camera_id": row.camera_id,
                    "image_width": row.image_width,
                    "image_height": row.image_height,
                },
                sort_keys=True,
            )
            for row in rows
        )
        + "\n"
    )
    output_bank = tmp_path / "bank.npz"
    summary_path = tmp_path / "summary.json"

    build_raw_landmark_bank_cli_main(
        [
            "--track_observations",
            str(track_path),
            "--token_manifest",
            str(manifest_path),
            "--layer_name",
            "radio_final",
            "--method",
            "geometry_weighted",
            "--min_observations",
            "2",
            "--output_bank",
            str(output_bank),
            "--summary_json",
            str(summary_path),
        ]
    )

    bank = load_selected_track_bank_npz(output_bank)
    summary = json.loads(summary_path.read_text())
    assert len(bank.tracks) == 1
    assert bank.feature_dim == 2
    assert summary["aggregation"]["method"] == "geometry_weighted"
    assert summary["mapability"]["track_count"] == 1
    assert summary["diagnostics"]["retrieval"]["query_count"] == 1


def test_sampled_track_observation_cache_round_trips_features_and_metadata(tmp_path):
    observations = [
        _obs(1, [1.0, 2.0], utility=0.5, image_id="img_a"),
        _obs(2, [3.0, 4.0], utility=2.0, image_id="img_b", geometry_valid=False),
    ]
    cache_path = tmp_path / "sampled_observations.npz"

    save_sampled_track_observations_npz(
        observations,
        cache_path,
        metadata={"sample_mode": "bilinear", "utility_mode": "inverse_reprojection_center_view"},
    )
    loaded, metadata = load_sampled_track_observations_npz(cache_path)

    assert metadata["sample_mode"] == "bilinear"
    assert metadata["utility_mode"] == "inverse_reprojection_center_view"
    assert [obs.track_id for obs in loaded] == [1, 2]
    assert [obs.image_id for obs in loaded] == ["img_a", "img_b"]
    assert [obs.geometry_valid for obs in loaded] == [True, False]
    np.testing.assert_allclose(loaded[0].feature, np.asarray([1.0, 2.0], dtype=np.float32))
    assert loaded[1].utility == pytest.approx(2.0)


def test_build_raw_vfm_landmark_bank_cli_reuses_sampled_observation_cache(tmp_path):
    cache_path = tmp_path / "sampled_cache.npz"
    save_sampled_track_observations_npz(
        [
            _obs(3, [1.0, 0.0], image_id="img_a"),
            _obs(3, [0.8, 0.2], image_id="img_b"),
            _obs(3, [0.9, 0.1], image_id="img_c"),
        ],
        cache_path,
        metadata={"source": "unit-test"},
    )
    output_bank = tmp_path / "bank.npz"
    summary_path = tmp_path / "summary.json"

    build_raw_landmark_bank_cli_main(
        [
            "--sampled_observation_cache",
            str(cache_path),
            "--method",
            "mean",
            "--min_observations",
            "2",
            "--output_bank",
            str(output_bank),
            "--summary_json",
            str(summary_path),
        ]
    )

    bank = load_selected_track_bank_npz(output_bank)
    summary = json.loads(summary_path.read_text())
    assert len(bank.tracks) == 1
    assert summary["sampled_observation_count"] == 3
    assert summary["input_files"]["sampled_observation_cache"]["path"] == str(cache_path)
    assert summary["sampled_observation_cache_metadata"]["source"] == "unit-test"


def test_visualize_raw_vfm_landmark_bank_cli_writes_pca_and_variance_ply(tmp_path):
    observations = [
        _obs(1, [1.0, 0.0], image_id="a"),
        _obs(1, [1.0, 0.1], image_id="b"),
        _obs(2, [0.0, 1.0], image_id="c"),
        _obs(2, [0.1, 1.0], image_id="d"),
    ]
    bank = aggregate_landmark_features(
        observations,
        LandmarkAggregationConfig(method="mean", min_observations=2),
    )
    bank_path = tmp_path / "bank.npz"
    from feature_extract.vfm.map_lifting import save_selected_track_bank_npz

    save_selected_track_bank_npz(bank, bank_path)
    track_path = tmp_path / "tracks.jsonl"
    track_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "track_id": 1,
                        "image_id": "a",
                        "point2d_idx": 0,
                        "xy": [0.0, 0.0],
                        "xyz": [0.0, 0.0, 0.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                    }
                ),
                json.dumps(
                    {
                        "track_id": 2,
                        "image_id": "c",
                        "point2d_idx": 0,
                        "xy": [0.0, 0.0],
                        "xyz": [1.0, 0.0, 0.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                    }
                ),
            ]
        )
        + "\n"
    )
    pca_ply = tmp_path / "pca.ply"
    variance_ply = tmp_path / "variance.ply"
    summary_path = tmp_path / "summary.json"

    visualize_raw_landmark_bank_cli_main(
        [
            "--bank",
            str(bank_path),
            "--track_observations",
            str(track_path),
            "--output_pca_ply",
            str(pca_ply),
            "--output_variance_ply",
            str(variance_ply),
            "--summary_json",
            str(summary_path),
        ]
    )

    assert "element vertex 2" in pca_ply.read_text()
    assert "element vertex 2" in variance_ply.read_text()
    summary = json.loads(summary_path.read_text())
    assert summary["visualized_track_count"] == 2
    assert summary["feature_dim"] == 2
