import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_stage_c0_compressed_features import main
from feature_extract.vfm.feature_compression import (
    apply_feature_compression_to_channel_first,
    fit_feature_compression,
    slice_feature_compression,
)
from feature_extract.vfm.map_lifting import (
    SelectedTrackFeatureBank,
    TrackFeature,
    load_selected_track_bank_npz,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def test_feature_compression_fits_pca_and_channel_selectors() -> None:
    features = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )

    pca = fit_feature_compression(features, method="pca", output_dim=2, seed=0)
    projected = pca.apply_rows(features)
    assert projected.shape == (4, 2)
    assert np.allclose(projected.mean(axis=0), 0.0, atol=1e-5)

    variance = fit_feature_compression(features, method="channel_variance", output_dim=2, seed=0)
    assert set(variance.selected_channels.tolist()) == {0, 3}

    idf = fit_feature_compression(features, method="idf", output_dim=1, seed=0, idf_threshold=0.5)
    assert idf.selected_channels.tolist() == [2]

    fisher = fit_feature_compression(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        method="fisher",
        output_dim=1,
        labels=np.asarray([0, 0, 1, 1]),
    )
    assert fisher.selected_channels.tolist() == [0]

    sliced = slice_feature_compression(pca, 1)
    assert sliced.output_dim == 1
    assert sliced.apply_rows(features).shape == (4, 1)


def test_feature_compression_applies_to_channel_first_maps() -> None:
    feature_map = np.zeros((4, 2, 2), dtype=np.float32)
    feature_map[0] = 1.0
    feature_map[3] = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    transform = fit_feature_compression(
        np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0]], dtype=np.float32),
        method="first_channels",
        output_dim=2,
        seed=0,
    )

    compressed = apply_feature_compression_to_channel_first(feature_map, transform)

    assert compressed.shape == (2, 2, 2)
    np.testing.assert_allclose(compressed[0], feature_map[0])
    np.testing.assert_allclose(compressed[1], feature_map[1])


def test_stage_c0_compression_cli_writes_matching_query_and_landmark_banks(tmp_path: Path) -> None:
    query_feature = np.zeros((4, 1, 2), dtype=np.float32)
    query_feature[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_feature[:, 0, 1] = np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    token_path = tmp_path / "tokens" / "q.npz"
    token_path.parent.mkdir(parents=True)
    np.savez_compressed(token_path, radio_final=query_feature)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", 4, 16),),
                split="test",
                scene="Synthetic",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)

    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(
                    1,
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    np.ones((4,), dtype=np.float32),
                    3,
                    1.0,
                    ("ref_a.png",),
                ),
                2: TrackFeature(
                    2,
                    np.asarray([0.0, 0.0, 1.0, 4.0], dtype=np.float32),
                    np.ones((4,), dtype=np.float32) * 2.0,
                    4,
                    1.0,
                    ("ref_b.png",),
                ),
            },
            feature_dim=4,
        ),
        bank_path,
    )

    out_dir = tmp_path / "compressed"
    output_manifest = out_dir / "manifest.json"
    output_bank = out_dir / "bank.npz"
    summary_path = out_dir / "summary.json"
    transform_path = out_dir / "transform.npz"

    main(
        [
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--layer_name",
            "radio_final",
            "--method",
            "channel_variance",
            "--output_dim",
            "2",
            "--output_query_dir",
            str(out_dir / "tokens"),
            "--output_query_manifest",
            str(output_manifest),
            "--output_landmark_bank",
            str(output_bank),
            "--output_transform",
            str(transform_path),
            "--summary_json",
            str(summary_path),
            "--device",
            "cpu",
            "--batch_tokens",
            "2",
        ]
    )

    compressed_manifest = TokenBankManifest.from_json(output_manifest)
    compressed_bank = load_selected_track_bank_npz(output_bank)
    summary = json.loads(summary_path.read_text())
    with np.load(compressed_manifest.records[0].token_path) as data:
        compressed_query = data["radio_final"]

    assert compressed_manifest.records[0].layers[0].channels == 2
    assert compressed_query.shape == (2, 1, 2)
    assert compressed_bank.feature_dim == 2
    assert all(track.mean_feature.shape == (2,) for track in compressed_bank.tracks.values())
    assert summary["method"] == "channel_variance"
    assert summary["output_dim"] == 2
    assert summary["query_record_count"] == 1
    assert Path(summary["outputs"]["query_manifest"]).exists()
    assert transform_path.exists()


def test_stage_c0_compression_cli_supports_fisher_labels(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens" / "q.npz"
    token_path.parent.mkdir(parents=True)
    np.savez_compressed(token_path, radio_final=np.zeros((3, 1, 1), dtype=np.float32))
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", 3, 16),),
                split="test",
                scene="Synthetic",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(1, np.asarray([0.0, 0.0, 0.0], dtype=np.float32), np.ones((3,), dtype=np.float32), 2, 1.0),
                2: TrackFeature(2, np.asarray([1.0, 0.0, 0.0], dtype=np.float32), np.ones((3,), dtype=np.float32), 2, 1.0),
            },
            feature_dim=3,
        ),
        bank_path,
    )
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, np.asarray([0, 1], dtype=np.int64))
    out_dir = tmp_path / "fisher"

    main(
        [
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--method",
            "fisher",
            "--fit_labels_npy",
            str(labels_path),
            "--output_dim",
            "1",
            "--output_query_dir",
            str(out_dir / "tokens"),
            "--output_query_manifest",
            str(out_dir / "manifest.json"),
            "--output_landmark_bank",
            str(out_dir / "bank.npz"),
            "--output_transform",
            str(out_dir / "transform.npz"),
            "--summary_json",
            str(out_dir / "summary.json"),
        ]
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["method"] == "fisher"
    assert summary["selected_channels_head"] == [0]
