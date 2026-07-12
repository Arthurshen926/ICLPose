import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.map_lifting import aggregate_selected_tracks
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)
from feature_extract.vfm import track_feature_sampling
from feature_extract.vfm.track_feature_sampling import sample_token_track_observations


def _token_manifest(tmp_path):
    feature = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    records = []
    for frame in ("frame0001", "frame0002"):
        token_path = tmp_path / f"seq1__{frame}.npz"
        write_npz_token_record(token_path, {"radio_final": feature})
        records.append(
            TokenBankRecord(
                image_id=f"seq1/{frame}.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "c-radio_v4-h", "final", 2, 16),),
                split="train",
                scene="Synthetic",
                checksum=compute_file_sha256(token_path),
            )
        )
    return TokenBankManifest(
        records=tuple(records)
    )


def test_sample_token_track_observations_uses_relative_colmap_coordinates(tmp_path):
    manifest = _token_manifest(tmp_path)
    observations = [
        ColmapTrackObservation(
            track_id=7,
            image_id="seq1/frame0001.png",
            point2d_idx=0,
            xy=(99.0, 79.0),
            xyz=np.asarray([0.0, 0.0, 1.0]),
            track_length=2,
            reprojection_error=0.1,
            camera_id=1,
            image_width=100,
            image_height=80,
        )
    ]

    sampled = sample_token_track_observations(observations, manifest, layer_name="radio_final")

    assert len(sampled) == 1
    assert sampled[0].track_id == 7
    assert sampled[0].feature.tolist() == [4.0, 40.0]
    assert sampled[0].feature.base is None


@pytest.mark.parametrize("sample_mode", ["nearest", "bilinear"])
def test_vectorized_feature_sampling_matches_scalar_path(sample_mode):
    rng = np.random.default_rng(7)
    feature_map = rng.normal(size=(5, 7, 11)).astype(np.float32)
    xy = np.asarray([[0.0, 0.0], [31.25, 19.75], [63.0, 47.0]], dtype=np.float64)
    widths = np.asarray([64, 64, 64], dtype=np.int64)
    heights = np.asarray([48, 48, 48], dtype=np.int64)
    expected = np.stack(
        [
            track_feature_sampling._sample_feature_vector(
                feature_map,
                tuple(coordinate),
                int(width),
                int(height),
                sample_mode,
            )
            for coordinate, width, height in zip(xy, widths, heights)
        ],
        axis=0,
    )

    actual = track_feature_sampling._sample_feature_vectors(
        feature_map,
        xy,
        widths,
        heights,
        sample_mode,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_jsonl_loader_deduplicates_track_image_observations_deterministically(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    common = {
        "track_id": 7,
        "image_id": "seq1/frame0001.png",
        "xyz": [0.0, 0.0, 1.0],
        "track_length": 2,
        "camera_id": 1,
        "image_width": 100,
        "image_height": 80,
    }
    path.write_text(
        "\n".join(
            [
                json.dumps({**common, "point2d_idx": 5, "xy": [9.0, 9.0], "reprojection_error": 0.2}),
                json.dumps({**common, "point2d_idx": 8, "xy": [8.0, 8.0], "reprojection_error": 0.1}),
                json.dumps({**common, "point2d_idx": 3, "xy": [3.0, 3.0], "reprojection_error": 0.1}),
            ]
        )
        + "\n"
    )

    deduplicated = track_feature_sampling.load_colmap_track_observations_jsonl(path)
    raw = track_feature_sampling.load_colmap_track_observations_jsonl(
        path,
        deduplicate_track_images=False,
    )

    assert len(deduplicated) == 1
    assert deduplicated[0].point2d_idx == 3
    assert len(raw) == 3


def test_jsonl_loader_filters_tracks_before_deduplication(tmp_path):
    path = tmp_path / "tracks.jsonl"
    rows = []
    for track_id in (7, 8):
        for point2d_idx, error in ((2, 0.2), (1, 0.1)):
            rows.append(
                {
                    "track_id": track_id,
                    "image_id": "seq1/frame0001.png",
                    "point2d_idx": point2d_idx,
                    "xy": [float(point2d_idx), float(point2d_idx)],
                    "xyz": [0.0, 0.0, 1.0],
                    "track_length": 2,
                    "reprojection_error": error,
                    "camera_id": 1,
                    "image_width": 100,
                    "image_height": 80,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    selected = track_feature_sampling.load_colmap_track_observations_jsonl(path, track_ids={8})

    assert len(selected) == 1
    assert selected[0].track_id == 8
    assert selected[0].point2d_idx == 1


def test_build_track_feature_bank_from_colmap_tokens_cli(tmp_path):
    manifest = _token_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    tracks_jsonl = tmp_path / "tracks.jsonl"
    tracks_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0001.png",
                        "point2d_idx": 0,
                        "xy": [0.0, 0.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0002.png",
                        "point2d_idx": 1,
                        "xy": [99.0, 79.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
            ]
        )
        + "\n"
    )
    output_bank = tmp_path / "bank.npz"
    summary_json = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_track_feature_bank_from_tokens",
            "--track_observations",
            str(tracks_jsonl),
            "--token_manifest",
            str(manifest_path),
            "--layer_name",
            "radio_final",
            "--min_observations",
            "2",
            "--output_bank",
            str(output_bank),
            "--summary_json",
            str(summary_json),
        ],
        check=True,
    )

    summary = json.loads(summary_json.read_text())
    assert summary["track_count"] == 1
    assert summary["feature_dim"] == 2
    assert output_bank.exists()


def test_build_track_feature_bank_cli_supports_first_channel_transform(tmp_path):
    manifest = _token_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    tracks_jsonl = tmp_path / "tracks.jsonl"
    tracks_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0001.png",
                        "point2d_idx": 0,
                        "xy": [0.0, 0.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0002.png",
                        "point2d_idx": 1,
                        "xy": [99.0, 79.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
            ]
        )
        + "\n"
    )
    output_bank = tmp_path / "bank_first.npz"
    summary_json = tmp_path / "summary_first.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_track_feature_bank_from_tokens",
            "--track_observations",
            str(tracks_jsonl),
            "--token_manifest",
            str(manifest_path),
            "--layer_name",
            "radio_final",
            "--transform",
            "first_channels",
            "--output_dim",
            "1",
            "--l2_normalize",
            "--min_observations",
            "2",
            "--output_bank",
            str(output_bank),
            "--summary_json",
            str(summary_json),
        ],
        check=True,
    )

    summary = json.loads(summary_json.read_text())
    assert summary["feature_dim"] == 1
    assert summary["transform"] == "first_channels"
    assert summary["l2_normalize"] is True


def test_build_track_feature_bank_cli_supports_selector_checkpoint(tmp_path):
    manifest = _token_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    tracks_jsonl = tmp_path / "tracks.jsonl"
    tracks_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0001.png",
                        "point2d_idx": 0,
                        "xy": [0.0, 0.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
                json.dumps(
                    {
                        "track_id": 7,
                        "image_id": "seq1/frame0002.png",
                        "point2d_idx": 1,
                        "xy": [99.0, 79.0],
                        "xyz": [0.0, 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1,
                        "camera_id": 1,
                        "image_width": 100,
                        "image_height": 80,
                    }
                ),
            ]
        )
        + "\n"
    )
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=1, group_size=1)
    with torch.no_grad():
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.bias.zero_()
    checkpoint = tmp_path / "selector.pt"
    torch.save(selector.state_dict(), checkpoint)
    output_bank = tmp_path / "bank_selector.npz"
    summary_json = tmp_path / "summary_selector.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.build_track_feature_bank_from_tokens",
            "--track_observations",
            str(tracks_jsonl),
            "--token_manifest",
            str(manifest_path),
            "--layer_name",
            "radio_final",
            "--transform",
            "selector",
            "--selector_checkpoint",
            str(checkpoint),
            "--selector_device",
            "cpu",
            "--min_observations",
            "2",
            "--output_bank",
            str(output_bank),
            "--summary_json",
            str(summary_json),
        ],
        check=True,
    )

    summary = json.loads(summary_json.read_text())
    bank = load_selected_track_bank_npz(output_bank)
    assert summary["feature_dim"] == 1
    assert summary["transform"] == "selector"
    assert summary["selector_output_dim"] == 1
    assert bank.tracks[7].mean_utility == pytest.approx(0.5)


def test_sampled_features_can_aggregate_to_selected_track_bank(tmp_path):
    manifest = _token_manifest(tmp_path)
    observations = [
        ColmapTrackObservation(7, "seq1/frame0001.png", 0, (0.0, 0.0), np.zeros(3), 2, 0.1, 1, 100, 80),
        ColmapTrackObservation(7, "seq1/frame0001.png", 1, (99.0, 79.0), np.zeros(3), 2, 0.1, 1, 100, 80),
    ]

    bank = aggregate_selected_tracks(
        sample_token_track_observations(observations, manifest, layer_name="radio_final"),
        min_observations=2,
    )

    assert len(bank) == 1
    assert bank.feature_dim == 2


def test_track_sampling_groups_observations_by_image(tmp_path, monkeypatch):
    manifest = _token_manifest(tmp_path)
    observations = [
        ColmapTrackObservation(7, "seq1/frame0001.png", 0, (0.0, 0.0), np.zeros(3), 2, 0.1, 1, 100, 80),
        ColmapTrackObservation(8, "seq1/frame0001.png", 1, (99.0, 79.0), np.zeros(3), 2, 0.1, 1, 100, 80),
    ]
    load_count = 0
    original_load_layer = track_feature_sampling._load_layer

    def counted_load_layer(path, layer_name):
        nonlocal load_count
        load_count += 1
        return original_load_layer(path, layer_name)

    monkeypatch.setattr(track_feature_sampling, "_load_layer", counted_load_layer)

    sampled = sample_token_track_observations(observations, manifest, layer_name="radio_final")

    assert len(sampled) == 2
    assert load_count == 1
