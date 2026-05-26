import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_track_utility_selector import main as train_track_utility_cli_main
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.track_utility_training import (
    TrackUtilityTrainingConfig,
    _build_track_supervision_groups,
    _pairwise_track_separation,
    run_track_utility_training,
)
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _synthetic_track_observations(track_count=8, obs_per_track=3, channels=6):
    rng = np.random.default_rng(17)
    observations = []
    for track_id in range(track_count):
        signal = rng.normal(size=(2,)).astype(np.float32)
        for obs_idx in range(obs_per_track):
            feature = rng.normal(scale=0.1, size=(channels,)).astype(np.float32)
            feature[:2] = signal + rng.normal(scale=0.02, size=(2,))
            feature[2:] = rng.normal(scale=0.5, size=(channels - 2,))
            feature[2] = 1.0 if obs_idx == 0 else -1.0
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    image_id=f"image_{obs_idx}.png",
                    feature=feature,
                    visible=True,
                    geometry_valid=True,
                    utility=1.0 if obs_idx == 0 else 0.2,
                )
            )
    return observations


def test_track_utility_training_learns_track_separation_and_utility_targets():
    observations = _synthetic_track_observations()

    run = run_track_utility_training(
        observations,
        TrackUtilityTrainingConfig(
            steps=120,
            batch_size=8,
            output_dim=4,
            group_size=2,
            seed=3,
            device="cpu",
            lr=0.03,
            utility_weight=1.0,
            contrastive_weight=1.0,
            consistency_weight=0.5,
        ),
    )

    assert run.summary.track_count == 8
    assert run.summary.initial_loss > run.summary.final_loss
    assert run.summary.final_positive_similarity > run.summary.final_negative_similarity + 0.2
    assert run.summary.final_utility_target_correlation > 0.5


def test_track_utility_training_can_warm_start_selector_checkpoint(tmp_path):
    observations = _synthetic_track_observations(track_count=4)
    checkpoint = tmp_path / "warm.pt"
    selector = LocalizableFeatureSelector(input_dim=6, output_dim=4, group_size=2)
    with torch.no_grad():
        selector.group_logits.fill_(3.0)
    torch.save(selector.state_dict(), checkpoint)

    run = run_track_utility_training(
        observations,
        TrackUtilityTrainingConfig(
            steps=1,
            batch_size=4,
            output_dim=4,
            group_size=2,
            seed=1,
            device="cpu",
            lr=1e-12,
            init_checkpoint=str(checkpoint),
        ),
    )

    assert torch.allclose(run.selector.group_logits, torch.full_like(run.selector.group_logits, 3.0), atol=1e-6)


def test_track_supervision_groups_reject_single_observation_tracks():
    observations = _synthetic_track_observations(track_count=2, obs_per_track=1)

    with pytest.raises(ValueError, match="at least two observations"):
        _build_track_supervision_groups(observations, min_observations=2)


def test_track_supervision_utility_targets_use_global_normalization():
    observations = [
        TrackObservation(1, "a", np.ones(4, dtype=np.float32), True, True, utility=1.0),
        TrackObservation(1, "b", np.ones(4, dtype=np.float32), True, True, utility=1.0),
        TrackObservation(2, "c", -np.ones(4, dtype=np.float32), True, True, utility=10.0),
        TrackObservation(2, "d", -np.ones(4, dtype=np.float32), True, True, utility=10.0),
    ]

    groups = _build_track_supervision_groups(observations, min_observations=2)

    assert groups[0].utility_targets.tolist() == pytest.approx([0.0, 0.0])
    assert groups[1].utility_targets.tolist() == pytest.approx([1.0, 1.0])


def test_pairwise_track_separation_reports_positive_above_negative_for_clean_features():
    observations = _synthetic_track_observations(track_count=5, obs_per_track=2)
    groups = _build_track_supervision_groups(observations, min_observations=2)
    selector = LocalizableFeatureSelector(input_dim=6, output_dim=2, group_size=2)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0

    positive, negative = _pairwise_track_separation(selector, groups, torch.device("cpu"))

    assert positive > negative


def _write_track_token_fixture(tmp_path):
    records = []
    lines = []
    for track_id, x in [(0, 0.0), (1, 9.0)]:
        for obs_idx in range(2):
            image_id = f"image_{track_id}_{obs_idx}.png"
            feature = np.zeros((6, 1, 2), dtype=np.float32)
            feature[0, 0, 0] = 1.0 if track_id == 0 else -1.0
            feature[1, 0, 0] = 0.5
            feature[0, 0, 1] = -feature[0, 0, 0]
            path = tmp_path / f"{image_id}.npz"
            write_npz_token_record(path, {"radio_final": feature})
            records.append(
                TokenBankRecord(
                    image_id=image_id,
                    token_path=path,
                    layers=(TokenLayerSpec("radio_final", "synthetic", "final", 6, 1),),
                    split="train",
                    scene="Synthetic",
                    checksum=compute_file_sha256(path),
                )
            )
            lines.append(
                json.dumps(
                    {
                        "track_id": track_id,
                        "image_id": image_id,
                        "point2d_idx": obs_idx,
                        "xy": [x, 0.0],
                        "xyz": [float(track_id), 0.0, 1.0],
                        "track_length": 2,
                        "reprojection_error": 0.1 if obs_idx == 0 else 1.0,
                        "camera_id": 1,
                        "image_width": 10,
                        "image_height": 1,
                    }
                )
            )
    manifest = TokenBankManifest(records=tuple(records))
    manifest_path = tmp_path / "manifest.json"
    tracks_path = tmp_path / "tracks.jsonl"
    manifest.to_json(manifest_path)
    tracks_path.write_text("\n".join(lines) + "\n")
    return manifest_path, tracks_path


def test_train_track_utility_selector_cli_writes_report_and_checkpoint(tmp_path):
    manifest_path, tracks_path = _write_track_token_fixture(tmp_path)
    output = tmp_path / "report.json"
    checkpoint = tmp_path / "selector.pt"

    train_track_utility_cli_main(
        [
            "--track_observations",
            str(tracks_path),
            "--token_manifest",
            str(manifest_path),
            "--layer_name",
            "radio_final",
            "--steps",
            "20",
            "--batch_size",
            "2",
            "--output_dim",
            "4",
            "--group_size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["result"]["track_count"] == 2
    assert payload["inputs"]["input_files"]["track_observations"]["sha256"]
    assert checkpoint.exists()
