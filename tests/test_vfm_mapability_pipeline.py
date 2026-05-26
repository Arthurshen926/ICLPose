import json
import subprocess
import sys

import numpy as np

from feature_extract.vfm.map_lifting import (
    TrackObservation,
    aggregate_selected_tracks,
    load_selected_track_bank_npz,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.mapability_metrics import compare_track_bank_mapability


def test_mapability_comparison_prefers_lower_track_variance():
    selected = aggregate_selected_tracks(
        [
            TrackObservation(0, "a", np.array([1.0, 0.0]), True, True),
            TrackObservation(0, "b", np.array([0.95, 0.05]), True, True),
            TrackObservation(1, "a", np.array([0.0, 1.0]), True, True),
            TrackObservation(1, "b", np.array([0.05, 0.95]), True, True),
        ],
        min_observations=2,
    )
    noisy = aggregate_selected_tracks(
        [
            TrackObservation(0, "a", np.array([1.0, 0.0]), True, True),
            TrackObservation(0, "b", np.array([0.0, 1.0]), True, True),
            TrackObservation(1, "a", np.array([0.0, 1.0]), True, True),
            TrackObservation(1, "b", np.array([1.0, 0.0]), True, True),
        ],
        min_observations=2,
    )

    selected_report = compare_track_bank_mapability(selected, expected_track_count=2)
    noisy_report = compare_track_bank_mapability(noisy, expected_track_count=2)

    assert selected_report.coverage == 1.0
    assert selected_report.mean_track_variance < noisy_report.mean_track_variance
    assert selected_report.separability_ratio > noisy_report.separability_ratio


def test_mapability_comparison_supports_bounded_pairwise_sampling():
    observations = []
    for track_id in range(20):
        feature = np.zeros(4, dtype=np.float32)
        feature[track_id % 4] = 1.0
        observations.append(TrackObservation(track_id, "a", feature, True, True))
        observations.append(TrackObservation(track_id, "b", feature, True, True))
    bank = aggregate_selected_tracks(observations, min_observations=2)

    report = compare_track_bank_mapability(bank, expected_track_count=20, max_pairwise_tracks=5, seed=0)

    assert report.track_count == 20
    assert report.coverage == 1.0
    assert report.separability_ratio > 0.0


def test_report_track_bank_mapability_cli_writes_json(tmp_path):
    bank = aggregate_selected_tracks(
        [
            TrackObservation(0, "a", np.array([1.0, 0.0]), True, True),
            TrackObservation(0, "b", np.array([0.9, 0.1]), True, True),
            TrackObservation(1, "a", np.array([0.0, 1.0]), True, True),
            TrackObservation(1, "b", np.array([0.1, 0.9]), True, True),
        ],
        min_observations=2,
    )
    bank_path = tmp_path / "tracks.npz"
    output_json = tmp_path / "mapability.json"
    save_selected_track_bank_npz(bank, bank_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.report_track_bank_mapability",
            "--track_bank",
            str(bank_path),
            "--expected_track_count",
            "4",
            "--max_pairwise_tracks",
            "2",
            "--output_json",
            str(output_json),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text())
    assert payload["track_count"] == 2
    assert payload["expected_track_count"] == 4
    assert payload["coverage"] == 0.5
    assert payload["separability_ratio"] > 0.0


def test_synthetic_pipeline_writes_core_artifacts(tmp_path):
    output = tmp_path / "synthetic"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.run_synthetic_pipeline",
            "--output_dir",
            str(output),
            "--query_count",
            "16",
            "--candidates_per_query",
            "4",
            "--seed",
            "2",
        ],
        check=True,
    )

    manifest = json.loads((output / "token_manifest.json").read_text())
    reports = json.loads((output / "score_reports.json").read_text())
    bank = load_selected_track_bank_npz(output / "selected_tracks.npz")

    assert manifest["records"]
    assert (output / "candidate_bank.jsonl").exists()
    assert (output / "feature_utility.md").exists()
    assert len(bank) > 0
    assert all(track.observation_image_ids for track in bank.tracks.values())
    assert reports["selected_feature"]["mean_top1_acc"] > reports["metadata_only"]["mean_top1_acc"]
