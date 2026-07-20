import json

import numpy as np
import pytest
import torch

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_overlay import (
    OVERLAY_FORMAT,
    build_independent_rgb_candidate_prior_overlay,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    fuse_candidate_log_likelihood_ratios,
)


def _write_base_and_evidence(tmp_path):
    base_path = tmp_path / "base.npz"
    evidence_path = tmp_path / "evidence.npz"
    tracks = np.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=np.int64)
    probabilities = np.asarray(
        [[0.10, 0.20, 0.30], [0.20, 0.10, 0.20], [0.25, 0.15, 0.10]],
        dtype=np.float32,
    )
    null = np.asarray([0.40, 0.50, 0.50], dtype=np.float32)
    np.savez_compressed(
        base_path,
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=null,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_prior_overlay_v1",
                    "proposals_sha256": "proposal-hash",
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                }
            )
        ),
    )
    rows = np.asarray([0, 2], dtype=np.int64)
    columns = np.asarray([[2, 0, 1], [1, 2, 0]], dtype=np.int64)
    selected_probability = np.take_along_axis(probabilities[rows], columns, axis=1)
    np.savez_compressed(
        evidence_path,
        selected_rows=rows,
        split_names=np.asarray(["validation", "validation"]),
        candidate_source_columns=columns,
        candidate_valid=np.ones_like(columns, dtype=bool),
        candidate_track_ids=np.take_along_axis(tracks[rows], columns, axis=1),
        candidate_prior_probabilities=selected_probability,
        unknown_probability=null[rows],
        metadata_json=np.asarray(
            json.dumps({"format": "candidate_evidence_v3", "proposals_sha256": "proposal-hash"})
        ),
    )
    return base_path, evidence_path, probabilities, null, rows, columns


def _write_prediction(path, *, evidence_path, rows, llr, measured):
    expected_hash = file_sha256_short(evidence_path)
    np.savez_compressed(
        path,
        evidence_row_indices=np.asarray(rows, dtype=np.int64),
        candidate_rgb_log_likelihood_ratios=np.asarray(llr, dtype=np.float32),
        candidate_rgb_measured=np.asarray(measured, dtype=bool),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "independent_rgb_candidate_predictions_v1",
                    "candidate_evidence_sha256": expected_hash,
                    "availability_evidence_sha256": expected_hash,
                    "checkpoint_sha256": "checkpoint-hash",
                    "prediction_splits": ["validation"],
                }
            )
        ),
    )


def test_rgb_overlay_maps_compact_candidates_back_to_source_columns(tmp_path) -> None:
    base, evidence, probability, null, rows, columns = _write_base_and_evidence(tmp_path)
    prediction = tmp_path / "prediction.npz"
    llr = np.asarray([[0.4, -0.2, 0.1]], dtype=np.float32)
    measured = np.asarray([[True, True, False]], dtype=bool)
    _write_prediction(
        prediction,
        evidence_path=evidence,
        rows=np.asarray([1], dtype=np.int64),
        llr=llr,
        measured=measured,
    )
    output = tmp_path / "overlay.npz"
    result = build_independent_rgb_candidate_prior_overlay(
        base_overlay_path=base,
        candidate_evidence_path=evidence,
        availability_evidence_path=evidence,
        prediction_paths=(prediction,),
        llr_weights=(0.5,),
        output_path=output,
    )

    with np.load(output, allow_pickle=False) as payload:
        fused = np.asarray(payload["candidate_probabilities"])
        fused_null = np.asarray(payload["null_probabilities"])
        metadata = json.loads(str(payload["metadata_json"].item()))
    expected, expected_null = fuse_candidate_log_likelihood_ratios(
        torch.from_numpy(probability[2, columns[1]][None]),
        torch.from_numpy(null[[2]]),
        torch.from_numpy(llr),
        measured_mask=torch.from_numpy(measured),
        candidate_valid=torch.from_numpy(np.ones_like(measured, dtype=bool)),
        evidence_weight=0.5,
    )
    expected_compact = expected.numpy()[0]
    np.testing.assert_allclose(fused[0], probability[0])
    np.testing.assert_allclose(fused_null, null)
    np.testing.assert_allclose(fused[2, columns[1]], expected_compact)
    np.testing.assert_allclose(expected_null.numpy(), null[[2]])
    np.testing.assert_allclose(fused.sum(axis=1) + fused_null, 1.0)
    assert metadata["format"] == OVERLAY_FORMAT
    assert metadata["updated_split_names"] == ["validation"]
    assert result["protocol"]["target_free"] is True


def test_rgb_overlay_rejects_stale_compact_prior(tmp_path) -> None:
    base, evidence, _probability, _null, rows, _columns = _write_base_and_evidence(tmp_path)
    with np.load(evidence, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata_json"}
        metadata = json.loads(str(payload["metadata_json"].item()))
    arrays["candidate_prior_probabilities"] = np.array(
        arrays["candidate_prior_probabilities"], copy=True
    )
    arrays["candidate_prior_probabilities"][0, 0] += 0.01
    stale = tmp_path / "stale_evidence.npz"
    np.savez_compressed(stale, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    prediction = tmp_path / "prediction.npz"
    _write_prediction(
        prediction,
        evidence_path=stale,
        rows=np.asarray([rows[0]], dtype=np.int64),
        llr=np.zeros((1, 3), dtype=np.float32),
        measured=np.ones((1, 3), dtype=bool),
    )
    with pytest.raises(ValueError, match="priors differ"):
        build_independent_rgb_candidate_prior_overlay(
            base_overlay_path=base,
            candidate_evidence_path=stale,
            availability_evidence_path=stale,
            prediction_paths=(prediction,),
            llr_weights=(1.0,),
            output_path=tmp_path / "never.npz",
        )


def test_rgb_overlay_mean_aggregation_preserves_declared_ensemble_weight(tmp_path) -> None:
    base, evidence, probability, null, rows, columns = _write_base_and_evidence(tmp_path)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    measured = np.asarray([[True, True, True]], dtype=bool)
    first_llr = np.asarray([[0.2, -0.4, 0.1]], dtype=np.float32)
    second_llr = np.asarray([[0.6, 0.2, -0.3]], dtype=np.float32)
    _write_prediction(
        first,
        evidence_path=evidence,
        rows=np.asarray([rows[0]], dtype=np.int64),
        llr=first_llr,
        measured=measured,
    )
    _write_prediction(
        second,
        evidence_path=evidence,
        rows=np.asarray([rows[0]], dtype=np.int64),
        llr=second_llr,
        measured=measured,
    )
    output = tmp_path / "mean_overlay.npz"
    build_independent_rgb_candidate_prior_overlay(
        base_overlay_path=base,
        candidate_evidence_path=evidence,
        availability_evidence_path=evidence,
        prediction_paths=(first, second),
        llr_weights=(0.25, 0.25),
        llr_aggregation="mean",
        output_path=output,
    )
    with np.load(output, allow_pickle=False) as payload:
        fused = np.asarray(payload["candidate_probabilities"])
        metadata = json.loads(str(payload["metadata_json"].item()))
    expected, _ = fuse_candidate_log_likelihood_ratios(
        torch.from_numpy(probability[0, columns[0]][None]),
        torch.from_numpy(null[[0]]),
        torch.from_numpy(0.5 * (first_llr + second_llr)),
        measured_mask=torch.from_numpy(measured),
        candidate_valid=torch.from_numpy(np.ones_like(measured, dtype=bool)),
        evidence_weight=0.25,
    )
    np.testing.assert_allclose(fused[0, columns[0]], expected.numpy()[0])
    assert metadata["llr_aggregation"] == "mean"
    assert [item["effective_llr_weight"] for item in metadata["prediction_artifacts"]] == [
        0.125,
        0.125,
    ]
