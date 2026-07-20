from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.fit_multiscale_context_attention_probe import (
    _MIXED_POINTS_CANDIDATE_INPUT,
    _load_candidate_prior_input,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_INTERMEDIATE,
)


def _write_mixed_points(path) -> None:
    metadata = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    np.savez_compressed(
        path,
        # Deliberately reverse NPZ order.  The fitter must use source IDs,
        # never incidental archive row order.
        source_point_ids=np.asarray([1, 0], dtype=np.int64),
        query_ids=np.asarray(["q1.png", "q0.png"]),
        split_names=np.asarray(["train", "validation"]),
        xy=np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        point_sources=np.asarray([POINT_SOURCE_ALIKE, POINT_SOURCE_RADIO_INTERMEDIATE]),
        source_detector_rows=np.asarray([7, -1], dtype=np.int64),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        candidate_bank_rows=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_track_ids=np.asarray([[101, 102], [201, 202]], dtype=np.int64),
        candidate_prototype_ids=np.zeros((2, 2), dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.5, 0.3], [0.6, 0.2]], dtype=np.float32),
        null_probabilities=np.asarray([0.2, 0.2], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_mixed_candidate_input_uses_explicit_dense_source_ids(tmp_path) -> None:
    points_path = tmp_path / "points.npz"
    _write_mixed_points(points_path)
    digest = file_sha256_short(points_path)
    source = _load_candidate_prior_input(
        contract={
            "candidate_input_kind": _MIXED_POINTS_CANDIDATE_INPUT,
            "candidate_input_lineage_sha256": digest,
            "proposals_sha256": digest,
        },
        proposals_path=None,
        base_overlay_path=None,
        verification_points_path=points_path,
    )

    assert source.kind == _MIXED_POINTS_CANDIDATE_INPUT
    # Source ID 0 was the second physical NPZ row.
    np.testing.assert_array_equal(source.candidate_track_ids[0], [201, 202])
    np.testing.assert_allclose(source.candidate_probabilities[0], [0.6, 0.2])
    np.testing.assert_allclose(source.null_probabilities, [0.2, 0.2])


def test_mixed_candidate_input_rejects_stale_contract_lineage(tmp_path) -> None:
    points_path = tmp_path / "points.npz"
    _write_mixed_points(points_path)

    with pytest.raises(ValueError, match="differ from frozen candidate lineage"):
        _load_candidate_prior_input(
            contract={
                "candidate_input_kind": _MIXED_POINTS_CANDIDATE_INPUT,
                "candidate_input_lineage_sha256": "stale",
                "proposals_sha256": "stale",
            },
            proposals_path=None,
            base_overlay_path=None,
            verification_points_path=points_path,
        )
