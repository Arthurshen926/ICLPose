from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_USAGE,
    build_global_context_candidate_probe_features,
)


def _source_metadata() -> str:
    return json.dumps(
        {
            "format": "multiscale_candidate_probe_features_v1",
            "contains_ground_truth": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": False,
            "support_view_selection": "fixed_maplet_coverage_rank_with_real_observation_fallback_v1",
            "proposals_sha256": "test-proposals-sha",
        }
    )


def _radio_metadata() -> str:
    return json.dumps(
        {
            "format": "radio_final_context_pca_v1",
            "pca_fit_scope": "mapping_train_images_only",
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "normalization": "input_row_l2_centered_pca_output_row_l2",
        }
    )


def test_builder_exports_only_fixed_candidate_support_global_context(tmp_path) -> None:
    layout = tmp_path / "layout.npz"
    radio = tmp_path / "radio.npz"
    output = tmp_path / "global_features.npz"
    summary = tmp_path / "summary.json"
    view_valid = np.asarray(
        [
            [[True, False], [True, True]],
            [[True, True], [False, True]],
        ],
        dtype=bool,
    )
    support_ids = np.asarray(
        [
            [["s1.png", ""], ["s2.png", "s3.png"]],
            [["s2.png", "s1.png"], ["", "s3.png"]],
        ]
    )
    source_features = np.zeros((2, 2, 2, 3), dtype=np.float32)
    source_features[..., 0] = 0.1
    source_features[..., 1] = 0.2
    source_features[..., 2] = 0.3
    np.savez(
        layout,
        source_row_indices=np.asarray([3, 7], dtype=np.int64),
        query_ids=np.asarray(["q1.png", "q2.png"]),
        split_names=np.asarray(["train", "validation"]),
        xy=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[11, 12], [13, 14]], dtype=np.int64),
        candidate_canonical_rows=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        candidate_features=source_features,
        candidate_view_valid=view_valid,
        candidate_support_image_ids=support_ids,
        candidate_support_coverage_counts=np.ones((2, 2, 2), dtype=np.int32),
        feature_names=np.asarray(
            [
                "radio_final_anchor_cosine",
                "radio_intermediate_anchor_cosine",
                "alike_anchor_cosine",
            ]
        ),
        metadata_json=np.asarray(_source_metadata()),
    )
    np.savez(
        radio,
        image_ids=np.asarray(["q1.png", "q2.png", "s1.png", "s2.png", "s3.png"]),
        global_descriptors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
        metadata_json=np.asarray(_radio_metadata()),
    )

    result = build_global_context_candidate_probe_features(
        frozen_layout_features=layout,
        radio_final_context_cache=radio,
        output=output,
        summary_json=summary,
        force=False,
    )

    assert result["protocol"]["image_retrieval_or_submap_used"] is False
    assert result["protocol"]["hard_image_retrieval_or_candidate_reselection"] is False
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        features = np.asarray(data["candidate_features"])
        assert data["feature_names"].tolist() == [
            "radio_final_anchor_cosine",
            "radio_intermediate_anchor_cosine",
            "alike_anchor_cosine",
            "radio_final_global_support_image_cosine",
        ]
        np.testing.assert_allclose(features[..., :3][view_valid], source_features[view_valid])
        np.testing.assert_allclose(features[0, 0, 0, -1], 1.0)
        np.testing.assert_allclose(features[0, 1, 0, -1], 0.0)
        np.testing.assert_allclose(features[0, 1, 1, -1], -1.0)
        np.testing.assert_allclose(features[1, 0, 0, -1], 1.0)
        np.testing.assert_allclose(features[1, 0, 1, -1], 0.0)
        assert np.all(features[~view_valid] == 0.0)
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["soft_global_context_factor"] is True
        assert metadata["global_context_usage"] == GLOBAL_CONTEXT_USAGE
        assert metadata["global_context_hard_retrieval_or_candidate_reselection"] is False
        assert metadata["whole_image_summary_or_global_used"] is True
        assert metadata["image_retrieval_or_submap_used"] is False
        assert metadata["proposals_sha256"] == "test-proposals-sha"
