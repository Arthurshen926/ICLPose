from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_global_context_support8_candidate_probe_features import (
    ARTIFACT_FORMAT,
    GLOBAL_CONTEXT_USAGE,
    build_global_context_support8_candidate_probe_features,
    parse_args,
)


def _layout_metadata() -> str:
    return json.dumps(
        {
            "format": "multiscale_candidate_probe_features_v1",
            "contains_ground_truth": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "whole_image_summary_or_global_used": False,
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
        }
    )


def test_cli_parser_accepts_required_support8_inputs() -> None:
    args = parse_args(
        [
            "--frozen_layout_features",
            "layout.npz",
            "--maplet_support_index",
            "maplet.npz",
            "--radio_final_context_cache",
            "radio.npz",
            "--output",
            "output.npz",
            "--summary_json",
            "summary.json",
        ]
    )
    assert args.maplet_support_index == "maplet.npz"
    assert args.force is False


def test_builder_expands_only_predeclared_maplet_support_views(tmp_path) -> None:
    layout = tmp_path / "layout.npz"
    maplet = tmp_path / "maplet.npz"
    radio = tmp_path / "radio.npz"
    output = tmp_path / "support8_features.npz"
    summary = tmp_path / "summary.json"
    original_view_valid = np.ones((2, 2, 2), dtype=bool)
    source_features = np.zeros((2, 2, 2, 3), dtype=np.float32)
    np.savez(
        layout,
        source_row_indices=np.asarray([3, 7], dtype=np.int64),
        query_ids=np.asarray(["q1.png", "q2.png"]),
        split_names=np.asarray(["train", "validation"]),
        xy=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[11, 12], [13, 14]], dtype=np.int64),
        candidate_canonical_rows=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        candidate_features=source_features,
        candidate_view_valid=original_view_valid,
        candidate_support_image_ids=np.full((2, 2, 2), "s1.png"),
        candidate_support_coverage_counts=np.ones((2, 2, 2), dtype=np.int32),
        feature_names=np.asarray(
            [
                "radio_final_anchor_cosine",
                "radio_intermediate_anchor_cosine",
                "alike_anchor_cosine",
            ]
        ),
        metadata_json=np.asarray(_layout_metadata()),
    )
    np.savez(
        maplet,
        anchor_track_ids=np.asarray([11, 12, 13, 14], dtype=np.int64),
        support_image_ids=np.asarray(["s1.png", "s2.png", "s3.png"]),
        support_image_indices=np.asarray([[0, 1], [1, 2], [2, 0], [1, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[8, 7], [6, 5], [4, 3], [2, 0]], dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps({"format": "local_maplet_support_index_npz", "max_support_views": 2})
        ),
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

    result = build_global_context_support8_candidate_probe_features(
        frozen_layout_features=layout,
        maplet_support_index=maplet,
        radio_final_context_cache=radio,
        output=output,
        summary_json=summary,
        force=False,
    )

    assert result["support_view_count"] == 2
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        support_ids = np.asarray(data["candidate_support_image_ids"]).astype(str)
        features = np.asarray(data["candidate_features"])
        valid = np.asarray(data["candidate_view_valid"], dtype=bool)
        assert data["feature_names"].tolist() == ["radio_final_global_support_image_cosine"]
        assert support_ids[0, 0].tolist() == ["s1.png", "s2.png"]
        assert support_ids[0, 1].tolist() == ["s2.png", "s3.png"]
        assert support_ids[1, 1].tolist() == ["s2.png", ""]
        np.testing.assert_allclose(features[0, 0, :, 0], [1.0, 0.0])
        np.testing.assert_allclose(features[0, 1, :, 0], [0.0, -1.0])
        np.testing.assert_allclose(features[1, 0, :, 0], [0.0, 0.0])
        assert valid[1, 1].tolist() == [True, False]
        assert features[1, 1, 0, 0] == 1.0
        assert features[1, 1, 1, 0] == 0.0
        assert metadata["format"] == ARTIFACT_FORMAT
        assert metadata["global_context_usage"] == GLOBAL_CONTEXT_USAGE
        assert metadata["global_context_support_view_count"] == 2
        assert metadata["global_context_hard_retrieval_or_candidate_reselection"] is False
