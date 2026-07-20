from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT,
)
from feature_extract.tools.vfm.build_radio_intermediate_fixed_candidate_rerank import (
    ARTIFACT_FORMAT,
    build_radio_intermediate_fixed_candidate_rerank,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.landmark_hybrid import save_landmark_index_npz
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def test_rerank_export_keeps_final_candidate_identity_fixed(tmp_path) -> None:
    context_path = tmp_path / "context.npz"
    contract = {
        "version": 1,
        "resolved_image_root": "/tmp/images",
        "image_count": 2,
        "image_ids_sha256": "ids",
        "sampled_content_manifest_sha256": "content",
        "source_image_dimensions": {"11x11": 2},
        "sampled_bytes_per_file_end": 32768,
    }
    grid = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=np.asarray(["q0.png", "q1.png"]),
            image_sizes=np.asarray([[11, 11], [11, 11]], dtype=np.int64),
            grids={2: grid},
            metadata={
                "format": PCA_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "render": False,
                "spatial_grid_sizes": [2],
                "projection_dim": 2,
                "image_source_contract": contract,
            },
        ),
        context_path,
    )
    bank_path = tmp_path / "bank.npz"
    save_landmark_index_npz(
        LandmarkMapIndex(
            track_ids=np.asarray([10, 11], dtype=np.int64),
            xyz=np.zeros((2, 3), dtype=np.float64),
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            mean_variances=np.zeros((2,), dtype=np.float32),
            observation_counts=np.ones((2,), dtype=np.int64),
            observation_image_ids=((), ()),
        ),
        bank_path,
        metadata={
            "stage": "radio_intermediate_pca_projected_observation_landmark_bank_v1",
            "descriptor_space_id": "intermediate-space",
            "descriptor_space_manifest": {
                "projection_source": "raw_radio_intermediate_pca_projected_observation_full_map",
                "context_cache_sha256": file_sha256_short(context_path),
                "grid_size": 2,
            },
        },
    )
    detector_path = tmp_path / "detector.npz"
    np.savez(
        detector_path,
        image_ids=np.asarray(["q0.png", "q1.png"]),
        offsets=np.asarray([0, 2, 4], dtype=np.int64),
        xy=np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "alike_detector_mapped_radio_query_cache_v1",
                    "detector_point_count": 4,
                    "detector_selection_version": "test",
                    "descriptor_space_id": "final-space",
                }
            )
        ),
    )
    selected_path = tmp_path / "selected.npz"
    np.savez(
        selected_path,
        selected_rows=np.asarray([0, 3], dtype=np.int64),
        selected_columns=np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "detector_maplet_geometry_features_v1",
                    "contains_ground_truth": False,
                    "contains_pose_derived_selection": False,
                    "supervision_mode": "none_inference_only",
                    "candidate_top_k": 2,
                    "detector_query_cache_sha256": file_sha256_short(detector_path),
                    "proposals_sha256": "proposals",
                }
            )
        ),
    )
    overlay_path = tmp_path / "overlay.npz"
    tracks = np.asarray([[10, 11], [10, 11], [10, 11], [10, 11]], dtype=np.int64)
    probabilities = np.asarray([[0.7, 0.2]] * 4, dtype=np.float32)
    np.savez(
        overlay_path,
        candidate_track_ids=tracks,
        candidate_probabilities=probabilities,
        null_probabilities=np.asarray([0.1] * 4, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "candidate_maplet_prior_overlay_v1",
                    "contains_ground_truth": False,
                    "contains_target_errors": False,
                    "probability_semantics": "candidate_identity_probability_plus_explicit_null_equals_one",
                    "proposals_sha256": "proposals",
                    "inference_data_manifest": {
                        "detector_query_cache_sha256": file_sha256_short(detector_path)
                    },
                }
            )
        ),
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["q0.png"], "validation": ["q1.png"], "test": []})
    )
    output = tmp_path / "rerank.npz"
    summary = tmp_path / "summary.json"
    result = build_radio_intermediate_fixed_candidate_rerank(
        context_cache=context_path,
        context_landmark_bank=bank_path,
        detector_query_cache=detector_path,
        selected_point_artifact=selected_path,
        candidate_prior_overlay=overlay_path,
        query_split=split_path,
        grid_size=2,
        output=output,
        summary_json=summary,
    )
    assert result["candidate_top_k"] == 2
    with np.load(output, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert metadata["format"] == ARTIFACT_FORMAT
        assert data["candidate_track_ids"].tolist() == [[11, 10], [10, 11]]
        assert data["split_names"].tolist() == ["train", "validation"]
        np.testing.assert_allclose(data["radio_intermediate_cosine"][0], [0.0, 1.0])
