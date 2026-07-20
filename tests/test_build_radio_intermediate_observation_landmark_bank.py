from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_radio_intermediate_image_context_pca_cache import (
    PCA_FORMAT,
)
from feature_extract.tools.vfm.build_radio_intermediate_observation_landmark_bank import (
    STAGE,
    build_radio_intermediate_observation_landmark_bank,
)
from feature_extract.vfm.localization.landmark_hybrid import (
    load_landmark_index_npz,
    save_landmark_index_npz,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def test_builder_creates_full_map_intermediate_observation_bank(tmp_path) -> None:
    contract = {
        "version": 1,
        "resolved_image_root": "/tmp/images",
        "image_count": 2,
        "image_ids_sha256": "ids",
        "sampled_content_manifest_sha256": "content",
        "source_image_dimensions": {"11x11": 2},
        "sampled_bytes_per_file_end": 32768,
    }
    cache_path = tmp_path / "context.npz"
    grids = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    save_spatial_image_context_cache(
        SpatialImageContextCache(
            image_ids=np.asarray(["a.png", "b.png"]),
            image_sizes=np.asarray([[11, 11], [11, 11]], dtype=np.int64),
            grids={2: grids},
            metadata={
                "format": PCA_FORMAT,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "render": False,
                "spatial_grid_sizes": [2],
                "projection_dim": 2,
                "radio_checkpoint_sha256": "radio",
                "intermediate_index": -6,
                "source_context_sha256": "raw",
                "pca_training_manifest_sha256": "pca-manifest",
                "pca_training_image_list_sha256": "pca-ids",
                "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
                "source_image_manifest_sha256": "content",
                "image_source_contract": contract,
            },
        ),
        cache_path,
    )
    manifest_path = tmp_path / "support.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "maplet_support_image_manifest_v1",
                "metadata": {
                    "support_image_count": 2,
                    "pca_fit_scope": "mapping_support_images_excluding_all_query_splits_v1",
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "render": False,
                },
                "records": [{"image_id": "a.png"}, {"image_id": "b.png"}],
            }
        )
    )
    geometry_path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("a.png", "b.png"),
            image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
            source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
            track_ids=np.asarray([10, 11, 10], dtype=np.int64),
            xy=np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float32),
            viewing_rays=np.ones((3, 3), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        geometry_path,
        metadata={
            "support_track_observations_sha256": "tracks",
            "observation_count": 3,
        },
    )
    source_path = tmp_path / "source.npz"
    save_landmark_index_npz(
        LandmarkMapIndex(
            track_ids=np.asarray([10, 11], dtype=np.int64),
            xyz=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            mean_variances=np.zeros((2,), dtype=np.float32),
            observation_counts=np.asarray([2, 1], dtype=np.int64),
            observation_image_ids=(("a.png", "b.png"), ("a.png",)),
        ),
        source_path,
        metadata={"track_observations_sha256": "tracks", "sampled_observation_count": 3},
    )
    output = tmp_path / "bank.npz"
    summary = tmp_path / "summary.json"
    result = build_radio_intermediate_observation_landmark_bank(
        context_cache=cache_path,
        mapping_support_manifest=manifest_path,
        support_geometry_index=geometry_path,
        source_landmark_bank=source_path,
        grid_size=2,
        output_index=output,
        summary_json=summary,
    )
    index, metadata = load_landmark_index_npz(output)
    assert result["stage"] == STAGE
    assert index.track_ids.tolist() == [10, 11]
    assert metadata["stage"] == STAGE
    assert metadata["descriptor_space_manifest"]["projection_source"] == (
        "raw_radio_intermediate_pca_projected_observation_full_map"
    )
    assert metadata["descriptor_space_manifest"]["descriptor_space_id"] == metadata[
        "descriptor_space_id"
    ]
