from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.build_multiscale_context_attention_probe_contract import (
    build_multiscale_context_attention_probe_contract,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
    save_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    save_spatial_image_context_cache,
)


def _unit_grid(count: int, size: int, *, axis: int) -> np.ndarray:
    values = np.zeros((count, size * size, 2), dtype=np.float32)
    values[..., axis] = 1.0
    return values


def _write_inputs(tmp_path):
    layout = tmp_path / "layout.npz"
    geometry = tmp_path / "geometry.npz"
    final = tmp_path / "final.npz"
    intermediate = tmp_path / "intermediate.npz"
    alike = tmp_path / "alike.npz"
    ids = np.asarray(["q.png", "s1.png", "s2.png"])
    np.savez(
        layout,
        source_row_indices=np.asarray([3], dtype=np.int64),
        query_ids=np.asarray(["q.png"]),
        split_names=np.asarray(["train"]),
        xy=np.asarray([[40.0, 40.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[11]], dtype=np.int64),
        candidate_canonical_rows=np.asarray([[0]], dtype=np.int64),
        candidate_features=np.asarray(
            [[[[0.2, 0.3, 0.4], [0.2, 0.35, 0.45]]]], dtype=np.float32
        ),
        candidate_view_valid=np.asarray([[[True, True]]]),
        candidate_support_image_ids=np.asarray([[["s1.png", "s2.png"]]]),
        candidate_support_coverage_counts=np.asarray([[[3, 2]]], dtype=np.int32),
        feature_names=np.asarray(
            [
                "radio_final_anchor_cosine",
                "radio_intermediate_anchor_cosine",
                "alike_anchor_cosine",
            ]
        ),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "multiscale_candidate_probe_features_v1",
                    "contains_ground_truth": False,
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "whole_image_summary_or_global_used": False,
                    "proposals_sha256": "test-proposals-sha",
                    "radio_checkpoint_sha256": "radio-test",
                    "support_view_selection": "fixed_test_support_views",
                    "is_complete_frozen_layout": True,
                }
            )
        ),
    )
    save_support_observation_geometry_index_npz(
        SupportObservationGeometryIndex(
            image_ids=("s1.png", "s2.png"),
            image_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            source_row_indices=np.asarray([0, 1], dtype=np.int64),
            track_ids=np.asarray([11, 11], dtype=np.int64),
            xy=np.asarray([[40.0, 40.0], [40.0, 40.0]], dtype=np.float32),
            viewing_rays=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            reprojection_errors=np.asarray([0.1, 0.1], dtype=np.float32),
        ),
        geometry,
        metadata={"coordinate_source": "sfm_observation_xy"},
    )
    np.savez(
        final,
        image_ids=ids,
        image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
        summary_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        global_descriptors=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        grid4_descriptors=_unit_grid(3, 4, axis=0),
        grid8_descriptors=_unit_grid(3, 8, axis=0),
        grid16_descriptors=_unit_grid(3, 16, axis=0),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "format": "radio_final_context_pca_v1",
                    "pose_or_ground_truth_used": False,
                    "image_retrieval_or_submap_used": False,
                    "render": False,
                    "pca_fit_scope": "mapping_train_images_only",
                    "spatial_grid_sizes": [4, 8, 16],
                    "radio_checkpoint_sha256": "radio-test",
                    "source_image_manifest_sha256": "image-manifest",
                }
            )
        ),
    )

    def write_spatial(path, *, format_name: str, size: int, axis: int, extra: dict) -> None:
        cache = SpatialImageContextCache(
            image_ids=ids,
            image_sizes=np.asarray([[80, 80]] * 3, dtype=np.int64),
            grids={size: _unit_grid(3, size, axis=axis)},
            metadata={
                "format": format_name,
                "pose_or_ground_truth_used": False,
                "image_retrieval_or_submap_used": False,
                "render": False,
                "spatial_grid_sizes": [size],
                "source_image_manifest_sha256": "image-manifest",
                **extra,
            },
        )
        save_spatial_image_context_cache(cache, path)

    write_spatial(
        intermediate,
        format_name="radio_intermediate_image_context_pca_v1",
        size=16,
        axis=1,
        extra={
            "pca_fit_scope": "mapping_train_images_only",
            "radio_checkpoint_sha256": "radio-test",
            "intermediate_index": -6,
        },
    )
    write_spatial(
        alike,
        format_name="alike_image_spatial_context_v1",
        size=32,
        axis=0,
        extra={"alike_checkpoint_sha256": "alike-test"},
    )
    return layout, geometry, final, intermediate, alike


def test_context_attention_contract_freezes_multiscale_target_free_sources(tmp_path) -> None:
    layout, geometry, final, intermediate, alike = _write_inputs(tmp_path)
    output = tmp_path / "contract.json"
    summary = tmp_path / "summary.json"

    build_multiscale_context_attention_probe_contract(
        frozen_layout_features=layout,
        support_geometry_index=geometry,
        radio_final_context_cache=final,
        radio_intermediate_context_cache=intermediate,
        alike_spatial_context_cache=alike,
        output=output,
        summary_json=summary,
        force=False,
    )

    contract = json.loads(output.read_text())
    assert contract["format"] == CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT
    assert contract["pose_or_ground_truth_used"] is False
    assert contract["image_retrieval_or_submap_used"] is False
    assert contract["whole_image_summary_or_global_used"] is False
    assert contract["candidate_input_kind"] == "proposal_overlay_v1"
    assert contract["candidate_input_lineage_sha256"] == "test-proposals-sha"
    assert contract["context_only_center_mask"]["shape"] == "3x3"
    assert [item["name"] for item in contract["source_scales"]] == [
        "radio_final",
        "radio_intermediate",
        "alike",
    ]
    assert contract["valid_candidate_view_count"] == 2
