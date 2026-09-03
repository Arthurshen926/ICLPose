from __future__ import annotations

import copy

import pytest

from feature_extract.tools.vfm.audit_goal_maplet_resident_render_equivalence import (
    _validate_render_equivalence,
)


def _reports() -> tuple[dict[str, object], dict[str, object]]:
    base = {
        "artifact_type": "goal_maplet_frozen_pnp_moge3_2dgs_render_consistency_v1",
        "query_count": 1,
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_moge3_role": "post_pose_metric_scale_relative_depth_and_normal_verification",
        "depth_scale_fit": "median",
        "affine_log_depth_fit": "bounded",
        "raw_metric_depth_retained": True,
        "query_depth_changes_frozen_pose": False,
        "frozen_pose_inventory_file_sha256": "pose-file",
        "frozen_pose_inventory_content_sha256": "pose-content",
        "physical_map_file_sha256": "map-file",
        "physical_map_content_sha256": "map-content",
        "query_camera_inventory_file_sha256": "camera-file",
        "query_camera_inventory_content_sha256": "camera-content",
        "frozen_correspondence_file_sha256": "corr-file",
        "frozen_correspondence_content_sha256": "corr-content",
        "moge3_manifest_file_sha256_in_order": ["moge-file"],
        "moge3_manifest_content_sha256_in_order": ["moge-content"],
        "minimum_front_incidence": 0.05,
        "renderer": "clean_2dgs_disks_alpha_transmittance_dominant_depth",
        "elapsed_seconds": 2.0,
        "rows": [{"name": "query", "usable": True, "normal_within_20deg": 0.5}],
    }
    optimized = copy.deepcopy(base)
    optimized.update({
        "renderer": "resident_batched_clean_2dgs_disks_alpha_transmittance_dominant_depth",
        "resident_batch_size": 4,
        "resident_gpu_composite": True,
        "elapsed_seconds": 1.0,
    })
    return base, optimized


def test_resident_render_equivalence_accepts_only_runtime_differences() -> None:
    legacy, optimized = _reports()
    _validate_render_equivalence(legacy, optimized)


def test_resident_render_equivalence_rejects_row_drift() -> None:
    legacy, optimized = _reports()
    optimized["rows"][0]["normal_within_20deg"] = 0.5000001
    with pytest.raises(ValueError, match="per-query diagnostics"):
        _validate_render_equivalence(legacy, optimized)


def test_resident_render_equivalence_rejects_lineage_drift() -> None:
    legacy, optimized = _reports()
    optimized["physical_map_content_sha256"] = "other-map"
    with pytest.raises(ValueError, match="physical_map_content_sha256"):
        _validate_render_equivalence(legacy, optimized)
