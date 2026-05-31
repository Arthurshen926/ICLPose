from feature_extract.vfm.local_pipeline_reporting import failure_bucket_summary, metric_row_from_summary


def test_metric_row_from_summary_extracts_canonical_localization_metrics() -> None:
    row = metric_row_from_summary(
        "OldHospital",
        "C2.5+LM",
        {
            "success_10cm_5deg": 0.1,
            "success_25cm_10deg": 0.4,
            "success_50cm_10deg": 0.7,
            "median_translation_error_m": 0.3,
            "median_rotation_error_deg": 0.6,
            "mean_pnp_inlier_patch_at_1": 0.65,
            "mean_pnp_inlier_count": 123.0,
            "pnp_solve_rate": 1.0,
            "elapsed_sec": 20.0,
            "query_count": 10,
        },
    )

    assert row["scene"] == "OldHospital"
    assert row["method"] == "C2.5+LM"
    assert row["success_25cm_10deg"] == 0.4
    assert row["runtime_sec_per_query"] == 2.0


def test_failure_bucket_summary_counts_actionable_failure_modes() -> None:
    rows = [
        {
            "query_id": "low_inliers.png",
            "success_25cm_10deg": False,
            "pnp_solve": True,
            "pnp_inlier_count": 5,
            "visible_landmark_recall": 0.8,
            "patch_geometry": {"patch_at_1": 0.5},
            "pnp_inlier_spatial": {"grid_4x4_occupancy_frac": 0.5, "depth_range_m": 2.0, "xyz_planarity_ratio": 0.05},
        },
        {
            "query_id": "high_wrong.png",
            "success_25cm_10deg": False,
            "pnp_solve": True,
            "pnp_inlier_count": 200,
            "translation_error_m": 0.4,
            "visible_landmark_recall": 0.9,
            "mean_similarity": 0.8,
            "patch_geometry": {"patch_at_1": 0.7},
            "pnp_inlier_spatial": {"grid_4x4_occupancy_frac": 0.7, "depth_range_m": 3.0, "xyz_planarity_ratio": 0.05},
        },
        {
            "query_id": "low_coverage.png",
            "success_25cm_10deg": False,
            "pnp_solve": True,
            "pnp_inlier_count": 90,
            "visible_landmark_recall": 0.2,
            "patch_geometry": {"patch_at_1": 0.6},
            "pnp_inlier_spatial": {"grid_4x4_occupancy_frac": 0.2, "depth_range_m": 0.2, "xyz_planarity_ratio": 0.0},
        },
        {
            "query_id": "success.png",
            "success_25cm_10deg": True,
            "pnp_solve": True,
            "pnp_inlier_count": 100,
        },
    ]
    baseline = [
        {"query_id": "high_wrong.png", "translation_error_m": 0.2, "rotation_error_deg": 2.0, "success_25cm_10deg": True},
    ]

    summary = failure_bucket_summary(rows, baseline_rows=baseline)

    assert summary["failed_query_count"] == 3
    assert "low_inliers.png" in summary["buckets"]["inlier_count_too_low"]["examples"]
    assert "high_wrong.png" in summary["buckets"]["inlier_count_high_but_wrong_pose"]["examples"]
    assert "low_coverage.png" in summary["buckets"]["reference_top10_coverage_low"]["examples"]
    assert "low_coverage.png" in summary["buckets"]["inliers_spatially_degenerate"]["examples"]
    assert "high_wrong.png" in summary["buckets"]["lm_refinement_worsens"]["examples"]
