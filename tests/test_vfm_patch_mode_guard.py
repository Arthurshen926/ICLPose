import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_patch_mode_guard import main


def _row(query_id, inliers, ratio, residual, success, translation):
    return {
        "query_id": query_id,
        "pnp_inlier_count": inliers,
        "pnp_inlier_ratio": ratio,
        "pnp_reprojection": {"pnp_reproj_inlier_median_px": residual},
        "success_25cm_10deg": success,
        "success_50cm_10deg": translation <= 0.5,
        "translation_error_m": translation,
        "rotation_error_deg": 2.0,
        "patch_geometry": {"pnp_inlier_patch_at_1": 0.5},
    }


def _write(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_summarize_patch_mode_guard_selects_rows_by_policy(tmp_path: Path) -> None:
    feature = tmp_path / "feature.jsonl"
    stats = tmp_path / "stats.jsonl"
    _write(
        feature,
        [
            _row("q1", 10, 0.5, 5.0, True, 0.1),
            _row("q2", 5, 0.2, 2.0, False, 0.8),
        ],
    )
    _write(
        stats,
        [
            _row("q1", 4, 0.4, 1.0, False, 0.7),
            _row("q2", 20, 0.7, 4.0, True, 0.2),
        ],
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    main(
        [
            "--run",
            f"SceneA,feature,{feature}",
            "--run",
            f"SceneA,stats,{stats}",
            "--policy",
            "inlier_count",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    row = report["rows"][0]
    assert row["policy"] == "inlier_count"
    assert row["method_counts"] == {"feature": 1, "stats": 1}
    assert row["success_25cm_10deg"] == 1.0
    assert "stats:1" in output_md.read_text()
