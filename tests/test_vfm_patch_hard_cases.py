import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_patch_hard_cases import main


def _row(query_id: str, success: bool, translation: float, inliers: int, coverage: float = 0.5):
    return {
        "query_id": query_id,
        "success_25cm_10deg": success,
        "success_50cm_10deg": translation <= 0.5,
        "translation_error_m": translation,
        "rotation_error_deg": 2.0,
        "visible_landmark_recall": coverage,
        "submap_gt_visible_tracks": 10,
        "pnp_inlier_count": inliers,
        "reference_prior": {
            "top1_translation_error_m": 2.0 if query_id == "q_far" else 0.1,
            "top1_rotation_error_deg": 2.0,
        },
        "patch_geometry": {"pnp_inlier_patch_at_1": 0.8 if success else 0.2},
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_summarize_patch_hard_cases_reports_rescues(tmp_path: Path) -> None:
    random_rows = tmp_path / "random.jsonl"
    learned_rows = tmp_path / "learned.jsonl"
    raw_rows = tmp_path / "raw.jsonl"
    _write_jsonl(
        random_rows,
        [
            _row("q_ok", True, 0.1, 20),
            _row("q_fail", False, 0.8, 3),
            _row("q_far", False, 0.9, 2),
        ],
    )
    _write_jsonl(
        learned_rows,
        [
            _row("q_ok", True, 0.1, 21),
            _row("q_fail", True, 0.2, 12),
            _row("q_far", False, 0.7, 3),
        ],
    )
    _write_jsonl(
        raw_rows,
        [
            _row("q_ok", True, 0.1, 18),
            _row("q_fail", False, 0.8, 4),
            _row("q_far", False, 0.9, 2),
        ],
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    main(
        [
            "--run",
            f"SceneA,random128,{random_rows}",
            "--run",
            f"SceneA,learned128,{learned_rows}",
            "--run",
            f"SceneA,raw1280,{raw_rows}",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    report = json.loads(output_json.read_text())
    assert report["stage"] == "patch_hard_case_summary"
    pair = [row for row in report["pairwise"] if row["method"] == "learned128"][0]
    assert pair["query_count"] == 2
    assert pair["rescued_count"] == 1
    assert pair["worsened_count"] == 0
    assert "reference_top1_far" in output_md.read_text()
