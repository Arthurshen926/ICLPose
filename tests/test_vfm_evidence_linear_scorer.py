import json

import pytest

from feature_extract.tools.vfm.fit_evidence_linear_scorer import main as fit_evidence_cli_main
from feature_extract.vfm.evidence_linear_scorer import fit_heldout_evidence_linear_scorer
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import ScoreRow, evaluate_score_table


def _baseline_rows():
    rows = []
    for query_id in ("q0", "q1", "q2"):
        rows.extend(
            [
                ScoreRow(query_id, "c0", 0.0, 0.1, True, ProtocolKind.RENDERED_POSE, "baseline"),
                ScoreRow(query_id, "c1", 1.0, 1.0, False, ProtocolKind.RENDERED_POSE, "baseline"),
            ]
        )
    return rows


def _evidence_rows():
    rows = []
    for query_id in ("q0", "q1", "q2"):
        rows.extend(
            [
                ScoreRow(
                    query_id,
                    "c0",
                    0.2,
                    0.1,
                    True,
                    ProtocolKind.RENDERED_POSE,
                    "rendered",
                    mean_similarity=0.9,
                    inlier_fraction=0.8,
                    match_count=32,
                    visibility_fraction=0.7,
                ),
                ScoreRow(
                    query_id,
                    "c1",
                    0.8,
                    1.0,
                    False,
                    ProtocolKind.RENDERED_POSE,
                    "rendered",
                    mean_similarity=0.1,
                    inlier_fraction=0.2,
                    match_count=4,
                    visibility_fraction=0.4,
                ),
            ]
        )
    return rows


def _dump_rows(path, rows):
    path.write_text(
        json.dumps(
            [{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in rows],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_heldout_evidence_scorer_learns_visual_feature_on_calibration_only():
    baseline = _baseline_rows()
    evidence = _evidence_rows()

    result = fit_heldout_evidence_linear_scorer(
        baseline,
        evidence,
        calibration_query_ids=["q0", "q1"],
        evaluation_query_ids=["q2"],
        feature_names=["baseline_score", "mean_similarity"],
        method="linear_evidence",
        l2=1e-4,
    )

    baseline_eval = evaluate_score_table([row for row in baseline if row.query_id == "q2"])
    assert baseline_eval.mean_top1_acc == pytest.approx(0.0)
    assert result.evaluation_report.mean_top1_acc == pytest.approx(1.0)
    assert {row.query_id for row in result.evaluation_rows} == {"q2"}
    mean_similarity_weight = result.model.weights[result.model.feature_names.index("mean_similarity")]
    assert mean_similarity_weight > 0.0


def test_heldout_evidence_scorer_refuses_overlapping_splits():
    with pytest.raises(ValueError, match="overlaps"):
        fit_heldout_evidence_linear_scorer(
            _baseline_rows(),
            _evidence_rows(),
            calibration_query_ids=["q0"],
            evaluation_query_ids=["q0"],
            feature_names=["baseline_score", "mean_similarity"],
            method="bad",
        )


def test_heldout_evidence_scorer_supports_query_normalized_ranking_target():
    result = fit_heldout_evidence_linear_scorer(
        _baseline_rows(),
        _evidence_rows(),
        calibration_query_ids=["q0", "q1"],
        evaluation_query_ids=["q2"],
        feature_names=["baseline_score", "mean_similarity"],
        method="linear_evidence",
        target="negative_query_zscore_cost",
    )

    assert result.model.target == "negative_query_zscore_cost"
    assert result.evaluation_report.mean_top1_acc == pytest.approx(1.0)


def test_heldout_evidence_scorer_cli_writes_evaluation_rows_and_model(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    evidence_path = tmp_path / "evidence.json"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    split_path = tmp_path / "split.json"
    _dump_rows(baseline_path, _baseline_rows())
    _dump_rows(evidence_path, _evidence_rows())

    fit_evidence_cli_main(
        [
            "--baseline_rows",
            str(baseline_path),
            "--evidence_rows",
            str(evidence_path),
            "--features",
            "baseline_score",
            "mean_similarity",
            "--calibration_prefix",
            "q0",
            "--calibration_prefix",
            "q1",
            "--evaluation_prefix",
            "q2",
            "--method",
            "linear_evidence",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
            "--output_split_queries",
            str(split_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    split = json.loads(split_path.read_text())
    assert {row["query_id"] for row in rows} == {"q2"}
    assert report["method"] == "linear_evidence"
    assert report["model"]["feature_names"] == ["baseline_score", "mean_similarity"]
    assert report["split"]["protocol"] == "query_prefix_holdout"
    assert split["evaluation_query_ids"] == ["q2"]
