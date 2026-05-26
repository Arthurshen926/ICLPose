import json

import pytest

from feature_extract.tools.vfm.fuse_score_rows import main as fuse_score_rows_cli_main
from feature_extract.tools.vfm.fit_fusion_weight import main as fit_fusion_weight_cli_main
from feature_extract.tools.vfm.slice_score_rows_by_queries import main as slice_rows_cli_main
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_fusion import (
    calibrate_two_score_fusion,
    fuse_two_score_rows,
    fuse_score_rows,
    normalize_scores_by_query,
    slice_score_rows_by_query_ids,
)
from feature_extract.vfm.score_table import ScoreRow, evaluate_score_table


def _rows(method, scores):
    rows = []
    for query_id, query_scores in scores.items():
        for idx, score in enumerate(query_scores):
            cost = 0.1 if idx == 0 else 1.0 + idx
            rows.append(
                ScoreRow(
                    query_id=query_id,
                    candidate_id=f"c{idx}",
                    score=float(score),
                    cost_m=cost,
                    basin_label=idx == 0,
                    protocol_kind=ProtocolKind.REAL_RETRIEVAL,
                    method=method,
                )
            )
    return rows


def test_score_fusion_normalizes_scores_per_query_before_combining():
    prior = _rows("prior", {"q1": [100.0, 90.0], "q2": [0.9, 1.0]})
    visual = _rows("visual", {"q1": [0.1, 0.9], "q2": [0.8, 0.2]})

    fused = fuse_score_rows(
        [prior, visual],
        weights=[1.0, 1.0],
        method="prior_visual_fusion",
        normalization="minmax",
    )
    report = evaluate_score_table(fused)

    assert [row.method for row in fused] == ["prior_visual_fusion"] * 4
    assert report.mean_top1_acc == pytest.approx(1.0)


def test_score_fusion_refuses_missing_candidate_alignment():
    prior = _rows("prior", {"q1": [1.0, 0.0]})
    visual = _rows("visual", {"q1": [1.0]})

    with pytest.raises(ValueError, match="candidate key"):
        fuse_score_rows([prior, visual], weights=[1.0, 1.0], method="bad")


def test_score_fusion_can_align_rows_by_query_rank_suffix():
    prior = [
        ScoreRow("q1", "frame:score:000", 1.0, 0.1, True, ProtocolKind.REAL_RETRIEVAL, "prior"),
        ScoreRow("q1", "frame:score:001", 0.0, 2.0, False, ProtocolKind.REAL_RETRIEVAL, "prior"),
    ]
    visual = [
        ScoreRow("q1", "frame:retrieval:000", 0.5, 0.1, True, ProtocolKind.REAL_RETRIEVAL, "visual"),
        ScoreRow("q1", "frame:retrieval:001", 0.2, 2.0, False, ProtocolKind.REAL_RETRIEVAL, "visual"),
    ]

    fused = fuse_score_rows(
        [prior, visual],
        weights=[0.75, 0.25],
        method="aligned",
        alignment="query_rank",
    )

    assert [row.candidate_id for row in fused] == ["frame:score:000", "frame:score:001"]


def test_score_fusion_can_align_legacy_init_lattice_candidate_ids():
    prior = [
        ScoreRow(
            "seq4/frame0001.png",
            "seq4__frame0001:init_lattice:000:000",
            1.0,
            0.1,
            True,
            ProtocolKind.RENDERED_POSE,
            "prior",
        ),
        ScoreRow(
            "seq4/frame0001.png",
            "seq4__frame0001:init_lattice:000:001",
            0.0,
            2.0,
            False,
            ProtocolKind.RENDERED_POSE,
            "prior",
        ),
    ]
    visual = [
        ScoreRow(
            "seq4/frame0001.png",
            "frame0001:init_lattice:000:000",
            0.5,
            0.1,
            True,
            ProtocolKind.RENDERED_POSE,
            "visual",
        ),
        ScoreRow(
            "seq4/frame0001.png",
            "frame0001:init_lattice:000:001",
            0.2,
            2.0,
            False,
            ProtocolKind.RENDERED_POSE,
            "visual",
        ),
    ]

    fused = fuse_score_rows(
        [prior, visual],
        weights=[0.75, 0.25],
        method="aligned",
        alignment="init_lattice_id",
    )

    assert [row.candidate_id for row in fused] == [
        "seq4__frame0001:init_lattice:000:000",
        "seq4__frame0001:init_lattice:000:001",
    ]


def test_score_fusion_alpha_zero_and_one_match_primary_and_secondary():
    prior = _rows("prior", {"q1": [0.9, 0.1]})
    visual = _rows("visual", {"q1": [0.1, 0.9]})

    only_prior = fuse_score_rows(
        [prior, visual],
        weights=[1.0, 0.0],
        method="only_prior",
        normalization="none",
    )
    only_visual = fuse_score_rows(
        [prior, visual],
        weights=[0.0, 1.0],
        method="only_visual",
        normalization="none",
    )

    assert [row.score for row in only_prior] == pytest.approx([0.9, 0.1])
    assert [row.score for row in only_visual] == pytest.approx([0.1, 0.9])


def test_fuse_two_score_rows_uses_alpha_semantics():
    prior = _rows("prior", {"q1": [0.9, 0.1]})
    visual = _rows("visual", {"q1": [0.1, 0.9]})

    fused = fuse_two_score_rows(
        prior,
        visual,
        alpha=0.25,
        method="two_score",
        normalization="none",
    )

    assert [row.score for row in fused] == pytest.approx([0.7, 0.3])


def test_score_fusion_cli_writes_rows_and_report(tmp_path):
    prior_path = tmp_path / "prior.json"
    visual_path = tmp_path / "visual.json"
    rows_path = tmp_path / "fused_rows.json"
    report_path = tmp_path / "fused_report.json"
    prior_rows = _rows("prior", {"q1": [0.9, 0.1], "q2": [0.0, 1.0]})
    visual_rows = _rows("visual", {"q1": [0.2, 0.8], "q2": [1.0, 0.0]})

    def dump(path, rows):
        path.write_text(
            json.dumps(
                [{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in rows],
                indent=2,
            )
        )

    dump(prior_path, prior_rows)
    dump(visual_path, visual_rows)

    fuse_score_rows_cli_main(
        [
            "--score_rows",
            str(prior_path),
            str(visual_path),
            "--weights",
            "1.0",
            "0.5",
            "--normalization",
            "zscore",
            "--method",
            "prior_plus_visual",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text())
    fused_rows = json.loads(rows_path.read_text())
    assert payload["method"] == "prior_plus_visual"
    assert payload["inputs"]["score_rows"][0]["path"] == str(prior_path)
    assert fused_rows[0]["method"] == "prior_plus_visual"


def test_score_fusion_cli_accepts_primary_secondary_alpha(tmp_path):
    prior_path = tmp_path / "prior.json"
    visual_path = tmp_path / "visual.json"
    rows_path = tmp_path / "fused_rows.json"
    report_path = tmp_path / "fused_report.json"
    prior_rows = _rows("prior", {"q1": [0.9, 0.1]})
    visual_rows = _rows("visual", {"q1": [0.1, 0.9]})
    prior_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in prior_rows]))
    visual_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in visual_rows]))

    fuse_score_rows_cli_main(
        [
            "--primary_rows",
            str(prior_path),
            "--secondary_rows",
            str(visual_path),
            "--alpha",
            "0.25",
            "--normalization",
            "none",
            "--method",
            "prior_plus_visual_alpha025",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    fused_rows = json.loads(rows_path.read_text())
    assert [row["score"] for row in fused_rows] == pytest.approx([0.7, 0.3])


def test_score_normalization_supports_rank_percentile():
    rows = _rows("visual", {"q1": [0.2, 0.8, 0.5]})

    normalized = normalize_scores_by_query(rows, normalization="rank_percentile")

    values = [row.score for row in normalized]
    assert values == pytest.approx([0.0, 1.0, 0.5])


def test_slice_score_rows_by_hard_case_query_ids_cli(tmp_path):
    rows_path = tmp_path / "rows.json"
    hard_cases_path = tmp_path / "hard_cases.json"
    output_rows = tmp_path / "slice_rows.json"
    output_report = tmp_path / "slice_report.json"
    rows = _rows("method", {"q1": [0.9, 0.1], "q2": [0.1, 0.9]})
    rows_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in rows]))
    hard_cases_path.write_text(json.dumps({"retrieval_top1_wrong": ["q2"]}))

    slice_rows_cli_main(
        [
            "--rows",
            str(rows_path),
            "--query_set_json",
            str(hard_cases_path),
            "--query_set_key",
            "retrieval_top1_wrong",
            "--output_rows",
            str(output_rows),
            "--output_report",
            str(output_report),
        ]
    )

    sliced_rows = json.loads(output_rows.read_text())
    report = json.loads(output_report.read_text())
    assert {row["query_id"] for row in sliced_rows} == {"q2"}
    assert report["query_count"] == 1
    assert report["inputs"]["query_set_key"] == "retrieval_top1_wrong"


def test_slice_score_rows_by_query_ids_library_refuses_empty_slice():
    rows = _rows("method", {"q1": [0.9, 0.1], "q2": [0.1, 0.9]})

    sliced = slice_score_rows_by_query_ids(rows, ["q2"])

    assert {row.query_id for row in sliced} == {"q2"}
    with pytest.raises(ValueError, match="no score rows"):
        slice_score_rows_by_query_ids(rows, ["missing"])


def test_calibrate_two_score_fusion_selects_alpha_on_calibration_only():
    primary = _rows("primary", {"q0": [0.9, 0.1], "q1": [0.9, 0.1], "q2": [0.9, 0.1]})
    secondary = _rows("secondary", {"q0": [0.1, 0.9], "q1": [0.1, 0.9], "q2": [0.9, 0.1]})

    result = calibrate_two_score_fusion(
        primary,
        secondary,
        alphas=[0.0, 0.5, 1.0],
        metric="top1_acc",
        calibration_query_ids=["q0", "q1"],
        evaluation_query_ids=["q2"],
        method_prefix="calibrated",
        normalization="none",
    )

    assert result.selected_alpha == pytest.approx(0.0)
    assert result.calibration_report.query_count == 2
    assert result.evaluation_report.query_count == 1
    assert result.alpha_reports[0.0].calibration.mean_top1_acc == pytest.approx(1.0)
    assert result.alpha_reports[1.0].calibration.mean_top1_acc == pytest.approx(0.0)


def test_fit_fusion_weight_cli_writes_split_calibrated_report(tmp_path):
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    output_rows = tmp_path / "selected_rows.json"
    output_report = tmp_path / "fit_report.json"
    primary = _rows("primary", {"q0": [0.9, 0.1], "q1": [0.9, 0.1], "q2": [0.1, 0.9], "q3": [0.1, 0.9]})
    secondary = _rows("secondary", {"q0": [0.1, 0.9], "q1": [0.1, 0.9], "q2": [0.9, 0.1], "q3": [0.9, 0.1]})
    primary_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in primary]))
    secondary_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in secondary]))

    fit_fusion_weight_cli_main(
        [
            "--primary_rows",
            str(primary_path),
            "--secondary_rows",
            str(secondary_path),
            "--alphas",
            "0.0",
            "0.5",
            "1.0",
            "--metric",
            "top1_acc",
            "--calibration_fraction",
            "0.5",
            "--split_seed",
            "0",
            "--normalization",
            "none",
            "--method_prefix",
            "fit_test",
            "--output_rows",
            str(output_rows),
            "--output_report",
            str(output_report),
        ]
    )

    payload = json.loads(output_report.read_text())
    rows = json.loads(output_rows.read_text())
    assert payload["selected_alpha"] == pytest.approx(0.5)
    assert payload["split"]["calibration_query_count"] == 2
    assert payload["split"]["evaluation_query_count"] == 2
    assert len(rows) == 4
    assert rows[0]["method"].startswith("fit_test_a")


def test_fit_fusion_weight_cli_uses_explicit_query_lists(tmp_path):
    primary_path = tmp_path / "primary.json"
    secondary_path = tmp_path / "secondary.json"
    calibration_queries = tmp_path / "calibration_queries.json"
    evaluation_queries = tmp_path / "evaluation_queries.json"
    output_rows = tmp_path / "selected_rows.json"
    output_report = tmp_path / "fit_report.json"
    output_split = tmp_path / "split_queries.json"
    primary = _rows("primary", {"q0": [0.9, 0.1], "q1": [0.9, 0.1], "q2": [0.1, 0.9]})
    secondary = _rows("secondary", {"q0": [0.1, 0.9], "q1": [0.1, 0.9], "q2": [0.9, 0.1]})
    primary_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in primary]))
    secondary_path.write_text(json.dumps([{**row.__dict__, "protocol_kind": row.protocol_kind.value} for row in secondary]))
    calibration_queries.write_text(json.dumps(["q0", "q1"]))
    evaluation_queries.write_text(json.dumps(["q2"]))

    fit_fusion_weight_cli_main(
        [
            "--primary_rows",
            str(primary_path),
            "--secondary_rows",
            str(secondary_path),
            "--alphas",
            "0.0",
            "0.5",
            "1.0",
            "--metric",
            "top1_acc",
            "--calibration_queries",
            str(calibration_queries),
            "--evaluation_queries",
            str(evaluation_queries),
            "--normalization",
            "none",
            "--method_prefix",
            "fit_explicit",
            "--output_rows",
            str(output_rows),
            "--output_report",
            str(output_report),
            "--output_split_queries",
            str(output_split),
        ]
    )

    payload = json.loads(output_report.read_text())
    rows = json.loads(output_rows.read_text())
    split_payload = json.loads(output_split.read_text())
    assert payload["split"]["protocol"] == "query_explicit_holdout"
    assert payload["selected_alpha"] == pytest.approx(0.0)
    assert {row["query_id"] for row in rows} == {"q2"}
    assert split_payload["calibration_query_ids"] == ["q0", "q1"]
    assert split_payload["evaluation_query_ids"] == ["q2"]
