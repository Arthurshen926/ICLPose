import json

import pytest

from feature_extract.tools.vfm.compare_score_rows import main as compare_score_rows_cli_main
from feature_extract.vfm.paired_score_comparison import compare_score_rows, load_score_rows_json
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import rows_from_arrays


def _write_rows(path, rows):
    payload = []
    for row in rows:
        item = dict(row.__dict__)
        item["protocol_kind"] = row.protocol_kind.value
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_paired_score_comparison_reports_query_level_delta_and_tests(tmp_path):
    query_ids = ["q0", "q1", "q2"]
    costs = [[0.1, 1.0, 2.0], [0.2, 1.2, 2.2], [0.3, 1.3, 2.3]]
    labels = [[True, False, False], [True, False, False], [True, False, False]]
    method_rows = rows_from_arrays(
        query_ids,
        scores=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        costs_m=costs,
        basin_labels=labels,
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        method="selected",
    )
    baseline_rows = rows_from_arrays(
        query_ids,
        scores=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        costs_m=costs,
        basin_labels=labels,
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        method="retrieval",
    )

    summary = compare_score_rows(
        method_rows,
        baseline_rows,
        label="selected_vs_retrieval",
        resamples=200,
        seed=0,
    )

    assert summary["label"] == "selected_vs_retrieval"
    assert summary["query_count"] == 3
    pred = summary["metrics"]["pred_cost_m"]
    assert pred["method_mean"] == pytest.approx(0.2)
    assert pred["baseline_mean"] == pytest.approx(1.5)
    assert pred["delta_mean"] < 0.0
    assert pred["bootstrap_ci_low"] < pred["bootstrap_ci_high"]
    assert pred["wilcoxon"]["nonzero_count"] == 3
    assert summary["metrics"]["top1_acc"]["mcnemar_pvalue"] <= 1.0


def test_compare_score_rows_cli_writes_json_and_markdown(tmp_path):
    query_ids = ["q0", "q1"]
    costs = [[0.1, 1.0], [0.2, 1.2]]
    labels = [[True, False], [True, False]]
    method_path = tmp_path / "method_rows.json"
    baseline_path = tmp_path / "baseline_rows.json"
    output_json = tmp_path / "comparison.json"
    output_md = tmp_path / "comparison.md"
    _write_rows(
        method_path,
        rows_from_arrays(
            query_ids,
            scores=[[1.0, 0.0], [1.0, 0.0]],
            costs_m=costs,
            basin_labels=labels,
            protocol_kind=ProtocolKind.REFERENCE_POSE,
            method="selected",
        ),
    )
    _write_rows(
        baseline_path,
        rows_from_arrays(
            query_ids,
            scores=[[0.0, 1.0], [0.0, 1.0]],
            costs_m=costs,
            basin_labels=labels,
            protocol_kind=ProtocolKind.REFERENCE_POSE,
            method="retrieval",
        ),
    )

    compare_score_rows_cli_main(
        [
            "--method_rows",
            str(method_path),
            "--baseline_rows",
            str(baseline_path),
            "--label",
            "selected_vs_retrieval",
            "--resamples",
            "100",
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ]
    )

    loaded_rows = load_score_rows_json(method_path)
    assert loaded_rows[0].method == "selected"
    payload = json.loads(output_json.read_text())
    assert payload["metrics"]["pred_cost_m"]["delta_mean"] < 0.0
    table = output_md.read_text()
    assert "selected_vs_retrieval" in table
    assert "| pred_cost_m |" in table
