"""Apply the frozen train-OOF refinement policy to deployment queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.evaluate_goal_maplet_adaptive_refinement_oof import (
    _replay_row,
)
from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    RUN_SOURCE_IDENTITY_KEYS,
    _refinement_execution_configuration,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--refinement_reports", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite deployment policy output")
    frozen_path = Path(args.frozen)
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("artifact_type") != "goal_maplet_g23_frozen_configuration_v1":
        raise ValueError("unsupported frozen G23 configuration")
    configuration = frozen["refinement_configuration"]
    paths = [Path(value) for value in args.refinement_reports]
    reports = [json.loads(path.read_text()) for path in paths]
    rows = [row for report in reports for row in report.get("rows", [])]
    image_ids = [str(row["image_id"]) for row in rows]
    if not rows or len(image_ids) != len(set(image_ids)):
        raise ValueError("deployment refinement reports are empty or overlap")
    for report in reports:
        if (
            str(report.get("refinement_candidate_policy")) != "score_topk"
            or float(report.get("refinement_score_margin", -1.0)) >= 0.0
            or str(report.get("postselection_evidence_contract", ""))
            != "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ):
            raise ValueError("deployment source is not a complete score-TopK run")
        if _refinement_execution_configuration(report) != frozen.get(
            "refinement_execution_configuration"
        ):
            raise ValueError("deployment refinement execution differs from frozen G23")
        if report.get("run_manifest", {}).get("numeric_contract", {}).get(
            "acceptance"
        ) != "strict_monotonic_common_exact_surface_score":
            raise ValueError("deployment refinement lacks exact-score run lineage")
        source_identity = {
            key: report.get("run_manifest", {}).get(key)
            for key in RUN_SOURCE_IDENTITY_KEYS
        }
        if source_identity != frozen.get("implementation_source_identity"):
            raise ValueError("deployment refinement implementation differs from frozen G23")
    validation_enabled = all(
        int(report.get("validation_splat_radius_tokens", -1)) >= 0
        for report in reports
    )
    require_consistency = all(
        bool(report.get("require_cross_splat_winner_consistency", False))
        for report in reports
    )
    selected = [
        _replay_row(
            row, policy=str(configuration["policy"]),
            budget=int(configuration["maximum_refinement_candidates"]),
            score_margin=float(configuration["score_margin"]),
            validation_enabled=validation_enabled,
            require_cross_splat_winner_consistency=require_consistency,
        )
        for row in rows
    ]
    selected.sort(key=lambda value: str(value["image_id"]))
    payload = {
        "artifact_type": "goal_maplet_frozen_policy_deployment_report_v1",
        "postselection_evidence_contract": (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ),
        "selection_uses_query_ground_truth": False,
        "ground_truth_retained_for_offline_metrics_only": True,
        "frozen_configuration": str(frozen_path),
        "frozen_configuration_sha256": file_sha256(frozen_path),
        "refinement_configuration": configuration,
        "source_reports": [str(path) for path in paths],
        "source_report_sha256": [file_sha256(path) for path in paths],
        "query_count": len(selected),
        "rows": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
