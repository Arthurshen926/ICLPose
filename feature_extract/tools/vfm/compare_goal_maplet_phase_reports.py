"""Compare two frozen phase rankings on one identical historical candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.evaluate_goal_maplet_conditional_energy import (
    _candidate_generator_contract,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_phase_operator_crossfit import (
    MODE,
    _compare_rows,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directional_report", required=True)
    parser.add_argument("--jacobian_report", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode_name", default=MODE)
    parser.add_argument("--label", default="historical_regression")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase-report comparison")
    directional_path = Path(args.directional_report)
    jacobian_path = Path(args.jacobian_report)
    directional = json.loads(directional_path.read_text())
    jacobian = json.loads(jacobian_path.read_text())
    if _candidate_generator_contract(directional) != _candidate_generator_contract(jacobian):
        raise ValueError("phase reports use different candidate generators")
    left = dict(directional.get("surface_verification_contract") or {})
    right = dict(jacobian.get("surface_verification_contract") or {})
    if left.get("candidate_pool_sha256") != right.get("candidate_pool_sha256"):
        raise ValueError("phase reports use different candidate pools")
    result = {
        "stage": "goal_maplet_frozen_phase_operator_pair_comparison",
        "label": str(args.label),
        "historical_regression_only": True,
        "method_selection_allowed": False,
        "null_or_abstention_evaluated": False,
        "directional_report": str(directional_path),
        "directional_report_sha256": file_sha256(directional_path),
        "jacobian_report": str(jacobian_path),
        "jacobian_report_sha256": file_sha256(jacobian_path),
        "candidate_generator_contract": _candidate_generator_contract(directional),
        **_compare_rows(directional, jacobian, mode_name=str(args.mode_name)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "label": result["label"],
        "metrics": result["metrics"],
        "top1_complementarity": result["top1_complementarity"],
        "operator_score_correlation_per_query": result[
            "operator_score_correlation_per_query"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
