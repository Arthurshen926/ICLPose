"""Summarize frozen Schur-complement diagnostics for the MoGe3 scale latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    values = np.asarray([
        float(row.get("conditional_scale_data_to_prior_information_ratio", np.nan))
        for row in rows
    ])
    accepted = np.asarray([bool(row.get("refinement_accepted", False)) for row in rows])
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("MoGe3 report has no scale observability diagnostics")
    quantiles = np.quantile(values[finite], [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "query_count": int(len(rows)),
        "diagnosed_count": int(np.sum(finite)),
        "accepted_count": int(np.sum(accepted)),
        "data_at_least_prior_observable_count": int(np.sum(finite & (values >= 1.0))),
        "accepted_and_data_at_least_prior_observable_count": int(
            np.sum(finite & accepted & (values >= 1.0))
        ),
        "data_to_prior_information_ratio_quantiles": {
            key: float(value)
            for key, value in zip(("minimum", "p10", "median", "p90", "maximum"), quantiles)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moge3_evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MoGe3 observability audit")
    source = json.loads(args.moge3_evaluation.read_text())
    if (
        source.get("artifact_type") != "goal_maplet_moge3_plane_scale_surface_refinement_evaluation_v2"
        or source.get("pose_frozen_before_query_pose_or_ground_truth_open") is not True
        or not isinstance(source.get("rows"), list)
    ):
        raise ValueError("MoGe3 evaluation contract differs")
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_moge3_scale_observability_audit_v1",
        "semantics": (
            "Schur complement of robust data Jacobian for log scale after six pose nuisance "
            "variables; metric prior row excluded"
        ),
        "decision_threshold": "conditional_data_information >= log_scale_prior_information",
        **_summary(source["rows"]),
        "moge3_evaluation_file_sha256": file_sha256(args.moge3_evaluation),
        "moge3_evaluation_content_sha256": source.get("content_sha256"),
        "query_pose_or_ground_truth_used_by_this_audit": False,
        "changes_pose_or_acceptance": False,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
