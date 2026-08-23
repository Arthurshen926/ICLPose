"""Evaluate the union of protected phase-refinement seed strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCHEMA = "goal_maplet_protected_multiseed_phase_refinement_summary_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value) for value in args.strategy]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite multiseed summary")
    reports = [json.loads(path.read_text()) for path in paths]
    for key in ("dataset_file_sha256", "energy_semantics", "score_semantics"):
        if len({str(report.get(key)) for report in reports}) != 1:
            raise ValueError(f"phase refinement strategies disagree on {key}")
    per_strategy = []
    expected_ids = None
    for path, report in zip(paths, reports):
        rows = {str(row["image_id"]): row for row in report["rows"]}
        if len(rows) != len(report["rows"]):
            raise ValueError("a phase strategy contains duplicate query IDs")
        ids = tuple(sorted(rows))
        if expected_ids is None:
            expected_ids = ids
        elif not set(ids).issubset(set(expected_ids)):
            raise ValueError("a supplemental phase strategy contains unknown queries")
        per_strategy.append((path, report, rows))
    result_rows = []
    for image_id in expected_ids or ():
        hypotheses = []
        for path, report, rows in per_strategy:
            if image_id not in rows:
                continue
            row = rows[image_id]
            hypotheses.extend((
                {
                    "strategy": report["initialization_semantics"], "role": "protected_seed",
                    "translation_m": float(row["initial_translation_m"]),
                    "rotation_deg": float(row["initial_rotation_deg"]),
                },
                {
                    "strategy": report["initialization_semantics"], "role": "refined_hypothesis",
                    "translation_m": float(row["final_translation_m"]),
                    "rotation_deg": float(row["final_rotation_deg"]),
                },
            ))
        strict = [
            h for h in hypotheses
            if h["translation_m"] <= 0.5 + 1.0e-6 and h["rotation_deg"] <= 5.0 + 1.0e-5
        ]
        loose = [
            h for h in hypotheses
            if h["translation_m"] <= 1.0 + 1.0e-6 and h["rotation_deg"] <= 10.0 + 1.0e-5
        ]
        result_rows.append({
            "image_id": image_id,
            "hypotheses": hypotheses,
            "strict_acquired": bool(strict),
            "loose_acquired": bool(loose),
            "best_translation_m": float(min(h["translation_m"] for h in hypotheses)),
            "best_rotation_deg": float(min(h["rotation_deg"] for h in hypotheses)),
        })
    summary = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": reports[0]["dataset_file_sha256"],
        "energy_semantics": reports[0]["energy_semantics"],
        "score_semantics": reports[0]["score_semantics"],
        "source_strategies": [
            {
                "path": str(path.resolve()), "file_sha256": file_sha256(path),
                "initialization_semantics": report["initialization_semantics"],
            }
            for path, report, _ in per_strategy
        ],
        "query_count": len(result_rows),
        "mean_hypotheses_per_query_before_physical_dedup": float(np.mean([
            len(row["hypotheses"]) for row in result_rows
        ])),
        "maximum_hypotheses_per_query_before_physical_dedup": int(max(
            len(row["hypotheses"]) for row in result_rows
        )),
        "strict_acquisition_rate": float(np.mean([row["strict_acquired"] for row in result_rows])),
        "loose_acquisition_rate": float(np.mean([row["loose_acquired"] for row in result_rows])),
        "protected_seed_means_refinement_cannot_delete_an_input_basin": True,
        "poses_unavailable_in_legacy_source_reports_so_physical_dedup_not_replayed": True,
        "selection_or_top1_claim": False,
        "production_eligible": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "rows": result_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": summary["query_count"],
        "strict_acquisition_rate": summary["strict_acquisition_rate"],
        "loose_acquisition_rate": summary["loose_acquisition_rate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
