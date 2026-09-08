"""Aggregate canonical multi-plane layout diagnostics across fixed shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


ARTIFACT = "goal_maplet_relative_multiplane_layout_pnp_postlabel_diagnostic_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite multi-plane aggregate")
    reports = []
    rows = []
    seen = set()
    for path in args.inputs:
        report = json.loads(path.read_text())
        expected = canonical_json_sha256({
            key: value for key, value in report.items() if key != "content_sha256"
        })
        if (
            report.get("artifact_type") != ARTIFACT
            or report.get("content_sha256") != expected
            or report.get("production_eligible") is not False
        ):
            raise ValueError(f"multi-plane input contract differs: {path}")
        split = str(report["split_name"])
        for source in report["postlabel_rows"]:
            row = dict(source)
            key = (split, str(row["name"]))
            if key in seen:
                raise ValueError(f"duplicate multi-plane row: {key}")
            seen.add(key)
            row["split_name"] = split
            rows.append(row)
        reports.append(report)

    fields = (
        "existing_selected_is_2m45",
        "layout_pool_oracle_is_2m45",
        "layout_supported_entities_selected_is_2m45",
        "union_existing_plus_layout_pool_oracle_is_2m45",
    )
    counts = {field: int(sum(bool(row[field]) for row in rows)) for field in fields}
    failures = [row for row in rows if not row["existing_selected_is_2m45"]]
    tail = {
        "existing_failure_count": int(len(failures)),
        "layout_pool_recovers_existing_failure_count": int(sum(
            bool(row["layout_pool_oracle_is_2m45"]) for row in failures
        )),
        "layout_selected_recovers_existing_failure_count": int(sum(
            bool(row["layout_supported_entities_selected_is_2m45"]) for row in failures
        )),
    }
    output = {
        "artifact_type": "goal_maplet_relative_multiplane_layout_pnp_aggregate_diagnostic_v1",
        "evaluation_role": "HISTORICAL_POSTHOC_DIAGNOSTIC_NOT_PROMOTION",
        "query_count": int(len(rows)),
        "coarse_2m45_counts": counts,
        "tail_recovery": tail,
        "decision": (
            "CANDIDATE_GENERATION_HEADROOM_BUT_NO_DEPLOYABLE_SELECTION_GAIN;_"
            "retain_as_bounded_tail_hypothesis_source_only"
        ),
        "input_lineage": [
            {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "content_sha256": report["content_sha256"],
                "split_name": report["split_name"],
            }
            for path, report in zip(args.inputs, reports)
        ],
        "production_eligible": False,
        "rows": rows,
    }
    output["content_sha256"] = canonical_json_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
