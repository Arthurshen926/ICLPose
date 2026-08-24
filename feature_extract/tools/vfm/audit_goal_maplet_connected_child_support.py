"""Audit lossless connected-component carriers for strict child retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.vfm.localization_goal_maplet.connected_support_audit import (
    audit_connected_support_carrier,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    child_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.tokens import compute_file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument("--maximum_normal_angle_degrees", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _content_sha256(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite connected-support audit")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    area = child_surface_area_m2(physical)
    summaries = [Path(value).resolve() for value in args.retrieval_summary]
    records: list[dict[str, object]] = []
    query_routes: set[str] = set()
    for summary_path in summaries:
        summary = _json_without_duplicates(summary_path)
        if (
            summary.get("artifact_type")
            != "goal_maplet_pure_radio_retrieval_run_v1"
            or summary.get("promotion_eligible") is not True
            or summary.get("control_only") is not False
            or str(summary.get("physical_map_sha256", ""))
            != physical.content_sha256
        ):
            raise ValueError("connected-support audit requires strict retrieval")
        query_routes.update(
            str(value)
            for value in summary.get("query_split_audit", {}).get(
                "query_trajectory_ids", []
            )
        )
        records.extend(list(summary.get("rows", [])))
    records.sort(key=lambda value: str(value["image_id"]))
    if (
        not records
        or len({str(value["image_id"]) for value in records}) != len(records)
        or (int(args.expected_queries) > 0 and len(records) != int(args.expected_queries))
    ):
        raise ValueError("connected-support retrieval inventory differs")
    rows: list[dict[str, object]] = []
    for record in records:
        path = Path(str(record["artifact"])).resolve()
        if compute_file_sha256(path) != str(record["artifact_sha256"]):
            raise ValueError("connected-support retrieval file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(path)
        if retrieval.content_sha256 != str(record["content_sha256"]):
            raise ValueError("connected-support retrieval content differs")
        rows.append({
            "image_id": retrieval.image_id,
            **audit_connected_support_carrier(
                retrieval,
                physical,
                maximum_normal_angle_degrees=float(
                    args.maximum_normal_angle_degrees
                ),
                precomputed_child_surface_area_m2=area,
            ),
        })
    metric_keys = (
        "selected_child_count",
        "connected_component_count",
        "singleton_component_fraction",
        "mean_children_per_component",
        "maximum_children_per_component",
        "largest_component_child_fraction",
        "largest_component_score_fraction",
        "largest_component_area_fraction",
        "selected_surface_area_m2",
    )
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "p50": float(np.quantile([float(row[key]) for row in rows], 0.50)),
            "p90": float(np.quantile([float(row[key]) for row in rows], 0.90)),
        }
        for key in metric_keys
    }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_connected_child_support_audit_v1",
        "retrieval_summaries": [str(path) for path in summaries],
        "retrieval_summary_file_sha256": [
            compute_file_sha256(path) for path in summaries
        ],
        "physical_map": str(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "query_trajectory_ids": sorted(query_routes),
        "query_count": len(rows),
        "maximum_normal_angle_degrees": float(
            args.maximum_normal_angle_degrees
        ),
        "aggregate": aggregate,
        "all_child_unions_preserved_exactly": all(
            bool(row["child_union_preserved_exactly"]) for row in rows
        ),
        "all_surface_areas_preserved": all(
            bool(row["surface_area_preserved"]) for row in rows
        ),
        "all_scores_preserved": all(bool(row["score_preserved"]) for row in rows),
        "claim_scope": {
            "lossless_retrieval_set_carrier_only": True,
            "not_a_new_selector": True,
            "not_pose_estimation": True,
            "uses_query_pose": False,
            "uses_query_ground_truth": False,
        },
        "rows": rows,
    }
    report["content_sha256"] = _content_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
