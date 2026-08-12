"""Validate and aggregate strict-fold six-axis refinement-basin diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.evaluate_goal_maplet_primitive_refinement_six_axis_basin import (
    AXES,
    _group_summary,
    _trial_definitions,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


EXPECTED_CONFIGURATION = {
    "translation_levels_m": [0.1, 0.25, 0.5, 1.0, 2.0],
    "rotation_levels_deg": [2.0, 5.0, 10.0, 20.0],
    "signs": [-1, 1],
    "include_zero": True,
    "translation_steps_m": [0.6, 0.4, 0.25, 0.12],
    "rotation_steps_deg": [5.0, 3.0, 2.0, 1.0],
    "iterations_per_scale": 2,
    "minimum_score_improvement": 1.0e-6,
    "maximum_splat_radius_tokens": 0,
}


def _parse_reports(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        fold_id, separator, path = str(value).partition("=")
        if not separator or not fold_id or not path:
            raise ValueError("reports must use fold_id=/path/report.json")
        if fold_id in result:
            raise ValueError(f"duplicate six-axis fold report: {fold_id}")
        result[fold_id] = Path(path)
    return result


def aggregate_reports(
    protocol_path: Path,
    report_paths: dict[str, Path],
    *,
    expected_queries_per_trajectory: int = 4,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("artifact_type") != "goal_maplet_official_train_oof_protocol_v1":
        raise ValueError("unsupported OOF protocol")
    fold_specs = {
        str(value["fold_id"]): value for value in protocol["development"]["folds"]
    }
    if set(report_paths) != set(fold_specs):
        raise ValueError("six-axis reports must cover every protocol fold exactly once")

    rows: list[dict[str, object]] = []
    fold_lineage = []
    query_to_fold: dict[str, str] = {}
    route_counts: Counter[str] = Counter()
    trial_keys: set[tuple[object, ...]] = set()
    for fold_id in sorted(fold_specs):
        path = report_paths[fold_id]
        report = json.loads(path.read_text())
        if report.get("artifact_type") != "goal_maplet_primitive_vfm_six_axis_basin_v1":
            raise ValueError(f"unsupported six-axis report: {path}")
        if (
            not bool(report.get("oracle_only_initialization"))
            or bool(report.get("refinement_uses_ground_truth", True))
            or not bool(report.get("exact_score_acceptance_is_monotonic"))
            or report.get("surface_score_semantics")
            != "all_geometry_occludes_fixed_query_grid_primitive_vfm"
        ):
            raise ValueError(f"invalid six-axis method contract: {path}")
        if report.get("configuration") != EXPECTED_CONFIGURATION:
            raise ValueError(f"six-axis configuration drift: {path}")
        fold_rows = list(report.get("rows", []))
        query_ids = sorted({str(value["image_id"]) for value in fold_rows})
        if int(report.get("query_count", -1)) != len(query_ids):
            raise ValueError(f"six-axis query count differs: {path}")
        if str(report.get("query_image_ids_sha256")) != ordered_id_sha256(query_ids):
            raise ValueError(f"six-axis query hash differs: {path}")
        held_routes = set(
            str(value) for value in fold_specs[fold_id]["held_query_trajectories"]
        )
        for image_id in query_ids:
            route = image_id.split("/", 1)[0]
            if route not in held_routes:
                raise ValueError(f"six-axis query is not held out in {fold_id}: {image_id}")
            if image_id in query_to_fold:
                raise ValueError(f"six-axis query occurs in multiple folds: {image_id}")
            query_to_fold[image_id] = fold_id
            route_counts[route] += 1
        for row in fold_rows:
            key = (
                str(row["image_id"]), str(row["perturbation_axis"]),
                float(row["perturbation_magnitude"]), int(row["perturbation_sign"]),
            )
            if key in trial_keys:
                raise ValueError(f"duplicate six-axis trial: {key}")
            trial_keys.add(key)
            if not bool(row.get("exact_score_monotonic")):
                raise ValueError(f"non-monotonic exact-score trial: {key}")
        rows.extend(fold_rows)
        fold_lineage.append({
            "fold_id": fold_id,
            "report": str(path),
            "report_sha256": file_sha256(path),
            "physical_map_sha256": str(report["physical_map_sha256"]),
            "canonical_field_sha256": str(report["canonical_field_sha256"]),
            "surface_mapper_sha256": str(report["surface_mapper_sha256"]),
            "query_count": len(query_ids),
        })

    expected_routes = {
        str(route)
        for fold in fold_specs.values()
        for route in fold["held_query_trajectories"]
    }
    if set(route_counts) != expected_routes or any(
        count != int(expected_queries_per_trajectory)
        for count in route_counts.values()
    ):
        raise ValueError("six-axis query selection is not route balanced")
    query_ids = sorted(query_to_fold)
    axes = set(AXES) | {"zero"}
    if {str(value["perturbation_axis"]) for value in rows} != axes:
        raise ValueError("six-axis report does not cover every declared axis")
    expected_trials = {
        (
            str(value["axis"]), float(value["magnitude"]), int(value["sign"]),
        )
        for value in _trial_definitions(
            EXPECTED_CONFIGURATION["translation_levels_m"],
            EXPECTED_CONFIGURATION["rotation_levels_deg"],
            include_zero=True,
        )
    }
    for image_id in query_ids:
        image_trials = {
            (
                str(row["perturbation_axis"]),
                float(row["perturbation_magnitude"]),
                int(row["perturbation_sign"]),
            )
            for row in rows if str(row["image_id"]) == image_id
        }
        if image_trials != expected_trials:
            raise ValueError(f"incomplete six-axis trials for query: {image_id}")

    axis_levels = sorted({
        (str(row["perturbation_axis"]), float(row["perturbation_magnitude"]))
        for row in rows
    })
    directions = sorted({
        (
            str(row["perturbation_axis"]), float(row["perturbation_magnitude"]),
            int(row["perturbation_sign"]),
        )
        for row in rows
    })
    return {
        "artifact_type": "goal_maplet_primitive_vfm_six_axis_basin_oof_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "oracle_only_initialization": True,
        "refinement_uses_ground_truth": False,
        "selection_uses_basin_outcomes": False,
        "strict_fold_geometry_required": True,
        "surface_score_semantics": (
            "all_geometry_occludes_fixed_query_grid_primitive_vfm"
        ),
        "exact_score_acceptance_is_monotonic": True,
        "configuration": EXPECTED_CONFIGURATION,
        "integrity": {
            "fold_count": len(fold_lineage),
            "query_count": len(query_ids),
            "query_image_ids_sha256": ordered_id_sha256(query_ids),
            "trial_count": len(rows),
            "queries_per_trajectory": dict(sorted(route_counts.items())),
            "every_route_balanced": True,
            "query_fold_overlap": False,
        },
        "summary": _group_summary(rows),
        "by_axis_level": {
            f"{axis}_{magnitude:g}": _group_summary([
                row for row in rows
                if str(row["perturbation_axis"]) == axis
                and float(row["perturbation_magnitude"]) == magnitude
            ])
            for axis, magnitude in axis_levels
        },
        "by_signed_direction": {
            f"{axis}_{magnitude:g}_{sign:+d}": _group_summary([
                row for row in rows
                if str(row["perturbation_axis"]) == axis
                and float(row["perturbation_magnitude"]) == magnitude
                and int(row["perturbation_sign"]) == sign
            ])
            for axis, magnitude, sign in directions
        },
        "folds": fold_lineage,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--expected_queries_per_trajectory", type=int, default=4)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite six-axis OOF aggregation")
    payload = aggregate_reports(
        Path(args.protocol), _parse_reports(args.reports),
        expected_queries_per_trajectory=int(args.expected_queries_per_trajectory),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
