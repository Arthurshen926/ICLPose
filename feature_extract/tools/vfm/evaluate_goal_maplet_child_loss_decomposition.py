"""Decompose frozen pure-RADIO child retrieval loss without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _contributor_inventory,
    _json_without_duplicates,
    _load_contributor,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
)
from feature_extract.vfm.localization_goal_maplet.child_loss_diagnostics import (
    DEFAULT_AREA_BUDGETS,
    evaluate_child_loss_decomposition,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    PhysicalIncidence,
    remap_pinhole_contributors_to_raw_grid,
    token_primitive_visibility,
)
from feature_extract.vfm.tokens import compute_file_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--base_evaluation", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument(
        "--area_budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_AREA_BUDGETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _mean(rows: list[dict[str, object]], *keys: str) -> float:
    values = []
    for row in rows:
        value: object = row
        for key in keys:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _content_sha256(payload: dict[str, object]) -> str:
    clean = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite child-loss diagnostic")
    summary_paths = [Path(value).resolve() for value in args.retrieval_summary]
    summaries = [_json_without_duplicates(path) for path in summary_paths]
    physical_path = Path(args.physical_map).resolve()
    canonical_path = Path(args.canonical_field).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    canonical = CanonicalSurfaceField.load_npz(canonical_path)
    if canonical.physical_map_sha256 != physical.content_sha256:
        raise ValueError("canonical field and physical map lineage differ")

    records: list[dict[str, object]] = []
    for summary in summaries:
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval run")
        if str(summary.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("retrieval physical map differs")
        for key in (
            "uses_query_pose",
            "uses_query_ground_truth",
            "uses_alike",
            "uses_pnp",
            "uses_sfm_points",
            "uses_sfm_tracks",
            "uses_mapping_rgb",
            "uses_image_retrieval",
        ):
            if summary.get(key) is not False:
                raise ValueError(f"retrieval summary requires {key}=false")
        records.extend(list(summary.get("rows", [])))
    records = sorted(records, key=lambda value: str(value["image_id"]))
    image_ids = [str(value["image_id"]) for value in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("retrieval query identities are duplicated")
    if int(args.expected_queries) > 0 and len(records) != int(args.expected_queries):
        raise ValueError("retrieval query count differs")

    # Validate the already-published metric report before opening contributor
    # GT.  It supplies only the frozen Top64 exact/tolerant control used for C5.
    base_path = Path(args.base_evaluation).resolve()
    base = _json_without_duplicates(base_path)
    if base.get("artifact_type") != "goal_maplet_pure_retrieval_surface_evaluation_v1":
        raise ValueError("base report is not a pure retrieval evaluation")
    declared_summary_paths = [Path(value).resolve() for value in base["retrieval_summaries"]]
    if declared_summary_paths != summary_paths:
        raise ValueError("base evaluation retrieval summaries differ")
    if list(base["retrieval_summary_sha256"]) != [
        compute_file_sha256(path) for path in summary_paths
    ]:
        raise ValueError("base evaluation summary hashes differ")
    base_rows = {str(value["image_id"]): value for value in base["rows"]}
    if sorted(base_rows) != image_ids:
        raise ValueError("base evaluation query inventory differs")

    incidence = PhysicalIncidence.from_physical_map(physical)
    supported = np.zeros((physical.primitive_ids.size,), dtype=bool)
    supported[canonical.primitive_rows] = True
    contributors = _contributor_inventory(Path(args.contributors))
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        retrieval_path = Path(str(record["artifact"]))
        if compute_file_sha256(retrieval_path) != str(record["artifact_sha256"]):
            raise ValueError("retrieval artifact file hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
        if retrieval.image_id != image_id or retrieval.content_sha256 != str(
            record["content_sha256"]
        ):
            raise ValueError("retrieval artifact identity/content differs")
        contributor_path = contributors.get(image_id)
        if contributor_path is None:
            raise ValueError(f"missing contributor truth for {image_id}")
        labels, model, width, height, params, metadata = _load_contributor(
            contributor_path
        )
        if str(metadata.get("image_id", "")) != image_id:
            raise ValueError("contributor identity differs")
        raw_labels, _ = remap_pinhole_contributors_to_raw_grid(
            labels,
            camera_model_id=model,
            camera_width=width,
            camera_height=height,
            camera_params=params,
        )
        token_primitive, _, _ = token_primitive_visibility(
            raw_labels,
            physical,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
        )
        row = evaluate_child_loss_decomposition(
            retrieval,
            token_primitive,
            incidence.primitive_to_parent,
            incidence.primitive_to_child,
            physical,
            canonical_supported_primitive=supported,
            current_surface_metrics=base_rows[image_id]["surface_sets"]["child_top64"],
            area_budgets=tuple(float(value) for value in args.area_budgets),
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "count": len(records),
                    "image_id": image_id,
                    "dominant": row["dominant_attribution"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    attribution_keys = sorted(rows[0]["attribution"]) if rows else []
    aggregate: dict[str, object] = {
        "query_count": len(rows),
        "current_scene_child_exact_recall": _mean(
            rows, "current_scene_child_exact_recall"
        ),
        "token_candidate_union_exact_recall_ceiling": _mean(
            rows, "token_candidate_union_exact_recall_ceiling"
        ),
        "current_parent_candidate_union_exact_recall_ceiling": _mean(
            rows, "current_parent_candidate_union_exact_recall_ceiling"
        ),
        "oracle_parent_current_candidate_exact_recall_ceiling": _mean(
            rows, "oracle_parent_current_candidate_exact_recall_ceiling"
        ),
        "token_candidate_unique_child_count": _mean(
            rows, "token_candidate_unique_child_count"
        ),
        "current_selected_parent_count": _mean(rows, "current_selected_parent_count"),
        "oracle_visible_parent_count": _mean(rows, "oracle_visible_parent_count"),
        "current_selected_child_count": _mean(rows, "current_selected_child_count"),
        "attribution_mean": {
            key: _mean(rows, "attribution", key) for key in attribution_keys
        },
        "dominant_attribution_count": dict(
            sorted(Counter(str(row["dominant_attribution"]) for row in rows).items())
        ),
        "area_curves": {},
    }
    for fraction in args.area_budgets:
        budget_key = f"area_{float(fraction):.2f}"
        aggregate["area_curves"][budget_key] = {}
        scenario_keys = sorted(rows[0]["area_curves"][budget_key]) if rows else []
        for scenario in scenario_keys:
            aggregate["area_curves"][budget_key][scenario] = {
                key: _mean(rows, "area_curves", budget_key, scenario, key)
                for key in (
                    "exact_visible_mass_recall",
                    "selected_child_count",
                    "selected_surface_area_m2",
                )
            }

    candidate_gap = float(
        aggregate["token_candidate_union_exact_recall_ceiling"]
    ) - float(aggregate["current_scene_child_exact_recall"])
    parent_gap = float(
        aggregate["oracle_parent_current_candidate_exact_recall_ceiling"]
    ) - float(aggregate["current_parent_candidate_union_exact_recall_ceiling"])
    aggregate["decision_diagnostics"] = {
        "scene_selection_gap_from_stored_token_candidates": candidate_gap,
        "parent_conditioning_gap_on_current_child_candidates": parent_gap,
        "allocator_is_primary_if_scene_gap_dominates": bool(
            candidate_gap > max(parent_gap, 0.10)
        ),
        "new_child_readout_not_authorized_by_this_diagnostic_alone": True,
    }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_child_loss_decomposition_v1",
        "retrieval_summaries": [str(path) for path in summary_paths],
        "retrieval_summary_sha256": [
            compute_file_sha256(path) for path in summary_paths
        ],
        "base_evaluation": str(base_path),
        "base_evaluation_sha256": compute_file_sha256(base_path),
        "physical_map": str(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field": str(canonical_path),
        "canonical_field_sha256": canonical.content_sha256,
        "contributors": str(Path(args.contributors).resolve()),
        "area_budgets": [float(value) for value in args.area_budgets],
        "aggregate": aggregate,
        "claim_scope": {
            "evaluator_only": True,
            "retrieval_artifacts_not_modified": True,
            "oracle_diagnostics_not_deployment_scores": True,
            "not_pose_recall": True,
            "uses_alike": False,
            "uses_pnp": False,
            "uses_sfm": False,
        },
        "rows": rows,
    }
    report["content_sha256"] = _content_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

