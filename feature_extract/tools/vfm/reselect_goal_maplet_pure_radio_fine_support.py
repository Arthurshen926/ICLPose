"""Reallocate frozen pure-RADIO child evidence under a physical-area budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    COMPONENT_SEMANTICS,
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    SELECTION_SEMANTICS,
    child_surface_area_m2,
    select_fine_supports_under_area_budget,
    total_map_surface_area_m2,
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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_area_fraction", type=float, required=True)
    parser.add_argument("--maximum_children", type=int, default=2048)
    parser.add_argument(
        "--target_candidate_posterior_mass_fraction", type=float, default=1.0
    )
    parser.add_argument("--maximum_child_primitive_iou", type=float, default=0.50)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _atomic_save(result: PureRadioPhysicalRetrieval, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp.npz")
    if temporary.exists():
        temporary.unlink()
    result.save_npz(temporary)
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite fine-support summary")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    child_area = child_surface_area_m2(physical)
    total_map_area = total_map_surface_area_m2(physical)
    source_paths = [Path(value).resolve() for value in args.retrieval_summary]
    source_summaries = [_json_without_duplicates(path) for path in source_paths]
    records: list[dict[str, object]] = []
    required_false = (
        "uses_query_pose",
        "uses_query_ground_truth",
        "uses_alike",
        "uses_pnp",
        "uses_sfm_points",
        "uses_sfm_tracks",
        "uses_mapping_rgb",
        "uses_image_retrieval",
    )
    for summary in source_summaries:
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval summary")
        if str(summary.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("retrieval and physical map differ")
        for key in required_false:
            if summary.get(key) is not False:
                raise ValueError(f"retrieval summary requires {key}=false")
        records.extend(list(summary.get("rows", [])))
    records = sorted(records, key=lambda row: str(row["image_id"]))
    identities = [str(row["image_id"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("retrieval query identities are duplicated")
    if int(args.expected_queries) > 0 and len(records) != int(args.expected_queries):
        raise ValueError("retrieval query count differs")
    if not records:
        raise ValueError("retrieval summary is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    selected_counts: list[int] = []
    area_fractions: list[float] = []
    posterior_masses: list[float] = []
    component_counts: list[int] = []
    for index, record in enumerate(records):
        source = Path(str(record["artifact"])).resolve()
        if compute_file_sha256(source) != str(record["artifact_sha256"]):
            raise ValueError("source retrieval artifact hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(source)
        if (
            retrieval.image_id != str(record["image_id"])
            or retrieval.content_sha256 != str(record["content_sha256"])
            or retrieval.physical_map_sha256 != physical.content_sha256
        ):
            raise ValueError("source retrieval identity/content differs")
        selection = select_fine_supports_under_area_budget(
            retrieval.token_child_rows,
            retrieval.token_child_probabilities,
            retrieval.scene_parent_ids,
            physical,
            maximum_area_fraction=float(args.maximum_area_fraction),
            maximum_children=int(args.maximum_children),
            maximum_primitive_iou=float(args.maximum_child_primitive_iou),
            target_candidate_posterior_mass_fraction=float(
                args.target_candidate_posterior_mass_fraction
            ),
            precomputed_child_surface_area_m2=child_area,
            precomputed_total_map_surface_area_m2=total_map_area,
        )
        components = connected_fine_support_components(
            selection.child_rows,
            physical,
            precomputed_child_surface_area_m2=child_area,
        )
        metadata = dict(retrieval.metadata)
        metadata.pop("content_sha256", None)
        metadata.update(
            {
                "fine_support_selection_semantics": SELECTION_SEMANTICS,
                "maximum_fine_support_area_fraction": float(
                    args.maximum_area_fraction
                ),
                "maximum_scene_children": int(args.maximum_children),
                "maximum_child_primitive_iou": float(
                    args.maximum_child_primitive_iou
                ),
                "selected_fine_support_area_m2": float(
                    selection.selected_surface_area_m2
                ),
                "maximum_fine_support_area_m2": float(
                    selection.maximum_surface_area_m2
                ),
                "fine_support_candidate_count": int(
                    selection.candidate_child_count
                ),
                "children_suppressed_by_primitive_iou_nms": int(
                    selection.suppressed_duplicate_count
                ),
                "source_retrieval_artifact": str(source),
                "source_retrieval_file_sha256": compute_file_sha256(source),
                "source_retrieval_content_sha256": retrieval.content_sha256,
                "selection_uses_ground_truth": False,
                "connected_fine_support_semantics": COMPONENT_SEMANTICS,
                "connected_fine_support_count": int(components.component_count),
                "target_candidate_posterior_mass_fraction": float(
                    args.target_candidate_posterior_mass_fraction
                ),
                "achieved_candidate_posterior_mass_fraction": float(
                    selection.posterior_mass.sum()
                    / max(selection.eligible_posterior_mass, 1e-12)
                ),
            }
        )
        result = PureRadioPhysicalRetrieval(
            **{
                **retrieval.__dict__,
                "scene_child_rows": selection.child_rows,
                "scene_child_scores": selection.posterior_mass,
                "metadata": metadata,
            }
        )
        destination = output_dir / (retrieval.image_id.replace("/", "__") + ".npz")
        if destination.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {destination}")
        _atomic_save(result, destination)
        selected_counts.append(int(selection.child_rows.size))
        area_fractions.append(
            float(selection.selected_surface_area_m2)
            / max(float(selection.maximum_surface_area_m2), 1e-12)
            * float(args.maximum_area_fraction)
        )
        posterior_masses.append(float(selection.posterior_mass.sum()))
        component_counts.append(int(components.component_count))
        output_rows.append(
            {
                "image_id": result.image_id,
                "artifact": str(destination),
                "artifact_sha256": compute_file_sha256(destination),
                "content_sha256": result.content_sha256,
                "source_artifact_sha256": compute_file_sha256(source),
                "selected_child_count": int(selection.child_rows.size),
                "selected_surface_area_m2": float(selection.selected_surface_area_m2),
                "connected_fine_support_count": int(components.component_count),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(
                json.dumps(
                    {"index": index + 1, "count": len(records), "image_id": result.image_id},
                    sort_keys=True,
                ),
                flush=True,
            )

    report: dict[str, object] = {
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "query_count": len(output_rows),
        "retrieval_source": "frozen_token_posterior_reselection",
        "source_retrieval_summaries": [str(path) for path in source_paths],
        "source_retrieval_summary_sha256": [
            compute_file_sha256(path) for path in source_paths
        ],
        "physical_map": str(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "method": "full_radio_36x64_parent_then_area_budgeted_child_set",
        "scene_aggregation": str(records and PureRadioPhysicalRetrieval.load_npz(Path(str(records[0]["artifact"]))).metadata["scene_aggregation"]),
        "fine_support_selection_semantics": SELECTION_SEMANTICS,
        "maximum_fine_support_area_fraction": float(args.maximum_area_fraction),
        "maximum_scene_children": int(args.maximum_children),
        "target_candidate_posterior_mass_fraction": float(
            args.target_candidate_posterior_mass_fraction
        ),
        "maximum_child_primitive_iou": float(args.maximum_child_primitive_iou),
        "mean_selected_child_count": float(sum(selected_counts) / len(selected_counts)),
        "mean_selected_map_area_fraction": float(sum(area_fractions) / len(area_fractions)),
        "mean_selected_token_posterior_mass": float(
            sum(posterior_masses) / len(posterior_masses)
        ),
        "connected_fine_support_semantics": COMPONENT_SEMANTICS,
        "mean_connected_fine_support_count": float(
            sum(component_counts) / len(component_counts)
        ),
        **{key: False for key in required_false},
        "rows": output_rows,
    }
    _atomic_json(report, summary_path)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
