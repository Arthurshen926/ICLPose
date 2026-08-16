"""Merge exact-identity and evidence-coverage child queues under one area cap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.tools.vfm.reselect_goal_maplet_pure_radio_fine_support import (
    _atomic_save,
)
from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    CHILD_PROBABILITY_SEMANTICS,
    child_surface_area_m2,
    total_map_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.tokens import compute_file_sha256


SELECTION_SEMANTICS = "equal_area_exact_identity_and_evidence_coverage_queues_v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage_summary", required=True)
    parser.add_argument("--identity_summary", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_area_fraction", type=float, default=0.10)
    parser.add_argument("--maximum_children", type=int, default=2048)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def interleave_equal_area_queues(
    coverage_rows: np.ndarray,
    identity_rows: np.ndarray,
    child_area_m2: np.ndarray,
    *,
    maximum_area_m2: float,
    maximum_children: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable two-queue union; queue choice depends only on charged area."""

    queues = (
        np.asarray(coverage_rows, dtype=np.int64).reshape(-1),
        np.asarray(identity_rows, dtype=np.int64).reshape(-1),
    )
    area = np.asarray(child_area_m2, dtype=np.float64).reshape(-1)
    if (
        any(np.any((queue < 0) | (queue >= area.size)) for queue in queues)
        or any(np.unique(queue).size != queue.size for queue in queues)
        or np.any(~np.isfinite(area))
        or np.any(area <= 0.0)
        or not np.isfinite(float(maximum_area_m2))
        or float(maximum_area_m2) <= 0.0
        or int(maximum_children) <= 0
    ):
        raise ValueError("invalid equal-area child queues")
    positions = [0, 0]
    charged = [0.0, 0.0]
    used = 0.0
    selected: list[int] = []
    selected_source: list[int] = []
    seen: set[int] = set()
    while len(selected) < int(maximum_children):
        available = [index for index in (0, 1) if positions[index] < queues[index].size]
        if not available:
            break
        source = min(available, key=lambda index: (charged[index], index))
        row = int(queues[source][positions[source]])
        positions[source] += 1
        if row in seen:
            continue
        cost = float(area[row])
        if used + cost > float(maximum_area_m2) + 1e-12:
            # This item cannot fit, but a later smaller item may fit.
            continue
        selected.append(row)
        selected_source.append(source)
        seen.add(row)
        charged[source] += cost
        used += cost
    return np.asarray(selected, dtype=np.int64), np.asarray(selected_source, dtype=np.int8)


def _rows(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    values = {str(row["image_id"]): row for row in summary.get("rows", [])}
    if len(values) != len(list(summary.get("rows", []))):
        raise ValueError("retrieval summary identities are duplicated")
    return values


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    coverage_path = Path(args.coverage_summary).resolve()
    identity_path = Path(args.identity_summary).resolve()
    physical_path = Path(args.physical_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite two-queue summary")
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    area = child_surface_area_m2(physical)
    maximum_area = float(args.maximum_area_fraction) * total_map_surface_area_m2(physical)
    if not 0.0 < float(args.maximum_area_fraction) <= 1.0:
        raise ValueError("maximum area fraction must lie in (0,1]")
    coverage_summary = _json_without_duplicates(coverage_path)
    identity_summary = _json_without_duplicates(identity_path)
    for summary in (coverage_summary, identity_summary):
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval summary")
        if str(summary.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("retrieval and physical map differ")
    coverage = _rows(coverage_summary)
    identity = _rows(identity_summary)
    if sorted(coverage) != sorted(identity):
        raise ValueError("two-queue query inventories differ")
    if int(args.expected_queries) > 0 and len(coverage) != int(args.expected_queries):
        raise ValueError("query count differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    source_counts = np.zeros((2,), dtype=np.int64)
    for image_id in sorted(coverage):
        rows = (coverage[image_id], identity[image_id])
        retrievals: list[PureRadioPhysicalRetrieval] = []
        for record in rows:
            path = Path(str(record["artifact"])).resolve()
            if compute_file_sha256(path) != str(record["artifact_sha256"]):
                raise ValueError("two-queue source file hash differs")
            result = PureRadioPhysicalRetrieval.load_npz(path)
            if result.image_id != image_id or result.content_sha256 != str(record["content_sha256"]):
                raise ValueError("two-queue source identity/content differs")
            retrievals.append(result)
        left, right = retrievals
        for key in (
            "token_xy", "token_parent_ids", "token_parent_probabilities",
            "token_out_of_map_probabilities", "token_in_map_tail_probabilities",
            "token_child_rows", "token_child_probabilities", "scene_parent_ids",
            "scene_parent_scores",
        ):
            if not np.array_equal(getattr(left, key), getattr(right, key)):
                raise ValueError(f"two-queue frozen retrieval field differs: {key}")
        selected, selected_source = interleave_equal_area_queues(
            left.scene_child_rows,
            right.scene_child_rows,
            area,
            maximum_area_m2=maximum_area,
            maximum_children=int(args.maximum_children),
        )
        source_counts += np.bincount(selected_source, minlength=2)
        score = np.linspace(1.0, 0.5, selected.size, dtype=np.float32)
        components = connected_fine_support_components(selected, physical)
        metadata = dict(left.metadata)
        metadata.pop("content_sha256", None)
        metadata.update(
            {
                "fine_support_selection_semantics": SELECTION_SEMANTICS,
                "maximum_fine_support_area_fraction": float(args.maximum_area_fraction),
                "selected_fine_support_area_m2": float(np.sum(area[selected])),
                "connected_fine_support_count": int(components.component_count),
                "coverage_queue_content_sha256": left.content_sha256,
                "identity_queue_content_sha256": right.content_sha256,
                "crossfit_held_sequence": right.metadata.get("crossfit_held_sequence"),
                "crossfit_training_sequences": right.metadata.get("crossfit_training_sequences"),
                "crossfit_uses_held_sequence_labels": False,
                "selection_uses_ground_truth": False,
                "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
                "child_probability_is_calibrated_credible_mass": False,
                "development_530_not_untouched_test": True,
            }
        )
        result = PureRadioPhysicalRetrieval(
            **{
                **left.__dict__,
                "scene_child_rows": selected,
                "scene_child_scores": score,
                "metadata": metadata,
            }
        )
        destination = output_dir / (image_id.replace("/", "__") + ".npz")
        if destination.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {destination}")
        _atomic_save(result, destination)
        output_rows.append(
            {
                "image_id": image_id,
                "artifact": str(destination),
                "artifact_sha256": compute_file_sha256(destination),
                "content_sha256": result.content_sha256,
                "selected_child_count": int(selected.size),
                "selected_surface_area_m2": float(np.sum(area[selected])),
                "connected_fine_support_count": int(components.component_count),
            }
        )
    required_false = {
        "uses_query_pose": False, "uses_query_ground_truth": False,
        "uses_alike": False, "uses_pnp": False, "uses_sfm_points": False,
        "uses_sfm_tracks": False, "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
    }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "query_count": len(output_rows),
        "method": "equal_area_exact_identity_and_evidence_coverage_queues",
        "scene_aggregation": "fixed_4x4_blocks_top4_max_sum_v1",
        "physical_map": str(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "coverage_summary": str(coverage_path),
        "coverage_summary_sha256": compute_file_sha256(coverage_path),
        "identity_summary": str(identity_path),
        "identity_summary_sha256": compute_file_sha256(identity_path),
        "fine_support_selection_semantics": SELECTION_SEMANTICS,
        "maximum_fine_support_area_fraction": float(args.maximum_area_fraction),
        "selected_from_coverage_queue": int(source_counts[0]),
        "selected_from_identity_queue": int(source_counts[1]),
        "development_530_not_untouched_test": True,
        **required_false,
        "rows": output_rows,
    }
    _atomic_json(report, summary_path)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

