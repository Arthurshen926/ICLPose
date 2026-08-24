"""Fuse strict baseline and parent-balanced scene-child retrieval branches.

Only the pose-free ``scene_child_rows/scores`` gate is fused.  Every token
posterior and scene-parent array must be bit-identical across branches.  The
union is a deterministic rank-paired interleave with child-row de-duplication;
query pose, labels, rendering, ALIKE, and PnP are outside this executable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _json_without_duplicates,
)
from feature_extract.vfm.localization_goal_maplet.hierarchical_child_allocator import (
    SEMANTICS as F50_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)


UNION_SEMANTICS = "stable_rank_paired_baseline_f50_scene_child_union_dedup_v1"
RUN_SCHEMA = "goal_maplet_pure_radio_retrieval_run_v1"
_FALSE_FLAGS = (
    "uses_alike", "uses_pnp", "uses_query_pose", "uses_query_ground_truth",
    "uses_sfm_points", "uses_sfm_tracks", "uses_mapping_rgb",
    "uses_image_retrieval",
)
_FROZEN_ARRAYS = (
    "token_xy", "token_parent_ids", "token_parent_probabilities",
    "token_out_of_map_probabilities", "token_in_map_tail_probabilities",
    "token_child_rows", "token_child_probabilities", "scene_parent_ids",
    "scene_parent_scores",
)
_SUMMARY_LINEAGE = (
    "physical_map_sha256", "canonical_field_sha256",
    "field_feature_contract_sha256", "validity_calibration_sha256",
    "query_manifest_sha256", "canonical_field_coordinate_contract",
    "canonical_field_coordinate_correct", "parent_score_semantics",
    "parent_scene_ranking_semantics", "parent_mode_temperature",
    "child_probability_semantics", "child_probability_is_calibrated_credible_mass",
)


def stable_rank_paired_scene_child_union(
    baseline_rows: np.ndarray,
    baseline_scores: np.ndarray,
    f50_rows: np.ndarray,
    f50_scores: np.ndarray,
    *,
    maximum_children: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Return B1,F1,B2,F2,... with first-occurrence de-duplication."""

    branch_rows = [
        np.asarray(baseline_rows, dtype=np.int64).reshape(-1),
        np.asarray(f50_rows, dtype=np.int64).reshape(-1),
    ]
    branch_scores = [
        np.asarray(baseline_scores, dtype=np.float32).reshape(-1),
        np.asarray(f50_scores, dtype=np.float32).reshape(-1),
    ]
    if (
        any(rows.shape != scores.shape for rows, scores in zip(branch_rows, branch_scores))
        or any(np.unique(rows).size != rows.size for rows in branch_rows)
        or any(np.any(rows < 0) for rows in branch_rows)
        or any(np.any(~np.isfinite(scores)) for scores in branch_scores)
        or int(maximum_children) <= 0
    ):
        raise ValueError("scene-child branches are invalid")
    rank_by_branch = [
        {int(row): rank + 1 for rank, row in enumerate(rows.tolist())}
        for rows in branch_rows
    ]
    score_by_branch = [
        {int(row): float(scores[rank]) for rank, row in enumerate(rows.tolist())}
        for rows, scores in zip(branch_rows, branch_scores)
    ]
    overlap = set(rank_by_branch[0]) & set(rank_by_branch[1])
    if any(
        np.asarray(score_by_branch[0][row], dtype=np.float32).tobytes()
        != np.asarray(score_by_branch[1][row], dtype=np.float32).tobytes()
        for row in overlap
    ):
        raise ValueError("overlapping scene-child branch scores are not bit-identical")
    selected: list[int] = []
    seen: set[int] = set()
    maximum_rank = max((rows.size for rows in branch_rows), default=0)
    for rank in range(maximum_rank):
        for rows in branch_rows:
            if rank >= rows.size:
                continue
            child = int(rows[rank])
            if child not in seen:
                selected.append(child)
                seen.add(child)
                if len(selected) >= int(maximum_children):
                    break
        if len(selected) >= int(maximum_children):
            break
    provenance = []
    for output_rank, child in enumerate(selected, start=1):
        baseline_rank = rank_by_branch[0].get(child)
        f50_rank = rank_by_branch[1].get(child)
        provenance.append({
            "union_rank": output_rank,
            "child_row": child,
            "baseline_rank": baseline_rank,
            "f50_rank": f50_rank,
            "branches": [
                name for name, rank in (
                    ("baseline", baseline_rank), ("f50", f50_rank)
                ) if rank is not None
            ],
        })
    scores = np.asarray([
        score_by_branch[0][child]
        if child in score_by_branch[0]
        else score_by_branch[1][child]
        for child in selected
    ], dtype=np.float32)
    return np.asarray(selected, dtype=np.int64), scores, provenance


def _validate_summary(summary: dict[str, object], *, allocator: bool) -> None:
    split = summary.get("query_split_audit")
    if (
        summary.get("artifact_type") != RUN_SCHEMA
        or summary.get("promotion_eligible") is not True
        or summary.get("control_only") is not False
        or list(summary.get("promotion_blockers", ()))
        or not isinstance(split, dict)
        or split.get("disjoint") is not True
        or list(split.get("blockers", ()))
        or any(summary.get(flag) is not False for flag in _FALSE_FLAGS)
    ):
        raise ValueError("retrieval branch is not strict/promotion eligible")
    if allocator and (
        summary.get("scene_child_selection_semantics") != F50_SEMANTICS
        or summary.get("hierarchical_child_allocator_tuning_route") != "seq10"
        or split.get("allocator_tuning_query_disjoint") is not True
        or split.get("allocator_tuning_trajectory_ids") != ["seq10"]
        or len(str(summary.get(
            "hierarchical_child_allocator_config_file_sha256", ""
        ))) != 64
        or len(str(summary.get(
            "hierarchical_child_allocator_config_content_sha256", ""
        ))) != 64
    ):
        raise ValueError("f50 retrieval branch allocator lineage differs")


def _rows_from_summaries(
    paths: list[Path], *, allocator: bool,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    result: dict[str, dict[str, object]] = {}
    summaries = []
    for path in paths:
        summary = _json_without_duplicates(path)
        _validate_summary(summary, allocator=allocator)
        summaries.append(summary)
        rows = summary.get("rows")
        if not isinstance(rows, list):
            raise ValueError("retrieval branch lacks rows")
        for row in rows:
            image_id = str(row.get("image_id", ""))
            if not image_id or image_id in result:
                raise ValueError("retrieval branch query inventory repeats")
            result[image_id] = row
    return result, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_summary", action="append", required=True)
    parser.add_argument("--f50_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_union_children", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite branch-union retrieval summary")
    baseline_paths = [Path(value).resolve() for value in args.baseline_summary]
    f50_path = Path(args.f50_summary).resolve()
    baseline_rows, baseline_summaries = _rows_from_summaries(
        baseline_paths, allocator=False,
    )
    f50_rows, f50_summaries = _rows_from_summaries(
        [f50_path], allocator=True,
    )
    if set(baseline_rows) != set(f50_rows):
        raise ValueError("baseline/f50 retrieval query inventories differ")
    template = f50_summaries[0]
    for summary in [*baseline_summaries, *f50_summaries]:
        if any(summary.get(key) != template.get(key) for key in _SUMMARY_LINEAGE):
            raise ValueError("baseline/f50 retrieval summary lineage differs")
    baseline_split = baseline_summaries[0]["query_split_audit"]
    f50_split = template["query_split_audit"]
    for key in (
        "canonical_mapping_trajectory_ids", "canonical_excluded_trajectory_ids",
        "mapper_training_trajectory_ids", "mapper_validation_trajectory_ids",
        "query_trajectory_ids", "validity_calibration_fit_trajectory_ids",
    ):
        if baseline_split.get(key) != f50_split.get(key):
            raise ValueError("baseline/f50 retrieval route splits differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    counts = []
    for image_id in sorted(baseline_rows):
        baseline_row = baseline_rows[image_id]
        f50_row = f50_rows[image_id]
        baseline_path = Path(str(baseline_row["artifact"])).resolve()
        f50_artifact_path = Path(str(f50_row["artifact"])).resolve()
        baseline_file_hash = file_sha256(baseline_path)
        f50_file_hash = file_sha256(f50_artifact_path)
        if (
            baseline_file_hash != baseline_row.get("artifact_sha256")
            or f50_file_hash != f50_row.get("artifact_sha256")
            or Path(str(f50_row.get("base_artifact", ""))).resolve() != baseline_path
            or f50_row.get("base_artifact_sha256") != baseline_file_hash
        ):
            raise ValueError("f50 row does not bind its baseline source")
        baseline = PureRadioPhysicalRetrieval.load_npz(baseline_path)
        f50 = PureRadioPhysicalRetrieval.load_npz(f50_artifact_path)
        if (
            baseline.image_id != image_id or f50.image_id != image_id
            or baseline.content_sha256 != baseline_row.get("content_sha256")
            or f50.content_sha256 != f50_row.get("content_sha256")
            or baseline.physical_map_sha256 != f50.physical_map_sha256
            or any(
                not np.array_equal(getattr(baseline, name), getattr(f50, name))
                for name in _FROZEN_ARRAYS
            )
            or f50.metadata.get("base_retrieval_content_sha256")
            != baseline.content_sha256
            or f50.metadata.get("base_retrieval_file_sha256") != baseline_file_hash
            or f50.metadata.get("hierarchical_child_allocator_tuning_route") != "seq10"
            or f50.metadata.get("uses_query_pose") is not False
            or f50.metadata.get("uses_query_ground_truth") is not False
        ):
            raise ValueError("baseline/f50 per-query frozen retrieval inputs differ")
        union_rows, union_scores, provenance = stable_rank_paired_scene_child_union(
            baseline.scene_child_rows, baseline.scene_child_scores,
            f50.scene_child_rows, f50.scene_child_scores,
            maximum_children=int(args.maximum_union_children),
        )
        uncompressed_union_count = int(np.unique(np.concatenate([
            baseline.scene_child_rows, f50.scene_child_rows,
        ])).size)
        metadata = dict(f50.metadata)
        metadata.pop("content_sha256", None)
        metadata.update({
            "scene_child_selection_semantics": UNION_SEMANTICS,
            "scene_child_union_baseline_content_sha256": baseline.content_sha256,
            "scene_child_union_baseline_file_sha256": baseline_file_hash,
            "scene_child_union_f50_content_sha256": f50.content_sha256,
            "scene_child_union_f50_file_sha256": f50_file_hash,
            "scene_child_union_baseline_count": int(baseline.scene_child_rows.size),
            "scene_child_union_f50_count": int(f50.scene_child_rows.size),
            "scene_child_union_count": int(union_rows.size),
            "scene_child_union_maximum_children": int(args.maximum_union_children),
            "scene_child_union_uncompressed_count": uncompressed_union_count,
            "scene_child_union_compression": (
                "none"
                if union_rows.size == uncompressed_union_count
                else "stable_rank_paired_prefix"
            ),
            "scene_child_union_provenance": provenance,
            "scene_child_union_uses_query_pose": False,
            "scene_child_union_uses_query_ground_truth": False,
            "scene_child_union_uses_held_labels": False,
            "promotion_eligible": True,
            "promotion_blockers": [],
            "control_only": False,
        })
        result = PureRadioPhysicalRetrieval(
            **{
                **baseline.__dict__,
                "scene_child_rows": union_rows,
                "scene_child_scores": union_scores,
                "metadata": metadata,
            }
        )
        output = output_dir / baseline_path.name
        if output.exists() and not bool(args.force):
            raise FileExistsError(f"refusing to overwrite {output}")
        result.save_npz(output)
        counts.append(int(union_rows.size))
        output_rows.append({
            "image_id": image_id,
            "artifact": str(output),
            "artifact_sha256": file_sha256(output),
            "content_sha256": result.content_sha256,
            "baseline_artifact": str(baseline_path),
            "baseline_artifact_sha256": baseline_file_hash,
            "baseline_content_sha256": baseline.content_sha256,
            "f50_artifact": str(f50_artifact_path),
            "f50_artifact_sha256": f50_file_hash,
            "f50_content_sha256": f50.content_sha256,
            "baseline_child_count": int(baseline.scene_child_rows.size),
            "f50_child_count": int(f50.scene_child_rows.size),
            "union_child_count": int(union_rows.size),
        })
    summary = {
        **{key: value for key, value in template.items() if key != "rows"},
        "query_count": len(output_rows),
        "rows": output_rows,
        "shard_count": 1,
        "shard_index": 0,
        "source_baseline_retrieval_summaries": [str(path) for path in baseline_paths],
        "source_baseline_retrieval_summary_sha256": [
            file_sha256(path) for path in baseline_paths
        ],
        "source_f50_retrieval_summary": str(f50_path),
        "source_f50_retrieval_summary_sha256": file_sha256(f50_path),
        "scene_child_selection_semantics": UNION_SEMANTICS,
        "scene_child_union_maximum_children": int(args.maximum_union_children),
        "scene_child_union_count_range": [min(counts), max(counts)],
        "scene_child_union_count_mean": float(np.mean(counts)),
        "final_pose_pool_budget": 64,
        "final_factorized_position_seed_budget": 4,
        "final_factorized_orientation_budget": 64,
        "union_uses_query_pose": False,
        "union_uses_query_ground_truth": False,
        "union_uses_held_labels": False,
        "promotion_eligible": True,
        "promotion_blockers": [],
        "control_only": False,
    }
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps({
        "summary_json": str(summary_path),
        "query_count": len(output_rows),
        "union_child_count_range": [min(counts), max(counts)],
        "union_child_count_mean": float(np.mean(counts)),
        "union_uses_held_labels": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
