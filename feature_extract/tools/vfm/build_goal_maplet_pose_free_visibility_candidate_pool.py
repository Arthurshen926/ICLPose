"""Build a pose-only candidate pool from pure RADIO retrieval and a frozen atlas.

This is deliberately a pre-GT operation.  It verifies the retrieval run's
negative dependency claims, recomputes atlas scores, applies deterministic
physical NMS, and serializes only candidate poses and pose-free scores.  The
result is compatible with the sparse pose-transport dataset builder, which
will ignore these scores and rerender every candidate from the frozen map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    diverse_pose_rows,
    hierarchical_location_orientation_pose_rows,
    score_visibility_pose_atlas,
)


SCHEMA = "goal_maplet_pose_free_visibility_candidate_pool_v1"
FORBIDDEN_TRUE_FLAGS = (
    "uses_alike", "uses_image_retrieval", "uses_mapping_rgb", "uses_pnp",
    "uses_query_ground_truth", "uses_query_pose", "uses_sfm_points",
    "uses_sfm_tracks",
)


def _select_pose_rows(
    atlas: ChildVisibilityPoseAtlas,
    score: np.ndarray,
    global_score: np.ndarray,
    layout_score: np.ndarray,
    *,
    semantics: str,
    maximum_modes: int,
    translation_nms_m: float,
    rotation_nms_deg: float,
    orientations_per_location: int,
    location_radius_m: float,
    orientation_nms_deg: float,
) -> np.ndarray:
    if semantics == "global_physical_nms_v1":
        return diverse_pose_rows(
            atlas.poses_w2c, global_score, maximum_modes=maximum_modes,
            translation_nms_m=translation_nms_m,
            rotation_nms_deg=rotation_nms_deg,
        )
    if semantics == "hierarchical_location_marginal_orientation_v2":
        return hierarchical_location_orientation_pose_rows(
            atlas.poses_w2c, global_score, layout_score,
            maximum_modes=maximum_modes,
            orientations_per_location=orientations_per_location,
            location_radius_m=location_radius_m,
            orientation_nms_degrees=orientation_nms_deg,
            translation_nms_m=translation_nms_m,
            rotation_nms_deg=rotation_nms_deg,
        )
    raise ValueError(f"unsupported candidate semantics: {semantics}")


def _requested_ids(paths: list[Path]) -> set[str] | None:
    if not paths:
        return None
    result: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload.get("records") or payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("query inventory lacks records/rows")
        for row in rows:
            image_id = str(row.get("image_id", ""))
            if not image_id:
                raise ValueError("query inventory contains an empty image ID")
            result.add(image_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--retrieval_run", action="append", required=True)
    parser.add_argument("--query_inventory", action="append", default=[])
    parser.add_argument("--query_route")
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--candidate_semantics",
        choices=("global_physical_nms_v1", "hierarchical_location_marginal_orientation_v2"),
        default="hierarchical_location_marginal_orientation_v2",
    )
    parser.add_argument("--maximum_modes", type=int, default=8)
    parser.add_argument("--layout_weight", type=float, default=0.0)
    parser.add_argument("--layout_tolerance_cells", type=int, default=0)
    parser.add_argument("--translation_nms_m", type=float, default=0.5)
    parser.add_argument("--rotation_nms_deg", type=float, default=5.0)
    parser.add_argument("--orientations_per_location", type=int, default=2)
    parser.add_argument("--location_radius_m", type=float, default=2.0)
    parser.add_argument("--orientation_nms_deg", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite pose-free candidate pool")
    if args.maximum_modes < 2:
        raise ValueError("candidate pool requires at least two modes")

    atlas_path = Path(args.atlas)
    atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
    matrices = atlas.sparse_matrices()
    requested = _requested_ids([Path(value) for value in args.query_inventory])
    source_rows: dict[str, tuple[Path, dict[str, object], dict[str, object]]] = {}
    run_bindings: list[dict[str, object]] = []
    for value in args.retrieval_run:
        run_path = Path(value)
        run = json.loads(run_path.read_text())
        if run.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval run")
        for flag in FORBIDDEN_TRUE_FLAGS:
            if run.get(flag) is not False:
                raise ValueError(f"pure retrieval dependency flag is not false: {flag}")
        if run.get("physical_map_sha256") != atlas.physical_map_sha256:
            raise ValueError("retrieval and visibility atlas physical maps differ")
        rows = run.get("rows")
        if not isinstance(rows, list):
            raise ValueError("pure retrieval run lacks rows")
        for row in rows:
            image_id = str(row.get("image_id", ""))
            if image_id in source_rows:
                raise ValueError("duplicate query across retrieval runs")
            source_rows[image_id] = (run_path, row, run)
        run_bindings.append({"path": str(run_path.resolve()), "file_sha256": file_sha256(run_path)})

    rows_out: list[dict[str, object]] = []
    for image_id in sorted(source_rows):
        if requested is not None and image_id not in requested:
            continue
        if args.query_route and not image_id.startswith(str(args.query_route) + "/"):
            continue
        run_path, source, _ = source_rows[image_id]
        artifact = Path(str(source["artifact"]))
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
        if retrieval.image_id != image_id:
            raise ValueError("retrieval row and artifact image IDs differ")
        if source.get("artifact_sha256") != file_sha256(artifact):
            raise ValueError("retrieval artifact file hash differs")
        if source.get("content_sha256") != retrieval.content_sha256:
            raise ValueError("retrieval artifact content hash differs")
        score, global_score, layout_score = score_visibility_pose_atlas(
            atlas, retrieval, layout_weight=float(args.layout_weight),
            layout_tolerance_cells=int(args.layout_tolerance_cells),
            selected_children_only=True, matrices=matrices,
        )
        selected = _select_pose_rows(
            atlas, score, global_score, layout_score,
            semantics=str(args.candidate_semantics), maximum_modes=int(args.maximum_modes),
            translation_nms_m=float(args.translation_nms_m),
            rotation_nms_deg=float(args.rotation_nms_deg),
            orientations_per_location=int(args.orientations_per_location),
            location_radius_m=float(args.location_radius_m),
            orientation_nms_deg=float(args.orientation_nms_deg),
        )
        details = []
        for rank, row_index in enumerate(np.asarray(selected, dtype=np.int64).tolist(), start=1):
            details.append({
                "rank": rank,
                "atlas_pose_row": int(row_index),
                "pose_w2c": atlas.poses_w2c[row_index].astype(float).tolist(),
                "score": float(score[row_index]),
                "global_score": float(global_score[row_index]),
                "layout_score": float(layout_score[row_index]),
            })
        rows_out.append({
            "image_id": image_id,
            "mode_details": {"actual_parent_actual_child": details},
            "retrieval_artifact": str(artifact.resolve()),
            "retrieval_content_sha256": retrieval.content_sha256,
            "retrieval_run": str(run_path.resolve()),
        })
    if not rows_out:
        raise ValueError("no requested pure retrieval queries were selected")
    if requested is not None:
        missing = sorted(requested - {str(row["image_id"]) for row in rows_out})
        if args.query_route:
            missing = [value for value in missing if value.startswith(str(args.query_route) + "/")]
        if missing:
            raise ValueError(f"query inventory is absent from retrieval runs: {missing[:3]}")

    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_count": len(rows_out),
        "query_route": args.query_route,
        "atlas": str(atlas_path.resolve()),
        "atlas_file_sha256": file_sha256(atlas_path),
        "atlas_content_sha256": atlas.content_sha256,
        "retrieval_runs": run_bindings,
        "candidate_semantics": str(args.candidate_semantics),
        "maximum_modes": int(args.maximum_modes),
        "layout_weight": float(args.layout_weight),
        "layout_tolerance_cells": int(args.layout_tolerance_cells),
        "translation_nms_m": float(args.translation_nms_m),
        "rotation_nms_deg": float(args.rotation_nms_deg),
        "orientations_per_location": int(args.orientations_per_location),
        "location_radius_m": float(args.location_radius_m),
        "orientation_nms_deg": float(args.orientation_nms_deg),
        "scores_are_pose_free_and_not_consumed_by_transport_builder": True,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_point_correspondences": False,
        "rows": rows_out,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
