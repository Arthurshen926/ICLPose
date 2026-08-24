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
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
    diverse_pose_rows,
    hierarchical_location_orientation_pose_rows,
    progressive_hierarchical_location_orientation_pose_rows,
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
    location_block_size: int = 8,
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
    if semantics == "progressive_hierarchical_location_orientation_v3":
        return progressive_hierarchical_location_orientation_pose_rows(
            atlas.poses_w2c, global_score, layout_score,
            maximum_modes=maximum_modes,
            orientations_per_location=orientations_per_location,
            location_block_size=location_block_size,
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


def _validate_route_disjoint_atlas(
    atlas: ChildVisibilityPoseAtlas, *, query_route: str,
) -> dict[str, object]:
    """Fail closed unless atlas poses come only from an explicit map allowlist."""

    route = str(query_route)
    metadata = dict(atlas.metadata or {})
    if not route:
        raise ValueError("route-disjoint atlas validation requires a query route")
    if metadata.get("route_allowlist_enforced") is not True:
        raise ValueError("visibility atlas lacks an enforced route allowlist")
    allowed = [str(value) for value in metadata.get("allowed_trajectories", ())]
    source_routes = [
        str(value) for value in metadata.get("source_contributor_trajectories", ())
    ]
    image_ids = [str(value) for value in metadata.get("source_contributor_image_ids", ())]
    counts = metadata.get("source_contributor_trajectory_counts")
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or sorted(source_routes) != sorted(allowed)
        or route in allowed
        or not isinstance(counts, dict)
        or sum(int(value) for value in counts.values()) != atlas.view_count
        or len(image_ids) != atlas.view_count
        or len(set(image_ids)) != atlas.view_count
        or int(metadata.get("source_contributor_inventory_count", -1))
        != atlas.view_count
        or len(str(metadata.get("source_contributor_inventory_sha256", ""))) != 64
        or metadata.get("source_contributor_inventory_semantics")
        != "ordered_image_id_resolved_path_file_sha256_v1"
        or metadata.get("coordinate_correct") is not True
        or metadata.get("coordinate_contract") != COORDINATE_CONTRACT
        or metadata.get(
            "coordinate_transform_applied_before_visibility_aggregation"
        ) is not True
        or int(metadata.get("coordinate_audit_count", -1)) != atlas.view_count
        or len(str(metadata.get("coordinate_audits_sha256", ""))) != 64
        or canonical_json_sha256(image_ids)
        != metadata.get("source_contributor_image_ids_sha256")
    ):
        raise ValueError("visibility atlas route allowlist lineage is inconsistent")
    derived_routes = [value.split("/", 1)[0] for value in image_ids]
    if (
        sorted(set(derived_routes)) != sorted(source_routes)
        or any(value == route for value in derived_routes)
        or {
            value: derived_routes.count(value) for value in sorted(set(derived_routes))
        } != {str(key): int(value) for key, value in counts.items()}
    ):
        raise ValueError("visibility atlas source IDs violate route disjointness")
    return {
        "route_allowlist_enforced": True,
        "query_route_excluded_from_atlas": True,
        "allowed_trajectories": sorted(allowed),
        "source_contributor_trajectory_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "source_contributor_image_ids_sha256": str(
            metadata["source_contributor_image_ids_sha256"]
        ),
        "source_contributor_inventory_sha256": str(
            metadata["source_contributor_inventory_sha256"]
        ),
        "coordinate_correct": True,
        "coordinate_contract": COORDINATE_CONTRACT,
        "coordinate_audits_sha256": str(metadata["coordinate_audits_sha256"]),
    }


def _validate_promotion_eligible_retrieval_run(
    run: dict[str, object],
    *,
    query_route: str,
    atlas: ChildVisibilityPoseAtlas,
) -> dict[str, object]:
    """Fail closed on the strict fit/val/calibration/test route split."""

    route = str(query_route)
    split = run.get("query_split_audit")
    atlas_routes = {
        str(value) for value in (atlas.metadata or {}).get(
            "allowed_trajectories", (),
        )
    }
    if (
        not route or not isinstance(split, dict)
        or run.get("promotion_eligible") is not True
        or run.get("control_only") is not False
        or list(run.get("promotion_blockers", ()))
        or split.get("disjoint") is not True
        or list(split.get("blockers", ()))
    ):
        raise ValueError("retrieval run is not strict-promotion eligible")
    query_routes = {str(value) for value in split.get("query_trajectory_ids", ())}
    calibration_routes = {
        str(value) for value in split.get(
            "validity_calibration_fit_trajectory_ids", (),
        )
    }
    canonical_routes = {
        str(value) for value in split.get("canonical_mapping_trajectory_ids", ())
    }
    mapper_routes = {
        str(value) for value in split.get("mapper_training_trajectory_ids", ())
    } | {
        str(value) for value in split.get("mapper_validation_trajectory_ids", ())
    }
    excluded_routes = {
        str(value) for value in split.get("canonical_excluded_trajectory_ids", ())
    }
    if (
        query_routes != {route}
        or calibration_routes != {"seq10"}
        or canonical_routes != atlas_routes
        or mapper_routes != atlas_routes
        or not {"seq10", "seq12", "seq14"}.issubset(excluded_routes)
        or route in atlas_routes or route in calibration_routes
        or calibration_routes & atlas_routes
    ):
        raise ValueError("retrieval run route split differs from the strict atlas contract")
    hashes = {
        key: str(run.get(key, ""))
        for key in (
            "validity_calibration_sha256", "canonical_field_sha256",
            "field_feature_contract_sha256", "physical_map_sha256",
        )
    }
    if any(len(value) != 64 for value in hashes.values()):
        raise ValueError("retrieval run strict lineage hashes are absent")
    return {
        "promotion_eligible": True,
        "control_only": False,
        "promotion_blockers": [],
        "query_route": route,
        "query_split_disjoint": True,
        "atlas_routes": sorted(atlas_routes),
        "validity_calibration_fit_trajectories": ["seq10"],
        "canonical_excluded_trajectories": sorted(excluded_routes),
        **hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--retrieval_run", action="append", required=True)
    parser.add_argument("--query_inventory", action="append", default=[])
    parser.add_argument("--query_route")
    parser.add_argument(
        "--require_route_disjoint_atlas", action="store_true",
        help="require an explicit atlas map-route allowlist excluding query_route",
    )
    parser.add_argument(
        "--require_promotion_eligible_retrieval", action="store_true",
        help="require the strict seq10-calibration/query-route-disjoint run audit",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--candidate_semantics",
        choices=(
            "global_physical_nms_v1",
            "hierarchical_location_marginal_orientation_v2",
            "progressive_hierarchical_location_orientation_v3",
        ),
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
    parser.add_argument(
        "--location_block_size", type=int, default=8,
        help=(
            "fixed number of new location seeds per progressive v3 stage; "
            "ignored by legacy candidate semantics"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite pose-free candidate pool")
    if args.maximum_modes < 2:
        raise ValueError("candidate pool requires at least two modes")

    atlas_path = Path(args.atlas)
    atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
    route_audit = None
    if bool(args.require_route_disjoint_atlas):
        route_audit = _validate_route_disjoint_atlas(
            atlas, query_route=str(args.query_route or ""),
        )
    matrices = atlas.sparse_matrices()
    requested = _requested_ids([Path(value) for value in args.query_inventory])
    source_rows: dict[str, tuple[Path, dict[str, object], dict[str, object]]] = {}
    run_bindings: list[dict[str, object]] = []
    strict_retrieval_audits: list[dict[str, object]] = []
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
        if bool(args.require_promotion_eligible_retrieval):
            strict_retrieval_audits.append(
                _validate_promotion_eligible_retrieval_run(
                    run, query_route=str(args.query_route or ""), atlas=atlas,
                )
            )
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
            location_block_size=int(args.location_block_size),
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
        "route_disjoint_atlas_audit": route_audit,
        "strict_retrieval_promotion_required": bool(
            args.require_promotion_eligible_retrieval
        ),
        "strict_retrieval_promotion_audits": strict_retrieval_audits,
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
    if str(args.candidate_semantics) == "progressive_hierarchical_location_orientation_v3":
        report["location_block_size"] = int(args.location_block_size)
        report["candidate_prefix_stable_across_budgets"] = True
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
