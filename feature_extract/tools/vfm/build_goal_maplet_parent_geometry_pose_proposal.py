"""Freeze query-parent-conditioned position cells crossed with analytic SO(3)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    LATTICE_SPACING_M,
    MAXIMUM_PARENT_PREFIX,
    MAXIMUM_UNIQUE_CELLS_PER_QUERY,
    ORIENTATION_COUNT,
    PARENT_AABB_EXPANSION_LINF_M,
    PARENT_PREFIX_BUDGETS,
    SCHEMA,
    SEMANTICS,
    _parent_cells,
    build_parent_geometry_proposal_arrays,
    exact_parent_primitive_rectangle_aabbs,
    lattice_origin_from_parent_geometry,
    load_parent_geometry_proposal,
    orientation_cover_certificate,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)


FROZEN_PADDING_SOURCE_CONTENT_SHA256 = (
    "068557fa7f4758d17da5c8cf0b17acec365627f0706ec666f1a1bc9a34fd57d6"
)
FROZEN_PADDING_CONFIGURATION_SHA256 = (
    "53ee5152469389be70a887e78a1b77e5028ab70bec626aaabae52a3064288088"
)
FORBIDDEN_TRUE_FLAGS = (
    "uses_alike", "uses_image_retrieval", "uses_mapping_rgb", "uses_pnp",
    "uses_query_ground_truth", "uses_query_pose", "uses_sfm_points",
    "uses_sfm_tracks",
)


def _atomic_save(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", action="append", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--query_route", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite parent-geometry proposal")
    route = str(args.query_route)
    if route not in {"seq10", "seq12", "seq14"}:
        raise ValueError("parent-geometry proposal route is outside the frozen screen")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    if physical.metadata.get("primitive_membership_coverage_fraction") != 1.0:
        raise ValueError("physical map lacks complete primitive membership")
    summary_bindings = []
    records = []
    for value in args.retrieval_summary:
        path = Path(value).resolve()
        run = json.loads(path.read_text())
        audit = run.get("query_split_audit", {})
        query_routes = {str(item) for item in audit.get("query_trajectory_ids", ())}
        if (
            run.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1"
            or run.get("physical_map_sha256") != physical.content_sha256
            or query_routes != {route}
            or any(run.get(flag) is not False for flag in FORBIDDEN_TRUE_FLAGS)
        ):
            raise ValueError("parent-geometry retrieval dependency contract differs")
        if route == "seq10":
            if (
                run.get("promotion_eligible") is not False
                or run.get("control_only") is not True
                or list(run.get("promotion_blockers", ()))
                != ["query_route_used_for_validity_calibration"]
            ):
                raise ValueError("seq10 parent-geometry screen must remain calibration control")
        elif (
            run.get("promotion_eligible") is not True
            or run.get("control_only") is not False
            or audit.get("disjoint") is not True
            or list(run.get("promotion_blockers", ()))
        ):
            raise ValueError("held parent-geometry retrieval is not strict-promotion eligible")
        records.extend(list(run.get("rows", ())))
        summary_bindings.append({
            "path": str(path), "file_sha256": file_sha256(path),
        })
    records.sort(key=lambda row: str(row.get("image_id", "")))
    image_ids = [str(row.get("image_id", "")) for row in records]
    if (
        not records or len(set(image_ids)) != len(records)
        or any(not value.startswith(route + "/") for value in image_ids)
    ):
        raise ValueError("parent-geometry retrieval query inventory differs")
    parent_ids, parent_scores = [], []
    artifact_bindings = []
    load_started = time.perf_counter()
    for record in records:
        artifact = Path(str(record["artifact"])).resolve()
        if file_sha256(artifact) != str(record["artifact_sha256"]):
            raise ValueError("parent-geometry retrieval artifact bytes differ")
        retrieval = PureRadioPhysicalRetrieval.load_npz(artifact)
        if (
            retrieval.content_sha256 != str(record["content_sha256"])
            or retrieval.image_id != str(record["image_id"])
            or retrieval.scene_parent_ids.size < MAXIMUM_PARENT_PREFIX
            or np.unique(
                retrieval.scene_parent_ids[:MAXIMUM_PARENT_PREFIX]
            ).size != MAXIMUM_PARENT_PREFIX
            or np.any(
                np.diff(retrieval.scene_parent_scores[:MAXIMUM_PARENT_PREFIX])
                > 1.0e-12
            )
        ):
            raise ValueError("parent-geometry retrieval parent prefix differs")
        parent_ids.append(np.asarray(
            retrieval.scene_parent_ids[:MAXIMUM_PARENT_PREFIX], dtype=np.int64,
        ))
        parent_scores.append(np.asarray(
            retrieval.scene_parent_scores[:MAXIMUM_PARENT_PREFIX], dtype=np.float64,
        ))
        artifact_bindings.append({
            "image_id": retrieval.image_id,
            "file_sha256": str(record["artifact_sha256"]),
            "content_sha256": retrieval.content_sha256,
        })
    retrieval_load_seconds = float(time.perf_counter() - load_started)

    parent_lower, parent_upper = exact_parent_primitive_rectangle_aabbs(physical)
    origin = lattice_origin_from_parent_geometry(parent_lower)
    maximum_single_parent_cells = 0
    for lower, upper in zip(parent_lower, parent_upper):
        first, last = _parent_cells(lower, upper, origin)
        maximum_single_parent_cells = max(
            maximum_single_parent_cells, int(np.prod(last - first + 1)),
        )
    map_wide_no_dedup_upper_bound = (
        maximum_single_parent_cells * MAXIMUM_PARENT_PREFIX
    )
    if map_wide_no_dedup_upper_bound > MAXIMUM_UNIQUE_CELLS_PER_QUERY:
        raise ValueError("fixed-map parent-cell upper bound exceeds the fail-closed cap")
    build_started = time.perf_counter()
    arrays = build_parent_geometry_proposal_arrays(
        np.asarray(image_ids), np.stack(parent_ids), np.stack(parent_scores), physical,
    )
    generation_seconds = float(time.perf_counter() - build_started)
    position_counts = arrays["unique_position_count_by_parent_prefix"]
    implicit_counts = arrays["implicit_pose_factor_count_by_parent_prefix"]
    certificate = orientation_cover_certificate()
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_route": route,
        "query_count": int(arrays["image_ids"].size),
        "retrieval_summaries": summary_bindings,
        "retrieval_artifact_inventory": artifact_bindings,
        "physical_map": str(physical_path),
        "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_content_sha256": physical.content_sha256,
        "parent_prefix_budgets": list(PARENT_PREFIX_BUDGETS),
        "parent_prefix_order_semantics": "scene_parent_score_descending_strict_prefix_v1",
        "parent_aabb_semantics": (
            "complete_membership_union_of_primitive_center_plus_or_minus_"
            "abs_tangent1_times_scale1_plus_abs_tangent2_times_scale2_v1"
        ),
        "primitive_rectangle_support_sigma": 1.0,
        "primitive_rectangle_support_sigma_source": (
            "physical_map_stores_scale1_scale2_without_an_additional_support_sigma"
        ),
        "parent_aabb_expansion_linf_m": PARENT_AABB_EXPANSION_LINF_M,
        "parent_aabb_expansion_status": (
            "preexisting_frozen_development_assumption_not_a_true_pose_containment_theorem"
        ),
        "parent_aabb_expansion_source_contract_content_sha256": (
            FROZEN_PADDING_SOURCE_CONTENT_SHA256
        ),
        "parent_aabb_expansion_source_frozen_configuration_sha256": (
            FROZEN_PADDING_CONFIGURATION_SHA256
        ),
        "lattice_spacing_m": LATTICE_SPACING_M,
        "lattice_cell_half_diagonal_m": float(
            math.sqrt(3.0) * LATTICE_SPACING_M / 2.0
        ),
        "translation_cover_proof": (
            "every_point_in_an_included_closed_2m_world_axis_cell_is_at_most_"
            "sqrt3_m_from_its_center_strictly_below_2m"
        ),
        "orientation_count": ORIENTATION_COUNT,
        "orientation_cover_certificate": certificate,
        "joint_2m45_cover_scope": (
            "conditional_on_query_camera_center_belonging_to_the_union_of_"
            "selected_parent_expanded_aabbs"
        ),
        "maximum_single_parent_cell_count_for_fixed_map": maximum_single_parent_cells,
        "maximum_16_parent_no_dedup_cell_upper_bound_for_fixed_map": (
            map_wide_no_dedup_upper_bound
        ),
        "maximum_unique_cells_per_query": MAXIMUM_UNIQUE_CELLS_PER_QUERY,
        "hard_cap_behavior": "fail_closed_never_truncate",
        "unique_position_count_range_by_parent_prefix": {
            str(budget): [
                int(np.min(position_counts[:, index])),
                int(np.max(position_counts[:, index])),
            ]
            for index, budget in enumerate(PARENT_PREFIX_BUDGETS)
        },
        "unique_position_count_mean_by_parent_prefix": {
            str(budget): float(np.mean(position_counts[:, index]))
            for index, budget in enumerate(PARENT_PREFIX_BUDGETS)
        },
        "implicit_pose_factor_count_range_by_parent_prefix": {
            str(budget): [
                int(np.min(implicit_counts[:, index])),
                int(np.max(implicit_counts[:, index])),
            ]
            for index, budget in enumerate(PARENT_PREFIX_BUDGETS)
        },
        "retrieval_load_seconds": retrieval_load_seconds,
        "position_generation_seconds": generation_seconds,
        "uses_mapping_camera_position_seed": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "uses_signed_surface_normal": False,
        "uses_gravity_for_position_exclusion": False,
        "uses_clearance_or_occupancy_pruning": False,
        "finite_2dgs_watertight_free_space_authority": False,
        "position_cells_are_physical_free_space": False,
        "cross_position_orientation_product_materialized": False,
        "direct_label_or_contributor_opened_during_generation": False,
        "raw_support_is_implicit_upper_bound_not_localization_success": True,
        "production_eligible": False,
    }
    _atomic_save(output, arrays, metadata)
    load_parent_geometry_proposal(output)
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "content_sha256": metadata["content_sha256"],
        "query_route": route,
        "query_count": metadata["query_count"],
        "unique_position_count_range_by_parent_prefix": metadata[
            "unique_position_count_range_by_parent_prefix"
        ],
        "unique_position_count_mean_by_parent_prefix": metadata[
            "unique_position_count_mean_by_parent_prefix"
        ],
        "maximum_16_parent_no_dedup_cell_upper_bound_for_fixed_map": (
            map_wide_no_dedup_upper_bound
        ),
        "position_generation_seconds": generation_seconds,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
