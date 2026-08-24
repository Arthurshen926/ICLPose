"""Freeze a pose-free baseline/layout seed-budget sweep artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    load_pose_free_candidate_pool,
)
from feature_extract.vfm.localization_goal_maplet.two_branch_seed_budget import (
    BRANCH_NAMES,
    ORIENTATION_BUDGET,
    POSITION_SEED_BUDGETS,
    SCHEMA,
    SEMANTICS,
    build_two_branch_seed_budget_arrays,
    load_two_branch_seed_budget,
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


def _validate_pool_branch(
    pool: dict[str, object], *, branch: str, query_route: str,
) -> None:
    if (
        pool.get("query_route") != query_route
        or int(pool.get("maximum_modes", -1)) != 64
        or pool.get("candidate_semantics")
        != "progressive_hierarchical_location_orientation_v3"
        or pool.get("uses_query_pose") is not False
        or pool.get("uses_query_ground_truth") is not False
        or pool.get("uses_alike") is not False
        or pool.get("uses_pnp") is not False
    ):
        raise ValueError(f"{branch} seed-sweep pool contract differs")
    bindings = list(pool.get("retrieval_runs", ()))
    if not bindings:
        raise ValueError(f"{branch} seed-sweep pool lacks retrieval lineage")
    for binding in bindings:
        path = Path(str(binding["path"])).resolve()
        run = json.loads(path.read_text())
        audit = run.get("query_split_audit", {})
        routes = {str(value) for value in audit.get("query_trajectory_ids", ())}
        if (
            file_sha256(path) != binding.get("file_sha256")
            or routes != {query_route}
            or run.get("uses_query_pose") is not False
            or run.get("uses_query_ground_truth") is not False
        ):
            raise ValueError(f"{branch} seed-sweep retrieval lineage differs")
        if query_route == "seq10":
            if (
                run.get("control_only") is not True
                or run.get("promotion_eligible") is not False
                or not list(run.get("promotion_blockers", ()))
            ):
                raise ValueError("seq10 seed sweep must remain a tuning control")
            if branch == "layout" and (
                run.get("layout_child_allocator_tuning_route") != "seq10"
                or run.get("seq10_same_route_layout_tuning_control") is not True
                or audit.get("allocator_tuning_same_route_control") is not True
            ):
                raise ValueError("seq10 layout seed sweep lacks same-route control lineage")
        else:
            if (
                query_route not in {"seq12", "seq14"}
                or run.get("control_only") is not False
                or run.get("promotion_eligible") is not True
                or audit.get("disjoint") is not True
            ):
                raise ValueError("held seed sweep retrieval is not strict-promotion eligible")
            if branch == "layout" and (
                run.get("layout_child_allocator_tuning_route") != "seq10"
                or audit.get("allocator_tuning_query_disjoint") is not True
            ):
                raise ValueError("held layout seed sweep allocator lineage differs")
        has_layout = bool(run.get("layout_child_allocator_config"))
        if has_layout != (branch == "layout"):
            raise ValueError("seed-sweep retrieval branch identity differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_pool", required=True)
    parser.add_argument("--layout_pool", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite two-branch seed sweep")
    pool_paths = [Path(args.baseline_pool).resolve(), Path(args.layout_pool).resolve()]
    pools = [load_pose_free_candidate_pool(path) for path in pool_paths]
    query_route = str(pools[0].get("query_route", ""))
    if (
        not query_route or pools[1].get("query_route") != query_route
        or pools[0].get("atlas_file_sha256") != pools[1].get("atlas_file_sha256")
        or pools[0].get("atlas_content_sha256")
        != pools[1].get("atlas_content_sha256")
    ):
        raise ValueError("two-branch seed-sweep pool lineage differs")
    for branch, pool in zip(BRANCH_NAMES, pools):
        _validate_pool_branch(pool, branch=branch, query_route=query_route)
    started = time.perf_counter()
    arrays = build_two_branch_seed_budget_arrays(pools[0], pools[1])
    build_seconds = float(time.perf_counter() - started)
    seed_counts = arrays["unique_position_seed_count_by_budget"]
    position_counts = arrays["unique_position_factor_count_by_budget"]
    orientation_counts = arrays["unique_orientation_factor_count"]
    implicit_counts = arrays["implicit_lattice_pose_pair_count_by_budget"]
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "query_count": int(arrays["image_ids"].size),
        "query_route": query_route,
        "branch_names": list(BRANCH_NAMES),
        "baseline_pool": str(pool_paths[0]),
        "baseline_pool_file_sha256": file_sha256(pool_paths[0]),
        "baseline_pool_content_sha256": str(pools[0]["content_sha256"]),
        "layout_pool": str(pool_paths[1]),
        "layout_pool_file_sha256": file_sha256(pool_paths[1]),
        "layout_pool_content_sha256": str(pools[1]["content_sha256"]),
        "atlas_file_sha256": str(pools[0]["atlas_file_sha256"]),
        "atlas_content_sha256": str(pools[0]["atlas_content_sha256"]),
        "position_seed_budgets": list(POSITION_SEED_BUDGETS),
        "orientation_budget": ORIENTATION_BUDGET,
        "position_step_m": 2.0,
        "position_xz_half_extent_m": 10.0,
        "position_y_half_extent_m": 4.0,
        "position_offsets_per_seed": int(arrays["position_offsets_camera"].shape[0]),
        "unique_position_seed_count_range_by_budget": {
            str(budget): [
                int(np.min(seed_counts[:, index])),
                int(np.max(seed_counts[:, index])),
            ]
            for index, budget in enumerate(POSITION_SEED_BUDGETS)
        },
        "unique_position_factor_count_range_by_budget": {
            str(budget): [
                int(np.min(position_counts[:, index])),
                int(np.max(position_counts[:, index])),
            ]
            for index, budget in enumerate(POSITION_SEED_BUDGETS)
        },
        "unique_orientation_factor_count_range": [
            int(np.min(orientation_counts)), int(np.max(orientation_counts)),
        ],
        "implicit_lattice_pose_pair_count_range_by_budget": {
            str(budget): [
                int(np.min(implicit_counts[:, index])),
                int(np.max(implicit_counts[:, index])),
            ]
            for index, budget in enumerate(POSITION_SEED_BUDGETS)
        },
        "phase1_build_seconds": build_seconds,
        "development_same_route_seq10_control": query_route == "seq10",
        "held_routes_used_for_budget_selection": False,
        "cross_branch_cartesian_products_included": False,
        "branch_local_cartesian_products_materialized": False,
        "direct_label_dataset_opened_during_generation": False,
        "query_pose_member_opened_during_generation": False,
        "query_ground_truth_member_opened_during_generation": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "raw_support_is_implicit_factor_upper_bound_only": True,
        "position_collision_free_space_certified": False,
        "production_eligible": False,
    }
    _atomic_save(output, arrays, metadata)
    load_two_branch_seed_budget(output)
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "content_sha256": metadata["content_sha256"],
        "query_route": query_route,
        "query_count": metadata["query_count"],
        "phase1_build_seconds": build_seconds,
        "unique_position_seed_count_range_by_budget": metadata[
            "unique_position_seed_count_range_by_budget"
        ],
        "unique_orientation_factor_count_range": metadata[
            "unique_orientation_factor_count_range"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
