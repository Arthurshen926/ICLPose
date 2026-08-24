"""Freeze the union of baseline and layout factor domains without labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_extract.vfm.localization_goal_maplet.factorized_branch_union import (
    BRANCH_NAMES,
    SCHEMA,
    SEMANTICS,
    build_factorized_branch_union_arrays,
    load_factorized_branch_union,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_free_proposal import (
    load_factorized_pose_free_proposal,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256


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


def _load_bound_candidate_pool(
    proposal_metadata: dict[str, object], *, branch: str,
) -> tuple[Path, dict[str, object]]:
    path = Path(str(proposal_metadata["candidate_pool"])).resolve()
    pool = json.loads(path.read_text())
    if (
        file_sha256(path) != proposal_metadata.get("candidate_pool_file_sha256")
        or pool.get("content_sha256")
        != proposal_metadata.get("candidate_pool_content_sha256")
        or pool.get("uses_query_pose") is not False
        or pool.get("uses_query_ground_truth") is not False
        or pool.get("strict_retrieval_promotion_required") is not True
    ):
        raise ValueError(f"{branch} factor branch candidate-pool lineage differs")
    return path, pool


def _validate_retrieval_runs(
    pool: dict[str, object], *, branch: str,
) -> None:
    bindings = list(pool.get("retrieval_runs", ()))
    if not bindings:
        raise ValueError(f"{branch} factor branch lacks retrieval lineage")
    for binding in bindings:
        path = Path(str(binding["path"])).resolve()
        run = json.loads(path.read_text())
        audit = run.get("query_split_audit", {})
        common_valid = (
            file_sha256(path) == binding.get("file_sha256")
            and run.get("promotion_eligible") is True
            and run.get("control_only") is False
            and audit.get("disjoint") is True
        )
        if branch == "baseline":
            branch_valid = (
                "layout_child_allocator_config" not in run
                and "layout_child_allocator_tuning_route" not in run
            )
        else:
            branch_valid = (
                run.get("layout_child_allocator_tuning_route") == "seq10"
                and audit.get("allocator_tuning_query_disjoint") is True
                and bool(run.get("layout_child_allocator_config"))
            )
        if not common_valid or not branch_valid:
            raise ValueError(f"{branch} factor branch retrieval lineage differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_proposal", required=True)
    parser.add_argument("--layout_proposal", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_npz)
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite factorized branch union")
    baseline_path = Path(args.baseline_proposal).resolve()
    layout_path = Path(args.layout_proposal).resolve()
    baseline, baseline_metadata = load_factorized_pose_free_proposal(baseline_path)
    layout, layout_metadata = load_factorized_pose_free_proposal(layout_path)
    common = (
        "query_route", "position_seed_count", "position_step_m",
        "position_xz_half_extent_m", "position_y_half_extent_m",
        "positions_per_seed", "orientation_source_prefix_budget",
        "candidate_pool_atlas_content_sha256",
    )
    if (
        any(baseline_metadata.get(key) != layout_metadata.get(key) for key in common)
        or baseline_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
        or layout_metadata.get("strict_v4_seq10_calibrated_retrieval_confirmed")
        is not True
        or int(baseline_metadata.get("position_seed_count", -1)) != 4
        or int(baseline_metadata.get("orientation_source_prefix_budget", -1)) != 64
    ):
        raise ValueError("factorized branch union strict/configuration lineage differs")
    baseline_pool_path, baseline_pool = _load_bound_candidate_pool(
        baseline_metadata, branch="baseline",
    )
    layout_pool_path, layout_pool = _load_bound_candidate_pool(
        layout_metadata, branch="layout",
    )
    _validate_retrieval_runs(baseline_pool, branch="baseline")
    _validate_retrieval_runs(layout_pool, branch="layout")
    arrays = build_factorized_branch_union_arrays(baseline, layout)
    seed_counts = np.sum(arrays["unique_position_seed_valid"], axis=1)
    orientation_counts = np.sum(arrays["unique_orientation_valid"], axis=1)
    unique_position_counts = []
    for query in range(int(arrays["image_ids"].size)):
        positions = arrays["unique_position_centers_world"][
            query, arrays["unique_position_seed_valid"][query]
        ].reshape(-1, 3)
        unique_position_counts.append(int(np.unique(
            positions.round(10), axis=0,
        ).shape[0]))
    metadata: dict[str, object] = {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "content_sha256": arrays_sha256(arrays),
        "branch_names": list(BRANCH_NAMES),
        "query_count": int(arrays["image_ids"].size),
        "query_route": str(baseline_metadata["query_route"]),
        "baseline_proposal": str(baseline_path),
        "baseline_proposal_file_sha256": file_sha256(baseline_path),
        "baseline_proposal_content_sha256": str(
            baseline_metadata["content_sha256"]
        ),
        "baseline_candidate_pool_file_sha256": str(
            baseline_metadata["candidate_pool_file_sha256"]
        ),
        "baseline_candidate_pool": str(baseline_pool_path),
        "baseline_candidate_pool_content_sha256": str(
            baseline_metadata["candidate_pool_content_sha256"]
        ),
        "layout_proposal": str(layout_path),
        "layout_proposal_file_sha256": file_sha256(layout_path),
        "layout_proposal_content_sha256": str(layout_metadata["content_sha256"]),
        "layout_candidate_pool_file_sha256": str(
            layout_metadata["candidate_pool_file_sha256"]
        ),
        "layout_candidate_pool": str(layout_pool_path),
        "layout_candidate_pool_content_sha256": str(
            layout_metadata["candidate_pool_content_sha256"]
        ),
        "strict_v4_seq10_calibrated_retrieval_confirmed": True,
        "layout_allocator_tuned_only_on_seq10": True,
        "branch_position_seed_budget": 4,
        "branch_orientation_budget": 64,
        "position_offsets_per_seed": int(
            arrays["position_offsets_camera"].shape[0]
        ),
        "maximum_unique_position_seed_count": 8,
        "unique_position_seed_count_range": [
            int(np.min(seed_counts)), int(np.max(seed_counts)),
        ],
        "unique_position_seed_count_mean": float(np.mean(seed_counts)),
        "unique_position_factor_count_range": [
            min(unique_position_counts), max(unique_position_counts),
        ],
        "unique_position_factor_count_mean": float(np.mean(unique_position_counts)),
        "unique_orientation_count_range": [
            int(np.min(orientation_counts)), int(np.max(orientation_counts)),
        ],
        "unique_orientation_count_mean": float(np.mean(orientation_counts)),
        "implicit_seed_orientation_pair_count_range": [
            int(np.min(arrays["implicit_seed_orientation_pair_count_by_query"])),
            int(np.max(arrays["implicit_seed_orientation_pair_count_by_query"])),
        ],
        "implicit_lattice_pose_pair_count_range": [
            int(np.min(arrays["implicit_lattice_pose_pair_count_by_query"])),
            int(np.max(arrays["implicit_lattice_pose_pair_count_by_query"])),
        ],
        "domain_union_is_branch_or_not_cartesian_hull": True,
        "cross_branch_cartesian_products_included": False,
        "branch_factor_provenance_stored": True,
        "factor_identity_deduplicated_at_1e_10": True,
        "cartesian_pose_product_materialized": False,
        "direct_label_dataset_opened_during_generation": False,
        "query_pose_member_opened_during_generation": False,
        "query_ground_truth_member_opened_during_generation": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "raw_coverage_is_implicit_factor_support_upper_bound_only": True,
        "position_collision_free_space_certified": False,
        "production_eligible": False,
    }
    _atomic_save(output, arrays, metadata)
    load_factorized_branch_union(output)
    sidecar.write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "content_sha256": metadata["content_sha256"],
        "query_count": metadata["query_count"],
        "unique_position_seed_count_range": metadata[
            "unique_position_seed_count_range"
        ],
        "unique_orientation_count_range": metadata[
            "unique_orientation_count_range"
        ],
        "cross_branch_cartesian_products_included": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
