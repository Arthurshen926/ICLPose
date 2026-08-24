"""Build a lightweight, hash-bound candidate dataset for direct rendering.

The pose-free candidate pool is fully validated before any contributor pose is
opened.  The output stores only immutable file references, candidate poses and
post-freeze error labels; rendered child grids and duplicated RADIO tensors are
deliberately absent.  Exact candidate evidence is produced later by the direct
canonical renderer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _load_contributors,
    _load_token_inventory,
    _pose_key,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
)


def _atomic_save(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
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


def _frozen_pose_free_prefix(
    source: dict[str, object], *, maximum_nonanchor: int,
) -> list[np.ndarray]:
    """Return the pool prefix without consulting or de-duplicating against GT."""

    details_by_mode = source.get("mode_details")
    details = (
        details_by_mode.get("actual_parent_actual_child")
        if isinstance(details_by_mode, dict) else None
    )
    if not isinstance(details, list) or not details:
        raise ValueError("candidate pool has no pose-free candidates")
    values: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for expected_rank, detail in enumerate(details, start=1):
        pose = np.asarray(detail.get("pose_w2c"), dtype=np.float64)
        if (
            int(detail.get("rank", -1)) != expected_rank
            or pose.shape != (4, 4)
            or np.any(~np.isfinite(pose))
        ):
            raise ValueError("candidate pool contains an invalid ranked pose")
        key = _pose_key(pose)
        if key in seen:
            raise ValueError("pose-free candidate pool contains duplicate poses")
        seen.add(key)
        if len(values) < int(maximum_nonanchor):
            values.append(pose)
    if not values:
        raise ValueError("candidate pool has no non-anchor pose")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--token_manifest", action="append", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", default="")
    parser.add_argument("--canonical_field_audit", default="")
    parser.add_argument(
        "--label_only_route_lineage_from_candidate_pool", action="store_true",
        help=(
            "build Phase-2 labels from the pool's strict atlas lineage without "
            "adding an unused canonical-field dependency"
        ),
    )
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--artifact_root", default=".")
    parser.add_argument("--maximum_candidates", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite direct pose candidate dataset")
    if int(args.maximum_candidates) < 2:
        raise ValueError("direct dataset requires at least two candidates")

    pool_path = Path(args.candidate_pool).resolve()
    pool = json.loads(pool_path.read_text())
    unhashed_pool = dict(pool)
    pool_content_sha256 = str(unhashed_pool.pop("content_sha256", ""))
    if (
        pool.get("artifact_type") != "goal_maplet_pose_free_visibility_candidate_pool_v1"
        or pool.get("uses_query_pose") is not False
        or pool.get("uses_query_ground_truth") is not False
        or pool.get("uses_alike") is not False
        or pool.get("uses_pnp") is not False
    ):
        raise ValueError("candidate pool is not a pose-free Goal-Maplet pool")
    if pool_content_sha256 != canonical_json_sha256(unhashed_pool):
        raise ValueError("candidate pool content hash differs")
    pool_rows = list(pool.get("rows", ()))
    if not pool_rows or int(pool.get("query_count", -1)) != len(pool_rows):
        raise ValueError("candidate pool inventory differs")
    # Freeze the complete candidate payload and byte identity before opening
    # contributor pose arrays below.
    frozen_pool_file_sha256 = file_sha256(pool_path)
    frozen_pool_content_sha256 = str(pool.get("content_sha256", ""))
    frozen_rows_json = json.dumps(pool_rows, sort_keys=True, separators=(",", ":"))
    strict_retrieval_audits = pool.get("strict_retrieval_promotion_audits", [])
    strict_v4_retrieval = bool(
        pool.get("strict_retrieval_promotion_required") is True
        and isinstance(strict_retrieval_audits, list)
        and strict_retrieval_audits
        and all(
            isinstance(value, dict)
            and value.get("promotion_eligible") is True
            and value.get("control_only") is False
            and value.get("query_split_disjoint") is True
            and value.get("validity_calibration_fit_trajectories") == ["seq10"]
            and value.get("query_route") == str(pool.get("query_route", ""))
            for value in strict_retrieval_audits
        )
    )
    if pool.get("strict_retrieval_promotion_required") is True and not strict_v4_retrieval:
        raise ValueError("strict candidate pool retrieval promotion audit differs")

    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field_path: Path | None = None
    audit_path: Path | None = None
    field: CanonicalSurfaceField | None = None
    audit: dict[str, object] | None = None
    if bool(args.label_only_route_lineage_from_candidate_pool):
        if args.canonical_field or args.canonical_field_audit:
            raise ValueError("label-only direct dataset must not bind an unused canonical field")
        route_audit = pool.get("route_disjoint_atlas_audit")
        if (
            not isinstance(route_audit, dict)
            or route_audit.get("route_allowlist_enforced") is not True
            or route_audit.get("query_route_excluded_from_atlas") is not True
            or route_audit.get("coordinate_correct") is not True
        ):
            raise ValueError("label-only direct dataset requires a strict pool atlas")
        mapping_routes = {
            str(value) for value in route_audit.get("allowed_trajectories", ())
        }
        atlas_path = Path(str(pool.get("atlas", ""))).resolve()
        atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
        if (
            file_sha256(atlas_path) != pool.get("atlas_file_sha256")
            or atlas.content_sha256 != pool.get("atlas_content_sha256")
            or atlas.physical_map_sha256 != physical.content_sha256
        ):
            raise ValueError("label-only direct dataset atlas/physical lineage differs")
    else:
        if not args.canonical_field or not args.canonical_field_audit:
            raise ValueError("direct dataset requires canonical field lineage or label-only mode")
        field_path = Path(args.canonical_field).resolve()
        audit_path = Path(args.canonical_field_audit).resolve()
        field = CanonicalSurfaceField.load_npz(field_path)
        audit = json.loads(audit_path.read_text())
        if (
            field.physical_map_sha256 != physical.content_sha256
            or audit.get("physical_map_sha256") != physical.content_sha256
            or audit.get("canonical_field_sha256") != field.content_sha256
            or audit.get("storage_contract", {}).get("coordinate_correct") is not True
        ):
            raise ValueError("direct dataset map/field coordinate lineage differs")
        mapping_routes = {
            str(value) for value in audit.get("mapping_trajectory_ids", ())
        }

    image_ids = [str(row.get("image_id", "")) for row in pool_rows]
    if len(set(image_ids)) != len(image_ids) or any(not value for value in image_ids):
        raise ValueError("candidate pool image IDs are empty or duplicated")
    if any(value.split("/", 1)[0] in mapping_routes for value in image_ids):
        raise ValueError("candidate query route appears in the canonical map")

    # The contributor inventory contains the complete training set.  Opening
    # every compressed NPZ merely to build a bounded query diagnostic made a
    # small post-freeze coverage audit take minutes before doing any useful
    # work.  Resolve only the already-frozen query IDs; the loader still checks
    # uniqueness and fails closed if any requested record is absent.
    contributors = _load_contributors(
        Path(args.contributors), required_image_ids=set(image_ids),
    )
    manifest_paths = [Path(value).resolve() for value in args.token_manifest]
    tokens = _load_token_inventory(
        manifest_paths, artifact_root=Path(args.artifact_root).resolve(),
    )
    if any(value not in contributors or value not in tokens for value in image_ids):
        raise ValueError("candidate query lacks contributor or RADIO token evidence")

    maximum = int(args.maximum_candidates)
    candidate_rows, valid_rows, translation_rows, rotation_rows = [], [], [], []
    radio_paths, radio_hashes, contributor_paths, contributor_hashes = [], [], [], []
    for image_id, source in zip(image_ids, pool_rows):
        contributor = contributors[image_id]
        with np.load(contributor, allow_pickle=False) as data:
            target_pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        # The GT anchor is a Phase-2 diagnostic only.  It must not alter the
        # membership/order of the already-frozen pose-free prefix.  A map pose
        # may legitimately equal GT across routes and must retain its 0 error.
        values = [target_pose] + _frozen_pose_free_prefix(
            source, maximum_nonanchor=maximum - 1,
        )
        valid_count = len(values)
        while len(values) < maximum:
            values.append(values[-1].copy())
        candidates = np.stack(values)
        translation, rotation = _pose_errors(candidates, target_pose)
        candidate_rows.append(candidates)
        valid_rows.append(np.arange(maximum) < valid_count)
        translation_rows.append(translation.astype(np.float32))
        rotation_rows.append(rotation.astype(np.float32))
        radio_paths.append(str(tokens[image_id]))
        radio_hashes.append(file_sha256(tokens[image_id]))
        contributor_paths.append(str(contributor))
        contributor_hashes.append(file_sha256(contributor))

    if (
        file_sha256(pool_path) != frozen_pool_file_sha256
        or str(json.loads(pool_path.read_text()).get("content_sha256", ""))
        != frozen_pool_content_sha256
        or json.dumps(json.loads(pool_path.read_text()).get("rows", ()), sort_keys=True, separators=(",", ":"))
        != frozen_rows_json
    ):
        raise RuntimeError("candidate pool changed after the pre-label freeze")

    arrays = {
        "image_ids": np.asarray(image_ids),
        "radio_token_paths": np.asarray(radio_paths),
        "radio_file_sha256": np.asarray(radio_hashes),
        "contributor_paths": np.asarray(contributor_paths),
        "contributor_file_sha256": np.asarray(contributor_hashes),
        "candidate_poses_w2c": np.stack(candidate_rows),
        "translation_m": np.stack(translation_rows),
        "rotation_deg": np.stack(rotation_rows),
        "candidate_valid": np.stack(valid_rows),
    }
    metadata = {
        "artifact_type": DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
        "content_sha256": arrays_sha256(arrays),
        "query_count": len(image_ids),
        "candidate_count": maximum,
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "nonanchor_candidates_preserve_pose_free_pool_exact_order": True,
        "gt_anchor_does_not_change_nonanchor_candidate_membership": True,
        "pose_free_pool_internal_duplicates_rejected_before_gt_join": True,
        "candidate_pool_frozen_before_target_pose_opened": True,
        "candidate_pool_file_sha256": frozen_pool_file_sha256,
        "candidate_pool_content_sha256": frozen_pool_content_sha256,
        "strict_retrieval_promotion_required": bool(
            pool.get("strict_retrieval_promotion_required", False)
        ),
        "strict_retrieval_promotion_audits": strict_retrieval_audits,
        "strict_v4_seq10_calibrated_retrieval_confirmed": strict_v4_retrieval,
        "candidate_pool_scores_consumed": False,
        "candidate_semantics": str(pool.get("candidate_semantics", "")),
        "candidate_prefix_stable_across_budgets": bool(
            pool.get("candidate_prefix_stable_across_budgets", False)
        ),
        "candidate_pool_maximum_modes": int(pool.get("maximum_modes", maximum - 1)),
        "candidate_location_block_size": (
            int(pool["location_block_size"])
            if "location_block_size" in pool else None
        ),
        "pose_errors_computed_only_after_candidate_freeze": True,
        "canonical_map_excludes_query_route": True,
        "map_training_routes": sorted(mapping_routes),
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": file_sha256(physical_path),
        "canonical_field_sha256": (
            field.content_sha256 if field is not None else None
        ),
        "canonical_field_file_sha256": (
            file_sha256(field_path) if field_path is not None else None
        ),
        "canonical_field_audit_file_sha256": (
            file_sha256(audit_path) if audit_path is not None else None
        ),
        "label_only_no_canonical_field_dependency": bool(
            args.label_only_route_lineage_from_candidate_pool
        ),
        "token_manifest_files": [
            {"path": str(path), "file_sha256": file_sha256(path)}
            for path in manifest_paths
        ],
        "stores_rendered_target_grids": False,
        "stores_radio_tensor_copies": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    _atomic_save(output, arrays, metadata)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({**metadata, "output_npz": str(output.resolve())}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()), "query_count": len(image_ids),
        "candidate_count": maximum, "content_sha256": metadata["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
