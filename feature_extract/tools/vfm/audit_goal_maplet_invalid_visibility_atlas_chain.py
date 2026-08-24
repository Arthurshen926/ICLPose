"""Write a post-hoc invalidation audit for a query-route-contaminated atlas chain."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_child_visibility_pose_atlas import (
    contributor_route_and_image_id,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.visibility_pose_atlas import (
    ChildVisibilityPoseAtlas,
)


def _artifact_binding(path: Path) -> dict[str, object]:
    value = Path(path).resolve()
    return {"path": str(value), "file_sha256": file_sha256(value)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--held_out_route", action="append", required=True)
    parser.add_argument("--candidate_pool", action="append", required=True)
    parser.add_argument("--direct_dataset", action="append", required=True)
    parser.add_argument("--affected_artifact", action="append", default=[])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite atlas invalidation audit")

    atlas_path = Path(args.atlas).resolve()
    atlas = ChildVisibilityPoseAtlas.load_npz(atlas_path)
    source_directory = Path(str(atlas.metadata.get("source_contributor_directory", "")))
    source_paths = sorted(source_directory.glob("*.npz"))
    routes = [contributor_route_and_image_id(path)[0] for path in source_paths]
    counts = Counter(routes)
    held = sorted({str(value) for value in args.held_out_route})
    held_counts = {route: int(counts.get(route, 0)) for route in held}
    if any(value <= 0 for value in held_counts.values()):
        raise ValueError("requested held-out route is not present in invalid atlas sources")
    if len(source_paths) != atlas.view_count:
        raise ValueError("atlas source directory count differs from atlas rows")

    direct_by_pool: dict[str, tuple[Path, dict[str, np.ndarray], dict[str, object]]] = {}
    direct_bindings: list[dict[str, object]] = []
    for value in args.direct_dataset:
        path = Path(value).resolve()
        arrays, metadata = load_pose_candidate_dataset(path, require_rendered_targets=False)
        key = str(metadata.get("candidate_pool_content_sha256", ""))
        if not key or key in direct_by_pool:
            raise ValueError("direct dataset candidate-pool binding is empty or duplicated")
        direct_by_pool[key] = (path, arrays, metadata)
        direct_bindings.append({
            **_artifact_binding(path),
            "content_sha256": str(metadata["content_sha256"]),
            "candidate_pool_content_sha256": key,
        })

    pool_audits: list[dict[str, object]] = []
    for value in args.candidate_pool:
        pool_path = Path(value).resolve()
        pool = json.loads(pool_path.read_text())
        pool_content = str(pool.get("content_sha256", ""))
        if (
            str(Path(str(pool.get("atlas", ""))).resolve()) != str(atlas_path)
            or pool.get("atlas_file_sha256") != file_sha256(atlas_path)
            or pool.get("atlas_content_sha256") != atlas.content_sha256
            or pool_content not in direct_by_pool
        ):
            raise ValueError("candidate pool does not bind the invalid atlas/direct labels")
        direct_path, arrays, direct_metadata = direct_by_pool[pool_content]
        pool_ids = [str(row["image_id"]) for row in pool["rows"]]
        direct_ids = [str(value) for value in np.asarray(arrays["image_ids"]).tolist()]
        if pool_ids != direct_ids:
            raise ValueError("invalid-chain pool/direct query identities differ")
        poses = np.stack([
            np.stack([
                np.asarray(detail["pose_w2c"], dtype=np.float64)
                for detail in row["mode_details"]["actual_parent_actual_child"]
            ])
            for row in pool["rows"]
        ])
        target_poses: list[np.ndarray] = []
        for contributor in np.asarray(arrays["contributor_paths"]).tolist():
            with np.load(Path(str(contributor)), allow_pickle=False) as data:
                target_poses.append(np.asarray(data["pose_w2c"], dtype=np.float64))
        target = np.stack(target_poses)
        collision = np.all(poses == target[:, None], axis=(2, 3))
        direct_pose = np.asarray(arrays["candidate_poses_w2c"], dtype=np.float64)
        direct_valid = np.asarray(arrays["candidate_valid"], dtype=bool)
        width = min(int(poses.shape[1]), int(direct_pose.shape[1] - 1))
        alignment = np.all(
            direct_pose[:, 1 : 1 + width] == poses[:, :width], axis=(1, 2, 3)
        ) & np.all(direct_valid[:, 1 : 1 + width], axis=1)
        pool_audits.append({
            **_artifact_binding(pool_path),
            "content_sha256": pool_content,
            "query_route": str(pool.get("query_route", "")),
            "query_count": len(pool_ids),
            "candidate_count": int(poses.shape[1]),
            "exact_query_gt_pose_collision_query_count": int(
                np.sum(np.any(collision, axis=1))
            ),
            "exact_query_gt_pose_collision_candidate_count": int(np.sum(collision)),
            "direct_dataset": str(direct_path),
            "direct_prefix_exact_pool_alignment_query_count": int(np.sum(alignment)),
            "direct_gt_conditioned_prefix_drift_query_count": int(np.sum(~alignment)),
            "direct_legacy_gt_deduplication_bug": bool(
                direct_metadata.get(
                    "gt_anchor_does_not_change_nonanchor_candidate_membership"
                ) is not True
            ),
        })

    affected = [_artifact_binding(Path(value)) for value in args.affected_artifact]
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_invalid_visibility_atlas_chain_audit_v1",
        "status": "invalid_leaky_atlas",
        "authority": "must_not_be_cited_as_pose_free_retrieval_or_pose_estimation_evidence",
        "invalid_atlas": {
            **_artifact_binding(atlas_path),
            "content_sha256": atlas.content_sha256,
            "source_contributor_count": atlas.view_count,
            "source_trajectory_counts": {
                route: int(counts[route]) for route in sorted(counts)
            },
            "held_out_route_source_counts": held_counts,
            "route_allowlist_enforced": bool(
                atlas.metadata.get("route_allowlist_enforced", False)
            ),
            "coordinate_correct": bool(atlas.metadata.get("coordinate_correct", False)),
        },
        "candidate_pool_gt_collision_audits": pool_audits,
        "invalid_direct_datasets": direct_bindings,
        "other_affected_artifacts": affected,
        "failure_reasons": [
            "held_out_query_routes_are_atlas_pose_sources",
            "candidate_pool_contains_exact_query_gt_pose_for_many_queries",
            "legacy_direct_builder_gt_deduplication_changes_pose_free_prefix",
            "atlas_coordinate_contract_is_not_proven_coordinate_correct",
        ],
        "query_ground_truth_used_only_for_posthoc_invalidation_audit": True,
        "query_ground_truth_used_for_candidate_generation_or_scoring": False,
        "files_deleted": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "status": report["status"],
        "atlas_file_sha256": report["invalid_atlas"]["file_sha256"],
        "held_out_route_source_counts": held_counts,
        "candidate_pool_gt_collision_audits": pool_audits,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
