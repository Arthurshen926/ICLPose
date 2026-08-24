"""Build a route-disjoint inventory for GT-relative pose-energy supervision.

This artifact is deliberately not a deployable candidate pool.  It opens each
training contributor pose only after the query inventory and held-out
canonical map have been validated, and stores that pose solely as row zero for
the deterministic SE(3) supervision builder downstream.  It never uses ALIKE,
point correspondences, PnP, or absolute-pose regression.
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
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA,
)


def _parse_route_limits(values: list[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for value in values:
        route, separator, count = str(value).partition(":")
        if not separator or not route or not count.isdigit() or int(count) <= 0:
            raise ValueError("route limits must use ROUTE:POSITIVE_COUNT")
        result.append((route, int(count)))
    if not result or len({route for route, _ in result}) != len(result):
        raise ValueError("controlled training routes must be nonempty and unique")
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--token_manifest", action="append", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--canonical_field_audit", required=True)
    parser.add_argument("--route_limits", nargs="+", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--artifact_root", default=".")
    args = parser.parse_args()

    output = Path(args.output_npz)
    if output.exists():
        raise FileExistsError("refusing to overwrite controlled pose training inventory")
    route_limits = _parse_route_limits(list(args.route_limits))
    requested_routes = {route for route, _ in route_limits}
    physical_path = Path(args.physical_map).resolve()
    field_path = Path(args.canonical_field).resolve()
    audit_path = Path(args.canonical_field_audit).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    audit = json.loads(audit_path.read_text())
    mapping_routes = {str(value) for value in audit.get("mapping_trajectory_ids", ())}
    if (
        requested_routes & mapping_routes
        or field.physical_map_sha256 != physical.content_sha256
        or audit.get("physical_map_sha256") != physical.content_sha256
        or audit.get("canonical_field_sha256") != field.content_sha256
        or audit.get("storage_contract", {}).get("coordinate_correct") is not True
    ):
        raise ValueError("controlled training map is not route-disjoint/coordinate-correct")

    contributors = _load_contributors(Path(args.contributors))
    manifest_paths = [Path(value).resolve() for value in args.token_manifest]
    tokens = _load_token_inventory(
        manifest_paths, artifact_root=Path(args.artifact_root).resolve(),
    )
    image_ids: list[str] = []
    for route, count in route_limits:
        eligible = sorted(
            image_id for image_id in set(contributors) & set(tokens)
            if image_id.split("/", 1)[0] == route
        )
        if len(eligible) < count:
            raise ValueError(f"route {route} lacks {count} contributor/token rows")
        image_ids.extend(eligible[:count])
    if len(set(image_ids)) != len(image_ids):
        raise AssertionError("controlled training inventory contains duplicate images")

    poses, radio_paths, radio_hashes, contributor_paths, contributor_hashes = [], [], [], [], []
    for image_id in image_ids:
        contributor = contributors[image_id]
        with np.load(contributor, allow_pickle=False) as data:
            pose = np.asarray(data["pose_w2c"], dtype=np.float64)
        if pose.shape != (4, 4) or np.any(~np.isfinite(pose)):
            raise ValueError("controlled training contributor pose differs")
        token = tokens[image_id]
        poses.append(pose)
        radio_paths.append(str(token)); radio_hashes.append(file_sha256(token))
        contributor_paths.append(str(contributor)); contributor_hashes.append(file_sha256(contributor))

    pose_array = np.asarray(poses, dtype=np.float64)[:, None]
    arrays = {
        "image_ids": np.asarray(image_ids),
        "radio_token_paths": np.asarray(radio_paths),
        "radio_file_sha256": np.asarray(radio_hashes),
        "contributor_paths": np.asarray(contributor_paths),
        "contributor_file_sha256": np.asarray(contributor_hashes),
        "candidate_poses_w2c": pose_array,
        "translation_m": np.zeros((len(image_ids), 1), dtype=np.float32),
        "rotation_deg": np.zeros((len(image_ids), 1), dtype=np.float32),
        "candidate_valid": np.ones((len(image_ids), 1), dtype=bool),
    }
    metadata = {
        "artifact_type": CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA,
        "content_sha256": arrays_sha256(arrays),
        "query_count": len(image_ids),
        "candidate_count": 1,
        "route_limits": [{"route": route, "count": count} for route, count in route_limits],
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "uses_gt_for_training_candidate_generation": True,
        "deployment_candidate_pool": False,
        "canonical_map_excludes_query_route": True,
        "map_training_routes": sorted(mapping_routes),
        "physical_map_sha256": physical.content_sha256,
        "physical_map_file_sha256": file_sha256(physical_path),
        "canonical_field_sha256": field.content_sha256,
        "canonical_field_file_sha256": file_sha256(field_path),
        "canonical_field_audit_file_sha256": file_sha256(audit_path),
        "token_manifest_files": [
            {"path": str(path), "file_sha256": file_sha256(path)} for path in manifest_paths
        ],
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "contains_standard_test_queries": False,
    }
    _atomic_save(output, arrays, metadata)
    output.with_suffix(".json").write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "query_count": len(image_ids),
        "content_sha256": metadata["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
