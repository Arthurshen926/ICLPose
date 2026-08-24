"""Build GT-controlled trajectories from frozen natural retrieval seeds.

The source candidate pools must have been frozen before target poses were
opened.  This tool is training-only: it interpolates camera centre linearly
and orientation along the shortest SO(3) arc from every natural seed toward
the diagnostic GT anchor.  It never creates a deployment candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.vfm.cambridge_pose_lattice import (
    quaternion_wxyz_to_rotation_matrix,
    rotation_matrix_to_quaternion_wxyz,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA,
    load_pose_candidate_dataset,
)


TRAJECTORY_SUPERVISION_SEMANTICS = (
    "frozen_natural_seed_to_gt_center_linear_so3_shortest_arc_training_only_v1"
)


def interpolate_pose_product(
    seed_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate world camera centre and shortest-arc w2c orientation."""
    seed = np.asarray(seed_pose_w2c, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    fraction = float(alpha)
    if np.any(~np.isfinite(seed)) or np.any(~np.isfinite(target)):
        raise ValueError("trajectory endpoint pose must be finite")
    if not 0.0 <= fraction <= 1.0 or not np.isfinite(fraction):
        raise ValueError("trajectory fraction must lie in [0,1]")
    if fraction == 0.0:
        return seed.copy()
    if fraction == 1.0:
        return target.copy()
    seed_rotation = seed[:3, :3]
    target_rotation = target[:3, :3]
    seed_center = -seed_rotation.T @ seed[:3, 3]
    target_center = -target_rotation.T @ target[:3, 3]
    first = np.asarray(rotation_matrix_to_quaternion_wxyz(seed_rotation))
    second = np.asarray(rotation_matrix_to_quaternion_wxyz(target_rotation))
    dot = float(first @ second)
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-10:
        quaternion = (1.0 - fraction) * first + fraction * second
    else:
        angle = float(np.arccos(dot))
        quaternion = (
            np.sin((1.0 - fraction) * angle) / np.sin(angle) * first
            + np.sin(fraction * angle) / np.sin(angle) * second
        )
    quaternion /= np.linalg.norm(quaternion)
    rotation = quaternion_wxyz_to_rotation_matrix(*quaternion.tolist())
    center = (1.0 - fraction) * seed_center + fraction * target_center
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ center
    return result


def _route_limits(values: list[str]) -> list[tuple[str, int]]:
    result = []
    for value in values:
        route, separator, count = str(value).partition(":")
        if not separator or not route or not count.isdigit() or int(count) <= 0:
            raise ValueError("route limits must use ROUTE:POSITIVE_COUNT")
        result.append((route, int(count)))
    if not result or len({route for route, _ in result}) != len(result):
        raise ValueError("trajectory route limits must be nonempty and unique")
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
    parser.add_argument("--source_dataset", action="append", required=True)
    parser.add_argument("--route_limits", nargs="+", required=True)
    parser.add_argument("--steps_per_seed", type=int, default=8)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    output = Path(args.output_npz)
    if output.exists():
        raise FileExistsError("refusing to overwrite trajectory supervision")
    steps = int(args.steps_per_seed)
    if steps < 2:
        raise ValueError("trajectory supervision requires at least two seed steps")

    by_id: dict[str, tuple[dict[str, np.ndarray], dict[str, object], Path, int]] = {}
    bindings = []
    common = None
    for value in args.source_dataset:
        path = Path(value).resolve()
        arrays, metadata = load_pose_candidate_dataset(path, require_rendered_targets=False)
        if (
            metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True
            or metadata.get("candidate_zero_is_diagnostic_gt_anchor") is not True
            or int(arrays["candidate_valid"].shape[1]) < 2
        ):
            raise ValueError("trajectory source is not a frozen natural candidate pool")
        lineage = (
            metadata.get("physical_map_sha256"), metadata.get("canonical_field_sha256"),
            metadata.get("canonical_map_excludes_query_route"),
        )
        if common is None:
            common = lineage
        elif common != lineage:
            raise ValueError("trajectory source map lineage differs")
        for row, image_id in enumerate(arrays["image_ids"].tolist()):
            key = str(image_id)
            if key in by_id:
                raise ValueError("duplicate trajectory source query")
            by_id[key] = (arrays, metadata, path, row)
        bindings.append({
            "path": str(path), "file_sha256": file_sha256(path),
            "content_sha256": metadata["content_sha256"],
        })

    selected = []
    for route, count in _route_limits(list(args.route_limits)):
        rows = sorted(key for key in by_id if key.split("/", 1)[0] == route)
        if len(rows) < count:
            raise ValueError(f"trajectory source route {route} lacks {count} rows")
        selected.extend(rows[:count])
    first_arrays = by_id[selected[0]][0]
    source_candidates = int(first_arrays["candidate_valid"].shape[1])
    candidate_count = 1 + (source_candidates - 1) * steps
    query_count = len(selected)
    poses = np.empty((query_count, candidate_count, 4, 4), dtype=np.float64)
    translation = np.empty((query_count, candidate_count), dtype=np.float32)
    rotation = np.empty_like(translation)
    seed_rows = np.full((query_count, candidate_count), -1, dtype=np.int32)
    alpha_rows = np.ones((query_count, candidate_count), dtype=np.float32)
    radio_paths = []
    radio_hashes = []
    contributor_paths = []
    contributor_hashes = []
    fractions = np.arange(steps, dtype=np.float64) / float(steps)
    for output_row, image_id in enumerate(selected):
        arrays, _, _, source_row = by_id[image_id]
        if arrays["candidate_valid"].shape[1] != source_candidates:
            raise ValueError("trajectory source candidate counts differ")
        if not np.all(arrays["candidate_valid"][source_row]):
            raise ValueError("trajectory source contains invalid natural candidates")
        target = np.asarray(arrays["candidate_poses_w2c"][source_row, 0], dtype=np.float64)
        values = [target.copy()]
        cursor = 1
        for seed_row in range(1, source_candidates):
            seed = arrays["candidate_poses_w2c"][source_row, seed_row]
            for fraction in fractions.tolist():
                values.append(interpolate_pose_product(seed, target, fraction))
                seed_rows[output_row, cursor] = seed_row
                alpha_rows[output_row, cursor] = fraction
                cursor += 1
        pose = np.asarray(values, dtype=np.float64)
        poses[output_row] = pose
        translation[output_row], rotation[output_row] = _pose_errors(pose, target)
        # Row zero is exactly the target matrix by construction.  Eliminate
        # trace/arccos roundoff so the serialized label preserves that identity.
        translation[output_row, 0] = 0.0
        rotation[output_row, 0] = 0.0
        radio_paths.append(str(arrays["radio_token_paths"][source_row]))
        radio_hashes.append(str(arrays["radio_file_sha256"][source_row]))
        contributor_paths.append(str(arrays["contributor_paths"][source_row]))
        contributor_hashes.append(str(arrays["contributor_file_sha256"][source_row]))

    output_arrays = {
        "image_ids": np.asarray(selected),
        "radio_token_paths": np.asarray(radio_paths),
        "radio_file_sha256": np.asarray(radio_hashes),
        "contributor_paths": np.asarray(contributor_paths),
        "contributor_file_sha256": np.asarray(contributor_hashes),
        "candidate_poses_w2c": poses,
        "translation_m": translation,
        "rotation_deg": rotation,
        "candidate_valid": np.ones((query_count, candidate_count), dtype=bool),
        "trajectory_seed_candidate_index": seed_rows,
        "trajectory_alpha": alpha_rows,
    }
    metadata = {
        "artifact_type": CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA,
        "content_sha256": arrays_sha256(output_arrays),
        "supervision_semantics": TRAJECTORY_SUPERVISION_SEMANTICS,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "source_natural_candidate_count_including_anchor": source_candidates,
        "steps_per_seed_excluding_gt_endpoint": steps,
        "trajectory_alpha_values": fractions.tolist(),
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "uses_gt_for_training_candidate_generation": True,
        "natural_candidate_pools_frozen_before_target_pose_opened": True,
        "deployment_candidate_pool": False,
        "canonical_map_excludes_query_route": True,
        "physical_map_sha256": common[0],
        "canonical_field_sha256": common[1],
        "source_dataset_bindings": bindings,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "contains_standard_test_queries": True,
    }
    _atomic_save(output, output_arrays, metadata)
    output.with_suffix(".json").write_text(json.dumps({
        **metadata, "output_npz": str(output.resolve()),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_npz": str(output.resolve()),
        "query_count": query_count,
        "candidate_count": candidate_count,
        "content_sha256": metadata["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
