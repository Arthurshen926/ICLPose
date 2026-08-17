"""Audit scalar/resident exact equivalence, permutation stability, and speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import feature_extract.vfm.localization_goal_maplet.resident_surface_renderer as resident_renderer_module
import feature_extract.vfm.localization_goal_maplet.surface_renderer as scalar_renderer_module

from feature_extract.tools.vfm.audit_goal_maplet_soft_child_renderer_contract import (
    _camera_and_pose,
)
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    left_retract_pose_w2c,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    render_soft_child_surface_field,
)


EXACT_FIELDS = ("child_rows", "child_feature_valid")
NUMERIC_FIELDS = (
    "child_weights", "child_features", "child_tail_weight",
    "unassigned_geometry_weight", "background_weight",
    "canonical_field_missing_weight", "payload_excluded_weight", "null_weight",
    "total_alpha",
)


def _maximum_error(left, right, name: str) -> float:
    a = np.asarray(getattr(left, name), dtype=np.float64)
    b = np.asarray(getattr(right, name), dtype=np.float64)
    return float(np.max(np.abs(a - b), initial=0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--contributor", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose_count", type=int, default=3)
    parser.add_argument("--numeric_tolerance", type=float, default=2e-6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= int(args.pose_count) <= 8:
        raise ValueError("pose_count must lie in [1,8]")
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite resident renderer audit")
    physical_path = Path(args.physical_map)
    field_path = Path(args.canonical_field)
    contributor_path = Path(args.contributor)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    field = CanonicalSurfaceField.load_npz(field_path)
    camera, center = _camera_and_pose(contributor_path)
    coordinates = np.asarray([
        [0, 0, 0, 0, 0, 0],
        [0.3, -0.2, 0.1, 0.2, -0.1, 0.15],
        [-0.4, 0.1, 0.25, -0.1, 0.3, -0.2],
        [0.8, 0.0, -0.2, 0.0, 0.5, 0.0],
        [0.0, -0.7, 0.3, 0.4, 0.0, -0.3],
        [0.2, 0.3, 0.6, -0.4, 0.2, 0.0],
        [-0.6, -0.2, 0.0, 0.1, -0.5, 0.2],
        [0.1, 0.5, -0.4, 0.3, 0.2, -0.4],
    ], dtype=np.float64)[: int(args.pose_count)]
    poses = np.asarray([
        left_retract_pose_w2c(
            center, coordinate, translation_step_m=0.5,
            rotation_step_degrees=10.0,
        ) if np.any(coordinate) else center
        for coordinate in coordinates
    ])
    common = dict(
        width=64, height=36, top_l=4, coordinate_supersample_factor=4,
    )
    scalar = []
    scalar_seconds = []
    for pose in poses:
        started = time.perf_counter()
        scalar.append(render_soft_child_surface_field(
            physical, field, pose, camera, device=str(args.device), **common,
        ))
        scalar_seconds.append(time.perf_counter() - started)
    initialized = time.perf_counter()
    resident = FrozenSoftSurfaceSceneGPU(physical, field, device=str(args.device))
    initialization_seconds = time.perf_counter() - initialized
    forward = resident.render_exact_batch(poses, camera, **common)
    reverse = resident.render_exact_batch(poses[::-1], camera, **common)
    exact = {
        name: bool(all(np.array_equal(getattr(a, name), getattr(b, name))
                       for a, b in zip(scalar, forward.rendered)))
        for name in EXACT_FIELDS
    }
    numeric = {
        name: float(max(
            _maximum_error(a, b, name) for a, b in zip(scalar, forward.rendered)
        ))
        for name in NUMERIC_FIELDS
    }
    permutation_exact = {
        name: bool(all(
            np.array_equal(getattr(forward.rendered[row], name),
                           getattr(reverse.rendered[-1-row], name))
            for row in range(len(poses))
        )) for name in EXACT_FIELDS
    }
    permutation_numeric = {
        name: float(max(
            _maximum_error(forward.rendered[row], reverse.rendered[-1-row], name)
            for row in range(len(poses))
        )) for name in NUMERIC_FIELDS
    }
    tolerance = float(args.numeric_tolerance)
    equivalence = bool(
        all(exact.values()) and all(permutation_exact.values())
        and max(numeric.values(), default=0.0) <= tolerance
        and max(permutation_numeric.values(), default=0.0) <= tolerance
    )
    scalar_total = float(sum(scalar_seconds))
    speedup = scalar_total / max(float(forward.audit.total_seconds), 1e-12)
    speed_gate = bool(
        len(poses) == 8
        and forward.audit.total_seconds < min(scalar_seconds)
        and forward.audit.gpu_child_reducer_implemented
    )
    report = {
        "artifact_type": "goal_maplet_resident_soft_surface_renderer_audit_v1",
        "input_file_sha256": {
            "physical_map": file_sha256(physical_path),
            "canonical_field": file_sha256(field_path),
            "contributor": file_sha256(contributor_path),
        },
        "source_file_sha256": {
            "audit": file_sha256(Path(__file__).resolve()),
            "resident_renderer": file_sha256(Path(resident_renderer_module.__file__).resolve()),
            "scalar_renderer": file_sha256(Path(scalar_renderer_module.__file__).resolve()),
        },
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "pose_count": int(len(poses)),
        "scalar_seconds_by_pose": scalar_seconds,
        "scalar_total_seconds": scalar_total,
        "resident_initialization_seconds": initialization_seconds,
        "resident_forward_audit": forward.audit.__dict__,
        "resident_reverse_audit": reverse.audit.__dict__,
        "scalar_to_resident_batch_speedup": speedup,
        "exact_fields": exact,
        "maximum_numeric_error_by_field": numeric,
        "permutation_exact_fields": permutation_exact,
        "maximum_permutation_error_by_field": permutation_numeric,
        "numeric_tolerance": tolerance,
        "equivalence_gate_passed": equivalence,
        "exact_top8_speed_gate_passed": speed_gate,
        "promotion_eligible": bool(equivalence and speed_gate),
        "blockers": ([] if speed_gate else [
            "deterministic child/feature reduction remains on CPU and exact Top8 speed gate is not passed",
        ]),
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
