"""Run actual multi-basin pattern search from two fixed 2m/45deg GT offsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_pose_capture_direction import (
    _camera_and_token, _errors, _normalized_directions,
)
from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.multibasin_pattern_search import batched_multibasin_pattern_search
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import FrozenSoftSurfaceSceneGPU
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import left_retract_pose_w2c
from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import score_hierarchical_spatial_soft_surface_pose_energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("physical_map", "canonical_field", "surface_mapper", "contributor", "retrieval", "query_pose_file", "image_id", "output_json"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--maximum_sweeps", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite pattern-capture audit")
    paths = {name: Path(getattr(args, name)) for name in (
        "physical_map", "canonical_field", "surface_mapper", "contributor",
        "retrieval", "query_pose_file",
    )}
    physical = GoalMapletPhysicalMap.load_npz(paths["physical_map"])
    field = CanonicalSurfaceField.load_npz(paths["canonical_field"])
    retrieval = PureRadioPhysicalRetrieval.load_npz(paths["retrieval"])
    gt = {row.image_id: row.pose_w2c for row in parse_cambridge_pose_file(paths["query_pose_file"])}[str(args.image_id)]
    camera, token_path = _camera_and_token(paths["contributor"])
    mapper, _ = load_surface_maplet_mapper(paths["surface_mapper"], device=str(args.device))
    query = np.asarray(mapper.project(_load_raw_final(token_path, "radio_final")).measurement_context)
    directions = _normalized_directions()
    initial = np.asarray([
        left_retract_pose_w2c(gt, direction, translation_step_m=2.0, rotation_step_degrees=45.0)
        for direction in directions
    ])
    scene = FrozenSoftSurfaceSceneGPU(physical, field, device=str(args.device))
    renderer_audits = []
    score_cache: dict[bytes, float] = {}

    def score_batch(poses: np.ndarray) -> np.ndarray:
        pose = np.asarray(poses, dtype=np.float64)
        result = np.empty((pose.shape[0],), dtype=np.float64)
        missing = [row for row in range(pose.shape[0]) if pose[row].tobytes() not in score_cache]
        for begin in range(0, len(missing), int(args.render_batch_size)):
            rows = missing[begin : begin + int(args.render_batch_size)]
            batch = scene.render_exact_batch(
                pose[rows], camera, width=query.shape[2], height=query.shape[1],
                selected_child_rows=retrieval.scene_child_rows, top_l=4,
                coordinate_supersample_factor=4,
            )
            renderer_audits.append(batch.audit.__dict__)
            for row, rendered in zip(rows, batch.rendered):
                energy = score_hierarchical_spatial_soft_surface_pose_energy(
                    query, retrieval, rendered,
                    child_to_parent_ids=physical.maplet_ids[physical.child_parent_rows],
                    radio_weight=0.5, spatial_kernel_radius=1,
                )
                score_cache[pose[row].tobytes()] = float(energy.combined_score)
        for row in range(pose.shape[0]):
            result[row] = score_cache[pose[row].tobytes()]
        return result

    initial_errors = [_errors(pose, gt) for pose in initial]
    started = time.perf_counter()
    search = batched_multibasin_pattern_search(
        initial, score_batch,
        translation_radius_m=1.0, rotation_radius_deg=22.5,
        minimum_translation_radius_m=0.125,
        minimum_rotation_radius_deg=2.8125,
        maximum_sweeps=int(args.maximum_sweeps),
        basin_location_ids=[0, 1], maximum_basins=2,
    )
    elapsed = time.perf_counter() - started
    final_by_source = {state.source_basin_index: state for state in search.basins}
    basins = []
    for row, (initial_t, initial_r) in enumerate(initial_errors):
        state = final_by_source[row]
        final_t, final_r = _errors(state.pose_w2c, gt)
        basins.append({
            "source_basin_index": row, "initial_translation_m": initial_t,
            "initial_rotation_deg": initial_r, "final_translation_m": final_t,
            "final_rotation_deg": final_r, "final_score": state.score,
            "accepted_updates": state.accepted_updates,
            "joint_error_scale_initial": max(initial_t / 2.0, initial_r / 45.0),
            "joint_error_scale_final": max(final_t / 2.0, final_r / 45.0),
            "improved_joint_error": max(final_t / 2.0, final_r / 45.0) < max(initial_t / 2.0, initial_r / 45.0),
            "captured_1m_10deg": final_t <= 1.0 and final_r <= 10.0,
            "captured_0_5m_5deg": final_t <= 0.5 and final_r <= 5.0,
        })
    report = {
        "artifact_type": "goal_maplet_full_map_pattern_search_capture_audit_v1",
        "image_id": str(args.image_id),
        "input_file_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "initialization": "two_frozen_coupled_directions_2m_45deg_gt_oracle",
        "maximum_sweeps": int(args.maximum_sweeps),
        "elapsed_seconds": elapsed,
        "evaluator_calls": search.evaluator_calls,
        "evaluated_pose_count": search.evaluated_pose_count,
        "completed_sweeps": search.completed_sweeps,
        "all_basins_improved_joint_error": bool(all(row["improved_joint_error"] for row in basins)),
        "capture_1m_10deg_count": int(sum(row["captured_1m_10deg"] for row in basins)),
        "capture_0_5m_5deg_count": int(sum(row["captured_0_5m_5deg"] for row in basins)),
        "basins": basins, "renderer_batches": renderer_audits,
        "claims": {
            "uses_gt_to_construct_initial_pose": True, "is_localization_result": False,
            "uses_alike": False, "uses_pnp": False, "uses_hard_correspondence": False,
        },
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "renderer_batches"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
