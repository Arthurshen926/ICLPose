"""Gate hierarchy-layout CUDA parity and q0 full-domain runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.global_child_layout_streaming_contract import (
    GATE_SCHEMA, MAXIMUM_QUERY_PARENTS, MAXIMUM_SCENE_CHILDREN,
    POSITION_CHUNK_SIZE, TOPK, TORCH_DTYPE, hierarchy_result_arrays,
    hierarchy_topk_arrays_sha256,
)
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    load_bound_retrieval, load_pose_free_camera_manifest, load_seq10_control_summary,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import load_all_parent_union_support
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    NORMAL_CONTRACT, ParentLayoutCamera, SCORE_SEMANTICS, TOKEN_FOOTPRINT_PHASE,
    score_parent_support_layout_guide,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.streaming_child_layout_guide_gpu import (
    CHILD_GEOMETRY_CONTRACT, CHILD_SCORE_SEMANTICS, FUSION_SEMANTICS, MODES,
    MULTI_STREAMING_SEMANTICS, exact_child_member_rectangle_obbs,
    query_child_layout, score_support_layout_numpy,
    stream_hierarchy_support_layout_topk_gpu,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_topk_npz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite hierarchy GPU gate")
    proposal_path, summary_path = Path(args.proposal).resolve(), Path(args.retrieval_summary).resolve()
    physical_path, camera_path = Path(args.physical_map).resolve(), Path(args.camera_manifest).resolve()
    proposal, proposal_meta = load_all_parent_union_support(proposal_path)
    summary = load_seq10_control_summary(summary_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    physical_file = file_sha256(physical_path)
    retrieval, retrieval_path = load_bound_retrieval(
        summary["rows"][0], physical_file_sha256=physical_file,
        physical_content_sha256=physical.content_sha256,
    )
    cameras, camera_hashes = load_pose_free_camera_manifest(camera_path)
    camera: ParentLayoutCamera = cameras[retrieval.image_id]
    positions = np.asarray(proposal["lattice_origin_world"], np.float64) + (
        np.asarray(proposal["cell_indices_world"], np.float64) + .5
    ) * float(proposal["lattice_spacing_m"])
    rotations = np.asarray(proposal["orientation_rotations_w2c"], np.float64)
    subset_rows = np.unique(np.linspace(0, len(positions) - 1, 64).round().astype(np.int64))
    subset = positions[subset_rows]
    pair_position = np.repeat(np.arange(len(subset), dtype=np.int64), len(rotations))
    pair_orientation = np.tile(np.arange(len(rotations), dtype=np.int64), len(subset))

    started = time.perf_counter()
    parent_cpu = score_parent_support_layout_guide(
        subset[:, None], rotations, np.ones(len(rotations), bool),
        np.arange(len(rotations)), retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        topk=len(subset) * len(rotations), candidate_batch_size=128,
    )
    raw_parent = np.empty(len(subset) * len(rotations), dtype=np.float64)
    parent_linear = (
        parent_cpu.top_position_factor_indices * len(rotations)
        + parent_cpu.top_orientation_factor_indices
    )
    raw_parent[parent_linear] = parent_cpu.top_scores
    child_rows, child_mass, child_integral, complete_child_mass = query_child_layout(
        retrieval, physical, maximum_scene_children=MAXIMUM_SCENE_CHILDREN,
    )
    child_centers, child_corners, child_normals, _ = exact_child_member_rectangle_obbs(
        physical, child_rows,
    )
    child_cpu = score_support_layout_numpy(
        subset, rotations, child_centers, child_corners, child_normals,
        child_integral, complete_child_mass, camera,
        token_height=int(retrieval.metadata["token_height"]),
        token_width=int(retrieval.metadata["token_width"]), candidate_batch_size=128,
    )
    cpu_seconds = time.perf_counter() - started
    raw = {
        "parent": raw_parent,
        "child": child_cpu["score"].reshape(-1),
        "geometric_mean": np.sqrt(raw_parent * child_cpu["score"].reshape(-1)),
    }
    forward = stream_hierarchy_support_layout_topk_gpu(
        subset, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        maximum_scene_children=MAXIMUM_SCENE_CHILDREN,
        topk=len(subset) * len(rotations), position_chunk_size=11,
        device=args.device, torch_dtype=TORCH_DTYPE,
    )
    reverse = stream_hierarchy_support_layout_topk_gpu(
        subset, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        maximum_scene_children=MAXIMUM_SCENE_CHILDREN,
        topk=len(subset) * len(rotations), position_chunk_size=17,
        device=args.device, torch_dtype=TORCH_DTYPE,
        reverse_position_chunks=True, reverse_orientation_evaluation=True,
    )
    parity = {}
    parity_pass = True
    for mode in MODES:
        order = np.lexsort((pair_orientation, pair_position, -raw[mode]))
        ranked = forward.rankings[mode]
        score_delta = float(np.max(np.abs(ranked.top_scores - raw[mode][order])))
        indices_exact = bool(
            np.array_equal(ranked.top_position_rows, pair_position[order])
            and np.array_equal(ranked.top_orientation_rows, pair_orientation[order])
        )
        reverse_exact = bool(all(np.array_equal(
            np.asarray(getattr(ranked, name)),
            np.asarray(getattr(reverse.rankings[mode], name)),
        ) for name in (
            "top_scores", "top_position_rows", "top_orientation_rows",
            "top_parent_scores", "top_child_scores",
            "top_parent_visible_counts", "top_child_visible_counts",
            "top_child_front_facing_counts", "top_child_positive_depth_counts",
            "top_child_center_in_image_counts",
            "top_child_projected_token_footprint_mass",
            "top_child_sqrt_overlap_mass",
        )))
        parity[mode] = {
            "indices_exact": indices_exact,
            "score_max_abs_delta": score_delta,
            "score_atol": 1e-12,
            "forward_reverse_all_arrays_exact": reverse_exact,
        }
        parity_pass &= indices_exact and reverse_exact and score_delta <= 1e-12
    child_order = np.lexsort((pair_orientation, pair_position, -raw["child"]))
    child_ranked = forward.rankings["child"]
    child_diagnostics = {
        "visible_counts_exact": bool(np.array_equal(
            child_ranked.top_child_visible_counts,
            child_cpu["visible"].reshape(-1)[child_order],
        )),
        "front_facing_counts_exact": bool(np.array_equal(
            child_ranked.top_child_front_facing_counts,
            child_cpu["front"].reshape(-1)[child_order],
        )),
        "positive_depth_counts_exact": bool(np.array_equal(
            child_ranked.top_child_positive_depth_counts,
            child_cpu["depth"].reshape(-1)[child_order],
        )),
        "center_in_image_counts_exact": bool(np.array_equal(
            child_ranked.top_child_center_in_image_counts,
            child_cpu["center_image"].reshape(-1)[child_order],
        )),
        "footprint_mass_exact": bool(np.array_equal(
            child_ranked.top_child_projected_token_footprint_mass,
            child_cpu["footprint"].reshape(-1)[child_order],
        )),
        "sqrt_overlap_max_abs_delta": float(np.max(np.abs(
            child_ranked.top_child_sqrt_overlap_mass
            - child_cpu["overlap"].reshape(-1)[child_order]
        ))),
        "sqrt_overlap_atol": 1.0e-9,
    }
    child_diagnostic_pass = bool(
        all(child_diagnostics[key] for key in (
            "visible_counts_exact", "front_facing_counts_exact",
            "positive_depth_counts_exact", "center_in_image_counts_exact",
            "footprint_mass_exact",
        ))
        and child_diagnostics["sqrt_overlap_max_abs_delta"] <= 1.0e-9
    )
    parity_pass &= child_diagnostic_pass
    if not parity_pass:
        raise RuntimeError(
            "hierarchy layout parity gate failed: "
            + json.dumps({"modes": parity, "child": child_diagnostics}, sort_keys=True)
        )
    full = stream_hierarchy_support_layout_topk_gpu(
        positions, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        maximum_scene_children=MAXIMUM_SCENE_CHILDREN, topk=TOPK,
        position_chunk_size=POSITION_CHUNK_SIZE, device=args.device,
        torch_dtype=TORCH_DTYPE,
    )
    runtime_pass = full.elapsed_seconds <= 15.0
    if args.output_topk_npz:
        topk_output = Path(args.output_topk_npz).resolve()
        if topk_output.exists() and not args.force:
            raise FileExistsError("refusing to overwrite hierarchy gate Top-K arrays")
        topk_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(topk_output, **hierarchy_result_arrays(retrieval.image_id, full))
    props = torch.cuda.get_device_properties(torch.device(args.device))
    report = {
        "artifact_type": GATE_SCHEMA,
        "decision": "PASS" if runtime_pass else "KILL_RUNTIME",
        "proposal": str(proposal_path), "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": proposal_meta["content_sha256"],
        "retrieval_summary": str(summary_path),
        "retrieval_summary_file_sha256": file_sha256(summary_path),
        "retrieval": str(retrieval_path), "retrieval_file_sha256": file_sha256(retrieval_path),
        "retrieval_content_sha256": retrieval.content_sha256,
        "physical_map": str(physical_path), "physical_map_file_sha256": physical_file,
        "physical_map_content_sha256": physical.content_sha256,
        "camera_manifest": str(camera_path),
        "camera_manifest_file_sha256": file_sha256(camera_path),
        "camera_intrinsics_content_sha256": camera_hashes[retrieval.image_id],
        "image_id": retrieval.image_id,
        "position_count": len(positions), "orientation_count": len(rotations),
        "total_factor_pair_count": int(len(positions) * len(rotations)),
        "maximum_query_parents": MAXIMUM_QUERY_PARENTS,
        "maximum_scene_children": MAXIMUM_SCENE_CHILDREN,
        "returned_topk": TOPK, "position_chunk_size": POSITION_CHUNK_SIZE,
        "torch_dtype": TORCH_DTYPE, "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)),
        "cuda_device_total_memory_bytes": int(props.total_memory),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda),
        "parent_score_semantics": SCORE_SEMANTICS,
        "child_score_semantics": CHILD_SCORE_SEMANTICS,
        "child_geometry_contract": CHILD_GEOMETRY_CONTRACT,
        "fusion_semantics": FUSION_SEMANTICS,
        "streaming_semantics": MULTI_STREAMING_SEMANTICS,
        "normal_contract": NORMAL_CONTRACT, "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
        "subset_gate": {
            "global_position_rows": subset_rows.tolist(), "factor_pair_count": len(subset) * len(rotations),
            "cpu_seconds": cpu_seconds, "gpu_forward_seconds": forward.elapsed_seconds,
            "gpu_reverse_seconds": reverse.elapsed_seconds, "modes": parity,
            "child_diagnostics": child_diagnostics,
            "decision": "PASS" if parity_pass else "FAIL",
        },
        "full_q0_runtime_gate": {
            "elapsed_seconds": full.elapsed_seconds, "scoring_seconds": full.scoring_seconds,
            "stable_merge_seconds": full.merge_seconds, "maximum_seconds": 15.0,
            "peak_cuda_allocated_bytes": full.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": full.peak_cuda_reserved_bytes,
            "topk_arrays_sha256": hierarchy_topk_arrays_sha256(full),
            "top_by_mode": {mode: {
                "score": float(full.rankings[mode].top_scores[0]),
                "position": int(full.rankings[mode].top_position_rows[0]),
                "orientation": int(full.rankings[mode].top_orientation_rows[0]),
            } for mode in MODES},
            "decision": "PASS" if runtime_pass else "KILL_RUNTIME",
        },
        "complete_child_probability_mass": full.complete_child_probability_mass,
        "selected_child_probability_mass_total": full.selected_child_probability_mass_total,
        "selected_child_probability_mass_fraction": (
            full.selected_child_probability_mass_total / full.complete_child_probability_mass
        ),
        "uses_query_pose": False, "uses_query_ground_truth": False,
        "uses_contributor_artifact": False, "uses_alike": False, "uses_pnp": False,
        "uses_optimizer": False, "uses_renderer": False,
        "cartesian_score_array_materialized": False, "phase2_labels_opened": False,
        "control_only": True, "production_eligible": False,
        "promotion_blockers": summary["promotion_blockers"],
        "held_evaluation_authorized": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"],
                      "elapsed_seconds": full.elapsed_seconds}, indent=2))


if __name__ == "__main__":
    main()
