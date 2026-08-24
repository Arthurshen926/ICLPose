"""Gate CUDA parity and full-domain runtime before seq10 Phase-1 scoring.

This executable accepts no pose/label/contributor input.  Camera calibration
comes only from an explicitly pose-free manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    GATE_SCHEMA,
    MAXIMUM_QUERY_PARENTS,
    POSITION_CHUNK_SIZE,
    RETURNED_TOPK,
    TORCH_DTYPE,
    load_bound_retrieval,
    load_pose_free_camera_manifest,
    load_seq10_control_summary,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    NORMAL_CONTRACT,
    SCORE_SEMANTICS,
    TOKEN_FOOTPRINT_PHASE,
    score_parent_support_layout_guide,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.streaming_parent_layout_guide_gpu import (
    STREAMING_SEMANTICS,
    stream_parent_support_layout_topk_gpu,
)


def _stable_result_hash(result: object) -> str:
    names = (
        "top_scores", "top_position_rows", "top_orientation_rows",
        "top_visible_parent_counts", "top_front_facing_parent_counts",
        "top_positive_depth_parent_counts", "top_center_in_image_parent_counts",
        "top_projected_token_footprint_mass", "top_sqrt_overlap_mass",
    )
    return arrays_sha256({name: np.asarray(getattr(result, name)) for name in names})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite GPU gate report")

    proposal_path = Path(args.proposal).resolve()
    summary_path = Path(args.retrieval_summary).resolve()
    physical_path = Path(args.physical_map).resolve()
    camera_path = Path(args.camera_manifest).resolve()
    proposal, proposal_metadata = load_all_parent_union_support(proposal_path)
    summary = load_seq10_control_summary(summary_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    physical_file = file_sha256(physical_path)
    proposal_physical = proposal_metadata.get("physical_map", {})
    if (
        not isinstance(proposal_physical, dict)
        or proposal_physical.get("content_sha256") != physical.content_sha256
        or proposal_physical.get("file_sha256") != physical_file
        or summary.get("physical_map_sha256") != physical.content_sha256
    ):
        raise ValueError("GPU gate physical lineage differs")
    cameras, camera_bindings = load_pose_free_camera_manifest(camera_path)
    row = summary["rows"][0]
    retrieval, retrieval_path = load_bound_retrieval(
        row, physical_file_sha256=physical_file,
        physical_content_sha256=physical.content_sha256,
    )
    if retrieval.image_id not in cameras:
        raise ValueError("GPU gate query camera is absent")
    camera = cameras[retrieval.image_id]
    origin = np.asarray(proposal["lattice_origin_world"], dtype=np.float64)
    spacing = float(proposal["lattice_spacing_m"])
    positions = origin + (
        np.asarray(proposal["cell_indices_world"], dtype=np.float64) + 0.5
    ) * spacing
    rotations = np.asarray(proposal["orientation_rotations_w2c"], dtype=np.float64)

    subset_global_rows = np.unique(
        np.linspace(0, positions.shape[0] - 1, 64).round().astype(np.int64),
    )
    subset = positions[subset_global_rows]
    started = time.perf_counter()
    cpu = score_parent_support_layout_guide(
        subset[:, None, :], rotations, np.ones(rotations.shape[0], dtype=bool),
        np.arange(rotations.shape[0], dtype=np.int64), retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        topk=int(subset.shape[0] * rotations.shape[0]), candidate_batch_size=256,
    )
    cpu_seconds = time.perf_counter() - started
    forward = stream_parent_support_layout_topk_gpu(
        subset, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        topk=int(subset.shape[0] * rotations.shape[0]), position_chunk_size=11,
        device=args.device, torch_dtype=TORCH_DTYPE,
    )
    reverse = stream_parent_support_layout_topk_gpu(
        subset, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS,
        topk=int(subset.shape[0] * rotations.shape[0]), position_chunk_size=17,
        device=args.device, torch_dtype=TORCH_DTYPE,
        reverse_position_chunks=True, reverse_orientation_evaluation=True,
    )
    score_delta = float(np.max(np.abs(cpu.top_scores - forward.top_scores)))
    overlap_delta = float(np.max(np.abs(
        cpu.top_sqrt_overlap_mass - forward.top_sqrt_overlap_mass
    )))
    exact_names = (
        (cpu.top_position_factor_indices, forward.top_position_rows),
        (cpu.top_orientation_factor_indices, forward.top_orientation_rows),
        (cpu.top_visible_parent_counts, forward.top_visible_parent_counts),
        (cpu.top_front_facing_parent_counts, forward.top_front_facing_parent_counts),
        (cpu.top_positive_depth_parent_counts, forward.top_positive_depth_parent_counts),
        (cpu.top_center_in_image_parent_counts, forward.top_center_in_image_parent_counts),
        (cpu.top_projected_token_footprint_mass,
         forward.top_projected_token_footprint_mass),
    )
    cpu_gpu_discrete_exact = bool(all(np.array_equal(a, b) for a, b in exact_names))
    forward_reverse_exact = _stable_result_hash(forward) == _stable_result_hash(reverse)
    parity_pass = bool(
        cpu_gpu_discrete_exact and forward_reverse_exact
        and score_delta <= 1.0e-12 and overlap_delta <= 1.0e-9
    )
    if not parity_pass:
        raise RuntimeError("GPU streaming layout parity gate failed")

    full = stream_parent_support_layout_topk_gpu(
        positions, rotations, retrieval, physical, camera,
        maximum_query_parents=MAXIMUM_QUERY_PARENTS, topk=RETURNED_TOPK,
        position_chunk_size=POSITION_CHUNK_SIZE, device=args.device,
        torch_dtype=TORCH_DTYPE,
    )
    runtime_pass = bool(full.elapsed_seconds <= 15.0)
    report = {
        "artifact_type": GATE_SCHEMA,
        "decision": "PASS" if runtime_pass else "KILL_RUNTIME",
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": proposal_metadata["content_sha256"],
        "retrieval_summary": str(summary_path),
        "retrieval_summary_file_sha256": file_sha256(summary_path),
        "retrieval": str(retrieval_path),
        "retrieval_file_sha256": file_sha256(retrieval_path),
        "retrieval_content_sha256": retrieval.content_sha256,
        "physical_map": str(physical_path),
        "physical_map_file_sha256": physical_file,
        "physical_map_content_sha256": physical.content_sha256,
        "camera_manifest": str(camera_path),
        "camera_manifest_file_sha256": file_sha256(camera_path),
        "camera_intrinsics_content_sha256": camera_bindings[retrieval.image_id],
        "image_id": retrieval.image_id,
        "position_count": int(positions.shape[0]),
        "orientation_count": int(rotations.shape[0]),
        "total_factor_pair_count": int(full.total_factor_pair_count),
        "maximum_query_parents": MAXIMUM_QUERY_PARENTS,
        "returned_topk": RETURNED_TOPK,
        "position_chunk_size": POSITION_CHUNK_SIZE,
        "torch_dtype": TORCH_DTYPE,
        "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)),
        "cuda_device_total_memory_bytes": int(
            torch.cuda.get_device_properties(torch.device(args.device)).total_memory
        ),
        "cuda_compute_capability": list(
            torch.cuda.get_device_capability(torch.device(args.device))
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "score_semantics": SCORE_SEMANTICS,
        "streaming_semantics": STREAMING_SEMANTICS,
        "normal_contract": NORMAL_CONTRACT,
        "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
        "subset_gate": {
            "global_position_rows": subset_global_rows.tolist(),
            "factor_pair_count": int(subset.shape[0] * rotations.shape[0]),
            "cpu_seconds": float(cpu_seconds),
            "gpu_forward_seconds": forward.elapsed_seconds,
            "gpu_reverse_seconds": reverse.elapsed_seconds,
            "cpu_gpu_discrete_exact": cpu_gpu_discrete_exact,
            "score_max_abs_delta": score_delta,
            "score_atol": 1.0e-12,
            "sqrt_overlap_max_abs_delta": overlap_delta,
            "sqrt_overlap_atol": 1.0e-9,
            "forward_reverse_all_arrays_exact": forward_reverse_exact,
            "forward_arrays_sha256": _stable_result_hash(forward),
            "reverse_arrays_sha256": _stable_result_hash(reverse),
            "decision": "PASS",
        },
        "full_q0_runtime_gate": {
            "elapsed_seconds": full.elapsed_seconds,
            "scoring_seconds": full.scoring_seconds,
            "stable_merge_seconds": full.merge_seconds,
            "maximum_seconds": 15.0,
            "peak_cuda_allocated_bytes": full.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": full.peak_cuda_reserved_bytes,
            "topk_arrays_sha256": _stable_result_hash(full),
            "top_score": float(full.top_scores[0]),
            "top_position_factor_index": int(full.top_position_rows[0]),
            "top_orientation_factor_index": int(full.top_orientation_rows[0]),
            "decision": "PASS" if runtime_pass else "KILL_RUNTIME",
        },
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_contributor_artifact": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_optimizer": False,
        "uses_renderer": False,
        "cartesian_score_array_materialized": False,
        "phase2_labels_opened": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output), "decision": report["decision"],
        "elapsed_seconds": full.elapsed_seconds,
        "peak_cuda_allocated_bytes": full.peak_cuda_allocated_bytes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
