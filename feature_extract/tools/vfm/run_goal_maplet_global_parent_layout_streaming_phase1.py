"""Freeze seq10 global-domain parent-layout Top-K before opening labels.

The executable intentionally has no contributor, pose, direct-dataset, or
ground-truth argument.  It scores every query from the frozen seq10 retrieval
control using only RADIO parent posteriors, physical geometry, a pose-free
camera manifest, and the frozen global-v2 factor domain.
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
    RANKING_KEY,
    RETURNED_TOPK,
    RUN_SCHEMA,
    SCORE_SCHEMA,
    TORCH_DTYPE,
    atomic_save_score,
    load_bound_retrieval,
    load_json_no_duplicate_keys,
    load_pose_free_camera_manifest,
    load_seq10_control_summary,
    load_streaming_score,
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
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.streaming_parent_layout_guide_gpu import (
    STREAMING_SEMANTICS,
    stream_parent_support_layout_topk_gpu,
)


def _result_hash(result: object) -> str:
    names = (
        "top_scores", "top_position_rows", "top_orientation_rows",
        "top_visible_parent_counts", "top_front_facing_parent_counts",
        "top_positive_depth_parent_counts", "top_center_in_image_parent_counts",
        "top_projected_token_footprint_mass", "top_sqrt_overlap_mass",
    )
    return arrays_sha256({name: np.asarray(getattr(result, name)) for name in names})


def _validate_gate(path: Path) -> dict[str, object]:
    gate = load_json_no_duplicate_keys(path)
    unhashed = dict(gate)
    content = unhashed.pop("content_sha256", "")
    if (
        gate.get("artifact_type") != GATE_SCHEMA
        or gate.get("decision") != "PASS"
        or content != canonical_json_sha256(unhashed)
        or gate.get("maximum_query_parents") != MAXIMUM_QUERY_PARENTS
        or gate.get("returned_topk") != RETURNED_TOPK
        or gate.get("position_chunk_size") != POSITION_CHUNK_SIZE
        or gate.get("torch_dtype") != TORCH_DTYPE
        or gate.get("score_semantics") != SCORE_SEMANTICS
        or gate.get("streaming_semantics") != STREAMING_SEMANTICS
        or gate.get("normal_contract") != NORMAL_CONTRACT
        or gate.get("token_footprint_phase") != TOKEN_FOOTPRINT_PHASE
        or gate.get("cuda_device_name")
        != torch.cuda.get_device_name(torch.device(str(gate.get("device"))))
        or gate.get("torch_version") != str(torch.__version__)
        or gate.get("torch_cuda_version") != str(torch.version.cuda)
        or gate.get("uses_query_pose") is not False
        or gate.get("uses_query_ground_truth") is not False
        or gate.get("uses_contributor_artifact") is not False
        or gate.get("phase2_labels_opened") is not False
        or gate.get("subset_gate", {}).get("decision") != "PASS"
        or gate.get("full_q0_runtime_gate", {}).get("decision") != "PASS"
    ):
        raise ValueError("Phase-1 requires a valid passing GPU gate")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--gpu_gate", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    proposal_path = Path(args.proposal).resolve()
    summary_path = Path(args.retrieval_summary).resolve()
    physical_path = Path(args.physical_map).resolve()
    camera_path = Path(args.camera_manifest).resolve()
    gate_path = Path(args.gpu_gate).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_path = output_dir / "phase1_score_run.json"
    if run_path.exists():
        raise FileExistsError("refusing to overwrite completed Phase-1 run")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    gate = _validate_gate(gate_path)
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
        raise ValueError("Phase-1 physical lineage differs")
    cameras, camera_bindings = load_pose_free_camera_manifest(camera_path)
    bindings = {
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": proposal_metadata["content_sha256"],
        "retrieval_summary_file_sha256": file_sha256(summary_path),
        "physical_map_file_sha256": physical_file,
        "physical_map_content_sha256": physical.content_sha256,
        "camera_manifest_file_sha256": file_sha256(camera_path),
    }
    if (
        gate.get("proposal_file_sha256") != bindings["proposal_file_sha256"]
        or gate.get("proposal_content_sha256") != bindings["proposal_content_sha256"]
        or gate.get("retrieval_summary_file_sha256")
        != bindings["retrieval_summary_file_sha256"]
        or gate.get("physical_map_file_sha256") != physical_file
        or gate.get("physical_map_content_sha256") != physical.content_sha256
        or gate.get("camera_manifest_file_sha256")
        != bindings["camera_manifest_file_sha256"]
        or gate.get("device") != str(args.device)
    ):
        raise ValueError("Phase-1 input/configuration differs from GPU gate")

    origin = np.asarray(proposal["lattice_origin_world"], dtype=np.float64)
    positions = origin + (
        np.asarray(proposal["cell_indices_world"], dtype=np.float64) + 0.5
    ) * float(proposal["lattice_spacing_m"])
    rotations = np.asarray(proposal["orientation_rotations_w2c"], dtype=np.float64)
    all_rows = []
    total_started = time.perf_counter()
    for query_index, row in enumerate(summary["rows"]):
        image_id = str(row["image_id"])
        camera = cameras.get(image_id)
        if camera is None:
            raise ValueError("Phase-1 query is absent from pose-free camera manifest")
        retrieval, retrieval_path = load_bound_retrieval(
            row, physical_file_sha256=physical_file,
            physical_content_sha256=physical.content_sha256,
        )
        output = scores_dir / (image_id.replace("/", "__") + ".npz")
        if output.exists():
            arrays, metadata = load_streaming_score(output)
            if (
                metadata.get("query_index") != query_index
                or str(np.asarray(arrays["image_id"]).item()) != image_id
                or metadata.get("retrieval_file_sha256") != file_sha256(retrieval_path)
                or metadata.get("camera_intrinsics_content_sha256")
                != camera_bindings[image_id]
                or any(metadata.get(key) != value for key, value in bindings.items())
                or metadata.get("position_count") != int(positions.shape[0])
                or metadata.get("orientation_count") != int(rotations.shape[0])
                or metadata.get("total_factor_pair_count")
                != int(positions.shape[0] * rotations.shape[0])
                or metadata.get("score_semantics") != SCORE_SEMANTICS
                or metadata.get("streaming_semantics") != STREAMING_SEMANTICS
                or metadata.get("normal_contract") != NORMAL_CONTRACT
                or metadata.get("token_footprint_phase") != TOKEN_FOOTPRINT_PHASE
                or metadata.get("device") != str(args.device)
            ):
                raise ValueError("existing partial Phase-1 score differs")
            elapsed = float(metadata["elapsed_seconds"])
            content = str(metadata["content_sha256"])
        else:
            result = stream_parent_support_layout_topk_gpu(
                positions, rotations, retrieval, physical, camera,
                maximum_query_parents=MAXIMUM_QUERY_PARENTS,
                topk=RETURNED_TOPK, position_chunk_size=POSITION_CHUNK_SIZE,
                device=args.device, torch_dtype=TORCH_DTYPE,
            )
            if query_index == 0 and _result_hash(result) != gate[
                "full_q0_runtime_gate"
            ]["topk_arrays_sha256"]:
                raise RuntimeError("Phase-1 q0 bytes differ from passing GPU gate")
            arrays = {
                "image_id": np.asarray(image_id),
                "selected_query_parent_ids": np.asarray(
                    result.selected_query_parent_ids, dtype=np.int64,
                ),
                "selected_query_parent_probability_mass": np.asarray(
                    result.selected_query_parent_probability_mass, dtype=np.float64,
                ),
                "top_scores": np.asarray(result.top_scores, dtype=np.float64),
                "top_position_factor_indices": np.asarray(
                    result.top_position_rows, dtype=np.int64,
                ),
                "top_orientation_factor_indices": np.asarray(
                    result.top_orientation_rows, dtype=np.int64,
                ),
                "top_visible_parent_counts": np.asarray(
                    result.top_visible_parent_counts, dtype=np.int16,
                ),
                "top_front_facing_parent_counts": np.asarray(
                    result.top_front_facing_parent_counts, dtype=np.int16,
                ),
                "top_positive_depth_parent_counts": np.asarray(
                    result.top_positive_depth_parent_counts, dtype=np.int16,
                ),
                "top_center_in_image_parent_counts": np.asarray(
                    result.top_center_in_image_parent_counts, dtype=np.int16,
                ),
                "top_projected_token_footprint_mass": np.asarray(
                    result.top_projected_token_footprint_mass, dtype=np.float64,
                ),
                "top_sqrt_overlap_mass": np.asarray(
                    result.top_sqrt_overlap_mass, dtype=np.float64,
                ),
            }
            content = arrays_sha256(arrays)
            elapsed = float(result.elapsed_seconds)
            metadata = {
                "artifact_type": SCORE_SCHEMA,
                "content_sha256": content,
                "image_id": image_id,
                "query_index": query_index,
                "proposal": str(proposal_path),
                **bindings,
                "retrieval_summary": str(summary_path),
                "retrieval": str(retrieval_path),
                "retrieval_file_sha256": file_sha256(retrieval_path),
                "retrieval_content_sha256": retrieval.content_sha256,
                "camera_manifest": str(camera_path),
                "camera_intrinsics_content_sha256": camera_bindings[image_id],
                "position_count": int(positions.shape[0]),
                "orientation_count": int(rotations.shape[0]),
                "total_factor_pair_count": int(result.total_factor_pair_count),
                "maximum_query_parents": MAXIMUM_QUERY_PARENTS,
                "selected_query_parent_count": int(
                    result.selected_query_parent_ids.size
                ),
                "complete_query_parent_probability_mass": float(
                    result.complete_query_parent_probability_mass
                ),
                "selected_query_parent_probability_mass_total": float(
                    result.selected_query_parent_probability_mass_total
                ),
                "returned_topk": RETURNED_TOPK,
                "position_chunk_size": POSITION_CHUNK_SIZE,
                "torch_dtype": TORCH_DTYPE,
                "device": str(args.device),
                "ranking_key": RANKING_KEY,
                "score_semantics": SCORE_SEMANTICS,
                "streaming_semantics": STREAMING_SEMANTICS,
                "normal_contract": NORMAL_CONTRACT,
                "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
                "elapsed_seconds": elapsed,
                "scoring_seconds": float(result.scoring_seconds),
                "stable_merge_seconds": float(result.merge_seconds),
                "peak_cuda_allocated_bytes": int(result.peak_cuda_allocated_bytes),
                "peak_cuda_reserved_bytes": int(result.peak_cuda_reserved_bytes),
                "control_only": True,
                "production_eligible": False,
                "promotion_blockers": summary["promotion_blockers"],
                "uses_query_image_features": True,
                "uses_query_rgb_directly": False,
                "uses_query_pose": False,
                "uses_query_ground_truth": False,
                "uses_contributor_artifact": False,
                "uses_alike": False,
                "uses_pnp": False,
                "uses_point_correspondences": False,
                "uses_optimizer": False,
                "uses_renderer": False,
                "cartesian_pose_product_materialized": False,
                "cartesian_score_array_materialized": False,
                "phase2_labels_opened": False,
            }
            atomic_save_score(output, arrays, metadata)
            load_streaming_score(output)
        all_rows.append({
            "query_index": query_index,
            "image_id": image_id,
            "artifact": str(output),
            "artifact_file_sha256": file_sha256(output),
            "content_sha256": content,
            "elapsed_seconds": elapsed,
        })
        print(json.dumps({
            "query_index": query_index, "image_id": image_id,
            "elapsed_seconds": elapsed,
        }, sort_keys=True), flush=True)

    report = {
        "artifact_type": RUN_SCHEMA,
        "query_route": "seq10",
        "query_count": len(all_rows),
        "rows": all_rows,
        "proposal": str(proposal_path),
        **bindings,
        "retrieval_summary": str(summary_path),
        "camera_manifest": str(camera_path),
        "gpu_gate": str(gate_path),
        "gpu_gate_file_sha256": file_sha256(gate_path),
        "gpu_gate_content_sha256": gate["content_sha256"],
        "position_count": int(positions.shape[0]),
        "orientation_count": int(rotations.shape[0]),
        "total_factor_pair_count_per_query": int(positions.shape[0] * rotations.shape[0]),
        "maximum_query_parents": MAXIMUM_QUERY_PARENTS,
        "returned_topk_per_query": RETURNED_TOPK,
        "position_chunk_size": POSITION_CHUNK_SIZE,
        "torch_dtype": TORCH_DTYPE,
        "device": str(args.device),
        "ranking_key": RANKING_KEY,
        "score_semantics": SCORE_SEMANTICS,
        "streaming_semantics": STREAMING_SEMANTICS,
        "normal_contract": NORMAL_CONTRACT,
        "token_footprint_phase": TOKEN_FOOTPRINT_PHASE,
        "total_elapsed_seconds": float(time.perf_counter() - total_started),
        "control_only": True,
        "production_eligible": False,
        "promotion_blockers": summary["promotion_blockers"],
        "score_before_label_contract": {
            "phase1_cli_has_no_label_pose_contributor_argument": True,
            "camera_source_is_pose_free_manifest_only": True,
            "contributor_bytes_or_members_opened": False,
            "phase2_labels_opened": False,
            "all_score_artifact_bytes_frozen_before_phase2": True,
        },
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_contributor_artifact": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_optimizer": False,
        "uses_renderer": False,
        "cartesian_score_array_materialized": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    run_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(run_path), "query_count": len(all_rows),
        "total_elapsed_seconds": report["total_elapsed_seconds"],
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
