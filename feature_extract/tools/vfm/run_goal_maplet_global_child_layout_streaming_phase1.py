"""Score one strict shard of seq10 hierarchy-layout Phase-1 on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.global_child_layout_streaming_contract import (
    GATE_SCHEMA, MAXIMUM_QUERY_PARENTS, MAXIMUM_SCENE_CHILDREN,
    POSITION_CHUNK_SIZE, SCORE_SCHEMA, SHARD_RUN_SCHEMA, TOPK, TORCH_DTYPE,
    hierarchy_result_arrays, hierarchy_topk_arrays_sha256, load_hierarchy_score,
)
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    load_bound_retrieval, load_json_no_duplicate_keys,
    load_pose_free_camera_manifest, load_seq10_control_summary,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import load_all_parent_union_support
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import NORMAL_CONTRACT, SCORE_SEMANTICS, TOKEN_FOOTPRINT_PHASE
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.streaming_child_layout_guide_gpu import (
    CHILD_GEOMETRY_CONTRACT, CHILD_SCORE_SEMANTICS, FUSION_SEMANTICS,
    MULTI_STREAMING_SEMANTICS, stream_hierarchy_support_layout_topk_gpu,
)


def _save(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gate(path: Path, device: str) -> dict:
    gate = load_json_no_duplicate_keys(path)
    unhashed = dict(gate); content = unhashed.pop("content_sha256", "")
    if (
        gate.get("artifact_type") != GATE_SCHEMA or gate.get("decision") != "PASS"
        or content != canonical_json_sha256(unhashed)
        or gate.get("maximum_query_parents") != MAXIMUM_QUERY_PARENTS
        or gate.get("maximum_scene_children") != MAXIMUM_SCENE_CHILDREN
        or gate.get("returned_topk") != TOPK
        or gate.get("position_chunk_size") != POSITION_CHUNK_SIZE
        or gate.get("torch_dtype") != TORCH_DTYPE
        or gate.get("parent_score_semantics") != SCORE_SEMANTICS
        or gate.get("child_score_semantics") != CHILD_SCORE_SEMANTICS
        or gate.get("child_geometry_contract") != CHILD_GEOMETRY_CONTRACT
        or gate.get("fusion_semantics") != FUSION_SEMANTICS
        or gate.get("streaming_semantics") != MULTI_STREAMING_SEMANTICS
        or gate.get("normal_contract") != NORMAL_CONTRACT
        or gate.get("token_footprint_phase") != TOKEN_FOOTPRINT_PHASE
        or gate.get("torch_version") != str(torch.__version__)
        or gate.get("torch_cuda_version") != str(torch.version.cuda)
        or gate.get("cuda_device_name") != torch.cuda.get_device_name(torch.device(device))
        or gate.get("subset_gate", {}).get("decision") != "PASS"
        or gate.get("full_q0_runtime_gate", {}).get("decision") != "PASS"
    ):
        raise ValueError("hierarchy Phase-1 GPU gate differs")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True); parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--physical_map", required=True); parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--gpu_gate", required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--shard_count", type=int, required=True); parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid hierarchy Phase-1 shard")
    output_dir = Path(args.output_dir).resolve(); run_path = output_dir / "phase1_shard.json"
    if run_path.exists(): raise FileExistsError("refusing to overwrite completed hierarchy shard")
    output_dir.mkdir(parents=True, exist_ok=True); scores_dir = output_dir / "scores"; scores_dir.mkdir(exist_ok=True)
    proposal_path=Path(args.proposal).resolve(); summary_path=Path(args.retrieval_summary).resolve()
    physical_path=Path(args.physical_map).resolve(); camera_path=Path(args.camera_manifest).resolve(); gate_path=Path(args.gpu_gate).resolve()
    gate=_gate(gate_path,args.device); proposal,proposal_meta=load_all_parent_union_support(proposal_path)
    summary=load_seq10_control_summary(summary_path); physical=GoalMapletPhysicalMap.load_npz(physical_path)
    physical_file=file_sha256(physical_path); cameras,camera_hashes=load_pose_free_camera_manifest(camera_path)
    bindings={"proposal_file_sha256":file_sha256(proposal_path),"proposal_content_sha256":proposal_meta["content_sha256"],"retrieval_summary_file_sha256":file_sha256(summary_path),"physical_map_file_sha256":physical_file,"physical_map_content_sha256":physical.content_sha256,"camera_manifest_file_sha256":file_sha256(camera_path),"gpu_gate_file_sha256":file_sha256(gate_path),"gpu_gate_content_sha256":gate["content_sha256"]}
    for key in ("proposal_file_sha256","proposal_content_sha256","retrieval_summary_file_sha256","physical_map_file_sha256","physical_map_content_sha256","camera_manifest_file_sha256"):
        if gate.get(key)!=bindings[key]: raise ValueError("hierarchy shard input differs from GPU gate")
    positions=np.asarray(proposal["lattice_origin_world"],np.float64)+(np.asarray(proposal["cell_indices_world"],np.float64)+.5)*float(proposal["lattice_spacing_m"])
    rotations=np.asarray(proposal["orientation_rotations_w2c"],np.float64)
    splits=np.array_split(np.arange(len(summary["rows"]),dtype=np.int64),args.shard_count); query_indices=splits[args.shard_index]
    rows=[]; started=time.perf_counter()
    for query_index in query_indices.tolist():
        row=summary["rows"][query_index]; image_id=str(row["image_id"]); camera=cameras[image_id]
        retrieval,retrieval_path=load_bound_retrieval(row,physical_file_sha256=physical_file,physical_content_sha256=physical.content_sha256)
        output=scores_dir/(image_id.replace("/","__")+".npz")
        if output.exists():
            arrays,metadata=load_hierarchy_score(output)
            if metadata.get("query_index")!=query_index or any(metadata.get(k)!=v for k,v in bindings.items()) or metadata.get("device")!=args.device:
                raise ValueError("stale hierarchy shard partial differs")
            elapsed=float(metadata["elapsed_seconds"]); content=str(metadata["content_sha256"])
        else:
            result=stream_hierarchy_support_layout_topk_gpu(positions,rotations,retrieval,physical,camera,maximum_query_parents=MAXIMUM_QUERY_PARENTS,maximum_scene_children=MAXIMUM_SCENE_CHILDREN,topk=TOPK,position_chunk_size=POSITION_CHUNK_SIZE,device=args.device,torch_dtype=TORCH_DTYPE)
            if query_index==0 and hierarchy_topk_arrays_sha256(result)!=gate["full_q0_runtime_gate"]["topk_arrays_sha256"]:
                raise RuntimeError("hierarchy q0 bytes differ from gate")
            arrays=hierarchy_result_arrays(image_id,result); content=arrays_sha256(arrays); elapsed=float(result.elapsed_seconds)
            metadata={"artifact_type":SCORE_SCHEMA,"content_sha256":content,"image_id":image_id,"query_index":query_index,**bindings,"retrieval":str(retrieval_path),"retrieval_file_sha256":file_sha256(retrieval_path),"retrieval_content_sha256":retrieval.content_sha256,"camera_intrinsics_content_sha256":camera_hashes[image_id],"position_count":len(positions),"orientation_count":len(rotations),"total_factor_pair_count":int(len(positions)*len(rotations)),"maximum_query_parents":MAXIMUM_QUERY_PARENTS,"maximum_scene_children":MAXIMUM_SCENE_CHILDREN,"returned_topk":TOPK,"position_chunk_size":POSITION_CHUNK_SIZE,"torch_dtype":TORCH_DTYPE,"device":args.device,"parent_score_semantics":SCORE_SEMANTICS,"child_score_semantics":CHILD_SCORE_SEMANTICS,"child_geometry_contract":CHILD_GEOMETRY_CONTRACT,"fusion_semantics":FUSION_SEMANTICS,"streaming_semantics":MULTI_STREAMING_SEMANTICS,"normal_contract":NORMAL_CONTRACT,"token_footprint_phase":TOKEN_FOOTPRINT_PHASE,"complete_parent_probability_mass":result.complete_parent_probability_mass,"complete_child_probability_mass":result.complete_child_probability_mass,"selected_parent_probability_mass_total":result.selected_parent_probability_mass_total,"selected_child_probability_mass_total":result.selected_child_probability_mass_total,"selected_child_probability_mass_fraction":result.selected_child_probability_mass_total/result.complete_child_probability_mass,"elapsed_seconds":elapsed,"scoring_seconds":result.scoring_seconds,"stable_merge_seconds":result.merge_seconds,"peak_cuda_allocated_bytes":result.peak_cuda_allocated_bytes,"peak_cuda_reserved_bytes":result.peak_cuda_reserved_bytes,"control_only":True,"production_eligible":False,"promotion_blockers":summary["promotion_blockers"],"uses_query_pose":False,"uses_query_ground_truth":False,"uses_contributor_artifact":False,"uses_alike":False,"uses_pnp":False,"uses_optimizer":False,"uses_renderer":False,"cartesian_score_array_materialized":False,"phase2_labels_opened":False}
            _save(output,arrays,metadata); load_hierarchy_score(output)
        rows.append({"query_index":query_index,"image_id":image_id,"artifact":str(output),"artifact_file_sha256":file_sha256(output),"content_sha256":content,"elapsed_seconds":elapsed})
        print(json.dumps({"query_index":query_index,"image_id":image_id,"elapsed_seconds":elapsed},sort_keys=True),flush=True)
    report={"artifact_type":SHARD_RUN_SCHEMA,"query_route":"seq10","shard_count":args.shard_count,"shard_index":args.shard_index,"query_indices":query_indices.tolist(),"query_count":len(rows),"rows":rows,**bindings,"proposal":str(proposal_path),"retrieval_summary":str(summary_path),"physical_map":str(physical_path),"camera_manifest":str(camera_path),"gpu_gate":str(gate_path),"position_count":len(positions),"orientation_count":len(rotations),"total_factor_pair_count_per_query":int(len(positions)*len(rotations)),"maximum_query_parents":MAXIMUM_QUERY_PARENTS,"maximum_scene_children":MAXIMUM_SCENE_CHILDREN,"returned_topk_per_query":TOPK,"position_chunk_size":POSITION_CHUNK_SIZE,"torch_dtype":TORCH_DTYPE,"device":args.device,"total_elapsed_seconds":time.perf_counter()-started,"control_only":True,"production_eligible":False,"phase2_labels_opened":False,"uses_query_pose":False,"uses_query_ground_truth":False,"uses_contributor_artifact":False}
    report["content_sha256"]=canonical_json_sha256(report); run_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(run_path),"query_count":len(rows),"elapsed":report["total_elapsed_seconds"]},indent=2))


if __name__ == "__main__": main()
