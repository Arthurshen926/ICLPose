"""Score one pose-free seq10 shard, ranking positions with all SO(3) factors retained."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    load_bound_retrieval, load_pose_free_camera_manifest, load_seq10_control_summary,
)
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import load_all_parent_union_support
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.streaming_child_layout_guide_gpu import (
    MODES, stream_hierarchy_position_topk_gpu,
)


SCHEMA = "goal_maplet_global_hierarchy_position_streaming_phase1_shard_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True); parser.add_argument("--retrieval_summary", required=True)
    parser.add_argument("--physical_map", required=True); parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--device", required=True); parser.add_argument("--shard_count", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True); parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    output = Path(args.output_npz).resolve()
    if output.exists(): raise FileExistsError("refusing to overwrite hierarchy position shard")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count: raise ValueError("invalid shard")
    proposal_path=Path(args.proposal).resolve(); summary_path=Path(args.retrieval_summary).resolve()
    physical_path=Path(args.physical_map).resolve(); camera_path=Path(args.camera_manifest).resolve()
    proposal,proposal_meta=load_all_parent_union_support(proposal_path); summary=load_seq10_control_summary(summary_path)
    physical=GoalMapletPhysicalMap.load_npz(physical_path); physical_file=file_sha256(physical_path)
    cameras,camera_hashes=load_pose_free_camera_manifest(camera_path)
    positions=np.asarray(proposal["lattice_origin_world"],np.float64)+(np.asarray(proposal["cell_indices_world"],np.float64)+.5)*float(proposal["lattice_spacing_m"])
    rotations=np.asarray(proposal["orientation_rotations_w2c"],np.float64)
    indices=np.array_split(np.arange(len(summary["rows"]),dtype=np.int64),args.shard_count)[args.shard_index]
    scores=[]; position_rows=[]; best_orientations=[]; image_ids=[]; retrieval_files=[]; retrieval_contents=[]; elapsed=[]
    started=time.perf_counter()
    for query_index in indices.tolist():
        row=summary["rows"][query_index]
        retrieval,path=load_bound_retrieval(row,physical_file_sha256=physical_file,physical_content_sha256=physical.content_sha256)
        result=stream_hierarchy_position_topk_gpu(positions,rotations,retrieval,physical,cameras[retrieval.image_id],device=args.device)
        scores.append(np.stack([result.rankings[m].top_scores for m in MODES]))
        position_rows.append(np.stack([result.rankings[m].top_position_rows for m in MODES]))
        best_orientations.append(np.stack([result.rankings[m].diagnostic_best_orientation_rows for m in MODES]))
        image_ids.append(retrieval.image_id); retrieval_files.append(file_sha256(path)); retrieval_contents.append(retrieval.content_sha256); elapsed.append(result.elapsed_seconds)
        print(json.dumps({"query_index":query_index,"image_id":retrieval.image_id,"elapsed":result.elapsed_seconds}),flush=True)
    arrays={
        "query_indices":indices,"image_ids":np.asarray(image_ids),
        "top_scores":np.stack(scores).astype(np.float64),
        "top_position_rows":np.stack(position_rows).astype(np.int64),
        "diagnostic_best_orientation_rows":np.stack(best_orientations).astype(np.int64),
        "retrieval_file_sha256":np.asarray(retrieval_files),
        "retrieval_content_sha256":np.asarray(retrieval_contents),
        "elapsed_seconds":np.asarray(elapsed,np.float64),
    }
    metadata={
        "artifact_type":SCHEMA,"query_route":"seq10","shard_count":args.shard_count,"shard_index":args.shard_index,
        "content_sha256":arrays_sha256(arrays),"proposal_file_sha256":file_sha256(proposal_path),
        "proposal_content_sha256":proposal_meta["content_sha256"],"retrieval_summary_file_sha256":file_sha256(summary_path),
        "physical_map_file_sha256":physical_file,"physical_map_content_sha256":physical.content_sha256,
        "camera_manifest_file_sha256":file_sha256(camera_path),"camera_intrinsics_hashes":[camera_hashes[x] for x in image_ids],
        "modes":list(MODES),"position_count":len(positions),"implicit_orientation_count":len(rotations),
        "topk_positions":4096,"support_semantics":"selected_positions_cartesian_all_analytic60_orientations_v1",
        "phase2_labels_opened":False,"uses_query_pose":False,"uses_query_ground_truth":False,
        "control_only":True,"production_eligible":False,"total_elapsed_seconds":time.perf_counter()-started,
    }
    metadata["metadata_content_sha256"]=canonical_json_sha256(metadata)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+".",suffix=".tmp",delete=False) as f: temporary=Path(f.name)
    try:
        with temporary.open("wb") as f: np.savez_compressed(f,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
        os.replace(temporary,output)
    finally: temporary.unlink(missing_ok=True)
    print(json.dumps({"output":str(output),"query_count":len(indices),"elapsed":metadata["total_elapsed_seconds"]},indent=2))


if __name__ == "__main__": main()
