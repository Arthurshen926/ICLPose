"""Evaluate frozen position rankings with the complete analytic60 factor retained."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_global_parent_layout_streaming_seq10 import (
    _load_direct_labels_after_freeze, _rotation_error_deg,
)
from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import EXPECTED_SEQ10_QUERY_COUNT
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import load_all_parent_union_support
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.streaming_child_layout_guide_gpu import MODES
from feature_extract.tools.vfm.run_goal_maplet_global_child_layout_position_phase1 import SCHEMA as SHARD_SCHEMA


SCHEMA="goal_maplet_global_hierarchy_position_streaming_seq10_retention_v1"
K_VALUES=(64,256,1024,4096)
ARRAYS=("query_indices","image_ids","top_scores","top_position_rows","diagnostic_best_orientation_rows","retrieval_file_sha256","retrieval_content_sha256","elapsed_seconds")


def _load(path: Path):
    expected={*(f"{x}.npy" for x in ARRAYS),"metadata_json.npy"}
    with zipfile.ZipFile(path,"r") as z:
        members=[x.filename for x in z.infolist()]
    if len(members)!=len(set(members)) or set(members)!=expected: raise ValueError("position shard members differ")
    with np.load(path,allow_pickle=False) as z:
        arrays={x:np.asarray(z[x]) for x in ARRAYS}; metadata=json.loads(str(np.asarray(z["metadata_json"]).item()))
    unhashed=dict(metadata); declared=unhashed.pop("metadata_content_sha256","")
    n=len(arrays["query_indices"])
    if (metadata.get("artifact_type")!=SHARD_SCHEMA or declared!=canonical_json_sha256(unhashed)
        or metadata.get("content_sha256")!=arrays_sha256(arrays) or metadata.get("phase2_labels_opened") is not False
        or metadata.get("uses_query_pose") is not False or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("modes")!=list(MODES) or metadata.get("topk_positions")!=4096
        or arrays["top_scores"].shape!=(n,3,4096) or arrays["top_scores"].dtype!=np.float64
        or arrays["top_position_rows"].shape!=(n,3,4096) or arrays["top_position_rows"].dtype!=np.int64
        or np.any(~np.isfinite(arrays["top_scores"]))): raise ValueError("position shard contract differs")
    for q in range(n):
        for m in range(3):
            score=arrays["top_scores"][q,m]; rows=arrays["top_position_rows"][q,m]
            if len(np.unique(rows))!=4096 or not np.array_equal(np.lexsort((rows,-score)),np.arange(4096)):
                raise ValueError("position shard ranking differs")
    return arrays,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--shard",action="append",required=True)
    p.add_argument("--proposal",required=True); p.add_argument("--direct_dataset",required=True); p.add_argument("--output_json",required=True)
    args=p.parse_args(); output=Path(args.output_json).resolve()
    if output.exists(): raise FileExistsError("refusing to overwrite position evaluation")
    shard_paths=[Path(x).resolve() for x in args.shard]; loaded=[_load(x) for x in shard_paths]
    reference=loaded[0][1]
    if sorted(x[1]["shard_index"] for x in loaded)!=list(range(len(loaded))) or len(loaded)!=reference["shard_count"]: raise ValueError("position shard set differs")
    for _,meta in loaded[1:]:
        for key in ("proposal_file_sha256","proposal_content_sha256","physical_map_file_sha256","physical_map_content_sha256","camera_manifest_file_sha256"):
            if meta[key]!=reference[key]: raise ValueError("position shard lineage differs")
    arrays={key:np.concatenate([x[0][key] for x in loaded],axis=0) for key in ARRAYS}
    order=np.argsort(arrays["query_indices"]); arrays={k:v[order] for k,v in arrays.items()}
    if not np.array_equal(arrays["query_indices"],np.arange(EXPECTED_SEQ10_QUERY_COUNT)): raise ValueError("position query inventory differs")
    proposal_path=Path(args.proposal).resolve(); proposal,pmeta=load_all_parent_union_support(proposal_path)
    if file_sha256(proposal_path)!=reference["proposal_file_sha256"] or pmeta["content_sha256"]!=reference["proposal_content_sha256"]: raise ValueError("position proposal differs")
    direct_path=Path(args.direct_dataset).resolve()
    direct,dmeta=_load_direct_labels_after_freeze(direct_path,arrays["image_ids"].astype(str).tolist(),reference["physical_map_file_sha256"],reference["physical_map_content_sha256"])
    target=np.asarray(direct["candidate_poses_w2c"],float)[:,0]; centers=np.stack([camera_center_from_pose_w2c(x) for x in target])
    positions=np.asarray(proposal["lattice_origin_world"],float)+(np.asarray(proposal["cell_indices_world"],float)+.5)*float(proposal["lattice_spacing_m"])
    rotations=np.asarray(proposal["orientation_rotations_w2c"],float)
    orientation_hit=np.asarray([np.min(_rotation_error_deg(rotations,x[:3,:3]))<=45 for x in target])
    hits={}
    for mi,mode in enumerate(MODES):
        distance=np.linalg.norm(positions[arrays["top_position_rows"][:,mi]]-centers[:,None],axis=2)
        hits[mode]={str(k):int(np.sum(np.any(distance[:,:k]<=2,axis=1)&orientation_hit)) for k in K_VALUES}
    required=int(math.ceil(.95*EXPECTED_SEQ10_QUERY_COUNT-1e-12))
    selected=next(({"mode":m,"k":k} for k in K_VALUES for m in ("child","geometric_mean") if hits[m][str(k)]>=required),None)
    report={"artifact_type":SCHEMA,"query_route":"seq10","query_count":EXPECTED_SEQ10_QUERY_COUNT,
        "shards":[{"path":str(x),"file_sha256":file_sha256(x),"content_sha256":meta["content_sha256"]} for x,(_,meta) in zip(shard_paths,loaded)],
        "proposal":str(proposal_path),"proposal_file_sha256":file_sha256(proposal_path),"proposal_content_sha256":pmeta["content_sha256"],
        "direct_dataset":str(direct_path),"direct_dataset_file_sha256":file_sha256(direct_path),"direct_dataset_content_sha256":dmeta["content_sha256"],
        "position_budget_preregistered":list(K_VALUES),"implicit_orientation_count":len(rotations),"orientation_45deg_hit_count":int(np.sum(orientation_hit)),
        "selection_required_hits":required,"hits_2m45":hits,"selected":selected,"decision":"GO" if selected else "KILL",
        "phase_separation_audit":{"both_shards_and_all_rank_arrays_frozen_before_direct_dataset_open":True},
        "held_route_labels_opened":False,"held_evaluation_authorized":selected is not None,"control_only":True,"production_eligible":False,
        "uses_alike":False,"uses_pnp":False,"uses_optimizer":False,"uses_renderer":False}
    report["content_sha256"]=canonical_json_sha256(report); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"decision":report["decision"],"hits":hits},indent=2,sort_keys=True))


if __name__=="__main__": main()
