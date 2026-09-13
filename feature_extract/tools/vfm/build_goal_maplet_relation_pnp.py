"""Additional PnP branch with matched-budget MoGe quartet proposal controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.moge_relation_proposals import RelationSampler
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--correspondences',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--policy',choices=['uniform_control','relation','shuffled'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=260901);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 c,cm=_load(a.correspondences);names=c['names'];poses=[];inliers=[];records=[];hashes={};start=time.perf_counter()
 for i,n in enumerate(names.astype(str)):
  path=a.moge_dir/n;qp,qn,qv=moge_tokens(path);hashes[str(path)]=file_sha256(path);sampler=RelationSampler(qp,qv,a.policy);lo,hi=c['correspondence_offsets'][i:i+2];w=c['world_points'][lo:hi];t=c['query_tokens'][lo:hi];xy=c['query_measurements_xy'][lo:hi];stats={}
  pose=solve(w,t,c['camera_matrices'][i],float(c['radial_k1'][i]),np.arange(len(w)),pixels=xy,iterations=5000,seed=a.seed,hypothesis_budget=1000,stats=stats,proposal_sampler=sampler)
  poses.append(np.full((4,4),np.nan) if pose is None else pose)
  if pose is None:count=0
  else:
   cw,ct,cxy=canonical_hypotheses(w,t,xy);groups=[np.flatnonzero(ct==u) for u in np.unique(ct)];count=score_pose(pose,cw,cxy,groups,c['camera_matrices'][i],float(c['radial_k1'][i]),return_selected=False)[0][0]
  inliers.append(count);records.append(dict(name=n,**stats,**sampler.summary()))
  if (i+1)%20==0:print(a.policy,i+1,flush=True)
 arr=dict(names=names,pose_w2c=np.asarray(poses),usable=np.isfinite(poses).all((1,2)),candidate_correspondence_count=np.diff(c['correspondence_offsets']),pnp_inlier_count=np.array(inliers))
 meta=dict(artifact_type='goal_maplet_relation_group_pnp_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,query_depth_or_scale_used_by_pose_solver=True,policy=a.policy,seed=a.seed,shuffle_seed=314,frozen_correspondence_file_sha256=file_sha256(a.correspondences),frozen_correspondence_content_sha256=cm['content_sha256'],moge_files_sha256=hashes,scored_hypothesis_budget=1000,maximum_attempts=5000,quartet_options_per_attempt=8,uniform_exploration_probability=.5,scope='additional global branch; primary preserved separately; not a full replacement of group solver; equal scored budgets not equal RGB runtime')
 meta['content_sha256']=canonical_json_sha256(meta);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)));a.output.with_suffix('.json').write_text(json.dumps(dict(query_labels_used=False,records=records,seconds=time.perf_counter()-start,pose_file_sha256=file_sha256(a.output)),indent=2))
if __name__=='__main__':main()
