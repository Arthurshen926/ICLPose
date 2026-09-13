"""Export an existing label-free PnP selection without refining its pose."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_stage
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256
ARTIFACT_TYPE='goal_maplet_retained_pnp_endpoint_v1'

def diverse_indices(poses, reference):
 # Inference-time pose separation, not localization error against truth.
 from scipy.spatial.transform import Rotation
 poses=np.asarray(poses,float);reference=np.asarray(reference,float)
 if not np.isfinite(reference).all():return np.flatnonzero(np.isfinite(poses).all((1,2)))
 ids=[];center=-reference[:3,:3].T@reference[:3,3]
 for j,p in enumerate(poses):
  if not np.isfinite(p).all():continue
  distance=np.linalg.norm(-p[:3,:3].T@p[:3,3]-center)
  angle=np.degrees(Rotation.from_matrix(p[:3,:3]@reference[:3,:3].T).magnitude())
  if distance>1.0 or angle>10.0:ids.append(j)
 return np.asarray(ids,np.int64)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--candidates',type=Path,required=True);p.add_argument('--correspondences',type=Path,required=True)
 p.add_argument('--rule',choices=['raw_inliers','balanced_support','supported_entities','diverse_support'],required=True)
 p.add_argument('--reference_pose',type=Path)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 x,m=_pose_stage(a.candidates);c,cm=_load(a.correspondences)
 if m.get('query_pose_or_ground_truth_opened') is not False or canonical_json_sha256({k:v for k,v in m.items() if k!='content_sha256'})!=m.get('content_sha256'):raise ValueError('PnP metadata contract differs')
 if not np.array_equal(x['names'],c['names']) or m['frozen_correspondence_file_sha256']!=file_sha256(a.correspondences):raise ValueError('PnP correspondence lineage differs')
 rule='raw_inliers' if a.rule=='diverse_support' else a.rule
 pose=x[rule+'_pose_w2c'].copy();inliers=x[rule+'_inlier_count'].copy();chosen=np.full(len(pose),-1,np.int64)
 if a.rule=='diverse_support':
  if a.reference_pose is None:raise ValueError('diverse support needs frozen reference')
  from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
  from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
  ref,_=_load_pose_candidate(a.reference_pose)
  if not np.array_equal(ref['names'],x['names']):raise ValueError('reference query order differs')
  for i in range(len(pose)):
   lo,hi=c['correspondence_offsets'][i:i+2];w,t,xy=canonical_hypotheses(c['world_points'][lo:hi],c['query_tokens'][lo:hi],c['query_measurements_xy'][lo:hi]);groups=[np.flatnonzero(t==u) for u in np.unique(t)]
   pl,ph=x['candidate_offsets'][i:i+2];candidates=x['candidate_pose_w2c'][pl:ph];eligible=diverse_indices(candidates,ref['pose_w2c'][i])
   scores=[score_pose(candidates[j],w,xy,groups,c['camera_matrices'][i],float(c['radial_k1'][i]),return_selected=False)[0] for j in eligible]
   if scores:
    j=int(eligible[max(range(len(scores)),key=lambda v:scores[v])]);pose[i]=candidates[j];inliers[i]=scores[max(range(len(scores)),key=lambda v:scores[v])][0];chosen[i]=pl+j
 elif a.reference_pose is not None:raise ValueError('unused reference pose')
 arr=dict(names=x['names'],pose_w2c=pose,usable=np.isfinite(pose).all((1,2)),candidate_correspondence_count=np.diff(c['correspondence_offsets']),pnp_inlier_count=inliers,source_candidate_index=chosen)
 meta=dict(artifact_type=ARTIFACT_TYPE,arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,query_depth_or_scale_used_by_pose_solver=False,selection_rule=a.rule,source_candidate_file_sha256=file_sha256(a.candidates),frozen_correspondence_file_sha256=file_sha256(a.correspondences),frozen_correspondence_content_sha256=cm['content_sha256'],scope='exact frozen PnP candidate, no refinement or label-based choice',reference_pose_file_sha256=file_sha256(a.reference_pose) if a.reference_pose else None,diversity_translation_m=1.0 if a.reference_pose else None,diversity_rotation_deg=10.0 if a.reference_pose else None)
 meta['content_sha256']=canonical_json_sha256(meta);a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
if __name__=='__main__':main()
