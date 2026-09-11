"""Mapping-only physical boundary candidates, frozen before pose supervision."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.token_hypothesis_ransac import solve
from feature_extract.tools.vfm.refine_goal_maplet_global_token_lm import refine
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['candidates','scores','features','reference','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--radius',type=float,required=True);a=p.parse_args()
 if a.radius<=0:raise ValueError('positive radius required')
 a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
 with np.load(a.scores) as z:
  priority=z['logits'][:,z['arm_names'].tolist().index('radius2')]
  if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('score lineage')
 with np.load(a.features) as z:
  names=z['source_names'].astype(str)
  if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('feature lineage')
 with np.load(a.reference) as z:sources=z['source_image'];region_ids=z['region_ids'];centers=z['centers']
 if not all(names[s].startswith('seq9__') for s in sources):raise ValueError('mapping-only reference required')
 raw=[];refined=[];stats=[]
 for count,s in enumerate(np.unique(sources)):
  rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];pr=c['prototype_rows'][rows];world=c['prototype_world'][pr];score=priority[rows];planes=c['prototype_plane'][pr]
  pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];tree=cKDTree(world)
  with np.load(a.contributors/names[s]) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
  for rid in region_ids[sources==s]:
   group=np.asarray(tree.query_ball_point(centers[rid],a.radius),int);st={}
   pose=solve(world,tok,K,k1,group,iterations=2048,seed=260911+int(rid),sampling_policy='geometry_score',planes=planes,scores=score,hypothesis_budget=256,stats=st)
   before=np.full((4,4),np.nan) if pose is None else pose;after=before if pose is None else refine(pose,world,pixels,tok,K,k1)[0]
   raw.append(before);refined.append(after);stats.append(st)
  if (count+1)%10==0:print('mapping boundaries',a.radius,count+1,flush=True)
 # Input reference order must be source-major, as produced by the original trainer.
 if np.any(np.diff(sources)<0):raise ValueError('reference must be source-major')
 np.savez_compressed(a.output/'frozen.npz',raw_poses=np.asarray(raw),poses=np.asarray(refined),source_image=sources,region_ids=region_ids,centers=centers,radius=a.radius)
 errors=[];gt={}
 for s,pose in zip(sources,refined):
  if s not in gt:
   with np.load(a.contributors/names[s]) as z:gt[s]=z['pose_w2c']
  errors.append(_pose_error(pose,gt[s]))
 e=np.asarray(errors);utility=np.exp(-e[:,0]/.5-e[:,1]/5)
 np.savez_compressed(a.output/'labels.npz',utility=utility,errors=e)
 (a.output/'audit.json').write_text(json.dumps({'candidate_sha256':file_sha256(a.candidates),'reference_sha256':file_sha256(a.reference),'poses_frozen_before_GT':True,'training_images':names[np.unique(sources)].tolist(),'stats':stats},indent=2))

if __name__=='__main__':main()
