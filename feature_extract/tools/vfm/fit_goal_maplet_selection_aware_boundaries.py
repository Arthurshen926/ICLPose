"""Mapping-only boundary optimization through a frozen four-region selector."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,score_pose
from feature_extract.tools.vfm.refine_goal_maplet_global_token_lm import refine
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error


def selected_quality(choice,quality,support,counts,residual,global_counts,global_residual,global_quality):
 cols=np.arange(len(choice));a=support[:,cols,choice];order=np.argsort(-a,axis=1,kind='stable')[:,:4];rr=np.arange(len(a))[:,None]
 k=np.c_[global_counts,counts[:,cols,choice][rr,order]];e=np.c_[global_residual,residual[:,cols,choice][rr,order]];u=np.c_[global_quality,quality[:,cols,choice][rr,order]]
 winner=np.argmax(np.where(k==k.max(1,keepdims=True),e,-np.inf),axis=1)
 return float(u[np.arange(len(a)),winner].mean())


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base;root=b/'learned_boundaries_v241';dest=root/'selection_aware';dest.mkdir(exist_ok=False)
 with np.load(b/'moge_full_input_v218.npz') as z:c={k:z[k] for k in z.files}
 with np.load(b/'adaptive_memory_v234/scale_models.npz') as z:priority=z['logits'][:,z['arm_names'].tolist().index('radius2')]
 with np.load(b/'adaptive_memory_v234/multiscale/moge_full_input_v218.npz') as z:names=z['source_names'].astype(str)
 with np.load(b/'learned_region_alignment_v240/budget256/frozen_initializers.npz') as z:sources=z['source_image'];ids=z['region_ids'];centers=z['centers'];poses6=z['poses']
 raw=[]
 for radius in [3,6,9]:
  if radius==6:raw.append(poses6);continue
  with np.load(root/f'r{radius}/frozen.npz') as z:
   assert np.array_equal(sources,z['source_image']) and np.array_equal(ids,z['region_ids']);raw.append(z['raw_poses'])
 unique=np.unique(sources);support=np.zeros((len(unique),len(centers),3));counts=np.full_like(support,-1.);residual=np.full_like(support,-np.inf);global_counts=[];global_residual=[];global_poses=[]
 contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')
 for i,s in enumerate(unique):
  rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];pr=c['prototype_rows'][rows];world=c['prototype_world'][pr];score=priority[rows];pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];groups=[np.flatnonzero(tok==t) for t in np.unique(tok)]
  with np.load(contributors/names[s]) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
  pose=solve(world,tok,K,k1,np.arange(len(tok)),iterations=2048,seed=260901,sampling_policy='geometry_score',planes=c['prototype_plane'][pr],scores=score,hypothesis_budget=256)
  key=(-1,-np.inf) if pose is None else score_pose(pose,world,pixels,groups,K,k1)[0];global_counts.append(key[0]);global_residual.append(key[1]);global_poses.append(np.full((4,4),np.nan) if pose is None else refine(pose,world,pixels,tok,K,k1)[0])
  for row in np.flatnonzero(sources==s):
   rid=ids[row];distance=np.linalg.norm(world-centers[rid],axis=1)
   for k,radius in enumerate([3,6,9]):
    members=np.flatnonzero(distance<=radius);ut=np.unique(tok[members])
    if len(ut)<6:continue
    support[i,rid,k]=sum(expit(score[members[tok[members]==t]]).max() for t in ut)
    if np.isfinite(raw[k][row]).all():counts[i,rid,k],residual[i,rid,k]=score_pose(raw[k][row],world,pixels,groups,K,k1)[0]
 np.savez_compressed(dest/'frozen_selector.npz',support=support,counts=counts,residual=residual,global_counts=global_counts,global_residual=global_residual,global_poses=global_poses)
 # All pose candidates and selection evidence seal before mapping quality is read.
 gq=[]
 for s,pose in zip(unique,global_poses):
  with np.load(contributors/names[s]) as z:te,re=_pose_error(pose,z['pose_w2c'])
  gq.append(np.exp(-te/.5-re/5))
 with np.load(root/'maps/training_proxy.npz') as z:q=z['quality'];observed=z['observed'];cost=z['cost']
 choice=np.ones(len(centers),int);budget=int(cost[:,1].sum());active=np.flatnonzero(observed.any(0));trace=[]
 def value(ch):return selected_quality(ch,q,support,counts,residual,global_counts,global_residual,gq)
 initial=value(choice)
 for iteration in range(30):
  current=value(choice);used=int(cost[np.arange(len(choice)),choice].sum());actions=[(int(i),k,int(cost[i,k]-cost[i,choice[i]])) for i in active for k in range(3) if k!=choice[i]];best=current+1e-12;winner=None
  proposals=[(x,) for x in actions if used+x[2]<=budget]
  proposals.extend((x,y) for x in actions if x[2]>0 for y in actions if y[2]<0 and x[0]!=y[0] and used+x[2]+y[2]<=budget)
  for acts in proposals:
   proposal=choice.copy()
   for i,k,_ in acts:proposal[i]=k
   score=value(proposal)
   if score>best:best,winner=score,proposal
  if winner is None:break
  choice=winner;trace.append({'value':best,'references':int(cost[np.arange(len(choice)),choice].sum())});print('selection-aware iteration',iteration,best,flush=True)
 radii=np.array([3.,6.,9.])[choice]
 from scipy.spatial import cKDTree
 members=cKDTree(c['prototype_world']).query_ball_point(centers,radii);model=json.loads((b/'learned_region_alignment_v240/budget256_global/model.json').read_text())
 np.savez_compressed(root/'maps/selection_aware.npz',centers=centers,radii=radii,prototype_rows=np.concatenate(members),offsets=np.r_[0,np.cumsum([len(x) for x in members])],world_sha256=np.asarray(hashlib.sha256(c['prototype_world'].tobytes()).hexdigest()),training_images=np.asarray(model['training_images']))
 (dest/'report.json').write_text(json.dumps({'initial_value':initial,'final_value':value(choice),'trace':trace,'radius_histogram':{str(r):int(np.sum(radii==r)) for r in [3,6,9]},'references':sum(map(len,members)),'budget':budget,'scope':'frozen sampled region universe, top4 support then geometry selection including global initializer; GT only for selected mapping utility','limitations':['sampled region universe rather than all eligible regions','mapping seed uses region identity; deployment uses active group order','does not include MoGe refinement in training utility']},indent=2))

if __name__=='__main__':main()
