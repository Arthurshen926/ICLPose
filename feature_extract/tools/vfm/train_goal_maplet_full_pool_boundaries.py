"""Mapping-only full eligible boundary pool, common random seeds, final backend utility.

All hypotheses are frozen before GT labels are read. This matches the optional
shared-regional-seed 224-query backend, not the historical two-branch consensus.
"""
import argparse, hashlib, json, multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,score_pose
from feature_extract.tools.vfm.refine_goal_maplet_global_token_lm import refine
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import _refine_pose_scale,_many_to_one_plane_associations
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics,_region_token_support
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

G={}
def process(task):
 si,s=task;c=G['c'];a=G['args'];name=G['names'][s];seed=a.seeds[si]
 rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];pr=c['prototype_rows'][rows];w=c['prototype_world'][pr];score=G['scores'][rows];planes=c['prototype_plane'][pr]
 pix=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];groups=[np.flatnonzero(tok==t) for t in np.unique(tok)]
 with np.load(a.contributors/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
 qp,meta=QueryPlaneRegions.load_npz(a.query_regions/name)
 if meta.get('uses_pose_or_ground_truth') is not False:raise ValueError('pose-dependent query geometry')
 tr=np.full(2304,-1,int)
 for r in range(len(qp.normals_camera)):
  ts,frac=_region_token_support(qp.labels,r);tr[ts[frac>=.75]]=r
 def finish(pose):
  if not np.isfinite(pose).all():return pose
  pose=refine(pose,w,pix,tok,K,k1)[0];key,sel=score_pose(pose,w,pix,groups,K,k1);sel=sel[tr[tok[sel]]>=0]
  prov=np.c_[tr[tok[sel]],planes[sel],pr[sel]];ass=_many_to_one_plane_associations(prov,np.arange(len(sel)));pm=G['map']
  proposal,_,accepted,_=_refine_pose_scale(pose,w[sel],pix[sel],prov,np.arange(len(sel)),ass,pm.normals_world,pm.offsets_world,qp.normals_camera,qp.offsets_camera,K,k1)
  return proposal if accepted and score_pose(proposal,w,pix,groups,K,k1)[0]>=key else pose
 def estimate(member,local_seed):
  st={};p=solve(w,tok,K,k1,member,iterations=2048,seed=local_seed,sampling_policy='geometry_score',planes=planes,scores=score,hypothesis_budget=256,stats=st)
  p=np.full((4,4),np.nan) if p is None else p
  key=(-1,-np.inf) if not np.isfinite(p).all() else score_pose(p,w,pix,groups,K,k1)[0]
  return p,finish(p),key,st
 n=len(G['centers']);raw=np.full((n,3,4,4),np.nan);final=raw.copy();support=np.zeros((n,3));counts=np.full((n,3),-1.);residual=np.full((n,3),-np.inf);eligible=np.zeros((n,3),bool);stats=[];tree=cKDTree(w)
 global_raw,global_final,global_key,st=estimate(np.arange(len(tok)),seed);stats.append(st)
 for k,r in enumerate([3.,6.,9.]):
  for j,members in enumerate(tree.query_ball_point(G['centers'],r)):
   members=np.asarray(members,int);ut,inv=np.unique(tok[members],return_inverse=True)
   if len(ut)<6:continue
   eligible[j,k]=True;v=np.zeros(len(ut));np.maximum.at(v,inv,expit(score[members]));support[j,k]=v.sum()
   p,f,key,st=estimate(members,seed+1);raw[j,k]=p;final[j,k]=f;counts[j,k],residual[j,k]=key;stats.append(st)
 np.savez_compressed(a.output/f'cache_{si}_{s}.npz',raw=raw,final=final,support=support,counts=counts,residual=residual,eligible=eligible,global_raw=global_raw,global_final=global_final,global_key=global_key,source_image=s,seed=seed)
 return si,int(s),int(eligible.sum()),sum(x.get('scored_hypotheses',0) for x in stats),sum(not x.get('budget_reached',True) for x in stats)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['candidates','scores','features','region_map','contributors','query_regions','planar_map','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--seeds',type=int,nargs='+',default=[260901,260902]);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
 sha=file_sha256(a.candidates)
 with np.load(a.scores) as z:
  assert str(z['candidate_sha256'])==sha;scores=z['logits'][:,z['arm_names'].tolist().index('radius2')]
 with np.load(a.features) as z:
  assert str(z['candidate_sha256'])==sha;names=z['source_names'].astype(str)
 with np.load(a.region_map) as z:centers=z['centers'];assert str(z['world_sha256'])==hashlib.sha256(c['prototype_world'].tobytes()).hexdigest()
 sources=np.unique(c['source_image']);assert all(names[s].startswith('seq9__') for s in sources)
 G.update(args=a,c=c,scores=scores,names=names,centers=centers,map=GeometryNativePlanarMap.load_npz(a.planar_map))
 tasks=[(i,int(s)) for i in range(len(a.seeds)) for s in sources];audit=[]
 with mp.get_context('fork').Pool(a.workers) as pool:
  for row in pool.imap_unordered(process,tasks):audit.append(row);print('full pool',len(audit),'/',len(tasks),row,flush=True)
 # Every worker has sealed hypotheses and final refinement before GT is accessed.
 arrays=[];quality=[];global_quality=[]
 for si,s in tasks:
  with np.load(a.output/f'cache_{si}_{s}.npz') as z:d={k:z[k] for k in z.files}
  with np.load(a.contributors/names[s]) as z:gt=z['pose_w2c']
  err=np.asarray([_pose_error(t,gt) for t in d['final'].reshape(-1,4,4)]).reshape(len(centers),3,2)
  quality.append(np.exp(-err[...,0]/.5-err[...,1]/5));e=_pose_error(d['global_final'],gt);global_quality.append(np.exp(-e[0]/.5-e[1]/5));arrays.append(d)
 np.savez_compressed(a.output/'training.npz',quality=quality,global_quality=global_quality,support=np.asarray([d['support'] for d in arrays]),counts=np.asarray([d['counts'] for d in arrays]),residual=np.asarray([d['residual'] for d in arrays]),global_counts=[d['global_key'][0] for d in arrays],global_residual=[d['global_key'][1] for d in arrays],eligible=np.asarray([d['eligible'] for d in arrays]),centers=centers,training_images=names[sources],sample_names=np.asarray([names[s] for _,s in tasks]),seeds=a.seeds)
 (a.output/'audit.json').write_text(json.dumps({'candidate_sha256':sha,'poses_frozen_before_GT':True,'seed_policy':'global seed; all regional groups seed+1','scope':'full eligible mapping pool with global LM and MoGe v2; single mapping route, 224-query backend not 438 consensus','worker_results':audit},indent=2))
if __name__=='__main__':main()
