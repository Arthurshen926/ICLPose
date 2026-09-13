"""Choose pose-update strength using fixed spatial cross-validation folds.

Compare alpha 0, 0.5 and 1 on held-out features, then apply the chosen fraction
between full-data regularized/free endpoints. Shared backbone receptive fields
preclude statistical independence. No pose labels enter the choice.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.local_pose_consensus import interpolate_pose
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine,sample_with_gradient
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def read(path):
 with np.load(path) as z:a={k:z[k] for k in z.files if k!='metadata_json'};m=json.loads(z['metadata_json'].item())
 if arrays_sha256(a)!=m['arrays_sha256'] or canonical_json_sha256({k:v for k,v in m.items() if k!='content_sha256'})!=m['content_sha256'] or m.get('query_pose_or_ground_truth_read') is not False:raise ValueError('frozen feature pose differs')
 return a,m


def feature_loss(pose,world,K,k1,grid,target,robust):
 pix=cv2.projectPoints(world,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2);f,_=sample_with_gradient(grid,pix);t=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12);sq=np.sum((f-t)**2,axis=1)
 return float(np.mean(.25*(np.sqrt(1+sq/.25)-1) if robust else .5*sq))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);a=p.parse_args();b=a.base;s=a.split;o=a.output;o.mkdir(parents=True,exist_ok=True);arms={'plain':('direct_feature_extended_v332','fine_direct512',False),'robust':('direct_feature_robust_free_v334','fine_robust_direct512',True)}
 if any((o/f'{s}_{arm}.npz').exists() for arm in arms):raise FileExistsError('output exists')
 initial=b/'sequential_depth_rescue_v327'/f'{s}.npz';base,_=_load_pose_candidate(initial);mp=b/'native_fine_v264/readout/map.npz'
 with np.load(mp) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(z['metadata_json'].item())
 if arrays_sha256(maps)!=mm['arrays_sha256']:raise ValueError('map differs')
 fallback_path=b/'direct_feature_extended_v332'/f'{s}_fine_robust512.npz';fallback,_=read(fallback_path);corrpath=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';corr,_=_load(corrpath)
 rows_path=b/'direct_feature_precision_v330'/f'{s}_fine_joint512_audit.json';rows=json.load(open(rows_path));sources={str(f):file_sha256(f) for f in [initial,mp,fallback_path,corrpath,rows_path,Path(__file__),Path(__file__).with_name('direct_anonymous_feature_pose.py')]};candidates={};fallbacks={};audits={k:[] for k in arms};outs={k:[] for k in arms};start=time.perf_counter()
 for arm,(root,kind,robust) in arms.items():
  path=b/root/f'{s}_{kind}.npz';c,m=read(path)
  if m['source_sha256'].get(str(initial))!=file_sha256(initial) or m['source_sha256'].get(str(mp))!=file_sha256(mp) or not np.array_equal(c['names'],base['names']):raise ValueError('candidate source binding differs')
  candidates[arm]=c;sources[str(path)]=file_sha256(path)
  fp=(b/'direct_feature_precision_v330'/f'{s}_fine_joint512.npz') if arm=='plain' else fallback_path
  fallbacks[arm],_=read(fp);sources[str(fp)]=file_sha256(fp)
 if not np.array_equal(fallback['names'],base['names']) or not np.array_equal(corr['names'],base['names']) or [r['name'] for r in rows]!=base['names'].tolist():raise ValueError('query alignment differs')
 for i,n in enumerate(base['names'].astype(str)):
  tokens=np.array(rows[i]['selected_tokens'],int);pr=np.array(rows[i]['prototype_rows'],int);world=maps['world_points'][pr].astype(float);target=maps['fine'][pr].astype(float);K=corr['camera_matrices'][i].astype(float);k1=float(corr['radial_k1'][i]);fold=(tokens//64//8+tokens%64//8)%2;cache=b/'native_fine_v264/query_cache'/n;grids,ck=load_grids(cache,mm['projection_sha256']);sources[str(cache)]=file_sha256(cache)
  if ck!=mm['checkpoint_sha256']:raise ValueError('query checkpoint differs')
  for arm,(_,_,robust) in arms.items():
   detail=[];enough=bool(min(np.sum(fold==0),np.sum(fold==1))>=32);alpha=0.
   if enough:
    losses=[]
    for f in [0,1]:
     fit=fold==f;val=~fit
     regularized,rm=refine(base['pose_w2c'][i],world[fit],K,k1,grids[1],target[fit],.02,robust=robust)
     free,fm=refine(base['pose_w2c'][i],world[fit],K,k1,grids[1],target[fit],0.,robust=robust)
     scores=[feature_loss(interpolate_pose(regularized,free,v),world[val],K,k1,grids[1],target[val],robust) for v in [0.,.5,1.]]
     losses.append(scores);detail.append(dict(fold=f,fit_tokens=tokens[fit].tolist(),heldout_tokens=tokens[val].tolist(),losses=scores,regularized_fit_accepted=rm['accepted'],free_fit_accepted=fm['accepted']))
    alpha=[0.,.5,1.][int(np.argmin(np.mean(losses,axis=0)))]
   if np.array_equal(candidates[arm]['pose_w2c'][i],base['pose_w2c'][i]):alpha=0.
   chosen=interpolate_pose(fallbacks[arm]['pose_w2c'][i],candidates[arm]['pose_w2c'][i],alpha);outs[arm].append(chosen);audits[arm].append(dict(name=n,accepted=alpha>0,selected_alpha=alpha,folds=detail))
  if (i+1)%20==0:print(s,i+1,flush=True)
 for arm,values in outs.items():
  arr=dict(names=base['names'],pose_w2c=np.array(values),usable=np.isfinite(values).all((1,2)),accepted=np.array([v['accepted'] for v in audits[arm]]));m=dict(artifact_type='goal_maplet_crossfit_feature_step_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,source_sha256=sources,arm=arm,not_statistically_independent=True,rule='minimize two-fold held-out feature loss over fixed alpha0,0.5,1; interpolate full regularized/free endpoints; no GT choice')
  m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(o/f'{s}_{arm}.npz',**arr,metadata_json=np.array(json.dumps(m,sort_keys=True)));(o/f'{s}_{arm}_audit.json').write_text(json.dumps(audits[arm]))
 (o/f'{s}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,scope='two modes, eight additional fold fits per query; cached features')))
if __name__=='__main__':main()
