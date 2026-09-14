"""Frozen two-stage refinement and common-token selection for regional proposals."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);p.add_argument('--retain-all',action='store_true');p.add_argument('--arms',nargs='+',choices=['mnn','independent','joint','shuffled','learned','overlap_mnn','weighted_mnn','spread_mnn','information_mnn','pose_information_mnn','two_stage_mnn','identity_mnn','coarse_sparse_mnn','identity_top1','uniform','complement'],default=['mnn','independent','joint','shuffled']);a=p.parse_args();b=a.base;s=a.split;o=a.output;o.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 mp=b/'native_fine_v264/readout/map.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';bp=b/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';base,bm=read(bp)
 with np.load(mp) as f:maps={k:f[k] for k in f.files if k!='metadata_json'};mm=json.loads(f['metadata_json'].item())
 assert arrays_sha256(maps)==mm['arrays_sha256']
 with np.load(cp) as f:Ks=f['camera_matrices'];ks=f['radial_k1'];assert np.array_equal(f['names'],base['names'])
 for arm in a.arms:
  ip=a.input/f'{s}_{arm}.npz';rp=a.input/f'{s}_{arm}_audit.json';candidate,cm=read(ip);records=json.load(open(rp));assert np.array_equal(candidate['names'],base['names']);sources={str(p):file_sha256(p) for p in [mp,cp,bp,ip,rp,Path(__file__),Path(__file__).with_name('direct_anonymous_feature_pose.py'),Path(__file__).with_name('token_hypothesis_ransac.py')]};values={k:[] for k in ['refined','selected']};audit=[]
  for i,n in enumerate(base['names'].astype(str)):
   assert records[i]['name']==n
   token=np.concatenate([np.asarray(r['selected_tokens'],int) for r in records[i]['regions']]) if records[i]['regions'] else np.array([],int);pr=np.concatenate([np.asarray(r['prototype_rows'],int) for r in records[i]['regions']]) if records[i]['regions'] else np.array([],int);xy=np.concatenate([np.asarray(r.get('query_pixels',np.c_[(np.asarray(r['selected_tokens'],int)%64)*4+1.5,(np.asarray(r['selected_tokens'],int)//64)*4+1.5]),float).reshape(-1,2) for r in records[i]['regions']]) if records[i]['regions'] else np.empty((0,2));w=maps['world_points'][pr].astype(float);pose=candidate['pose_w2c'][i].copy();K=Ks[i].astype(float);k1=float(ks[i]);steps=[]
   cache=b/'native_fine_v264/query_cache'/n;grids,ck=load_grids(cache,mm['projection_sha256']);assert ck==mm['checkpoint_sha256'];sources[str(cache)]=file_sha256(cache)
   candidates=[None if p is None else np.array(p) for p in records[i]['candidate_poses']] if a.retain_all else [pose]
   refined_candidates=[];all_steps=[]
   for proposed in candidates:
    pose=np.full((4,4),np.nan) if proposed is None else proposed.copy();steps=[]
    if np.isfinite(pose).all() and len(w):
     for stage in ['multiscale','fine']:
      proj=cv2.projectPoints(w,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2);z=(w@pose[:3,:3].T+pose[:3,3])[:,2];err=np.linalg.norm(proj-xy,axis=1);valid=(z>0)&(err<=4)&maps['available'][pr]&(proj>=4).all(1)&(proj<=np.array([251,139])).all(1);rows=np.flatnonzero(valid);rows=rows[np.lexsort((pr[rows],err[rows],token[rows]))];_,first=np.unique(token[rows],return_index=True);rows=rows[first]
      if stage=='multiscale' and len(rows)>512:rows=rows[np.linspace(0,len(rows)-1,512,dtype=int)]
      if len(rows)<32:steps.append(dict(stage=stage,accepted=False,support=len(rows)));continue
      pose,detail=refine(pose,w[rows],K,k1,grids[1],maps['fine'][pr[rows]],0 if stage=='multiscale' else .02,robust=stage=='multiscale',constrained=True,additional_grid=grids[0] if stage=='multiscale' else None,additional_target=maps['coarse'][pr[rows]] if stage=='multiscale' else None);steps.append(dict(stage=stage,support=len(rows),**detail))
    refined_candidates.append(pose);all_steps.append(steps)
   if len(w):
    cw,ct,cxy=canonical_hypotheses(w,token,xy);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)]
    finite=[p for p in refined_candidates if np.isfinite(p).all()]
    pose=max(finite,key=lambda p:score_pose(p,cw,cxy,groups,K,k1,return_selected=False)[0]) if finite else np.full((4,4),np.nan)
   else:pose=np.full((4,4),np.nan)
   take=False;keys=None
   if len(w) and np.isfinite(pose).all():
    cw,ct,cxy=canonical_hypotheses(w,token,xy);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)];kb=score_pose(base['pose_w2c'][i],cw,cxy,groups,K,k1,return_selected=False)[0];kn=score_pose(pose,cw,cxy,groups,K,k1,return_selected=False)[0];take=kn[0]>=24 and kn>kb;keys=[kb,kn]
   values['refined'].append(pose);values['selected'].append(pose if take else base['pose_w2c'][i]);audit.append(dict(name=n,accepted=bool(take),steps=all_steps,refined_candidate_poses=[p.tolist() if np.isfinite(p).all() else None for p in refined_candidates],common_token_scores=keys))
  for kind,poses in values.items():
   dest=o/f'{s}_{arm}_{kind}.npz'
   if dest.exists():raise FileExistsError(dest)
   arrays=dict(names=base['names'],pose_w2c=np.array(poses),usable=np.isfinite(poses).all((1,2)),accepted=np.array([r['accepted'] for r in audit]));m=dict(artifact_type='goal_maplet_configuration_refinement_v1',arrays_sha256=arrays_sha256(arrays),source_sha256=sources,query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,arm=arm,stage=kind,retain_all_regional_candidates=a.retain_all,base_pose_preserved_if_not_selected=kind=='selected',selection='same frozen unique-token reprojection score; lexicographic strict improvement; at least24 supporting tokens',selection_is_not_calibrated_confidence=True,refinement='frozen multiscale robust then fine plain regularized; region-generated support')
   m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(m,sort_keys=True)))
  (o/f'{s}_{arm}_audit.json').write_text(json.dumps(audit));print(s,arm,'done',flush=True)
 (o/f'{s}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,controls=len(a.arms),scope='cached refinement; excludes retrieval and RGB feature extraction')))
if __name__=='__main__':main()
