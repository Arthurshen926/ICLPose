"""Cross-route mapping-only, region-only next-region PnP counterfactuals.

This deliberately separate diagnostic does not represent the full hybrid frontend.
Pose labels are opened only after every candidate pose and feature is frozen.
"""
import argparse,json,concurrent.futures,hashlib
from pathlib import Path
import numpy as np
import torch
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_native_region_training_inventory import write
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import load_mapping_subtoken_head
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support,_scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
from feature_extract.tools.vfm.token_hypothesis_ransac import solve


def counterfactual(job):
 path,output=map(Path,job[:2]);seed_offset=int(job[2]) if len(job)>2 else 0
 with np.load(path) as z:d={k:z[k] for k in z.files if k!='metadata_json'}
 offsets=d['offsets'];base=np.arange(offsets[8]);poses=[];features=[];stats=[]
 seed=260912+seed_offset+int(hashlib.sha256(path.name.encode()).hexdigest()[:6],16)
 base_tokens=np.unique(d['tokens'][base]);base_score=d['scores'][base]
 for candidate in [-1]+list(range(8,16)):
  extra=np.arange(offsets[candidate],offsets[candidate+1]) if candidate>=0 else np.empty(0,int)
  rows=np.r_[base,extra];ss={};pose=solve(d['world'],d['tokens'],d['K'],float(d['k1']),rows,pixels=d['pixels'],iterations=1024,hypothesis_budget=256,seed=seed,stats=ss)
  poses.append(np.full((4,4),np.nan) if pose is None else pose);stats.append(ss)
  if candidate>=0:
   tok=np.unique(d['tokens'][extra]);new=np.setdiff1d(tok,base_tokens);v=d['scores'][extra];context=d['context_scores'][candidate]
   # Features describe addition conditional on the fixed first-eight set; no pose labels.
   from feature_extract.tools.vfm.native_region_value_features import addition_features
   features.append(addition_features(base_tokens,base_score,d['tokens'][extra],v,context,d['context_scores'][:8],d['centers'][candidate],d['centers'][:8]))
 write(output,dict(poses=np.asarray(poses),features=np.asarray(features)),dict(artifact_type='goal_maplet_native_region_counterfactual_v1',source_rows_sha256=file_sha256(path),query_pose_or_ground_truth_read=False,seed=seed,scope='region-only top8 versus top8 plus one of ranks9..16; same token AP3P budget; not hybrid mainline',stats=stats))
 return path.name


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--workers',type=int,default=8);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False);(o/'rows').mkdir();(o/'frozen').mkdir();fold=b/'native_region_training_v276'
 atlaspath=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz'
 with np.load(atlaspath) as z:world=z['world_points'];desc=z['radio_features'].astype(np.float32)
 with np.load(b/'native_fine_v264/readout/map.npz') as z:fm=json.loads(str(z['metadata_json']))
 with np.load(fold/'retrieval_candidates.npz') as z:names=z['names'].astype(str);regions=z['regions'];scores=z['quadrant_scores'];meta=json.loads(str(z['metadata_json']))
 head,hm=load_mapping_subtoken_head(b/'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz');head.eval();gpu=torch.as_tensor(desc,device=a.device);contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean');libs={}
 for route in sorted({n.split('__')[0] for n in names}):
  path=fold/(route+'_map.npz');assert file_sha256(path)==meta['fold_map_sha256'][route]
  with np.load(path) as z:libs[route]={k:z[k] for k in z.files if k!='metadata_json'}
 for i,name in enumerate(names):
  route=name.split('__')[0];lib=libs[route];grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/name,fm['projection_sha256']);assert ck==fm['checkpoint_sha256'];q=grids[0].reshape(2304,64).astype(np.float32)
  planes,pm=QueryPlaneRegions.load_npz(fold/'planes'/route/name);assert pm['uses_pose_or_ground_truth'] is False;valid=[]
  for rid in range(len(planes.pixel_counts)):
   tt,vv=_region_token_support(planes.labels,rid);valid.extend(tt[vv>=.75])
  valid=np.unique(valid).astype(int);tq=torch.as_tensor(q[valid],device=a.device);tokens=[];protos=[];values=[];offsets=[0]
  for rid in regions[i]:
   g=lib['geometry_member_rows'][lib['geometry_offsets'][rid]:lib['geometry_offsets'][rid+1]]
   if len(g) and len(valid):
    sim=tq@gpu[g].T;val,ind=sim.max(1);mutual=sim.argmax(0)[ind]==torch.arange(len(valid),device=a.device);tt=valid[mutual.cpu().numpy()];pp=g[ind[mutual].cpu().numpy()];vv=val[mutual].cpu().numpy();order=np.argsort(-vv,kind='stable')[:1024];tokens.extend(tt[order]);protos.extend(pp[order]);values.extend(vv[order])
   offsets.append(len(tokens))
  tokens=np.asarray(tokens,int);protos=np.asarray(protos,int)
  with torch.no_grad():mean,var,logit=head(torch.from_numpy(q[tokens]),torch.from_numpy(desc[protos]),torch.from_numpy(tokens))
  delta=mean.numpy().astype(float)
  if hm.get('coordinate_affine_matrix') is None:delta*=float(hm.get('coordinate_shrinkage',1.))
  else:delta=delta@np.asarray(hm['coordinate_affine_matrix'])+np.asarray(hm['coordinate_affine_bias_px'])
  with np.load(contributors/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
  write(o/'rows'/name,dict(world=world[protos],tokens=tokens,prototype_rows=protos,pixels=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]+np.clip(delta,-2,2),scores=np.asarray(values),offsets=np.asarray(offsets),K=K,k1=np.asarray(k1),context_scores=scores[i],centers=lib['centers'][regions[i]]),dict(query_pose_or_ground_truth_read=False,fold_library_sha256=meta['fold_map_sha256'][route],native_atlas_sha256=file_sha256(atlaspath),geometry_file_sha256=file_sha256(fold/'planes'/route/name)))
  if (i+1)%20==0:print('mapping regional matches',i+1,flush=True)
 del gpu,tq;torch.cuda.empty_cache()
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for i,name in enumerate(pool.map(counterfactual,[(o/'rows'/n,o/'frozen'/n) for n in names])):
   if (i+1)%20==0:print('frozen region counterfactuals',i+1,flush=True)
 (o/'protocol.json').write_text(json.dumps(dict(scope=__doc__,queries=len(names),query_pose_or_ground_truth_read=False,candidate_sha256=file_sha256(fold/'retrieval_candidates.npz'),hypothesis_budget=256,attempt_budget=1024,base_regions=8,candidate_ranks=list(range(9,17))),indent=2))

if __name__=='__main__':main()
