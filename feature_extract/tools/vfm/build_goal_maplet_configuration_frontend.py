"""Pose-free fixed-region retrieval and joint correspondence configuration controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens,local_modes,retrieve_local_regions,retrieve_anchor_regions
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import select_context_modes
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors,normalise
from feature_extract.tools.vfm.surface_configuration_matching import configuration_match
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);p.add_argument('--seed',type=int,default=260901);p.add_argument('--overlap-aware',action='store_true');p.add_argument('--native-normals',action='store_true');p.add_argument('--anchor-retrieval',action='store_true');p.add_argument('--query-relative-null',action='store_true');p.add_argument('--regions-per-query',type=int,default=4);p.add_argument('--local-memory',action='store_true');p.add_argument('--distinctive-sampling',action='store_true');p.add_argument('--fine-matching',action='store_true');p.add_argument('--arms',nargs='+',choices=['mnn','independent','joint','shuffled'],default=['mnn','independent','joint','shuffled']);a=p.parse_args();b=a.base;s=a.split;o=a.output;o.mkdir(parents=True,exist_ok=True);arms=a.arms;start=time.perf_counter()
 if any((o/f'{s}_{arm}.npz').exists() for arm in arms):raise FileExistsError(o)
 ap=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';mp=b/'native_context_v278/map.npz';fp=b/'native_fine_v264/readout/map.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz'
 with np.load(ap) as f:am=json.loads(f['metadata_json'].item());world=f['world_points'].astype(float);mf=normalise(f['radio_features'].astype(float));texel=f['texel_identity'];offset=f['plane_texel_offsets']
 with np.load(mp) as f:centers=f['centers'];modes=f['descriptors'];mode_regions=f['mode_regions'];mm=json.loads(f['metadata_json'].item());ma={k:f[k] for k in f.files if k!='metadata_json'}
 assert arrays_sha256(ma)==mm['arrays_sha256'] and mm['native_atlas_sha256']==file_sha256(ap)
 assert mm['query_pose_or_ground_truth_used_for_retrieval'] is False
 assert canonical_json_sha256({k:v for k,v in mm.items() if k!='content_sha256'})==mm['content_sha256']
 with np.load(fp) as f:fm=json.loads(f['metadata_json'].item());fine_map=f['fine'].astype(float);map_available=f['available']
 camera_inventory,_=_load(cp);names=camera_inventory['names'];Ks=camera_inventory['camera_matrices'];ks=camera_inventory['radial_k1']
 projection=b/'stmarys_chart_local_radio_projection_64d_v2.npz'
 with np.load(projection) as f:projection_meta=json.loads(f['metadata_json'].item())
 assert not {'seq10','seq13'}&set(projection_meta['fit_mapping_routes']+[projection_meta['validation_mapping_route']])
 calibration=projection_meta['validation_learned_projection'];absolute_null=.5*(calibration['positive_cosine_mean']+calibration['same_plane_negative_cosine_mean']) if a.overlap_aware else None
 assert not set(names.astype(str))&set(mm['offline_mapping_source_names'])
 normals=np.zeros_like(world)
 for lo,hi in zip(offset[:-1],offset[1:]):
  points=world[lo:hi]
  if len(points)>=3:normals[lo:hi]=np.linalg.svd(points-points.mean(0),full_matrices=False)[2][-1]
 gp=b.parent/'stmarys_rendered_ransac_fused_planes_v1.npz'
 if a.native_normals:
  assert file_sha256(gp)==am['planar_map_file_sha256']
  with np.load(gp) as f:normals=np.repeat(f['normals_world'],np.diff(offset),axis=0)
  assert normals.shape==world.shape
 local_path=b/'native_region_training_v276/offline_native_contexts.npz'
 if a.local_memory:
  with np.load(local_path) as f:
   local_context=f['context'];source_names=f['source_names'].astype(str)
  assert not any(n.startswith(('seq10__','seq13__')) for n in source_names)
  mode_rows=local_modes(world,local_context,map_available);modes=normalise(local_context[mode_rows]);centers=world[mode_rows];mode_regions=np.arange(len(mode_rows))
  np.savez_compressed(o/f'{s}_local_memory.npz',prototype_rows=mode_rows,centers=centers,descriptors=modes)
 if a.anchor_retrieval:
  if a.local_memory:raise ValueError('choose one retrieval control')
  centers=world
 if not 1<=a.regions_per_query<=16:raise ValueError('region count must be in [1,16]')
 tree=cKDTree(world);regions={};qi=np.arange(2304).reshape(36,64);patches=[qi[y:y+18,x:x+32].ravel() for y in [0,18] for x in [0,32]]
 cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1]);sources={str(f):file_sha256(f) for f in [ap,mp,fp,cp,projection,Path(__file__),Path(__file__).with_name('surface_configuration_matching.py'),Path(__file__).with_name('token_hypothesis_ransac.py')]};sources.update({str(gp):file_sha256(gp)} if a.native_normals else {});sources[str(Path(__file__).with_name('partial_visibility_memory.py'))]=file_sha256(Path(__file__).with_name('partial_visibility_memory.py'));sources.update({str(local_path):file_sha256(local_path)} if a.local_memory else {});out={k:[] for k in arms};audits={k:[] for k in arms}
 for ix,name in enumerate(names.astype(str)):
  cache=b/'native_fine_v264/query_cache'/name;grids,ck=load_grids(cache,fm['projection_sha256']);assert ck==fm['checkpoint_sha256'];sources[str(cache)]=file_sha256(cache);q=normalise(grids[0].reshape(2304,64).astype(float));qp,qn,qv=moge_tokens(md/name);sources[str(md/name)]=file_sha256(md/name)
  context=normalise(sector_descriptors(q.reshape(36,64,64),1).reshape(2304,256));scores=normalise(np.array([context[g].mean(0) for g in patches]))@modes.T;chosen=[int(mode_regions[j]) for j in select_context_modes(scores,mode_regions,a.regions_per_query)]
  retrieval_evidence=[]
  if a.local_memory:chosen,retrieval_evidence=retrieve_local_regions(context.reshape(36,64,256),modes,centers,count=a.regions_per_query)
  if a.anchor_retrieval:chosen,retrieval_evidence=retrieve_anchor_regions(q,mf,world,count=a.regions_per_query)
  ts=diverse_tokens(q,qv) if a.distinctive_sampling else np.flatnonzero(qv)
  if len(ts)>256:ts=ts[np.linspace(0,len(ts)-1,256,dtype=int)]
  query_null=(q[ts]@mf.T).max(1)-.1 if a.query_relative_null else absolute_null
  xy=np.c_[(ts%64)*4+1.5,(ts//64)*4+1.5];inventories=[];configs={arm:[] for arm in arms};detail={arm:[] for arm in arms}
  for rid in chosen:
   if rid not in regions:regions[rid]=np.array(tree.query_ball_point(centers[rid],6.,return_sorted=True),int)
   g=regions[rid]
   if len(g)<32 or len(ts)<6:continue
   sim=q[ts]@mf[g].T;rank=np.argsort(-sim,axis=1,kind='stable')[:,:32];ids=[]
   for candidates in g[rank]:
    _,first=np.unique(texel[candidates],return_index=True);first=np.sort(first)[:4]
    if len(first)<4:raise ValueError('insufficient distinct cell modes')
    ids.append(candidates[first])
   ids=np.array(ids);unary=np.sum(q[ts,None,:]*mf[ids],axis=-1);inventories.append((world[ids].reshape(-1,3),np.repeat(ts,4),np.repeat(xy,4,axis=0)))
   for arm in arms:
    selected,valid,belief,record=configuration_match(unary,world[ids],normals[ids],qp[ts],qn[ts],xy,'independent' if arm=='mnn' else arm,absolute_null=query_null,exclusive=a.overlap_aware and arm!='mnn')
    if arm=='mnn':valid &= np.argmax(sim,axis=0)[rank[:,0]]==np.arange(len(ts))
    take=np.flatnonzero(valid);pr=ids[take,selected[take]];stats={}
    measured=xy[take].copy()
    if a.fine_matching and len(pr):
     offsets_xy=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
     samples=sample_grid(grids[1],(measured[:,None]+offsets_xy).reshape(-1,2)).reshape(len(pr),9,64)
     fs=np.einsum('nkd,nd->nk',samples,normalise(fine_map[pr]));best=np.argmax(fs,axis=1)
     # Preserve the center on ties; unavailable map descriptors cannot move points.
     best[(fs[:,4]>=fs.max(1))|~map_available[pr]]=4
     measured+=offsets_xy[best]
    pose=solve(world[pr],ts[take],Ks[ix],float(ks[ix]),np.arange(len(pr)),pixels=measured,iterations=1250,seed=a.seed+rid,hypothesis_budget=1000//a.regions_per_query,stats=stats)
    configs[arm].append(pose);detail[arm].append(dict(region=rid,selected_tokens=ts[take].tolist(),prototype_rows=pr.tolist(),query_pixels=measured.tolist(),matching=record,pnp=stats))
  if inventories:
   cw,ct,cxy=canonical_hypotheses(*[np.concatenate([v[j] for v in inventories]) for j in range(3)]);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)]
  for arm in arms:
   valid=[p for p in configs[arm] if p is not None];best=max(valid,key=lambda p:score_pose(p,cw,cxy,groups,Ks[ix],float(ks[ix]),return_selected=False)[0]) if valid else np.full((4,4),np.nan)
   out[arm].append(best);audits[arm].append(dict(name=name,accepted=bool(np.isfinite(best).all()),regions=detail[arm],region_ids=chosen,candidate_poses=[None if p is None else p.tolist() for p in configs[arm]],valid_query_tokens=len(ts),sampled_tokens=ts.tolist(),retrieval_evidence=retrieval_evidence))
  if (ix+1)%10==0:print(s,ix+1,flush=True)
 for arm,values in out.items():
  arrays=dict(names=names,pose_w2c=np.array(values),usable=np.isfinite(values).all((1,2)));m=dict(artifact_type='goal_maplet_configuration_frontend_v1',arrays_sha256=arrays_sha256(arrays),source_sha256=sources,query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,arm=arm,seed=a.seed,overlap_aware=a.overlap_aware,absolute_null=absolute_null,calibration_is_not_probability=True,fixed_region_radius_m=6,regions_per_query=a.regions_per_query,anchor_retrieval=a.anchor_retrieval,query_relative_null=a.query_relative_null,query_tokens_cap=256,alternatives_per_token_per_region=4,pnp_hypothesis_cap_per_region=1000//a.regions_per_query,subtoken_head_used=False,fine_grid_search_used=a.fine_matching,local_memory=a.local_memory,distinctive_sampling=a.distinctive_sampling,local_memory_is_learned=False,local_context_source_sha256=file_sha256(local_path) if a.local_memory else None,geometry_anchors_unchanged=True,normals='hash-bound native plane normals; unsigned angular relationships only' if a.native_normals else 'per-atlas-plane PCA; unsigned angular relationships only',scope='pose-free new correspondence frontend; fixed-radius metric neighborhoods; no learned memory members')
  m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(o/f'{s}_{arm}.npz',**arrays,metadata_json=np.array(json.dumps(m,sort_keys=True)));(o/f'{s}_{arm}_audit.json').write_text(json.dumps(audits[arm]))
 (o/f'{s}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,controls=len(arms),scope='requested controls with cached features; not RGB-to-pose latency')))
if __name__=='__main__':main()
