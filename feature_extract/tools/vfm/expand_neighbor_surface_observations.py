"""Frozen-feature neighboring-region observation experiment (v410).

Run from the repository root with ``python -m``. This deliberately retains the
historical experiment paths. Each query opens up to four new neighboring regions,
removes correspondence pairs already present in the MNN inventory, and solves
these new measurements separately. Uniform selection is compared with marginal
appearance coverage. This is not geometry-aware complementarity, joint old/new
PnP, calibrated visibility, or a complete adaptive LoD loop. Query poses are never
loaded here. Evaluation and held-out acceptance are separate experiment drivers.
"""
from pathlib import Path
import argparse,json,time,numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.partial_overlap_matcher import make_pair
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.shared_identity_matcher import reciprocal_identity
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=260901);a=p.parse_args();s=a.split;o=a.output;o.mkdir(exist_ok=True,parents=True);b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');maps,meta,sources=load_map(b);world=maps['world'];tree=cKDTree(world)
# One actual anchor per 2m cell, no averaged coordinates.
_,rep=np.unique(np.floor(world/2).astype(int),axis=0,return_index=True);ap=b/'overlap_lod_v402/lod_fixed'/f'{s}_mnn_audit.json';old=json.load(open(ap));cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz'
with np.load(cp) as f:names=f['names'];Ks=f['camera_matrices'];ks=f['radial_k1']
with np.load(b/'stmarys_chart_local_radio_projection_64d_v2.npz') as f:cal=json.loads(f['metadata_json'].item())['validation_learned_projection'];threshold=.5*(cal['positive_cosine_mean']+cal['same_plane_negative_cosine_mean'])
weights=np.array(json.load(open(b/'shared_identity_v408/model_appearance/metadata.json'))['feature_weights']);cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1]);arms=['uniform','complement'];values={x:[] for x in arms};records={x:[] for x in arms};sources.update({str(p):file_sha256(p) for p in [ap,cp,Path(__file__),Path('feature_extract/tools/vfm/shared_identity_matcher.py'),Path('feature_extract/tools/vfm/run_overlap_lod_frontend.py'),b/'shared_identity_v408/model_appearance/metadata.json',b/'stmarys_chart_local_radio_projection_64d_v2.npz']});regions={}
def region(rid):
 if rid not in regions:regions[rid]=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int)
 return regions[rid]
for i,n in enumerate(names.astype(str)):
 start=time.perf_counter();oldrec=old[i];assert oldrec['name']==n;centers=oldrec['region_ids'];tokens=np.array(oldrec['sampled_tokens']);opened=np.unique(np.concatenate([region(rid) for rid in centers])) if centers else np.array([],int);excluded=np.zeros(len(world),bool);excluded[opened]=True
 cache=b/'native_fine_v264/query_cache'/n;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];qp,qn,qv=moge_tokens(md/n);sources[str(cache)]=file_sha256(cache);sources[str(md/n)]=file_sha256(md/n);q=grids[0].reshape(2304,64)[tokens];q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
 distance=np.min(np.linalg.norm(world[rep,None]-world[np.array(centers)][None],axis=-1),axis=1) if centers else np.full(len(rep),np.inf);front=[]
 for rid in rep[np.argsort(distance,kind='stable')]:
  d=np.min(np.linalg.norm(world[rid]-world[np.array(centers)],axis=1)) if centers else np.inf
  if d<6 or d>12 or (front and np.min(np.linalg.norm(world[rid]-world[front],axis=1))<3):continue
  ids=region(int(rid))
  if len(ids)>=16:front.append(int(rid))
  if len(front)==16:break
 bank=[region(rid) for rid in front];responses=[]
 # Equal shortlist scoring is charged to both arms. Descriptor reads are audited.
 for ids in bank:
  samples=ids[np.linspace(0,len(ids)-1,min(32,len(ids)),dtype=int)];responses.append((q@maps['coarse_map'][samples].T).max(1))
 original_rows=np.concatenate([np.array(x['prototype_rows'],int) for x in oldrec['regions']]) if oldrec['regions'] else np.array([],int)
 baseline=(q@maps['coarse_map'][np.unique(original_rows)].T).max(1) if len(original_rows) else np.zeros(len(q));density=np.maximum(np.maximum(q@q.T,0)**8@np.ones(len(q)),1);chosen=[];cover=baseline.copy()
 for _ in range(min(4,len(front))):
  scores=[np.sum(np.maximum(x-cover,0)/density) if j not in chosen else -np.inf for j,x in enumerate(responses)];j=int(np.argmax(scores));chosen.append(j);cover=np.maximum(cover,responses[j])
 oldpairs=set((int(t),int(p)) for rr in oldrec['regions'] for t,p in zip(rr['selected_tokens'],rr['prototype_rows']));choices=dict(complement=chosen,uniform=np.random.default_rng(a.seed+i).choice(len(front),min(4,len(front)),replace=False).tolist());shared_seconds=time.perf_counter()-start
 for arm in arms:
  rr=[];poses=[];inventory=[];armstart=time.perf_counter()
  for j in choices[arm]:
   ids=bank[j];pair=make_pair(grids,qp,qn,qv,tokens,ids,**maps);logits=pair['similarity']@weights;take,cols=reciprocal_identity(logits,pair['ids'],pair['similarity'][...,0],threshold);pr=pair['ids'][take,cols];fresh=np.array([(int(tokens[t]),int(p)) not in oldpairs for t,p in zip(take,pr)],bool);take=take[fresh];cols=cols[fresh];pr=pr[fresh];xy=fine_coordinates(grids,pair['pixels'][take],pr[:,None],maps)[:,0];t=tokens[take];stats={};pose=solve(world[pr],t,Ks[i],float(ks[i]),np.arange(len(pr)),pixels=xy,iterations=1250,seed=a.seed+front[j],hypothesis_budget=250,stats=stats,sampling_policy='context_prior',scores=np.ones(len(pr)));assert all((int(tt),int(pp)) not in oldpairs for tt,pp in zip(t,pr));poses.append(pose);inventory.append((world[pr],t,xy));rr.append(dict(region=front[j],selected_tokens=t.tolist(),prototype_rows=pr.tolist(),query_pixels=xy.tolist(),pnp=stats))
  finite=[x for x in poses if x is not None]
  if finite:
   w,t,xy=canonical_hypotheses(*[np.concatenate([x[j] for x in inventory]) for j in range(3)]);groups=[np.flatnonzero(t==v) for v in np.unique(t)];pose=max(finite,key=lambda p:score_pose(p,w,xy,groups,Ks[i],float(ks[i]),return_selected=False)[0])
  else:pose=np.full((4,4),np.nan)
  values[arm].append(pose);records[arm].append(dict(name=n,accepted=bool(finite),regions=rr,region_ids=[front[j] for j in choices[arm]],candidate_poses=[p.tolist() if p is not None else None for p in poses],sampled_tokens=tokens.tolist(),frontier=front,all_correspondence_pairs_are_new=True,new_anchor_outside_original_fraction=float(np.mean([not excluded[p] for reg in rr for p in reg['prototype_rows']])) if any(reg['prototype_rows'] for reg in rr) else None,shared_search_seconds=shared_seconds,arm_seconds=time.perf_counter()-armstart,new_anchor_comparisons=int(sum(len(bank[j])*len(tokens) for j in choices[arm])),representative_comparisons=int(sum(min(32,len(ids))*len(tokens) for ids in bank))))
 if (i+1)%10==0:print(s,i+1,flush=True)
for arm in arms:
 arr=dict(names=names,pose_w2c=np.array(values[arm]),usable=np.isfinite(values[arm]).all((1,2)));m=dict(artifact_type='neighbor_surface_expansion_v410',arrays_sha256=arrays_sha256(arr),source_sha256=sources,query_pose_or_ground_truth_read=False,arm=arm,seed=a.seed,max_new_regions=4,original_query_tokens_preserved=True,map_geometry_unchanged=True,projection_gate_used=False,selection='marginal appearance coverage' if arm=='complement' else 'uniform random neighboring centers',not_learned_lod=True);m['content_sha256']=canonical_json_sha256(m);dest=o/f'{s}_{arm}.npz';assert not dest.exists();np.savez_compressed(dest,**arr,metadata_json=np.array(json.dumps(m)));(o/f'{s}_{arm}_audit.json').write_text(json.dumps(records[arm]))
