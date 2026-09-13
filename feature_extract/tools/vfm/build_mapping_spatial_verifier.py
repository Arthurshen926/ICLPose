from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.partial_overlap_matcher import make_pair
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.tools.vfm.token_hypothesis_ransac import solve
from feature_extract.tools.vfm.spatial_pose_evidence import evidence,FEATURE_NAMES
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine
from feature_extract.tools.vfm.prepare_overlap_lod_training import project
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
p=argparse.ArgumentParser();p.add_argument('--route',required=True);a=p.parse_args();route=a.route
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');o=b/'spatial_verifier_v404/mapping';o.mkdir(parents=True,exist_ok=True);dest=o/(route+'.json');assert not dest.exists();maps,meta,sources=load_map(b);w=maps['world'];tree=cKDTree(w)
cp=b/'native_hybrid_mapping_v286'/route/'corr.npz';mask=cp.with_name('mask.npz')
with np.load(cp) as f:names=f['names'].astype(str);Ks=f['camera_matrices'];ks=f['radial_k1'];cm=json.loads(f['metadata_json'].item())
with np.load(mask) as f:eligible=f['eligible'];mm=json.loads(f['metadata_json'].item());assert mm['excluded_mapping_route']==route
assert cm['atlas_member_mask_file_sha256']==file_sha256(mask)
with np.load(b/'native_region_training_v276/offline_native_contexts.npz') as f:owners=f['source_names'].astype(str)[f['native_source']]
assert not eligible[np.char.startswith(owners,route+'__')].any()
with np.load(b/'stmarys_chart_local_radio_projection_64d_v2.npz') as f:pm=json.loads(f['metadata_json'].item())['validation_learned_projection'];threshold=.5*(pm['positive_cosine_mean']+pm['same_plane_negative_cosine_mean'])
records=[]
for i,name in enumerate(names):
 grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/name,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];qp,qn,qv=moge_tokens(b/'native_region_training_v276/moge'/route/name);tokens=diverse_tokens(normalise(grids[0].reshape(2304,64)),qv)
 proposals=json.load(open(b/'overlap_lod_v402/training'/(name+'.retrieval.json')))['lod'];inventories=[];poses=[]
 for rid in proposals:
  region=np.array(tree.query_ball_point(w[rid],6.,return_sorted=True),int);region=region[eligible[region]]
  if len(region)<16:continue
  pair=make_pair(grids,qp,qn,qv,tokens,region,**maps);take=np.flatnonzero(pair['mutual']&(pair['similarity'][:,0,0]>=threshold));ids=pair['ids'][take,0];xy=fine_coordinates(grids,pair['pixels'][take],pair['ids'][take,:1],maps)[:,0];t=tokens[take]
  inventories.append((ids,t,xy));pose=solve(w[ids],t,Ks[i],float(ks[i]),np.arange(len(ids)),pixels=xy,iterations=1250,seed=260901+rid,hypothesis_budget=250,sampling_policy='context_prior',scores=np.ones(len(ids)));poses.append(pose)
 if not inventories:continue
 ids,t,xy=[np.concatenate([v[j] for v in inventories]) for j in range(3)];world=w[ids].astype(float)
 candidates=[]
 for pose in poses:
  if pose is None:continue
  candidates.append(('raw',pose.copy()))
  for stage in ['multiscale','fine']:
   uv,z,_=project(world,pose,Ks[i],float(ks[i]));err=np.linalg.norm(uv-xy,axis=1);rows=np.flatnonzero((z>0)&(err<=4)&maps['available'][ids]&(uv>=4).all(1)&(uv<=np.array([251,139])).all(1));rows=rows[np.lexsort((ids[rows],err[rows],t[rows]))];_,first=np.unique(t[rows],return_index=True);rows=rows[first]
   if len(rows)<32:continue
   pose,_=refine(pose,world[rows],Ks[i],float(ks[i]),grids[1],maps['fine_map'][ids[rows]],0 if stage=='multiscale' else .02,robust=stage=='multiscale',constrained=True,additional_grid=grids[0] if stage=='multiscale' else None,additional_target=maps['coarse_map'][ids[rows]] if stage=='multiscale' else None)
  candidates.append(('refined',pose))
 for stage,pose in candidates:records.append(dict(name=name,route=route,stage=stage,pose=pose.tolist(),features=evidence(pose,world,t,xy,Ks[i],float(ks[i])).tolist()))
 print(route,i+1,flush=True)
# Freeze all candidates/features before reading this route's label files.
frozen=o/(route+'_unlabelled.json');frozen.write_text(json.dumps(records));frozen_sha=file_sha256(frozen)
for r in records:
 with np.load(Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/r['name']) as f:r['error']=list(_pose_error(np.array(r['pose']),f['pose_w2c']))
 r['positive']=bool(r['error'][0]<=.5 and r['error'][1]<=5)
dest.write_text(json.dumps(dict(records=records,feature_names=FEATURE_NAMES,unlabelled_sha256=frozen_sha,excluded_mapping_route=route,query_routes_read=False,geometry_rebuilt_per_route=False),indent=2))
