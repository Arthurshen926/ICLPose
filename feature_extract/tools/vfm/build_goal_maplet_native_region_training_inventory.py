"""Prepare native cross-route regional retrieval candidates without utility labels.

This is a data prerequisite, not a learned selector or a pose-utility oracle.
Each mapping query route is excluded from both context modes and local native
prototype memberships. Physical world coordinates retain exact native identity.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import select_context_modes
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise,sector_descriptors
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def write(path,arrays,meta):
 meta=dict(meta,arrays_sha256=arrays_sha256(arrays));meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(path,**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
 atlaspath=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';lineagepath=b/'native_fine_v264/mapping_lineage.npz';finepath=b/'native_fine_v264/readout/map.npz'
 with np.load(atlaspath) as z:world=z['world_points']
 with np.load(finepath) as z:fm=json.loads(str(z['metadata_json']));available=z['available'];assert np.array_equal(world,z['world_points'])
 if fm['atlas_sha256']!=file_sha256(atlaspath) or fm['lineage_sha256']!=file_sha256(lineagepath):raise ValueError('native lineage differs')
 with np.load(lineagepath) as z:names=z['source_names'].astype(str);source=z['prototype_source_and_cell'][:,0];assert bool(z['offline_only'])
 routes=np.array([name.split('__')[0] for name in names]);assert not np.isin(routes,['seq10','seq13']).any()
 with np.load(b/'region_frontend_v246/map.npz') as z:centers=z['centers']
 with np.load(b/'native_fine_calibration_v267/mapping_rows.npz') as z:targets=np.unique(z['source']);assert np.array_equal(names,z['source_names'])
 contexts=np.zeros((len(world),256),np.float16);query_contexts={};ledger={};contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')
 for count,src in enumerate(np.unique(source)):
  rows=np.flatnonzero(source==src);cache=b/'adaptive_memory_v234/fine_cache'/names[src];grids,ck=load_grids(cache,fm['projection_sha256'])
  if ck!=fm['checkpoint_sha256']:raise ValueError('mapping feature checkpoint differs')
  context=normalise(sector_descriptors(grids[0],1).reshape(36,64,256));camera=contributors/names[src]
  with np.load(camera) as z:pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
  xy,cam=_project(pose,world[rows],K,k1);contexts[rows]=np.where(available[rows,None],sample_grid(context,np.nan_to_num(xy)),0.)
  if src in targets:query_contexts[src]=context
  ledger[names[src]]=dict(cache_sha256=file_sha256(cache),mapping_camera_sha256=file_sha256(camera))
  if (count+1)%100==0:print('native contexts',count+1,flush=True)
 # Some target views may not own a selected atlas prototype; load their RGB features only.
 for src in targets:
  if src not in query_contexts:
   grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/names[src],fm['projection_sha256']);assert ck==fm['checkpoint_sha256'];query_contexts[src]=normalise(sector_descriptors(grids[0],1).reshape(36,64,256))
 np.savez_compressed(o/'offline_native_contexts.npz',context=contexts,native_source=source,source_names=names,offline_only=np.array(True))
 groups=cKDTree(world).query_ball_point(centers,6.);folds={};audit=[];foldsha={};origins={}
 for route in sorted(set(routes[targets])):
  modes=[];mode_regions=[];geometry=[];offsets=[0];chosen_sources=[]
  for rid,g in enumerate(groups):
   g=np.array(g,int);g=g[(routes[source[g]]!=route)&available[g]];geometry.extend(g);offsets.append(len(geometry))
   sources=np.unique(source[g]);sg=[g[source[g]==s] for s in sources];order=np.argsort([-len(v) for v in sg],kind='stable')[:4]
   for j in order:
    if len(sg[j])<6:continue
    modes.append(normalise(contexts[sg[j]].astype(np.float32).mean(0)));mode_regions.append(rid);chosen_sources.append(int(sources[j]))
  arr=dict(descriptors=np.asarray(modes,np.float32),mode_regions=np.asarray(mode_regions,int),centers=centers,geometry_member_rows=np.asarray(geometry,int),geometry_offsets=np.asarray(offsets,int));path=o/f'{route}_map.npz'
  assert not np.any(routes[source[arr['geometry_member_rows']]]==route);assert not np.any(routes[chosen_sources]==route)
  write(path,arr,dict(artifact_type='goal_maplet_native_crossroute_region_library_v1',native_atlas_sha256=file_sha256(atlaspath),excluded_mapping_route=route,radius_m=6.,maximum_modes_per_region=4,source_identity_retained_at_runtime=False,query_pose_or_ground_truth_used_for_retrieval=False));foldsha[route]=file_sha256(path);folds[route]=arr;origins[route]=[names[s] for s in chosen_sources]
  audit.append(dict(route=route,mode_count=len(modes),geometry_memberships=len(geometry),query_route_source_excluded_from_context_and_geometry=True))
 qi=np.arange(2304).reshape(36,64);patches=[qi[y:y+18,x:x+32].ravel() for y in [0,18] for x in [0,32]];regions=[];scores=[]
 for src in targets:
  lib=folds[routes[src]];qd=query_contexts[src].reshape(2304,256);aggregate=normalise(np.array([qd[g].mean(0) for g in patches]));patch_scores=aggregate@lib['descriptors'].T;chosen=select_context_modes(patch_scores,lib['mode_regions'],16);regions.append(lib['mode_regions'][chosen]);scores.append(patch_scores[:,chosen].T)
 write(o/'retrieval_candidates.npz',dict(names=names[targets],regions=np.asarray(regions),quadrant_scores=np.asarray(scores),mapping_routes=routes[targets]),dict(artifact_type='goal_maplet_native_mapping_region_candidates_v1',contains_pose_or_utility_labels=False,fold_map_sha256=foldsha,retrieved_distinct_regions=16,geometry='exact current native atlas membership, not old row transfer'))
 (o/'offline_source_audit.json').write_text(json.dumps(dict(ledger=ledger,context_mode_sources=origins),indent=2))
 (o/'report.json').write_text(json.dumps(dict(scope=__doc__,mapping_queries=len(targets),routes=sorted(set(routes[targets])),folds=audit,pending=['MoGe mapping query regions','original plane-front-end correspondences on same mapping folds','per-next-region frozen pose counterfactuals and postfreeze targets','learned next-region marginal pose utility'],current_query_routes_used=False,default_runtime_map_changed=False),indent=2));print('native fold libraries and mapping retrieval candidates complete',flush=True)

if __name__=='__main__':main()
