"""Mapping-route-excluded context retrieval and conservative visibility supervision.

This prepares training data; it does not fit member weights or use test queries.
Occluded/ambiguous regions remain unknown rather than automatic negatives.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_multiview_feature_map import depth_visible
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import select_context_modes
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise,sector_descriptors
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def visibility_label(inside,visible,identities,negative_name='negative_outside_patch'):
 inside=np.asarray(inside,bool);visible=np.asarray(visible,bool);identities=np.asarray(identities)
 if inside.shape!=visible.shape or inside.shape!=identities.shape or np.any(visible&~inside):raise ValueError('visibility contract differs')
 if len(np.unique(identities[visible]))>=6:return 'positive_visible'
 if len(inside) and not inside.any():return negative_name
 return 'unknown'


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
 context_path=b/'native_region_training_v276/offline_native_contexts.npz';map_path=b/'native_context_v278/map.npz';atlas_path=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';fine_path=b/'native_fine_v264/readout/map.npz'
 with np.load(context_path) as f:context=f['context'].astype(float);owners=f['source_names'].astype(str)[f['native_source']];assert bool(f['offline_only'])
 with np.load(map_path) as f:centers=f['centers'];mm=json.loads(f['metadata_json'].item())
 assert file_sha256(context_path)==mm['context_cache_sha256'] and file_sha256(atlas_path)==mm['native_atlas_sha256']
 with np.load(atlas_path) as f:world=f['world_points'];texel=f['texel_identity']
 with np.load(fine_path) as f:available=f['available'];fm=json.loads(f['metadata_json'].item())
 assert not any(n.startswith(('seq10__','seq13__')) for n in owners)
 groups=cKDTree(world).query_ball_point(centers,6.);qi=np.arange(2304).reshape(36,64);patches=[qi[y:y+18,x:x+32].ravel() for y in [0,18] for x in [0,32]];summary={};manifest=[]
 for route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']:
  output=o/route;output.mkdir();cp=b/'native_hybrid_mapping_v286'/route/'corr.npz';mask_path=cp.with_name('mask.npz');c,cm=_load(cp)
  with np.load(mask_path) as f:eligible=f['eligible'].astype(bool);mask_meta=json.loads(f['metadata_json'].item())
  assert cm['atlas_member_mask_file_sha256']==file_sha256(mask_path) and mask_meta['excluded_mapping_route']==route
  assert not eligible[np.array([n.startswith(route+'__') for n in owners])].any()
  assert set(n.split('__')[0] for n in c['names'].astype(str))=={route}
  descriptors=[];mode_regions=[];member_groups=[]
  for rid,g in enumerate(groups):
   g=np.array(g,int);g=g[eligible[g]&available[g]];member_groups.append(g);sources=sorted(set(owners[g]));sg=[g[owners[g]==src] for src in sources]
   for j in np.argsort([-len(v) for v in sg],kind='stable')[:4]:
    if len(sg[j])>=6:descriptors.append(normalise(context[sg[j]].mean(0)));mode_regions.append(rid)
  arrays=dict(descriptors=np.array(descriptors,np.float32),mode_regions=np.array(mode_regions,int),centers=centers,member_offsets=np.r_[0,np.cumsum([len(g) for g in member_groups])],member_rows=np.concatenate(member_groups));meta=dict(artifact_type='goal_maplet_route_excluded_member_training_context_v1',arrays_sha256=arrays_sha256(arrays),excluded_mapping_route=route,atlas_sha256=file_sha256(atlas_path),context_sha256=file_sha256(context_path),mask_sha256=file_sha256(mask_path),members_learned=False,query_pose_or_ground_truth_used_for_retrieval=False,geometry_not_rebuilt_per_route=True);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(output/'context.npz',**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
  rankings=[];hashes={}
  for name in c['names'].astype(str):
   cache=b/'adaptive_memory_v234/fine_cache'/name;grids,ck=load_grids(cache,fm['projection_sha256']);assert ck==fm['checkpoint_sha256'];hashes[str(cache)]=file_sha256(cache);q=normalise(grids[0].reshape(2304,64).astype(float));ctx=normalise(sector_descriptors(q.reshape(36,64,64),1).reshape(2304,256));scores=normalise(np.array([ctx[g].mean(0) for g in patches]))@arrays['descriptors'].T;chosen=select_context_modes(scores,arrays['mode_regions'],8);rankings.append(dict(name=name,regions=[int(arrays['mode_regions'][j]) for j in chosen],scores=[float(scores[:,j].max()) for j in chosen],patch_scores=[scores[:,j].tolist() for j in chosen]))
  # Freeze actual retrieval before opening mapping-camera supervision.
  ranking_path=output/'pose_free_rankings.json';ranking_path.write_text(json.dumps(rankings,indent=2));labels=[]
  for i,r in enumerate(rankings):
   path=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/r['name'];hashes[str(path)]=file_sha256(path)
   with np.load(path) as f:pose=f['pose_w2c'];depth=f['dominant_depth']
   for rid in r['regions']:
    rows=member_groups[rid];xy=cv2.projectPoints(world[rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],c['camera_matrices'][i],np.array([c['radial_k1'][i],0.,0.,0.,0.]))[0].reshape(-1,2);z=(world[rows]@pose[:3,:3].T+pose[:3,3])[:,2];inside=(z>0)&np.isfinite(xy).all(1)&(xy>=0).all(1)&(xy<=np.array([255,143])).all(1);visible=depth_visible(xy,z,depth)
    state=visibility_label(inside,visible,texel[rows],'negative_outside_fov')
    patch_labels=[]
    for patch in range(4):
     in_patch=inside&(xy[:,0]>=128*(patch%2))&(xy[:,0]<128*(patch%2+1))&(xy[:,1]>=72*(patch//2))&(xy[:,1]<72*(patch//2+1));seen=visible&in_patch;unique_visible=len(np.unique(texel[rows[seen]]));status=visibility_label(in_patch,seen,texel[rows]);patch_labels.append(dict(patch=patch,supervision=status,visible_anchor_rows=rows[seen].tolist(),visible_distinct_cells=unique_visible))
    labels.append(dict(name=r['name'],region=rid,supervision=state,patch_labels=patch_labels,visible_anchor_rows=rows[visible].tolist(),scope='mapping clean-2DGS visibility proxy, not independent physical-identity ground truth'))
  (output/'mapping_visibility_labels.json').write_text(json.dumps(labels));(output/'sources.json').write_text(json.dumps(hashes,indent=2));split='train' if route in ['seq1','seq2','seq4','seq6'] else 'development_calibration';manifest.append(dict(route=route,split=split,query_names=c['names'].astype(str).tolist(),context=str(output/'context.npz'),rankings_sha256=file_sha256(ranking_path),labels=str(output/'mapping_visibility_labels.json')));summary[route]=dict(queries=len(rankings),contexts=len(descriptors),labels={k:sum(r['supervision']==k for r in labels) for k in ['positive_visible','negative_outside_fov','unknown']});summary[route]['patch_labels']={k:sum(v['supervision']==k for r in labels for v in r['patch_labels']) for k in ['positive_visible','negative_outside_patch','unknown']};print(route,summary[route],flush=True)
 (o/'manifest.json').write_text(json.dumps(dict(routes=manifest,test_routes_used=False,members_trained=False,independent_confirmation=False,scope='140 existing mapping queries; source-route excluded appearance, shared frozen geometry and backbone'),indent=2));(o/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
