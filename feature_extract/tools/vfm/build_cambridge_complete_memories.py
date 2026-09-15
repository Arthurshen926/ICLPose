"""Build both original source-mode and native region memories from scene assets."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import (
    _load_observation_bank,PlaneVisibilityAtlas,_load_projection,GeometryNativePlanarMap,
    _normalise,_source_view_mode_pool,_diverse_mode_indices,_records,_radio)
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.build_goal_maplet_native_region_training_inventory import write
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise,sector_descriptors
from feature_extract.vfm.localization_goal_maplet.metric_region_memory import build_regions
from feature_extract.vfm.localization_goal_maplet.mapping_image_partition import image_groups
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--atlas-dir',type=Path,required=True);p.add_argument('--radio-manifest',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--mapping-image-partition',type=Path);a=p.parse_args();b=a.base;ad=a.atlas_dir
    bank,bm=_load_observation_bank(ad/'observations.npz');vis,vm=PlaneVisibilityAtlas.load_npz(ad/'visibility.npz');planes=GeometryNativePlanarMap.load_npz(ad/'planes.npz');proj,pm=_load_projection(ad/'projection64.npz')
    if pm['observation_bank_content_sha256']!=bm['content_sha256'] or bm['visibility_atlas_content_sha256']!=vm['content_sha256']:raise ValueError('Memory training lineage differs')
    names,source=np.unique(vis.view_names.astype(str),return_inverse=True);obs=np.repeat(np.arange(len(vis.view_names)),np.diff(bank['observation_offsets']));src=source[obs];plane=np.repeat(np.arange(len(planes.plane_ids)),np.diff(vis.plane_offsets))[obs]
    if a.mapping_image_partition:
        group=image_groups(vis.view_names.astype(str),a.mapping_image_partition);fit=np.flatnonzero(group[obs]=='mapping_images_fit')
    else:
        route=np.array([n.split('__')[0] for n in vis.view_names.astype(str)]);fit=np.flatnonzero(route[obs]!=pm['validation_mapping_route'])
    world=bank['world_points'].astype(float);features=np.empty((len(world),64),np.float32)
    for lo in range(0,len(world),8192):features[lo:lo+8192]=_normalise(bank['radio_features'][lo:lo+8192]@proj.T)
    uv=np.einsum('ni,nji->nj',world-planes.centers_world[plane],planes.frames_world[plane,:2]);identities,identity=np.unique(np.c_[plane,np.floor(uv/.5)],axis=0,return_inverse=True)
    keys,mf,mw=_source_view_mode_pool(identity[fit],src[fit],world[fit],features[fit]);selected=[]
    for ident in np.unique(keys[:,0]):
        rows=np.flatnonzero(keys[:,0]==ident)
        if len(rows)>=2:selected.extend(rows[_diverse_mode_indices(mf[rows],4)])
    selected=np.asarray(selected,int);keys=keys[selected];mw=mw[selected];mp=identities[keys[:,0],0].astype(int);del features,mf
    descriptors=np.zeros((len(keys),4,64),np.float32);tokens=[None]*len(keys);records=_records([a.radio_manifest])
    for j,s in enumerate(np.unique(keys[:,1])):
        sectors=sector_descriptors(_normalise(_radio(names[s],records)@proj.T).reshape(36,64,64),1).reshape(2304,4,64);br=np.flatnonzero(src==s)
        for row in np.flatnonzero(keys[:,1]==s):
            t=bank['token_ids'][br[identity[br]==keys[row,0]]].astype(int);tokens[row]=np.unique(t);descriptors[row]=normalise(sectors[t].mean(0))
        if (j+1)%100==0:print('source contexts',j+1,flush=True)
    old=b/'adaptive_memory_v234';old.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(old/'topology.npz',prototype_keys=keys,identity_keys=identities,prototype_world=mw,prototype_plane=mp,source_names=names,token_offsets=np.r_[0,np.cumsum([len(t) for t in tokens])],token_ids=np.concatenate(tokens))
    desc=normalise(descriptors.astype(np.float16).astype(np.float32).reshape(-1,256));centers,_=build_regions(mw,6.)
    fixed_groups=cKDTree(mw).query_ball_point(centers,6.)
    fixed=b/'full_pool_boundaries_v245/maps/fixed.npz';fixed.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fixed,centers=centers,radii=np.full(len(centers),6.),prototype_rows=np.concatenate(fixed_groups),offsets=np.r_[0,np.cumsum([len(g) for g in fixed_groups])],world_sha256=np.asarray(hashlib.sha256(mw.tobytes()).hexdigest()),training_images=names)
    modes=[];ids=[];members=[]
    for rid,g in enumerate(fixed_groups):
        g=np.asarray(g,int);sg=[g[keys[g,1]==s] for s in np.unique(keys[g,1])]
        for j in np.argsort([-len(np.unique(keys[v,0])) for v in sg],kind='stable')[:4]:
            if len(sg[j])>=6:modes.append(normalise(desc[sg[j]].mean(0)));ids.append(rid);members.append(sg[j])
    if not modes:raise ValueError('No valid source-mode regions')
    (b/'region_frontend_v246').mkdir(exist_ok=True)
    np.savez_compressed(b/'region_frontend_v246/map.npz',descriptors=np.asarray(modes,np.float32),mode_regions=np.asarray(ids),member_rows=np.concatenate(members),offsets=np.r_[0,np.cumsum([len(g) for g in members])],centers=centers)
    native=b/'native_fine_v264';atlas=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz'
    with np.load(atlas) as z:world=z['world_points']
    with np.load(native/'mapping_lineage.npz') as z:names=z['source_names'].astype(str);source=z['prototype_source_and_cell'][:,0]
    with np.load(native/'readout/map.npz') as z:available=z['available'];fm=json.loads(str(z['metadata_json']))
    contexts=np.zeros((len(world),256),np.float16)
    for j,s in enumerate(np.unique(source)):
        rows=np.flatnonzero(source==s);grids,ck=load_grids(old/'fine_cache'/names[s],fm['projection_sha256'])
        if ck!=fm['checkpoint_sha256']:raise ValueError('Native feature checkpoint differs')
        context=normalise(sector_descriptors(grids[0],1).reshape(36,64,256))
        with np.load(a.contributors/names[s]) as z:pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        xy=cv2.projectPoints(world[rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
        contexts[rows]=np.where(available[rows,None],sample_grid(context,np.nan_to_num(xy)),0.)
    cache=b/'native_region_training_v276/offline_native_contexts.npz';cache.parent.mkdir(exist_ok=True);np.savez_compressed(cache,context=contexts,native_source=source,source_names=names,offline_only=np.array(True))
    modes=[];ids=[]
    for rid,g in enumerate(cKDTree(world).query_ball_point(centers,6.)):
        g=np.asarray(g,int);g=g[available[g]];sg=[g[source[g]==s] for s in np.unique(source[g])]
        for j in np.argsort([-len(v) for v in sg],kind='stable')[:4]:
            if len(sg[j])>=6:modes.append(normalise(contexts[sg[j]].astype(np.float32).mean(0)));ids.append(rid)
    (b/'native_context_v278').mkdir(exist_ok=True)
    write(b/'native_context_v278/map.npz',dict(descriptors=np.asarray(modes,np.float32),mode_regions=np.asarray(ids,int),centers=centers),dict(artifact_type='goal_maplet_native_full_context_v1',native_atlas_sha256=file_sha256(atlas),excluded_mapping_route=None,query_pose_or_ground_truth_used_for_retrieval=False,offline_mapping_source_names=names[np.unique(source)].tolist(),context_cache_sha256=file_sha256(cache),geometry_readout_unchanged=True))
    print('both original region memories built',flush=True)


if __name__=='__main__':main()
