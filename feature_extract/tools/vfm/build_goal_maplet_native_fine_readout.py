"""Matched native-atlas coarse/fine reads with fixed anchors and pose initializers.

Offline map cameras are recovered from exact atlas construction lineage. Query
poses are frozen predictions, never ground truth. Local joint alignment uses
one image translation per small same-plane/query-region block, not a global
homography across a multi-plane region.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import _load_frozen_poses as _load_pose_candidate
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def select_offsets(similarity, groups, joint=False):
    scores=np.asarray(similarity,float)
    if scores.ndim!=2 or scores.shape[1]!=9 or not np.isfinite(scores[:,4]).all():raise ValueError('invalid fine search scores')
    def best(v):
        maximum=v.max()
        return 4 if v[4]==maximum else int(np.argmax(v))
    if not joint:return np.array([best(v) for v in scores],int)
    out=np.full(len(scores),4,int)
    for group in groups:
        if len(group)>=3:out[group]=best(scores[group].sum(0))
    return out


def load_grids(path, projection_sha):
    with np.load(path) as z:
        m=json.loads(str(z['metadata_json']))
        if m.get('poses_or_labels_opened') is not False or m.get('format')!='goal_fine_radio_v2' or m.get('final_projection_sha256')!=projection_sha:raise ValueError('fine cache contract differs')
        if m.get('coarse_rgb_size')!=[1024,576] or m.get('fine_rgb_size')!=[1536,864]:raise ValueError('fine spatial contract differs')
        return [z[k].astype(np.float32) for k in ['coarse_final','fine_final']],m['checkpoint_sha256']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['map','query'],required=True)
    a=p.parse_args();b=a.base;root=a.output;root.mkdir(parents=True,exist_ok=True);native=b/'native_fine_v264';atlaspath=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';projsha=file_sha256(b/'stmarys_chart_local_radio_projection_64d_v2.npz')
    with np.load(atlaspath) as z:world=z['world_points'];plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
    if a.phase=='map':
        out=root/'map.npz'
        if out.exists():raise FileExistsError(out)
        with np.load(native/'reconstructed_atlas.npz') as z,np.load(atlaspath) as old:
            for k in old.files:
                if k!='metadata_json' and not np.array_equal(old[k],z[k]):raise ValueError('native atlas reconstruction differs')
        with np.load(native/'mapping_lineage.npz') as z:names=z['source_names'].astype(str);lineage=z['prototype_source_and_cell'];assert bool(z['offline_only'])
        if any(n.startswith(('seq10__','seq13__')) for n in names):raise ValueError('map/query source overlap')
        desc=np.zeros((2,len(world),64),np.float16);available=np.zeros(len(world),bool);hashes={};checkpoint=None
        for j,src in enumerate(np.unique(lineage[:,0])):
            name=names[src];rows=np.flatnonzero(lineage[:,0]==src);path=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/name
            with np.load(path) as z:pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
            xy=cv2.projectPoints(world[rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2);camera=world[rows]@pose[:3,:3].T+pose[:3,3];valid=(camera[:,2]>0)&np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=255)&(xy[:,1]>=0)&(xy[:,1]<=143);available[rows]=valid
            cp=b/'adaptive_memory_v234/fine_cache'/name;grids,ck=load_grids(cp,projsha)
            if checkpoint is not None and ck!=checkpoint:raise ValueError('checkpoint differs')
            checkpoint=ck
            for d,g in zip(desc,grids):d[rows]=np.where(valid[:,None],sample_grid(g,np.nan_to_num(xy)),0)
            hashes[str(path)]=file_sha256(path);hashes[str(cp)]=file_sha256(cp)
            if (j+1)%100==0:print('map',j+1,flush=True)
        arrays=dict(world_points=world,coarse=desc[0],fine=desc[1],available=available)
        source_audit=root/'offline_map_sources.json'
        source_audit.write_text(json.dumps(hashes,sort_keys=True,indent=2))
        meta=dict(artifact_type='goal_maplet_native_fine_map_v1',arrays_sha256=arrays_sha256(arrays),atlas_sha256=file_sha256(atlaspath),lineage_sha256=file_sha256(native/'mapping_lineage.npz'),checkpoint_sha256=checkpoint,projection_sha256=projsha,query_ground_truth_read=False,offline_source_audit_sha256=file_sha256(source_audit),source_view_identity_retained_at_runtime=False)
        meta['content_sha256']=canonical_json_sha256(meta)
        np.savez_compressed(out,**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)));return
    with np.load(root/'map.npz') as z:m=json.loads(str(z['metadata_json']));maps={k:z[k] for k in z.files if k!='metadata_json'}
    if arrays_sha256(maps)!=m['arrays_sha256'] or not np.array_equal(maps['world_points'],world) or m['atlas_sha256']!=file_sha256(atlaspath):raise ValueError('fine map binding differs')
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
    protocol=dict(map_sha256=file_sha256(root/'map.npz'),maximum_tokens=128,patch='same physical plane, query plane, and 8x8 coarse-token image block; at least 3 distinct tokens',mask='smallest frozen MoGe pose residual per token; uniformly spaced token ids up to128; map available; residual<=4px',arms=['coarse','fine','joint'],anchors_fixed=True,uncertainty='original frozen variances retained; not recalibrated for fine updates',query_ground_truth_read=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for split in ['seq10','shard0','shard1','shard2','shard3']:
        corrpath=b/'native_scope_v254/wide'/f'{split}_appearance.npz';corr,meta=_load(corrpath);initialpath=b/'native_scope_mainline_v255'/(split+'_wide')/'moge.npz';poses,_=_load_pose_candidate(initialpath)
        if not np.array_equal(corr['names'],poses['names']):raise ValueError('query order differs')
        if not np.array_equal(corr['world_points'],world[corr['prototype_atlas_row']]):raise ValueError('native correspondence geometry differs from atlas')
        variants={k:{f:v.copy() for f,v in corr.items()} for k in protocol['arms']};audit=[]
        for i,name in enumerate(corr['names'].astype(str)):
            l,h=map(int,corr['correspondence_offsets'][i:i+2]);tok=corr['query_tokens'][l:h];pr=corr['prototype_atlas_row'][l:h];xy=corr['query_measurements_xy'][l:h];pose=poses['pose_w2c'][i]
            if not poses['usable'][i]:audit.append(dict(name=name,eligible_tokens=0));continue
            K=corr['camera_matrices'][i];k1=float(corr['radial_k1'][i]);cam=world[pr]@pose[:3,:3].T+pose[:3,3];project=cv2.projectPoints(world[pr],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2);res=np.linalg.norm(project-xy,axis=1);valid=(cam[:,2]>0)&np.isfinite(res)&(res<=4)&maps['available'][pr]
            rows=np.flatnonzero(valid);order=np.lexsort((rows,res[rows],tok[rows]));rows=rows[order];_,first=np.unique(tok[rows],return_index=True);rows=rows[first]
            if len(rows)>128:rows=rows[np.linspace(0,len(rows)-1,128,dtype=int)]
            provenance=corr['provenance_region_plane_atlas_row'][l:h]
            keys=np.c_[plane[pr[rows]],provenance[rows,0],tok[rows]//64//8,tok[rows]%64//8];_,labels=np.unique(keys,axis=0,return_inverse=True);groups=[np.flatnonzero(labels==v) for v in np.unique(labels)];eligible=np.concatenate([g for g in groups if len(g)>=3]) if any(len(g)>=3 for g in groups) else np.array([],int);rows=rows[np.sort(eligible)]
            if not len(rows):audit.append(dict(name=name,eligible_tokens=0));continue
            keys=np.c_[plane[pr[rows]],provenance[rows,0],tok[rows]//64//8,tok[rows]%64//8];_,labels=np.unique(keys,axis=0,return_inverse=True);groups=[np.flatnonzero(labels==v) for v in np.unique(labels)]
            cp=native/'query_cache'/name;grids,ck=load_grids(cp,projsha)
            if ck!=m['checkpoint_sha256']:raise ValueError('query checkpoint differs')
            probes=xy[rows,None]+offsets[None];center=np.c_[(tok[rows]%64)*4+1.5,(tok[rows]//64)*4+1.5];inside=(np.abs(probes-center[:,None])<=2+1e-8).all(2)
            sim=[np.sum(sample_grid(g,probes)*maps[k][pr[rows],None].astype(np.float32),axis=-1) for g,k in zip(grids,['coarse','fine'])]
            for v in sim:v[~inside]=-np.inf
            updates={}
            for arm,v,joint in [('coarse',sim[0],False),('fine',sim[1],False),('joint',sim[1],True)]:
                best=select_offsets(v,groups,joint);variants[arm]['query_measurements_xy'][l+rows]=probes[np.arange(len(rows)),best];updates[arm]=float(np.mean(best!=4))
            audit.append(dict(name=name,eligible_tokens=len(rows),changed_fraction=updates,query_cache_sha256=file_sha256(cp)))
        for arm,arrays in variants.items():
            out=root/f'{split}_{arm}.npz'
            if out.exists():raise FileExistsError(out)
            outmeta=dict(meta);outmeta.pop('content_sha256',None);outmeta.update(arrays_sha256=arrays_sha256(arrays),fine_readout_protocol=protocol,fine_readout_arm=arm,coarse_correspondence_file_sha256=file_sha256(corrpath),frozen_initial_pose_sha256=file_sha256(initialpath),query_measurement_semantics='native_matched_fine_query_pixel_update_inside_original_4x4_token',fine_measurement_variance_recalibrated=False);outmeta['content_sha256']=canonical_json_sha256(outmeta)
            temporary=out.with_name(out.name+'.temporary.npz')
            np.savez_compressed(temporary,**arrays,metadata_json=np.array(json.dumps(outmeta,sort_keys=True)));_load(temporary);temporary.replace(out)
        (root/f'{split}_audit.json').write_text(json.dumps(audit,indent=2));print(split,'built',flush=True)


if __name__=='__main__':main()
