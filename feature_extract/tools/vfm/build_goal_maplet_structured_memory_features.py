"""Frozen-candidate RADIO arrangement and MoGe relative geometry; no labels read."""
import argparse,json,time
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_mapping_crossplane_retrieval import (
 _load_observation_bank,PlaneVisibilityAtlas,_load_projection,GeometryNativePlanarMap,
 _normalise,_source_view_mode_pool,_diverse_mode_indices,_records,_radio)
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors,arrangement_scores,relative_shape_features,normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def moge_tokens(path):
    with np.load(path,allow_pickle=False) as z:
        points=z['points_camera'];normals=z['normal_camera'];valid=z['valid'].astype(bool)
        meta=json.loads(str(z['metadata_json'].item()))
    if points.shape!=(144,256,3) or normals.shape!=points.shape:raise ValueError('MoGe grid differs')
    if meta.get('uses_pose') is not False or meta.get('uses_ground_truth') is not False:raise ValueError('MoGe is not pose-free')
    valid &= np.isfinite(points).all(-1)&np.isfinite(normals).all(-1)&(points[...,2]>0)
    def block(x):return x.reshape(36,4,64,4,*x.shape[2:]).transpose(0,2,1,3,*range(4,x.ndim+2)).reshape(2304,16,*x.shape[2:])
    v=block(valid);p=block(points);n=block(normals)
    weights=v/np.maximum(v.sum(1,keepdims=True),1)
    center=np.sum(np.where(v[...,None],p,0)*weights[...,None],axis=1)
    mean=np.sum(np.where(v[...,None],n,0)*weights[...,None],axis=1)
    reliable=(v.sum(1)>=12)&(np.linalg.norm(mean,axis=1)>=.8)
    return center,normalise(mean),reliable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','observation_bank','visibility_atlas','planar_map','radio_projection','radio_manifest','moge_dir','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--held_route',default='seq9')
    p.add_argument('--radius',type=int,default=2)
    p.add_argument('--topology_output',type=Path)
    p.add_argument('--descriptor_output',type=Path)
    a=p.parse_args();start=time.monotonic()
    if a.radius<1:raise ValueError('positive context radius required')
    for extra in [a.topology_output,a.descriptor_output]:
        if extra is not None and extra.exists():raise FileExistsError(extra)
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    bank,bm=_load_observation_bank(a.observation_bank);vis,vm=PlaneVisibilityAtlas.load_npz(a.visibility_atlas)
    proj,pm=_load_projection(a.radio_projection);planes=GeometryNativePlanarMap.load_npz(a.planar_map)
    if bm['visibility_atlas_content_sha256']!=vm['content_sha256'] or pm['observation_bank_content_sha256']!=bm['content_sha256']:raise ValueError('lineage differs')
    if bm['radio_manifest_file_sha256']!=file_sha256(a.radio_manifest):raise ValueError('RADIO extraction differs')
    offsets=bank['observation_offsets'];obs=np.repeat(np.arange(len(vis.view_names)),np.diff(offsets))
    op=np.repeat(np.arange(len(planes.plane_ids)),np.diff(vis.plane_offsets));plane=op[obs]
    names=vis.view_names.astype(str);source_names,source=np.unique(names,return_inverse=True);src=source[obs]
    candidate_report=json.loads(a.candidates.with_suffix('.json').read_text())
    if candidate_report['candidate_file_sha256']!=file_sha256(a.candidates):raise ValueError('candidate report differs')
    for key in ['observation_bank','visibility_atlas','planar_map','radio_projection']:
        if candidate_report['input_sha256'][key]!=file_sha256(getattr(a,key)):raise ValueError('candidate input lineage differs: '+key)
    exclusions=candidate_report.get('map_excluded_routes',[a.held_route])
    query_names=candidate_report.get('query_image_names',[r['name'] for r in candidate_report['moge_region_lineage']])
    extras=sorted(set(query_names)-set(source_names))
    source_names=np.r_[source_names,np.array(extras,dtype=str)]
    held=np.array([n.split('__')[0] in exclusions for n in names]);fit=np.flatnonzero(~held[obs])
    world=bank['world_points'].astype(float);features=np.empty((len(world),64),np.float32)
    for lo in range(0,len(world),8192):features[lo:lo+8192]=_normalise(bank['radio_features'][lo:lo+8192]@proj.T)
    uv=np.einsum('ni,nji->nj',world-planes.centers_world[plane],planes.frames_world[plane,:2])
    identities,identity=np.unique(np.c_[plane,np.floor(uv/.5)],axis=0,return_inverse=True)
    keys,mf,mw=_source_view_mode_pool(identity[fit],src[fit],world[fit],features[fit])
    selected=[]
    for ident in np.unique(keys[:,0]):
        rows=np.flatnonzero(keys[:,0]==ident)
        if len(rows)>=2:selected.extend(rows[_diverse_mode_indices(mf[rows],4)])
    selected=np.asarray(selected,int);keys=keys[selected];mw=mw[selected]
    mp=identities[keys[:,0],0].astype(int)
    if not np.array_equal(mw,c['prototype_world']) or not np.array_equal(mp,c['prototype_plane']):raise ValueError('prototype reconstruction differs')
    del features,mf
    descriptors=np.zeros((len(keys),4,64),np.float32);query={};records=_records([a.radio_manifest])
    prototype_tokens=[None]*len(keys)
    needed=sorted(set(keys[:,1])|set(c['source_image']))
    hashes={}
    for count,s in enumerate(needed):
        name=source_names[s];raw=_radio(name,records);grid=_normalise(raw@proj.T).reshape(36,64,64)
        sectors=sector_descriptors(grid,a.radius).reshape(2304,4,64)
        if s in c['source_image']:query[int(s)]=sectors
        protos=np.flatnonzero(keys[:,1]==s);br=np.flatnonzero(src==s)
        for r in protos:
            tok=bank['token_ids'][br[identity[br]==keys[r,0]]].astype(int)
            prototype_tokens[r]=np.unique(tok)
            descriptors[r]=normalise(sectors[tok].mean(axis=0))
        if (count+1)%100==0:print('appearance sources',count+1,'/',len(needed),flush=True)
    if a.topology_output is not None:
        token_offsets=np.r_[0,np.cumsum([len(t) for t in prototype_tokens])]
        np.savez_compressed(a.topology_output,prototype_keys=keys,identity_keys=identities,
            prototype_world=mw,prototype_plane=mp,source_names=source_names,
            token_offsets=token_offsets,token_ids=np.concatenate(prototype_tokens),
            candidate_sha256=np.asarray(file_sha256(a.candidates)))
    if a.descriptor_output is not None:
        np.savez_compressed(a.descriptor_output,descriptors=descriptors.astype(np.float16),
            radius=np.asarray(a.radius),candidate_sha256=np.asarray(file_sha256(a.candidates)))
    out=np.zeros((len(c['query_token']),6),np.float32)
    for count,s in enumerate(np.unique(c['source_image'])):
        rows=np.flatnonzero(c['source_image']==s);tok=c['query_token'][rows];proto=c['prototype_rows'][rows]
        out[rows,:3]=arrangement_scores(query[int(s)][tok],descriptors[proto])
        kept=rows[c['homography_keep'][rows]];path=a.moge_dir/source_names[s]
        qp,qn,qv=moge_tokens(path);hashes[str(path)]=file_sha256(path)
        proto=c['prototype_rows'][kept]
        out[kept,3:]=relative_shape_features(c['query_token'][kept],qp,qn,qv,mw[proto],planes.normals_world[mp[proto]],c['association_features'][kept,0])
        if (count+1)%10==0:print('relative shape images',count+1,flush=True)
    names_out=['unordered_token_context','ordered_token_context','horizontal_arrangement_contrast','relative_normal_agreement','relative_incidence_agreement','relative_shape_available_fraction']
    np.savez_compressed(a.output,features=out,feature_names=np.array(names_out),candidate_sha256=np.asarray(file_sha256(a.candidates)),source_names=source_names)
    report={'scope':__doc__,'candidate_sha256':file_sha256(a.candidates),'feature_sha256':file_sha256(a.output),'feature_names':names_out,'source_names':source_names.tolist(),
            'radius':a.radius,'neighbours':8,'prototype_count':len(descriptors),'additional_prototype_descriptor_bytes_float32':descriptors.nbytes,
            'seconds':time.monotonic()-start,'labels_opened':False,'candidate_and_coordinate_arrays_changed':False,'map_excluded_routes':exclusions,
            'moge_sha256':hashes,'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['observation_bank','visibility_atlas','planar_map','radio_projection','radio_manifest']}}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['source_names','moge_sha256']},indent=2))


if __name__=='__main__':main()
