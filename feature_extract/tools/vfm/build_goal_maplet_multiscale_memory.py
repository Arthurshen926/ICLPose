"""Finite, metric-anchored memory choices with explicit cross-surface references.

Context windows are alternative visual units, not merged geometric planes.
All alternatives have four descriptors and at most sixteen geometry references.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import (
 _load_observation_bank,PlaneVisibilityAtlas,GeometryNativePlanarMap,_load_projection,_records,_radio,_normalise)
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors,normalise,arrangement_scores
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['topology','observation_bank','visibility_atlas','planar_map','radio_projection','radio_manifest','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidates',nargs='+',type=Path,required=True)
    p.add_argument('--source_names',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with np.load(a.topology) as z:t={k:z[k] for k in z.files}
    names=np.asarray(json.loads(a.source_names.read_text()));keys=t['prototype_keys'];n=len(keys);radii=[1,2,4]
    if not np.array_equal(names[:len(t['source_names'])],t['source_names']):raise ValueError('source indices differ')
    bank,_=_load_observation_bank(a.observation_bank);vis,_=PlaneVisibilityAtlas.load_npz(a.visibility_atlas)
    planes=GeometryNativePlanarMap.load_npz(a.planar_map);proj,_=_load_projection(a.radio_projection)
    obs=np.repeat(np.arange(len(vis.view_names)),np.diff(bank['observation_offsets']))
    lookup={name:i for i,name in enumerate(names)};src=np.array([lookup[str(name)] for name in vis.view_names])[obs]
    plane=np.repeat(np.arange(len(planes.plane_ids)),np.diff(vis.plane_offsets))[obs]
    bt=bank['token_ids'];world=bank['world_points'];records=_records([a.radio_manifest])
    descriptors=np.zeros((n,3,4,64),np.float16);references=np.full((n,3,16),-1,np.int64)
    attrs=np.zeros((n,3,8),np.float32)
    for count,s in enumerate(np.unique(keys[:,1])):
        raw=_radio(names[s],records);grid=_normalise(raw@proj.T).reshape(36,64,64)
        sectors=[sector_descriptors(grid,r).reshape(2304,4,64) for r in radii]
        br=np.flatnonzero(src==s);xy=np.c_[bt[br]%64,bt[br]//64];tree=cKDTree(xy)
        for j in np.flatnonzero(keys[:,1]==s):
            tok=t['token_ids'][t['token_offsets'][j]:t['token_offsets'][j+1]]
            center=np.c_[tok%64,tok//64].mean(0)
            for k,r in enumerate(radii):
                descriptors[j,k]=normalise(sectors[k][tok].mean(0))
                near=np.unique(np.concatenate([tree.query_ball_point([q%64,q//64],r,p=np.inf) for q in tok])).astype(int)
                rows=br[near]
                if len(rows):
                    # Unique physical rows; deterministic near-first cap. References
                    # retain original plane/world identities and never vote twice.
                    order=np.lexsort((rows,np.sum((xy[near]-center)**2,axis=1)))
                    chosen=rows[order[:16]];references[j,k,:len(chosen)]=chosen
                    normal=planes.normals_world[plane[rows]]
                    span=1-float(np.linalg.norm(normal.mean(0)))
                    extent=float(np.linalg.norm(np.std(world[rows],axis=0)))
                    attrs[j,k,:6]=[np.log1p(len(np.unique(bt[rows]))),np.log1p(len(np.unique(plane[rows]))),span,np.log1p(extent),np.log1p(len(tok)),r/4]
        if (count+1)%100==0:print('map sources',count+1,flush=True)
    # Cross-view stability, computed only from map appearance modes.
    for ident in np.unique(keys[:,0]):
        rows=np.flatnonzero(keys[:,0]==ident)
        d=descriptors[rows].astype(np.float32)
        avg=normalise(d.mean(0))
        attrs[rows,:,6]=np.mean(np.sum(d*avg,axis=-1),axis=-1)
        attrs[rows,:,7]=np.log1p(len(rows))
    np.savez_compressed(a.output/'map_choices.npz',descriptors=descriptors,reference_bank_rows=references,
        attributes=attrs,radii=np.array(radii),prototype_world=t['prototype_world'],prototype_plane=t['prototype_plane'],
        prototype_keys=keys,source_names=names,geometry_bank_sha256=np.asarray(file_sha256(a.observation_bank)))
    del bank,world,obs,src,bt
    reports=[]
    for candidate in a.candidates:
        with np.load(candidate) as z:c={k:z[k] for k in z.files}
        if not np.array_equal(c['prototype_world'],t['prototype_world']):raise ValueError('candidate geometry differs')
        out=np.zeros((len(c['query_token']),3,3),np.float32)
        for s in np.unique(c['source_image']):
            grid=_normalise(_radio(names[s],records)@proj.T).reshape(36,64,64)
            rows=np.flatnonzero(c['source_image']==s);tok=c['query_token'][rows];pr=c['prototype_rows'][rows]
            for k,r in enumerate(radii):
                qs=sector_descriptors(grid,r).reshape(2304,4,64)
                out[rows,k]=arrangement_scores(qs[tok],descriptors[pr,k].astype(np.float32))
        target=a.output/(candidate.stem+'.npz')
        np.savez_compressed(target,features=out,source_names=names,candidate_sha256=np.asarray(file_sha256(candidate)))
        reports.append({'candidates':str(candidate),'features':str(target),'rows':len(out)})
    (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'radii':radii,'prototypes':n,
        'candidate_choice_count':n*3,'fixed_descriptor_bytes_per_unit':4*64*2,'fixed_reference_bytes_per_unit':16*8,
        'all_choice_uncompressed_bytes':descriptors.nbytes+references.nbytes+attrs.nbytes,
        'one_choice_uncompressed_bytes':descriptors.nbytes//3+references.nbytes//3+n,
        'cross_surface_choice_fraction':np.mean(attrs[:,:,1]>np.log(2)+1e-5,axis=0).tolist(),
        'mapping_geometry_changed':False,'labels_or_poses_opened':False,'reports':reports,'seconds':time.monotonic()-start,
        'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['topology','observation_bank','visibility_atlas','planar_map','radio_projection','radio_manifest']}},indent=2))


if __name__=='__main__':main()
