"""Bake anonymous directional feature modes at identical fixed world anchors."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256

def depth_visible(xy,z,depth):
    valid=np.isfinite(xy).all(1)&np.isfinite(z)&(z>0)&(xy[:,0]>=0)&(xy[:,0]<=255)&(xy[:,1]>=0)&(xy[:,1]<=143)
    ij=np.rint(np.nan_to_num(xy)).astype(int);ij[:,0]=np.clip(ij[:,0],0,255);ij[:,1]=np.clip(ij[:,1],0,143)
    d=depth[ij[:,1],ij[:,0]]
    return valid&np.isfinite(d)&(d>0)&(np.abs(z-d)<=np.maximum(.1,.01*d))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;a.output.mkdir(parents=True,exist_ok=True)
    mp=b/'native_fine_v264/readout/map.npz'
    if (a.output/'map.npz').exists():raise FileExistsError(a.output)
    with np.load(mp) as f:arr={k:f[k] for k in f.files if k!='metadata_json'};meta=json.loads(f['metadata_json'].item())
    assert arrays_sha256(arr)==meta['arrays_sha256']
    lp=b/'native_fine_v264/mapping_lineage.npz'
    with np.load(lp) as f:names=f['source_names'].astype(str);owner=f['prototype_source_and_cell'][:,0]
    assert not any(n.startswith(('seq10__','seq13__')) for n in names)
    root=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean');poses=[];cameras=[];hashes={str(mp):file_sha256(mp),str(lp):file_sha256(lp)}
    for name in names:
        path=root/name
        with np.load(path) as f:
            pose=f['pose_w2c'];poses.append(pose);cameras.append(_scaled_intrinsics(int(f['camera_model_id']),f['camera_params'],int(f['camera_width']),int(f['camera_height'])))
        hashes[str(path)]=file_sha256(path)
    poses=np.array(poses);centers=-np.einsum('nji,nj->ni',poses[:,:3,:3],poses[:,:3,3]);neighbors=cKDTree(centers).query(centers,k=9)[1]
    world=arr['world_points'];sums=np.stack([arr['coarse'],arr['fine']]).astype(np.float32);sums/=np.maximum(np.linalg.norm(sums,axis=-1,keepdims=True),1e-12);count=arr['available'].astype(int)
    # Fixed camera-index order and at most four additional visible observations.
    modes=np.zeros((len(world),5,64),np.float16);modes[:,0]=arr['fine'];directions=np.zeros((len(world),5,3),np.float32);v=centers[owner]-world;directions[:,0]=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    requests={}
    for src in np.unique(owner):
        rows=np.flatnonzero((owner==src)&arr['available'])
        for other in neighbors[src]:
            if other!=src:requests.setdefault(int(other),[]).append(rows)
    for j,src in enumerate(sorted(requests)):
        rows=np.concatenate(requests[src]);rows=rows[count[rows]<5];pose=poses[src];K,k1=cameras[src]
        xy=cv2.projectPoints(world[rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2) if len(rows) else np.empty((0,2))
        z=(world[rows]@pose[:3,:3].T+pose[:3,3])[:,2]
        with np.load(root/names[src]) as f:valid=depth_visible(xy,z,f['dominant_depth'])
        rows=rows[valid];xy=xy[valid];cp=b/'adaptive_memory_v234/fine_cache'/names[src]
        if len(rows) and cp.exists():
            grids,ck=load_grids(cp,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];hashes[str(cp)]=file_sha256(cp)
            for level,grid in enumerate(grids):
                sampled=sample_grid(grid,xy);sums[level,rows]+=sampled
                if level==1:modes[rows,count[rows]]=sampled
            v=centers[src]-world[rows];directions[rows,count[rows]]=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            count[rows]+=1
        if (j+1)%100==0:print('mapping views',j+1,flush=True)
    sums/=np.maximum(np.linalg.norm(sums,axis=-1,keepdims=True),1e-12);arr['coarse']=sums[0].astype(np.float16);arr['fine']=sums[1].astype(np.float16)
    arr.update(fine_modes=modes,view_directions=directions,mode_count=count)
    audit=a.output/'offline_sources.json';audit.write_text(json.dumps(hashes,indent=2,sort_keys=True))
    meta.pop('content_sha256');meta.update(artifact_type='goal_maplet_directional_fixed_anchor_map_v1',arrays_sha256=arrays_sha256(arr),offline_source_audit_sha256=file_sha256(audit),aggregation='normalized mean including original; up to four additional visible observations; eight nearest source-camera neighbors',visibility='declared clean 2DGS dominant depth; nearest pixel; absolute tolerance max(0.1m,1 percent depth)',builder_sha256=file_sha256(Path(__file__)))
    meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(a.output/'map.npz',**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
    (a.output/'coverage.json').write_text(json.dumps(dict(anchors=len(count),additional_view_fraction=float(np.mean(count>1)),mean_observations=float(count.mean()),query_ground_truth_read=False),indent=2))
if __name__=='__main__':main()
