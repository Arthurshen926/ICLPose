"""Matched-coordinate coarse/intermediate/fine RADIO candidate readout.

Only mapping-camera poses project fixed anchors into their source views.
Query pose/depth/labels are never accessed. A coarse interpolated search is
retained as the control for actual finer RGB observation.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def sample_grid(grid,pixels):
    """Bilinear readout under the same pixel-center contract at any grid size."""
    h,w,d=grid.shape;p=np.asarray(pixels,float)
    x=(p[...,0]+.5)*w/256-.5;y=(p[...,1]+.5)*h/144-.5
    x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    wx=(x-x0)[...,None];wy=(y-y0)[...,None]
    return normalise((1-wx)*(1-wy)*grid[y0,x0]+wx*(1-wy)*grid[y0,x1]+(1-wx)*wy*grid[y1,x0]+wx*wy*grid[y1,x1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['topology','cache','contributors','source_names','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidates',nargs='+',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.topology) as z:t={k:z[k] for k in z.files}
    names=np.asarray(json.loads(a.source_names.read_text()));keys=t['prototype_keys'];world=t['prototype_world'];n=len(world)
    descriptors=[np.zeros((n,d),np.float32) for d in [64,128,64]];available=np.zeros(n,bool)
    cache_keys=['coarse_final','coarse_intermediate','fine_final'];hashes={};map_pose_hashes={}
    started=time.monotonic()
    def grids(s):
        path=a.cache/names[s]
        with np.load(path) as z:
            meta=json.loads(str(z['metadata_json']))
            if meta['poses_or_labels_opened'] is not False or meta['format']!='goal_fine_radio_v2':raise ValueError('unsafe/incomplete fine cache')
            value=[z[k].astype(np.float32) for k in cache_keys]
        hashes[str(path)]=file_sha256(path)
        return value
    for count,s in enumerate(np.unique(keys[:,1])):
        if names[s].startswith(('seq9__','seq12__','seq14__')):raise ValueError('query camera requested for map construction')
        rows=np.flatnonzero(keys[:,1]==s);path=a.contributors/names[s]
        with np.load(path) as z:
            pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        xy=cv2.projectPoints(world[rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
        valid=(world[rows]@pose[:3,:3].T+pose[:3,3])[:,2]>0
        valid &= np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=255)&(xy[:,1]>=0)&(xy[:,1]<=143)
        available[rows]=valid
        for d,g in zip(descriptors,grids(s)):d[rows]=np.where(valid[:,None],sample_grid(g,np.nan_to_num(xy)),0)
        map_pose_hashes[str(path)]=file_sha256(path)
        if (count+1)%100==0:print('map fine',count+1,flush=True)
    np.savez_compressed(a.output/'map_readouts.npz',coarse=descriptors[0].astype(np.float16),intermediate=descriptors[1].astype(np.float16),fine=descriptors[2].astype(np.float16),available=available,
        prototype_world=world,prototype_plane=t['prototype_plane'])
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
    reports=[]
    for candidate in a.candidates:
        with np.load(candidate) as z:c={k:z[k] for k in z.files}
        if not np.array_equal(c['prototype_world'],world):raise ValueError('candidate geometry differs')
        features=np.zeros((len(c['query_token']),7),np.float32);pixels=np.zeros((len(features),2,2),np.float32)
        for s in np.unique(c['source_image']):
            rows=np.flatnonzero(c['source_image']==s);tok=c['query_token'][rows];pr=c['prototype_rows'][rows]
            xy=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];gs=grids(s)
            for k,(g,d) in enumerate(zip(gs,descriptors)):
                features[rows,k]=np.sum(sample_grid(g,xy)*d[pr],axis=1)
                if k in [0,2]:
                    probes=xy[:,None,:]+offsets[None]
                    sim=np.sum(sample_grid(g,probes)*d[pr,None],axis=-1)
                    # Center wins exact ties, preventing arbitrary flat-feature drift.
                    sim[:,4]+=1e-7;best=np.argmax(sim,axis=1)
                    col=0 if k==0 else 1
                    pixels[rows,col]=probes[np.arange(len(rows)),best]
                    features[rows,3+col]=sim[np.arange(len(rows)),best]-features[rows,k]
                    features[rows,5+col]=available[pr]
        target=a.output/(candidate.stem+'.npz')
        np.savez_compressed(target,features=features,refined_pixels=pixels,source_names=names,candidate_sha256=np.asarray(file_sha256(candidate)))
        reports.append({'candidate':str(candidate),'rows':len(features),'artifact':str(target)})
    (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'reports':reports,'map_available_fraction':float(available.mean()),
        'search_offsets_pixels_256x144':offsets.tolist(),'query_labels_or_poses_opened':False,
        'map_readout_bytes_float16':[d.size*2 for d in descriptors],'seconds':time.monotonic()-started,
        'feature_extraction_is_full_image_not_roi':True,'cache_sha256':hashes,'map_camera_sha256':map_pose_hashes},indent=2))


if __name__=='__main__':main()
