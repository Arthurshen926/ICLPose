"""Freeze one query ROI and mapping ROIs from selected units, without query GT."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_fine_readout_scores import selection
from feature_extract.tools.vfm.fit_goal_maplet_memory_transfer import design
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def quadrant(pixels):
    p=np.asarray(pixels);return (p[...,0]>=127.5).astype(int)+2*(p[...,1]>=71.5).astype(int)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['topology','policy','base_models','contributors','source_names','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidates',nargs='+',type=Path,required=True);p.add_argument('--shape',nargs='+',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.topology) as z:t={k:z[k] for k in z.files}
    with np.load(a.policy) as z:selected=z['budget25']
    names=np.array(json.loads(a.source_names.read_text()));beta=np.array(json.loads(a.base_models.read_text())['models']['combined'])
    map_xy=np.zeros((len(selected),2));map_quad=np.full(len(selected),-1,int);records={};selected_ids=np.flatnonzero(selected)
    for s in np.unique(t['prototype_keys'][selected,1]):
        if names[s].startswith(('seq9__','seq12__','seq14__')):raise ValueError('map ROI attempted query camera')
        rows=selected_ids[t['prototype_keys'][selected,1]==s]
        with np.load(a.contributors/names[s]) as z:
            pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        xy=cv2.projectPoints(t['prototype_world'][rows],cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
        valid=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=255)&(xy[:,1]>=0)&(xy[:,1]<=143)&((t['prototype_world'][rows]@pose[:3,:3].T+pose[:3,3])[:,2]>0)
        rows=rows[valid];xy=xy[valid];map_xy[rows]=xy;map_quad[rows]=quadrant(xy)
        records[names[s]]=sorted(np.unique(map_quad[rows]).tolist())
    for candidate,shape_path in zip(a.candidates,a.shape):
        with np.load(candidate) as z:c={k:z[k] for k in z.files}
        with np.load(shape_path) as z:
            shape=z['features']
            if str(z['candidate_sha256'])!=file_sha256(candidate):raise ValueError('shape lineage differs')
        priority=design(c['association_features'],shape,True)@beta
        allowed=selection(c,priority,selected);roi_mask=np.zeros(len(allowed),bool);query_quads=[]
        for s in np.unique(c['source_image']):
            rows=np.flatnonzero((c['source_image']==s)&allowed);tok=c['query_token'][rows]
            xy=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];q=quadrant(xy)
            mass=[]
            for qi in range(4):
                tr=rows[q==qi];ut,inv=np.unique(c['query_token'][tr],return_inverse=True)
                v=np.full(len(ut),-np.inf);np.maximum.at(v,inv,priority[tr]);mass.append(float(np.sum(1/(1+np.exp(-v)))))
            chosen=int(np.argmax(mass));roi_mask[rows[q==chosen]]=True;query_quads.append([int(s),chosen])
            records[names[s]]=[chosen]
        np.savez_compressed(a.output/(candidate.stem+'.npz'),roi_mask=roi_mask,query_quadrants=np.array(query_quads),candidate_sha256=np.asarray(file_sha256(candidate)))
    np.savez_compressed(a.output/'mapping.npz',map_xy=map_xy,map_quadrant=map_quad,prototype_source=t['prototype_keys'][:,1],prototype_world=t['prototype_world'])
    (a.output/'plan.json').write_text(json.dumps({'records':[{'name':n,'quadrants':q} for n,q in sorted(records.items())],
        'native_quadrants':2,'roi_input_size':[768,432],'query_roi_budget':1,'query_GT_opened':False,
        'policy_sha256':file_sha256(a.policy),'mapping_geometry_unchanged':True},indent=2))


if __name__=='__main__':main()
