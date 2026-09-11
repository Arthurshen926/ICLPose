"""Mapping-only precision utility: metric-valid probability times subpixel utility.

Fixed 1.5px Gaussian target, no threshold sweep and no test-route training.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_memory_transfer import design
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit,token_image_weights
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','features','labels','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.features) as z:
        f=z['features'];names=z['source_names'].astype(str)
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('feature lineage differs')
    with np.load(a.labels) as z:
        y=z['labels']
        if str(z['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('label lineage differs')
    sources=c['source_image'];tokens=c['query_token'];images=np.unique(sources);keep=c['homography_keep'];target=np.zeros(len(y))
    if not all(names[s].startswith('seq9__') for s in images):raise ValueError('quality fitting is seq9-only')
    for s in images:
        rows=np.flatnonzero(sources==s);world=c['prototype_world'][c['prototype_rows'][rows]]
        with np.load(a.contributors/names[s]) as z:
            gt=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        xy=np.c_[(tokens[rows]%64)*4+1.5,(tokens[rows]//64)*4+1.5]
        projected,_=cv2.projectPoints(world,cv2.Rodrigues(gt[:3,:3])[0],gt[:3,3],K,np.array([k1,0.,0.,0.,0.]))
        error=np.linalg.norm(projected.reshape(-1,2)-xy,axis=1);front=(world@gt[:3,:3].T+gt[:3,3])[:,2]>0
        target[rows]=np.where(front&(y[rows]==1),np.exp(-.5*(error/1.5)**2),0.)
    models={};out=np.zeros((len(y),2),np.float32);reports={}
    for col,name in enumerate(['quality_base','quality_combined']):
        x=design(c['association_features'],f,'combined' in name);valid=keep&(y>=0)
        models[name]=fit(x[valid],target[valid],token_image_weights(sources[valid],tokens[valid])).tolist()
        for fold in [0,1]:
            tr=np.isin(sources,images[fold::2])&valid;ev=~np.isin(sources,images[fold::2])
            beta=fit(x[tr],target[tr],token_image_weights(sources[tr],tokens[tr]));out[ev,col]=x[ev]@beta
        prob=1/(1+np.exp(-out[valid,col]));reports[name]={'crossfit_soft_target_mse':float(np.mean((prob-target[valid])**2))}
    np.savez_compressed(a.output.with_suffix('.npz'),logits=out,arm_names=np.array(list(models)),candidate_sha256=np.asarray(file_sha256(a.candidates)))
    model={'scope':__doc__,'models':models,'training_images':names[images].tolist(),'target_sigma_pixels':1.5,'uses_mapping_pose_for_supervision_only':True,
           'training_route':'seq9','reports':reports,'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['candidates','features','labels']},
           'not_a_calibrated_match_probability':True,'evaluation_route_labels_used':False}
    a.output.write_text(json.dumps(model,indent=2))


if __name__=='__main__':main()
