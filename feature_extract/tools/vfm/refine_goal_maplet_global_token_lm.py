"""Expand a frozen regional initializer to all consistent unique-token observations."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def refine(pose,world,pixels,tokens,K,k1,steps=3,loss="linear"):
    if loss not in ("linear","soft_l1"):raise ValueError("invalid loss")
    best=pose.copy();accepted=0
    if not np.isfinite(best).all() or not len(tokens):return best,accepted
    # Canonicalize alternatives before selection, so duplicates/order cannot bias LM.
    unique=np.unique(np.c_[tokens,world,pixels],axis=0)
    tokens=unique[:,0].astype(int);world=unique[:,1:4];pixels=unique[:,4:6]
    groups=[np.flatnonzero(tokens==t) for t in np.unique(tokens)]
    key,selected=score_pose(best,world,pixels,groups,K,k1)
    for _ in range(steps):
        if len(selected)<6:break
        try:
            distortion=np.array([k1,0.,0.,0.,0.])
            if loss=='linear':
                rv,tv=cv2.solvePnPRefineLM(world[selected],pixels[selected],np.asarray(K,float),distortion,cv2.Rodrigues(best[:3,:3])[0],best[:3,3].copy())
            else:
                from scipy.optimize import least_squares
                def residual(x):
                    projected,_=cv2.projectPoints(world[selected],x[:3],x[3:],np.asarray(K,float),distortion)
                    return (projected.reshape(-1,2)-pixels[selected]).reshape(-1)
                x0=np.r_[cv2.Rodrigues(best[:3,:3])[0].reshape(3),best[:3,3]]
                fit=least_squares(residual,x0,loss='soft_l1',f_scale=1.,max_nfev=100)
                rv,tv=fit.x[:3],fit.x[3:]
        except cv2.error:break
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():break
        proposal=np.eye(4);proposal[:3,:3]=cv2.Rodrigues(rv)[0];proposal[:3,3]=tv.reshape(3)
        new_key,new_selected=score_pose(proposal,world,pixels,groups,K,k1)
        if new_key<=key:break
        best,key,selected=proposal,new_key,new_selected;accepted+=1
    return best,accepted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','poses','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pixels',type=Path,help='Optional matched readout; fixed candidate geometry.')
    p.add_argument('--pixel_mode',type=int,choices=[0,1],default=1)
    p.add_argument('--loss',choices=['linear','soft_l1'],default='linear')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.poses) as z:names=z['names'];sources=z['source_image'];poses=z['pose_w2c']
    refined_pixels=None
    if a.pixels:
        with np.load(a.pixels) as z:
            if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('pixel lineage differs')
            refined_pixels=z['refined_pixels'][:,a.pixel_mode];mask=z['refinement_mask'].astype(bool)
        if refined_pixels.shape!=(len(c['query_token']),2) or mask.shape!=(len(refined_pixels),) or not np.isfinite(refined_pixels).all():raise ValueError('invalid pixel inventory')
    output=[];accepted=[]
    for s,name,pose in zip(sources,names,poses):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];world=c['prototype_world'][c['prototype_rows'][rows]]
        pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
        if refined_pixels is not None:pixels=np.where(mask[rows,None],refined_pixels[rows],pixels)
        with np.load(a.contributors/str(name)) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        result,n=refine(pose,world,pixels,tok,K,k1,loss=a.loss);output.append(result);accepted.append(n)
    np.savez_compressed(a.output,names=names,source_image=sources,pose_w2c=np.asarray(output))
    a.output.with_suffix('.json').write_text(json.dumps({'scope':__doc__,'pixel_sha256':file_sha256(a.pixels) if a.pixels else None,'pixel_mode':a.pixel_mode if a.pixels else None,'pose_labels_accessed':False,'loss':a.loss,'robust_scale_px':1. if a.loss=='soft_l1' else None,'max_lm_steps':3,'accepted_steps':accepted,'candidate_sha256':file_sha256(a.candidates),'initial_pose_sha256':file_sha256(a.poses)},indent=2))


if __name__=='__main__':main()
