"""Post-label diagnostic: separate association quality from fixed-coordinate limits.

All selectors here use labels/GT and are explicitly non-deployable oracles.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _solve
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','features','labels','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.features) as z:names=z['source_names'].astype(str)
    with np.load(a.labels) as z:
        old=z['original_labels'];new=z['labels']
        if str(z['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('labels differ')
    arms={k:[] for k in ['old_positive','extended_positive','reprojection_1px_oracle','reprojection_2px_oracle']};counts={k:[] for k in arms}
    for s in np.unique(c['source_image']):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];world=c['prototype_world'][c['prototype_rows'][rows]]
        with np.load(a.contributors/names[s]) as z:
            gt=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        camera=world@gt[:3,:3].T+gt[:3,3];pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
        projected,_=cv2.projectPoints(world,cv2.Rodrigues(gt[:3,:3])[0],gt[:3,3],K,np.array([k1,0.,0.,0.,0.]))
        err=np.linalg.norm(projected.reshape(-1,2)-pixels,axis=1);err[camera[:,2]<=0]=np.inf
        masks=[old[rows]==1,new[rows]==1,err<=1,err<=2]
        for arm,mask in zip(arms,masks):
            chosen=np.flatnonzero(mask);counts[arm].append(len(np.unique(tok[chosen])))
            pose=_solve(world,tok,K,k1,chosen)
            arms[arm].append(_pose_error(np.full((4,4),np.nan) if pose is None else pose,gt))
    reports={}
    for arm,errors in arms.items():
        e=np.array(errors);reports[arm]={'hits':[int(((e[:,0]<=t)&(e[:,1]<=r)).sum()) for t,r in [(.1,1),(.25,2),(.5,5),(1,10),(2,45)]],
                    'median_translation':float(np.median(e[:,0])),'median_unique_tokens':float(np.median(counts[arm])),'images_fewer_than_6_tokens':int((np.array(counts[arm])<6).sum())}
    report={'scope':__doc__,'candidate_sha256':file_sha256(a.candidates),'production_eligible':False,'reports':reports,'errors':arms}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
