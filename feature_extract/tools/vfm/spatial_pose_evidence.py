"""Pose evidence from unique query support, coverage, and local observability.

No GT, semantic labels, or fitted scene thresholds are consumed here. Covariance
is a unit-pixel-noise linearization diagnostic, not calibrated pose uncertainty.
"""
import cv2
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose

FEATURE_NAMES=['log_tokens','inlier_fraction','robust_mass_fraction','covered_cells_fraction',
               'hull_fraction','log_depth_ratio','log_condition','log_translation_noise',
               'log_inliers','spatial_mass_fraction']


def evidence(pose,world,tokens,pixels,K,k1):
    w,t,xy=canonical_hypotheses(world,tokens,pixels)
    groups=[np.flatnonzero(t==v) for v in np.unique(t)]
    if pose is None or not np.isfinite(pose).all() or not groups:
        return np.array([0,0,0,0,0,0,20,20,0,0],float)
    _,sel=score_pose(pose,w,xy,groups,K,k1)
    n=len(groups);count=len(sel)
    if count<3:return np.array([np.log1p(n),count/n,0,0,0,0,20,20,np.log1p(count),0],float)
    camera=w[sel]@pose[:3,:3].T+pose[:3,3]
    uv,jac=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),K,np.array([k1,0.,0.,0.,0.]))
    error=np.sum((uv.reshape(-1,2)-xy[sel])**2,axis=1)
    mass=np.maximum(1-error/16,0)
    cell=np.clip(np.floor(xy[sel]/16).astype(int),[0,0],[15,8]);cid=cell[:,1]*16+cell[:,0]
    spatial=np.zeros(144);np.maximum.at(spatial,cid,mass)
    hull=cv2.contourArea(cv2.convexHull(xy[sel].astype(np.float32)))/(256*144)
    J=jac[:,:6];scaled=J/np.maximum(np.linalg.norm(J,axis=0),1e-12)
    singular=np.linalg.svd(scaled,compute_uv=False);condition=singular[0]/max(singular[-1],1e-12)
    sv=np.linalg.svd(J,compute_uv=False)
    if sv[-1]<=sv[0]*1e-10:noise=1e8
    else:noise=np.sqrt(np.trace(np.linalg.inv(J.T@J)[3:,3:]))
    z=camera[:,2];depth_ratio=np.percentile(z,90)/max(np.percentile(z,10),1e-8)
    return np.array([np.log1p(n),count/n,mass.sum()/n,len(np.unique(cid))/144,
                     hull,np.log1p(depth_ratio),np.log1p(condition),np.log1p(noise),
                     np.log1p(count),spatial.sum()/144],float)
