"""Reuse the established MoGe plane/scale refiner on unique-token memory poses."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import _refine_pose_scale,_many_to_one_plane_associations
from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics,_region_token_support
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','poses','planar_map','query_regions','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pixels',type=Path);p.add_argument('--pixel_mode',type=int,default=1)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.poses) as z:names=z['names'].astype(str);sources=z['source_image'];initial=z['pose_w2c']
    pixel_readout=None
    if a.pixels:
        with np.load(a.pixels) as z:
            if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('pixel lineage differs')
            pixel_readout=z['refined_pixels'][:,a.pixel_mode];mask=z['refinement_mask']
    planes=GeometryNativePlanarMap.load_npz(a.planar_map);poses=[];diagnostics=[]
    for i,(s,name,pose) in enumerate(zip(sources,names,initial)):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];pr=c['prototype_rows'][rows]
        world=c['prototype_world'][pr];pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
        if pixel_readout is not None:pixels=np.where(mask[rows,None],pixel_readout[rows],pixels)
        with np.load(a.contributors/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        qp,qmeta=QueryPlaneRegions.load_npz(a.query_regions/name)
        if qmeta.get('uses_pose_or_ground_truth') is not False:raise ValueError('query regions must be pose-free')
        token_region=np.full(2304,-1,int)
        for region in range(len(qp.normals_camera)):
            tokens,fraction=_region_token_support(qp.labels,region)
            # At least 12/16 observed pixels must identify the region. Two
            # disjoint masks cannot both pass; do not assign boundary tokens
            # to whichever region happened to be enumerated last.
            token_region[tokens[fraction>=.75]]=region
        valid=token_region[tok]>=0;groups=[np.flatnonzero(tok==t) for t in np.unique(tok)]
        if not np.isfinite(pose).all() or not len(rows):poses.append(pose);diagnostics.append({'accepted':False});continue
        key,selected=score_pose(pose,world,pixels,groups,K,k1);selected=selected[valid[selected]]
        provenance=np.c_[token_region[tok[selected]],c['prototype_plane'][pr[selected]],pr[selected]]
        associations=_many_to_one_plane_associations(provenance,np.arange(len(selected)))
        proposal,scale,accepted,detail=_refine_pose_scale(pose,world[selected],pixels[selected],provenance,np.arange(len(selected)),associations,
            planes.normals_world,planes.offsets_world,qp.normals_camera,qp.offsets_camera,K,k1)
        # Reuse original geometric score for a second explicit unique-token guard.
        # An overlapping memory cannot increase acceptance by duplicating rows.
        final_key,_=score_pose(proposal,world,pixels,groups,K,k1)
        accepted=accepted and final_key>=key
        poses.append(proposal if accepted else pose)
        diagnostics.append({'accepted':bool(accepted),'normal_offset_refiner':detail,'scale':scale,'unique_support':len(selected),'associations':len(associations)})
        if (i+1)%25==0:print('moge refinement',i+1,flush=True)
    np.savez_compressed(a.output,names=names,source_image=sources,pose_w2c=np.asarray(poses))
    errors=[];before=[]
    for name,pose,old in zip(names,poses,initial):
        with np.load(a.contributors/name) as z:gt=z['pose_w2c']
        errors.append(_pose_error(pose,gt));before.append(_pose_error(old,gt))
    def hits(e):
        e=np.asarray(e);return [int(((e[:,0]<=t)&(e[:,1]<=r)).sum()) for t,r in [(.1,1),(.25,2),(.5,5),(1,10),(2,45)]]
    a.output.with_suffix('.json').write_text(json.dumps({'scope':__doc__,'before_hits':hits(before),'hits':hits(errors),
        'accepted':sum(d['accepted'] for d in diagnostics),'images':len(names),'errors':errors,'diagnostics':diagnostics,
        'candidate_sha256':file_sha256(a.candidates),'initial_pose_sha256':file_sha256(a.poses),
        'query_GT_used_for_refinement':False,'same_mainline_refiner_function':True,
        'additional_unique_token_acceptance_guard':True,'not_full_two_branch_consensus':True},indent=2))


if __name__=='__main__':main()
