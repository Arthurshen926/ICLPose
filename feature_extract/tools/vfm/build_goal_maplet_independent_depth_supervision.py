"""Candidate-independent mapping depth targets, checked against held-route-excluded renders.

These are same-map rendered pseudo-targets, not independent measured 3D truth.
No candidate plane is used to construct, snap, select, or validate a target.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_plane_pnp_observation_bank import _load_contributor_geometry
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import inverse_simple_radial
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def depth_targets(depth,pose,K,k1):
    if depth.shape!=(144,256):raise ValueError('depth grid differs')
    block=depth.reshape(36,4,64,4).transpose(0,2,1,3).reshape(2304,16)
    valid=np.isfinite(block)&(block>0);safe=np.where(valid,block,np.nan)
    # Empty tokens stay unknown. No fill from neighbouring pixels or surfaces.
    nonempty=valid.any(1);z=np.zeros(2304);spread=np.full(2304,np.inf)
    z[nonempty]=np.nanmedian(safe[nonempty],axis=1)
    spread[nonempty]=np.nanquantile(safe[nonempty],.9,axis=1)-np.nanquantile(safe[nonempty],.1,axis=1)
    reliable=(valid.sum(1)>=12)&(spread<=np.maximum(.25,.02*z))
    tokens=np.arange(2304);xy=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]
    uv=inverse_simple_radial((xy-K[:2,2])/np.diag(K)[:2],k1)
    camera=np.c_[uv*z[:,None],z];world=(camera-pose[:3,3])@pose[:3,:3]
    return world,reliable,spread


def reprojection_depth_agreement(world,depth,pose,K,k1):
    camera=world@pose[:3,:3].T+pose[:3,3];z=camera[:,2]
    uv=camera[:,:2]/np.maximum(z[:,None],1e-8)
    uv=uv*(1+k1*np.sum(uv**2,axis=1))[:,None]
    xy=uv*np.diag(K)[:2]+K[:2,2]
    finite=np.isfinite(xy).all(1)&(z>0)
    xy=np.where(finite[:,None],xy,-100.)
    x=np.floor(np.clip(xy[:,0],-1000,1000)).astype(int);y=np.floor(np.clip(xy[:,1],-1000,1000)).astype(int)
    inside=finite&(x>=0)&(y>=0)&(x<255)&(y<143)
    result=np.zeros(len(world),bool);rows=np.flatnonzero(inside)
    if not len(rows):return result
    xx=x[rows];yy=y[rows]
    d=np.c_[depth[yy,xx],depth[yy,xx+1],depth[yy+1,xx],depth[yy+1,xx+1]]
    good=np.isfinite(d).all(1)&(d>0).all(1)
    # Reject support straddling a render depth discontinuity.
    good &= np.ptp(d,axis=1)<=np.maximum(.25,.02*z[rows])
    fx=xy[rows,0]-xx;fy=xy[rows,1]-yy
    pred=np.sum(d*np.c_[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],axis=1)
    good &= np.abs(pred-z[rows])<=np.maximum(.1,.01*z[rows])
    result[rows]=good
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','labels','visibility_atlas','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--held_route',default='seq9');a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.targets.npz').exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:sources=z['source_image'];tokens=z['query_token'];keep=z['homography_keep']
    vis,_=PlaneVisibilityAtlas.load_npz(a.visibility_atlas);names=np.unique(vis.view_names.astype(str))
    images=np.unique(sources)
    if any(names[s].split('__')[0]!=a.held_route for s in images):raise ValueError('not held mapping images')
    refs=np.array([i for i,n in enumerate(names) if n.split('__')[0]!=a.held_route])
    inventory={};hashes={}
    for i in refs:
        path=a.contributors/names[i];inventory[int(i)]=_load_contributor_geometry(path);hashes[str(path)]=file_sha256(path)
    centers=np.array([-inventory[int(i)][1][:3,:3].T@inventory[int(i)][1][:3,3] for i in refs])
    targets=[];valid=[];supports=[];spreads=[];selected_refs=[]
    for s in images:
        path=a.contributors/names[s];depth,pose,K,k1=_load_contributor_geometry(path);hashes[str(path)]=file_sha256(path)
        world,good,spread=depth_targets(depth,pose,K,k1)
        center=-pose[:3,:3].T@pose[:3,3]
        selected=refs[np.argsort(np.sum((centers-center)**2,axis=1),kind='stable')[:8]]
        support=np.zeros(2304,np.int16)
        for r in selected:support+=reprojection_depth_agreement(world,*inventory[int(r)])
        targets.append(world);valid.append(good&(support>=2));supports.append(support);spreads.append(spread);selected_refs.append(names[selected])
    targets=np.array(targets);valid=np.array(valid);supports=np.array(supports)
    target_path=a.output.with_suffix('.targets.npz')
    np.savez_compressed(target_path,source_image=images,world=targets,valid=valid,reference_support=supports,depth_spread=np.array(spreads),reference_names=np.array(selected_refs))
    # Candidate geometry is opened only after independent targets are frozen.
    with np.load(a.candidates) as z:world=z['prototype_world'][z['prototype_rows']]
    with np.load(a.labels) as z:
        original=z['labels']
        if str(z['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('candidate/label lineage differs')
    image_row=np.searchsorted(images,sources);reliable=valid[image_row,tokens]
    distance=np.linalg.norm(world-targets[image_row,tokens],axis=1)
    metric=np.where(reliable,np.where(distance<=.25,1,np.where(distance>.5,0,-1)),-2).astype(np.int8)
    # Positive extension is metric correspondence supervision, not plane identity.
    extended=original.copy();extension=(original==-2)&(metric>=0);extended[extension]=metric[extension]
    np.savez_compressed(a.output,labels=extended,original_labels=original,independent_metric_labels=metric,
                        newly_supervised=extension,supervision_valid=extended>=0,
                        frozen_candidate_sha256=np.asarray(file_sha256(a.candidates)),target_sha256=np.asarray(file_sha256(target_path)))
    old_known=keep&(original>=0)&(metric>=0)
    report={'scope':__doc__,'candidate_sha256':file_sha256(a.candidates),'target_sha256':file_sha256(target_path),'label_sha256':file_sha256(a.output),
            'reference_routes_exclude':a.held_route,'reference_count':8,'minimum_consistent_references':2,
            'targets_valid_unique_tokens':int(valid.sum()),'known_comparable_rows':int(old_known.sum()),
            'known_label_disagreements':int((old_known&(metric!=original)).sum()),
            'new_positive_after_homography':int((keep&extension&(metric==1)).sum()),'new_negative_after_homography':int((keep&extension&(metric==0)).sum()),
            'remaining_unknown_after_homography':int((keep&(extended==-2)).sum()),
            'original_known_or_ambiguous_labels_preserved':bool(np.array_equal(extended[original!=-2],original[original!=-2])),
            'candidate_used_for_target_construction':False,'new_positive_semantics':'distance<=0.25m to cross-view-consistent rendered token target; not exact plane identity',
            'contributor_sha256':hashes,'query_test_data_used':False,'uses_mapping_pose_for_supervision_only':True}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='contributor_sha256'},indent=2))


if __name__=='__main__':main()
