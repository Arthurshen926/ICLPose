"""Soft MoGe depth-order and relative-depth verification of an immutable pose pool."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def depth_relations(tokens,query_depth,predicted_depth,valid):
    """Ratios cancel common positive scale; uncertain order contributes nothing."""
    tok=np.asarray(tokens);q=np.asarray(query_depth);z=np.asarray(predicted_depth)
    keep=np.asarray(valid)&np.isfinite(q)&np.isfinite(z)&(q>0)&(z>0)
    tok,q,z=tok[keep],q[keep],z[keep]
    _,first=np.unique(tok,return_index=True);tok,q,z=tok[first],q[first],z[first]
    if len(tok)<2:return 0.,0.,0
    xy=np.c_[tok%64,tok//64];dist=np.sum((xy[:,None]-xy[None])**2,axis=-1)
    i,j=np.where(np.triu((dist>=4)&(dist<=144),1))
    if not len(i):return 0.,0.,0
    qr=np.log(q[i]/q[j]);zr=np.log(z[i]/z[j]);certain=np.abs(qr)>=.05
    order=np.where(certain,np.sign(qr)*np.sign(zr),0.)
    ratio=2*np.exp(-np.abs(qr-zr)/.1)-1
    # Equalize endpoints: dense neighboring clusters do not multiply their vote.
    count=np.bincount(np.r_[i,j],minlength=len(tok));weight=1/count[i]+1/count[j];weight/=weight.sum()
    return float(order@weight),float(ratio@weight),len(i)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','poses','moge','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.poses) as z:names=z['names'].astype(str);sources=z['source_image'];original=z['pose_w2c'];pool=z['candidate_pose_w2c'];offsets=z['candidate_offsets']
    selections={'depth_order':[],'depth_ratio':[]};diagnostics=[]
    for idx,(s,name) in enumerate(zip(sources,names)):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];world=c['prototype_world'][c['prototype_rows'][rows]]
        xy=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];groups=[np.flatnonzero(tok==t) for t in np.unique(tok)]
        with np.load(a.contributors/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        points,_,valid=moge_tokens(a.moge/name);choices=pool[offsets[idx]:offsets[idx+1]];scores=[]
        for pose in choices:
            key,selected=score_pose(pose,world,xy,groups,K,k1)
            depth=(world[selected]@pose[:3,:3].T+pose[:3,3])[:,2]
            order,ratio,count=depth_relations(tok[selected],points[tok[selected],2],depth,valid[tok[selected]])
            support=key[0];base=support+key[1]/max(len(groups)*16,1)
            scores.append([base+.2*support*order,base+.2*support*ratio,count,order,ratio])
        for k,arm in enumerate(selections):
            selections[arm].append(choices[np.argmax(np.array(scores)[:,k])] if len(scores) else original[idx])
        diagnostics.append({'name':name,'candidate_scores':scores})
    for arm,poses in selections.items():np.savez_compressed(a.output/(arm+'.npz'),names=names,source_image=sources,pose_w2c=np.asarray(poses))
    reports={}
    for arm,poses in [('original',original)]+list(selections.items()):
        errors=[]
        for name,pose in zip(names,poses):
            with np.load(a.contributors/name) as z:gt=z['pose_w2c']
            errors.append(_pose_error(pose,gt))
        e=np.asarray(errors);reports[arm]={'hits':[int(((e[:,0]<=t)&(e[:,1]<=r)).sum()) for t,r in [(.1,1),(.25,2),(.5,5),(1,10),(2,45)]],'errors':errors}
    (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'reports':reports,'diagnostics':diagnostics,
        'coefficient_fixed':.2,'query_GT_opened_before_sealing':False,'no_hard_depth_rejection':True,
        'limitations':['relative depth ratios do not cancel local MoGe deformation','candidate pool remains small and conditional'],
        'candidate_sha256':file_sha256(a.candidates),'pose_pool_sha256':file_sha256(a.poses)},indent=2))


if __name__=='__main__':main()
