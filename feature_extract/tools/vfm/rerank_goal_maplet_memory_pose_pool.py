"""Freeze pose-pool reranking by independent-token precision support before GT.

Fixed 1.5px kernel, with unweighted-soft and learned-quality controls. Neither
new hypotheses nor coordinate updates are introduced.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def soft_token_support(pose,world,tokens,K,k1,weight,sigma=1.5):
    camera=world@pose[:3,:3].T+pose[:3,3];z=camera[:,2]
    uv=camera[:,:2]/np.where(np.abs(z[:,None])>1e-8,z[:,None],1.)
    pred=uv*(1+k1*np.sum(uv**2,axis=1))[:,None]*np.diag(K)[:2]+K[:2,2]
    xy=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]
    error=np.sum((pred-xy)**2,axis=1)
    likelihood=np.where((z>0)&np.isfinite(error),np.exp(-.5*error/sigma**2),0.)
    _,inverse=np.unique(tokens,return_inverse=True)
    value=np.zeros(len(np.unique(tokens)))
    np.maximum.at(value,inverse,weight*likelihood)
    return float(value.sum())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','class_scores','quality_scores','pose_pool','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    priorities={}
    for path in [a.class_scores,a.quality_scores]:
        with np.load(path) as z:
            if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('score lineage differs')
            priorities.update({str(n):expit(z['logits'][:,i]) for i,n in enumerate(z['arm_names'])})
    with np.load(a.pose_pool) as z:pool={k:z[k] for k in z.files}
    policies=['original','soft_uniform','soft_base','soft_combined','soft_extended_base','soft_extended_combined','soft_quality_base','soft_quality_combined']
    selected={k:[] for k in policies};selected_rows={k:[] for k in policies}
    for i,(s,name) in enumerate(zip(pool['source_image'],pool['names'].astype(str))):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tokens=c['query_token'][rows];world=c['prototype_world'][c['prototype_rows'][rows]]
        with np.load(a.contributors/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        lo,hi=pool['candidate_offsets'][i:i+2];poses=pool['candidate_pose_w2c'][lo:hi]
        selected['original'].append(pool['pose_w2c'][i]);selected_rows['original'].append(-1)
        for policy in policies[1:]:
            weight=np.ones(len(rows)) if policy=='soft_uniform' else priorities[policy[5:]][rows]
            scores=[soft_token_support(pose,world,tokens,K,k1,weight) for pose in poses]
            j=int(np.argmax(scores)) if len(scores) else -1
            selected[policy].append(poses[j] if j>=0 else np.full((4,4),np.nan));selected_rows[policy].append(j)
    np.savez_compressed(a.output.with_suffix('.npz'),names=pool['names'],**{k:np.array(v) for k,v in selected.items()},**{k+'_index':np.array(v) for k,v in selected_rows.items()})
    errors={k:[] for k in policies};routes=[]
    for i,name in enumerate(pool['names'].astype(str)):
        with np.load(a.contributors/name) as z:gt=z['pose_w2c']
        routes.append(name.split('__')[0])
        for policy in policies:errors[policy].append(_pose_error(selected[policy][i],gt))
    summaries={}
    for policy in policies:
        e=np.array(errors[policy]);summaries[policy]={'hits':[int(((e[:,0]<=t)&(e[:,1]<=r)).sum()) for t,r in [(.1,1),(.25,2),(.5,5),(1,10),(2,45)]],'median_translation':float(np.median(e[:,0]))}
    report={'scope':__doc__,'reports':summaries,'errors':errors,'routes':routes,'sigma_pixels':1.5,
            'candidate_sha256':file_sha256(a.candidates),'pose_pool_sha256':file_sha256(a.pose_pool),'class_scores_sha256':file_sha256(a.class_scores),'quality_scores_sha256':file_sha256(a.quality_scores),
            'labels_used_for_ranking':False,'pool_or_coordinates_changed':False,'exploratory_followup_after_development_pool_audit':True}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
