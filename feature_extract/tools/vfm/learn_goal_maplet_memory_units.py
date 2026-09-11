"""Learn a frozen map policy from cross-fitted mapping-view pose outcomes.

The supervised utility combines valid precise identity evidence and actual
fixed-backend pose success. It is a credit-assignment surrogate, not a causal
per-unit removal effect or a proof of globally optimal map organization.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def policy_design(attributes):
    a=np.asarray(attributes,float)
    return np.concatenate([a,a*a,a[...,:5]*a[...,5:6]],axis=-1)


def capacity_mask(values,fraction):
    v=np.asarray(values,float)
    if not np.isfinite(v).all() or not 0<fraction<=1:raise ValueError('invalid utility/capacity')
    count=int(np.floor(len(v)*fraction));mask=np.zeros(len(v),bool)
    mask[np.lexsort((np.arange(len(v)),-v))[:count]]=True
    return mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['map_choices','candidates','labels','scores','contributors','pose_root','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.map_choices) as z:m={k:z[k] for k in z.files}
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    digest=file_sha256(a.candidates)
    with np.load(a.labels) as z:
        y=z['labels']
        if str(z['frozen_candidate_sha256'])!=digest:raise ValueError('label lineage differs')
    with np.load(a.scores) as z:
        scores=z['logits']
        if str(z['candidate_sha256'])!=digest or z['arm_names'].tolist()!=['radius1','radius2','radius4']:raise ValueError('score lineage differs')
    if not np.array_equal(m['prototype_world'],c['prototype_world']) or not np.array_equal(m['prototype_plane'],c['prototype_plane']):raise ValueError('map choice geometry differs')
    names=m['source_names'];images=np.unique(c['source_image']);n=len(m['prototype_world'])
    if not all(names[s].startswith('seq9__') for s in images):raise ValueError('policy training must be mapping seq9 only')
    x=policy_design(m['attributes']);training_x=[];training_y=[];training_w=[];pose_reports={}
    for k,r in enumerate([1,2,4]):
        path=a.pose_root/f'radius{r}'/f'radius{r}.npz'
        pose_report=json.loads((path.parent/'summary.json').read_text())
        if pose_report['candidate_sha256']!=digest or pose_report['score_sha256']!=file_sha256(a.scores):raise ValueError('pose training lineage differs')
        with np.load(path) as z:poses=dict(zip(z['names'].astype(str),z['pose_w2c']))
        rewards=[]
        for s in images:
            rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']&(y>=0));pr=c['prototype_rows'][rows]
            with np.load(a.contributors/names[s]) as z:
                gt=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
            te,re=_pose_error(poses[names[s]],gt)
            reward=float(np.exp(-te/.25-re/2)) if np.isfinite(te+re) else 0.
            rewards.append(reward)
            if not len(rows):continue
            world=c['prototype_world'][pr];tok=c['query_token'][rows]
            xy=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
            projected=cv2.projectPoints(world,cv2.Rodrigues(gt[:3,:3])[0],gt[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
            front=(world@gt[:3,:3].T+gt[:3,3])[:,2]>0
            precision=np.exp(-.5*np.sum((projected-xy)**2,axis=1)/1.5**2)*(y[rows]==1)*front
            # Probability expresses recognition confidence; mapping reprojection
            # utility includes precision, and the frozen actual pose supplies
            # configuration-level value. Unknown rows never become negatives.
            value=precision*expit(scores[rows,k])*(.5+.5*reward)
            ids,inverse=np.unique(pr,return_inverse=True);sums=np.bincount(inverse,weights=value);counts=np.bincount(inverse)
            training_x.append(x[ids,k]);training_y.append(sums/counts);training_w.append(np.full(len(ids),1/len(ids)))
        pose_reports[str(r)]={'mean_mapping_pose_utility':float(np.mean(rewards)),'pose_sha256':file_sha256(path)}
    tx=np.concatenate(training_x);ty=np.concatenate(training_y);w=np.concatenate(training_w);w/=w.sum()
    mean=np.sum(tx*w[:,None],axis=0);std=np.sqrt(np.sum((tx-mean)**2*w[:,None],axis=0));std=np.maximum(std,1e-3)
    design=np.c_[np.ones(len(tx)),(tx-mean)/std];reg=np.eye(design.shape[1])*.01;reg[0,0]=0
    beta=np.linalg.solve(design.T@(design*w[:,None])+reg,design.T@(ty*w))
    prediction=beta[0]+((x-mean)/std)@beta[1:];choice=np.argmax(prediction,axis=1)
    utility=prediction[np.arange(n),choice];selected=capacity_mask(utility,.25)
    # Same-capacity fixed-context comparator, using map-only cross-view stability.
    fixed_selected=capacity_mask(m['attributes'][:,1,6],.25)
    for name,mask in [('full',np.ones(n,bool)),('budget25',selected)]:
        ids=np.flatnonzero(mask);ch=choice[ids]
        np.savez_compressed(a.output/(name+'.npz'),unit_anchor_prototype=ids,scale_index=ch,
            descriptors=m['descriptors'][ids,ch],reference_bank_rows=m['reference_bank_rows'][ids,ch],
            prototype_world=m['prototype_world'],prototype_plane=m['prototype_plane'],
            geometry_bank_sha256=m['geometry_bank_sha256'])
    np.savez_compressed(a.output/'policy.npz',choice=choice,utility=utility,budget25=selected,fixed_budget25=fixed_selected,
        prediction=prediction,prototype_world=m['prototype_world'],prototype_plane=m['prototype_plane'])
    (a.output/'model.json').write_text(json.dumps({'scope':__doc__,'mean':mean.tolist(),'std':std.tolist(),'coefficients':beta.tolist(),
        'training_images':names[images].tolist(),'training_records':len(ty),'mapping_pose_reports':pose_reports,
        'radius_histogram':np.bincount(choice,minlength=3).tolist(),'budget25_radius_histogram':np.bincount(choice[selected],minlength=3).tolist(),
        'full_descriptor_bytes':n*4*64*2,'budget25_descriptor_bytes':int(selected.sum())*4*64*2,
        'full_reference_bytes':n*16*8,'budget25_reference_bytes':int(selected.sum())*16*8,
        'geometry_changed':False,'query_adaptation_changes_map':False,'fitted_on_evaluation_routes':False,
        'objective':'ridge prediction of identification x precision x (0.5 + 0.5 fixed-PnP pose utility)',
        'not_globally_optimal_or_causal_unit_effect':True,
        'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['map_choices','candidates','labels','scores']}},indent=2))


if __name__=='__main__':main()
