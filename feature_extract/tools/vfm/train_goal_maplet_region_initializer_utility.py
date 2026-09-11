"""Learn regional usefulness from each region's own frozen PnP initializer.

No shared image reward is copied to all units. This predicts standalone region
initialization quality; it does not claim a causal leave-one-region-out effect.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.metric_region_memory import build_regions,activate_regions,region_features
from feature_extract.tools.vfm.token_hypothesis_ransac import solve
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','scores','features','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--radius',type=float,default=3.)
    p.add_argument('--hypothesis_budget',type=int,default=128)
    p.add_argument('--global_refine',action='store_true')
    a=p.parse_args()
    if a.hypothesis_budget<1:raise ValueError('positive hypothesis budget required')
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.scores) as z:
        priority=z['logits'][:,z['arm_names'].tolist().index('radius2')]
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('score lineage differs')
    with np.load(a.features) as z:names=z['source_names'].astype(str)
    sources=np.unique(c['source_image'])
    if not all(names[s].startswith('seq9__') for s in sources):raise ValueError('mapping-only training required')
    centers,members=build_regions(c['prototype_world'],a.radius);records=[];poses=[];xx=[];image_ids=[];region_ids=[]
    for count,s in enumerate(sources):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];pr=c['prototype_rows'][rows]
        world=c['prototype_world'][pr];planes=c['prototype_plane'][pr];score=priority[rows]
        with np.load(a.contributors/names[s]) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
        groups,ids=activate_regions(world,tok,score,centers,a.radius,8,False)
        groups2,ids2=activate_regions(world,tok,score,centers,a.radius,8,True)
        union=dict(zip(ids,groups));union.update(zip(ids2,groups2))
        for rid,group in sorted(union.items()):
            pose=solve(world,tok,K,k1,group,iterations=2048,seed=260911+rid,sampling_policy='geometry_score',planes=planes,scores=score,hypothesis_budget=a.hypothesis_budget)
            if pose is not None and a.global_refine:
                from feature_extract.tools.vfm.refine_goal_maplet_global_token_lm import refine
                pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
                pose,_=refine(pose,world,pixels,tok,K,k1)
            poses.append(np.full((4,4),np.nan) if pose is None else pose);image_ids.append(s);region_ids.append(rid)
            xx.append(region_features(world,tok,score,group,len(np.unique(tok))))
        if (count+1)%10==0:print('mapping regional initializers',count+1,flush=True)
    np.savez_compressed(a.output/'frozen_initializers.npz',poses=np.asarray(poses),source_image=np.array(image_ids),region_ids=np.array(region_ids),features=np.array(xx),centers=centers)
    # Mapping pose labels are opened only after all regional initializers seal.
    gt={};target=[];errors=[]
    for s,pose in zip(image_ids,poses):
        if s not in gt:
            with np.load(a.contributors/names[s]) as z:gt[s]=z['pose_w2c']
        te,re=_pose_error(pose,gt[s]);errors.append([te,re]);target.append(np.exp(-te/.5-re/5) if np.isfinite(te+re) else 0.)
    x=np.asarray(xx);target=np.array(target);source=np.array(image_ids)
    _,inverse,count=np.unique(source,return_inverse=True,return_counts=True);w=1/count[inverse];w/=w.sum()
    mean=np.sum(x*w[:,None],axis=0);std=np.maximum(np.sqrt(np.sum((x-mean)**2*w[:,None],axis=0)),1e-3)
    design=np.c_[np.ones(len(x)),(x-mean)/std];beta=fit(design,target,w)
    model={'hypothesis_budget':a.hypothesis_budget,'global_refine':a.global_refine,'scope':__doc__,'radius':a.radius,'mean':mean.tolist(),'std':std.tolist(),'coefficients':beta.tolist(),
        'training_images':names[sources].tolist(),'training_regions':len(target),'map_world_sha256':hashlib.sha256(c['prototype_world'].tobytes()).hexdigest(),
        'target':'exp(-translation_error/.5 - rotation_error/5), zero on failure','candidate_sha256':file_sha256(a.candidates),
        'training_errors':errors,'no_transfer_pose_labels':True,'candidate_generation_remains_fixed':True}
    (a.output/'model.json').write_text(json.dumps(model,indent=2))


if __name__=='__main__':main()
