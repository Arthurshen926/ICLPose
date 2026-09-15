"""Post-label diagnostic: identical global consensus and local-field readout.

No labels are loaded in this tool. Its DESIGN followed inspection of v415 seed1
results, so these outputs are exploratory, not a new untouched holdout.
"""
import argparse,json,time,cv2
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT,PROTOCOL
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine
from feature_extract.tools.vfm.local_precision_evidence import paired_features,project_support
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256


def global_refit(base,world,tokens,pixels,K,k):
    if not np.isfinite(base).all():return base.copy(),dict(accepted_steps=0)
    groups=[np.flatnonzero(tokens==t) for t in np.unique(tokens)];best=base.copy();key,_=score_pose(best,world,pixels,groups,K,k);steps=0
    for _ in range(3):
        _,selected=score_pose(best,world,pixels,groups,K,k)
        if len(selected)<6:break
        try:rv,tv=cv2.solvePnPRefineLM(world[selected],pixels[selected],K,np.array([k,0.,0.,0.,0.]),cv2.Rodrigues(best[:3,:3])[0],best[:3,3].copy())
        except cv2.error:break
        candidate=np.eye(4);candidate[:3,:3]=cv2.Rodrigues(rv)[0];candidate[:3,3]=tv.ravel()
        if not np.isfinite(candidate).all():break
        other,_=score_pose(candidate,world,pixels,groups,K,k)
        if other<=key:break
        best,key=candidate,other;steps+=1
    return best,dict(accepted_steps=steps,unique_support=key[0])


def field_readout(base,world,pr,tt,xy,K,k,grids,maps):
    if not np.isfinite(base).all():return base.copy(),{}
    pixels,visible=project_support(base,world[pr],K,k);error=np.linalg.norm(pixels-xy,axis=1);valid=np.flatnonzero(visible&(error<=4.));valid=valid[np.lexsort((pr[valid],error[valid],tt[valid]))];_,first=np.unique(tt[valid],return_index=True);valid=valid[first]
    if len(valid)<6:return base.copy(),dict(support=len(valid))
    rows=pr[valid];raw,info=refine(base,world[rows],K,k,grids[1],maps['fine_map'][rows],constrained=True);features,n=paired_features(base,raw,world[rows],xy[valid],tt[valid],K,k,grids,[maps['coarse_map'][rows],maps['fine_map'][rows]]);accept=bool(n>=6 and features[0]>0 and features[6]>0)
    return raw if accept else base.copy(),dict(support=len(valid),optimizer=info,paired_features=features.tolist(),agreement_accept=accept)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--seed',type=int,default=260901);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args();r=ROOT/a.scene;out=r/f'readout_seed{a.seed}';out.mkdir(exist_ok=True);names=json.load(open(r/'test_names.json'))[a.shard::a.shards]
    with np.load(r/'map.npz') as z:maps={k:z[k] for k in ['world','coarse_map','fine_map']}
    world=maps['world'];maps['available']=np.ones(len(world),bool);lod=LocalizationLoD(world,maps['coarse_map']);tree=cKDTree(world);regions={}
    with np.load(r/'camera_only.npz') as z:cameras={str(n):(K,float(k)) for n,K,k in zip(z['names'],z['camera_matrices'],z['radial_k1'])}
    for index,name in enumerate(names):
        dest=out/name
        if dest.exists():continue
        started=time.perf_counter()
        with np.load(r/'features'/name) as z:grids=[normalise(z[k].reshape(-1,64)).reshape(z[k].shape) for k in ['coarse_final','fine_final']]
        q=grids[0].reshape(-1,64);K,k=cameras[name];tokens=diverse_tokens(q,np.ones(2304,bool));chosen,_=lod.proposals(q);inventory=[]
        for rid in chosen:
            if rid not in regions:regions[rid]=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int)
            region=regions[rid]
            if len(region)<6:continue
            similarity=q[tokens]@maps['coarse_map'][region].T;nearest=similarity.argmax(1);mutual=similarity.argmax(0)[nearest]==np.arange(len(tokens));take=np.flatnonzero(mutual&(similarity[np.arange(len(tokens)),nearest]>=PROTOCOL['descriptor_threshold']));pr=region[nearest[take]];tt=tokens[take];xy=np.c_[tt%64*4+1.5,tt//64*4+1.5];xy=fine_coordinates(grids,xy,pr[:,None],maps)[:,0];inventory.append((pr,tt,xy))
        with np.load(r/f'predictions_seed{a.seed}'/name) as z:base_region=z['regional'];previous=json.loads(z['metadata_json'].item())
        with np.load(r/f'multistart_seed{a.seed}'/name) as z:base_pool=z['pooled_multistart']
        outputs={};details={};inventory_hash=None
        if inventory:
            pr,tt,xy=[np.concatenate([v[j] for v in inventory]) for j in range(3)];cw,ct,cp=canonical_hypotheses(world[pr],tt,xy);inventory_hash=arrays_sha256(dict(world=cw,tokens=ct,pixels=cp));assert inventory_hash==previous['shared_inventory_sha256']
            for label,base in [('regional',base_region),('pooled',base_pool)]:
                updated,info=global_refit(base,cw,ct,cp,K,k);fine,fi=field_readout(updated,world,pr,tt,xy,K,k,grids,maps);outputs[label+'_global']=updated;outputs[label+'_global_field']=fine;details[label]=dict(global_refit=info,field=fi)
        else:
            for label,base in [('regional',base_region),('pooled',base_pool)]:outputs[label+'_global']=outputs[label+'_global_field']=base.copy()
        np.savez_compressed(dest,**outputs,metadata_json=np.array(json.dumps(dict(post_label_exploratory_design=True,query_labels_consumed=False,shared_inventory_sha256=inventory_hash,implementation_sha256=file_sha256(Path(__file__)),details=details,seconds=time.perf_counter()-started))))
        if (index+1)%20==0:print(a.scene,a.seed,a.shard,index+1,len(names),flush=True)
if __name__=='__main__':main()
