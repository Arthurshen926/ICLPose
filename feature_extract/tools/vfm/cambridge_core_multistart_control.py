"""Equal multi-start/LM opportunity control, frozen before v415 label readout."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT,PROTOCOL
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--seed',type=int,default=260901);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args();r=ROOT/a.scene;out=r/f'multistart_seed{a.seed}';out.mkdir(exist_ok=True);names=json.load(open(r/'test_names.json'))[a.shard::a.shards]
    with np.load(r/'map.npz') as z:maps={k:z[k] for k in ['world','coarse_map','fine_map']}
    world=maps['world'];maps['available']=np.ones(len(world),bool);lod=LocalizationLoD(world,maps['coarse_map']);tree=cKDTree(world);regions={}
    with np.load(r/'camera_only.npz') as z:cameras={str(n):(K,float(k)) for n,K,k in zip(z['names'],z['camera_matrices'],z['radial_k1'])}
    for index,name in enumerate(names):
        dest=out/name
        if dest.exists():continue
        start=time.perf_counter()
        with np.load(r/'features'/name) as z:grids=[normalise(z[k].reshape(-1,64)).reshape(z[k].shape) for k in ['coarse_final','fine_final']]
        q=grids[0].reshape(-1,64);K,k=cameras[name];tokens=diverse_tokens(q,np.ones(2304,bool));chosen,_=lod.proposals(q);inventory=[];seeds=[]
        for rid in chosen:
            if rid not in regions:regions[rid]=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int)
            region=regions[rid]
            if len(region)<6:continue
            similarity=q[tokens]@maps['coarse_map'][region].T;nearest=similarity.argmax(1);mutual=similarity.argmax(0)[nearest]==np.arange(len(tokens));take=np.flatnonzero(mutual&(similarity[np.arange(len(tokens)),nearest]>=PROTOCOL['descriptor_threshold']));pr=region[nearest[take]];tt=tokens[take];xy=np.c_[tt%64*4+1.5,tt//64*4+1.5];xy=fine_coordinates(grids,xy,pr[:,None],maps)[:,0];inventory.append((pr,tt,xy));seeds.append(a.seed+int(rid))
        pose=np.full((4,4),np.nan);stats=[];inventory_hash=None
        if inventory:
            pr,tt,xy=[np.concatenate([v[j] for v in inventory]) for j in range(3)];cw,ct,cp=canonical_hypotheses(world[pr],tt,xy);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)];inventory_hash=arrays_sha256(dict(world=cw,tokens=ct,pixels=cp))
            with np.load(r/f'predictions_seed{a.seed}'/name) as z:previous=json.loads(z['metadata_json'].item());assert inventory_hash==previous['shared_inventory_sha256']
            candidates=[]
            for seed in seeds:
                st={};candidate=solve(cw,ct,K,k,np.arange(len(ct)),pixels=cp,iterations=1250,seed=seed,hypothesis_budget=250,stats=st);stats.append(st)
                if candidate is not None:candidates.append(candidate)
            if candidates:pose=max(candidates,key=lambda p:score_pose(p,cw,cp,groups,K,k,return_selected=False)[0])
        np.savez_compressed(dest,pooled_multistart=pose,metadata_json=np.array(json.dumps(dict(shared_inventory_sha256=inventory_hash,exact_inventory_replay=True,query_pose_or_labels_read=False,implementation_sha256=file_sha256(Path(__file__)),seeds=seeds,pnp=stats,seconds=time.perf_counter()-start))))
        if (index+1)%20==0:print(a.scene,a.seed,a.shard,index+1,len(names),flush=True)
if __name__=='__main__':main()
