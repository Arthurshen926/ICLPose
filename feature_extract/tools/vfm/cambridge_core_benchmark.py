"""Five official Cambridge splits: frozen portable-core paired mechanism assay.

Inference never loads pose labels. Geometry is supplied by train-only 2DGS.
This assay is NOT the historical full scene-specific v414 cascade.
"""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine
from feature_extract.tools.vfm.local_precision_evidence import paired_features,project_support
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256

ROOT=Path('output/cambridge_core_v415')
PROTOCOL=dict(version='v415_core_assay_1',map_images=120,max_regions=4,region_radius_m=6.,query_tokens=256,descriptor_threshold=.6040024161338806,region_hypothesis_cap=250,total_hypothesis_cap=1000,pnp_threshold_px=4.,seed=260901,near_clip_m=1.,disk_sigma=3.,projection='frozen StMarys mapping-trained 64d',arms=['pooled','regional','regional_fine','regional_agreement'],test_specific_tuning=False,query_geometry_used=False,full_v414_cascade=False)


def build(scene):
    r=ROOT/scene;names=json.load(open(r/'mapping_names.json'));world=[];coarse=[];fine=[];sources=[];tokens=[];lineage={};coverage=[]
    for si,n in enumerate(names):
        gp=r/'geometry_v1'/n;fp=r/'features'/n
        with np.load(gp) as g,np.load(fp) as f:
            t=g['tokens'];w=g['world'];xy=np.c_[t%64*4+1.5,t//64*4+1.5];world.append(w);coarse.append(normalise(f['coarse_final'].reshape(-1,64)[t]));fine.append(sample_grid(f['fine_final'],xy));sources.append(np.full(len(t),si));tokens.append(t);coverage.append(dict(name=n,anchors=len(t),primitive_count=len(np.unique(g['primitive_ids']))))
        for p in [gp,fp]:lineage[str(p)]=file_sha256(p)
    dest=r/'map.npz';np.savez_compressed(dest,world=np.concatenate(world),coarse_map=np.concatenate(coarse),fine_map=np.concatenate(fine),source_ids=np.concatenate(sources),source_tokens=np.concatenate(tokens),names=np.array(names),metadata_json=np.array(json.dumps(dict(source_sha256=lineage,coverage=coverage,protocol=PROTOCOL))))
    print(scene,'map',sum(len(w) for w in world),flush=True)


def localize(scene,shard,shards,seed):
    r=ROOT/scene;out=r/f'predictions_seed{seed}';out.mkdir(exist_ok=True)
    protocol=dict(PROTOCOL,seed=seed,implementation_sha256=file_sha256(Path(__file__)),map_sha256=file_sha256(r/'map.npz'),camera_sha256=file_sha256(r/'camera_only.npz'));seal=r/f'protocol_seed{seed}.json'
    if seal.exists():assert json.load(open(seal))==protocol
    else:seal.write_text(json.dumps(protocol,indent=2))
    with np.load(r/'map.npz') as z:maps={k:z[k] for k in ['world','coarse_map','fine_map']}
    maps['available']=np.ones(len(maps['world']),bool);world=maps['world'];lod=LocalizationLoD(world,maps['coarse_map']);tree=cKDTree(world)
    with np.load(r/'camera_only.npz') as z:cameras={str(n):(K,float(k)) for n,K,k in zip(z['names'],z['camera_matrices'],z['radial_k1'])}
    names=json.load(open(r/'test_names.json'))[shard::shards];regions={}
    for index,name in enumerate(names):
        dest=out/name
        if dest.exists():continue
        started=time.perf_counter()
        with np.load(r/'features'/name) as z:grids=[normalise(z[k].reshape(-1,64)).reshape(z[k].shape) for k in ['coarse_final','fine_final']]
        q=grids[0].reshape(-1,64);K,k=cameras[name];tokens=diverse_tokens(q,np.ones(2304,bool));chosen,cost=lod.proposals(q);inventory=[];region_poses=[];stats=[]
        for rid in chosen:
            if rid not in regions:regions[rid]=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int)
            region=regions[rid]
            if len(region)<6:continue
            similarity=q[tokens]@maps['coarse_map'][region].T;nearest=similarity.argmax(1);mutual=similarity.argmax(0)[nearest]==np.arange(len(tokens));take=np.flatnonzero(mutual&(similarity[np.arange(len(tokens)),nearest]>=PROTOCOL['descriptor_threshold']));pr=region[nearest[take]];tt=tokens[take];xy=np.c_[tt%64*4+1.5,tt//64*4+1.5];xy=fine_coordinates(grids,xy,pr[:,None],maps)[:,0];inventory.append((pr,tt,xy));st={};p=solve(world[pr],tt,K,k,np.arange(len(pr)),pixels=xy,iterations=1250,seed=seed+int(rid),hypothesis_budget=250,stats=st);stats.append(st)
            if p is not None:region_poses.append(p)
        nan=np.full((4,4),np.nan);pooled=regional=raw=agreement=nan;details=dict(region_ids=chosen,matches=[len(x[0]) for x in inventory],regional_pnp=stats,query_pose_or_labels_read=False)
        if inventory:
            pr,tt,xy=[np.concatenate([v[j] for v in inventory]) for j in range(3)];cw,ct,cp=canonical_hypotheses(world[pr],tt,xy);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)]
            st={};pooled=solve(cw,ct,K,k,np.arange(len(ct)),pixels=cp,iterations=5000,seed=seed,hypothesis_budget=1000,stats=st);pooled=nan if pooled is None else pooled;details['pooled_pnp']=st;details['shared_inventory_sha256']=arrays_sha256(dict(world=cw,tokens=ct,pixels=cp))
            if region_poses:regional=max(region_poses,key=lambda p:score_pose(p,cw,cp,groups,K,k,return_selected=False)[0])
            raw=agreement=regional.copy()
            if np.isfinite(regional).all():
                # Deterministic one-support-per-token bank, fixed before optimizing.
                pixels,visible=project_support(regional,world[pr],K,k);error=np.linalg.norm(pixels-xy,axis=1);valid=np.flatnonzero(visible&(error<=4.));valid=valid[np.lexsort((pr[valid],error[valid],tt[valid]))];_,first=np.unique(tt[valid],return_index=True);valid=valid[first];details['refinement_support']=len(valid)
                if len(valid)>=6:
                    rows=pr[valid];raw,info=refine(regional,world[rows],K,k,grids[1],maps['fine_map'][rows],constrained=True);features,n=paired_features(regional,raw,world[rows],xy[valid],tt[valid],K,k,grids,[maps['coarse_map'][rows],maps['fine_map'][rows]]);accept=bool(n>=6 and features[0]>0 and features[6]>0);agreement=raw if accept else regional;details.update(refinement=info,paired_features=features.tolist(),agreement_accept=accept)
        np.savez_compressed(dest,pooled=pooled,regional=regional,regional_fine=raw,regional_agreement=agreement,metadata_json=np.array(json.dumps(dict(**details,seconds=time.perf_counter()-started,seed=seed,query_feature_sha256=file_sha256(r/'features'/name)))))
        if (index+1)%10==0:print(scene,shard,index+1,len(names),round(time.perf_counter()-started,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','infer']);p.add_argument('--scene',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--seed',type=int,default=260901);a=p.parse_args();build(a.scene) if a.command=='build' else localize(a.scene,a.shard,a.shards,a.seed)
if __name__=='__main__':main()
