"""Retain competing poses from identical pooled-support RANSAC samples.

The optional collector in solve() is observational: it does not change sampling,
scoring or the historical winner. Compare score-ranked retention with geometric
mode retention at four final proposals and the same 1000 scored PnP hypotheses.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def retain_modes(candidates, limit=4, diverse=True):
    if limit<1:
        raise ValueError('positive retention budget required')
    retained=[];seen=set()
    for key,pose in sorted(candidates,key=lambda item:item[0],reverse=True):
        identity=np.asarray(pose,dtype=np.float64).tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        if diverse and any(not separated_mode(other,pose) for other in retained):
            continue
        retained.append(pose.copy())
        if len(retained)==limit:
            break
    return retained


def polish_pose(pose,world,tokens,pixels,K,k1):
    world,tokens,pixels=canonical_hypotheses(world,tokens,pixels)
    groups=[np.flatnonzero(tokens==t) for t in np.unique(tokens)]
    best=pose.copy();best_key=score_pose(best,world,pixels,groups,K,k1)[0]
    for _ in range(3):
        _,selected=score_pose(best,world,pixels,groups,K,k1)
        if len(selected)<6:break
        try:
            rv,tv=cv2.solvePnPRefineLM(world[selected],pixels[selected],K,np.array([k1,0.,0.,0.,0.]),cv2.Rodrigues(best[:3,:3])[0],best[:3,3].copy())
        except cv2.error:break
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():break
        candidate=np.eye(4);candidate[:3,:3]=cv2.Rodrigues(rv)[0];candidate[:3,3]=tv.reshape(3)
        key=score_pose(candidate,world,pixels,groups,K,k1)[0]
        if key<=best_key:break
        best,best_key=candidate,key
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--polish',action='store_true');p.add_argument('--split',required=True);p.add_argument('--seed',type=int,default=260901)
    p.add_argument('--arms',nargs='+',choices=['uniform','complement'],default=['uniform','complement'],help='uniform=ranked; complement=mode-diverse; historical backend arm names')
    a=p.parse_args();b=a.base;s=a.split;o=a.output;o.mkdir(parents=True,exist_ok=True)
    if any((o/f'{s}_{arm}.npz').exists() for arm in a.arms):raise FileExistsError(o)
    mp=b/'native_fine_v264/readout/map.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';op=b/'overlap_lod_v402/lod_fixed'/f'{s}_mnn_audit.json'
    with np.load(mp) as z:world=z['world_points']
    with np.load(cp) as z:names=z['names'];Ks=z['camera_matrices'];ks=z['radial_k1']
    original=json.loads(op.read_text());values={arm:[] for arm in a.arms};audits={arm:[] for arm in a.arms}
    for i,name in enumerate(names.astype(str)):
        record=original[i];assert record['name']==name;regions=record['regions']
        ids=np.concatenate([r['prototype_rows'] for r in regions]).astype(int);tokens=np.concatenate([r['selected_tokens'] for r in regions]).astype(int);pixels=np.concatenate([np.array(r['query_pixels']).reshape(-1,2) for r in regions]);w,t,xy=canonical_hypotheses(world[ids],tokens,pixels);groups=[np.flatnonzero(t==v) for v in np.unique(t)]
        pool=[];stats=[];winners=[]
        for region in regions:
            stat={};pose=solve(world[ids],tokens,Ks[i],float(ks[i]),np.arange(len(ids)),pixels=pixels,iterations=1250,seed=a.seed+int(region['region']),hypothesis_budget=250,stats=stat,sampling_policy='context_prior',scores=np.ones(len(ids)),proposal_collector=pool)
            stats.append(stat)
            if pose is not None:
                pool.append((score_pose(pose,w,xy,groups,Ks[i],float(ks[i]))[0],pose));winners.append(pose.tolist())
        for arm in a.arms:
            poses=retain_modes(pool,diverse=arm=='complement');poses=[polish_pose(p,world[ids],tokens,pixels,Ks[i],float(ks[i])) for p in poses] if a.polish else poses;pose=poses[0] if poses else np.full((4,4),np.nan);values[arm].append(pose)
            audits[arm].append(dict(name=name,regions=regions,region_ids=record['region_ids'],sampled_tokens=record['sampled_tokens'],candidate_poses=[v.tolist() for v in poses],joint_solver_stats=stats,collected_scored_models=len(pool),historical_restart_winners=winners,retention='mode_diverse' if arm=='complement' else 'score_ranked'))
        if (i+1)%20==0:print(s,i+1,flush=True)
    for arm in a.arms:
        arr=dict(names=names,pose_w2c=np.array(values[arm]),usable=np.isfinite(values[arm]).all((1,2)))
        meta=dict(artifact_type='joint_pose_mode_retention_v412',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,arm=arm,seed=a.seed,max_final_proposals=4,extra_polish_rounds_per_retained_pose=3 if a.polish else 0,max_scored_pnp_hypotheses=1000,correspondence_inventory_unchanged=True,retention='mode_diverse' if arm=='complement' else 'score_ranked',source_sha256={str(f):file_sha256(f) for f in [mp,cp,op,Path(__file__),Path(__file__).with_name('token_hypothesis_ransac.py'),Path(__file__).with_name('select_goal_maplet_separated_modes.py')]});meta['content_sha256']=canonical_json_sha256(meta)
        np.savez_compressed(o/f'{s}_{arm}.npz',**arr,metadata_json=np.array(json.dumps(meta)));(o/f'{s}_{arm}_audit.json').write_text(json.dumps(audits[arm]))


if __name__=='__main__':main()
