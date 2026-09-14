"""Experimental eligibility-before-ranking and dual-domain support rescue.

Retains the chosen baseline. The old and proposed inventories share query tokens;
canonical scoring prevents duplicate token votes. This reuses the historical
verification bank and does not claim independent evidence from earlier decisions.
"""
from feature_extract.tools.vfm.pose_evidence_selection import select_candidate
from pathlib import Path
import argparse,json,joblib,numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.spatial_pose_evidence import evidence
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.heldout_pose_gate import heldout_gate
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.tools.vfm.verification_bank_contract import validate_bank,validate_poses
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['eligible','dual_support'],required=True);p.add_argument('--root',required=True);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--baseline-root',default='heldout_evidence_v405/mnn_seed1');p.add_argument('--baseline-arm',default='corroborate');a=p.parse_args();b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/a.root;mp=b/'native_fine_v264/readout/map.npz';modelpath=b/'spatial_verifier_v404/pairwise_verifier.joblib';fit=joblib.load(modelpath);model=fit['model'];meta=fit['metadata'];scale=meta['scale'];threshold=meta['threshold'];assert not meta['query_routes_read']
    with np.load(mp) as f:world=f['world_points'];mm=json.loads(f['metadata_json'].item())
    maphash=file_sha256(mp);atlas=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';assert file_sha256(atlas)==mm['atlas_sha256']
    for arm in a.arms:
     o=b/a.output/(arm+'_verified');o.mkdir(exist_ok=True,parents=True)
     for s in ['seq10','shard0','shard1','shard2','shard3']:
      ap=r/'lod'/f'{s}_{arm}_audit.json';rp=r/'lod_final'/f'{s}_{arm}_audit.json';bp=b/a.baseline_root/f'{s}_{a.baseline_arm}.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';bank=b/'heldout_evidence_v405/banks'/f'{s}.npz'
      with np.load(bp) as f:names=f['names'];base=f['pose_w2c']
      with np.load(cp) as f:Ks=f['camera_matrices'];ks=f['radial_k1'];assert np.array_equal(names,f['names'])
      with np.load(bank) as f:bk={k:f[k] for k in f.files if k!='metadata_json'};bm=json.loads(f['metadata_json'].item())
      validate_bank(bk,bm,world,mp,maphash,mm['atlas_sha256'],names,Ks,ks);validate_poses(base);oldap=b/'overlap_lod_v402/lod_fixed'/f'{s}_mnn_audit.json';oldrp=b/'overlap_lod_v402/lod_fixed_final'/f'{s}_mnn_audit.json';oldraw=json.load(open(oldap));oldref=json.load(open(oldrp));raw=json.load(open(ap));ref=json.load(open(rp));values={k:[] for k in ['confirmed']};accepts={k:[] for k in values};audit=[]
      for i,n in enumerate(names.astype(str)):
       assert raw[i]['name']==ref[i]['name']==n;assert oldraw[i]['name']==n and raw[i]['sampled_tokens']==oldraw[i]['sampled_tokens'];regs=oldraw[i]['regions']+raw[i]['regions'];ids=np.concatenate([a['prototype_rows'] for a in regs]).astype(int) if regs else np.array([],int);t=np.concatenate([a['selected_tokens'] for a in regs]).astype(int) if regs else np.array([],int);xy=np.concatenate([np.array(a['query_pixels']).reshape(-1,2) for a in regs]) if regs else np.empty((0,2));pool=[base[i]]+[np.array(p) for p in oldref[i]['refined_candidate_poses']+ref[i]['refined_candidate_poses'] if p is not None];features=np.array([evidence(p,world[ids],t,xy,Ks[i],float(ks[i])) for p in pool]);prob=expit(scale*model.decision_function(features-features[:1]));j=int(prob.argmax());take=j>0 and prob[j]>=threshold and separated_mode(base[i],pool[j]);lo,hi=bk['offsets'][i:i+2];assert not set(bk['query_tokens'][lo:hi])&set(raw[i]['sampled_tokens']);w,tt,xx=canonical_hypotheses(world[bk['prototype_rows'][lo:hi]],bk['query_tokens'][lo:hi],bk['pixels'][lo:hi]);groups=[np.flatnonzero(tt==v) for v in np.unique(tt)];kb=score_pose(base[i],w,xx,groups,Ks[i],float(ks[i]))[0];kn=score_pose(pool[j],w,xx,groups,Ks[i],float(ks[i]))[0];confirm=bool(take and heldout_gate(kb,kn,'corroborate'))
       pw,pt,px=canonical_hypotheses(world[ids],t,xy);pg=[np.flatnonzero(pt==v) for v in np.unique(pt)]
       proposal_scores=[score_pose(p,pw,px,pg,Ks[i],float(ks[i]))[0] for p in pool]
       bank_scores=[score_pose(p,w,xx,groups,Ks[i],float(ks[i]))[0] for p in pool]
       separated=[separated_mode(base[i],p) for p in pool]
       choice,reason=select_candidate(prob,separated,proposal_scores,bank_scores,threshold,a.policy)
       values['confirmed'].append(pool[choice]);accepts['confirmed'].append(choice>0)
       audit.append(dict(name=n,chosen=choice,reason=reason,probabilities=prob.tolist(),proposal_scores=proposal_scores,bank_scores=bank_scores,separated=separated,legacy_chosen=j,legacy_confirmed=confirm))
      for k in values:
       arr=dict(names=names,pose_w2c=np.array(values[k]),accepted=np.array(accepts[k]));validate_poses(arr['pose_w2c']);m=dict(artifact_type='eligible_candidate_verification_v413',selection_policy=a.policy,baseline_root=a.baseline_root,baseline_arm=a.baseline_arm,arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,arm=arm,mode=k,source_sha256={str(p):file_sha256(p) for p in [oldap,oldrp,ap,rp,bp,cp,bank,mp,atlas,modelpath,Path(__file__),Path('feature_extract/tools/vfm/spatial_pose_evidence.py'),Path('feature_extract/tools/vfm/heldout_pose_gate.py'),Path('feature_extract/tools/vfm/pose_evidence_selection.py'),Path('feature_extract/tools/vfm/verification_bank_contract.py')]});m['content_sha256']=canonical_json_sha256(m);dest=o/f'{s}_{k}.npz';assert not dest.exists();np.savez_compressed(dest,**arr,metadata_json=np.array(json.dumps(m)))
      (o/f'{s}_audit.json').write_text(json.dumps(audit));print(arm,s,flush=True)

if __name__ == "__main__":
    main()
