from pathlib import Path
import json,numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.tools.vfm.heldout_pose_gate import heldout_gate
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'heldout_evidence_v405';previous=b/'spatial_verifier_v404';threshold=json.load(open(b/'decision_risk_v405_expanded/calibration.json'))['threshold']
with np.load(b/'native_fine_v264/readout/map.npz') as f:world=f['world_points']
for label,refroot,arm in [('mnn_seed1','overlap_lod_v402/lod_fixed_final','mnn'),('mnn_seed2','overlap_lod_v402/lod_seed2_final','mnn'),('soft_seed1','overlap_weighted_v403/lod_final','weighted_mnn'),('soft_seed2','overlap_weighted_seed2_v403/lod_final','weighted_mnn')]:
 out=r/label;out.mkdir(exist_ok=True)
 for s in ['seq10','shard0','shard1','shard2','shard3']:
  bank=r/'banks'/(s+'.npz');old=previous/(label+'_pairwise')/f'{s}_verified.npz';audit=json.load(open(previous/(label+'_pairwise')/f'{s}_audit.json'));bp=b/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz'
  with np.load(bank) as f:bk={k:f[k] for k in f.files if k!='metadata_json'}
  with np.load(old) as f:names=f['names'];proposed=f['pose_w2c'];accepted=f['accepted']
  with np.load(bp) as f:base=f['pose_w2c'];assert np.array_equal(names,f['names'])
  assert np.array_equal(names,bk['names']);values={k:[] for k in ['veto','corroborate','threshold']};accepts={k:[] for k in values};details=[]
  for i,n in enumerate(names.astype(str)):
   lo,hi=bk['offsets'][i:i+2];w,t,xy=canonical_hypotheses(world[bk['prototype_rows'][lo:hi]],bk['query_tokens'][lo:hi],bk['pixels'][lo:hi]);groups=[np.flatnonzero(t==v) for v in np.unique(t)]
   kb=score_pose(base[i],w,xy,groups,bk['camera_matrices'][i],float(bk['radial_k1'][i]))[0];kn=score_pose(proposed[i],w,xy,groups,bk['camera_matrices'][i],float(bk['radial_k1'][i]))[0];a=audit[i];assert a['name']==n
   for mode in values:
    take=bool(accepted[i]) and (a['probabilities'][a['chosen']]>=threshold if mode=='threshold' else heldout_gate(kb,kn,mode));values[mode].append(proposed[i] if take else base[i]);accepts[mode].append(take)
   details.append(dict(name=n,base_score=kb,new_score=kn,prior_accepted=bool(accepted[i])))
  for mode in values:
   p=out/f'{s}_{mode}.npz';assert not p.exists();arr=dict(names=names,pose_w2c=np.array(values[mode]),accepted=np.array(accepts[mode]));meta=dict(artifact_type='heldout_token_pose_gate_v405',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,mode=mode,threshold=threshold if mode=='threshold' else None,source_sha256={str(p):file_sha256(p) for p in [bank,old,bp,Path(__file__),Path('feature_extract/tools/vfm/heldout_pose_gate.py')]},shared_encoder_and_map=True);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(p,**arr,metadata_json=np.array(json.dumps(meta)))
  (out/f'{s}_gate_audit.json').write_text(json.dumps(details));print(label,s,flush=True)
