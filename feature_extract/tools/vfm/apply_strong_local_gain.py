"""Apply a mapping-only update selector where the frozen strong base still holds."""
from pathlib import Path
import json,numpy as np,joblib,os,sys
from feature_extract.tools.vfm.strong_local_selection import choose_local
from feature_extract.tools.vfm.verification_bank_contract import validate_poses
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def main():
 blocked=[]
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0]))
   if ('contributors' in str(p) and p.name.startswith(('seq10__','seq13__'))) or p.name in ['metrics.json','comparisons.json','mapping_confirmation.json']:
    blocked.append(str(p));raise PermissionError('evaluation input is prohibited during inference')
 sys.addaudithook(guard)
 b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'strong_local_precision_v414';mp=r/'gain_model/selector.joblib';fit=joblib.load(mp);assert fit['metadata']['query_routes_read'] is False and fit['metadata']['mapping_baseline_is_candidate_proxy'] is False
 for seed in [1,2]:
  out=r/f'query_seed{seed}';out.mkdir(exist_ok=False)
  for s in ['seq10','shard0','shard1','shard2','shard3']:
   up=r/'query/pairs'/f'{s}_unlabelled.json';f=json.load(open(up));bp=Path(f['base_path']);assert file_sha256(bp)==f['sources'][str(bp)];current=b/f'eligible_verification_v413/seed{seed}/dual_support/uniform_verified'/f'{s}_confirmed.npz'
   with np.load(bp) as z:names=z['names'];base=z['pose_w2c']
   with np.load(current) as z:np.testing.assert_array_equal(names,z['names']);cur=z['pose_w2c'];validate_poses(cur)
   candidates={}
   for arm,path in f['candidate_paths'].items():
    assert file_sha256(Path(path))==f['sources'][path]
    with np.load(path) as z:np.testing.assert_array_equal(names,z['names']);candidates[arm]=z['pose_w2c']
   groups={str(n):[] for n in names}
   for x in f['pairs']:groups[x['name']].append(x)
   values={arm:[] for arm in ['learned','fine_mean','plain','robust']};accepted={k:[] for k in values};audit=[]
   for i,n in enumerate(names.astype(str)):
    rows=groups[n];scores=fit['model'].predict(np.array([x['features'] for x in rows]));bound=np.array_equal(base[i],cur[i]);choices={}
    for policy in values:
     if not bound:j=-1
     elif policy=='learned':j=choose_local(rows,scores,fit['metadata']['threshold'],fit['metadata']['enabled'])
     elif policy=='fine_mean':j=choose_local(rows,[x['features'][6] for x in rows],np.nextafter(0.,1.))
     else:
      target='stage_plain_reg_all' if policy=='plain' else 'stage_robust_reg_all';j=next((j for j,x in enumerate(rows) if x['arm']==target and x['valid_local_pair']),-1)
     pose=candidates[rows[j]['arm']][i] if j>=0 else cur[i];values[policy].append(pose);accepted[policy].append(j>=0);choices[policy]=rows[j]['arm'] if j>=0 else None
    audit.append(dict(name=n,base_exactly_matches_current=bound,choices=choices,predicted_gain=scores.tolist(),support=[x['support'] for x in rows]))
   sources={str(p):file_sha256(p) for p in [up,bp,current,mp,Path(__file__),Path('feature_extract/tools/vfm/strong_local_selection.py')]+[Path(p) for p in f['candidate_paths'].values()]}
   for policy in values:
    arr=dict(names=names,pose_w2c=np.array(values[policy]),accepted=np.array(accepted[policy]));validate_poses(arr['pose_w2c']);meta=dict(artifact_type='strong_endpoint_local_update_v414',arrays_sha256=arrays_sha256(arr),source_sha256=sources,query_pose_or_ground_truth_read=False,policy=policy,base_bound_to_real_v357=True,preserves_current_robust_endpoint_unless_exact_base_match=True,predicted_gain_not_probability=True,independent_risk_guarantee=False);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(out/f'{s}_{policy}.npz',**arr,metadata_json=np.array(json.dumps(meta)))
   (out/f'{s}_audit.json').write_text(json.dumps(audit));print(seed,s,'complete',flush=True)
  (out/'read_guard.json').write_text(json.dumps(dict(prohibited_reads=blocked,query_label_and_metric_reads_blocked=True)))


if __name__=='__main__':main()
