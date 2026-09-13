from pathlib import Path
import argparse,json,numpy as np,joblib,sys,os
from scipy.special import expit
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.local_precision_evidence import paired_features
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.verification_bank_contract import validate_bank,validate_poses
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['control','control_seed2','appearance','appearance_seed2']);a=p.parse_args();v=a.variant
blocked=[]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0]));s=str(path)
  if ('contributors' in s and path.name.startswith(('seq10__','seq13__'))) or path.name in ['metrics.json','comparison.json','candidate_pool_oracle.json']:
   blocked.append(s);raise PermissionError('query label/metric read prohibited')
sys.addaudithook(guard)
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'local_precision_v409';o=r/v;o.mkdir(exist_ok=True);maps,meta,common=load_map(b);mp=b/'native_fine_v264/readout/map.npz';mh=file_sha256(mp);modelpath=r/'local_selector.joblib';fit=joblib.load(modelpath);assert not fit['metadata']['query_routes_read'];common[str(modelpath)]=file_sha256(modelpath);seed=2 if v.endswith('seed2') else 1
for s in ['seq10','shard0','shard1','shard2','shard3']:
 bp=b/'heldout_evidence_v405'/f'mnn_seed{seed}'/f'{s}_corroborate.npz';oldpath=b/'overlap_lod_v402/lod_fixed_final'/f'{s}_mnn_audit.json';newpath=b/'overlap_lod_v402/lod_seed2_final'/f'{s}_mnn_audit.json' if v.startswith('control') else b/'shared_identity_v408'/('pose_'+v)/'lod_final'/f'{s}_identity_mnn_audit.json';bank=b/'heldout_evidence_v405/banks'/f'{s}.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz'
 with np.load(bp) as f:names=f['names'];base=f['pose_w2c'];validate_poses(base)
 with np.load(bank) as f:bk={k:f[k] for k in f.files if k!='metadata_json'};bm=json.loads(f['metadata_json'].item())
 with np.load(cp) as f:validate_bank(bk,bm,maps['world'],mp,mh,meta['atlas_sha256'],names,f['camera_matrices'],f['radial_k1'])
 old=json.load(open(oldpath));new=json.load(open(newpath));sources=dict(common);sources.update({str(p):file_sha256(p) for p in [bp,oldpath,newpath,bank,cp,Path(__file__),Path('feature_extract/tools/vfm/local_precision_evidence.py'),Path('feature_extract/tools/vfm/select_goal_maplet_separated_modes.py'),Path('feature_extract/tools/vfm/verification_bank_contract.py')]});audit=[];values={arm:[] for arm in ['learned','fine_mean']};accepted={arm:[] for arm in values}
 for i,n in enumerate(names.astype(str)):
  assert old[i]['name']==new[i]['name']==n
  pool=[np.array(p) for p in old[i]['refined_candidate_poses']+new[i]['refined_candidate_poses'] if p is not None and not separated_mode(base[i],np.array(p))]
  cache=b/'native_fine_v264/query_cache'/n;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];sources[str(cache)]=file_sha256(cache);lo,hi=bk['offsets'][i:i+2];ids=bk['prototype_rows'][lo:hi];valid=maps['available'][ids];ids=ids[valid];xy=bk['pixels'][lo:hi][valid];t=bk['query_tokens'][lo:hi][valid];X=[];counts=[]
  for pose in pool:
   f,count=paired_features(base[i],pose,maps['world'][ids],xy,t,bk['camera_matrices'][i],float(bk['radial_k1'][i]),grids,[maps['coarse_map'][ids],maps['fine_map'][ids]]);X.append(f);counts.append(count)
  probabilities=expit(fit['metadata']['scale']*fit['model'].decision_function(np.array(X))) if X else np.array([]);choices={}
  for arm in values:
   score=probabilities.copy() if arm=='learned' else np.array([x[6] for x in X]);threshold=fit['metadata']['threshold'] if arm=='learned' else 0.
   if len(score):score[np.array(counts)<6]=-np.inf
   j=int(score.argmax()) if len(score) else -1;take=j>=0 and (score[j]>=threshold if arm=='learned' else score[j]>threshold)
   values[arm].append(pool[j] if take else base[i]);accepted[arm].append(bool(take));choices[arm]=j if take else -1
  audit.append(dict(name=n,choices=choices,support=counts,features=[x.tolist() for x in X],probabilities=probabilities.tolist(),candidate_poses=[p.tolist() for p in pool]))
 for arm in values:
  dest=o/f'{s}_{arm}.npz';assert not dest.exists();arr=dict(names=names,pose_w2c=np.array(values[arm]),accepted=np.array(accepted[arm]));validate_poses(arr['pose_w2c']);md=dict(artifact_type='local_paired_precision_v409',arm=arm,variant=v,arrays_sha256=arrays_sha256(arr),source_sha256=sources,query_pose_or_ground_truth_read=False,local_only=True,preserves_v405_if_abstain=True,independent_observation_claim=False);md['content_sha256']=canonical_json_sha256(md);np.savez_compressed(dest,**arr,metadata_json=np.array(json.dumps(md)))
 (o/f'{s}_audit.json').write_text(json.dumps(audit));print(v,s,flush=True)
(o/'read_guard.json').write_text(json.dumps(dict(prohibited_reads=blocked,query_labels_and_metrics_blocked=True)))
