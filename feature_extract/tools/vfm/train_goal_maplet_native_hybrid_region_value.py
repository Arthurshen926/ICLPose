"""Fit hybrid next-region value after all primary-pipeline counterfactuals finish."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.train_goal_maplet_native_region_marginal_value import fit_ridge,predict
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--loss',choices=['bounded_reward','log_pose_cost'],default='bounded_reward');a=p.parse_args();o=a.output;o.mkdir(exist_ok=False,parents=True);assert (a.input/'complete.json').exists()
 feature_path=a.features or a.input/'features.npz'
 with np.load(feature_path) as z:
  names=z['names'].astype(str);x=z['features'];feature_meta=json.loads(str(z['metadata_json'])) if 'metadata_json' in z else {}
  if feature_meta:
   assert arrays_sha256({k:z[k] for k in z.files if k!='metadata_json'})==feature_meta['arrays_sha256']
   assert canonical_json_sha256({k:v for k,v in feature_meta.items() if k!='content_sha256'})==feature_meta['content_sha256']
 width=x.shape[-1];contract=feature_meta.get('feature_contract','hybrid19');assert (contract,width) in [('hybrid19',19),('pose33',33)] and np.isfinite(x).all()
 with np.load(a.input/'features.npz') as z:assert np.array_equal(names,z['names'].astype(str))
 assert feature_meta.get('query_pose_ground_truth_used',False) is False
 poses={};hashes={}
 for route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']:
  for arm in range(9):
   path=a.input/route/f'arm{arm}'/'final.npz';arr,m=_load_pose_candidate(path);hashes[str(path)]=file_sha256(path)
   for n,pose in zip(arr['names'].astype(str),arr['pose_w2c']):poses[(n,arm)]=pose
 if contract=='pose33':
  assert feature_meta['inferred_pose_stage']=='native8_primary_final'
  assert feature_meta['base_feature_sha256']==file_sha256(a.input/'features.npz')
  for route,sha in feature_meta['base_pose_sha256'].items():assert sha==hashes[str(a.input/route/'arm0/final.npz')]
 (o/'frozen_hashes.json').write_text(json.dumps(hashes,indent=2));errors=[];contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')
 for n in names:
  with np.load(contributors/n) as z:gt=z['pose_w2c']
  errors.append([_pose_error(poses[(n,arm)],gt) for arm in range(9)])
 errors=np.asarray(errors);reward=np.exp(-errors[:,:,0]/.5-errors[:,:,1]/5);reward[~np.isfinite(reward)]=0
 if a.loss=='log_pose_cost':reward=-np.minimum(np.log1p(errors[:,:,0]/.25)+np.log1p(errors[:,:,1]/2),20.);reward[~np.isfinite(reward)]=-20.
 target=reward[:,1:]-reward[:,:1];routes=np.array([n.split('__')[0] for n in names]);train=np.isin(routes,['seq1','seq2','seq4','seq6']);assert train.sum()==80 and (~train).sum()==60
 model=fit_ridge(x[train].reshape(-1,width),target[train].reshape(-1));pred=predict(model,x.reshape(-1,width)).reshape(len(names),8);choice=pred.argmax(1)+1;stop=np.where(pred.max(1)>0,choice,0);indices=dict(base8=np.zeros(len(names),int),retrieval9=np.ones(len(names),int),novel9=x[:,:,16].argmax(1)+1,learned9=choice,learned_stop=stop,oracle_diagnostic=reward.argmax(1));reports={}
 for split,mask in [('train',train),('heldout',~train)]:
  reports[split]={}
  for arm,c in indices.items():reports[split][arm]=dict(metrics(errors[np.arange(len(names)),c][mask]),mean_reward=float(reward[np.arange(len(names)),c][mask].mean()),stop_rate=float((c[mask]==0).mean()))
 model.update(feature_contract=contract,feature_provenance=feature_meta,query_GT_used=False,heldout_used_to_fit=False,training_routes=['seq1','seq2','seq4','seq6'],heldout_routes=['seq7','seq8','seq11'],excluded_mapping_route='seq9',target=('delta exp(-translation/.5-rotation/5), invalid reward zero' if a.loss=='bounded_reward' else 'reduction in min(log1p(t/.25)+log1p(r/2),20); invalid cost20'),target_stage='primary final refinement, no sparse/dense verifier',frozen_hashes_sha256=file_sha256(o/'frozen_hashes.json'),feature_file_sha256=file_sha256(feature_path))
 (o/'model.json').write_text(json.dumps(model,indent=2));(o/'report.json').write_text(json.dumps(reports,indent=2));np.savez_compressed(o/'offline_evaluation.npz',names=names,errors=errors,training_mask=train,features=x,prediction=pred,target=target,**indices)
 print(json.dumps(reports['heldout'],indent=2),flush=True)

if __name__=='__main__':main()
