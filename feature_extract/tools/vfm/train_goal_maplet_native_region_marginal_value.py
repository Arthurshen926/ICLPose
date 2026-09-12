"""Fit next-region value from frozen mapping-only pose counterfactuals.

Fixed first-eight-region state; this is not a general sequential set policy.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def fit_ridge(x,y,ridge=.1):
 mean=x.mean(0);scale=np.maximum(x.std(0),1e-5);a=np.c_[np.ones(len(x)),(x-mean)/scale];penalty=np.eye(a.shape[1])*ridge;penalty[0,0]=0
 beta=np.linalg.solve(a.T@a/len(a)+penalty,a.T@y/len(a))
 return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),ridge=ridge)


def predict(model,x):
 return np.c_[np.ones(len(x)),(x-np.array(model['mean']))/np.array(model['scale'])]@np.array(model['beta'])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output;o.mkdir(exist_ok=False,parents=True);protocol=json.load(open(a.input/'protocol.json'));assert protocol['query_pose_or_ground_truth_read'] is False
 paths=sorted((a.input/'frozen').glob('*.npz'));assert len(paths)==160;poses=[];features=[];hashes={}
 for path in paths:
  with np.load(path) as z:d={k:z[k] for k in z.files if k!='metadata_json'};m=json.loads(str(z['metadata_json']))
  assert m['query_pose_or_ground_truth_read'] is False and m['source_rows_sha256']==file_sha256(a.input/'rows'/path.name)
  assert arrays_sha256(d)==m['arrays_sha256'] and canonical_json_sha256({k:v for k,v in m.items() if k!='content_sha256'})==m['content_sha256']
  poses.append(d['poses']);features.append(d['features']);hashes[path.name]=file_sha256(path)
 # Seal every inference artifact before opening any pose label.
 (o/'frozen_input_hashes.json').write_text(json.dumps(hashes,indent=2))
 names=np.array([p.name for p in paths]);assert not any(n.startswith(('seq10__','seq13__')) for n in names);errors=[];contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')
 for name,ps in zip(names,poses):
  with np.load(contributors/name) as z:gt=z['pose_w2c']
  errors.append([_pose_error(p,gt) for p in ps])
 errors=np.array(errors);x=np.array(features);reward=np.exp(-errors[:,:,0]/.5-errors[:,:,1]/5);reward[~np.isfinite(reward)]=0;target=reward[:,1:]-reward[:,:1];routes=np.array([n.split('__')[0] for n in names]);train=np.isin(routes,['seq1','seq2','seq4','seq6']);assert train.sum()==80
 model=fit_ridge(x[train].reshape(-1,x.shape[-1]),target[train].reshape(-1));pred=predict(model,x.reshape(-1,x.shape[-1])).reshape(160,8);choice=pred.argmax(1)+1;stop=np.where(pred.max(1)>0,choice,0);rng=np.random.default_rng(260912);random=rng.integers(1,9,len(names));indices=dict(base8=np.zeros(len(names),int),retrieval9=np.ones(len(names),int),random9=random,learned9=choice,learned_stop=stop,oracle_diagnostic=reward.argmax(1));reports={}
 for split,mask in [('train',train),('heldout',~train)]:
  reports[split]={}
  for arm,idx in indices.items():
   e=errors[np.arange(len(names)),idx][mask];reports[split][arm]=dict(metrics(e),mean_bounded_pose_reward=float(reward[np.arange(len(names)),idx][mask].mean()),stop_rate=float((idx[mask]==0).mean()))
 model.update(scope=__doc__,target='exp(-t/.5-r/5) difference relative to fixed first eight; invalid pose reward zero',query_GT_used=False,heldout_used_to_fit=False,training_routes=['seq1','seq2','seq4','seq6'],heldout_routes=['seq7','seq8','seq9','seq11'],frozen_inputs_sha256=file_sha256(o/'frozen_input_hashes.json'),feature_names=['added_rows','added_unique_tokens','novel_tokens','base_tokens','added_cosine_mean','added_cosine_max','base_cosine_mean','context_max','context_mean','context_marginal_gain','nearest_center_distance','token_x_spread','token_y_spread'])
 (o/'model.json').write_text(json.dumps(model,indent=2));(o/'report.json').write_text(json.dumps(reports,indent=2));np.savez_compressed(o/'offline_evaluation.npz',names=names,errors=errors,reward=reward,target=target,prediction=pred,training_mask=train,**indices)
 print(json.dumps(reports['heldout'],indent=2),flush=True)

if __name__=='__main__':main()
