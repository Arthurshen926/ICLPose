"""Fit only mapping routes after real strong-endpoint pairs were frozen."""
from pathlib import Path
import argparse,json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from scipy.special import expit
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.strong_local_selection import calibrate_forward,choose_local,threshold_harm
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
 b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'strong_local_precision_v414';parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args();out=args.output or r/'model';out.mkdir(exist_ok=False);routes=['seq1','seq2','seq4','seq6','seq7','seq8','seq11'];data={};sources={};gt=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')
 for s in routes:
  up=r/'mapping/pairs'/f'{s}_unlabelled.json';f=json.load(open(up));assert f['mapping_baseline_is_candidate_proxy'] is False
  for path in [f['base_path'],*f['candidate_paths'].values()]:assert file_sha256(Path(path))==f['sources'][path]
  with np.load(f['base_path']) as z:names=z['names'].astype(str);base=z['pose_w2c'];assert all(n.startswith(s+'__') for n in names)
  cand={}
  for arm,path in f['candidate_paths'].items():
   with np.load(path) as z:np.testing.assert_array_equal(names,z['names']);cand[arm]=z['pose_w2c']
  targets=[]
  for n in names:
   lp=gt/n
   with np.load(lp) as z:targets.append(z['pose_w2c'])
   sources[str(lp)]=file_sha256(lp)
  index={n:i for i,n in enumerate(names)};rows=f['pairs']
  for x in rows:
   i=index[x['name']];x.update(base_error=_pose_error(base[i],targets[i]),candidate_error=_pose_error(cand[x['arm']][i],targets[i]))
  labelpath=out/f'{s}_labelled.json';labelpath.write_text(json.dumps(dict(pairs=rows,unlabelled_sha256=file_sha256(up),mapping_pose_labels_joined_after_freeze=True),indent=2));sources[str(up)]=file_sha256(up);sources[str(labelpath)]=file_sha256(labelpath);data[s]=rows
 def examples(rows):
  X=[];y=[];names=[]
  for x in rows:
   if not x['valid_local_pair'] or x['support']<6:continue
   e0=max(x['base_error'][0]/.1,x['base_error'][1]);e1=max(x['candidate_error'][0]/.1,x['candidate_error'][1])
   if not np.isfinite([e0,e1]).all() or max(e0,e1)<1.25*max(min(e0,e1),1e-8):continue
   for sign in [1,-1]:X.append(sign*np.asarray(x['features']));y.append(e1<e0 if sign==1 else e0<e1);names.append(x['name'])
  names=np.asarray(names);weights=np.array([1/np.sum(names==n) for n in names]);weights=weights*len(weights)/weights.sum() if len(weights) else weights
  return np.asarray(X),np.asarray(y),weights
 X,y,w=examples(sum([data[s] for s in routes[:4]],[]))
 if len(X)==0 or len(np.unique(y))<2:
  status=dict(enabled=False,reason='no informative mapping pairs under fixed 25 percent margin',training_directed_examples=len(X),query_routes_read=False,mapping_baseline_is_candidate_proxy=False)
  (out/'metadata.json').write_text(json.dumps(status,indent=2));print(status);return
 model=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=.1,fit_intercept=False,max_iter=1000,random_state=414));model.fit(X,y,logisticregression__sample_weight=w)
 cx,cy,cw=examples(data['seq7']);scale=float(minimize_scalar(lambda a:np.average(np.logaddexp(0,a*model.decision_function(cx))-cy*a*model.decision_function(cx),weights=cw),bounds=(.001,20),method='bounded').x) if len(cx) else 1.
 def grouped(rows):
  groups={}
  for x in rows:groups.setdefault(x['name'],[]).append(x)
  groups=list(groups.values());return groups,[expit(scale*model.decision_function(np.array([x['features'] for x in g]))) for g in groups]
 groups,prob=grouped(data['seq7']);threshold,enabled=calibrate_forward(groups,prob)
 meta=dict(training_routes=routes[:4],calibration_routes=['seq7'],confirmation_routes=['seq8','seq11'],query_routes_read=False,mapping_baseline_is_candidate_proxy=False,base_endpoint='full source-excluded mapping v357',scale=scale,threshold=threshold,enabled=enabled,training_directed_examples=len(X),calibration_directed_examples=len(cx),reverse_pairs_for_fit_only=True,threshold_rule='actual forward seq7 decisions: no observed 10/25/50cm harm and >=5 distinct queries; otherwise explicitly disabled',risk_guarantee=False,source_sha256=sources)
 joblib.dump(dict(model=model,metadata=meta),out/'selector.joblib');(out/'metadata.json').write_text(json.dumps(meta,indent=2));report={}
 for s in routes[4:]:
  groups,prob=grouped(data[s]);decisions=[]
  for g,p in zip(groups,prob):
   j=choose_local(g,p,threshold,enabled);decisions.append(dict(name=g[0]['name'],choice=g[j]['arm'] if j>=0 else None,base_error=g[0]['base_error'],error=g[j]['candidate_error'] if j>=0 else g[0]['base_error'],harm=bool(j>=0 and threshold_harm(g[j]['base_error'],g[j]['candidate_error']))))
  report[s]=decisions
 (out/'mapping_confirmation.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in meta.items() if k!='source_sha256'})


if __name__=='__main__':main()
