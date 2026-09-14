"""Learn continuous signed update utility from frozen real mapping endpoints."""
from pathlib import Path
import json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from feature_extract.tools.vfm.strong_local_selection import calibrate_forward,choose_local,threshold_harm
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
 b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'strong_local_precision_v414';out=r/'gain_model';out.mkdir(exist_ok=False);routes=['seq1','seq2','seq4','seq6','seq7','seq8','seq11'];data={s:json.load(open(r/'model'/f'{s}_labelled.json'))['pairs'] for s in routes};X=[];y=[];names=[]
 for s in routes[:4]:
  for x in data[s]:
   if not x['valid_local_pair'] or x['support']<6:continue
   a=max(x['base_error'][0]/.1,x['base_error'][1]);c=max(x['candidate_error'][0]/.1,x['candidate_error'][1])
   if not np.isfinite([a,c]).all():continue
   gain=(a-c)/max(a,c,1e-12)
   for sign in [1,-1]:X.append(sign*np.array(x['features']));y.append(sign*gain);names.append(x['name'])
 X=np.asarray(X);y=np.asarray(y);names=np.array(names);weight=np.array([1/np.sum(names==n) for n in names]);weight*=len(weight)/weight.sum()
 model=make_pipeline(StandardScaler(with_mean=False),Ridge(alpha=1.,fit_intercept=False));model.fit(X,y,ridge__sample_weight=weight)
 def grouped(rows):
  g={}
  for x in rows:g.setdefault(x['name'],[]).append(x)
  groups=list(g.values());return groups,[model.predict(np.array([x['features'] for x in group])) for group in groups]
 groups,scores=grouped(data['seq7']);threshold,enabled=calibrate_forward(groups,scores,minimum_score=0.)
 meta=dict(training_routes=routes[:4],calibration_routes=['seq7'],confirmation_routes=['seq8','seq11'],query_routes_read=False,mapping_baseline_is_candidate_proxy=False,base_endpoint='source-excluded mapping v357',threshold=threshold,enabled=enabled,model_output='signed normalized utility gain, not probability',utility='u=max(translation_m/0.1,rotation_deg); target=(u_base-u_candidate)/max(u_base,u_candidate,1e-12)',training_directed_examples=len(X),reverse_pairs_for_fit_only=True,threshold_rule='actual forward seq7: nonnegative predicted utility, no observed 10/25/50cm harm, >=5 distinct queries; otherwise explicitly disabled',risk_guarantee=False,source_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('feature_extract/tools/vfm/strong_local_selection.py')]+[r/'model'/f'{s}_labelled.json' for s in routes]})
 joblib.dump(dict(model=model,metadata=meta),out/'selector.joblib');(out/'metadata.json').write_text(json.dumps(meta,indent=2));report={}
 for s in routes[4:]:
  groups,scores=grouped(data[s]);decisions=[]
  for g,score in zip(groups,scores):
   j=choose_local(g,score,threshold,enabled);decisions.append(dict(name=g[0]['name'],choice=g[j]['arm'] if j>=0 else None,base_error=g[0]['base_error'],error=g[j]['candidate_error'] if j>=0 else g[0]['base_error'],harm=bool(j>=0 and threshold_harm(g[j]['base_error'],g[j]['candidate_error'])),scores=score.tolist()))
  report[s]=decisions
 (out/'mapping_confirmation.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in meta.items() if k!='source_sha256'})


if __name__=='__main__':main()
