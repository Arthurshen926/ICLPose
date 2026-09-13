from pathlib import Path
import json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.optimize import minimize_scalar
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'local_precision_v409';routes=['seq1','seq2','seq4','seq6','seq7','seq8','seq11'];data={s:json.load(open(r/'mapping'/f'{s}.json'))['pairs'] for s in routes}
def directed(rows):
 out=[]
 for x in rows:
  if x['support']<6:continue
  out.append(x);out.append(dict(x,base=x['candidate'],candidate=x['base'],features=(-np.array(x['features'])).tolist(),base_error=x['candidate_error'],candidate_error=x['base_error']))
 return out
def examples(rows):
 xx=[];yy=[];names=[]
 for x in directed(rows):
  a=max(x['base_error'][0]/.1,x['base_error'][1]);b=max(x['candidate_error'][0]/.1,x['candidate_error'][1])
  if max(a,b)<1.25*max(min(a,b),1e-8):continue
  xx.append(x['features']);yy.append(b<a);names.append(x['name'])
 names=np.array(names);weight=np.array([1/np.sum(names==n) for n in names]);weight*=len(weight)/weight.sum()
 return np.array(xx),np.array(yy),weight
X,y,w=examples(sum([data[s] for s in routes[:4]],[]));model=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=.1,fit_intercept=False,max_iter=1000,random_state=409));model.fit(X,y,logisticregression__sample_weight=w)
x,y,w=examples(data['seq7']);z=model.decision_function(x);fit=minimize_scalar(lambda a:np.average(np.logaddexp(0,a*z)-y*a*z,weights=w),bounds=(.001,20),method='bounded');scale=float(fit.x)
def decisions(rows):
 grouped={}
 for x in directed(rows):grouped.setdefault((x['name'],x['base']),[]).append(x)
 out=[]
 for group in grouped.values():
  p=expit(scale*model.decision_function(np.array([x['features'] for x in group])));j=int(p.argmax());x=group[j];e0=x['base_error'];e1=x['candidate_error'];harm=any(e0[0]<=t and e0[1]<=d and not(e1[0]<=t and e1[1]<=d) for t,d in [(.1,1),(.25,2),(.5,5)])
  out.append(dict(x,probability=float(p[j]),harm=harm))
 return out
cal=decisions(data['seq7']);threshold=1.
for t in sorted(set([.5]+[x['probability'] for x in cal if x['probability']>=.5])):
 accepted=[x for x in cal if x['probability']>=t]
 if len(set(x['name'] for x in accepted))>=5 and not any(x['harm'] for x in accepted):threshold=t;break
meta=dict(training_routes=routes[:4],calibration_routes=['seq7'],confirmation_routes=['seq8','seq11'],query_routes_read=False,scale=scale,threshold=threshold,mapping_baseline_is_candidate_proxy=True,threshold_rule='seq7: no observed 10/25/50cm harm across available local baselines, at least5 distinct queries; else abstain',independent_risk_guarantee=False,source_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('feature_extract/tools/vfm/local_precision_evidence.py')]+[r/'mapping'/f'{s}.json' for s in routes]})
joblib.dump(dict(model=model,metadata=meta),r/'local_selector.joblib');(r/'model_metadata.json').write_text(json.dumps(meta,indent=2));report={}
for s in routes[4:]:
 d=decisions(data[s]);a=[x for x in d if x['probability']>=threshold];x,y,w=examples(data[s]);report[s]=dict(pair_accuracy_percent=100*np.average((model.decision_function(x)>0)==y,weights=w),accepted_query_fraction=len(set(x['name'] for x in a))/20,harmed_query_fraction=len(set(x['name'] for x in a if x['harm']))/20,decisions=d)
(r/'mapping_validation.json').write_text(json.dumps(report,indent=2));print(meta);print({s:{k:v for k,v in x.items() if k!='decisions'} for s,x in report.items()})
