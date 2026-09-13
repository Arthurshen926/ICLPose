from pathlib import Path
import json,joblib,numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'spatial_verifier_v404';data={route:json.load(open(r/'mapping'/(route+'.json')))['records'] for route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']}
train=sum([data[x] for x in ['seq1','seq2','seq4','seq6']],[]);X=np.array([a['features'] for a in train]);y=np.array([a['positive'] for a in train]);names=np.array([a['name'] for a in train]);weights=np.array([1/np.sum(names==n) for n in names]);weights*=len(weights)/weights.sum();model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=404));model.fit(X,y,logisticregression__sample_weight=weights)
cal=data['seq7'];z=model.decision_function([a['features'] for a in cal]);ycal=np.array([a['positive'] for a in cal],float)
fit=minimize(lambda ab:np.mean(np.logaddexp(0,ab[0]*z+ab[1])-ycal*(ab[0]*z+ab[1])),[1.,0.],bounds=[(.001,20),(-20,20)],method='L-BFGS-B');scale,bias=fit.x

def selected(records):
 z=model.decision_function([a['features'] for a in records]);p=expit(scale*z+bias);result=[]
 for name in dict.fromkeys(a['name'] for a in records):
  ids=[i for i,a in enumerate(records) if a['name']==name];j=max(ids,key=lambda i:p[i]);result.append(dict(name=name,p=float(p[j]),positive=records[j]['positive'],error=records[j]['error']))
 return result
chosen=selected(cal);threshold=1.;valid_thresholds=[]
for t in sorted(set(x['p'] for x in chosen)):
 keep=[x for x in chosen if x['p']>=t]
 if len(keep)>=5 and np.mean([x['positive'] for x in keep])>=.95:valid_thresholds.append(t)
if valid_thresholds:threshold=min(valid_thresholds)
metadata=dict(training_routes=['seq1','seq2','seq4','seq6'],calibration_routes=['seq7'],confirmation_routes=['seq8','seq11'],query_routes_read=False,calibration=[float(scale),float(bias)],threshold=float(threshold),threshold_rule='minimum seq7 selected-query probability with >=5 accepted queries and >=95% precision; 1 if unavailable',calibration_is_finite_sample_not_guarantee=True,sources={str(r/'mapping'/(x+'.json')):file_sha256(r/'mapping'/(x+'.json')) for x in data},features=json.load(open(r/'mapping/seq7.json'))['feature_names'],unit_noise_covariance_is_not_calibrated_uncertainty=True)
joblib.dump(dict(model=model,metadata=metadata),r/'verifier.joblib');(r/'model_metadata.json').write_text(json.dumps(metadata,indent=2));report={}
for route in ['seq7','seq8','seq11']:
 records=selected(data[route]);keep=[x for x in records if x['p']>=threshold];report[route]=dict(selected=records,coverage=len(keep)/20,precision=float(np.mean([x['positive'] for x in keep])) if keep else None)
(r/'mapping_validation.json').write_text(json.dumps(report,indent=2));print('threshold',threshold,'train',len(train),'positive',y.mean());print({k:{a:b for a,b in v.items() if a!='selected'} for k,v in report.items()})
