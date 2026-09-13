from pathlib import Path
import json,numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from scipy.special import expit
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'spatial_verifier_v404';routes=['seq1','seq2','seq4','seq6','seq7','seq8','seq11'];data={x:json.load(open(r/'mapping'/(x+'.json')))['records'] for x in routes}
def pairs(records):
 X=[];y=[];weights=[]
 for name in dict.fromkeys(a['name'] for a in records):
  a=[v for v in records if v['name']==name];local=[]
  for i in range(len(a)):
   for j in range(i):
    e1=max(a[i]['error'][0]/.5,a[i]['error'][1]/5);e2=max(a[j]['error'][0]/.5,a[j]['error'][1]/5)
    if max(e1,e2)<1.25*max(min(e1,e2),1e-8):continue
    diff=np.array(a[i]['features'])-a[j]['features']
    if np.linalg.norm(diff)<1e-10:continue
    local.extend([(diff,e1<e2),(-diff,e2<e1)])
  for diff,label in local:X.append(diff);y.append(label);weights.append(1/len(local))
 return np.array(X),np.array(y),np.array(weights)
X,y,weight=pairs(sum([data[x] for x in routes[:4]],[]));weight*=len(weight)/weight.sum();model=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=.1,fit_intercept=False,max_iter=1000,random_state=404));model.fit(X,y,logisticregression__sample_weight=weight)
xc,yc,wc=pairs(data['seq7']);z=model.decision_function(xc);fit=minimize_scalar(lambda a:np.average(np.logaddexp(0,a*z)-yc*a*z,weights=wc),bounds=(.001,20),method='bounded');scale=float(fit.x);p=expit(scale*z);threshold=1.
for t in sorted(set(p[p>=.5])):
 mask=p>=t
 if mask.sum()>=20 and np.average(yc[mask],weights=wc[mask])>=.95:threshold=float(t);break
meta=dict(training_routes=routes[:4],calibration_routes=['seq7'],confirmation_routes=routes[5:],query_routes_read=False,scale=scale,threshold=threshold,label='relative max(t/0.5,r/5) improvement of >=25%; actual mapping candidates only',threshold_rule='seq7 query-balanced pair precision >=95%, >=20 directed pairs, threshold>=0.5; not confidence guarantee',pair_count=len(y))
joblib.dump(dict(model=model,metadata=meta),r/'pairwise_verifier.joblib');(r/'pairwise_metadata.json').write_text(json.dumps(meta,indent=2));report={}
for route in routes[4:]:
 x,y,w=pairs(data[route]);p=expit(scale*model.decision_function(x));keep=p>=threshold;report[route]=dict(pair_accuracy=float(np.average((p>=.5)==y,weights=w)),accepted_pair_fraction=float(np.average(keep,weights=w)),accepted_precision=float(np.average(y[keep],weights=w[keep])) if keep.any() else None)
(r/'pairwise_mapping_validation.json').write_text(json.dumps(report,indent=2));print(meta);print(report)
