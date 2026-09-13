from pathlib import Path
import json,joblib,numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');old=b/'spatial_verifier_v404';out=b/'decision_risk_v405_expanded';out.mkdir(exist_ok=True);fit=joblib.load(old/'pairwise_verifier.joblib');model=fit['model'];scale=fit['metadata']['scale'];minimum=fit['metadata']['threshold']
def decisions(route):
 records=json.load(open(old/'mapping'/(route+'.json')))['records'];result=[]
 for name in dict.fromkeys(a['name'] for a in records):
  rows=[a for a in records if a['name']==name];f=np.array([a['features'] for a in rows]);utility=model.decision_function(f);j=int(utility.argmax())
  for i,a in enumerate(rows):
   if i==j or not separated_mode(np.array(a['pose']),np.array(rows[j]['pose'])):continue
   e0=a['error'];e1=rows[j]['error'];harm=any(e0[0]<=t and e0[1]<=r and not(e1[0]<=t and e1[1]<=r) for t,r in [(.25,2),(.5,5)])
   result.append(dict(name=name,base=i,chosen=j,p=float(expit(scale*(utility[j]-utility[i]))),harm=harm,base_error=e0,new_error=e1))
 return result
cal=decisions('seq7')+decisions('seq8');threshold=float(np.nextafter(1.,2.))
for t in sorted(set([minimum]+[x['p'] for x in cal if x['p']>=minimum])):
 accepted=[x for x in cal if x['p']>=t]
 if len(set(x['name'] for x in accepted))>=5 and not any(x['harm'] for x in accepted):threshold=float(t);break
meta=dict(threshold=threshold,previous_threshold=minimum,calibration_routes=['seq7','seq8'],confirmation_routes=['seq11'],query_routes_read=False,criterion='no observed loss of 25cm/2deg or50cm/5deg under any available mapping baseline; >=5 queries accepted; empirical only',mapping_baseline_is_candidate_proxy=True,independent_confidence_guarantee=False)
report={}
for route in ['seq7','seq8','seq11']:
 d=decisions(route);accepted=[x for x in d if x['p']>=threshold];report[route]=dict(decisions=d,accepted_query_fraction=len(set(x['name'] for x in accepted))/20,harmed_query_fraction=len(set(x['name'] for x in accepted if x['harm']))/20)
(out/'calibration.json').write_text(json.dumps(meta,indent=2));(out/'mapping_decisions.json').write_text(json.dumps(report,indent=2));print(meta);print({r:{k:v for k,v in a.items() if k!='decisions'} for r,a in report.items()})
