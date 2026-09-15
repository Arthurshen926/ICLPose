"""Evaluate complete, sealed inference inventories; never choose configurations."""
import json,argparse
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT,PROTOCOL
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
ARMS=PROTOCOL['arms']+['pooled_multistart']
SCENES=['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']

def paired(a,b):
    out={}
    rng=np.random.default_rng(415)
    for t,r in [(.1,1),(.25,2),(.5,5)]:
        x=(a[:,0]<=t)&(a[:,1]<=r);y=(b[:,0]<=t)&(b[:,1]<=r);d=y.astype(float)-x.astype(float);samples=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
        out[f'{t}m_{r}deg']=dict(delta_pp=float(d.mean()*100),gain_rate_percent=float(np.mean(~x&y)*100),harm_rate_percent=float(np.mean(x&~y)*100),paired_iid_bootstrap_95ci_pp=np.percentile(samples,[2.5,97.5]).tolist(),caveat='temporally correlated video frames; iid CI is descriptive, not independent-scene significance')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=260901);a=p.parse_args();seal={};report={};allerrors={k:[] for k in ARMS}
    # Must have every official query endpoint before loading ANY query pose label.
    for scene in SCENES:
        r=ROOT/scene;names=json.load(open(r/'test_names.json'));d=r/f'predictions_seed{a.seed}';actual={p.name for p in d.glob('*.npz')};assert actual==set(names),(scene,len(actual),len(names));seal[scene]={n:file_sha256(d/n) for n in names};control=r/f'multistart_seed{a.seed}';assert {p.name for p in control.glob('*.npz')}==set(names);seal[scene].update({'control/'+n:file_sha256(control/n) for n in names})
    (ROOT/f'endpoint_seal_seed{a.seed}.json').write_text(json.dumps(seal,indent=2))
    for scene in SCENES:
        r=ROOT/scene;names=json.load(open(r/'test_names.json'));gtfile=Path('/hy-tmp/Cambridge_stdloc')/scene/'dataset_test.txt';gt={x.image_id:x.pose_w2c for x in parse_cambridge_pose_file(gtfile)};assert set(gt)=={n[:-4].replace('__','/') for n in names};errors={k:[] for k in ARMS};audits=[]
        for n in names:
            f=r/f'predictions_seed{a.seed}'/n;assert file_sha256(f)==seal[scene][n]
            with np.load(f) as z:
                for arm in PROTOCOL['arms']:errors[arm].append(_pose_error(z[arm],gt[n[:-4].replace('__','/')]) if np.isfinite(z[arm]).all() else (np.inf,np.inf))
                audits.append(json.loads(z['metadata_json'].item()))
            control=r/f'multistart_seed{a.seed}'/n;assert file_sha256(control)==seal[scene]['control/'+n]
            with np.load(control) as z:errors['pooled_multistart'].append(_pose_error(z['pooled_multistart'],gt[n[:-4].replace('__','/')]) if np.isfinite(z['pooled_multistart']).all() else (np.inf,np.inf))
        errors={k:np.array(v) for k,v in errors.items()};result={k:metrics(v) for k,v in errors.items()};comparisons={f'{b}_vs_{a}':paired(errors[a],errors[b]) for a,b in [('pooled','regional'),('pooled_multistart','regional'),('regional','regional_fine'),('regional','regional_agreement')]}
        report[scene]=dict(metrics=result,paired_comparisons=comparisons,ground_truth_sha256=file_sha256(gtfile),seconds_per_query_mean=float(np.mean([v['seconds'] for v in audits])),refinement_accept_rate=float(np.mean([v.get('refinement',{}).get('accepted',False) for v in audits])),agreement_accept_rate=float(np.mean([v.get('agreement_accept',False) for v in audits])))
        np.savez_compressed(r/f'errors_seed{a.seed}.npz',names=np.array(names),**errors)
        for k,v in errors.items():allerrors[k].append(v)
    macro={k:{threshold:float(np.mean([report[s]['metrics'][k]['recall_percent'][threshold] for s in SCENES])) for threshold in report[SCENES[0]]['metrics'][k]['recall_percent']} for k in allerrors}
    result=dict(seed=a.seed,scope='portable core mechanism assay, not full v414 system benchmark',scenes=report,macro_scene_recall_percent=macro,pooled_images_metrics={k:metrics(np.concatenate(v)) for k,v in allerrors.items()},all_query_means_keep_infinite_failures=True)
    (ROOT/f'report_seed{a.seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps({s:r['metrics']['regional_agreement'] for s,r in report.items()},indent=2))
if __name__=='__main__':main()
