"""Paired, route-separated summary of immutable structured-memory pose runs."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


THRESHOLDS=[(.1,1),(.25,2),(.5,5),(1,10),(2,45)]

def compare_errors(a,b):
    out=[]
    for t,r in THRESHOLDS:
        x=(a[:,0]<=t)&(a[:,1]<=r);y=(b[:,0]<=t)&(b[:,1]<=r)
        gain=int((~x&y).sum());loss=int((x&~y).sum())
        out.append({'threshold':[t,r],'baseline':int(x.sum()),'method':int(y.sum()),'gained':gain,'lost':loss,
                    'mcnemar_exact_p':float(binomtest(gain,gain+loss).pvalue) if gain+loss else 1.})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    runs=[]
    for seed,version in [(0,227),(1,232)]:
        for arm in ['cosine','base','combined']:
            runs.append((f'transfer_seed{seed}_{arm}',a.root/f'memory_transfer_pose_v{version}_{arm}'/(arm+'.npz')))
    for seed in [0,1]:
        for arm in ['cosine','base','combined','extended_base','extended_combined']:
            runs.append((f'seq9_seed{seed}_{arm}',a.root/f'structured_memory_pose_v223_seed{seed}'/(arm+'.npz')))
    errors={};names_by_run={};gt={};reports={}
    for name,path in runs:
        with np.load(path) as z:names=z['names'].astype(str);poses=z['pose_w2c']
        local=[]
        for n,pose in zip(names,poses):
            if n not in gt:
                with np.load(a.contributors/n) as z:gt[n]=z['pose_w2c']
            local.append(_pose_error(pose,gt[n]))
        e=np.array(local);errors[name]=e;names_by_run[name]=names
        routes=np.array([n.split('__')[0] for n in names]);report={}
        for route in ['all']+sorted(set(routes)):
            rows=np.ones(len(e),bool) if route=='all' else routes==route;x=e[rows]
            report[route]={'images':int(rows.sum()),'hits':[int(((x[:,0]<=t)&(x[:,1]<=r)).sum()) for t,r in THRESHOLDS],
                           'median_translation_m':float(np.median(x[:,0]))}
        reports[name]={'pose_sha256':file_sha256(path),'by_route':report}
    paired={}
    for population in ['seq9','transfer']:
        for seed in [0,1]:
            key=f'{population}_seed{seed}';old=errors[key+'_base'];new=errors[key+'_combined']
            if not np.array_equal(names_by_run[key+'_base'],names_by_run[key+'_combined']):raise ValueError('paired inventory differs')
            paired[key]=compare_errors(old,new)
    report={'reports':reports,'paired_base_to_combined':paired,'unique_image_count':len(gt),
            'repeated_seeds_are_not_independent_images':True,'no_policy_selected_by_GT':True,'same_scene_development_routes_not_multiscene_blind_test':True}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(paired,indent=2))


if __name__=='__main__':main()
