"""Post-label aggregate of corrected consensus versus hash-bound historical poses."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _load, _errors, _threshold_hits, _temporal_block_ci, THRESHOLDS
from feature_extract.tools.vfm.replay_goal_maplet_corrected_baseline import resolve
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256, canonical_json_sha256
from feature_extract.vfm.statistics import mcnemar_exact_pvalue
from feature_extract.tools.vfm.aggregate_goal_maplet_pose_failure_stages import _load_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--contributors_root',type=Path,required=True)
    p.add_argument('--baseline_prefix',help='Optional frozen split-prefix instead of historical authority poses.')
    p.add_argument('--method_prefix',default='corrected_baseline_v183')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    names=[];bt=[];br=[];mt=[];mr=[];splits=[]
    for split in ['seq10','shard0','shard1','shard2','shard3']:
        label=split if split=='seq10' else 'seq13_'+split
        authority=_load_report(a.root/f'pose_failure_attribution_{label}_v114.json')
        ref=authority['frozen_input_lineage']['selected']
        old=(a.root/f'{a.baseline_prefix}_{split}_consensus/selected.npz' if a.baseline_prefix else
             resolve(a.root,ref['content_sha256'],ref['file_sha256'],'*consensus*.npz'))
        new=a.root/f'{a.method_prefix}_{split}_consensus/selected.npz'
        ba,_=_load(old);ma,_=_load(new)
        if not np.array_equal(ba['names'],ma['names']):raise ValueError('query identity differs')
        c=a.contributors_root/('contributors_alltrain_clean' if split=='seq10' else 'contributors_official_test_clean')
        t0,r0=_errors(ba,c);t1,r1=_errors(ma,c)
        names.extend(ba['names'].astype(str));bt.extend(t0);br.extend(r0);mt.extend(t1);mr.extend(r1)
        counts=lambda t,r:[int(_threshold_hits(t,r,x,y).sum()) for x,y in THRESHOLDS]
        splits.append({'split':split,'query_count':len(t0),'historical':counts(t0,r0),'corrected':counts(t1,r1),
            'historical_file':str(old),'historical_sha256':file_sha256(old),'corrected_file':str(new),'corrected_sha256':file_sha256(new)})
    if len(names)!=438 or len(set(names))!=438:raise ValueError('expected 438 unique queries')
    bt,br,mt,mr=map(np.asarray,[bt,br,mt,mr]);thresholds=[]
    for x,y in THRESHOLDS:
        before=_threshold_hits(bt,br,x,y);after=_threshold_hits(mt,mr,x,y)
        gains=int((after&~before).sum());losses=int((before&~after).sum())
        thresholds.append({'threshold':[x,y],'historical_hits':int(before.sum()),'corrected_hits':int(after.sum()),
            'gains':gains,'losses':losses,'mcnemar_p':mcnemar_exact_pvalue(before,after)})
    finite=np.isfinite(mt)&np.isfinite(bt)
    delta=np.full(mt.shape,np.nan)
    delta[finite]=mt[finite]-bt[finite]
    report={'artifact_type':'corrected_baseline_all438_postlabel_audit_v1','query_count':438,
        'evaluation_role':'reused_development_routes_not_blind_test','split_count':5,'route_count':2,
        'baseline_prefix':a.baseline_prefix,'method_prefix':a.method_prefix,
        'scope':'same_frozen_networks_and_correspondences_corrected_backend_and_recomputed_consensus',
        'historical_median_m':float(np.median(bt)),'corrected_median_m':float(np.median(mt)),
        'thresholds':thresholds,'splits':splits,
        'translation_block_ci':{str(k):_temporal_block_ci(names,delta,k,10000,260909) for k in [5,10,20]},
        'production_eligible':False,'query_pose_or_ground_truth_read':True}
    report['content_sha256']=canonical_json_sha256(report)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
