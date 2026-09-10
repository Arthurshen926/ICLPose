"""Aggregate paired fixed-initializer transfer without selecting a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _load,_errors,_threshold_hits,_temporal_block_ci,THRESHOLDS
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.statistics import mcnemar_exact_pvalue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directories',type=Path,nargs='+',required=True)
    p.add_argument('--contributors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    names=[];all_errors=[];lineage=[]
    for d in a.directories:
        x,_=_load(d/'local_final.npz');y,_=_load(d/'joint_final.npz')
        if not np.array_equal(x['names'],y['names']):raise ValueError('query mismatch')
        names.extend(x['names'].astype(str));all_errors.append(np.array([*_errors(x,a.contributors),*_errors(y,a.contributors)]))
        lineage.append({k:file_sha256(d/k) for k in ['local_final.npz','joint_final.npz','control.json']})
    if len(names)!=len(set(names)):raise ValueError('duplicate query across splits')
    bt,br,mt,mr=np.concatenate(all_errors,axis=1);thresholds=[]
    for t,r in THRESHOLDS:
        old=_threshold_hits(bt,br,t,r);new=_threshold_hits(mt,mr,t,r)
        thresholds.append({'threshold':[t,r],'baseline_hits':int(old.sum()),'method_hits':int(new.sum()),'gains':int((new&~old).sum()),'losses':int((old&~new).sum()),'mcnemar_p':mcnemar_exact_pvalue(old,new)})
    valid=np.isfinite(bt)&np.isfinite(mt);delta=np.full(len(bt),np.nan);delta[valid]=mt[valid]-bt[valid]
    report={'scope':'fixed_initial_coordinate_transfer_not_full_pipeline_or_blind_test','query_count':len(names),'thresholds':thresholds,
            'baseline_median_translation_m':float(np.median(bt)),'method_median_translation_m':float(np.median(mt)),
            'translation_improved_count':int((mt<bt).sum()),'translation_block_ci':{str(k):_temporal_block_ci(names,delta,k,10000,260910) for k in [5,10,20]},'lineage':lineage}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
