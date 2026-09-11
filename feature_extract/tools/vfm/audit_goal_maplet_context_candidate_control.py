"""Verify context adds a score without changing any frozen measurement or label."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def compare(baseline, context):
    with np.load(baseline,allow_pickle=False) as a,np.load(context,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('candidate schema differs')
        equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files if k!='association_features'}
        equal['original_association_features']=bool(np.array_equal(a['association_features'],b['association_features'][:,:4]))
        if not all(equal.values()):raise ValueError(f'context changed candidates: {equal}')
        if b['association_features'].shape!=(len(b['query_token']),5) or not np.isfinite(b['association_features']).all():
            raise ValueError('invalid context feature')
    for p in [baseline,context]:
        with np.load(p.with_suffix('.labels.npz')) as labels:
            if str(labels['frozen_candidate_sha256'])!=file_sha256(p):raise ValueError('label lineage mismatch')
    with np.load(baseline.with_suffix('.labels.npz')) as a,np.load(context.with_suffix('.labels.npz')) as b:
        for k in ['labels','supervision_valid','supervision_available']:
            if not np.array_equal(a[k],b[k]):raise ValueError('supervision changed')
    return {'identical_candidate_arrays':equal,'identical_supervision':True,
            'baseline_sha256':file_sha256(baseline),'context_sha256':file_sha256(context),
            'scope':'fixed candidate and coordinate control; does not evaluate pose accuracy'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','context','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=compare(a.baseline,a.context)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
