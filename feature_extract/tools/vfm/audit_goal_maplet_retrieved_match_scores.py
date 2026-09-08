"""Held mapping match-score audit on identical mined negatives; no deployment prior claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def metrics(positive,negative):
    p=np.clip(np.asarray(positive,float).reshape(-1),1e-7,1-1e-7)
    n=np.clip(np.asarray(negative,float).reshape(-1),1e-7,1-1e-7)
    if not len(p) or len(p)!=len(n) or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('invalid match audit scores')
    ranks=rankdata(np.r_[p,n]);count=len(p)
    return {'balanced_bce':float(-.5*np.mean(np.log(p)+np.log1p(-n))),
            'positive_recall_at_half':float(np.mean(p>=.5)),
            'negative_false_positive_at_half':float(np.mean(n>=.5)),
            'sampled_auc':float((ranks[:count].sum()-count*(count+1)/2)/(count*count)),
            'pairwise_positive_beats_negative':float(np.mean(p>n))}


def positive_recall_gate(bank):
    calibration=bank['calibration_rows'];evaluation=bank['evaluation_rows']
    if set(bank['source_views'][calibration])&set(bank['source_views'][evaluation]):
        raise ValueError('score gate calibration leakage')
    positive=bank['positive_probability'].reshape(-1)
    negative=bank['negative_probability'].reshape(-1)
    threshold=float(np.quantile(positive[calibration],.05))
    valid=np.isfinite(negative[evaluation])
    return {'head_content_sha256':str(bank['head_content_sha256']),
            'query_data_used':False,'threshold':threshold,'calibration_positive_recall_target':.95,
            'evaluation_positive_recall':float(np.mean(positive[evaluation]>=threshold)),
            'evaluation_negative_false_positive':float(np.mean(negative[evaluation][valid]>=threshold)),
            'semantics':'mapping_positive_quantile_rejection_not_deployment_probability'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--banks',type=Path,nargs='+',required=True)
    p.add_argument('--heads',type=Path,nargs='+',required=True)
    p.add_argument('--gate_output',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    banks=[]
    for path in args.banks:
        with np.load(path) as z:banks.append({k:z[k] for k in z.files})
    if len(args.heads)!=len(banks):raise ValueError('head/bank count differs')
    heads=[]
    for path,bank in zip(args.heads,banks):
        with np.load(path) as z:head=json.loads(str(z['metadata_json']))
        if head['content_sha256']!=str(bank['head_content_sha256']):raise ValueError('bank/head lineage differs')
        heads.append(head)
    for head in heads[1:]:
        for key in ('observation_bank_file_sha256','radio_projection_file_sha256','mapping_contributor_inventory_sha256',
                    'prototype_policy','null_mining_policy'):
            if head[key]!=heads[0][key]:raise ValueError('mining input/configuration differs: '+key)
        for split in ('fit','validation'):
            a=head['retrieved_negative_audit'][split].copy();b=heads[0]['retrieved_negative_audit'][split].copy()
            ah=a.pop('selected_pool_rows_and_valid_sha256',None);bh=b.pop('selected_pool_rows_and_valid_sha256',None)
            if a!=b or (ah is not None and bh is not None and ah!=bh):raise ValueError('mined candidates differ')
    first=banks[0];rows=first['evaluation_rows']
    for bank in banks[1:]:
        for key in ('evaluation_rows','source_views'):
            np.testing.assert_array_equal(bank[key],first[key])
    valid=np.ones(len(rows),bool)
    for bank in banks:valid &= np.isfinite(bank['negative_probability'].reshape(-1)[rows])
    rows=rows[valid]
    result={'common_held_pairs':len(rows),'query_data_used':False,'deployment_probability_calibrated':False,
            'heads':[],'mining_configuration_and_diagnostics_identical':True,
            'limitation':'balanced constructed positives/negatives, same known physical plane; not deployed prior calibration'}
    for path,bank in zip(args.banks,banks):
        result['heads'].append({'head_content_sha256':str(bank['head_content_sha256']),
            'bank_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'metrics':metrics(bank['positive_probability'].reshape(-1)[rows],bank['negative_probability'].reshape(-1)[rows])})
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if args.gate_output is not None:
        args.gate_output.write_text(json.dumps(positive_recall_gate(banks[-1]),indent=2)+'\n')


if __name__=='__main__':main()
