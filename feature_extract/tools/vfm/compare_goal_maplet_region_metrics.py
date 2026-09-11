"""Paired route-block uncertainty for regional pose improvements, using rates."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1]);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    reports=json.loads(a.metrics.read_text())['reports'];out={};rng=np.random.default_rng(260912)
    for seed in a.seeds:
        baseline=reports[f'baseline_s{seed}_moge'];old=np.asarray(baseline['errors']);names=np.asarray(baseline['names']);routes=np.array([n.split('__')[0] for n in names])
        samplings={}
        for length in [5,10,20]:
            parts=[]
            for route in sorted(set(routes)):
                rows=np.flatnonzero(routes==route);starts=rng.integers(0,len(rows),size=(3000,int(np.ceil(len(rows)/length))))
                positions=(starts[:,:,None]+np.arange(length))%len(rows);parts.append(rows[positions.reshape(3000,-1)[:,:len(rows)]])
            samplings[length]=np.concatenate(parts,axis=1)
        for label,report in reports.items():
            if not label.startswith('r') or not label.endswith(f's{seed}_moge'):continue
            if report['names']!=baseline['names']:raise ValueError('query inventory differs')
            new=np.asarray(report['errors']);comparison={}
            if not np.isfinite(new).all() or not np.isfinite(old).all():raise ValueError('unconditional infinite-error CI needs explicit treatment')
            for length,idx in samplings.items():
                row={}
                for dim,quantity in [(0,'translation_m'),(1,'rotation_deg')]:
                    for stat,fun in [('mean',np.mean),('median',np.median)]:
                        delta=fun(new[:,dim])-fun(old[:,dim]);boot=fun(new[idx,dim],axis=1)-fun(old[idx,dim],axis=1)
                        row[quantity+'_'+stat]={'method_minus_baseline':float(delta),'ci95':np.quantile(boot,[.025,.975]).tolist()}
                for t,r in [(.1,1),(.25,2),(.5,5),(1,10),(2,45)]:
                    diff=((new[:,0]<=t)&(new[:,1]<=r)).astype(float)-((old[:,0]<=t)&(old[:,1]<=r)).astype(float)
                    row[f'recall_{t}m_{r}deg']={'percentage_point_change':float(diff.mean()*100),'ci95':(np.quantile(diff[idx].mean(1),[.025,.975])*100).tolist()}
                comparison[str(length)]=row
            out[label]=comparison
    a.output.write_text(json.dumps({'comparisons':out,'resamples':3000,'block_lengths':[5,10,20],
        'resampling':'paired circular blocks within each route, seeds reported separately',
        'scope':'same-scene development evidence; multiple comparisons are exploratory'},indent=2))


if __name__=='__main__':main()
