"""All-query mean/median pose errors and recall rates, including failures."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def metrics(errors):
    e=np.asarray(errors,float);thresholds=[(.1,1),(.25,2),(.5,5),(1,10),(2,45)]
    valid=np.isfinite(e).all(1)
    return {'images':len(e),'invalid_pose_rate':float(1-valid.mean()),
        'translation_mean_m':float(np.mean(e[:,0])),'translation_median_m':float(np.median(e[:,0])),
        'rotation_mean_deg':float(np.mean(e[:,1])),'rotation_median_deg':float(np.median(e[:,1])),
        'recall_percent':{f'{t}m_{r}deg':float(np.mean((e[:,0]<=t)&(e[:,1]<=r))*100) for t,r in thresholds}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['manifest','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sources=json.loads(a.manifest.read_text());gt={};reports={}
    for label,path in sources.items():
        with np.load(path) as z:names=z['names'].astype(str);poses=z['pose_w2c']
        errors=[]
        for name,pose in zip(names,poses):
            if name not in gt:
                with np.load(a.contributors/name) as z:gt[name]=z['pose_w2c']
            errors.append(_pose_error(pose,gt[name]))
        e=np.array(errors);routes=np.array([n.split('__')[0] for n in names])
        reports[label]={'all':metrics(e),'by_route':{r:metrics(e[routes==r]) for r in sorted(set(routes))},
            'errors':e.tolist(),'names':names.tolist(),'pose_sha256':file_sha256(path)}
    a.output.write_text(json.dumps({'reports':reports,'mean_policy':'all queries; invalid estimates retain infinite error, never silently excluded'},indent=2))


if __name__=='__main__':main()
