"""Post-label initializer generation/retention audit; never selects a policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_stage,_pose_error
from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import THRESHOLDS
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--contributors_root',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['seq10','shard0','shard1','shard2','shard3'])
    p.add_argument('--guided_version',type=int,default=None)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    reports=[]
    for split in a.splits:
        c=a.contributors_root/('contributors_alltrain_clean' if split=='seq10' else 'contributors_official_test_clean')
        version=183 if split in ['seq10','shard3'] else 182
        paths={'corrected_opencv':a.root/f'corrected_baseline_v{version}_{split}_primary/pnp.npz'}
        for policy in ['uniform','geometry','geometry_score']:
            v=186 if split=='seq10' else 192
            if policy!='uniform' and a.guided_version is not None:v=a.guided_version
            paths[policy]=a.root/f'token_sampling_v{v}_{split}_{policy}.npz'
        for policy,path in paths.items():
            bank,meta=_pose_stage(path);selected=[];pool_hits=np.zeros(5,int);top4_hits=np.zeros(5,int)
            for i,name in enumerate(bank['names'].astype(str)):
                with np.load(c/name,allow_pickle=False) as z:gt=z['pose_w2c']
                selected.append(_pose_error(bank['raw_inliers_pose_w2c'][i],gt))
                lo,hi=bank['candidate_offsets'][i:i+2];poses=bank['candidate_pose_w2c'][lo:hi]
                errors=np.array([_pose_error(pose,gt) for pose in poses])
                order=np.argsort(-bank['candidate_inlier_count'][lo:hi],kind='stable')[:4]
                for j,(t,r) in enumerate(THRESHOLDS):
                    hit=(errors[:,0]<=t)&(errors[:,1]<=r)
                    pool_hits[j]+=int(hit.any());top4_hits[j]+=int(hit[order].any())
            errors=np.array(selected);records=meta.get('sampling_records',[])
            reports.append({'split':split,'policy':policy,'queries':len(errors),'sha256':file_sha256(path),
                'sampling_implementation':meta.get('token_sampling_implementation'),
                'selected_hits':[int(((errors[:,0]<=t)&(errors[:,1]<=r)).sum()) for t,r in THRESHOLDS],
                'pool_oracle_hits':pool_hits.tolist(),'top4_oracle_hits':top4_hits.tolist(),
                'median_translation_m':float(np.median(errors[:,0])),
                'scored_hypotheses':sum(v.get('scored_hypotheses',0) for v in records),
                'budget_unreached_groups':sum(not v.get('budget_reached',False) for v in records),
                'solve_seconds':sum(v.get('seconds',0) for v in records)})
    totals={}
    for policy in paths:
        subset=[r for r in reports if r['policy']==policy]
        totals[policy]={key:np.sum([r[key] for r in subset],axis=0).tolist() for key in
            ['queries','selected_hits','pool_oracle_hits','top4_oracle_hits','scored_hypotheses','budget_unreached_groups','solve_seconds']}
    report={'reports':reports,'totals':totals,'query_labels_used_only_for_audit':True,
        'top4_rule':'stable_descending_unique_token_inlier_count_not_GT',
        'limitations':['pool/top4 are postlabel upper bounds, not deployed selections',
            'OpenCV baseline is contextual, not a matched internal-hypothesis budget',
            'guided and uniform arms match finite scored model count, not unique models or wall time',
            'reused routes are not independent blind evaluation']}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
