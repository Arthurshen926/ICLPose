"""Mapping-only fixed-token, fixed-Jacobian shared-pose error projection."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def linear_bias(jacobian,residual,depth_scale):
    j=np.asarray(jacobian,float).reshape(-1,6).copy();r=np.asarray(residual,float).reshape(-1)
    if j.shape[0]!=len(r) or not np.isfinite(j).all() or not np.isfinite(r).all() or depth_scale<=0:
        raise ValueError('invalid shared pose input')
    j[:,3:]*=depth_scale
    u,s,vt=np.linalg.svd(j,full_matrices=False)
    if len(s)<6 or s[-1]<=s[0]*1e-6:return None
    coefficients=(u.T@r)/s;delta=vt.T@coefficients;delta[3:]*=depth_scale
    return {'translation_bias_m':float(np.linalg.norm(delta[3:])),
            'rotation_bias_deg':float(np.rad2deg(np.linalg.norm(delta[:3]))),
            'weakest_mode_bias_dimensionless':float(abs(coefficients[-1])),
            'condition_number_dimensionless':float(s[0]/s[-1])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    banks=[np.load(path,allow_pickle=False) for path in a.banks]
    for key in ['source_views','token_ids','plane_ids','evaluation_rows']:
        np.testing.assert_array_equal(banks[0][key],banks[1][key])
    b=banks[0];evalrows=b['evaluation_rows'];images=[];planes=[]
    for source in np.unique(b['source_views'][evalrows]):
        rows=evalrows[b['source_views'][evalrows]==source]
        # The first physically enumerated row is fixed without looking at errors.
        _,first=np.unique(b['token_ids'][rows],return_index=True);rows=rows[first]
        for plane in np.unique(b['plane_ids'][rows]):
            selected=rows[b['plane_ids'][rows]==plane]
            if len(selected)<3:continue
            means=[bank['residual'][selected].mean(axis=0) for bank in banks]
            planes.append({'source':int(source),'plane':int(plane),'tokens':len(selected),
                           'signed_mean_px':[m.tolist() for m in means],
                           'bias_norm_px':[float(np.linalg.norm(m)) for m in means]})
        if len(rows)<6 or len(np.unique(b['plane_ids'][rows]))<2:continue
        depth=float(np.median(b['camera_points'][rows,2]))
        result=[linear_bias(b['pose_jacobian'][rows],bank['residual'][rows],depth) for bank in banks]
        if result[0] is not None:
            images.append({'source':int(source),'tokens':len(rows),'heads':result})
    before=np.array([v['heads'][0]['translation_bias_m'] for v in images]);after=np.array([v['heads'][1]['translation_bias_m'] for v in images])
    report={'bank_sha256':[file_sha256(p) for p in a.banks],'query_data_used':False,
        'fixed_first_mode_per_source_token':True,'jacobian_frozen_to_first_head':True,
        'usable_multi_plane_source_images':len(images),'plane_groups':len(planes),
        'linear_translation_bias_median_m':[float(np.median(before)),float(np.median(after))],
        'linear_translation_bias_improved_images':int((after<before).sum()),
        'linear_translation_bias_worsened_images':int((after>before).sum()),
        'plane_bias_improved_groups':sum(p['bias_norm_px'][1]<p['bias_norm_px'][0] for p in planes),
        'plane_bias_worsened_groups':sum(p['bias_norm_px'][1]>p['bias_norm_px'][0] for p in planes),
        'images':images,'planes':planes,
        'limitations':['conditional correct-cell mapping examples; not retrieved query candidates',
            'linear unweighted error projection at mapping pose; not nonlinear deployed pose accuracy',
            'first-mode support is a fixed diagnostic, not a proposed runtime selector',
            'camera Jacobian parameterization is left camera SE3; translation scaled by median depth for conditioning']}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['images','planes']},indent=2))


if __name__=='__main__':main()
