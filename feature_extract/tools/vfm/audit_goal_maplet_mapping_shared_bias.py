"""Signed per-source-image joint-coordinate bias, not a pose-error estimator."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def grouped_bias(residual, sources):
    residual=np.asarray(residual,dtype=np.float64);sources=np.asarray(sources)
    if residual.shape!=(len(sources),2) or not np.isfinite(residual).all():
        raise ValueError('invalid paired residuals')
    result=[]
    for source in np.unique(sources):
        values=residual[sources==source];mean=values.mean(axis=0)
        bias=float(mean@mean);centered=float(np.mean(np.sum((values-mean)**2,axis=1)))
        mse=float(np.mean(np.sum(values**2,axis=1)))
        if not np.isclose(mse,bias+centered):raise ValueError('bias decomposition failed')
        result.append({'source':str(source),'rows':len(values),'signed_mean_px':mean.tolist(),
                       'bias_norm_px':float(np.sqrt(bias)),'mse_px2':mse,'centered_mse_px2':centered})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    banks=[np.load(path,allow_pickle=False) for path in a.banks]
    for key in ['evaluation_rows','calibration_rows','source_views']:
        np.testing.assert_array_equal(banks[0][key],banks[1][key])
    rows=banks[0]['evaluation_rows'];views=banks[0]['source_views'][rows]
    results=[grouped_bias(b['residual'][rows],views) for b in banks]
    before,after=[np.array([r['bias_norm_px'] for r in result]) for result in results]
    report={'artifact_type':'mapping_shared_image_bias_audit_v1','bank_sha256':[file_sha256(p) for p in a.banks],
        'evaluation_images':len(before),'evaluation_rows':len(rows),'heads':results,
        'mean_per_image_bias_norm_px':[float(before.mean()),float(after.mean())],
        'bias_norm_improved_images':int((after<before).sum()),'bias_norm_worsened_images':int((after>before).sum()),
        'query_data_used':False,'limitations':[
            'known-cell mapping candidates, not actual cross-plane retrieval',
            'multiple candidate modes share query tokens; row counts are not independent observations',
            'banks lack plane IDs and camera Jacobians; not per-plane bias or shared-pose consistency',
            'source image mean residual is descriptive; it does not establish pose-error causality']}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='heads'},indent=2))


if __name__=='__main__':main()
