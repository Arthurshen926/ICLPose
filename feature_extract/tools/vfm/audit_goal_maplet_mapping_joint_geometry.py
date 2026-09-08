"""Mapping-only error decomposition; oracle coordinates are diagnostic, never deployed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(bank):
    rows=bank['evaluation_rows']
    image=bank['image_error'][rows].astype(float)
    world=bank['projected_world_error'][rows].astype(float)
    normal=bank['offplane_image_error'][rows].astype(float)
    uv=world-normal
    residual=bank['residual'][rows].astype(float)
    np.testing.assert_allclose(residual,image-uv-normal,atol=1e-10)
    errors={'predicted_both':residual,'oracle_image_only':-world,
            'oracle_uv_only':image-normal,'oracle_image_and_uv':-normal,
            'oracle_normal_only':image-uv}
    result={'evaluation_count':len(rows),'query_data_used':False,
            'head_content_sha256':str(bank['head_content_sha256']),
            'normal_residual_m_median_p90_p99':np.percentile(bank['normal_residual_m'][rows],[50,90,99]).tolist(),
            'coordinate_oracles_are_diagnostics_not_runtime_measurements':True,'errors':{}}
    for name,error in errors.items():
        norm=np.linalg.norm(error,axis=1)
        result['errors'][name]={'median_p90_p99_px':np.percentile(norm,[50,90,99]).tolist(),
                               'mean_squared_px':float(np.mean(norm**2))}
    terms={'image':np.sum(image**2,axis=1),'uv':np.sum(uv**2,axis=1),'normal':np.sum(normal**2,axis=1),
           'image_uv_cross':-2*np.sum(image*uv,axis=1),'image_normal_cross':-2*np.sum(image*normal,axis=1),
           'uv_normal_cross':2*np.sum(uv*normal,axis=1)}
    result['squared_residual_decomposition_px2']={k:float(np.mean(v)) for k,v in terms.items()}
    assert np.isclose(sum(result['squared_residual_decomposition_px2'].values()),np.mean(np.sum(residual**2,axis=1)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks',type=Path,nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    banks=[]
    for path in args.banks:
        with np.load(path,allow_pickle=False) as z:banks.append({k:z[k] for k in z.files})
    for key in ('source_views','calibration_rows','evaluation_rows'):
        np.testing.assert_array_equal(banks[0][key],banks[1][key])
    rows=banks[0]['evaluation_rows']; views=banks[0]['source_views'][rows]
    delta=np.linalg.norm(banks[1]['residual'][rows],axis=1)-np.linalg.norm(banks[0]['residual'][rows],axis=1)
    image_delta=np.array([np.mean(delta[views==v]) for v in np.unique(views)])
    rng=np.random.default_rng(260918)
    boot=image_delta[rng.integers(len(image_delta),size=(10000,len(image_delta)))].mean(axis=1)
    report={'heads':[summarize(b) for b in banks],
            'bank_file_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.banks],
            'paired_local_minus_baseline_mean_error_px_equal_image_weight':float(np.mean(image_delta)),
            'source_image_bootstrap_95ci':np.percentile(boot,[2.5,97.5]).tolist(),
            'evaluation_source_images':len(image_delta),
            'limitations':['mapping-only held images, not query localization accuracy',
                           'source-image bootstrap does not remove adjacent-view temporal correlation',
                           'oracle UV does not alter anchor normal height']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
