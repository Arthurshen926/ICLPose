"""All-query dense-only ablation of sparse gating; not an identity verifier.

Both poses use the same complete query-valid normal denominator. Pose arrays
are frozen before any external evaluation tool opens query labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import _load_render,_normal_good_ray_score
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def select_dense(scores,usable):
    scores=np.asarray(scores,float);usable=np.asarray(usable,bool)
    if scores.ndim!=2 or scores.shape[1]!=2 or usable.shape!=scores.shape or np.isnan(scores).any() or np.isposinf(scores).any() or np.any((scores[np.isfinite(scores)]<0)|(scores[np.isfinite(scores)]>1)):
        raise ValueError('invalid paired dense scores')
    selected=np.zeros(len(scores),np.int8)
    selected[(~usable[:,0])&usable[:,1]]=1
    selected[usable.all(1)&(scores[:,1]>scores[:,0])]=1
    # This is missing/zero evidence, not calibrated physical-identity rejection.
    insufficient=~np.any(usable&np.isfinite(scores)&(scores>0),axis=1)
    return selected,insufficient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses',type=Path,nargs=2,required=True)
    p.add_argument('--renders',type=Path,nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    poses=[_load_pose_candidate(x) for x in a.poses];renders=[_load_render(x) for x in a.renders];names=poses[0][0]['names'].astype(str)
    for j,(pa,pm) in enumerate(poses):
        r=renders[j]
        if not np.array_equal(names,pa['names'].astype(str)) or [x['name'] for x in r['rows']]!=names.tolist():raise ValueError('query alignment differs')
        if r['frozen_pose_inventory_file_sha256']!=file_sha256(a.poses[j]) or r['frozen_pose_inventory_content_sha256']!=pm['content_sha256']:raise ValueError('render endpoint binding differs')
        if r.get('sparse_first_render_plan_file_sha256') is not None or any('dense_score_evaluated' in x for x in r['rows']):raise ValueError('dense-only control requires ungated complete rendering')
    for key in ['physical_map_file_sha256','query_camera_inventory_file_sha256','moge3_manifest_file_sha256_in_order']:
        if renders[0].get(key) is None or renders[0][key]!=renders[1][key]:raise ValueError('paired dense evidence differs: '+key)
    usable=np.stack([pa['usable'] for pa,_ in poses],axis=1).astype(bool)
    for i in np.flatnonzero(usable.all(1)):
        if renders[0]['rows'][i].get('query_valid_pixel_count')!=renders[1]['rows'][i].get('query_valid_pixel_count'):raise ValueError('paired fixed query denominator differs')
    scores=np.array([[_normal_good_ray_score(r['rows'][i]) for r in renders] for i in range(len(names))]);choice,insufficient=select_dense(scores,usable);idx=np.arange(len(names))
    arr=dict(names=poses[0][0]['names'],pose_w2c=np.stack([pa['pose_w2c'] for pa,_ in poses],axis=1)[idx,choice],usable=usable[idx,choice],selected_branch=choice,candidate_dense_score=scores,insufficient_positive_dense_evidence=insufficient)
    meta=dict(artifact_type='goal_maplet_full_dense_gate_ablation_v1',query_pose_or_ground_truth_read=False,arrays_sha256=arrays_sha256(arr),source_sha256={str(p):file_sha256(p) for p in a.poses+a.renders},selection='strict dense normal good-ray improvement; ties retain primary; no sparse gate',scope='uncalibrated geometric control, no claim of identity correctness or independent evidence')
    meta['content_sha256']=canonical_json_sha256(meta);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)))


if __name__=='__main__':main()
