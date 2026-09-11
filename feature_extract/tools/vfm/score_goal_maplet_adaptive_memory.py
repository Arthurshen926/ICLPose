"""Apply frozen offline unit choices with equal descriptor and dot-product budgets."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_multiscale_scores import matrix
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','multiscale','shape','scale_models','base_models','policy','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    digest=file_sha256(a.candidates)
    with np.load(a.multiscale) as z:
        f=z['features'];names=z['source_names']
        if str(z['candidate_sha256'])!=digest:raise ValueError('multiscale lineage differs')
    with np.load(a.shape) as z:
        shape=z['features']
        if str(z['candidate_sha256'])!=digest:raise ValueError('shape lineage differs')
    with np.load(a.policy) as z:policy={k:z[k] for k in z.files}
    if not np.array_equal(policy['prototype_world'],c['prototype_world']):raise ValueError('policy map differs')
    models=json.loads(a.scale_models.read_text());base=json.loads(a.base_models.read_text())
    training=set(models['training_images'])
    if set(names[np.unique(c['source_image'])])&training:raise ValueError('transfer overlaps model training')
    scores=np.column_stack([matrix(c,f[:,k],shape)@np.asarray(models['models']['radius'+str(r)]) for k,r in enumerate([1,2,4])])
    baseline=np.c_[np.ones(len(f)),c['association_features'][:,:4]]@np.asarray(base['models']['base'])
    pr=c['prototype_rows'];chosen=policy['choice'][pr];adaptive=scores[np.arange(len(f)),chosen]
    arms=['fixed1','fixed2','fixed4','adaptive','fixed25','adaptive25','adaptive_scale_fixed25','fixed_scale_learned25']
    out=np.column_stack([scores,adaptive,np.where(policy['fixed_budget25'][pr],scores[:,1],baseline),np.where(policy['budget25'][pr],adaptive,baseline),
        np.where(policy['fixed_budget25'][pr],adaptive,baseline),np.where(policy['budget25'][pr],scores[:,1],baseline)])
    np.savez_compressed(a.output,logits=out,arm_names=np.asarray(arms),candidate_sha256=np.asarray(file_sha256(a.candidates)))
    kept=c['homography_keep']
    a.output.with_suffix('.json').write_text(json.dumps({'arms':arms,'training_overlap':False,'policy_sha256':file_sha256(a.policy),
        'selected_candidate_fraction_budget25':float(policy['budget25'][pr[kept]].mean()),
        'selected_candidate_fraction_fixed25':float(policy['fixed_budget25'][pr[kept]].mean()),
        'candidate_or_geometry_changes':False,'query_pose_or_labels_opened':False,'map_choice_is_fixed_for_all_queries':True},indent=2))


if __name__=='__main__':main()
