"""Small mapping-only cross-plane association probe, not a deployed null prior."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def token_image_weights(sources,tokens):
    _,inv,count=np.unique(np.c_[sources,tokens],axis=0,return_inverse=True,return_counts=True)
    weight=1/count[inv]
    for s in np.unique(sources):
        rows=sources==s;weight[rows]/=weight[rows].sum()
    return weight/weight.sum()


def fit(x,y,weight):
    def objective(beta):
        z=x@beta
        loss=np.sum(weight*(np.logaddexp(0,z)-y*z))+.0005*np.sum(beta[1:]**2)
        gradient=x.T@(weight*(expit(z)-y));gradient[1:]+=.001*beta[1:]
        return loss,gradient
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not result.success:raise RuntimeError(result.message)
    return result.x


def auc(y,score):
    positive=y==1;n=int(positive.sum());m=len(y)-n
    return float((rankdata(score)[positive].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    c=np.load(a.candidates,allow_pickle=False);l=np.load(a.labels,allow_pickle=False)
    if str(l['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('label lineage differs')
    sources=c['source_image'];tokens=c['query_token'];y=l['labels'];keep=c['homography_keep']
    images=np.unique(sources);train=np.isin(sources,images[::2])&keep&(y>=0)
    evaluation=np.isin(sources,images[1::2])&keep&(y>=0)
    if set(sources[train])&set(sources[evaluation]):raise ValueError('source image leakage')
    x=np.c_[np.ones(len(y)),c['association_features']]
    weight=token_image_weights(sources[train],tokens[train])
    beta=fit(x[train],y[train],weight);score=expit(x@beta)
    cosine_beta=fit(x[train,:2],y[train],weight)
    cosine_probability=expit(x[evaluation,:2]@cosine_beta)
    ev=y[evaluation];pred=score[evaluation]
    candidate_report_path=a.candidates.with_suffix('.json')
    candidate_report=json.loads(candidate_report_path.read_text()) if candidate_report_path.exists() else {}
    if candidate_report and candidate_report.get('candidate_file_sha256')!=file_sha256(a.candidates):raise ValueError('candidate report lineage differs')
    region_policy=candidate_report.get('query_region_policy','mapping_observation_regions')
    report={'scope':'mapping source-image-disjoint association probe after real MNN/homography primitives; '+region_policy,
        'query_region_policy':region_policy,
        'candidate_sha256':file_sha256(a.candidates),'label_sha256':file_sha256(a.labels),
        'train_images':len(np.unique(sources[train])),'evaluation_images':len(np.unique(sources[evaluation])),
        'train_rows':int(train.sum()),'evaluation_rows':int(evaluation.sum()),
        'ignored_ambiguous_rows':int((keep&(y<0)).sum()),'coefficients':beta.tolist(),
        'features':['intercept','RADIO_cosine','plane_rank_div9','homography_residual_div025_clipped8','homography_valid'],
        'fixed_l2':.001,'cosine_auc':auc(ev,c['association_features'][evaluation,0]),'probe_auc':auc(ev,pred),
        'probe_brier':float(np.mean((pred-ev)**2)),
        'cosine_only_brier':float(np.mean((cosine_probability-ev)**2)),
        'cosine_only_bce':float(np.mean(np.logaddexp(0,x[evaluation,:2]@cosine_beta)-ev*(x[evaluation,:2]@cosine_beta))),
        'probe_bce':float(np.mean(np.logaddexp(0,x[evaluation]@beta)-ev*(x[evaluation]@beta))),
        'coordinate_or_descriptor_weights_updated':False,'query_test_data_used':False,
        'limitations':['same mapping route adjacent images, not unseen-route test',
            'binary metrics condition on geometrically unambiguous labels; do not calibrate missing/null candidates',
            'MoGe region mode still restricts inputs to mapping-supervised tokens' if region_policy.startswith('moge') else 'query regions are mapping-defined, not independent MoGe3 segmentation',
            'five-parameter score probe only; no runtime pose promotion']}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
