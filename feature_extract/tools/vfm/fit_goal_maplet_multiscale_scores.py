"""Freeze scale ablations before selecting map units from held mapping poses."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit,token_image_weights,auc
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def matrix(c,context,shape):
    return np.c_[np.ones(len(context)),c['association_features'][:,:4],context[:,:2],shape[:,3:]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','multiscale','shape','labels','output']:p.add_argument('--'+k,type=Path,required=True)
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
    with np.load(a.labels) as z:
        y=z['labels']
        if str(z['frozen_candidate_sha256'])!=digest:raise ValueError('label lineage differs')
    images=np.unique(c['source_image']);known=c['homography_keep']&(y>=0)
    out=np.zeros((len(y),3),np.float32);models={};metrics={}
    for k,r in enumerate([1,2,4]):
        x=matrix(c,f[:,k],shape);arm='radius'+str(r)
        models[arm]=fit(x[known],y[known],token_image_weights(c['source_image'][known],c['query_token'][known])).tolist()
        for fold in [0,1]:
            train=np.isin(c['source_image'],images[fold::2])&known;ev=~np.isin(c['source_image'],images[fold::2])
            beta=fit(x[train],y[train],token_image_weights(c['source_image'][train],c['query_token'][train]))
            out[ev,k]=x[ev]@beta
        metrics[arm]={'known_oof_auc':auc(y[known],out[known,k])}
    a.output.write_text(json.dumps({'models':models,'metrics':metrics,'training_images':names[images].tolist(),
        'crossfit_scores':True,'candidate_sha256':file_sha256(a.candidates),'multiscale_sha256':file_sha256(a.multiscale)},indent=2))
    np.savez_compressed(a.output.with_suffix('.npz'),logits=out,arm_names=np.array(list(models)),candidate_sha256=np.asarray(file_sha256(a.candidates)))


if __name__=='__main__':main()
