"""Fixed ablations and image-crossfit scores for structured local memory."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit,auc,token_image_weights
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def metrics(y,z):
    if not len(y):return {'rows':0}
    prob=expit(z)
    return {'rows':len(y),'positive':int((y==1).sum()),'auc':auc(y,z),
            'brier':float(np.mean((prob-y)**2)),'bce':float(np.mean(np.logaddexp(0,z)-y*z))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','features','labels','extended_labels','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.npz').exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.features) as z:
        f=z['features']
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('feature lineage differs')
    with np.load(a.labels) as z:y=z['labels']
    with np.load(a.extended_labels) as z:
        ey=z['labels']
        if str(z['frozen_candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('extended labels differ')
    sources=c['source_image'];tokens=c['query_token'];keep=c['homography_keep'];images=np.unique(sources)
    base=np.c_[np.ones(len(y)),c['association_features'][:,:4]]
    matrices={'base':base,'unordered':np.c_[base,f[:,0]],'ordered':np.c_[base,f[:,:2]],
              'reflected':np.c_[base,f[:,0],f[:,1]-f[:,2]],'moge':np.c_[base,f[:,3:]],
              'combined':np.c_[base,f[:,:2],f[:,3:]],'extended_base':base,'extended_combined':np.c_[base,f[:,:2],f[:,3:]]}
    scores=np.zeros((len(y),len(matrices)),np.float32);reports={};coefs={}
    for col,(name,x) in enumerate(matrices.items()):
        target=ey if name.startswith('extended') else y
        models=[]
        for fold in [0,1]:
            train=np.isin(sources,images[fold::2])&keep&(target>=0);evaluation=~np.isin(sources,images[fold::2])
            beta=fit(x[train],target[train],token_image_weights(sources[train],tokens[train]))
            scores[evaluation,col]=x[evaluation]@beta
            models.append({'trained_images':images[fold::2].tolist(),'coefficients':beta.tolist()})
        coefs[name]=models
        odd=np.isin(sources,images[1::2]);known=keep&(y>=0);new=keep&(y==-2)&(ey>=0)
        reports[name]={'old_known_eval49':metrics(y[odd&known],scores[odd&known,col]),
            'new_pseudo_eval49':metrics(ey[odd&new],scores[odd&new,col]),
            'all_known_crossfit98':metrics(y[known],scores[known,col]),
            'all_extended_crossfit98':metrics(ey[keep&(ey>=0)],scores[keep&(ey>=0),col])}
    np.savez_compressed(a.output.with_suffix('.npz'),logits=scores,arm_names=np.array(list(matrices)),candidate_sha256=np.asarray(file_sha256(a.candidates)))
    report={'scope':__doc__,'reports':reports,'models':coefs,'fixed_l2':.001,'source_image_crossfit':True,
            'candidate_sha256':file_sha256(a.candidates),'feature_sha256':file_sha256(a.features),'extended_label_sha256':file_sha256(a.extended_labels),
            'selection_on_evaluation_labels':False,'pose_followup_preregistered_arms':['base','combined','extended_base','extended_combined'],
            'limitations':['same-route image crossfit is not new-route generalisation','render-derived new positives mean metric correspondence not exact plane identity','multiple fixed ablations are exploratory; no significance promotion']}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
