"""Fit fixed model arms on seq9 only; freeze before cross-route pose evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit,token_image_weights
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def design(association,features,combined):
    base=np.c_[np.ones(len(association)),association[:,:4]]
    return np.c_[base,features[:,:2],features[:,3:]] if combined else base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','features','labels','extended_labels','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.features) as z:
        f=z['features'];names=z['source_names'].astype(str)
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('feature lineage differs')
    sources=c['source_image'];tokens=c['query_token'];keep=c['homography_keep'];models={}
    with np.load(a.labels) as z:y=z['labels']
    with np.load(a.extended_labels) as z:ey=z['labels']
    for name in ['base','combined','extended_base','extended_combined']:
        target=ey if name.startswith('extended') else y
        train=keep&(target>=0);x=design(c['association_features'],f,'combined' in name)
        beta=fit(x[train],target[train],token_image_weights(sources[train],tokens[train]))
        models[name]=beta.tolist()
    report={'models':models,'training_images':names[np.unique(sources)].tolist(),'training_route':'seq9',
            'fixed_l2':.001,'query_labels_opened':False,'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['candidates','features','labels','extended_labels']}}
    if not all(n.startswith('seq9__') for n in report['training_images']):raise ValueError('unexpected training route')
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
