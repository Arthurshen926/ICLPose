"""Apply frozen seq9 models to disjoint routes; no fitting or labels here."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_memory_transfer import design
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','features','training_candidates','models','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    model=json.loads(a.models.read_text())
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.features) as z:
        f=z['features'];names=z['source_names'].astype(str)
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('feature lineage differs')
    with np.load(a.training_candidates) as z:
        if file_sha256(a.training_candidates)!=model['input_sha256']['candidates']:raise ValueError('training lineage differs')
        if not np.array_equal(z['prototype_world'],c['prototype_world']) or not np.array_equal(z['prototype_plane'],c['prototype_plane']):raise ValueError('transfer map differs')
    if set(names[np.unique(c['source_image'])])&set(model['training_images']):raise ValueError('training/evaluation images overlap')
    logits=np.c_[tuple(design(c['association_features'],f,'combined' in name)@np.array(beta) for name,beta in model['models'].items())]
    np.savez_compressed(a.output,logits=logits.astype(np.float32),arm_names=np.array(list(model['models'])),candidate_sha256=np.asarray(file_sha256(a.candidates)))
    a.output.with_suffix('.json').write_text(json.dumps({'model_sha256':file_sha256(a.models),'candidate_sha256':file_sha256(a.candidates),'feature_sha256':file_sha256(a.features),'same_prototype_geometry':True,'training_query_images_disjoint':True,'fitting_performed':False},indent=2))


if __name__=='__main__':main()
