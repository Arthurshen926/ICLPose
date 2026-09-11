"""Freeze matched-readout ablations and identical selective refinement support."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.fit_goal_maplet_memory_transfer import design
from feature_extract.tools.vfm.train_goal_maplet_retrieved_association_probe import fit,token_image_weights,auc
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def selection(c,priority,map_mask,maximum_tokens=128):
    selected=np.zeros(len(priority),bool)
    for s in np.unique(c['source_image']):
        rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']&map_mask[c['prototype_rows']])
        tok=c['query_token'][rows];unique,inverse=np.unique(tok,return_inverse=True)
        value=np.full(len(unique),-np.inf);np.maximum.at(value,inverse,priority[rows])
        chosen=unique[np.lexsort((unique,-value))[:maximum_tokens]]
        selected[rows]=np.isin(tok,chosen)
    return selected


def matrices(c,shape,fine,mask):
    base=design(c['association_features'],shape,True)
    coarse=np.c_[base,fine[:,0],fine[:,5]]
    return {'coarse_readout':coarse,
            'intermediate_readout':np.c_[coarse,mask,mask*fine[:,1]],
            'highres_readout':np.c_[coarse,mask,mask*fine[:,2]]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['train_candidates','train_shape','train_fine','labels','test_candidates','test_shape','test_fine','base_models','policy','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--train_roi',type=Path);p.add_argument('--test_roi',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.policy) as z:map_mask=z['budget25'];map_world=z['prototype_world']
    if bool(a.train_roi)!=bool(a.test_roi):raise ValueError('both ROI sides required')
    models=json.loads(a.base_models.read_text());data=[];roi_data=[]
    for prefix in ['train','test']:
        candidate=getattr(a,prefix+'_candidates');digest=file_sha256(candidate)
        with np.load(candidate) as z:c={k:z[k] for k in z.files}
        if not np.array_equal(c['prototype_world'],map_world):raise ValueError('fine policy map differs')
        with np.load(getattr(a,prefix+'_shape')) as z:
            shape=z['features'];names=z['source_names']
            if str(z['candidate_sha256'])!=digest:raise ValueError('shape differs')
        with np.load(getattr(a,prefix+'_fine')) as z:
            f=z['features'];pixels=z['refined_pixels']
            if str(z['candidate_sha256'])!=digest:raise ValueError('fine cache differs')
        priority=design(c['association_features'],shape,True)@np.array(models['models']['combined'])
        mask=selection(c,priority,map_mask)&(f[:,5]>0)
        data.append((c,shape,f,mask,names))
        np.savez_compressed(a.output/(prefix+'_pixels.npz'),refined_pixels=pixels,refinement_mask=mask,candidate_sha256=np.asarray(digest))
        roi_path=getattr(a,prefix+'_roi')
        if roi_path:
            with np.load(roi_path) as z:
                if str(z['candidate_sha256'])!=digest:raise ValueError('ROI candidate differs')
                roi_mask=z['roi_mask'];roi_cos=z['cosine'];roi_pixels=z['refined_pixels']
            roi_data.append((roi_mask,roi_cos))
            np.savez_compressed(a.output/(prefix+'_window_pixels.npz'),refined_pixels=pixels,refinement_mask=roi_mask,candidate_sha256=np.asarray(digest))
            pixels=pixels.copy();pixels[:,1]=roi_pixels
            np.savez_compressed(a.output/(prefix+'_roi_pixels.npz'),refined_pixels=pixels,refinement_mask=roi_mask,candidate_sha256=np.asarray(digest))
    c,shape,f,mask,names=data[0]
    with np.load(a.labels) as z:
        y=z['labels']
        if str(z['frozen_candidate_sha256'])!=file_sha256(a.train_candidates):raise ValueError('labels differ')
    train_names=set(names[np.unique(c['source_image'])]);test_names=set(data[1][4][np.unique(data[1][0]['source_image'])])
    if train_names&test_names:raise ValueError('train/test image overlap')
    known=c['homography_keep']&(y>=0);images=np.unique(c['source_image']);coefs={};reports={}
    train_matrices=matrices(c,shape,f,mask);test_matrices=matrices(*data[1][:4]);oof=[];test_scores=[]
    if roi_data:
        for matrices_out,datum,roi in zip([train_matrices,test_matrices],data,roi_data):
            rm,rc=roi;coarse=matrices_out['coarse_readout']
            matrices_out['full_window_readout']=np.c_[coarse,rm,rm*datum[2][:,2]]
            matrices_out['roi_readout']=np.c_[coarse,rm,rm*rc]
    for arm,x in train_matrices.items():
        beta=fit(x[known],y[known],token_image_weights(c['source_image'][known],c['query_token'][known]));coefs[arm]=beta.tolist()
        values=np.zeros(len(y),np.float32)
        for fold in [0,1]:
            tr=np.isin(c['source_image'],images[fold::2])&known;ev=~np.isin(c['source_image'],images[fold::2])
            b=fit(x[tr],y[tr],token_image_weights(c['source_image'][tr],c['query_token'][tr]));values[ev]=x[ev]@b
        reports[arm]={'known_image_crossfit_auc':auc(y[known],values[known])}
        oof.append(values);test_scores.append(test_matrices[arm]@beta)
    for prefix,values,path in [('train',oof,a.train_candidates),('test',test_scores,a.test_candidates)]:
        np.savez_compressed(a.output/(prefix+'_scores.npz'),logits=np.column_stack(values),arm_names=np.array(list(coefs)),candidate_sha256=np.asarray(file_sha256(path)))
    (a.output/'models.json').write_text(json.dumps({'models':coefs,'reports':reports,'training_images':sorted(train_names),
        'test_image_count':len(test_names),'refinement_budget_tokens_per_image':128,'extra_fine_map_descriptor_bytes':int(map_mask.sum())*64*2,
        'extra_intermediate_map_descriptor_bytes':int(map_mask.sum())*128*2,
        'train_selected_rows':int(data[0][3].sum()),'test_selected_rows':int(data[1][3].sum()),
        'query_GT_used_for_selection':False,'limitation':'mapping-trained map policy and activation scorer see seq9; seq9 readout crossfit is diagnostic only, transfer is disjoint',
        'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['train_candidates','train_shape','train_fine','labels','test_candidates','test_shape','test_fine','base_models','policy']}},indent=2))


if __name__=='__main__':main()
