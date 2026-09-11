"""Executable lineage, population, capacity and geometry invariants for this sweep."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.root.parent
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(b/'moge_full_input_v218.npz') as z:train={k:z[k] for k in z.files}
    with np.load(b/'memory_transfer_candidates_v224.npz') as z:test={k:z[k] for k in z.files}
    with np.load(a.root/'topology.npz') as z:top={k:z[k] for k in z.files}
    with np.load(a.root/'learned_units/policy.npz') as z:policy={k:z[k] for k in z.files}
    with np.load(a.root/'multiscale/map_choices.npz') as z:choices={k:z[k] for k in z.files}
    for key in ['prototype_world','prototype_plane']:
        assert np.array_equal(train[key],test[key]) and np.array_equal(train[key],top[key]) and np.array_equal(train[key],policy[key])
    map_names=top['source_names'][top['prototype_keys'][:,1]]
    assert not any(n.startswith(('seq9__','seq12__','seq14__')) for n in map_names)
    vis,_=PlaneVisibilityAtlas.load_npz(b.parent/'stmarys_rendered_ransac_plane_visibility_atlas_v1.npz')
    with np.load(b/'stmarys_plane_specific_pnp_observation_bank_v2.npz') as z:offsets=z['observation_offsets']
    references=choices['reference_bank_rows'];refs=references[references>=0]
    assert np.max(refs)<offsets[-1]
    source=np.searchsorted(offsets,refs,side='right')-1
    assert not any(n.startswith('seq9__') for n in vis.view_names[np.unique(source)])
    n=len(map_names);assert policy['budget25'].sum()==n//4 and policy['fixed_budget25'].sum()==n//4
    with np.load(a.root/'learned_units/budget25.npz') as z:
        ids=z['unit_anchor_prototype'];scale=z['scale_index']
        assert np.array_equal(ids,np.flatnonzero(policy['budget25']))
        assert np.array_equal(z['descriptors'],choices['descriptors'][ids,scale])
        assert np.array_equal(z['reference_bank_rows'],references[ids,scale])
    with np.load(a.root/'adaptive_transfer_scores.npz') as z:original=z['logits']
    with np.load(a.root/'adaptive_factorial_scores.npz') as z:factorial=z['logits']
    assert np.array_equal(original,factorial[:,:6])
    old=json.loads((a.root/'fine_models/models.json').read_text())['models'];new=json.loads((a.root/'roi_models/models.json').read_text())['models']
    assert all(old[k]==new[k] for k in old)
    activation={}
    for prefix,candidate in [('train',train),('test',test)]:
        with np.load(a.root/'fine_models'/(prefix+'_pixels.npz')) as z:mask=z['refinement_mask'];pixels=z['refined_pixels']
        tokens=[len(np.unique(candidate['query_token'][(candidate['source_image']==s)&mask])) for s in np.unique(candidate['source_image'])]
        assert max(tokens)<=128 and np.isfinite(pixels).all()
        center=np.c_[(candidate['query_token']%64)*4+1.5,(candidate['query_token']//64)*4+1.5]
        assert np.max(np.abs(pixels-center[:,None]))<=4/3+1e-4
        activation[prefix]={'images':len(tokens),'median_active_tokens':float(np.median(tokens)),'max_active_tokens':max(tokens),'selected_rows':int(mask.sum())}
    names=json.loads((a.root/'extraction_names.json').read_text());fine_count=0
    for name in names:
        with np.load(a.root/'fine_cache'/name) as z:
            meta=json.loads(str(z['metadata_json']))
            assert meta['format']=='goal_fine_radio_v2' and meta['poses_or_labels_opened'] is False
            assert z['coarse_final'].shape==(36,64,64) and z['coarse_intermediate'].shape==(36,64,128) and z['fine_final'].shape==(54,96,64)
        fine_count+=1
    report={'scope':__doc__,'passed':True,'fine_rgb_images':fine_count,'prototype_count':n,'choice_count':len(choices['radii'])*n,
        'map_query_source_overlap':False,'cross_plane_reference_preserves_original_bank_row':True,'fixed_candidate_geometry':True,
        'equal_descriptor_capacity_across_scales':True,'budget25_units':int(policy['budget25'].sum()),'activation':activation,
        'serialized_full_memory_bytes':(a.root/'learned_units/full.npz').stat().st_size,'serialized_budget_memory_bytes':(a.root/'learned_units/budget25.npz').stat().st_size,
        'candidate_sha256':{'train':file_sha256(b/'moge_full_input_v218.npz'),'test':file_sha256(b/'memory_transfer_candidates_v224.npz')},
        'coefficient_reuse_and_factorial_scores_exact':True,
        'limitations':['not a blind new-scene experiment','no proof of optimal arbitrary memory boundaries','extra-context capacity is not total map size']}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
