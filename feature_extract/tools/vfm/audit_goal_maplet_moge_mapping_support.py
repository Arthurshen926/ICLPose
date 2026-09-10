"""Audit actual MoGe region support against mapping-only supervised tokens.

This does not build deployment-equivalent candidates or call map geometry truth.
It quantifies which real region tokens existing mapping supervision can cover.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['observation_bank','visibility_atlas','query_plane_dir','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.observation_bank) as z:offsets=z['observation_offsets'];tokens=z['token_ids'];bm=json.loads(str(z['metadata_json'].item()))
    with np.load(a.visibility_atlas) as z:names=z['view_names'].astype(str);vm=json.loads(str(z['metadata_json'].item()))
    if bm['visibility_atlas_content_sha256']!=vm['content_sha256']:raise ValueError('mapping lineage mismatch')
    rows=[];region_rows=[]
    for path in sorted(a.query_plane_dir.glob('*.npz')):
        value,meta=QueryPlaneRegions.load_npz(path)
        if meta.get('uses_pose_or_ground_truth') is not False:raise ValueError('non-query-only regions')
        name=path.name;observations=np.flatnonzero(names==name)
        if not len(observations):raise ValueError('region image has no mapping observations: '+name)
        supervised=set(int(t) for o in observations for t in tokens[offsets[o]:offsets[o+1]])
        actual=set()
        for region in range(len(value.pixel_counts)):
            ts,_=_region_token_support(value.labels,region);support=set(ts.tolist());actual.update(support)
            region_rows.append({'name':name,'region':region,'tokens':len(support),'supervised_tokens':len(support&supervised)})
        rows.append({'name':name,'regions':len(value.pixel_counts),'actual_tokens':len(actual),'mapping_supervised_tokens':len(supervised),
                     'intersection_tokens':len(actual&supervised),'missing_supervision_tokens':len(actual-supervised),
                     'mapping_tokens_outside_moge':len(supervised-actual),'covered_pixel_fraction':float(np.mean(value.labels>=0)),
                     'region_file_sha256':file_sha256(path)})
    sums={k:sum(r[k] for r in rows) for k in ['regions','actual_tokens','mapping_supervised_tokens','intersection_tokens','missing_supervision_tokens','mapping_tokens_outside_moge']}
    report={'scope':__doc__,'query_count':len(rows),'totals':sums,'images':rows,'regions':region_rows,
            'regions_with_less_than_four_supervised_tokens':sum(r['supervised_tokens']<4 for r in region_rows),
            'query_test_labels_used':False,'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['observation_bank','visibility_atlas']},
            'limitations':['mapping supervision coverage is not physical correspondence correctness','overlapping region tokens counted once per image in totals','does not train or evaluate an association model']}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'queries':len(rows),'totals':sums},indent=2))


if __name__=='__main__':main()
