"""Mapping-held cross-plane retrieval with runtime MNN/homography primitives.

Query regions come from mapping plane observations, NOT independent MoGe3
segmentation. No correct query plane/cell prefilter is used in candidate ranking.
This is a controlled distribution audit, not full deployment-equivalent training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.train_goal_maplet_chart_local_radio_projection import _load_observation_bank
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import _load_projection
from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import _normalise, _source_view_mode_pool, _diverse_mode_indices
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _mutual_matches
from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import _metric_homography_filter_with_projection
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support


def physical_labels(distance,same_plane):
    distance=np.asarray(distance,float)
    # Nearby cross-plane support and cell boundaries are ambiguous, not negatives.
    return np.where((distance<=.25)&same_plane,1,np.where(distance>.5,0,-1)).astype(np.int8)


def ambiguous_source_tokens(sources,tokens,world,threshold=.25):
    """Conservatively mask conflicting mapping targets; never select a closer one."""
    keys,inverse=np.unique(np.c_[sources,tokens],axis=0,return_inverse=True)
    order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1,len(order)]
    bad=set()
    for lo,hi in zip(starts[:-1],starts[1:]):
        rows=order[lo:hi]
        if len(rows)>1 and np.max(np.linalg.norm(world[rows]-world[rows[0]],axis=1))>threshold:
            bad.add(tuple(keys[inverse[rows[0]]].tolist()))
    return bad


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['observation_bank','visibility_atlas','planar_map','radio_projection','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--held_route',default='seq9')
    p.add_argument('--query_plane_dir',type=Path,help='Actual MoGe regions, restricted to mapping-supervised tokens; not full deployment candidates.')
    a=p.parse_args()
    frozen=a.output.with_suffix('.npz')
    if a.output.exists() or frozen.exists() or a.output.with_suffix('.labels.npz').exists():raise FileExistsError(a.output)
    bank,bm=_load_observation_bank(a.observation_bank);vis,vm=PlaneVisibilityAtlas.load_npz(a.visibility_atlas)
    projection,pm=_load_projection(a.radio_projection);planes=GeometryNativePlanarMap.load_npz(a.planar_map)
    if bm['visibility_atlas_content_sha256']!=vm['content_sha256'] or pm['observation_bank_content_sha256']!=bm['content_sha256']:
        raise ValueError('mapping lineage differs')
    if vm['planar_map_file_sha256']!=file_sha256(a.planar_map):raise ValueError('map lineage differs')
    offsets=bank['observation_offsets'];obs=np.repeat(np.arange(len(vis.view_names)),np.diff(offsets))
    op=np.repeat(np.arange(len(planes.plane_ids)),np.diff(vis.plane_offsets));plane=op[obs]
    names=vis.view_names.astype(str);held=np.array([n.split('__')[0]==a.held_route for n in names])
    _,source=np.unique(names,return_inverse=True)
    world=bank['world_points'].astype(float);features=np.empty((len(world),64),np.float32)
    for lo in range(0,len(world),8192):features[lo:lo+8192]=_normalise(bank['radio_features'][lo:lo+8192].astype(np.float32)@projection.T)
    uv=np.einsum('ni,nji->nj',world-planes.centers_world[plane],planes.frames_world[plane,:2])
    identities,identity=np.unique(np.c_[plane,np.floor(uv/.5)],axis=0,return_inverse=True)
    fit=np.flatnonzero(~held[obs])
    keys,mf,mw=_source_view_mode_pool(identity[fit],source[obs[fit]],world[fit],features[fit])
    selected=[]
    for ident in np.unique(keys[:,0]):
        rows=np.flatnonzero(keys[:,0]==ident)
        if len(rows)>=2:selected.extend(rows[_diverse_mode_indices(mf[rows],4)])
    selected=np.asarray(selected,np.int64);mf=mf[selected];mw=mw[selected]
    mp=identities[keys[selected,0],0].astype(int)
    muv=np.einsum('ni,nji->nj',mw-planes.centers_world[mp],planes.frames_world[mp,:2])
    pool={int(v):np.flatnonzero(mp==v) for v in np.unique(mp)}
    # Regional RADIO1280 descriptor and max-over-mapping-observation aggregation.
    desc=[];valid_obs=[]
    for o,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        if hi-lo<4:continue
        desc.append(_normalise(bank['radio_features'][lo:hi].astype(np.float32).mean(axis=0)));valid_obs.append(o)
    valid_obs=np.asarray(valid_obs);desc=np.asarray(desc);fitdesc=desc[~held[valid_obs]];fitplanes=op[valid_obs[~held[valid_obs]]]
    plane_list=np.array(sorted(pool));regional=[];qrows=[];mrows=[];keep_rows=[];scorer_features=[]
    queries=[];region_lineage=[]
    if a.query_plane_dir is None:
        for index in np.flatnonzero(held[valid_obs]):
            o=valid_obs[index];lo,hi=offsets[o:o+2]
            queries.append((int(o),np.arange(lo,hi),desc[index]))
    else:
        for name in sorted(set(names[held])):
            path=a.query_plane_dir/name;regions,rm=QueryPlaneRegions.load_npz(path)
            if rm.get('uses_pose_or_ground_truth') is not False:raise ValueError('query regions not pose-free')
            image_obs=np.flatnonzero(names==name)
            image_rows=np.concatenate([np.arange(offsets[o],offsets[o+1]) for o in image_obs])
            # Fixed first row per token, never selected by distance or plane identity.
            _,first=np.unique(bank['token_ids'][image_rows],return_index=True);image_rows=image_rows[first]
            for region in range(len(regions.pixel_counts)):
                tokens,_=_region_token_support(regions.labels,region)
                rows=image_rows[np.isin(bank['token_ids'][image_rows],tokens)]
                if len(rows)<4:continue
                queries.append((len(region_lineage),rows,_normalise(bank['radio_features'][rows].astype(np.float32).mean(axis=0))))
                region_lineage.append({'name':name,'region':region,'actual_tokens':len(tokens),'supervised_tokens':len(rows),'file_sha256':file_sha256(path)})
    for o,query_rows,query_descriptor in queries:
        similarity=fitdesc@query_descriptor
        scores=np.full(len(planes.plane_ids),-np.inf);np.maximum.at(scores,fitplanes,similarity)
        ranking=plane_list[np.lexsort((plane_list,-scores[plane_list]))[:10]]
        regional.append((int(o),ranking.tolist()))
        for rank,physical in enumerate(ranking):
            candidates=pool[int(physical)]
            qi,mi,cosine=_mutual_matches(features[query_rows],mf[candidates])
            if len(qi)<4:continue
            retained,projected,valid=_metric_homography_filter_with_projection(bank['token_ids'][query_rows[qi]],muv[candidates[mi]],threshold_m=.25)
            error=np.linalg.norm(projected-muv[candidates[mi]],axis=1)/.25
            error=np.where(valid&np.isfinite(error),np.minimum(error,8),8)
            scorer_features.extend(np.c_[cosine,np.full(len(qi),rank/9),error,valid.astype(float)].tolist())
            qrows.extend(query_rows[qi].tolist());mrows.extend(candidates[mi].tolist());keep_rows.extend(retained.tolist())
    qrows=np.asarray(qrows,np.int64);mrows=np.asarray(mrows,np.int64);keep=np.asarray(keep_rows,bool)
    arrays={'query_rows':qrows,'prototype_rows':mrows,'homography_keep':keep,
            'association_features':np.asarray(scorer_features,np.float32),
            'source_image':source[obs[qrows]],'query_token':bank['token_ids'][qrows],
            'prototype_world':mw,'prototype_plane':mp,'ranked_observations':np.asarray([v[0] for v in regional]),
            'ranked_planes':np.asarray([v[1] for v in regional])}
    np.savez_compressed(frozen,**arrays)
    # Geometry labels are computed only after all ranked candidate rows are frozen.
    distance=np.linalg.norm(world[qrows]-mw[mrows],axis=1);same=plane[qrows]==mp[mrows]
    labels=physical_labels(distance,same)
    conflicting_rows=0
    if a.query_plane_dir is not None:
        held_rows=np.flatnonzero(held[obs])
        conflicts=ambiguous_source_tokens(source[obs[held_rows]],bank['token_ids'][held_rows],world[held_rows])
        mask=np.array([(int(source[obs[q]]),int(bank['token_ids'][q])) in conflicts for q in qrows])
        conflicting_rows=int(mask.sum());labels[mask]=-1
    np.savez_compressed(a.output.with_suffix('.labels.npz'),labels=labels,
                        frozen_candidate_sha256=np.asarray(file_sha256(frozen)))
    def summary(mask):
        values=labels[mask]
        return {'rows':int(mask.sum()),'positive':int((values==1).sum()),'negative':int((values==0).sum()),
                'ambiguous':int((values==-1).sum()),'cross_plane_rows':int((~same[mask]).sum()),
                'unique_observation_rows':int(len(np.unique(qrows[mask]))),
                'unique_query_tokens':int(len(np.unique(np.c_[source[obs[qrows[mask]]],bank['token_ids'][qrows[mask]]],axis=0)))}
    report={'scope':__doc__,'candidate_file_sha256':file_sha256(frozen),'candidate_arrays_sha256':arrays_sha256(arrays),
            'query_region_count':len(regional),'correct_plane_in_top10':None if a.query_plane_dir else sum(op[o] in ranked for o,ranked in regional),
            'query_region_policy':'moge_regions_mapping_supervised_token_subset' if a.query_plane_dir else 'mapping_observation_regions',
            'moge_region_lineage':region_lineage,
            'mapping_conflicting_target_rows_masked':conflicting_rows,
            'before_homography':summary(np.ones(len(qrows),bool)),'after_homography':summary(keep),
            'mapping_held_route':a.held_route,'query_test_data_used':False,'whole_held_route_excluded_from_map':True,
            'candidate_geometry_labels_used_for_ranking':False,'trained_new_model':False,
            'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['observation_bank','visibility_atlas','planar_map','radio_projection']}}
    if a.query_plane_dir:report['scope']='Actual pose-free MoGe regions, restricted to mapping-supervised tokens; not full deployment candidate distribution. First mapping row defines ambiguous token supervision.'
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='moge_region_lineage'},indent=2))


if __name__=='__main__':main()
