"""Mapping-held cross-plane retrieval with runtime MNN/homography primitives.

Supports mapping regions, supervised MoGe subsets, and complete MoGe RADIO
inputs with independently attached supervision. No correct plane/cell prefilter
is used in candidate ranking. This is a controlled mapping-held audit.
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
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support, _records, _radio


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
        if len(rows)>1 and any(np.any(np.linalg.norm(world[rows[i+1:]]-world[rows[i]],axis=1)>threshold) for i in range(len(rows)-1)):
            bad.add(tuple(keys[inverse[rows[0]]].tolist()))
    return bad


def attach_supervision(tokens, mapping_tokens, mapping_rows):
    """Attach fixed-first supervision without removing any visual input."""
    lookup = {}
    for token, row in zip(mapping_tokens, mapping_rows):
        lookup.setdefault(int(token), int(row))
    return np.asarray([lookup.get(int(t), -1) for t in tokens], np.int64)


def context_ring(tokens, radius, height=36, width=64):
    """Observed image context only; never creates geometric measurements."""
    tokens=np.unique(np.asarray(tokens,np.int64))
    if radius < 0 or np.any(tokens < 0) or np.any(tokens >= height*width):
        raise ValueError('invalid context support')
    mask=np.zeros((height,width),bool)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            y=tokens//width+dy;x=tokens%width+dx
            valid=(y>=0)&(y<height)&(x>=0)&(x<width)
            mask[y[valid],x[valid]]=True
    mask.flat[tokens]=False
    return np.flatnonzero(mask)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['observation_bank','visibility_atlas','planar_map','radio_projection','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--held_route',default='seq9')
    p.add_argument('--query_plane_dir',type=Path,help='Actual MoGe regions, restricted to mapping-supervised tokens; not full deployment candidates.')
    p.add_argument('--radio_manifest',type=Path,nargs='+',help='Use complete real region RADIO inputs; requires query_plane_dir.')
    p.add_argument('--context_radius',type=int,default=0,help='Extra anonymous cross-surface RADIO context score; preserves original Top10 and MNN candidates.')
    p.add_argument('--query_routes',nargs='+',help='Optional query route inventory, independent of map exclusions.')
    p.add_argument('--map_excluded_routes',nargs='+',help='Freeze the same source-excluded map for cross-route transfer.')
    a=p.parse_args()
    query_routes=set(a.query_routes or [a.held_route]);map_exclusions=set(a.map_excluded_routes or [a.held_route])
    if a.context_radius<0 or (a.context_radius and not a.radio_manifest):raise ValueError('context requires full RADIO input')
    if a.radio_manifest and not a.query_plane_dir:raise ValueError('full input requires MoGe regions')
    frozen=a.output.with_suffix('.npz')
    if a.output.exists() or frozen.exists() or a.output.with_suffix('.labels.npz').exists():raise FileExistsError(a.output)
    bank,bm=_load_observation_bank(a.observation_bank);vis,vm=PlaneVisibilityAtlas.load_npz(a.visibility_atlas)
    projection,pm=_load_projection(a.radio_projection);planes=GeometryNativePlanarMap.load_npz(a.planar_map)
    if bm['visibility_atlas_content_sha256']!=vm['content_sha256'] or pm['observation_bank_content_sha256']!=bm['content_sha256']:
        raise ValueError('mapping lineage differs')
    if vm['planar_map_file_sha256']!=file_sha256(a.planar_map):raise ValueError('map lineage differs')
    offsets=bank['observation_offsets'];obs=np.repeat(np.arange(len(vis.view_names)),np.diff(offsets))
    op=np.repeat(np.arange(len(planes.plane_ids)),np.diff(vis.plane_offsets));plane=op[obs]
    names=vis.view_names.astype(str);held=np.array([n.split('__')[0] in query_routes for n in names]);map_held=np.array([n.split('__')[0] in map_exclusions for n in names])
    _,source=np.unique(names,return_inverse=True)
    world=bank['world_points'].astype(float);features=np.empty((len(world),64),np.float32)
    for lo in range(0,len(world),8192):features[lo:lo+8192]=_normalise(bank['radio_features'][lo:lo+8192].astype(np.float32)@projection.T)
    uv=np.einsum('ni,nji->nj',world-planes.centers_world[plane],planes.frames_world[plane,:2])
    identities,identity=np.unique(np.c_[plane,np.floor(uv/.5)],axis=0,return_inverse=True)
    fit=np.flatnonzero(~map_held[obs])
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
    valid_obs=np.asarray(valid_obs);desc=np.asarray(desc);fitdesc=desc[~map_held[valid_obs]];fitplanes=op[valid_obs[~map_held[valid_obs]]]
    plane_list=np.array(sorted(pool));regional=[];qrows=[];mrows=[];keep_rows=[];scorer_features=[]
    queries=[];region_lineage=[]
    query_names=sorted(set(names[held]))
    query_tokens=bank['token_ids'].copy();query_sources=source[obs].copy()
    supervision_rows=np.arange(len(world))
    radio_records=_records(a.radio_manifest) if a.radio_manifest else None
    full_features=[];full_tokens=[];full_sources=[];full_supervision=[]
    next_row=0
    fit_context=None
    if a.context_radius:
        fitobs=valid_obs[~map_held[valid_obs]]
        fit_context=np.zeros_like(fitdesc)
        for name in sorted(set(names[fitobs])):
            raw=_radio(name,radio_records)
            if raw.shape!=(2304,1280):raise ValueError('expected runtime 36x64 RADIO grid')
            for i in np.flatnonzero(names[fitobs]==name):
                o=fitobs[i];lo,hi=offsets[o:o+2]
                ring=context_ring(bank['token_ids'][lo:hi],a.context_radius)
                if len(ring):fit_context[i]=_normalise(raw[ring].mean(axis=0))


    if a.query_plane_dir is None:
        for index in np.flatnonzero(held[valid_obs]):
            o=valid_obs[index];lo,hi=offsets[o:o+2]
            queries.append((int(o),np.arange(lo,hi),desc[index],None))
    else:
        query_names=sorted(set(names[held]))
        source_lookup={n:int(source[np.flatnonzero(names==n)[0]]) for n in set(names)}
        if radio_records is not None:
            query_names=sorted(p.name for p in a.query_plane_dir.glob('*.npz') if p.name.split('__')[0] in query_routes)
            if not query_names:raise ValueError('no held-route region inputs')
            for n in query_names:
                if n not in source_lookup:source_lookup[n]=len(source_lookup)
        for name in query_names:
            path=a.query_plane_dir/name;regions,rm=QueryPlaneRegions.load_npz(path)
            if rm.get('uses_pose_or_ground_truth') is not False:raise ValueError('query regions not pose-free')
            image_obs=np.flatnonzero(names==name)
            image_rows=np.concatenate([np.arange(offsets[o],offsets[o+1]) for o in image_obs]) if len(image_obs) else np.empty(0,np.int64)
            # Fixed first row per token, never selected by distance or plane identity.
            _,first=np.unique(bank['token_ids'][image_rows],return_index=True);image_rows=image_rows[first]
            if radio_records is not None:
                raw=_radio(name,radio_records)
                if raw.shape != (36*64,1280):raise ValueError('expected runtime 36x64 RADIO grid')
                image_start=next_row;next_row+=len(raw)
                full_features.append(_normalise(raw@projection.T))
                full_tokens.append(np.arange(len(raw)))
                full_sources.append(np.full(len(raw),source_lookup[name]))
                attached=attach_supervision(np.arange(len(raw)),bank['token_ids'][image_rows],image_rows)
                full_supervision.append(attached)
            for region in range(len(regions.pixel_counts)):
                tokens,_=_region_token_support(regions.labels,region)
                rows=image_rows[np.isin(bank['token_ids'][image_rows],tokens)]
                supervised_count=len(rows)
                if radio_records is not None:
                    rows=image_start+tokens
                    descriptor=_normalise(raw[tokens].mean(axis=0)) if len(tokens) else None
                else:
                    descriptor=_normalise(bank['radio_features'][rows].astype(np.float32).mean(axis=0)) if len(rows) else None
                if len(rows)<4:continue
                context=None
                if a.context_radius:
                    ring=context_ring(tokens,a.context_radius)
                    context=_normalise(raw[ring].mean(axis=0)) if len(ring) else np.zeros(1280,np.float32)
                queries.append((len(region_lineage),rows,descriptor,context))
                region_lineage.append({'name':name,'region':region,'actual_tokens':len(tokens),'supervised_tokens':supervised_count,'file_sha256':file_sha256(path)})
    if radio_records is not None:
        features=np.concatenate(full_features);query_tokens=np.concatenate(full_tokens)
        query_sources=np.concatenate(full_sources);supervision_rows=np.concatenate(full_supervision)
    for o,query_rows,query_descriptor,context in queries:
        similarity=fitdesc@query_descriptor
        scores=np.full(len(planes.plane_ids),-np.inf);np.maximum.at(scores,fitplanes,similarity)
        ranking=plane_list[np.lexsort((plane_list,-scores[plane_list]))[:10]]
        context_scores=None
        if context is not None:
            context_scores=np.full(len(planes.plane_ids),-1.)
            # Read context at the mode selected by LOCAL evidence only.
            # Do not confound context with an extra raw local-cosine feature.
            ring_similarity=fit_context@context
            for physical in ranking:
                modes=np.flatnonzero(fitplanes==physical)
                if len(modes):context_scores[physical]=ring_similarity[modes[np.argmax(similarity[modes])]]
        regional.append((int(o),ranking.tolist()))
        for rank,physical in enumerate(ranking):
            candidates=pool[int(physical)]
            qi,mi,cosine=_mutual_matches(features[query_rows],mf[candidates])
            if len(qi)<4:continue
            retained,projected,valid=_metric_homography_filter_with_projection(query_tokens[query_rows[qi]],muv[candidates[mi]],threshold_m=.25)
            error=np.linalg.norm(projected-muv[candidates[mi]],axis=1)/.25
            error=np.where(valid&np.isfinite(error),np.minimum(error,8),8)
            row_features=np.c_[cosine,np.full(len(qi),rank/9),error,valid.astype(float)]
            if context_scores is not None:row_features=np.c_[row_features,np.full(len(qi),context_scores[physical])]
            scorer_features.extend(row_features.tolist())
            qrows.extend(query_rows[qi].tolist());mrows.extend(candidates[mi].tolist());keep_rows.extend(retained.tolist())
    qrows=np.asarray(qrows,np.int64);mrows=np.asarray(mrows,np.int64);keep=np.asarray(keep_rows,bool)
    arrays={'query_rows':qrows,'prototype_rows':mrows,'homography_keep':keep,
            'association_features':np.asarray(scorer_features,np.float32),
            'source_image':query_sources[qrows],'query_token':query_tokens[qrows],
            'prototype_world':mw,'prototype_plane':mp,'ranked_observations':np.asarray([v[0] for v in regional]),
            'ranked_planes':np.asarray([v[1] for v in regional])}
    np.savez_compressed(frozen,**arrays)
    # Geometry labels are computed only after all ranked candidate rows are frozen.
    target=supervision_rows[qrows];known=target>=0;safe=np.maximum(target,0)
    distance=np.linalg.norm(world[safe]-mw[mrows],axis=1);same=plane[safe]==mp[mrows]
    labels=physical_labels(distance,same);labels[~known]=-2

    conflicting_rows=0
    if a.query_plane_dir is not None:
        held_rows=np.flatnonzero(held[obs])
        conflicts=ambiguous_source_tokens(source[obs[held_rows]],bank['token_ids'][held_rows],world[held_rows])
        mask=np.array([(int(query_sources[q]),int(query_tokens[q])) in conflicts for q in qrows],dtype=bool)
        mask &= known
        conflicting_rows=int(mask.sum());labels[mask]=-1
    np.savez_compressed(a.output.with_suffix('.labels.npz'),labels=labels,supervision_valid=labels>=0,supervision_available=known,
                        frozen_candidate_sha256=np.asarray(file_sha256(frozen)))
    def summary(mask):
        values=labels[mask]
        return {'rows':int(mask.sum()),'positive':int((values==1).sum()),'negative':int((values==0).sum()),
                'ambiguous':int((values==-1).sum()),'unknown':int((values==-2).sum()),'cross_plane_rows':int((~same[mask]&known[mask]).sum()),
                'unique_observation_rows':int(len(np.unique(qrows[mask]))),
                'unique_query_tokens':int(len(np.unique(np.c_[query_sources[qrows[mask]],query_tokens[qrows[mask]]],axis=0)))}
    report={'scope':__doc__,'candidate_file_sha256':file_sha256(frozen),'candidate_arrays_sha256':arrays_sha256(arrays),
            'context_radius':a.context_radius,'context_changes_candidate_pool':False,'query_row_semantics':'full_image_token_inventory' if radio_records is not None else 'mapping_bank_row','query_image_names':query_names,'query_region_count':len(regional),'correct_plane_in_top10':None if a.query_plane_dir else sum(op[o] in ranked for o,ranked in regional),
            'query_region_policy':'moge_regions_mapping_supervised_token_subset' if a.query_plane_dir else 'mapping_observation_regions',
            'moge_region_lineage':region_lineage,
            'mapping_conflicting_target_rows_masked':conflicting_rows,
            'before_homography':summary(np.ones(len(qrows),bool)),'after_homography':summary(keep),
            'query_routes':sorted(query_routes),'map_excluded_routes':sorted(map_exclusions),'mapping_held_route':a.held_route,'query_test_data_used':False,'whole_held_route_excluded_from_map':bool(all(not any(n.split('__')[0]==r and not map_held[i] for i,n in enumerate(names)) for r in query_routes)),
            'candidate_geometry_labels_used_for_ranking':False,'trained_new_model':False,
            'input_sha256':{k:file_sha256(getattr(a,k)) for k in ['observation_bank','visibility_atlas','planar_map','radio_projection']}}
    if a.query_plane_dir:report['scope']='Actual pose-free MoGe regions, restricted to mapping-supervised tokens; not full deployment candidate distribution. First mapping row defines ambiguous token supervision.'
    if radio_records is not None:
        report['query_region_policy']='moge_regions_full_radio_input_independent_supervision'
        report['scope']='Complete real MoGe region inputs; mapping supervision attached after candidate freezing. Unknown=-2, ambiguous=-1.'
        report['radio_manifest_sha256']={str(p):file_sha256(p) for p in a.radio_manifest}
        report['input_population']={'region_token_occurrences':sum(r['actual_tokens'] for r in region_lineage),
            'supervised_region_token_occurrences':sum(r['supervised_tokens'] for r in region_lineage)}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='moge_region_lineage'},indent=2))


if __name__=='__main__':main()
