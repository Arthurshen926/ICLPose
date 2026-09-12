"""Append pose-free region-retrieved native-atlas matches to frozen correspondences.

Appearance and random retrieval share fixed metric regions, matching and the
original frozen subtoken head. Original query rows are preserved exactly.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _records, _radio, _region_token_support
from feature_extract.tools.vfm.refine_goal_maplet_plane_uv_pose_by_view_geometry import _load_atlas
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import load_mapping_subtoken_head
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors, normalise
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def append_query_rows(original, additions):
    """Repack ragged rows while preserving every old per-query prefix."""
    fixed = {'names', 'camera_matrices', 'radial_k1', 'correspondence_offsets'}
    result = {k: v.copy() for k, v in original.items() if k in fixed}
    row_keys = set(original) - fixed
    chunks = {k: [] for k in row_keys}
    offsets = [0]
    if len(additions) != len(original['names']):
        raise ValueError('query count differs')
    for i, extra in enumerate(additions):
        lo, hi = map(int, original['correspondence_offsets'][i:i+2])
        if set(extra) != row_keys or len({len(v) for v in extra.values()}) != 1:
            raise ValueError('incomplete added-row schema')
        for k in row_keys:
            chunks[k].extend([original[k][lo:hi], np.asarray(extra[k], dtype=original[k].dtype)])
        offsets.append(offsets[-1] + hi-lo + len(extra['query_tokens']))
    result.update({k: np.concatenate(v) for k, v in chunks.items()})
    result['correspondence_offsets'] = np.asarray(offsets, dtype=original['correspondence_offsets'].dtype)
    return result




def select_context_modes(patch_scores, mode_regions, number, policy='max'):
    """Select distinct regions, optionally maximizing marginal quadrant coverage.

    The first mode agrees with max retrieval. Later modes improve the sum of
    best similarities over fixed image quadrants; scores are not probabilities.
    """
    scores = np.asarray(patch_scores)
    ids = np.asarray(mode_regions)
    if scores.ndim != 2 or scores.shape[1] != len(ids) or not np.isfinite(scores).all():
        raise ValueError('invalid context score matrix')
    if policy not in ('max', 'coverage') or not 1 <= number <= len(np.unique(ids)):
        raise ValueError('invalid context selection policy or width')
    maximum = scores.max(0)
    rank = np.argsort(-maximum, kind='stable')
    chosen, seen = [], set()
    covered = None
    for _ in range(number):
        if policy == 'coverage' and chosen:
            gain = np.maximum(scores-covered[:, None], 0).sum(0)
            rank = np.lexsort((np.arange(len(ids)), -maximum, -gain))
        j = next(int(j) for j in rank if int(ids[j]) not in seen)
        chosen.append(j)
        seen.add(int(ids[j]))
        covered = scores[:, j].copy() if covered is None else np.maximum(covered, scores[:, j])
    return chosen


def allocate_added_matches(triples, owners, limit=1024, policy='cosine'):
    """Balance origin/quadrant buckets only when truncation is necessary.

    A duplicate pair has one deterministic owner; output retains cosine order
    so uncapped inventories exactly reproduce the existing solver inputs.
    """
    if limit < 1 or policy not in ('cosine', 'balanced'):
        raise ValueError('invalid addition budget')
    ordered = sorted(triples, key=lambda k: (-triples[k], k))
    if policy == 'cosine' or len(ordered) <= limit:
        return ordered[:limit]
    buckets = {}
    for k in ordered:
        t = k[0]
        bucket = (owners[k], int(t // 64 >= 18) * 2 + int(t % 64 >= 32))
        buckets.setdefault(bucket, []).append(k)
    keys = sorted(buckets)
    selected = set()
    depth = 0
    while len(selected) < limit:
        candidates = [buckets[k][depth] for k in keys if depth < len(buckets[k])]
        if not candidates:
            break
        # Partial final round prefers higher cosine, without fixed region bias.
        candidates.sort(key=lambda k: (-triples[k], k))
        selected.update(candidates[:limit-len(selected)])
        depth += 1
    return [k for k in ordered if k in selected]


def quadrant_support(patch_scores, mode_index, token):
    """Bounded rank prior (not a probability); ties remain equivalent."""
    values = np.asarray(patch_scores)[:, mode_index]
    quadrant = int(token // 64 >= 18) * 2 + int(token % 64 >= 32)
    return float((1 + np.sum(values < values[quadrant])) / 4.)


def readout_radii(path, centers, query_names=()):
    """Transfer metric envelopes, never prototype indices, onto a native atlas."""
    if path is None:
        return np.full(len(centers), 6., np.float64)
    with np.load(path, allow_pickle=False) as z:
        if 'member_policy' in z and str(z['member_policy']) != 'sphere':
            raise ValueError('explicit membership cannot be transferred by metric envelopes')
        if not np.array_equal(z['centers'], centers):
            raise ValueError('retrieval center identity differs')
        if 'training_images' in z and set(z['training_images'].astype(str)) & set(query_names):
            raise ValueError('boundary training/query overlap')
        radii = np.asarray(z['radii'] if 'radii' in z else z['radius'], np.float64)
    if radii.ndim == 0:
        radii = np.full(len(centers), float(radii))
    if radii.shape != (len(centers),) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('invalid metric readout radii')
    return radii


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--splits', nargs='+', default=['seq10','shard0','shard1','shard2','shard3'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--addition_budget_policy', choices=['cosine','balanced'], default='cosine')
    p.add_argument('--local_geometry_scope', choices=['region','recalled_planes'], default='region')
    p.add_argument('--plane_ranking_manifest', type=Path)
    p.add_argument('--region_boundaries', type=Path)
    p.add_argument('--context_library', type=Path)
    p.add_argument('--marginal_value_model', type=Path)
    p.add_argument('--selection_pose_manifest', type=Path)
    p.add_argument('--novel_token_control', action='store_true')
    p.add_argument('--marginal_stop', action='store_true')
    p.add_argument('--retrieved_regions', type=int, default=4)
    p.add_argument('--appearance_selection', choices=['max','coverage'], default='max')
    args = p.parse_args()
    value_model = None
    if args.marginal_stop and not args.marginal_value_model:
        raise ValueError('marginal stopping requires a value model')
    if args.novel_token_control and (args.marginal_value_model or args.context_library is None or args.retrieved_regions != 9):
        raise ValueError('novel-token control requires native eight plus one without a learned model')
    if args.marginal_value_model:
        if args.context_library is None or args.retrieved_regions != 9:
            raise ValueError('next-region model requires native context and eight plus one regions')
        value_model = json.loads(args.marginal_value_model.read_text())
        if value_model.get('query_GT_used') is not False or value_model.get('heldout_used_to_fit') is not False:
            raise ValueError('next-region model supervision contract differs')
    pose_paths = json.loads(args.selection_pose_manifest.read_text()) if args.selection_pose_manifest else {}
    if value_model is not None and value_model.get('feature_contract') == 'pose33':
        if not pose_paths:
            raise ValueError('pose33 requires inferred native8 primary-final poses')
    elif pose_paths:
        raise ValueError('selection poses require pose33 model')
    if args.retrieved_regions < 1:
        raise ValueError('positive retrieval width required')
    b = args.base
    ranking_paths = json.loads(args.plane_ranking_manifest.read_text()) if args.plane_ranking_manifest else {}
    if args.local_geometry_scope == "recalled_planes" and not ranking_paths:
        raise ValueError("recalled-plane control requires original ranking manifest")
    args.output.mkdir(exist_ok=False, parents=True)
    atlas_path = b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz'
    atlas, atlas_meta = _load_atlas(atlas_path)
    if value_model is not None and value_model.get('feature_contract') == 'pose33':
        if value_model['feature_provenance']['native_atlas_sha256'] != file_sha256(atlas_path):
            raise ValueError('pose33 atlas binding differs')
    head_path = b/'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz'
    head, hm = load_mapping_subtoken_head(head_path)
    head.eval()
    proj_path = b/'stmarys_chart_local_radio_projection_64d_v2.npz'
    with np.load(proj_path) as z:
        weight = z['weight']
    region_path = args.context_library or b/'region_frontend_v246/map.npz'
    with np.load(region_path) as z:
        modes, mode_regions, centers = z['descriptors'], z['mode_regions'], z['centers']
        if args.context_library:
            library_meta = json.loads(str(z['metadata_json']))
            library_arrays = {k:z[k] for k in z.files if k != 'metadata_json'}
            if (library_meta.get('native_atlas_sha256') != file_sha256(atlas_path)
                    or library_meta.get('excluded_mapping_route') is not None
                    or arrays_sha256(library_arrays) != library_meta.get('arrays_sha256')
                    or canonical_json_sha256({k:v for k,v in library_meta.items() if k != 'content_sha256'}) != library_meta.get('content_sha256')
                    or library_meta.get('query_pose_or_ground_truth_used_for_retrieval') is not False):
                raise ValueError('native full context library contract differs')
    with np.load(b/'adaptive_memory_v234/topology.npz') as z:
        source_names = z['source_names'].astype(str)
        used_sources = source_names[np.unique(z['prototype_keys'][:,1])]
    if args.context_library:
        used_sources = np.asarray(library_meta['offline_mapping_source_names'])
    radii = readout_radii(args.region_boundaries, centers)
    if args.retrieved_regions > len(np.unique(mode_regions)):
        raise ValueError('retrieval width exceeds region inventory')
    groups = [np.asarray(g, np.int64) for g in cKDTree(atlas['world_points']).query_ball_point(centers, radii)]
    plane_ids = np.repeat(np.arange(len(atlas['plane_texel_offsets'])-1), np.diff(atlas['plane_texel_offsets']))
    descriptors = torch.as_tensor(atlas['radio_features'].astype(np.float32), device=args.device)
    records = _records([Path('output/vfm_tokens/StMarysChurch/full_1024x576')/f'{split}_manifest.json' for split in ['train','test']])
    manifest = dict(query_GT_used=False, inferred_query_pose_used=bool(pose_paths), radius_m=radii.tolist(), retrieved_regions=args.retrieved_regions, maximum_added_rows=1024,
                    selection_pose_sha256={s:file_sha256(Path(pose_paths[s])) for s in args.splits} if pose_paths else {},
                    marginal_value_model_sha256=file_sha256(args.marginal_value_model) if args.marginal_value_model else None,
                    novel_token_control=bool(args.novel_token_control),
                    marginal_stop=bool(args.marginal_stop),
                    boundary_sha256=file_sha256(args.region_boundaries) if args.region_boundaries else None,
                    boundary_scope='metric local readout envelopes; retrieval context modes and geometry unchanged',
                    appearance_selection=args.appearance_selection, addition_budget_policy=args.addition_budget_policy,
                    local_geometry_scope=args.local_geometry_scope,
                    plane_rankings_sha256={k:file_sha256(Path(v)) for k,v in ranking_paths.items()},
                    native_region_memberships=sum(map(len,groups)), minimum_visible_fraction=0.75,
                    local_matcher='native atlas RADIO64 mutual nearest neighbors on observed MoGe tokens',
                    retrieval='four quadrant context RADIO256, source-mode means, configured number of distinct physical regions',
                    sampling_control='original per-query group-call caps; adaptive iterations and frontend cost not matched',
                    input_sha256={str(x):file_sha256(x) for x in [atlas_path,head_path,proj_path,region_path]},
                    map_source_names=used_sources.tolist(), splits=args.splits)
    (args.output/'protocol.json').write_text(json.dumps(manifest, indent=2))
    qi = np.arange(2304).reshape(36,64)
    patches = [qi[y:y+18,x:x+32].ravel() for y in [0,18] for x in [0,32]]
    for split in args.splits:
        protocol = json.loads((b/f'guarded_lm_v197_{split}_primary/protocol.json').read_text())
        old_path = Path(protocol['correspondences'])
        old, meta = _load(old_path)
        selection_poses = {}
        if pose_paths:
            from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
            pa, pm = _load_pose_candidate(Path(pose_paths[split]))
            if not np.array_equal(pa['usable'].astype(bool),np.isfinite(pa['pose_w2c']).all(axis=(1,2))):
                raise ValueError('selection pose validity contract differs')
            selection_poses = dict(zip(pa['names'].astype(str),pa['pose_w2c']))
            if len(selection_poses) != len(pa['names']):
                raise ValueError('duplicate selection pose query names')
            if set(selection_poses) != set(old['names'].astype(str)):
                raise ValueError('selection pose query inventory differs')
        rankings = {}
        if args.local_geometry_scope == 'recalled_planes':
            rp = Path(ranking_paths[split])
            rankdata = json.loads(rp.read_text())
            if file_sha256(rp) != meta['plane_ranking_file_sha256'] or rankdata.get('uses_pose_or_ground_truth') is not False or rankdata.get('contains_postlabel_fields') is not False:
                raise ValueError('original pose-free ranking contract differs')
            rankings = {r['image']: sorted({int(v) for g in r['regions'] for v in g['top10']}) for r in rankdata['rows']}
        readout_radii(args.region_boundaries, centers, old['names'].astype(str))
        if set(old['names'].astype(str)) & set(used_sources):
            raise ValueError('query in appearance memory')
        if meta['mapping_subtoken_head_content_sha256'] != hm['content_sha256']:
            raise ValueError('original coordinate head differs')
        plane_cmd = dict(protocol['commands'])['moge']
        plane_dir = Path(plane_cmd[plane_cmd.index('--query_plane_dir')+1])
        additions = {arm: [] for arm in ['appearance','random']}
        audit = []
        trace = {arm: [] for arm in additions}
        for i,name in enumerate(old['names'].astype(str)):
            start = time.perf_counter()
            q = normalise(_radio(name, records) @ weight.T).astype(np.float32)
            context = normalise(sector_descriptors(q.reshape(36,64,64),1).reshape(2304,256))
            patch_scores = normalise(np.asarray([context[g].mean(0) for g in patches])) @ modes.T
            scores = patch_scores.max(0)
            regions, pm = QueryPlaneRegions.load_npz(plane_dir/name)
            if pm.get('uses_pose_or_ground_truth') is not False:
                raise ValueError('query geometry is not pose-free')
            labels = np.full(2304,-1,np.int64)
            visible = np.zeros(2304,np.float32)
            for rid in range(len(regions.pixel_counts)):
                ts,vs = _region_token_support(regions.labels,rid)
                keep=vs>=0.75
                ts,vs=ts[keep],vs[keep]
                if np.any(labels[ts]>=0):
                    raise ValueError('ambiguous token membership')
                labels[ts],visible[ts] = rid,vs
            valid_tokens = np.flatnonzero(labels>=0)
            tq = torch.as_tensor(q[valid_tokens],device=args.device)
            lo,hi = map(int,old['correspondence_offsets'][i:i+2])
            existing = set(zip(old['query_tokens'][lo:hi].tolist(),old['prototype_atlas_row'][lo:hi].tolist()))
            row_audit = dict(name=name)
            for arm in additions:
                rank = np.argsort(-scores,kind='stable') if arm=='appearance' else np.random.default_rng(
                    260924+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)).permutation(len(modes))
                search_width = 16 if (value_model is not None or args.novel_token_control) and arm == 'appearance' else args.retrieved_regions
                if arm=='appearance':rank=select_context_modes(patch_scores,mode_regions,search_width,args.appearance_selection)
                chosen=[]
                for j in rank:
                    rid=int(mode_regions[j])
                    if rid not in chosen:chosen.append(rid)
                    if len(chosen)==search_width:break
                triples={}
                origins={}
                regional_matches={rid:[] for rid in chosen}
                chosen_modes = {int(mode_regions[j]): int(j) for j in rank if int(mode_regions[j]) in chosen}
                # First (highest ranked) appearance mode is the retrieval authority.
                for j in reversed(list(rank)):
                    if int(mode_regions[j]) in chosen: chosen_modes[int(mode_regions[j])] = int(j)
                if len(valid_tokens):
                    for rid in chosen:
                        g=groups[rid]
                        if args.local_geometry_scope == "recalled_planes": g=g[np.isin(plane_ids[g],rankings[name])]
                        if not len(g):continue
                        sim=tq@descriptors[g].T
                        val,ind=sim.max(1)
                        mutual=sim.argmax(0)[ind]==torch.arange(len(valid_tokens),device=args.device)
                        ts=valid_tokens[mutual.cpu().numpy()]
                        ps=g[ind[mutual].cpu().numpy()]
                        for t,r,v in zip(ts,ps,val[mutual].cpu().numpy()):
                            pair=(int(t),int(r))
                            regional_matches[rid].append((int(t),int(r),float(v)))
                            if pair not in existing:
                                triples[pair]=float(v)
                                origins.setdefault(pair,[]).append(rid)
                if (value_model is not None or args.novel_token_control) and arm == 'appearance':
                    from feature_extract.tools.vfm.native_region_value_features import addition_features
                    from feature_extract.tools.vfm.train_goal_maplet_native_region_marginal_value import predict
                    # Training features use each region's top 1024 MNN rows before
                    # deduplication against the original plane frontend.
                    rm={r:sorted(regional_matches[r],key=lambda x:-x[2])[:1024] for r in chosen}
                    base_pairs=[x for r in chosen[:8] for x in rm[r]]
                    bt=np.array([x[0] for x in base_pairs],int);bs=np.array([x[2] for x in base_pairs])
                    bc=patch_scores[:,[chosen_modes[r] for r in chosen[:8]]].T
                    xx=[addition_features(bt,bs,np.array([x[0] for x in rm[r]],int),np.array([x[2] for x in rm[r]]),patch_scores[:,chosen_modes[r]],bc,centers[r],centers[chosen[:8]]) for r in chosen[8:]]
                    if value_model is not None and value_model.get('feature_contract') in ('hybrid19','pose33'):
                        from feature_extract.tools.vfm.native_hybrid_region_value import hybrid_features
                        xx=hybrid_features(old['query_tokens'][lo:hi],old['prototype_atlas_row'][lo:hi],old['radio_match_score'][lo:hi],
                            [np.array([x[0] for x in rm[r]],int) for r in chosen],
                            [np.array([x[1] for x in rm[r]],int) for r in chosen],
                            [np.array([x[2] for x in rm[r]]) for r in chosen],patch_scores[:,[chosen_modes[r] for r in chosen]].T,centers[chosen])
                    if value_model is not None and value_model.get('feature_contract') == 'pose33':
                        from feature_extract.tools.vfm.native_pose_conditioned_region_value import pose_features
                        gx=pose_features(old['query_tokens'][lo:hi],old['prototype_atlas_row'][lo:hi],
                            [np.array([x[0] for x in rm[r]],int) for r in chosen],
                            [np.array([x[1] for x in rm[r]],int) for r in chosen],
                            [np.array([x[2] for x in rm[r]]) for r in chosen],
                            atlas['world_points'],selection_poses[name],old['camera_matrices'][i],float(old['radial_k1'][i]))
                        xx=np.concatenate([xx,gx],axis=1)
                    gain=np.asarray(xx)[:,2] if args.novel_token_control else predict(value_model,np.asarray(xx))
                    chosen=chosen[:8]+([] if args.marginal_stop and np.max(gain)<=0 else [chosen[8+int(np.argmax(gain))]])
                    origins={k:[r for r in rs if r in chosen] for k,rs in origins.items()}
                    origins={k:rs for k,rs in origins.items() if rs};triples={k:v for k,v in triples.items() if k in origins}
                owners={k:v[0] for k,v in origins.items()}
                ordered=allocate_added_matches(triples,owners,1024,args.addition_budget_policy)
                trace[arm].append(dict(name=name, pairs=[list(k) for k in ordered],
                    prior=[max(quadrant_support(patch_scores,chosen_modes[rid],k[0]) for rid in origins[k]) for k in ordered],
                    origins=[origins[k] for k in ordered],selected_modes=chosen_modes,patch_scores=patch_scores[:,[chosen_modes[r] for r in chosen]].tolist()))
                token=np.asarray([k[0] for k in ordered],np.int64)
                proto=np.asarray([k[1] for k in ordered],np.int64)
                extra={k:old[k][:0].copy() for k in old if k not in ['names','camera_matrices','radial_k1','correspondence_offsets']}
                if len(token):
                    with torch.no_grad():
                        mean,var,logit=head(torch.from_numpy(q[token]),torch.from_numpy(atlas['radio_features'][proto].astype(np.float32)),torch.from_numpy(token))
                    offset=mean.numpy().astype(np.float64)
                    if hm.get('coordinate_affine_matrix') is None:offset*=float(hm.get('coordinate_shrinkage',1.))
                    else:offset=offset@np.asarray(hm['coordinate_affine_matrix'])+np.asarray(hm['coordinate_affine_bias_px'])
                    extra.update(world_points=atlas['world_points'][proto],query_tokens=token,
                        provenance_region_plane_atlas_row=np.c_[labels[token],plane_ids[proto],atlas['texel_identity'][proto]],
                        prototype_atlas_row=proto,query_plane_visible_fraction=visible[token],
                        radio_match_score=np.asarray([triples[k] for k in ordered]),
                        query_measurements_xy=np.c_[(token%64)*4+1.5,(token//64)*4+1.5]+np.clip(offset,-2,2),
                        query_measurement_variance_px2=float(hm.get('measurement_variance_scale',1.))*var.numpy().reshape(-1),
                        correspondence_match_probability=torch.sigmoid(logit).numpy().reshape(-1))
                    for k in ['prototype_world_covariance_m2','prototype_plane_pixel_purity','prototype_plane_depth_dispersion_m']:extra[k]=atlas[k][proto]
                additions[arm].append(extra)
                row_audit[arm]=dict(regions=chosen,added_rows=len(token),precap_rows=len(triples),
                    precap_unique_tokens=len({k[0] for k in triples}),postcap_unique_tokens=len(set(token.tolist())),
                    precap_by_origin={str(r):len({k[0] for k in triples if r in origins[k]}) for r in chosen},
                    postcap_by_origin={str(r):len({k[0] for k in ordered if r in origins[k]}) for r in chosen},
                    precap_by_quadrant=[len({k[0] for k in triples if (int(k[0]//64>=18)*2+int(k[0]%64>=32))==q}) for q in range(4)],
                    postcap_by_quadrant=[len({k[0] for k in ordered if (int(k[0]//64>=18)*2+int(k[0]%64>=32))==q}) for q in range(4)])
            row_audit['seconds']=time.perf_counter()-start
            audit.append(row_audit)
            if (i+1)%10==0:print(split,i+1,'/',len(old['names']),flush=True)
        for arm,extras in additions.items():
            arrays=append_query_rows(old,extras)
            outmeta=dict(meta)
            outmeta.pop('content_sha256',None)
            outmeta.update(arrays_sha256=arrays_sha256(arrays),correspondence_count=len(arrays['query_tokens']),
                           augmentation_protocol=manifest,augmentation_arm=arm,original_correspondences_sha256=file_sha256(old_path),
                           original_metadata_scope='original per-query prefixes only; new matches use native regional MNN without homography filtering',
                           maximum_hypotheses_per_token_observed=max((int(np.unique(arrays['query_tokens'][l:h],return_counts=True)[1].max()) if h>l else 0) for l,h in zip(arrays['correspondence_offsets'][:-1],arrays['correspondence_offsets'][1:])))
            outmeta['content_sha256']=canonical_json_sha256(outmeta)
            out=args.output/f'{split}_{arm}.npz'
            np.savez_compressed(out,**arrays,metadata_json=np.asarray(json.dumps(outmeta,sort_keys=True)))
            checked,_=_load(out)
            weights=np.ones(len(arrays['query_tokens']),np.float64)
            for i,tr in enumerate(trace[arm]):
                start=int(arrays['correspondence_offsets'][i])+int(old['correspondence_offsets'][i+1]-old['correspondence_offsets'][i])
                weights[start:start+len(tr['prior'])]=tr['prior']
            prior_arrays=dict(names=arrays['names'],correspondence_offsets=arrays['correspondence_offsets'],sampling_weights=weights)
            prior_meta=dict(artifact_type='goal_maplet_region_sampling_prior_v1',query_pose_or_ground_truth_read=bool(pose_paths),query_ground_truth_used=False,inferred_query_pose_used=bool(pose_paths),correspondence_file_sha256=file_sha256(out),arrays_sha256=arrays_sha256(prior_arrays),semantics='quadrant rank among four for each retrieved mode; max over origins; old rows weight one; uncalibrated sampling prior')
            prior_meta['content_sha256']=canonical_json_sha256(prior_meta)
            np.savez_compressed(out.with_name(out.stem+'_prior.npz'),**prior_arrays,metadata_json=np.asarray(json.dumps(prior_meta,sort_keys=True)))
            out.with_name(out.stem+'_trace.json').write_text(json.dumps(dict(correspondence_file_sha256=file_sha256(out),rows=trace[arm]),indent=2))
            for i in range(len(old['names'])):
                l,h=map(int,old['correspondence_offsets'][i:i+2]);s=int(checked['correspondence_offsets'][i])
                for k in extras[i]:
                    if not np.array_equal(old[k][l:h],checked[k][s:s+h-l]):raise AssertionError('old row changed')
        (args.output/f'{split}_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
