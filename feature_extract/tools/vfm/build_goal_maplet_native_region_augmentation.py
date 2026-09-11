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
    p.add_argument('--region_boundaries', type=Path)
    p.add_argument('--retrieved_regions', type=int, default=4)
    p.add_argument('--appearance_selection', choices=['max','coverage'], default='max')
    args = p.parse_args()
    if args.retrieved_regions < 1:
        raise ValueError('positive retrieval width required')
    b = args.base
    args.output.mkdir(exist_ok=False, parents=True)
    atlas_path = b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz'
    atlas, atlas_meta = _load_atlas(atlas_path)
    head_path = b/'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz'
    head, hm = load_mapping_subtoken_head(head_path)
    head.eval()
    proj_path = b/'stmarys_chart_local_radio_projection_64d_v2.npz'
    with np.load(proj_path) as z:
        weight = z['weight']
    region_path = b/'region_frontend_v246/map.npz'
    with np.load(region_path) as z:
        modes, mode_regions, centers = z['descriptors'], z['mode_regions'], z['centers']
    with np.load(b/'adaptive_memory_v234/topology.npz') as z:
        source_names = z['source_names'].astype(str)
        used_sources = source_names[np.unique(z['prototype_keys'][:,1])]
    radii = readout_radii(args.region_boundaries, centers)
    if args.retrieved_regions > len(np.unique(mode_regions)):
        raise ValueError('retrieval width exceeds region inventory')
    groups = [np.asarray(g, np.int64) for g in cKDTree(atlas['world_points']).query_ball_point(centers, radii)]
    plane_ids = np.repeat(np.arange(len(atlas['plane_texel_offsets'])-1), np.diff(atlas['plane_texel_offsets']))
    descriptors = torch.as_tensor(atlas['radio_features'].astype(np.float32), device=args.device)
    records = _records([Path('output/vfm_tokens/StMarysChurch/full_1024x576')/f'{split}_manifest.json' for split in ['train','test']])
    manifest = dict(query_GT_used=False, radius_m=radii.tolist(), retrieved_regions=args.retrieved_regions, maximum_added_rows=1024,
                    boundary_sha256=file_sha256(args.region_boundaries) if args.region_boundaries else None,
                    boundary_scope='metric local readout envelopes; retrieval context modes and geometry unchanged',
                    appearance_selection=args.appearance_selection,
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
        readout_radii(args.region_boundaries, centers, old['names'].astype(str))
        if set(old['names'].astype(str)) & set(used_sources):
            raise ValueError('query in appearance memory')
        if meta['mapping_subtoken_head_content_sha256'] != hm['content_sha256']:
            raise ValueError('original coordinate head differs')
        plane_cmd = dict(protocol['commands'])['moge']
        plane_dir = Path(plane_cmd[plane_cmd.index('--query_plane_dir')+1])
        additions = {arm: [] for arm in ['appearance','random']}
        audit = []
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
                if arm=='appearance':rank=select_context_modes(patch_scores,mode_regions,args.retrieved_regions,args.appearance_selection)
                chosen=[]
                for j in rank:
                    rid=int(mode_regions[j])
                    if rid not in chosen:chosen.append(rid)
                    if len(chosen)==args.retrieved_regions:break
                triples={}
                if len(valid_tokens):
                    for rid in chosen:
                        g=groups[rid]
                        if not len(g):continue
                        sim=tq@descriptors[g].T
                        val,ind=sim.max(1)
                        mutual=sim.argmax(0)[ind]==torch.arange(len(valid_tokens),device=args.device)
                        ts=valid_tokens[mutual.cpu().numpy()]
                        ps=g[ind[mutual].cpu().numpy()]
                        for t,r,v in zip(ts,ps,val[mutual].cpu().numpy()):
                            if (int(t),int(r)) not in existing:triples[(int(t),int(r))]=float(v)
                ordered=sorted(triples,key=lambda k:(-triples[k],k))[:1024]
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
                row_audit[arm]=dict(regions=chosen,added_rows=len(token))
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
            for i in range(len(old['names'])):
                l,h=map(int,old['correspondence_offsets'][i:i+2]);s=int(checked['correspondence_offsets'][i])
                for k in extras[i]:
                    if not np.array_equal(old[k][l:h],checked[k][s:s+h-l]):raise AssertionError('old row changed')
        (args.output/f'{split}_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
