"""Joint old/new support control using frozen v410 correspondence inventories.

No new matches are generated here: this isolates joint initialization from the
v410 independent-region solver. Each new region is combined with all old MNN
regions, while unique query tokens remain the unit of PnP support. No query pose
or evaluation ground truth is read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import solve, canonical_hypotheses, score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256, arrays_sha256, canonical_json_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--seed', type=int, default=260901)
    p.add_argument('--old-only', action='store_true', help='equal-hypothesis pooled old inventory control')
    p.add_argument('--arms', nargs='+', choices=['uniform','complement'], default=['uniform','complement'])
    a = p.parse_args(); b = a.base; s = a.split
    a.output.mkdir(parents=True, exist_ok=True)
    arms = a.arms
    if any((a.output / f'{s}_{arm}.npz').exists() for arm in arms):
        raise FileExistsError('frozen output exists')
    mp = b / 'native_fine_v264/readout/map.npz'
    cp = b / 'native_reliability_v274/readout' / f'{s}_reliability_rank32.npz'
    op = b / 'overlap_lod_v402/lod_fixed' / f'{s}_mnn_audit.json'
    with np.load(mp) as z: world = z['world_points']
    with np.load(cp) as z: names = z['names']; Ks = z['camera_matrices']; ks = z['radial_k1']
    old = json.loads(op.read_text())
    for arm in arms:
        ip = op if a.old_only else b / 'neighbor_overlap_v410/lod' / f'{s}_{arm}_audit.json'
        records = json.loads(ip.read_text()); output = []; audits = []
        for i, name in enumerate(names.astype(str)):
            before, new = old[i], records[i]
            assert before['name'] == new['name'] == name
            assert before['sampled_tokens'] == new['sampled_tokens']
            regions = before['regions'] if a.old_only else before['regions'] + new['regions']
            def inventory(rr):
                ids = np.concatenate([r['prototype_rows'] for r in rr]).astype(int)
                tt = np.concatenate([r['selected_tokens'] for r in rr]).astype(int)
                xy = np.concatenate([np.asarray(r['query_pixels']).reshape(-1, 2) for r in rr])
                return ids, tt, xy
            poses = []; stats = []
            for j, region in enumerate(before['regions'] if a.old_only else new['regions']):
                ids, tt, xy = inventory(before['regions'] if a.old_only else before['regions'] + [region]); stat = {}
                pose = solve(world[ids], tt, Ks[i], float(ks[i]), np.arange(len(ids)),
                             pixels=xy, iterations=1250, seed=a.seed + int(region['region']),
                             hypothesis_budget=250, stats=stat, sampling_policy='context_prior',
                             scores=np.ones(len(ids)))
                poses.append(pose); stats.append(stat)
            ids, tt, xy = inventory(regions)
            w, tt, xy = canonical_hypotheses(world[ids], tt, xy)
            groups = [np.flatnonzero(tt == t) for t in np.unique(tt)]
            finite = [pose for pose in poses if pose is not None]
            pose = max(finite, key=lambda v: score_pose(v, w, xy, groups, Ks[i], float(ks[i]))[0]) if finite else np.full((4,4), np.nan)
            output.append(pose)
            audits.append(dict(name=name, regions=regions, sampled_tokens=new['sampled_tokens'],
                               region_ids=before['region_ids'] if a.old_only else new['region_ids'], candidate_poses=[None if v is None else v.tolist() for v in poses],
                               joint_solver_stats=stats, joint_old_new_support=not a.old_only))
        arrays = dict(names=names, pose_w2c=np.array(output), usable=np.isfinite(output).all((1,2)))
        metadata = dict(artifact_type='joint_neighbor_observations_v411', arrays_sha256=arrays_sha256(arrays),
                        query_pose_or_ground_truth_read=False, arm=arm, seed=a.seed,
                        correspondence_inventory='pooled old MNN only' if a.old_only else 'frozen old MNN plus v410 new pairs',
                        max_hypotheses_per_region=250, max_new_regions=4,
                        source_sha256={str(f): file_sha256(f) for f in [mp, cp, op, ip, Path(__file__), Path(__file__).with_name('token_hypothesis_ransac.py')]})
        metadata['content_sha256'] = canonical_json_sha256(metadata)
        np.savez_compressed(a.output / f'{s}_{arm}.npz', **arrays, metadata_json=np.array(json.dumps(metadata)))
        (a.output / f'{s}_{arm}_audit.json').write_text(json.dumps(audits))
        print(s, arm, 'done', flush=True)


if __name__ == '__main__':
    main()
