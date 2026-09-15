"""Evaluate only a complete, sealed 530-query v414 official-test inventory."""
import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.prepare_cambridge_full_inventory import names_only
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/cambridge_full_v417'))
    p.add_argument('--test-file', type=Path, default=Path('/hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_test.txt'))
    a = p.parse_args()
    out = a.root / 'StMarysChurch/full_test_evaluation'
    official = [n.replace('/', '__') + '.npz' for n in names_only(a.test_file)]
    if len(official) != 530 or len(set(official)) != 530:
        raise ValueError('Official StMarys inventory differs')
    by_seed, seals = {}, {}
    for seed in [1, 2]:
        poses = {}
        for split in ['shard0', 'shard1', 'shard2', 'shard3', 'seq3', 'seq5']:
            backend = (a.root / 'original_equivalence' / ('backend_' + split)
                       if split.startswith('shard') else a.root / 'StMarysChurch/full_test' / split / 'backend')
            marker = backend / 'COMPLETE.json'
            contract = json.loads(marker.read_text())
            if contract['query_labels_opened'] or contract['split'] != split:
                raise ValueError('Incomplete or mismatched inference contract')
            path = backend / 'runtime_base/strong_local_precision_v414' / f'query_agreement_seed{seed}' / f'{split}_agreement.npz'
            seals[str(path)] = file_sha256(path)
            seals[str(marker)] = file_sha256(marker)
            with np.load(path) as z:
                for n, pose in zip(z['names'].astype(str), z['pose_w2c']):
                    if n in poses:
                        raise ValueError(f'Duplicate official prediction: {n}')
                    poses[n] = pose
        if set(poses) != set(official):
            raise ValueError('Missing, extra or excluded official queries')
        by_seed[seed] = np.array([poses[n] for n in official])
    # The complete endpoint set is sealed before opening any pose values.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'endpoint_seal.json').write_text(json.dumps(dict(
        sources=seals, query_names=official, seeds=[1, 2],
        complete_method='v414 with all accepted ancestor branches',
        test_pose_values_not_yet_opened=True), indent=2))
    gt = {v.image_id.replace('/', '__') + '.npz': v.pose_w2c for v in parse_cambridge_pose_file(a.test_file)}
    report = {}
    routes = np.array([n.split('__')[0] for n in official])
    for seed, poses in by_seed.items():
        errors = np.array([_pose_error(pose, gt[n]) for n, pose in zip(official, poses)])
        report[str(seed)] = dict(all=metrics(errors),
                                by_route={r: metrics(errors[routes == r]) for r in sorted(set(routes))})
        np.savez_compressed(out / f'errors_seed{seed}.npz', names=np.array(official), errors=errors, pose_w2c=poses)
    for path, digest in seals.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('An endpoint changed during evaluation')
    (out / 'metrics.json').write_text(json.dumps(dict(
        scene='StMarysChurch', official_test_queries=530, reports=report,
        source_test_file_sha256=file_sha256(a.test_file), all_queries_in_denominator=True,
        failure_policy='invalid poses retain infinite error; no silent exclusion',
        scope='Old complete method on official test; historical seq13 was repeatedly inspected, so this is not a pristine blind test',
        five_scene_validation_complete=False), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
