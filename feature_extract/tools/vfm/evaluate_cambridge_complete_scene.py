"""Evaluate a sealed complete-method official Cambridge scene, including failures."""
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
    p.add_argument('--root', type=Path, default=Path('output/cambridge_full_v418'))
    p.add_argument('--scene', required=True, choices=['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch'])
    p.add_argument('--dataset-root', type=Path, default=Path('/hy-tmp/Cambridge_stdloc'))
    a = p.parse_args()
    a.test_file = a.dataset_root / a.scene / 'dataset_test.txt'
    scene_root = a.root / a.scene
    ready = json.loads((scene_root / 'runtime_base/READY.json').read_text())
    out = scene_root / 'full_test_evaluation'
    official = [n.replace('/', '__') + '.npz' for n in names_only(a.test_file)]
    expected = dict(GreatCourt=760, KingsCollege=343, OldHospital=182, ShopFacade=103, StMarysChurch=530)[a.scene]
    if len(official) != expected or len(set(official)) != expected or set(ready['query_names']) != set(official):
        raise ValueError('Official scene inventory differs')
    by_seed, seals = {}, {}
    for seed in [1, 2]:
        poses = {}
        for split in ready['splits']:
            backend = scene_root / 'full_test' / split / 'backend'
            marker = backend / 'COMPLETE.json'
            contract = json.loads(marker.read_text())
            if contract['query_labels_opened'] or contract['split'] != split:
                raise ValueError('Incomplete or mismatched inference contract')
            path = backend / 'runtime_base/strong_local_precision_v414' / f'query_agreement_seed{seed}' / f'{split}_agreement.npz'
            seals[str(path)] = file_sha256(path)
            seals[str(marker)] = file_sha256(marker)
            with np.load(path) as z:
                if z['pose_w2c'].shape != (len(z['names']), 4, 4):
                    raise ValueError('Endpoint pose shape differs from name inventory')
                for n, pose in zip(z['names'].astype(str), z['pose_w2c']):
                    if n in poses:
                        raise ValueError(f'Duplicate official prediction: {n}')
                    poses[n] = pose
        if set(poses) != set(official):
            raise ValueError('Missing, extra or excluded official queries')
        by_seed[seed] = np.array([poses[n] for n in official])
    # The complete endpoint set is sealed before opening any pose values.
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError("Completed evaluation must not be silently overwritten")
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
        scene=a.scene, official_test_queries=expected, reports=report,
        source_test_file_sha256=file_sha256(a.test_file), all_queries_in_denominator=True,
        failure_policy='invalid poses retain infinite error; no silent exclusion',
        scope='Complete v414 inference on user-supplied Gaussian prior; scene maps and coordinate heads fitted on official train; generic selectors frozen from StMarysChurch. StMarysChurch historical queries were previously inspected; no pristine blind claim.',
        protocol_amendment=(json.loads((scene_root/'validated_runtime_head_override.json').read_text()) if (scene_root/'validated_runtime_head_override.json').exists() else None),
        prior_contract=json.loads((scene_root / 'full_map/input_contract.json').read_text()),
        camera_lookup_audit=json.loads((a.root / 'camera_lookup_audit.json').read_text()),
        prior_provenance=json.loads((a.root / 'user_prior_provenance.json').read_text()),
        generic_models=json.loads((scene_root / 'runtime_base/frozen_generic_models.json').read_text()),
        five_scene_validation_complete=False), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
