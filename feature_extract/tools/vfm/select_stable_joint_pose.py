"""Require two verified solver restarts to agree before replacing a fixed endpoint.

The two restarts share observations: agreement is an optimizer stability control,
not independent evidence or a calibrated risk guarantee. Fixed first-seed priority
avoids selecting a winner from evaluation errors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
from feature_extract.tools.vfm.verification_bank_contract import validate_poses
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def stable_update(base, first, second, first_accepted, second_accepted):
    accept = bool(first_accepted and second_accepted and not separated_mode(first, second))
    return (first if accept else base).copy(), accept


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['baseline', 'first', 'second', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    baseline, bm = read(a.baseline); first, fm = read(a.first); second, sm = read(a.second)
    for arrays, metadata in [(first, fm), (second, sm)]:
        if metadata.get('mode') != 'confirmed':
            raise ValueError('both restarts must pass frozen extra-evidence verification')
        if metadata['source_sha256'].get(str(a.baseline)) != file_sha256(a.baseline):
            raise ValueError('restart did not compare against this exact baseline')
        if not np.array_equal(arrays['names'], baseline['names']):
            raise ValueError('query order differs')
        if arrays['accepted'].shape != (len(baseline['names']),) or arrays['pose_w2c'].shape != baseline['pose_w2c'].shape:
            raise ValueError('restart array cardinality differs')
        validate_poses(arrays['pose_w2c'])
    validate_poses(baseline['pose_w2c'])
    result = [stable_update(b, f, s, fa, sa) for b, f, s, fa, sa in zip(
        baseline['pose_w2c'], first['pose_w2c'], second['pose_w2c'], first['accepted'], second['accepted'])]
    arrays = dict(names=baseline['names'], pose_w2c=np.array([x[0] for x in result]), accepted=np.array([x[1] for x in result]))
    metadata = dict(artifact_type='verified_solver_agreement_v411', arrays_sha256=arrays_sha256(arrays),
                    query_pose_or_ground_truth_read=False, query_source_rgb_read=False,
                    rule='both confirmed and within existing 0.5m/3deg mode boundary; fixed first-seed priority',
                    statistically_independent_evidence=False, calibrated_risk_bound=False,
                    source_sha256={str(f):file_sha256(f) for f in [a.baseline,a.first,a.second,Path(__file__),Path(__file__).with_name('select_goal_maplet_separated_modes.py')]})
    metadata['content_sha256'] = canonical_json_sha256(metadata)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output, **arrays, metadata_json=np.array(json.dumps(metadata)))


if __name__ == '__main__':
    main()
