"""Replay the archived, complete multi-branch frontend on its original inputs.

This is a numerical equivalence gate, not a new-scene benchmark. The original
coarse correspondences, maps, trained heads, RGB features and MoGe caches are
explicit frozen inputs. Every pose operator through v309 runs again, including
the surface alternative, both fine readouts and all geometry consensus passes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def compare(new, old):
    with np.load(new) as x, np.load(old) as y:
        if set(x.files) - {'metadata_json'} != set(y.files) - {'metadata_json'}:
            raise ValueError(f'Archive output schema differs: {new}')
        result = {}
        for k in x.files:
            if k == 'metadata_json':
                continue
            if k not in y.files:
                raise ValueError(f'Unexpected output array: {new}:{k}')
            numeric = np.issubdtype(x[k].dtype, np.number)
            result[k] = bool(np.array_equal(x[k], y[k], equal_nan=True)
                             if numeric else np.array_equal(x[k], y[k]))
        if not all(result.values()):
            raise ValueError(f'Archive equivalence failed: {new}: {result}')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--new-query-inference', action='store_true',
                   help='Run materialized frozen protocols on new queries, which have no archived predictions to compare.')
    a = p.parse_args()
    b, out, split = a.base, a.output, a.split
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    translations = {}
    commands = []
    equivalence = json.loads((out / 'equivalence.json').read_text()) if (out / 'equivalence.json').exists() else {}
    completion_path = out / 'completed_commands.json'
    completed = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    labels = out / 'LABELS_NOT_AVAILABLE'
    if labels.exists():
        raise ValueError('Equivalence replay must not have evaluation labels')

    def remap(value):
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        if isinstance(value, str):
            for old, new in sorted(translations.items(), key=lambda kv: -len(kv[0])):
                if value == old or value.startswith(old + '/'):
                    return new + value[len(old):]
        return value

    def execute(label, cmd):
        cmd = list(cmd)
        cmd[0] = sys.executable
        if '--query_contributors' in cmd:
            cmd[cmd.index('--query_contributors') + 1] = str(labels)
        if '--query_contributors' in cmd or cmd[2] == 'feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus':
            if '--defer-evaluation' not in cmd:
                cmd.append('--defer-evaluation')
        if '--device' in cmd:
            cmd[cmd.index('--device') + 1] = a.device
        if label in completed:
            if completed[label] != cmd:
                raise ValueError(f'Cannot resume a changed command: {label}')
            commands.append(dict(label=label, argv=cmd))
            return
        commands.append(dict(label=label, argv=cmd))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2))
        logfile = out / (label.replace('/', '_') + '.log')
        with logfile.open('w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        completed[label] = cmd
        completion_path.write_text(json.dumps(completed, indent=2))

    def record(new, old):
        if a.new_query_inference:
            return
        equivalence[str(new.relative_to(out))] = compare(new, old)
        (out / 'equivalence.json').write_text(json.dumps(equivalence, indent=2))

    def protocol(relative):
        src, dest = b / relative, out / relative
        translations[str(src)] = str(dest)
        dest.mkdir(parents=True, exist_ok=True)
        original = src / 'protocol.json'
        spec = json.loads(original.read_text())
        for lp in src.glob('*lineage.json'):
            (dest / lp.name).write_text(json.dumps(remap(json.loads(lp.read_text())), indent=2))
        # v307's exported endpoint is produced separately, before its consumers.
        if relative.startswith('diverse_candidate_retention_v307/'):
            cmd = [sys.executable, '-m', 'feature_extract.tools.vfm.export_goal_maplet_pnp_endpoint',
                   '--candidates', remap(str(b / f'native_hybrid_mainline_v290/{split}_risk_stop/pnp.npz')),
                   '--correspondences', str(b / f'native_hybrid_transfer_v289/risk_stop/{split}_appearance.npz'),
                   '--rule', 'diverse_support', '--reference_pose',
                   remap(str(b / f'native_reliability_mainline_v275/{split}_reliability_rank32_consensus/selected.npz')),
                   '--output', str(dest / 'endpoint.npz')]
            execute(relative + '_export', cmd)
            record(dest / 'endpoint.npz', src / 'endpoint.npz')
        new_spec = remap(spec)
        new_spec['replay_source_protocol_sha256'] = file_sha256(original)
        (dest / 'protocol.json').write_text(json.dumps(new_spec, indent=2))
        for stage, cmd in new_spec['commands']:
            execute(relative + '_' + stage, cmd)
        for new in sorted(dest.glob('*.npz')):
            if (src / new.name).exists():
                record(new, src / new.name)
        print(split, relative, 'inference complete' if a.new_query_inference else 'exact array replay', flush=True)

    def readout(family, coarse, initial):
        dest = out / family / 'readout'
        manifest = out / (family + '_manifest.json')
        manifest.write_text(json.dumps({split: dict(correspondences=coarse, initial_pose=initial)}, indent=2))
        execute(family, [sys.executable, '-m', 'feature_extract.tools.vfm.build_goal_maplet_native_reliability',
                        '--base', str(b), '--output', str(dest), '--input_manifest', str(manifest),
                        '--protocol-reference', str(b / 'native_token_fine_v273/readout/reference_protocol.json'
                            if (b / 'native_token_fine_v273/readout/reference_protocol.json').exists() else
                            b / 'native_token_fine_v273/readout/seq10_fine_calibrated.npz'),
                        '--arms', 'reliability_rank32'])
        name = f'{split}_reliability_rank32.npz'
        record(dest / name, b / family / 'readout' / name)
        translations[str(b / family / 'readout' / name)] = str(dest / name)

    protocol(f'guarded_lm_v200_{split}_alternate')
    protocol(f'native_global_proposals_v258/{split}_token')
    readout('native_reliability_v274', str(b / f'native_scope_v254/wide/{split}_appearance.npz'),
            remap(str(b / f'native_global_proposals_v258/{split}_token/moge.npz')))
    protocol(f'native_reliability_mainline_v275/{split}_reliability_rank32')
    protocol(f'native_reliability_mainline_v275/{split}_reliability_rank32_consensus')
    protocol(f'native_hybrid_mainline_v290/{split}_risk_stop')
    readout('native_hybrid_fine_v301', str(b / f'native_hybrid_transfer_v289/risk_stop/{split}_appearance.npz'),
            remap(str(b / f'native_hybrid_mainline_v290/{split}_risk_stop/moge.npz')))
    for relative in [f'native_hybrid_fine_mainline_v302/{split}_reliability_rank32',
                     f'native_hybrid_fine_mainline_v302/{split}_reliability_rank32_consensus',
                     f'native_fine_joint_retention_v304/{split}_joint_fine_retained_consensus',
                     f'diverse_candidate_retention_v307/{split}_diverse_support_consensus',
                     f'diverse_refined_retention_v309/{split}_branch',
                     f'diverse_refined_retention_v309/{split}_diverse_refined_consensus']:
        protocol(relative)
    (out / 'COMPLETE.json').write_text(json.dumps(dict(
        scope=('New-query complete pose frontend through v309' if a.new_query_inference else
               'Original-input numerical replay of full pose frontend through v309'),
        split=split, evaluated_query_labels=False, new_scene_benchmark=False,
        commands=len(commands), compared_artifacts=len(equivalence),
        archive_comparison_performed=not a.new_query_inference,
        frozen_inputs=['maps', 'trained models', 'coarse correspondences', 'query features', 'MoGe caches'],
        subsequent_stages=['v313', 'v319', 'v326', 'v327', 'v344', 'v357', 'v402-v414']), indent=2))


if __name__ == '__main__':
    main()
