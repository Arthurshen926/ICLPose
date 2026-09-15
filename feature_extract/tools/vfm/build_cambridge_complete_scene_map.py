"""Build the full learned surface-map hierarchy from user-supplied Cambridge priors.

This is map preparation, not a localization benchmark. Only official train
poses and train RADIO enter fitting; test inventory supplies exclusion names.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prior-root', type=Path, default=Path('/root/matcha_prior'))
    p.add_argument('--feature-root', type=Path, default=Path('output/cambridge_full_v417'))
    p.add_argument('--output-root', type=Path, default=Path('output/cambridge_full_v418'))
    a = p.parse_args()
    scene = a.scene
    out = a.output_root / scene / 'full_map'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    prior = a.prior_root / (scene.lower() + '.ply')
    data = Path('/hy-tmp/Cambridge_stdloc') / scene
    manifest = (Path('output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json')
                if scene == 'StMarysChurch' else a.feature_root / scene / 'raw_train_manifest.json')
    payload = json.loads(manifest.read_text())
    cameras = json.loads((a.feature_root / scene / 'native_camera_manifest.json').read_text())['cameras']
    missing = [r['image_id'] for r in payload['records'] if r['image_id'] not in cameras]
    if missing and (scene != 'GreatCourt' or missing != ['seq5/frame00297.png']):
        raise ValueError(f'Unexpected missing mapping calibration: {missing}')
    payload['records'] = [r for r in payload['records'] if r['image_id'] in cameras]
    fit_manifest = out / 'mapping_manifest.json'
    manifest_text = json.dumps(payload, indent=2)
    if fit_manifest.exists() and fit_manifest.read_text() != manifest_text:
        raise ValueError('Mapping inventory changed')
    fit_manifest.write_text(manifest_text)
    train_routes = sorted({r['image_id'].split('/')[0] for r in payload['records']})
    test_routes = sorted({n.split('__')[0] for n in json.loads((a.feature_root / scene / 'test_names.json').read_text())})
    if set(train_routes) & set(test_routes):
        raise ValueError('Train and test trajectories overlap')
    contract = dict(scene=scene, prior=str(prior), prior_sha256=file_sha256(prior),
                    mapping_manifest_sha256=file_sha256(fit_manifest),
                    missing_mapping_calibration=missing, mapping_subsampling=False,
                    test_pose_values_read=False, train_routes=train_routes, test_routes=test_routes,
                    prior_policy='user supplied complete MAtCha prior; existing clean_surface_elements entrypoint',
                    mapper_fit='fixed 120 epochs from scratch, train-only; no test selection',
                    localization_complete=False)
    seal = out / 'input_contract.json'
    if seal.exists() and json.loads(seal.read_text()) != contract:
        raise ValueError('Frozen map inputs changed')
    seal.write_text(json.dumps(contract, indent=2))
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    commands_file = out / 'commands.json'
    commands = json.loads(commands_file.read_text()) if commands_file.exists() else {}

    def run(label, module, args, sentinel):
        cmd = [sys.executable, '-m', 'feature_extract.tools.vfm.' + module, *map(str, args)]
        if label in commands and commands[label] != cmd:
            raise ValueError('Changed command: ' + label)
        commands[label] = cmd
        commands_file.write_text(json.dumps(commands, indent=2))
        done = out / (label + '.complete.json')
        if done.exists():
            d = json.loads(done.read_text())
            if not sentinel.exists() or d['sentinel_sha256'] != file_sha256(sentinel):
                raise ValueError('Completed artifact changed: ' + label)
            return
        with (out / (label + '.log')).open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        done.write_text(json.dumps({'sentinel': str(sentinel), 'sentinel_sha256': file_sha256(sentinel)}, indent=2))
        print(scene, label, 'complete', flush=True)

    camera_args = ['--reference_manifest', fit_manifest, '--reference_pose_file', data / 'dataset_train.txt',
                   '--camera_model_dir', data / 'sparse/0', '--require_camera_for_every_view']
    base = out / 'bootstrap_map'
    final = out / 'final_map'
    mapper = out / 'surface_mapper.pt'
    for stage, directory in [('bootstrap_map', base), ('final_map', final)]:
        directory.mkdir(exist_ok=True)
        args = ['--gaussian_ply', prior, *camera_args, '--canonical_vfm_2dgs',
                '--canonical_token_supply', 'broad_grid', '--disable_virtual_surface_cells',
                '--surface_adjacency_element_radius_cap', '.1', '--output_npz', directory / 'region_map.npz',
                '--summary_json', directory / 'region_summary.json',
                '--descriptor_index_npz', directory / 'region_descriptor_index.npz',
                '--observation_bank_npz', directory / 'observation_bank.npz',
                '--contribution_dir', base / 'contributions', '--contribution_device', a.device,
                '--mapper_device', a.device]
        if stage == 'bootstrap_map':
            args += ['--surface_npz', base / 'surface_elements.npz']
        else:
            args += ['--reuse_surface_npz', base / 'surface_elements.npz',
                     '--reuse_contribution_dir', base / 'contributions', '--surface_maplet_mapper_checkpoint', mapper]
        run(stage, 'build_vfm_2dgs_anchor_map', args, directory / 'region_summary.json')
        surface = out / ('bootstrap_surface' if stage == 'bootstrap_map' else 'final_surface')
        surface.mkdir(exist_ok=True)
        args = ['--surface_elements', base / 'surface_elements.npz', '--region_map', directory / 'region_map.npz',
                '--observation_bank', directory / 'observation_bank.npz', '--radio_final_manifest', fit_manifest,
                '--reference_pose_file', data / 'dataset_train.txt', '--camera_model_dir', data / 'sparse/0',
                '--require_camera_for_every_view', '--output_maplets', surface / 'surface_maplets.npz',
                '--output_anchors', surface / 'stable_surface_anchors.npz', '--summary_json', surface / 'summary.json']
        if stage == 'final_map':
            args += ['--surface_maplet_mapper_checkpoint', mapper, '--mapper_device', a.device]
        run(surface.name, 'build_2dgs_surface_map', args, surface / 'summary.json')
        if stage == 'bootstrap_map':
            run('mapper', 'train_surface_maplet_mapper',
                ['--surface_maplets', surface / 'surface_maplets.npz', '--radio_final_manifest', fit_manifest,
                 '--output_checkpoint', mapper, '--summary_json', out / 'mapper_summary.json',
                 '--checkpoint_protocol', 'fixed_epoch_no_selection', '--training_trajectory_ids', *train_routes,
                 '--strict_holdout_trajectory_ids', *test_routes, '--epochs', '120', '--steps_per_epoch', '8',
                 '--batch_maplets', '64', '--seed', '2350', '--device', a.device], out / 'mapper_summary.json')
    run('physical_map', 'build_goal_maplet_physical_map',
        ['--surface_elements', base / 'surface_elements.npz', '--clean_surface_elements',
         '--legacy_maplets', out / 'final_surface/surface_maplets.npz', '--region_map', final / 'region_map.npz',
         '--mapping_pose_file', data / 'dataset_train.txt', '--output_map', out / 'physical_map.npz',
         '--audit_json', out / 'physical_map_audit.json'], out / 'physical_map_audit.json')
    (out / 'MAP_COMPLETE.json').write_text(json.dumps(dict(contract, map_complete=True), indent=2))


if __name__ == '__main__':
    main()
