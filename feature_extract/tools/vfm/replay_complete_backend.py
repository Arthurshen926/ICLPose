"""Connect a successful archived frontend replay to the complete v414 endpoint.

Static maps, features and fitted weights are shared read-only. All descendant
pose families are fresh directories; an absent descendant cannot fall back to
an archived prediction. Results are compared to the archive before proceeding.
This remains an original-input equivalence gate, not a cross-scene evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from feature_extract.tools.vfm.replay_complete_frontend import compare
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--frontend', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--new-query-inference', action='store_true')
    a = p.parse_args()
    b, front, out, s = a.base, a.frontend, a.output, a.split
    if not (front / 'COMPLETE.json').is_file():
        raise ValueError('Connected frontend equivalence must finish first')
    out.mkdir(parents=True, exist_ok=True)
    rb = out / 'runtime_base'
    rb.mkdir(exist_ok=True)
    fresh = ['projected_structure_v313', 'depth_visibility_v319',
             'structure_rescue_replay_v326', 'sequential_depth_rescue_v327',
             'direct_feature_consistency_v344', 'regularized_stage_precision_v357',
             'overlap_lod_v402', 'spatial_verifier_v404', 'heldout_evidence_v405',
             'local_precision_v409', 'joint_old_corrected_v411', 'joint_old_corrected_seed2_v411',
             'mainline_joint_v411', 'joint_modes_v412', 'joint_modes_seed2_v412',
             'eligible_verification_v413', 'strong_local_precision_v414']
    for family in fresh:
        if family not in ['projected_structure_v313', 'depth_visibility_v319']:
            (rb / family).mkdir(exist_ok=True)
    for src in b.iterdir():
        dst = rb / src.name
        if dst.exists() or src.name in fresh:
            continue
        replacement = front / src.name
        if replacement.exists():
            # The readout family also owns its immutable trained model.
            if src.is_dir():
                for item in src.iterdir():
                    if item.is_file() and not (replacement / item.name).exists():
                        (replacement / item.name).symlink_to(item.resolve())
            dst.symlink_to(replacement.resolve(), target_is_directory=src.is_dir())
        else:
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    # A new scene has no archived readout directory before the frontend runs.
    # Expose every newly generated frontend family, including v301, even when
    # that family was absent from the immutable input scaffold.
    for src in front.iterdir():
        dst = rb / src.name
        if src.is_dir() and src.name not in fresh and not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
    for rel in ['overlap_lod_v402/model_covariance_fixed',
                'spatial_verifier_v404/pairwise_verifier.joblib',
                'local_precision_v409/local_selector.joblib']:
        if not (rb / rel).exists():
            (rb / rel).symlink_to((b / rel).resolve(), target_is_directory=(b / rel).is_dir())
    gp = 'stmarys_rendered_ransac_fused_planes_v1.npz'
    if not (out / gp).exists():
        (out / gp).symlink_to((b.parent / gp).resolve())
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    commands, results = [], {}
    completion_path = out / 'completed_commands.json'
    completed = json.loads(completion_path.read_text()) if completion_path.exists() else {}

    def run(label, argv):
        commands.append(dict(label=label, argv=argv))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2))
        if label in completed:
            if completed[label] != argv:
                raise ValueError(f'Cannot resume changed command: {label}')
            return
        with (out / (label + '.log')).open('w') as f:
            subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        completed[label] = argv
        completion_path.write_text(json.dumps(completed, indent=2))
        print(s, label, 'complete', flush=True)

    def module(label, name, *args):
        run(label, [sys.executable, '-m', 'feature_extract.tools.vfm.' + name, *map(str, args)])

    def check(rel):
        if a.new_query_inference:
            if not (rb / rel).is_file():
                raise FileNotFoundError(rb / rel)
            return
        results[rel] = compare(rb / rel, b / rel)
        (out / 'equivalence.json').write_text(json.dumps(results, indent=2))

    def archived(label, source, args=(), substitutions=()):
        original = b / source
        text = original.read_text()
        old = "Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1')"
        if old not in text:
            raise ValueError(f'Unknown archived base declaration: {source}')
        text = text.replace(old, f'Path({str(rb)!r})')
        text = text.replace("['seq10','shard0','shard1','shard2','shard3']", repr([s]))
        for before, after in substitutions:
            if before not in text:
                raise ValueError(f'Unknown archived interface: {before}')
            text = text.replace(before, after)
        script = out / (label + '.py')
        script.write_text(text)
        (out / (label + '_source.json')).write_text(json.dumps(dict(
            original=str(original), original_sha256=file_sha256(original),
            adapted_sha256=file_sha256(script), interface_only=True), indent=2))
        run(label, [sys.executable, str(script), *args])

    for family, depth in [('projected_structure_v313', False), ('depth_visibility_v319', True)]:
        module(family, 'build_goal_maplet_projected_context_verification',
               '--base', rb, '--output', rb / family, '--splits', s,
               '--include_centered', '--device', a.device,
               *(['--depth_visibility', 'observed'] if depth else []))
        check(f'{family}/{s}_refined_structure_guarded.npz')
    for family in ['structure_rescue_replay_v326', 'sequential_depth_rescue_v327']:
        cmd = json.loads((b / family / f'{s}_command.json').read_text())
        mapped = []
        for arg in cmd:
            if arg.startswith(str(b) + '/'):
                rel = arg[len(str(b)) + 1:]
                # Verifier lineage keys bind the actual frontend paths, not
                # their runtime-base symlink aliases; preserve that binding.
                arg = str(front / rel) if (front / rel).is_file() else str(rb / rel)
            mapped.append(arg)
        cmd = mapped
        cmd[0] = sys.executable
        run(family, cmd)
        check(f'{family}/{s}.npz')
    module('v344', 'build_goal_maplet_direct_feature_refinement', '--base', rb,
           '--output', rb / 'direct_feature_consistency_v344', '--split', s,
           '--feature_controls', '--only-arm', 'multiscale_robust512')
    check(f'direct_feature_consistency_v344/{s}_multiscale_robust512.npz')
    module('v357', 'build_goal_maplet_direct_feature_refinement', '--base', rb,
           '--output', rb / 'regularized_stage_precision_v357', '--split', s,
           '--stage-followup', '--only-arm', 'stage_plain_reg_all', '--initial',
           rb / f'direct_feature_consistency_v344/{s}_multiscale_robust512.npz')
    check(f'regularized_stage_precision_v357/{s}_stage_plain_reg_all.npz')
    for seed in [1, 2]:
        label = 'lod_fixed' if seed == 1 else 'lod_seed2'
        raw = rb / 'overlap_lod_v402' / label
        final = raw.with_name(label + '_final')
        module(label, 'run_overlap_lod_frontend', '--base', rb, '--model',
               rb / 'overlap_lod_v402/model_covariance_fixed/matcher.pt',
               '--output', raw, '--split', s, '--lod', '--seed', 260900 + seed, '--arms', 'mnn')
        module(label + '_final', 'refine_goal_maplet_configuration_frontend',
               '--base', rb, '--input', raw, '--output', final,
               '--split', s, '--retain-all', '--arms', 'mnn')
        check(f'overlap_lod_v402/{label}_final/{s}_mnn_refined.npz')
    archived('bank405', 'heldout_evidence_v405/reproducers/g25_holdout_bank405.py', ['--split', s])
    archived('pairwise404', 'spatial_verifier_v404/reproducers/g25_apply_pairwise404.py',
             substitutions=[(",('overlap_weighted_v403/lod_final','overlap_weighted_v403/lod','weighted_mnn','soft_seed1'),('overlap_weighted_seed2_v403/lod_final','overlap_weighted_seed2_v403/lod','weighted_mnn','soft_seed2')", '')])
    archived('gate405', 'heldout_evidence_v405/reproducers/g25_apply_holdout405.py',
             substitutions=[(",('soft_seed1','overlap_weighted_v403/lod_final','weighted_mnn'),('soft_seed2','overlap_weighted_seed2_v403/lod_final','weighted_mnn')", '')])
    # v409 requires both seed inventories even when reporting only seed 1.
    for seed in [1, 2]:
        variant = 'control' if seed == 1 else 'control_seed2'
        module('local409_' + str(seed), 'apply_local_precision_selector',
               '--base', rb, '--splits', s, '--variant', variant)
        check(f'local_precision_v409/{variant}/{s}_learned.npz')
        root = 'joint_old_corrected_v411' if seed == 1 else 'joint_old_corrected_seed2_v411'
        raw, final = rb / root / 'lod', rb / root / 'lod_final'
        module('pooled411_' + str(seed), 'solve_joint_neighbor_observations',
               '--base', rb, '--output', raw, '--split', s, '--old-only',
               '--arms', 'uniform', '--seed', 260900 + seed)
        module('pooled411_refine_' + str(seed), 'refine_goal_maplet_configuration_frontend',
               '--base', rb, '--input', raw, '--output', final,
               '--split', s, '--retain-all', '--arms', 'uniform')
        archived('union411_' + str(seed), 'mainline_joint_v411/reproducers/g25_union_baseline411.py',
                 ['--root', root, '--arms', 'uniform', '--output', f'mainline_joint_v411/corrected_seed{seed}',
                  '--baseline-root', f'local_precision_v409/{variant}', '--baseline-arm', 'learned'])
        check(f'mainline_joint_v411/corrected_seed{seed}/uniform_verified/{s}_confirmed.npz')
        root = 'joint_modes_v412' if seed == 1 else 'joint_modes_seed2_v412'
        raw, final = rb / root / 'lod', rb / root / 'lod_final'
        module('modes412_' + str(seed), 'retain_joint_pose_modes', '--base', rb,
               '--output', raw, '--split', s, '--seed', 260900 + seed)
        module('modes412_refine_' + str(seed), 'refine_goal_maplet_configuration_frontend',
               '--base', rb, '--input', raw, '--output', final,
               '--split', s, '--retain-all', '--arms', 'uniform')
        module('verify413_' + str(seed), 'verify_eligible_pose_candidates', '--base', rb,
               '--splits', s, '--root', root, '--output', f'eligible_verification_v413/seed{seed}/dual_support',
               '--arms', 'uniform', '--baseline-root', f'mainline_joint_v411/corrected_seed{seed}/uniform_verified',
               '--baseline-arm', 'confirmed', '--policy', 'dual_support')
        check(f'eligible_verification_v413/seed{seed}/dual_support/uniform_verified/{s}_confirmed.npz')
    for arm in ['stage_plain_reg_all', 'stage_robust_reg_all']:
        module('local414_' + arm, 'build_goal_maplet_direct_feature_refinement', '--base', rb,
               '--output', rb / 'strong_local_precision_v414/query/candidates', '--split', s,
               '--stage-followup', '--only-arm', arm, '--initial',
               rb / f'regularized_stage_precision_v357/{s}_stage_plain_reg_all.npz')
    module('pairs414', 'build_strong_local_pairs', '--base', rb, '--domain', 'query', '--split', s)
    module('agreement414', 'apply_multiscale_local_agreement', '--base', rb, '--domain', 'query', '--splits', s)
    for seed in [1, 2]:
        check(f'strong_local_precision_v414/query_agreement_seed{seed}/{s}_agreement.npz')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(
        scope=('Connected new-query complete v414 pose chain' if a.new_query_inference else
               'Connected original-input complete v414 pose chain'), split=s,
        compared_artifacts=len(results), query_labels_opened=False,
        archive_comparison_performed=not a.new_query_inference,
        new_scene_benchmark=False, frozen_training_reused=True), indent=2))


if __name__ == '__main__':
    main()
