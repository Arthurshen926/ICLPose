"""Materialize the old seq13 complete protocol for official seq3/seq5 queries.

This uses the existing StMarys map and all its fitted models unchanged. Only
query inventories and output paths change. In particular it preserves the
official seq13 MoGe setting (level 5 / fp16 / resized camera canvas), which
differs from the historical seq10 development setting.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--query-cache', type=Path, required=True)
    p.add_argument('--coarse-equivalence', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    proof = json.loads(a.coarse_equivalence.read_text())
    if set(proof) != {'point', 'surface'} or not all(
            row['exact'] or row['float32_roundoff_only'] for v in proof.values() for row in v.values()):
        raise ValueError('Both original coarse branches must pass the numerical gate')
    b, nb = a.base, a.output
    nb.mkdir(parents=True, exist_ok=True)
    templates = ['guarded_lm_v197_shard0_primary', 'guarded_lm_v200_shard0_alternate',
                 'native_global_proposals_v258/shard0_token',
                 'native_reliability_mainline_v275/shard0_reliability_rank32',
                 'native_reliability_mainline_v275/shard0_reliability_rank32_consensus',
                 'native_hybrid_mainline_v290/shard0_risk_stop',
                 'native_hybrid_fine_mainline_v302/shard0_reliability_rank32',
                 'native_hybrid_fine_mainline_v302/shard0_reliability_rank32_consensus',
                 'native_fine_joint_retention_v304/shard0_joint_fine_retained_consensus',
                 'diverse_candidate_retention_v307/shard0_diverse_support_consensus',
                 'diverse_refined_retention_v309/shard0_branch',
                 'diverse_refined_retention_v309/shard0_diverse_refined_consensus']
    reserved = {x.split('/')[0] for x in templates} | {
        'native_scope_v254', 'native_hybrid_transfer_v289', 'native_fine_v264',
        'structure_rescue_replay_v326', 'sequential_depth_rescue_v327'}
    for src in b.iterdir():
        dst = nb / src.name
        if src.name not in reserved and not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    for name in ['native_scope_v254', 'native_hybrid_transfer_v289', 'native_fine_v264',
                 'structure_rescue_replay_v326', 'sequential_depth_rescue_v327']:
        (nb / name).mkdir(exist_ok=True)
    for src in (b / 'native_fine_v264').iterdir():
        dst = nb / 'native_fine_v264' / src.name
        if not dst.exists():
            dst.symlink_to(a.query_cache.resolve() if src.name == 'query_cache' else src.resolve(),
                           target_is_directory=src.is_dir())
    gp = 'stmarys_rendered_ransac_fused_planes_v1.npz'
    if not (nb.parent / gp).exists():
        (nb.parent / gp).symlink_to((b.parent / gp).resolve())
    commands, sources = [], {}
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def execute(label, cmd, expected):
        commands.append(dict(label=label, argv=cmd))
        (nb / 'preparation_commands.json').write_text(json.dumps(commands, indent=2))
        if expected.exists():
            return
        with (nb / (label + '.log')).open('w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        if not expected.exists():
            raise FileNotFoundError(expected)
        print(label, 'complete', flush=True)

    for route in ['seq3', 'seq5']:
        substitutions = {
            str(b.parent / 'query_planes_official_test_seq13_all_v3_shard0'):
                str(b.parent / f'query_planes_official_test_{route}_all_v3'),
            str(b.parent / 'query_camera_official_test_seq13_all_v3.npz'):
                str(b.parent / f'query_camera_official_test_{route}_all_v3.npz'),
            str(b.parent / 'moge3_official_test_seq13_resize1024_v2'):
                str(b.parent / f'moge3_official_test_{route}_resize1024_v2'),
        }

        def convert(value):
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, str):
                if value in substitutions:
                    return substitutions[value]
                return value.replace(str(b), str(nb)).replace('shard0', route)
            return value

        for template in templates:
            src, dst = b / template, nb / template.replace('shard0', route)
            dst.mkdir(parents=True, exist_ok=True)
            for source in [src / 'protocol.json', *src.glob('*lineage.json')]:
                sources[str(source)] = file_sha256(source)
                (dst / source.name).write_text(json.dumps(convert(json.loads(source.read_text())), indent=2))
        for family in ['structure_rescue_replay_v326', 'sequential_depth_rescue_v327']:
            source = b / family / 'shard0_command.json'
            sources[str(source)] = file_sha256(source)
            (nb / family / f'{route}_command.json').write_text(json.dumps(convert(json.loads(source.read_text())), indent=2))
        for arm, head, directory in [
                ('point', 'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz', f'guarded_lm_v197_{route}_primary'),
                ('surface', 'stmarys_mapping_surface_coordinate_homography_context_head_v34.npz', f'guarded_lm_v200_{route}_alternate')]:
            protocol = json.loads((nb / directory / 'protocol.json').read_text())
            dest = Path(protocol['correspondences'])
            cmd = [sys.executable, '-m', 'feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences',
                   '--plane_uv_atlas', str(nb / 'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz'),
                   '--query_plane_dir', str(b.parent / f'query_planes_official_test_{route}_all_v3'),
                   '--plane_ranking', str(b.parent / f'direct_radio_plane_ranking_official_test_{route}_all_score_only_v3.json'),
                   '--radio_manifest', 'output/vfm_tokens/StMarysChurch/full_1024x576/test_manifest.json',
                   '--radio_projection', str(nb / 'stmarys_chart_local_radio_projection_64d_v2.npz'),
                   '--query_camera_inventory', str(b.parent / f'query_camera_official_test_{route}_all_v3.npz'),
                   '--homography_threshold_m', '.25', '--query_measurement_policy', 'mapping_pair_subtoken',
                   '--mapping_subtoken_head', str(nb / head), '--output_correspondences', str(dest),
                   '--output', str(dest.with_suffix('.json'))]
            if arm == 'surface':
                cmd += ['--planar_map', str(b.parent / gp)]
            execute(route + '_' + arm, cmd, dest)
            protocol['correspondence_sha256'] = file_sha256(dest)
            (nb / directory / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    common = [sys.executable, '-m', 'feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation',
              '--base', str(nb), '--splits', 'seq3', 'seq5', '--device', a.device]
    execute('wide_augmentation', common + ['--output', str(nb / 'native_scope_v254/wide'), '--retrieved_regions', '8'],
            nb / 'native_scope_v254/wide/seq5_appearance.npz')
    execute('risk_augmentation', common + ['--output', str(nb / 'native_hybrid_transfer_v289/risk_stop'),
            '--context_library', str(nb / 'native_context_v278/map.npz'), '--retrieved_regions', '9',
            '--marginal_value_model', str(nb / 'native_hybrid_risk_v291/model.json'), '--marginal_stop'],
            nb / 'native_hybrid_transfer_v289/risk_stop/seq5_appearance.npz')
    (nb / 'READY.json').write_text(json.dumps(dict(
        query_routes=['seq3', 'seq5'], queries=180, method_reference='old official seq13 complete v414 chain',
        original_protocol_sha256=sources, query_labels_opened=False,
        map_and_fitted_weights_unchanged=True, new_pose_inference_complete=False), indent=2))


if __name__ == '__main__':
    main()
