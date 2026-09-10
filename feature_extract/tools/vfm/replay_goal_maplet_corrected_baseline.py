"""Replay frozen historical coordinate branches with explicit correctness fixes.

This is a branch-backend replay, not RGB-to-pose or final geometry consensus.
Historical artifacts are never overwritten. Timing includes subprocess startup
and evaluation I/O and must not be reported as online localization latency.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def metadata(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data['metadata_json'].item()))


def resolve(directory, content_hash, file_hash, pattern):
    for path in sorted(directory.glob(pattern)):
        try:
            matched = metadata(path).get('content_sha256') == content_hash
        except (ValueError, KeyError):
            continue
        if matched and file_sha256(path) == file_hash:
            return path
    raise ValueError('missing exact historical input: '+str(content_hash))


def commands(corr, atlas, planar_map, planes, contributors, dest, moge_meta, final_meta):
    common=['--frozen_correspondences',str(corr),'--query_contributors',str(contributors)]
    stages=[
        ('pnp','evaluate_goal_maplet_direct_plane_pnp_multihypothesis',common+[
            '--solver_policy','unique_token_lm','--seed_group_support','unique_tokens',
            '--maximum_groups_per_kind','16','--association_policy','all',
            '--output_frozen_candidates',str(dest/'pnp.npz')]),
        ('view','refine_goal_maplet_plane_uv_pose_by_view_geometry',common+[
            '--initial_candidates',str(dest/'pnp.npz'),'--plane_uv_atlas',str(atlas),
            '--solver_policy','unique_token_lm','--output_frozen_pose_inventory',str(dest/'view.npz')]),
        ('moge','refine_goal_maplet_plane_pose_with_moge3',common+[
            '--frozen_pose_inventory',str(dest/'view.npz'),'--planar_map',str(planar_map),
            '--query_plane_dir',str(planes),'--query_support_weighting',moge_meta['query_support_weighting'],
            '--plane_association_policy',moge_meta['plane_association_policy'],
            '--output_frozen_pose_inventory',str(dest/'moge.npz')]),
        ('final','refine_goal_maplet_plane_pose_with_uncertainty',common+[
            '--frozen_pose_inventory',str(dest/'moge.npz'),
            '--hypothesis_selection_policy',final_meta.get('fixed_hypothesis_selection_policy','nearest_reprojection'),
            '--reprojection_covariance_policy','isotropic',
            '--output_frozen_pose_inventory',str(dest/'final.npz')])]
    return [(name,[sys.executable,'-m','feature_extract.tools.vfm.'+module]+args+
             ['--output',str(dest/(name+'.json'))]) for name,module,args in stages]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['historical_final','plane_uv_atlas','planar_map','query_plane_dir','query_contributors','output_dir']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--initial_candidates',type=Path,help='Opt-in frozen initializer experiment; not correctness-only baseline.')
    p.add_argument('--solver_policy',choices=('unique_token_lm','unique_token_guarded_lm'),default='unique_token_lm')
    a=p.parse_args(); final=metadata(a.historical_final); root=a.historical_final.parent
    corr=resolve(root,final['frozen_correspondence_content_sha256'],final['frozen_correspondence_file_sha256'],'*corr.npz')
    old_moge=resolve(root,final['frozen_pose_inventory_content_sha256'],final['frozen_pose_inventory_file_sha256'],'*moge.npz')
    mm=metadata(old_moge)
    old_view=resolve(root,mm['frozen_pose_inventory_content_sha256'],mm['frozen_pose_inventory_file_sha256'],'*view.npz')
    vm=metadata(old_view)
    if file_sha256(a.plane_uv_atlas)!=vm['plane_uv_atlas_file_sha256'] or file_sha256(a.planar_map)!=mm['planar_map_file_sha256']:
        raise ValueError('map differs from historical baseline')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    jobs=commands(corr,a.plane_uv_atlas,a.planar_map,a.query_plane_dir,a.query_contributors,a.output_dir,mm,final)
    for _,cmd in jobs:
        if '--solver_policy' in cmd:cmd[cmd.index('--solver_policy')+1]=a.solver_policy
    if a.initial_candidates is not None:
        initial_meta=metadata(a.initial_candidates)
        if initial_meta.get('frozen_correspondence_file_sha256')!=file_sha256(corr) or initial_meta.get('query_pose_or_ground_truth_opened') is not False:
            raise ValueError('initializer experiment lineage differs')
        jobs=jobs[1:]
        cmd=jobs[0][1];cmd[cmd.index('--initial_candidates')+1]=str(a.initial_candidates)
    protocol={'scope':'frozen_coordinate_branch_backend_not_final_consensus',
              'historical_final':str(a.historical_final),'historical_sha256':file_sha256(a.historical_final),
              'correspondences':str(corr),'correspondence_sha256':file_sha256(corr),
              'commands':jobs,'python':sys.version,'numpy':np.__version__,
              'timing_semantics':'wall_seconds_including_process_startup_and_postlabel_evaluation',
              'correctness_fixes':['unique_token_LM','positive_depth','unique_token_seed_budget','zero_residual_sort'],
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if a.initial_candidates is not None:
        protocol.update(scope='frozen_initializer_experiment_not_correctness_baseline',
                        initial_candidates_sha256=file_sha256(a.initial_candidates))
    if a.solver_policy!='unique_token_lm':protocol.update(scope='guarded_LM_experiment_not_correctness_baseline',solver_policy=a.solver_policy)
    (a.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name,cmd in jobs:
        start=time.perf_counter()
        with (a.output_dir/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        record={'stage':name,'seconds':time.perf_counter()-start,'returncode':result.returncode}
        with (a.output_dir/'timing.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        if result.returncode:raise RuntimeError('stage failed: '+name)
    cmd=[sys.executable,'-m','feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade',
         '--baseline',str(a.historical_final),'--method',str(a.output_dir/'final.npz'),
         '--query_contributors',str(a.query_contributors),'--temporal_block_lengths','5','10','20',
         '--output',str(a.output_dir/'paired.json')]
    with (a.output_dir/'paired.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)


if __name__=='__main__':main()
