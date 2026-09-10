"""Recompute sparse/dense consensus for two corrected frozen coordinate branches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.tools.vfm.replay_goal_maplet_corrected_baseline import resolve
from feature_extract.tools.vfm.aggregate_goal_maplet_pose_failure_stages import _load_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['primary','alternate','historical_audit','planar_map','physical_map','query_plane_dir',
                'query_contributors','query_camera_inventory','moge3_query','output_dir']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--replay_branches',action='store_true',help='Single-command frozen-input backend and consensus replay; requires new branch directories.')
    p.add_argument('--plane_uv_atlas',type=Path)
    a=p.parse_args()
    authority=_load_report(a.historical_audit)
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if a.replay_branches:
        if a.plane_uv_atlas is None:raise ValueError('--plane_uv_atlas required for branch replay')
        if a.primary.exists() or a.alternate.exists():raise FileExistsError('branch output directories must be new')
        for role,dest in [('primary_final',a.primary),('alternate_final',a.alternate)]:
            ref=authority['frozen_input_lineage'][role]
            old=resolve(a.historical_audit.parent,ref['content_sha256'],ref['file_sha256'],'*final.npz')
            subprocess.run([sys.executable,'-m','feature_extract.tools.vfm.replay_goal_maplet_corrected_baseline',
                '--historical_final',str(old),'--plane_uv_atlas',str(a.plane_uv_atlas),
                '--planar_map',str(a.planar_map),'--query_plane_dir',str(a.query_plane_dir),
                '--query_contributors',str(a.query_contributors),'--output_dir',str(dest)],check=True)
    protocols=[json.loads((d/'protocol.json').read_text()) for d in [a.primary,a.alternate]]
    for role,protocol in zip(['primary_final','alternate_final'],protocols):
        if protocol['historical_sha256']!=authority['frozen_input_lineage'][role]['file_sha256']:
            raise ValueError('branch is not the historical baseline '+role)
    a.output_dir.mkdir(parents=True,exist_ok=False);out=a.output_dir
    ppose=str(a.primary/'final.npz');apose=str(a.alternate/'final.npz')
    pc,ac=[v['correspondences'] for v in protocols]
    stages=[('sparse','select_goal_maplet_cross_coordinate_moge3_geometry_pose',[
        '--point_pose',ppose,'--surface_pose',apose,'--point_correspondences',pc,'--surface_correspondences',ac,
        '--planar_map',str(a.planar_map),'--query_plane_dir',str(a.query_plane_dir),
        '--query_contributors',str(a.query_contributors),'--plane_association_policy','many_query_fragments_per_map_plane',
        '--output_frozen_pose_inventory',str(out/'sparse.npz'),'--output',str(out/'sparse.json')]),
        ('plan','build_goal_maplet_sparse_first_render_plan',[
        '--primary_pose',ppose,'--alternate_pose',apose,'--plane_geometry',str(out/'sparse.npz'),
        '--output',str(out/'plan.npz')])]
    for role,pose,corr in [('primary',ppose,pc),('alternate',apose,ac)]:
        stages.append((role+'_render','build_goal_maplet_direct_plane_pnp_render_consistency',[
            '--frozen_pose_inventory',pose,'--frozen_correspondence',corr,
            '--physical_map',str(a.physical_map),'--query_camera_inventory',str(a.query_camera_inventory),
            '--moge3_query',str(a.moge3_query),'--sparse_first_render_plan',str(out/'plan.npz'),
            '--device',a.device,'--resident_batch_size','4','--output',str(out/(role+'_render.json'))]))
    stages.append(('selected','select_goal_maplet_coordinate_pose_geometry_consensus',[
        '--primary_pose',ppose,'--alternate_pose',apose,'--plane_geometry',str(out/'sparse.npz'),
        '--primary_render',str(out/'primary_render.json'),'--alternate_render',str(out/'alternate_render.json'),
        '--sparse_first_render_plan',str(out/'plan.npz'),'--primary_evaluation',str(a.primary/'final.json'),
        '--alternate_evaluation',str(a.alternate/'final.json'),'--dense_normal_domain','all_valid_query',
        '--output_frozen_pose',str(out/'selected.npz'),'--output',str(out/'selected.json')]))
    jobs=[(name,[sys.executable,'-m','feature_extract.tools.vfm.'+module]+args) for name,module,args in stages]
    (out/'protocol.json').write_text(json.dumps({'commands':jobs,'historical_audit_sha256':file_sha256(a.historical_audit),
        'branches':protocols,'scope':'frozen_correspondences_to_single_pose_with_sparse_dense_consensus'},indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name,cmd in jobs:
        start=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        with (out/'timing.jsonl').open('a') as log:log.write(json.dumps({'stage':name,'seconds':time.perf_counter()-start,'returncode':r.returncode})+'\n')
        if r.returncode:raise RuntimeError('failed stage '+name)


if __name__=='__main__':main()
