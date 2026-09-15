"""Continue a complete learned physical map through original chart/head fitting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--root',type=Path,default=Path('output/cambridge_full_v418'));p.add_argument('--prepare-contributors-only',action='store_true');a=p.parse_args()
    m=a.root/a.scene/'full_map';contract=json.loads((m/('input_contract.json' if a.prepare_contributors_only else 'MAP_COMPLETE.json')).read_text());out=a.root/a.scene/'full_atlas';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if file_sha256(Path(contract['prior']))!=contract['prior_sha256']:raise ValueError('Prior changed')
    manifest=m/'mapping_manifest.json';data=Path('/hy-tmp/Cambridge_stdloc')/a.scene
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    commands_path=out/'commands.json';commands=json.loads(commands_path.read_text()) if commands_path.exists() else {}
    def run(label,module,args,sentinel):
        cmd=[sys.executable,'-m','feature_extract.tools.vfm.'+module,*map(str,args)]
        if label in commands and commands[label]!=cmd:raise ValueError('Changed command '+label)
        commands[label]=cmd;commands_path.write_text(json.dumps(commands,indent=2));done=out/(label+'.complete.json')
        if done.exists():
            if file_sha256(sentinel)!=json.loads(done.read_text())['sha256']:raise ValueError('Artifact changed '+label)
            return
        with (out/(label+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        done.write_text(json.dumps({'sha256':file_sha256(sentinel)},indent=2));print(a.scene,label,'complete',flush=True)
    contributors=out/'mapping_contributors';planes=out/'planes.npz';visibility=out/'visibility.npz';bank=out/'observations.npz';projection=out/'projection64.npz';obsdir=out/'rendered_planes'
    run('contributors','build_v6_contributor_cache',
        ['--gaussian_ply',contract['prior'],'--clean_surface_elements',m/'bootstrap_map/surface_elements.npz',
         '--mapping_manifest',manifest,'--mapping_pose_file',data/'dataset_train.txt',
         '--mapping_camera_manifest',Path('output/cambridge_full_v417')/a.scene/'native_camera_manifest.json',
         '--output_dir',contributors,'--summary_json',out/'contributors.json','--trajectory_ids',*contract['train_routes'],
         '--views_per_trajectory','10000','--width','256','--height','144','--top_k','4','--device',a.device],out/'contributors.json')
    if a.prepare_contributors_only:return
    run('rendered_planes','build_goal_maplet_rendered_plane_observations',
        ['--physical_map',m/'physical_map.npz','--contributors',contributors,'--output_dir',obsdir,'--workers','4','--routes',*contract['train_routes']],obsdir/'manifest.json')
    run('fused_planes','fuse_goal_maplet_rendered_plane_observations',
        ['--physical_map',m/'physical_map.npz','--observation_dir',obsdir,'--minimum_views','1','--output',planes],planes)
    run('visibility','build_goal_maplet_plane_visibility_atlas',
        ['--observation_dir',obsdir,'--lineage',planes.with_suffix('.lineage.npz'),'--contributors',contributors,'--planar_map',planes,'--output',visibility],visibility)
    run('observations','build_goal_maplet_plane_pnp_observation_bank',
        ['--visibility_atlas',visibility,'--radio_manifest',manifest,'--mapping_contributors',contributors,'--planar_observation_dir',obsdir,'--output',bank],bank)
    if a.scene=='ShopFacade':
        split=['--mapping_image_partition',m/'mapping_image_partition.json']
    else:
        route='seq9' if 'seq9' in contract['train_routes'] else sorted(contract['train_routes'])[-1]
        split=['--validation_route',route]
    common=['--observation_bank',bank,'--visibility_atlas',visibility,'--planar_map',planes,*split]
    # This trainer has no device CLI; constrain the visible GPU in its process.
    old_visible=env.get('CUDA_VISIBLE_DEVICES');env['CUDA_VISIBLE_DEVICES']=a.device.split(':')[-1]
    run('projection','train_goal_maplet_chart_local_radio_projection',[*common,'--output',projection,'--steps','1200','--seed','260903'],projection)
    for label,seed,flags in [('point_head',260906,[]),('surface_head',260918,['--predict_chart_uv','--geometric_context','--homography_context'])]:
        run(label,'train_goal_maplet_mapping_canonical_subtoken_head',
            [*common,'--radio_projection',projection,'--mapping_contributors',contributors,'--output',out/(label+'.npz'),
             '--calibrate_coordinate_shrinkage','--steps','1200','--seed',str(seed),*flags],out/(label+'.npz'))
    if old_visible is None:env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=old_visible
    run('atlas','build_goal_maplet_plane_uv_radio_atlas',
        ['--planar_map',planes,'--visibility_atlas',visibility,'--observation_bank',bank,'--radio_projection',projection,
         '--cell_size_m','.5','--minimum_views','2','--maximum_prototypes_per_texel','4','--output',out/'atlas.npz',
         '--output_mapping_lineage',out/'mapping_lineage.npz'],out/'atlas.npz')
    (out/'ATLAS_COMPLETE.json').write_text(json.dumps(dict(scene=a.scene,query_pose_values_read=False,localization_complete=False,prior_sha256=contract['prior_sha256']),indent=2))


if __name__=='__main__':main()
