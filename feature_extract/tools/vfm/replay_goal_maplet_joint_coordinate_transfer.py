"""Fixed-initializer coordinate-head transfer; no per-query policy selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=['shard0','shard1','shard2','shard3'],required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    a=p.parse_args();b=a.root;parent=b.parent;s=a.split
    a.output_dir.mkdir(parents=True,exist_ok=False)
    version=183 if s=='shard3' else 182
    initial=b/f'corrected_baseline_v{version}_{s}_primary/moge.npz'
    contributors=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_official_test_clean')
    commands=[]
    def run(module,args,label):
        cmd=[sys.executable,'-m','feature_extract.tools.vfm.'+module]+list(map(str,args))
        commands.append(cmd)
        (a.output_dir/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (a.output_dir/(label+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    for arm,head in [('local','stmarys_mapping_source_modes_v157.npz'),('joint','mapping_joint_coordinate_head_v201.npz')]:
        corr=a.output_dir/(arm+'_corr.npz')
        run('build_goal_maplet_plane_uv_radio_correspondences',[
            '--plane_uv_atlas',b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz',
            '--query_plane_dir',parent/f'query_planes_official_test_seq13_all_v3_{s}',
            '--plane_ranking',parent/f'direct_radio_plane_ranking_official_test_seq13_all_score_only_v3_{s}.json',
            '--radio_manifest','output/vfm_tokens/StMarysChurch/full_1024x576/test_manifest.json',
            '--radio_projection',b/'stmarys_chart_local_radio_projection_64d_v2.npz',
            '--query_camera_inventory',parent/'query_camera_official_test_seq13_all_v3.npz',
            '--homography_threshold_m','0.25','--query_measurement_policy','mapping_pair_subtoken',
            '--mapping_subtoken_head',b/head,'--planar_map',parent/'stmarys_rendered_ransac_fused_planes_v1.npz',
            '--output_correspondences',corr,'--output',a.output_dir/(arm+'_corr.json')],arm+'_build')
        run('refine_goal_maplet_plane_pose_with_uncertainty',[
            '--frozen_pose_inventory',initial,'--frozen_correspondences',corr,
            '--query_contributors',contributors,'--hypothesis_selection_policy','nearest_reprojection',
            '--reprojection_covariance_policy','isotropic','--output_frozen_pose_inventory',a.output_dir/(arm+'_final.npz'),
            '--output',a.output_dir/(arm+'_final.json')],arm+'_refine')
    with np.load(a.output_dir/'local_corr.npz') as x,np.load(a.output_dir/'joint_corr.npz') as y:
        keys=['names','correspondence_offsets','query_tokens','provenance_region_plane_atlas_row','prototype_atlas_row','radio_match_score','camera_matrices','radial_k1']
        equal={k:bool(np.array_equal(x[k],y[k])) for k in keys}
        if not all(equal.values()):raise ValueError('candidate identity/control mismatch')
    (a.output_dir/'control.json').write_text(json.dumps({'identical_arrays':equal,'initial':str(initial),'scope':'fixed_initial_pose_coordinate_transfer_not_full_consensus'},indent=2)+'\n')
    run('compare_goal_maplet_pose_upgrade',[
        '--baseline',a.output_dir/'local_final.npz','--method',a.output_dir/'joint_final.npz',
        '--query_contributors',contributors,'--temporal_block_lengths','5','10','20',
        '--output',a.output_dir/'paired.json'],'paired')


if __name__=='__main__':main()
