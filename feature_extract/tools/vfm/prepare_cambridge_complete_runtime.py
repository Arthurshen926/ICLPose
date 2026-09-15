"""Prepare full-resolution reads, fitted reliability and both region memories."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--root',type=Path,default=Path('output/cambridge_full_v418'));a=p.parse_args();r=a.root/a.scene;ad=r/'full_atlas';m=r/'full_map';json.loads((ad/'ATLAS_COMPLETE.json').read_text());contract=json.loads((m/'MAP_COMPLETE.json').read_text());b=r/'runtime_base';b.mkdir(exist_ok=True)
    old=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');features=r/'features';inputs=Path('output/cambridge_full_v417')/a.scene
    def link(dst,src):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.resolve()!=src.resolve():raise ValueError('Changed runtime asset '+str(dst))
        else:dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
    assets={'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz':ad/'atlas.npz',
            'stmarys_chart_local_radio_projection_64d_v2.npz':ad/'projection64.npz',
            'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz':ad/'point_head.npz',
            'stmarys_mapping_surface_coordinate_homography_context_head_v34.npz':ad/'surface_head.npz',
            'native_fine_v264/mapping_lineage.npz':ad/'mapping_lineage.npz'}
    override = r/'validated_runtime_head_override.json'
    if override.exists():
        amendment=json.loads(override.read_text())
        name='stmarys_mapping_surface_coordinate_homography_context_head_v34.npz'
        src=Path(amendment['head_path'])
        if file_sha256(src)!=amendment['head_sha256']:raise ValueError('Amended head changed')
        with np.load(src) as z:head_meta=json.loads(str(z['metadata_json']))
        if head_meta.get('mapping_validation_gate_pass') is not True:raise ValueError('Amended head failed original gate')
        assets[name]=src
    for dst,src in assets.items():link(b/dst,src)
    link(r/'stmarys_rendered_ransac_fused_planes_v1.npz',ad/'planes.npz')
    model_paths=['native_hybrid_risk_v291/model.json','decision_risk_v405_expanded/calibration.json','overlap_lod_v402/model_covariance_fixed',
                 'spatial_verifier_v404/pairwise_verifier.joblib','local_precision_v409/local_selector.joblib',
                 'spatial_verifier_v404/reproducers','heldout_evidence_v405/reproducers','mainline_joint_v411/reproducers']
    for rel in model_paths:link(b/rel,old/rel)
    (b/'frozen_generic_models.json').write_text(json.dumps(dict(source_scene='StMarysChurch',source_base=str(old),policy='Retain original generic inference model weights; no map hashes rewritten',models={str(v):file_sha256(v) for rel in model_paths for v in ([old/rel] if (old/rel).is_file() else sorted((old/rel).glob('*'))) if v.is_file()}),indent=2))
    manifests=([Path('output/vfm_tokens/StMarysChurch/full_1024x576')/(s+'_manifest.json') for s in ['train','test']] if a.scene=='StMarysChurch' else [inputs/('raw_'+s+'_manifest.json') for s in ['train','test']])
    scene_inputs=dict(scene=a.scene,radio_manifests=list(map(str,manifests)),radio_manifest_sha256={str(p):file_sha256(p) for p in manifests},query_pose_values_included=False)
    (b/'scene_inputs.json').write_text(json.dumps(scene_inputs,indent=2))
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');logs=r/'runtime_preparation';logs.mkdir(exist_ok=True);cp=logs/'commands.json';commands=json.loads(cp.read_text()) if cp.exists() else {}
    def run(label,module,args,sentinel):
        cmd=[sys.executable,'-m','feature_extract.tools.vfm.'+module,*map(str,args)]
        if label in commands and commands[label]!=cmd:raise ValueError('Changed command '+label)
        commands[label]=cmd;cp.write_text(json.dumps(commands,indent=2));done=logs/(label+'.complete.json')
        if done.exists():
            if file_sha256(sentinel)!=json.loads(done.read_text())['sha256']:raise ValueError('Completed asset changed '+label)
            return
        with (logs/(label+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        done.write_text(json.dumps({'sha256':file_sha256(sentinel)},indent=2));print(a.scene,label,'complete',flush=True)
    # Original output shapes and PIL interpolation; projection changes require fresh caches.
    run('features','extract_goal_maplet_fine_radio',['--names',inputs/'feature_names.json','--image_root',Path('/hy-tmp/Cambridge_stdloc')/a.scene,'--radio_projection',ad/'projection64.npz','--output',features,'--device',a.device],features/'shard0.json')
    link(b/'adaptive_memory_v234/fine_cache',features);link(b/'native_fine_v264/query_cache',features)
    run('atlas_reconstruction','build_goal_maplet_plane_uv_radio_atlas',
        ['--planar_map',ad/'planes.npz','--visibility_atlas',ad/'visibility.npz','--observation_bank',ad/'observations.npz','--radio_projection',ad/'projection64.npz','--minimum_views','2','--maximum_prototypes_per_texel','4','--output',b/'native_fine_v264/reconstructed_atlas.npz'],b/'native_fine_v264/reconstructed_atlas.npz')
    contributors=ad/'mapping_contributors'
    run('native_fine_map','build_goal_maplet_native_fine_readout',
        ['--base',b,'--output',b/'native_fine_v264/readout','--phase','map','--mapping-contributors',contributors,'--query-names',inputs/'test_names.json'],b/'native_fine_v264/readout/map.npz')
    split=[]
    if a.scene=='ShopFacade':split=['--mapping-image-partition',m/'mapping_image_partition.json']
    elif a.scene!='StMarysChurch':
        routes=contract['train_routes'];i=max(1,len(routes)//2);split=['--training-routes',*routes[:i],'--heldout-routes',*routes[i:]]
    run('fine_calibration','calibrate_goal_maplet_native_fine',['--base',b,'--output',b/'native_fine_calibration_v267','--mapping-contributors',contributors,*split],b/'native_fine_calibration_v267/calibration.json')
    reference=old/'native_token_fine_v273/readout/seq10_fine_calibrated.npz'
    with np.load(reference) as z:policy=json.loads(str(z['metadata_json']))['fine_readout_protocol']
    calibration=b/'native_fine_calibration_v267/calibration.json';cal=json.loads(calibration.read_text())
    policy=dict(policy,map_sha256=file_sha256(b/'native_fine_v264/readout/map.npz'),mapping_calibration=dict(alpha=cal['alpha'],variance_scale=cal['variance_scale'],file_sha256=file_sha256(calibration),heldout_used_to_fit=False,query_ground_truth_read=False))
    reference_config=b/'native_token_fine_v273/readout/reference_protocol.json';reference_config.parent.mkdir(parents=True,exist_ok=True);reference_config.write_text(json.dumps(dict(fine_readout_protocol=policy,source_policy_reference_sha256=file_sha256(reference),query_arrays_read=False),indent=2))
    run('fine_reliability','train_goal_maplet_native_fine_reliability',['--base',b,'--output',b/'native_reliability_v274'],b/'native_reliability_v274/model.json')
    run('memories','build_cambridge_complete_memories',
        ['--base',b,'--atlas-dir',ad,'--radio-manifest',m/'mapping_manifest.json','--contributors',contributors,
         *(['--mapping-image-partition',m/'mapping_image_partition.json'] if a.scene=='ShopFacade' else [])],b/'native_context_v278/map.npz')
    run('plane_field','build_goal_maplet_direct_radio_plane_field',
        ['--visibility_atlas',ad/'visibility.npz','--radio_manifest',m/'mapping_manifest.json','--output',b/'direct_plane_field.npz'],b/'direct_plane_field.npz')
    (b/'RUNTIME_ASSETS_COMPLETE.json').write_text(json.dumps(dict(scene=a.scene,full_inference_complete=False,query_labels_opened=False),indent=2))


if __name__=='__main__':main()
