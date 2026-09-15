"""Materialize the original complete v414 protocols for every official test query."""
import argparse,json,os,subprocess,sys
import numpy as np
from pathlib import Path
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256, arrays_sha256, canonical_json_sha256


TEMPLATES=['guarded_lm_v197_shard0_primary','guarded_lm_v200_shard0_alternate',
 'native_global_proposals_v258/shard0_token','native_reliability_mainline_v275/shard0_reliability_rank32',
 'native_reliability_mainline_v275/shard0_reliability_rank32_consensus','native_hybrid_mainline_v290/shard0_risk_stop',
 'native_hybrid_fine_mainline_v302/shard0_reliability_rank32','native_hybrid_fine_mainline_v302/shard0_reliability_rank32_consensus',
 'native_fine_joint_retention_v304/shard0_joint_fine_retained_consensus',
 'diverse_candidate_retention_v307/shard0_diverse_support_consensus','diverse_refined_retention_v309/shard0_branch',
 'diverse_refined_retention_v309/shard0_diverse_refined_consensus']



def write_camera_inventory(inputs, names, output):
 """Expose native intrinsics in the original camera-only consumer schema."""
 from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics, _camera_inventory
 manifest=inputs/'native_camera_manifest.json';payload=json.loads(manifest.read_text());cameras=payload['cameras']
 rows=[cameras[n[:-4].replace('__','/')] for n in names]
 arrays=dict(names=np.asarray(names),camera_model_id=np.asarray([r['model_id'] for r in rows]),
  camera_width=np.asarray([r['width'] for r in rows]),camera_height=np.asarray([r['height'] for r in rows]),
  camera_params=np.asarray([r['params'] for r in rows]),source_contributor_file_sha256=np.full(len(names),'',dtype='U64'))
 meta=dict(artifact_type='goal_maplet_query_camera_only_inventory_v1',pose_or_ground_truth_member_read=False,
  consumer_may_use_before_pose_freeze=True,query_count=len(names),source_archives_are_pose_bearing=False,
  source_contributor_archives_used=False,unused_contributor_hash_fields='empty; calibration comes from native intrinsics manifest',
  native_camera_manifest_sha256=file_sha256(manifest),arrays_sha256=arrays_sha256(arrays))
 meta['content_sha256']=canonical_json_sha256(meta)
 with np.load(inputs/'native_camera_only.npz') as z:
  reference={str(n):(k,r) for n,k,r in zip(z['names'],z['camera_matrices'],z['radial_k1'])}
 for name,row in zip(names,rows):
  K,k=_scaled_intrinsics(row['model_id'],row['params'],row['width'],row['height'])
  if not np.array_equal(K,reference[name][0]) or k!=reference[name][1]:raise ValueError('Camera schema conversion changed intrinsics')
 if output.exists():
  with np.load(output) as z:
   if json.loads(str(z['metadata_json']))!=meta or any(not np.array_equal(z[k],v) for k,v in arrays.items()):raise ValueError('Camera inventory changed')
 else:np.savez_compressed(output,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
 loaded,_=_camera_inventory(output)
 if set(loaded)!=set(names):raise ValueError('Camera inventory coverage differs')
 return output


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--root',type=Path,default=Path('output/cambridge_full_v418'));a=p.parse_args()
 r=a.root/a.scene;b=r/'runtime_base';json.loads((b/'RUNTIME_ASSETS_COMPLETE.json').read_text());old=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');inputs=Path('output/cambridge_full_v417')/a.scene;names=json.loads((inputs/'test_names.json').read_text());assert len(names)==len(set(names));m=r/'full_map';qroot=r/'query_inputs';qroot.mkdir(exist_ok=True)
 if len(names)!=json.loads((inputs/'official_inventory.json').read_text())['test_images']:raise ValueError('Official test coverage differs')
 camera=write_camera_inventory(inputs,names,qroot/'camera_inventory.npz')
 scene_inputs=json.loads((b/'scene_inputs.json').read_text());test_manifest=Path(scene_inputs['radio_manifests'][1]);env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');ledger={};commands={};cp=qroot/'commands.json'
 if cp.exists():commands=json.loads(cp.read_text())
 def run(label,module,args,sentinel):
  cmd=[sys.executable,'-m','feature_extract.tools.vfm.'+module,*map(str,args)]
  if label in commands and commands[label]!=cmd:raise ValueError('Query command changed: '+label)
  commands[label]=cmd;cp.write_text(json.dumps(commands,indent=2));done=qroot/(label+'.complete.json')
  if done.exists():
   if file_sha256(sentinel)!=json.loads(done.read_text())['sha256']:raise ValueError('Query artifact changed')
   return
  with (qroot/(label+'.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  done.write_text(json.dumps({'sha256':file_sha256(sentinel)},indent=2));print(a.scene,label,'complete',flush=True)
 splits=[];parent_manifests={}
 for index,start in enumerate(range(0,len(names),88)):
  split='batch'+str(index);splits.append(split);selected=names[start:start+88];qd=qroot/split;planes=qd/'planes';moge=qd/'moge';planes.mkdir(parents=True,exist_ok=True);moge.mkdir(exist_ok=True)
  plane_rows=[];moge_rows=[];parent_hashes={}
  for n in selected:
   route=n.split('__')[0]
   if a.scene=='StMarysChurch':
    sources=[old.parent/f'query_planes_official_test_{route}_all_v3'/n,old.parent/f'moge3_official_test_{route}_resize1024_v2'/n]
   else:sources=[inputs/'query_planes_official'/route/n,inputs/'moge_official'/route/n]
   for kind,source in enumerate(sources):
    parent=source.parent/'manifest.json'
    if parent not in parent_manifests:
     meta=json.loads(parent.read_text());keyed={str(row[0]) if kind==0 else str(row['image_id']).replace('/','__')+'.npz':row for row in meta['rows']};parent_manifests[parent]=(meta,keyed,file_sha256(parent))
    meta,keyed,digest=parent_manifests[parent];parent_hashes[str(parent)]=digest
    if kind==0:
     if meta.get('uses_pose_or_ground_truth') is not False or meta.get('sparse_occlusion_carrier',False):raise ValueError('Query plane generation differs from original protocol')
     row=keyed[n]
     if len(row) not in [4,5] or (len(row)==5 and row[4]!=0):raise ValueError('Unexpected query plane manifest schema')
     plane_rows.append(row[:4])
    else:
     if meta['resolution_level']!=5 or meta['use_fp16'] is not True or meta['output_height']!=144 or meta['output_width']!=256:raise ValueError('MoGe official configuration differs')
     moge_rows.append(keyed[n])
   for dst,src in zip([planes/n,moge/n],sources):
    if not src.is_file():raise FileNotFoundError(src)
    if dst.is_symlink() or dst.exists():
     if dst.resolve()!=src.resolve():raise ValueError('Query input changed')
    else:dst.symlink_to(src.resolve())
  (planes/'manifest.json').write_text(json.dumps(dict(artifact_type='goal_maplet_query_plane_region_cache_run_v3',uses_pose_or_ground_truth=False,rows=plane_rows,selected_names_in_order=sorted(selected),query_count=len(selected),source_manifests_sha256=parent_hashes,sparse_occlusion_carrier=False),indent=2))
  (moge/'manifest.json').write_text(json.dumps(dict(artifact_type='goal_maplet_moge_query_geometry_v2_manifest',output_height=144,output_width=256,rows=moge_rows,query_count=len(selected),source_manifests_sha256=parent_hashes,resolution_level=5,use_fp16=True,refine_steps=3,uses_pose=False,uses_ground_truth=False,scope='Exact subset of original sealed query geometry caches'),indent=2))
  (qd/'names.json').write_text(json.dumps(selected,indent=2))
  replacements={str(old.parent/'query_planes_official_test_seq13_all_v3_shard0'):str(planes),
   str(old.parent/'query_camera_official_test_seq13_all_v3.npz'):str(camera),
   str(old.parent/'moge3_official_test_seq13_resize1024_v2'):str(moge),
   str(old.parent/'stmarys_rendered_ransac_fused_planes_v1.npz'):str(r/'full_atlas/planes.npz'),
   'output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/physical_map_v4.npz':str(m/'physical_map.npz'),
   'output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_official_test_clean':str(b/'LABELS_NOT_AVAILABLE')}
  def convert(v):
   if isinstance(v,list):return [convert(x) for x in v]
   if isinstance(v,dict):return {k:convert(x) for k,x in v.items() if k not in ['historical_final','historical_sha256']}
   if isinstance(v,str):return replacements.get(v,v.replace(str(old),str(b)).replace('shard0',split))
   return v
  for t in TEMPLATES:
   src=old/t;dst=b/t.replace('shard0',split);dst.mkdir(parents=True,exist_ok=True)
   for f in [src/'protocol.json',*src.glob('*lineage.json')]:ledger[str(f)]=file_sha256(f);(dst/f.name).write_text(json.dumps(convert(json.loads(f.read_text())),indent=2))
  for family in ['structure_rescue_replay_v326','sequential_depth_rescue_v327']:
   src=old/family/'shard0_command.json';dst=b/family/(split+'_command.json');dst.parent.mkdir(exist_ok=True);ledger[str(src)]=file_sha256(src);dst.write_text(json.dumps(convert(json.loads(src.read_text())),indent=2))
  ranking=qd/'plane_ranking.json'
  run(split+'_ranking','build_goal_maplet_direct_radio_plane_ranking',['--plane_field',b/'direct_plane_field.npz','--query_plane_dir',planes,'--radio_manifest',test_manifest,'--output',ranking],ranking)
  for arm,head,directory in [('point','stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz',f'guarded_lm_v197_{split}_primary'),('surface','stmarys_mapping_surface_coordinate_homography_context_head_v34.npz',f'guarded_lm_v200_{split}_alternate')]:
   pp=b/directory/'protocol.json';protocol=json.loads(pp.read_text());dest=Path(protocol['correspondences'])
   args=['--plane_uv_atlas',b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz','--query_plane_dir',planes,'--plane_ranking',ranking,'--radio_manifest',test_manifest,'--radio_projection',b/'stmarys_chart_local_radio_projection_64d_v2.npz','--query_camera_inventory',camera,'--homography_threshold_m','.25','--query_measurement_policy','mapping_pair_subtoken','--mapping_subtoken_head',b/head,'--output_correspondences',dest,'--output',dest.with_suffix('.json')]
   if arm=='surface':args+=['--planar_map',r/'full_atlas/planes.npz']
   run(split+'_'+arm,'build_goal_maplet_plane_uv_radio_correspondences',args,dest);protocol['correspondence_sha256']=file_sha256(dest);pp.write_text(json.dumps(protocol,indent=2))
 common=['--base',b,'--splits',*splits,'--device',a.device]
 run('wide','build_goal_maplet_native_region_augmentation',[*common,'--output',b/'native_scope_v254/wide','--retrieved_regions','8'],b/'native_scope_v254/wide'/(splits[-1]+'_appearance.npz'))
 run('risk','build_goal_maplet_native_region_augmentation',[*common,'--output',b/'native_hybrid_transfer_v289/risk_stop','--context_library',b/'native_context_v278/map.npz','--retrieved_regions','9','--marginal_value_model',b/'native_hybrid_risk_v291/model.json','--marginal_stop'],b/'native_hybrid_transfer_v289/risk_stop'/(splits[-1]+'_appearance.npz'))
 (b/'READY.json').write_text(json.dumps(dict(scene=a.scene,splits=splits,query_names=names,queries=len(names),method_reference='original complete v414 inference',original_protocol_sha256=ledger,query_labels_opened=False,full_inference_complete=False),indent=2))


if __name__=='__main__':main()
