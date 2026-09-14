"""Freeze continuous anonymous coarse/fine feature-pose alignment controls."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from feature_extract.tools.vfm.direct_anonymous_feature_pose import refine
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',required=True);p.add_argument('--extended',action='store_true');p.add_argument('--unregularized',action='store_true');p.add_argument('--visibility',choices=['observed','shuffled']);p.add_argument('--constrained',action='store_true');p.add_argument('--box_slsqp',action='store_true');p.add_argument('--feature_controls',action='store_true');p.add_argument('--map',type=Path);p.add_argument('--context-map',type=Path);p.add_argument('--query-cache',type=Path);p.add_argument('--directional',action='store_true');p.add_argument('--contrast',action='store_true');p.add_argument('--initial',type=Path);p.add_argument('--only-arm');p.add_argument('--stage-followup',action='store_true');p.add_argument('--shuffle-query-spatial',action='store_true');a=p.parse_args();b=a.base;s=a.split;a.output.mkdir(parents=True,exist_ok=True)
 arms={'coarse_joint128':(0,.02,128),'fine_joint128':(1,.02,128),'fine_direct128':(1,0.,128),'fine_joint512':(1,.02,512)}
 if a.extended:arms={'fine_joint_all':(1,.02,2304),'fine_robust512':(1,.02,512),'fine_robust_all':(1,.02,2304),'fine_balanced512':(1,.02,512),'fine_direct512':(1,0.,512)}
 if a.unregularized:
  if a.extended:raise ValueError('choose one experiment family')
  arms={'fine_robust_direct512':(1,0.,512),'fine_robust_direct_all':(1,0.,2304),'fine_direct_all':(1,0.,2304)}
 if a.visibility:arms={'visibility_joint512':(1,.02,512),'visibility_direct512':(1,0.,512),'visibility_robust_direct512':(1,0.,512)}
 if a.constrained:arms={'constrained_joint512':(1,.02,512),'constrained_direct512':(1,0.,512),'constrained_robust_direct512':(1,0.,512)}
 if a.box_slsqp:
  if a.constrained:raise ValueError('choose constrained or box-only solver')
  arms={'box_joint512':(1,.02,512),'box_direct512':(1,0.,512),'box_robust_direct512':(1,0.,512)}
 if a.feature_controls:arms={'multiscale_robust512':(1,0.,512),'multiscale_plain512':(1,0.,512),'coarse_robust512':(0,0.,512),'frozen_bias_robust512':(1,0.,512),'profiled_bias_robust512':(1,0.,512)}
 if a.directional:arms={'direction_nearest_robust512':(1,0.,512),'direction_kernel_robust512':(1,0.,512),'direction_nearest_plain512':(1,0.,512),'direction_kernel_plain512':(1,0.,512)}
 if a.contrast:arms={'contrast_robust512':(1,0.,512),'contrast_plain512':(1,0.,512),'contrast_plusfine_robust512':(1,0.,512)}
 if a.stage_followup:arms={'stage_robust_reg512':(1,.02,512),'stage_robust_reg_all':(1,.02,2304),'stage_plain_reg_all':(1,.02,2304)}
 if a.only_arm:
  if a.only_arm not in arms:raise ValueError('arm is not in selected family')
  arms={a.only_arm:arms[a.only_arm]}
 if any((a.output/f'{s}_{k}.npz').exists() for k in arms):raise FileExistsError('frozen output exists')
 with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
 initial=a.initial or b/'sequential_depth_rescue_v327'/f'{s}.npz'
 if a.initial:
  from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
  poses,pm=read(initial)
  if pm.get('artifact_type')!='goal_maplet_direct_anonymous_feature_pose_v1' or pm.get('source_rgb_stored_or_consumed_at_runtime') is not False or not {'names','pose_w2c','usable'}.issubset(poses):raise ValueError('initial feature pose contract differs')
 else:poses,pm=_load_pose_candidate(initial)
 mp=a.map or b/'native_fine_v264/readout/map.npz'
 with np.load(mp) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(z['metadata_json'].item())
 if arrays_sha256(maps)!=mm['arrays_sha256'] or mm.get('query_ground_truth_read') is not False or canonical_json_sha256({k:v for k,v in mm.items() if k!='content_sha256'})!=mm['content_sha256']:raise ValueError('native map authority differs')
 cp=[b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz',b/'native_hybrid_fine_v301/readout'/f'{s}_reliability_rank32.npz'];corrs=[_load(v)[0] for v in cp]
 for c in corrs:
  if not np.array_equal(c['names'],poses['names']) or not np.array_equal(c['world_points'],maps['world_points'][c['prototype_atlas_row']]):raise ValueError('query/anchor binding differs')
 if not np.array_equal(corrs[0]['camera_matrices'],corrs[1]['camera_matrices']) or not np.array_equal(corrs[0]['radial_k1'],corrs[1]['radial_k1']):raise ValueError('camera differs')
 context_path=a.context_map or b/'native_context_v278/map.npz'
 with np.load(context_path) as z:context_arrays={k:z[k] for k in z.files if k!='metadata_json'};context_meta=json.loads(z['metadata_json'].item())
 exclusion=None;lineage_path=None
 if context_meta.get('artifact_type')=='goal_maplet_native_crossroute_region_library_v1':
  from feature_extract.tools.vfm.refinement_source_contract import validate_crossroute_support
  if arrays_sha256(context_arrays)!=context_meta['arrays_sha256'] or canonical_json_sha256({k:v for k,v in context_meta.items() if k!='content_sha256'})!=context_meta['content_sha256']:raise ValueError('cross-route map authority differs')
  if context_meta['native_atlas_sha256']!=mm['atlas_sha256']:raise ValueError('cross-route atlas differs')
  lineage_path=b/'native_fine_v264/mapping_lineage.npz'
  if file_sha256(lineage_path)!=mm['lineage_sha256']:raise ValueError('native feature lineage differs')
  with np.load(lineage_path) as z:
   exclusion=validate_crossroute_support(poses['names'],np.concatenate([c['prototype_atlas_row'] for c in corrs]),context_arrays['geometry_member_rows'],z['prototype_source_and_cell'][:,0],z['source_names'],context_meta['excluded_mapping_route'])
 else:
  source_names=set(context_meta['offline_mapping_source_names'])
  if source_names&set(poses['names'].astype(str)):raise ValueError('mapping/query overlap')
 if a.visibility:
  from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
  from feature_extract.tools.vfm.projected_surface_context import foreground_visibility
  cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];moge_dir=Path(cmd[cmd.index('--moge3_query')+1])
 sources={str(f):file_sha256(f) for f in [initial,mp,*cp,context_path,*([lineage_path,Path(__file__).with_name('refinement_source_contract.py')] if lineage_path else []),Path(__file__),Path(__file__).with_name('direct_anonymous_feature_pose.py')]};sources.update({str(Path(__file__).with_name('directional_feature_memory.py')):file_sha256(Path(__file__).with_name('directional_feature_memory.py'))} if a.directional else {});out={k:[] for k in arms};audit={k:[] for k in arms};start=time.perf_counter()
 for i,n in enumerate(poses['names'].astype(str)):
  pose=poses['pose_w2c'][i];K=corrs[0]['camera_matrices'][i].astype(float);k1=float(corrs[0]['radial_k1'][i]);records=[]
  for c in corrs:
   lo,hi=c['correspondence_offsets'][i:i+2];records.append(np.c_[c['query_tokens'][lo:hi],c['prototype_atlas_row'][lo:hi],c['query_measurements_xy'][lo:hi]])
  records=np.unique(np.concatenate(records),axis=0);tokens=records[:,0].astype(int);pr=records[:,1].astype(int);world=maps['world_points'][pr].astype(float)
  projected=cv2.projectPoints(world,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2);cam=world@pose[:3,:3].T+pose[:3,3];error=np.linalg.norm(projected-records[:,2:],axis=1)
  mask=maps['available'][pr]&(cam[:,2]>0)&np.isfinite(error)&(error<=4)&(projected>=4).all(1)&(projected<=np.array([251,139])).all(1);rows=np.flatnonzero(mask);order=np.lexsort((pr[rows],error[rows],tokens[rows]));rows=rows[order];_,first=np.unique(tokens[rows],return_index=True);rows=rows[first]
  visibility=None
  if a.visibility:
   path=moge_dir/n;qp,qn,qv=moge_tokens(path);sources[str(path)]=file_sha256(path);qd=qp[:,2].copy()
   if a.visibility=='shuffled':
    ids=np.flatnonzero(qv);qd[ids]=qd[np.random.default_rng(314).permutation(ids)]
   vm,visibility=foreground_visibility([cam[rows,2],cam[rows,2]],[np.ones(len(rows),bool)]*2,qd[tokens[rows]],qv[tokens[rows]])
   rows=rows[vm]
  cache=(a.query_cache or b/'native_fine_v264/query_cache')/n;grids,checkpoint=load_grids(cache,mm['projection_sha256']);sources[str(cache)]=file_sha256(cache)
  if checkpoint!=mm['checkpoint_sha256']:raise ValueError('feature checkpoint differs')
  if a.shuffle_query_spatial:
   grids=[g.reshape(-1,g.shape[-1])[np.random.default_rng(355).permutation(g.shape[0]*g.shape[1])].reshape(g.shape) for g in grids]
  for arm,(level,reg,budget) in arms.items():
   selected=rows if len(rows)<=budget else rows[np.linspace(0,len(rows)-1,budget,dtype=int)]
   if len(selected)<32:result=pose.copy();record=dict(accepted=False,reason='insufficient fixed support')
   else:
    weights=None
    if 'balanced' in arm:
     _,ix,counts=np.unique(plane[pr[selected]],return_inverse=True,return_counts=True);weights=1/counts[ix]
    target=maps['coarse' if level==0 else 'fine'][pr[selected]]
    if a.directional:
     from feature_extract.tools.vfm.directional_feature_memory import directional_target
     target=directional_target(maps['fine_modes'][pr[selected]],maps['view_directions'][pr[selected]],maps['mode_count'][pr[selected]],maps['world_points'][pr[selected]],pose,'nearest' if 'nearest' in arm else 'kernel')
    if a.contrast:target=maps['fine'][pr[selected]].astype(float)-maps['coarse'][pr[selected]].astype(float)
    result,record=refine(pose,world[selected],K,k1,grids[level],target,reg,weights=weights,robust='robust' in arm,constrained=a.constrained or a.feature_controls or a.directional or a.contrast or a.stage_followup,box_slsqp=a.box_slsqp,bias_mode='frozen' if 'frozen_bias' in arm else ('profiled' if 'profiled_bias' in arm else 'none'),additional_grid=grids[0] if 'multiscale' in arm else (grids[1] if 'plusfine' in arm else None),additional_target=maps['coarse'][pr[selected]] if 'multiscale' in arm else (maps['fine'][pr[selected]] if 'plusfine' in arm else None),contrast_grid=grids[0] if a.contrast else None)
   out[arm].append(result);audit[arm].append(dict(name=n,selected_tokens=tokens[selected].tolist(),prototype_rows=pr[selected].tolist(),depth_visibility=visibility,**record))
  if (i+1)%20==0:print(s,i+1,flush=True)
 for arm,value in out.items():
  arr=dict(names=poses['names'],pose_w2c=np.array(value),usable=np.isfinite(value).all((1,2)));meta=dict(artifact_type='goal_maplet_direct_anonymous_feature_pose_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,mapping_source_exclusion=exclusion,source_sha256=sources,arm=arm,constrained_optimizer=a.constrained or a.feature_controls or a.directional or a.contrast or a.stage_followup,bias_mode='frozen' if 'frozen_bias' in arm else ('profiled' if 'profiled_bias' in arm else 'none'),multiscale='multiscale' in arm,box_slsqp=a.box_slsqp,visibility=a.visibility,diagnostic_spatial_shuffle=a.shuffle_query_spatial,contrast_feature=a.contrast,direction_conditioned=a.directional,direction_frozen_at_initializer=a.directional,robust_descriptor_delta=.5 if 'robust' in arm else None,plane_balance='equal physical plane mass' if 'balanced' in arm else None,fixed_unique_token_support=True,maximum_center_step_m=.5,maximum_rotation_step_deg=3,maximum_image_motion_px=4,feature_residual_not_calibrated_likelihood=True,regularization_scope='image motion relative to initializer, not independent evidence')
  meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(a.output/f'{s}_{arm}.npz',**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)));(a.output/f'{s}_{arm}_audit.json').write_text(json.dumps(audit[arm]))
 (a.output/f'{s}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,scope='cached feature input; all four controls; not RGB-to-pose latency')))
if __name__=='__main__':main()
