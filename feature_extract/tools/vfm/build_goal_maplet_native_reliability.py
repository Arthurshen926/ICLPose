"""Conditional fine readout and matched measurement-budget controls on token initializers."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.native_fine_reliability import reliability_features,predict
from feature_extract.tools.vfm.native_fine_measurement import eligible_rows
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import _load_frozen_poses
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--input_manifest',type=Path);p.add_argument('--arms',nargs='+',choices=['reliability_mean','reliability_both','reliability_full','reliability_uniform32','reliability_rank32','fixed_full']);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False);modelpath=b/'native_reliability_v274/model.json';model=json.load(open(modelpath));mapfile=b/'native_fine_v264/readout/map.npz'
 if model['query_ground_truth_read'] is not False or model['heldout_used_to_fit'] is not False or model['map_sha256']!=file_sha256(mapfile) or model['content_sha256']!=canonical_json_sha256({k:v for k,v in model.items() if k!='content_sha256'}):raise ValueError('reliability binding differs')
 with np.load(mapfile) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(str(z['metadata_json']))
 if arrays_sha256(maps)!=mm['arrays_sha256']:raise ValueError('map arrays differ')
 with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
 inputs=json.loads(a.input_manifest.read_text()) if a.input_manifest else {}
 arms=a.arms or ['reliability_mean','reliability_both','reliability_full','reliability_uniform32','reliability_rank32','fixed_full'];offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
 for split in ['seq10','shard0','shard1','shard2','shard3']:
  cp=Path(inputs[split]['correspondences']) if inputs else b/'native_scope_v254/wide'/f'{split}_appearance.npz';corr,coarse_meta=_load(cp);reference,meta=_load(b/'native_token_fine_v273/readout'/f'{split}_fine_calibrated.npz');initial=Path(inputs[split]['initial_pose']) if inputs else b/'native_global_proposals_v258'/(split+'_token')/'moge.npz';poses,_=_load_frozen_poses(initial);variants={arm:{k:v.copy() for k,v in corr.items()} for arm in arms};audit=[]
  if inputs:
   meta=dict(coarse_meta,fine_readout_protocol=meta['fine_readout_protocol'],coarse_correspondence_file_sha256=file_sha256(cp),frozen_initial_pose_sha256=file_sha256(initial),query_measurement_semantics='native_matched_fine_query_pixel_update_inside_original_4x4_token')
  if not np.array_equal(corr['names'],poses['names']) or not np.array_equal(corr['world_points'],maps['world_points'][corr['prototype_atlas_row']]):raise ValueError('anchor/query binding differs')
  for i,name in enumerate(corr['names'].astype(str)):
   if not poses['usable'][i]:audit.append(dict(name=name,eligible=0));continue
   lo,hi=map(int,corr['correspondence_offsets'][i:i+2]);project,cam=_project(poses['pose_w2c'][i],corr['world_points'][lo:hi],corr['camera_matrices'][i],corr['radial_k1'][i]);rows,_=eligible_rows(corr,lo,hi,project,cam,maps['available'],plane,False);group,_=eligible_rows(corr,lo,hi,project,cam,maps['available'],plane)
   if not len(rows):audit.append(dict(name=name,eligible=0));continue
   absolute=lo+rows;token=corr['query_tokens'][absolute];pr=corr['prototype_atlas_row'][absolute];xy=corr['query_measurements_xy'][absolute];variance=corr['query_measurement_variance_px2'][absolute];probes=xy[:,None]+offsets;center=np.c_[(token%64)*4+1.5,(token//64)*4+1.5]
   grids,ck=load_grids(b/'native_fine_v264/query_cache'/name,mm['projection_sha256'])
   if ck!=mm['checkpoint_sha256']:raise ValueError('query feature checkpoint differs')
   sim=np.sum(sample_grid(grids[1],probes)*maps['fine'][pr,None].astype(np.float32),axis=-1);sim[(np.abs(probes-center[:,None])>2+1e-8).any(2)]=-np.inf;best=select_offsets(sim,[],False);delta=probes[np.arange(len(rows)),best]-xy;features=reliability_features(sim,delta,variance);alpha,var=predict(model['model'],features,variance)
   groupix=np.flatnonzero(np.isin(rows,group));budget=min(32,len(rows));uniform=np.linspace(0,len(rows)-1,budget,dtype=int);benefit=alpha**2*np.sum(delta**2,axis=1);ranked=np.lexsort((token,-benefit))[:budget]
   masks={'reliability_mean':groupix,'reliability_both':groupix,'reliability_full':np.arange(len(rows)),'reliability_uniform32':uniform,'reliability_rank32':ranked,'fixed_full':np.arange(len(rows))}
   for arm,ix in masks.items():
    if arm not in variants:continue
    aa=np.full(len(rows),model['fixed_alpha']) if arm=='fixed_full' else alpha;vv=variance*model['fixed_variance_scale'] if arm in ['fixed_full','reliability_mean'] else var
    variants[arm]['query_measurements_xy'][absolute[ix]]=xy[ix]+aa[ix,None]*delta[ix];variants[arm]['query_measurement_variance_px2'][absolute[ix]]=vv[ix]
   for key in ([] if inputs or 'fixed_full' not in variants else ['query_measurements_xy','query_measurement_variance_px2']):
    if not np.array_equal(variants['fixed_full'][key][lo+group],reference[key][lo+group]):raise ValueError('fixed baseline overlap replay differs')
   audit.append(dict(name=name,eligible=len(rows),selected={arm:len(ix) for arm,ix in masks.items()},alpha_mean=float(alpha.mean()),alpha_range=[float(alpha.min()),float(alpha.max())],variance_multiplier_mean=float(np.mean(var/variance)),rank_uniform_overlap_fraction=float(len(set(ranked)&set(uniform))/budget),mean_predicted_gain_selected=float(benefit[ranked].mean()),mean_predicted_gain_uniform=float(benefit[uniform].mean())))
  for arm,arrays in variants.items():
   om=dict(meta);om.pop('content_sha256',None);om.update(arrays_sha256=arrays_sha256(arrays),fine_readout_arm='fine_calibrated128' if arm=='fixed_full' else arm,fine_measurement_variance_recalibrated=True);om['fine_readout_protocol']=dict(om['fine_readout_protocol'],reliability_model_sha256=file_sha256(modelpath),reliability_query_ground_truth_read=False,reliability_heldout_used_to_fit=False,measurement_policy=arm,arms=arms,mask_reference='one per token with predicted residual<=4px and map available; uniformly spaced token cap128; no same-plane group restriction for rank32/uniform32/full',surface_readout='ordinary bilinear fine feature sampling within original 4x4 token; no observed-surface feature mask',actual_measurement_budget_changed=arm in ['reliability_rank32','reliability_uniform32'],maximum_updated_rows=min(32,128) if arm in ['reliability_rank32','reliability_uniform32'] else 128,input_manifest_sha256=file_sha256(a.input_manifest) if a.input_manifest else None,maximum_probe_tokens=128,ranking_utility='predicted alpha squared times raw shift squared; estimated coordinate MSE reduction, not learned region-set or pose utility',ranking_after_fine_probe=True);om['content_sha256']=canonical_json_sha256(om)
   dest=o/f'{split}_{arm}.npz';np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(om,sort_keys=True)));_load(dest)
  (o/f'{split}_audit.json').write_text(json.dumps(audit,indent=2));print(split,'built',flush=True)

if __name__=='__main__':main()
