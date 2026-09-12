"""Frozen native readout controls: mapping calibration and observed-surface readout."""
import argparse,json,time
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.native_fine_measurement import same_surface_sample,eligible_rows
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import _load_frozen_poses
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
    mapfile=b/'native_fine_v264/readout/map.npz';calfile=b/'native_fine_calibration_v267/calibration.json';cal=json.load(open(calfile))
    if cal.get('query_ground_truth_read') is not False or cal.get('heldout_used_to_fit') is not False or canonical_json_sha256({k:v for k,v in cal.items() if k!='content_sha256'})!=cal['content_sha256'] or cal['map_sha256']!=file_sha256(mapfile):raise ValueError('calibration provenance differs')
    with np.load(mapfile) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(str(z['metadata_json']))
    if arrays_sha256(maps)!=mm['arrays_sha256']:raise ValueError('map hash differs')
    with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]]);arms=['fine_shrink','fine_calibrated','boundary','boundary_joint']
    prot=dict(anchors_fixed=True,query_ground_truth_read=False,map_sha256=file_sha256(mapfile),arms=arms,mapping_calibration=dict(file_sha256=file_sha256(calfile),alpha=cal['alpha'],variance_scale=cal['variance_scale'],query_ground_truth_read=False,heldout_used_to_fit=False),surface_readout='bilinear feature centers and probe must have same observed MoGe plane id; absent center support freezes original; frozen backbone mixing cannot be undone',maximum_tokens=128,mask_reference='exact v264 same-plane >=3 group eligibility',map_descriptors_unchanged=True)
    (o/'protocol.json').write_text(json.dumps(prot,indent=2))
    for split in ['seq10','shard0','shard1','shard2','shard3']:
        cp=b/'native_scope_v254/wide'/f'{split}_appearance.npz';corr,meta=_load(cp);initial=b/'native_scope_mainline_v255'/(split+'_wide')/'moge.npz';poses,_=_load_frozen_poses(initial)
        if not np.array_equal(corr['names'],poses['names']) or not np.array_equal(corr['world_points'],maps['world_points'][corr['prototype_atlas_row']]):raise ValueError('pose/anchor binding differs')
        oldfine,_=_load(b/'native_fine_v264/readout'/f'{split}_fine.npz');oldjoint,_=_load(b/'native_fine_v264/readout'/f'{split}_joint.npz')
        stage=dict(json.load(open(b/'native_scope_mainline_v255'/(split+'_wide')/'protocol.json'))['commands'])['moge'];qpdir=Path(stage[stage.index('--query_plane_dir')+1]);variants={k:{f:v.copy() for f,v in corr.items()} for k in arms};audit=[]
        for i,name in enumerate(corr['names'].astype(str)):
            started=time.perf_counter();lo,hi=map(int,corr['correspondence_offsets'][i:i+2])
            if not poses['usable'][i]:audit.append(dict(name=name,eligible=0));continue
            proj,cam=_project(poses['pose_w2c'][i],corr['world_points'][lo:hi],corr['camera_matrices'][i],corr['radial_k1'][i]);rows,groups=eligible_rows(corr,lo,hi,proj,cam,maps['available'],plane)
            if not len(rows):audit.append(dict(name=name,eligible=0));continue
            absolute=lo+rows;tok=corr['query_tokens'][absolute];pr=corr['prototype_atlas_row'][absolute];xy=corr['query_measurements_xy'][absolute];center=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];probes=xy[:,None]+offsets;inside=(np.abs(probes-center[:,None])<=2+1e-8).all(2)
            gridpath=b/'native_fine_v264/query_cache'/name;grids,ck=load_grids(gridpath,mm['projection_sha256'])
            if ck!=mm['checkpoint_sha256']:raise ValueError('query checkpoint differs')
            sim=np.sum(sample_grid(grids[1],probes)*maps['fine'][pr,None].astype(np.float32),axis=-1);sim[~inside]=-np.inf
            replay=probes[np.arange(len(rows)),select_offsets(sim,groups,False)];replayjoint=probes[np.arange(len(rows)),select_offsets(sim,groups,True)]
            if not np.array_equal(replay.astype(oldfine['query_measurements_xy'].dtype),oldfine['query_measurements_xy'][absolute]) or not np.array_equal(replayjoint.astype(oldjoint['query_measurements_xy'].dtype),oldjoint['query_measurements_xy'][absolute]):raise ValueError('v264 fine/joint pixel replay differs')
            for arm in ['fine_shrink','fine_calibrated']:variants[arm]['query_measurements_xy'][absolute]=xy+cal['alpha']*(replay-xy)
            variants['fine_calibrated']['query_measurement_variance_px2'][absolute]*=cal['variance_scale']
            qp,qm=QueryPlaneRegions.load_npz(qpdir/name)
            if qm.get('uses_pose_or_ground_truth') is not False:raise ValueError('unsafe query surface labels')
            desc,supported=same_surface_sample(grids[1],probes,qp.labels,corr['provenance_region_plane_atlas_row'][absolute,0,None]);bsim=np.sum(desc*maps['fine'][pr,None].astype(np.float32),axis=-1);bsim[~(inside&supported)]=-np.inf
            absent=~supported[:,4];bsim[absent]=-np.inf;bsim[absent,4]=0.
            for arm,joint in [('boundary',False),('boundary_joint',True)]:variants[arm]['query_measurements_xy'][absolute]=probes[np.arange(len(rows)),select_offsets(bsim,groups,joint)]
            audit.append(dict(name=name,eligible=len(rows),groups=[len(g) for g in groups],center_supported_fraction=float(np.mean(supported[:,4])),probe_supported_fraction=float(np.mean(supported)),changed_fraction={arm:float(np.mean(np.any(variants[arm]['query_measurements_xy'][absolute]!=xy,axis=1))) for arm in arms},query_cache_sha256=file_sha256(gridpath),query_planes_sha256=file_sha256(qpdir/name),seconds=time.perf_counter()-started))
        for arm,arrays in variants.items():
            om=dict(meta);om.pop('content_sha256',None);om.update(arrays_sha256=arrays_sha256(arrays),fine_readout_protocol=prot,fine_readout_arm=arm,coarse_correspondence_file_sha256=file_sha256(cp),frozen_initial_pose_sha256=file_sha256(initial),query_measurement_semantics='native_matched_fine_query_pixel_update_inside_original_4x4_token',fine_measurement_variance_recalibrated=arm=='fine_calibrated');om['content_sha256']=canonical_json_sha256(om)
            dest=o/f'{split}_{arm}.npz';temp=dest.with_name(dest.name+'.temporary.npz');np.savez_compressed(temp,**arrays,metadata_json=np.array(json.dumps(om,sort_keys=True)));_load(temp);temp.replace(dest)
        (o/f'{split}_audit.json').write_text(json.dumps(audit,indent=2));print(split,'built',flush=True)

if __name__=='__main__':main()
