"""Expand independent calibrated fine readout to its already fixed 128-token cap."""
import argparse,json,time
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.native_fine_measurement import eligible_rows
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import _load_frozen_poses
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial_backend',choices=['wide','token'],default='wide');p.add_argument('--measurement_mask',choices=['full','group'],default='full');a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False);calfile=b/'native_fine_calibration_v267/calibration.json';cal=json.load(open(calfile));mapfile=b/'native_fine_v264/readout/map.npz'
    if cal['query_ground_truth_read'] is not False or cal['heldout_used_to_fit'] is not False or cal['map_sha256']!=file_sha256(mapfile) or canonical_json_sha256({k:v for k,v in cal.items() if k!='content_sha256'})!=cal['content_sha256']:raise ValueError('calibration binding differs')
    with np.load(mapfile) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(str(z['metadata_json']))
    if arrays_sha256(maps)!=mm['arrays_sha256']:raise ValueError('map arrays differ')
    with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
    for split in ['seq10','shard0','shard1','shard2','shard3']:
        cp=b/'native_scope_v254/wide'/f'{split}_appearance.npz';corr,meta=_load(cp);initial=b/('native_scope_mainline_v255' if a.initial_backend=='wide' else 'native_global_proposals_v258')/(split+'_'+a.initial_backend)/'moge.npz';poses,_=_load_frozen_poses(initial);result={k:v.copy() for k,v in corr.items()};old,oldmeta=_load(b/'native_fine_calibration_v267/readout'/f'{split}_fine_calibrated.npz');audit=[]
        if not np.array_equal(corr['names'],poses['names']) or not np.array_equal(corr['world_points'],maps['world_points'][corr['prototype_atlas_row']]):raise ValueError('query/anchor binding differs')
        for i,name in enumerate(corr['names'].astype(str)):
            start=time.perf_counter()
            if not poses['usable'][i]:audit.append(dict(name=name,eligible=0));continue
            lo,hi=map(int,corr['correspondence_offsets'][i:i+2]);proj,cam=_project(poses['pose_w2c'][i],corr['world_points'][lo:hi],corr['camera_matrices'][i],corr['radial_k1'][i]);rows,_=eligible_rows(corr,lo,hi,proj,cam,maps['available'],plane,a.measurement_mask=='group');previous,_=eligible_rows(corr,lo,hi,proj,cam,maps['available'],plane)
            if not len(rows):audit.append(dict(name=name,eligible=0));continue
            absolute=lo+rows;tok=corr['query_tokens'][absolute];pr=corr['prototype_atlas_row'][absolute];xy=corr['query_measurements_xy'][absolute];probes=xy[:,None]+offsets;center=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
            cache=b/'native_fine_v264/query_cache'/name;grids,ck=load_grids(cache,mm['projection_sha256'])
            if ck!=mm['checkpoint_sha256']:raise ValueError('checkpoint differs')
            sim=np.sum(sample_grid(grids[1],probes)*maps['fine'][pr,None].astype(np.float32),axis=-1);sim[(np.abs(probes-center[:,None])>2+1e-8).any(2)]=-np.inf;best=select_offsets(sim,[],False)
            result['query_measurements_xy'][absolute]=xy+cal['alpha']*(probes[np.arange(len(rows)),best]-xy);result['query_measurement_variance_px2'][absolute]*=cal['variance_scale']
            for key in ['query_measurements_xy','query_measurement_variance_px2']:
                if a.initial_backend=='wide' and not np.array_equal(result[key][lo+previous],old[key][lo+previous]):raise ValueError('previous eligible readout changed')
            audit.append(dict(name=name,eligible=len(rows),previous_eligible=len(previous),changed_fraction=float(np.mean(best!=4)),query_cache_sha256=file_sha256(cache),seconds=time.perf_counter()-start))
        om=dict(oldmeta);om.pop('content_sha256',None);om.update(arrays_sha256=arrays_sha256(result),fine_readout_arm=('fine_calibrated128' if a.measurement_mask=='full' else 'fine_calibrated'),frozen_initial_pose_sha256=file_sha256(initial));om['fine_readout_protocol']=dict(om['fine_readout_protocol'],mask_reference='one per token with predicted residual<=4px and map available; uniformly spaced token cap128; group restriction='+str(a.measurement_mask=='group'),actual_measurement_budget_changed=a.measurement_mask=='full',initial_backend=a.initial_backend);om['content_sha256']=canonical_json_sha256(om)
        arm='fine_calibrated128' if a.measurement_mask=='full' else 'fine_calibrated';dest=o/f'{split}_{arm}.npz';np.savez_compressed(dest,**result,metadata_json=np.array(json.dumps(om,sort_keys=True)));_load(dest);(o/f'{split}_audit.json').write_text(json.dumps(audit,indent=2));print(split,'built',flush=True)

if __name__=='__main__':main()
