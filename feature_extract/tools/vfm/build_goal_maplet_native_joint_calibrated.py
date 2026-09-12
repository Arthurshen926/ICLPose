"""Apply mapping-only shared-shift calibration and freeze correlated group sidecars."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.native_fine_measurement import eligible_rows,load_measurement_groups
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import _load_frozen_poses
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False);calfile=b/'native_joint_calibration_v269/calibration.json';cal=json.load(open(calfile));mapfile=b/'native_fine_v264/readout/map.npz'
    if cal['query_ground_truth_read'] is not False or cal['heldout_used_to_fit'] is not False or cal['map_sha256']!=file_sha256(mapfile) or canonical_json_sha256({k:v for k,v in cal.items() if k!='content_sha256'})!=cal['content_sha256']:raise ValueError('joint calibration binding differs')
    with np.load(mapfile) as z:available=z['available']
    with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
    for split in ['seq10','shard0','shard1','shard2','shard3']:
        corrfile=b/'native_scope_v254/wide'/f'{split}_appearance.npz';corr,meta=_load(corrfile);jointfile=b/'native_fine_v264/readout'/f'{split}_joint.npz';joint,_=_load(jointfile);initial=b/'native_scope_mainline_v255'/(split+'_wide')/'moge.npz';poses,_=_load_frozen_poses(initial)
        if not np.array_equal(corr['names'],poses['names']):raise ValueError('query order differs')
        result={k:v.copy() for k,v in corr.items()};ids=np.full(len(corr['query_tokens']),-1,np.int64);gid=0
        for i in range(len(corr['names'])):
            if not poses['usable'][i]:continue
            lo,hi=map(int,corr['correspondence_offsets'][i:i+2]);proj,cam=_project(poses['pose_w2c'][i],corr['world_points'][lo:hi],corr['camera_matrices'][i],corr['radial_k1'][i]);rows,groups=eligible_rows(corr,lo,hi,proj,cam,available,plane);absolute=lo+rows
            result['query_measurements_xy'][absolute]+=cal['alpha']*(joint['query_measurements_xy'][absolute]-corr['query_measurements_xy'][absolute]);result['query_measurement_variance_px2'][absolute]*=cal['variance_scale']
            for g in groups:ids[absolute[g]]=gid;gid+=1
        protocol=dict(anchors_fixed=True,query_ground_truth_read=False,mapping_calibration=dict(file_sha256=file_sha256(calfile),alpha=cal['alpha'],variance_scale=cal['variance_scale'],query_ground_truth_read=False,heldout_used_to_fit=False),joint_readout_file_sha256=file_sha256(jointfile))
        om=dict(meta);om.pop('content_sha256',None);om.update(arrays_sha256=arrays_sha256(result),fine_readout_protocol=protocol,fine_readout_arm='joint_calibrated',coarse_correspondence_file_sha256=file_sha256(corrfile),frozen_initial_pose_sha256=file_sha256(initial),query_measurement_semantics='native_matched_fine_query_pixel_update_inside_original_4x4_token',fine_measurement_variance_recalibrated=True);om['content_sha256']=canonical_json_sha256(om)
        dest=o/f'{split}_joint_calibrated.npz';np.savez_compressed(dest,**result,metadata_json=np.array(json.dumps(om,sort_keys=True)));_load(dest)
        ga=dict(names=corr['names'],correspondence_offsets=corr['correspondence_offsets'],group_ids=ids);gm=dict(artifact_type='goal_maplet_correlated_measurement_groups_v1',arrays_sha256=arrays_sha256(ga),query_ground_truth_read=False,rho=cal['rho'],calibration_file_sha256=file_sha256(calfile),correspondence_file_sha256=file_sha256(dest));gm['content_sha256']=canonical_json_sha256(gm)
        groupfile=o/f'{split}_groups.npz';np.savez_compressed(groupfile,**ga,metadata_json=np.array(json.dumps(gm,sort_keys=True)));load_measurement_groups(groupfile,dest,result);print(split,'built',flush=True)

if __name__=='__main__':main()
