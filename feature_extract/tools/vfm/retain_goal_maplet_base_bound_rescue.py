"""Apply an evaluated rescue only when its reference is the actual current pose."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def eligible_rescue(current,reference,choice,alternative_usable):
    current=np.asarray(current);reference=np.asarray(reference);choice=np.asarray(choice)
    if current.shape!=reference.shape or current.shape!=(len(choice),4,4):raise ValueError('rescue pose dimensions differ')
    return (choice==1)&np.asarray(alternative_usable,bool)&np.isfinite(current).all((1,2))&np.all(current==reference,axis=(1,2))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['current','reference','alternative','verifier','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 poses=[_load_pose_candidate(path) for path in [a.current,a.reference,a.alternative]]
 with np.load(a.verifier) as z:v={k:z[k] for k in z.files if k!='metadata_json'};vm=json.loads(z['metadata_json'].item())
 if arrays_sha256(v)!=vm['arrays_sha256'] or canonical_json_sha256({k:w for k,w in vm.items() if k!='content_sha256'})!=vm['content_sha256'] or vm.get('query_pose_or_ground_truth_read') is not False:raise ValueError('verifier authority differs')
 if not all(np.array_equal(v['names'],pa['names']) for pa,_ in poses):raise ValueError('query order differs')
 if vm.get('artifact_type')=='goal_maplet_projected_surface_context_selection_v1':
  if vm.get('arm')!='structure_guarded':raise ValueError('expected frozen spatially guarded verifier')
  for path in [a.reference,a.alternative]:
   if vm['source_sha256'].get(str(path))!=file_sha256(path):raise ValueError('verifier reference/alternate binding differs')
 elif vm.get('artifact_type')=='goal_maplet_coordinate_pose_geometry_consensus_v1':
  if vm.get('candidate_order')!='primary_then_alternate':raise ValueError('candidate ordering differs')
  for role,path,(_,pm) in zip(['primary','alternate'],[a.reference,a.alternative],poses[1:]):
   if vm.get(role+'_pose_file_sha256')!=file_sha256(path) or vm.get(role+'_pose_content_sha256')!=pm['content_sha256']:raise ValueError('consensus reference/alternate binding differs')
 else:raise ValueError('unsupported frozen verifier')
 for i,c in enumerate(v['selected_branch']):
  if c not in (0,1) or not np.array_equal(v['pose_w2c'][i],poses[1+int(c)][0]['pose_w2c'][i],equal_nan=True):raise ValueError('verifier is not an exact endpoint choice')
 choose=eligible_rescue(poses[0][0]['pose_w2c'],poses[1][0]['pose_w2c'],v['selected_branch'],poses[2][0]['usable']);arr=dict(names=v['names'],pose_w2c=np.where(choose[:,None,None],poses[2][0]['pose_w2c'],poses[0][0]['pose_w2c']),usable=np.where(choose,poses[2][0]['usable'],poses[0][0]['usable']),selected_branch=choose.astype(np.int8))
 m=dict(artifact_type='goal_maplet_base_bound_rescue_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,source_sha256={str(p):file_sha256(p) for p in [a.current,a.reference,a.alternative,a.verifier]},rule='apply frozen rescue only to an exactly identical current/reference pose; otherwise preserve current',no_per_query_nonregression_guarantee=True)
 m['content_sha256']=canonical_json_sha256(m);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arr,metadata_json=np.array(json.dumps(m,sort_keys=True)))
if __name__=='__main__':main()
