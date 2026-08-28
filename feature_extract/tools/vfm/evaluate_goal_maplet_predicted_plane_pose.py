"""Pose-free Top-1 RADIO-plane matches plus RADIO-basin regularized pose."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions
from feature_extract.tools.vfm.evaluate_goal_maplet_moge3_planes_against_rendered_map import rotation_from_normals
ALPHA=.25;SCALE_PRIOR=.04
def solve(A,b,C0,weight):
 root=np.sqrt(weight/max(weight.mean(),1e-15));Aw=A*root[:,None];bw=b*root
 augmented=np.r_[Aw,np.c_[np.sqrt(ALPHA)*np.eye(3),np.zeros(3)],np.array([[0,0,0,np.sqrt(SCALE_PRIOR)]])];target=np.r_[bw,np.sqrt(ALPHA)*C0,np.sqrt(SCALE_PRIOR)];x=np.linalg.lstsq(augmented,target,rcond=None)[0];residual=A@x-b;return x,float(np.average(residual**2,weights=weight)+ALPHA*np.sum((x[:3]-C0)**2)+SCALE_PRIOR*(x[3]-1)**2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--correspondence_report',type=Path,required=True);p.add_argument('--direct_candidates',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plane=GeometryNativePlanarMap.load_npz(a.planar_map);matching=json.loads(a.correspondence_report.read_text())
 with np.load(a.direct_candidates,allow_pickle=False) as d:image_ids=np.asarray(d['image_ids']);poses=np.asarray(d['candidate_poses_w2c'])[:,1:]
 by_image={x['image']:x for x in matching['rows']};rows=[]
 for qi,image_id in enumerate(image_ids.tolist()):
  name=image_id.replace('/','__')+'.npz';records=by_image[name]['regions']
  with np.load(a.moge_dir/name,allow_pickle=False) as d:q=extract_query_plane_regions(d['points_camera'],d['normal_camera'],d['valid'])
  # One query region per predicted map plane, retaining strongest pixel-mass evidence.
  selected={}
  for record in records:
   key=(record['predicted_score']*record['pixels'],record['pixels'],-record['region']);row=record['predicted_plane']
   if row not in selected or key>selected[row][0]:selected[row]=(key,record)
  records=[x[1] for x in selected.values() if x[1]['predicted_score']>0];result={'image_id':image_id,'match_count':len(records),'usable':False}
  if len(records)>=4:
   qn=np.asarray([q.normals_camera[x['region']] for x in records]);qd=np.asarray([q.offsets_camera[x['region']] for x in records]);maprows=np.asarray([x['predicted_plane'] for x in records]);base_mn=plane.normals_world[maprows];md=np.einsum('ij,ij->i',base_mn,plane.centers_world[maprows]);weight=np.asarray([max(x['predicted_score']*x['pixels'],1e-6) for x in records]);candidates=[]
   for pose in poses[qi]:
    R0=pose[:3,:3];C0=-R0.T@pose[:3,3];mn=base_mn.copy();predicted=qn@R0;sign=np.where(np.einsum('ij,ij->i',mn,predicted)>=0,1.,-1.);mn*=sign[:,None];signed_md=md*sign;Rcw=rotation_from_normals(qn,mn);A=np.c_[mn,qd];x,objective=solve(A,signed_md,C0,weight);candidates.append((objective,x,Rcw))
   index=min(range(len(candidates)),key=lambda i:(candidates[i][0],i));objective,x,Rcw=candidates[index]
   with np.load(a.contributors/name,allow_pickle=False) as d:gt=np.asarray(d['pose_w2c'],np.float64)
   gtcenter=-gt[:3,:3].T@gt[:3,3];translation=float(np.linalg.norm(x[:3]-gtcenter));rotation=float(Rotation.from_matrix(Rcw.T@gt[:3,:3].T).magnitude()*180/np.pi);result.update(usable=True,selected_rank=index+1,translation_error_m=translation,rotation_error_deg=rotation,scale=float(x[3]),objective=objective)
  rows.append(result)
 usable=[x for x in rows if x['usable']];t=np.asarray([x['translation_error_m'] for x in usable]);r=np.asarray([x['rotation_error_deg'] for x in usable]);report={'artifact_type':'goal_maplet_predicted_top1_plane_radio_basin_pose_v1','query_count':len(rows),'usable_count':len(usable),'uses_gt_in_pose_scoring_or_selection':False,'gt_used_only_for_final_error':True,'alpha':ALPHA,'scale_prior':SCALE_PRIOR,'median_translation_m':float(np.median(t)) if len(t) else None,'median_rotation_deg':float(np.median(r)) if len(r) else None,'recall_2m45':float(np.mean((t<=2)&(r<=45))) if len(t) else 0.,'recall_1m10':float(np.mean((t<=1)&(r<=10))) if len(t) else 0.,'production_eligible':False,'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
