"""RADIO-basin-conditioned global selection from per-plane Top-K matches."""
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions,extract_query_plane_regions
from feature_extract.tools.vfm.evaluate_goal_maplet_moge3_planes_against_rendered_map import rotation_from_normals
from feature_extract.tools.vfm.evaluate_goal_maplet_predicted_plane_pose import solve
TOPK=5;RANK_COST=.15;NORMAL_SCALE_DEG=15.
def robust_rotation(qn,mn,weight):
 combinations=list(itertools.combinations(range(len(qn)),3));indices=np.linspace(0,len(combinations)-1,min(256,len(combinations)),dtype=int);best=None
 for index in indices:
  subset=np.asarray(combinations[index]);R=rotation_from_normals(qn[subset],mn[subset]);predicted=qn@R.T;angle=np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i',predicted,mn)),0,1)));inlier=angle<=10.;key=(float(weight[inlier].sum()),int(inlier.sum()),-float(np.median(angle)))
  if best is None or key>best[0]:best=(key,R,inlier,angle)
 if best is None:return rotation_from_normals(qn,mn),np.ones(len(qn),bool)
 inlier=best[2]
 if inlier.sum()>=3:
  reference=qn[inlier]@best[1].T;target=mn[inlier].copy();target*=np.where(np.einsum('ij,ij->i',reference,target)>=0,1.,-1.)[:,None];return rotation_from_normals(qn[inlier],target),inlier
 return best[1],inlier
def main():
 p=argparse.ArgumentParser();p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--query_plane_dir',type=Path);p.add_argument('--correspondence_report',type=Path,required=True);p.add_argument('--direct_candidates',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plane=GeometryNativePlanarMap.load_npz(a.planar_map);matching=json.loads(a.correspondence_report.read_text());by_image={x['image']:x for x in matching['rows']}
 with np.load(a.direct_candidates,allow_pickle=False) as d:image_ids=np.asarray(d['image_ids']);poses=np.asarray(d['candidate_poses_w2c'])[:,1:]
 rows=[]
 for qi,image_id in enumerate(image_ids.tolist()):
  name=image_id.replace('/','__')+'.npz';records=by_image[name]['regions']
  if a.query_plane_dir:q,_=QueryPlaneRegions.load_npz(a.query_plane_dir/name)
  else:
   with np.load(a.moge_dir/name,allow_pickle=False) as d:q=extract_query_plane_regions(d['points_camera'],d['normal_camera'],d['valid'])
  candidate_solutions=[]
  for rank_pose,pose in enumerate(poses[qi]):
   R0=pose[:3,:3];C0=-R0.T@pose[:3,3];chosen=[]
   for record in records:
    nq=q.normals_camera[record['region']];predicted=nq@R0;options=[]
    for rank,maprow in enumerate(record['top10'][:TOPK]):
     nm=plane.normals_world[maprow];cosine=np.clip(abs(float(nm@predicted)),0,1);angle=np.degrees(np.arccos(cosine));cost=(angle/NORMAL_SCALE_DEG)**2+RANK_COST*rank;options.append((cost,rank,maprow))
    if options:chosen.append((min(options),record))
   # Enforce one query region per map plane after candidate-conditioned choice.
   unique={}
   for option,record in chosen:
    maprow=option[2];key=(-option[0],record['pixels'],-record['region'])
    if maprow not in unique or key>unique[maprow][0]:unique[maprow]=(key,option,record)
   selected=list(unique.values())
   if len(selected)<4:continue
   qn=np.asarray([q.normals_camera[x[2]['region']] for x in selected]);qd=np.asarray([q.offsets_camera[x[2]['region']] for x in selected]);maprows=np.asarray([x[1][2] for x in selected]);mn=plane.normals_world[maprows].copy();predicted=qn@R0;sign=np.where(np.einsum('ij,ij->i',mn,predicted)>=0,1.,-1.);mn*=sign[:,None];md=np.einsum('ij,ij->i',mn,plane.centers_world[maprows]);weight=np.asarray([max(x[2]['pixels']/(1+x[1][0]),1e-6) for x in selected]);Rcw,inlier=robust_rotation(qn,mn,weight);use=inlier if inlier.sum()>=4 else np.ones(len(inlier),bool);x,plane_objective=solve(np.c_[mn[use],qd[use]],md[use],C0,weight[use]);assignment=float(np.average([z[1][0] for z in selected],weights=weight));outlier_penalty=float(1-weight[use].sum()/weight.sum());candidate_solutions.append((assignment+plane_objective+outlier_penalty,rank_pose,x,Rcw,int(use.sum()),assignment,plane_objective,outlier_penalty))
  result={'image_id':image_id,'usable':False}
  if candidate_solutions:
   objective,rank_pose,x,Rcw,count,assignment,plane_objective,outlier_penalty=min(candidate_solutions,key=lambda z:(z[0],z[1]));
   with np.load(a.contributors/name,allow_pickle=False) as d:gt=np.asarray(d['pose_w2c'],np.float64)
   Cgt=-gt[:3,:3].T@gt[:3,3];result.update(usable=True,selected_radio_rank=rank_pose+1,inlier_count=count,translation_error_m=float(np.linalg.norm(x[:3]-Cgt)),rotation_error_deg=float(Rotation.from_matrix(Rcw.T@gt[:3,:3].T).magnitude()*180/np.pi),scale=float(x[3]),objective=objective,assignment_objective=assignment,plane_objective=plane_objective,outlier_penalty=outlier_penalty)
  rows.append(result)
 usable=[x for x in rows if x['usable']];t=np.asarray([x['translation_error_m'] for x in usable]);r=np.asarray([x['rotation_error_deg'] for x in usable]);report={'artifact_type':'goal_maplet_plane_top5_radio_basin_robust_consistency_pose_v2','query_count':len(rows),'usable_count':len(usable),'uses_gt_in_scoring_or_selection':False,'topk':TOPK,'rank_cost':RANK_COST,'normal_scale_degrees':NORMAL_SCALE_DEG,'rotation_ransac_threshold_degrees':10.,'median_translation_m':float(np.median(t)) if len(t) else None,'median_rotation_deg':float(np.median(r)) if len(r) else None,'recall_2m45':float(np.mean((t<=2)&(r<=45))) if len(t) else 0.,'recall_1m10':float(np.mean((t<=1)&(r<=10))) if len(t) else 0.,'production_eligible':False,'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
