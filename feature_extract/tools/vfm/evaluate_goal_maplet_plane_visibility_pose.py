"""RADIO + true mapping-view plane-mask consistency + robust direct R,t,s."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.tools.vfm.evaluate_goal_maplet_moge3_planes_against_rendered_map import rotation_from_normals
from feature_extract.tools.vfm.evaluate_goal_maplet_predicted_plane_pose import solve

TOPK=5;RANK_COST=.15;NORMAL_SCALE_DEG=15.;BETAS=(0.,.25,.5,1.)
def coarse_mask(counts):return np.asarray(counts,np.float64).reshape(9,4,16,4).sum((1,3))/256.
def dice(a,b):return float(2*np.minimum(a,b).sum()/max(a.sum()+b.sum(),1e-12))
def nearest_templates(atlas,plane_row,R,C,limit=4):
 lo,hi=map(int,atlas.plane_offsets[plane_row:plane_row+2])
 if lo==hi:return np.zeros(0,np.int64),0.
 Rm=atlas.poses_w2c[lo:hi,:3,:3];relative=np.einsum('ij,nkj->nik',R,Rm);trace=np.trace(relative,axis1=1,axis2=2);angle=np.arccos(np.clip((trace-1)*.5,-1,1));distance=np.linalg.norm(atlas.centers_world[lo:hi]-C,axis=1);cost=(angle/np.deg2rad(30.))**2+(distance/10.)**2
 order=np.lexsort((np.arange(hi-lo),cost));reliability=min(1.,(hi-lo)/4.)*float(np.exp(-.5*cost[order[0]]));return lo+order[:limit],reliability
def robust_rotation(qn,mn,weight):
 combinations=list(itertools.combinations(range(len(qn)),3));indices=np.linspace(0,len(combinations)-1,min(256,len(combinations)),dtype=int);best=None
 for index in indices:
  subset=np.asarray(combinations[index]);R=rotation_from_normals(qn[subset],mn[subset]);angle=np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i',qn@R.T,mn)),0,1)));inlier=angle<=10.;key=(float(weight[inlier].sum()),int(inlier.sum()),-float(np.median(angle)))
  if best is None or key>best[0]:best=(key,R,inlier)
 if best is None:return rotation_from_normals(qn,mn),np.ones(len(qn),bool)
 if best[2].sum()>=3:
  reference=qn[best[2]]@best[1].T;target=mn[best[2]].copy();target*=np.where(np.einsum('ij,ij->i',reference,target)>=0,1.,-1.)[:,None];return rotation_from_normals(qn[best[2]],target),best[2]
 return best[1],best[2]
def solve_query(q,records,poses,plane,atlas,beta):
 query_masks=[coarse_mask((q.labels==row).reshape(36,4,64,4).sum((1,3))) for row in range(len(q.normals_camera))];solutions=[]
 for rank_pose,pose in enumerate(poses):
  R0=pose[:3,:3];C0=-R0.T@pose[:3,3];template_cache={};chosen=[]
  for record in records:
   region=int(record['region']);nq=q.normals_camera[region];predicted=nq@R0;options=[]
   for rank,maprow in enumerate(record['top10'][:TOPK]):
    maprow=int(maprow);nm=plane.normals_world[maprow];angle=np.degrees(np.arccos(np.clip(abs(float(nm@predicted)),0,1)))
    if maprow not in template_cache:template_cache[maprow]=nearest_templates(atlas,maprow,R0,C0)
    indices,reliability=template_cache[maprow];visibility=max((dice(query_masks[region],coarse_mask(atlas.token_pixel_counts[x])) for x in indices),default=0.);evidence=reliability*visibility
    cost=(angle/NORMAL_SCALE_DEG)**2+RANK_COST*rank-beta*evidence;options.append((cost,rank,maprow,visibility,reliability))
   if options:chosen.append((min(options),record))
  unique={}
  for option,record in chosen:
   maprow=option[2];key=(-option[0],record['pixels'],-record['region'])
   if maprow not in unique or key>unique[maprow][0]:unique[maprow]=(key,option,record)
  selected=list(unique.values())
  if len(selected)<4:continue
  qn=np.asarray([q.normals_camera[x[2]['region']] for x in selected]);qd=np.asarray([q.offsets_camera[x[2]['region']] for x in selected]);maprows=np.asarray([x[1][2] for x in selected]);mn=plane.normals_world[maprows].copy();predicted=qn@R0;mn*=np.where(np.einsum('ij,ij->i',mn,predicted)>=0,1.,-1.)[:,None];md=np.einsum('ij,ij->i',mn,plane.centers_world[maprows]);weight=np.asarray([max(x[2]['pixels']/(1+x[1][0]),1e-6) for x in selected]);Rcw,inlier=robust_rotation(qn,mn,weight);use=inlier if inlier.sum()>=4 else np.ones(len(inlier),bool);x,plane_objective=solve(np.c_[mn[use],qd[use]],md[use],C0,weight[use]);assignment=float(np.average([z[1][0] for z in selected],weights=weight));outlier=float(1-weight[use].sum()/weight.sum());visibility=float(np.average([z[1][3] for z in selected],weights=weight));reliability=float(np.average([z[1][4] for z in selected],weights=weight));solutions.append((assignment+plane_objective+outlier,rank_pose,x,Rcw,int(use.sum()),assignment,plane_objective,outlier,visibility,reliability))
 return min(solutions,key=lambda z:(z[0],z[1])) if solutions else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--visibility_atlas',type=Path,required=True);p.add_argument('--query_plane_dir',type=Path,required=True);p.add_argument('--correspondence_report',type=Path,required=True);p.add_argument('--direct_candidates',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--betas',nargs='+',type=float,default=list(BETAS));a=p.parse_args();plane=GeometryNativePlanarMap.load_npz(a.planar_map);atlas,_=PlaneVisibilityAtlas.load_npz(a.visibility_atlas);matching=json.loads(a.correspondence_report.read_text());by_image={x['image']:x for x in matching['rows']}
 with np.load(a.direct_candidates,allow_pickle=False) as data:image_ids=np.asarray(data['image_ids']);poses=np.asarray(data['candidate_poses_w2c'])[:,1:]
 reports=[]
 for beta in a.betas:
  rows=[]
  for qi,image_id in enumerate(image_ids.tolist()):
   name=image_id.replace('/','__')+'.npz';q,_=QueryPlaneRegions.load_npz(a.query_plane_dir/name);solution=solve_query(q,by_image[name]['regions'],poses[qi],plane,atlas,beta);row={'image_id':image_id,'usable':solution is not None}
   if solution is not None:
    objective,rank_pose,x,Rcw,count,assignment,plane_objective,outlier,visibility,reliability=solution
    with np.load(a.contributors/name,allow_pickle=False) as data:gt=np.asarray(data['pose_w2c'],np.float64)
    Cgt=-gt[:3,:3].T@gt[:3,3];row.update(selected_radio_rank=rank_pose+1,inlier_count=count,translation_error_m=float(np.linalg.norm(x[:3]-Cgt)),rotation_error_deg=float(Rotation.from_matrix(Rcw.T@gt[:3,:3].T).magnitude()*180/np.pi),scale=float(x[3]),objective=objective,assignment_objective=assignment,plane_objective=plane_objective,outlier_penalty=outlier,selected_visibility_dice=visibility,selected_visibility_reliability=reliability)
   rows.append(row)
  usable=[x for x in rows if x['usable']];t=np.asarray([x['translation_error_m'] for x in usable]);r=np.asarray([x['rotation_error_deg'] for x in usable]);reports.append({'visibility_weight':beta,'usable_count':len(usable),'median_translation_m':float(np.median(t)) if len(t) else None,'median_rotation_deg':float(np.median(r)) if len(r) else None,'recall_2m45':float(np.mean((t<=2)&(r<=45))) if len(t) else 0.,'recall_1m10':float(np.mean((t<=1)&(r<=10))) if len(t) else 0.,'rows':rows})
 report={'artifact_type':'goal_maplet_true_visibility_plane_top5_pose_v1','query_count':len(image_ids),'uses_gt_in_scoring_or_selection':False,'visibility':'maximum 9x16 fractional-mask Dice among four nearest mapping observations; reward attenuated by min(1,n_views/4)*exp(-0.5*((angle/30deg)^2+(distance/10m)^2))','weights_preregistered_on_this_route':list(map(float,a.betas)),'topk':TOPK,'production_eligible':False,'results':reports};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in reports],indent=2))
if __name__=='__main__':main()
