"""Test whether plane offsets can refine pose-free RADIO camera basins.

Plane correspondence/sign and final errors are label-side oracle quantities.
The 64 non-anchor RADIO poses are frozen pose-free candidates; candidate zero
from the direct dataset is explicitly removed before any hybrid computation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions
from feature_extract.tools.vfm.evaluate_goal_maplet_moge3_planes_against_rendered_map import rotation_from_normals

ALPHAS=(.25,.04,.01)
SCALE_PRIOR=.04
def regularized(A,b,center,alpha):
 augmented=np.r_[A,np.c_[np.sqrt(alpha)*np.eye(3),np.zeros(3)],np.array([[0.,0.,0.,np.sqrt(SCALE_PRIOR)]])]
 target=np.r_[b,np.sqrt(alpha)*center,np.sqrt(SCALE_PRIOR)]
 x=np.linalg.lstsq(augmented,target,rcond=None)[0]
 residual=A@x-b;objective=float(np.mean(residual**2)+alpha*np.sum((x[:3]-center)**2)+SCALE_PRIOR*(x[3]-1)**2)
 return x,objective
def projected_polygon_dice(plane,row,query_mask,Rwc,center,K):
 lo,hi=map(int,plane.boundary_offsets[row:row+2]);uv=plane.boundary_uv[lo:hi]
 if len(uv)<3:return 0.
 world=plane.centers_world[row]+uv[:,:1]*plane.frames_world[row,0]+uv[:,1:]*plane.frames_world[row,1];camera=(world-center)@Rwc.T
 if np.sum(camera[:,2]>.1)<3:return 0.
 camera=camera[camera[:,2]>.1];xy=np.c_[K[0,0]*camera[:,0]/camera[:,2]+K[0,2]-.5,K[1,1]*camera[:,1]/camera[:,2]+K[1,2]-.5];polygon=cv2.convexHull(np.rint(xy).astype(np.int32));mask=np.zeros(query_mask.shape,np.uint8);cv2.fillConvexPoly(mask,polygon,1);predicted=mask.astype(bool)
 intersection=np.sum(predicted&query_mask);return float(2*intersection/max(int(predicted.sum()+query_mask.sum()),1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--direct_candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 physical=GoalMapletPhysicalMap.load_npz(a.physical_map);plane=GeometryNativePlanarMap.load_npz(a.planar_map,primitive_count=len(physical.primitive_ids));owner=np.full(len(physical.primitive_ids),-1,np.int64)
 for row in range(len(plane.plane_ids)):lo,hi=map(int,plane.member_offsets[row:row+2]);owner[plane.member_primitive_rows[lo:hi]]=row
 row_by_id=np.full(int(physical.primitive_ids.max())+1,-1,np.int64);row_by_id[physical.primitive_ids]=np.arange(len(physical.primitive_ids))
 with np.load(a.direct_candidates,allow_pickle=False) as d:image_ids=np.asarray(d['image_ids']);poses=np.asarray(d['candidate_poses_w2c'])[:,1:];candidate_error=np.asarray(d['translation_m'])[:,1:]
 results=[]
 for qi,image_id in enumerate(image_ids.tolist()):
  name=image_id.replace('/','__')+'.npz'
  with np.load(a.moge_dir/name,allow_pickle=False) as d:q=extract_query_plane_regions(d['points_camera'],d['normal_camera'],d['valid'])
  with np.load(a.contributors/name,allow_pickle=False) as d:
   ids=np.asarray(d['topk_ids'][:,:,0],np.int64);gt=np.asarray(d['pose_w2c'],np.float64);params=np.asarray(d['camera_params'],np.float64);cw=int(d['camera_width']);ch=int(d['camera_height'])
  h,w=ids.shape;K=np.array([[params[0]*w/cw,0,params[1]*w/cw],[0,params[0]*h/ch,params[2]*h/ch],[0,0,1]],np.float64)
  primitive=np.full(ids.shape,-1,np.int64);valid=(ids>=0)&(ids<row_by_id.size);primitive[valid]=row_by_id[ids[valid]];maplabel=np.full(ids.shape,-1,np.int64);valid=primitive>=0;maplabel[valid]=owner[primitive[valid]];matches={}
  for region in range(len(q.normals_camera)):
   mask=q.labels==region;values=maplabel[mask];values=values[values>=0]
   if values.size<40:continue
   count=np.bincount(values,minlength=len(plane.plane_ids));match=int(count.argmax());purity=float(count[match]/values.size);full=float(count[match]/mask.sum())
   if purity<.5 or full<.25:continue
   nq=q.normals_camera[region].copy();nm=plane.normals_world[match].copy()
   if nm@(gt[:3,:3].T@nq)<0:nm=-nm
   key=(int(count[match]),purity,full);value=(key,nq,float(q.offsets_camera[region]),nm,float(nm@plane.centers_world[match]),match,mask.copy())
   if match not in matches or key>matches[match][0]:matches[match]=value
  row={'image_id':image_id,'correspondence_count':len(matches),'usable':False,'radio_oracle_min_translation_m':float(candidate_error[qi].min())}
  if len(matches)>=4:
   values=list(matches.values());qn=np.asarray([x[1] for x in values]);qd=np.asarray([x[2] for x in values]);mn=np.asarray([x[3] for x in values]);md=np.asarray([x[4] for x in values]);Rcw=rotation_from_normals(qn,mn);Rwc=Rcw.T;rotation_error=float(Rotation.from_matrix(Rwc@gt[:3,:3].T).magnitude()*180/np.pi);gtcenter=-gt[:3,:3].T@gt[:3,3];A=np.c_[mn,qd]
   centers=-np.einsum('nij,nj->ni',poses[qi,:,:3,:3].transpose(0,2,1),poses[qi,:,:3,3]);row.update(usable=True,rotation_error_deg=rotation_error)
   polygon_scores=[]
   for center in centers:
    score=sum(projected_polygon_dice(plane,x[5],x[6],Rwc,center,K)*x[0][0] for x in values)/max(sum(x[0][0] for x in values),1);polygon_scores.append(score)
   polygon_selected=int(np.argmax(polygon_scores));row['polygon_selected_rank']=polygon_selected+1;row['polygon_score']=float(polygon_scores[polygon_selected]);row['polygon_selected_radio_center_error_m']=float(candidate_error[qi,polygon_selected])
   for alpha in ALPHAS:
    candidates=[regularized(A,md,c,alpha) for c in centers];selected=int(np.argmin([x[1] for x in candidates]));x,objective=candidates[selected];row[f'a{alpha:g}']={'selected_rank':selected+1,'translation_error_m':float(np.linalg.norm(x[:3]-gtcenter)),'scale':float(x[3]),'objective':objective,'selected_radio_center_error_m':float(candidate_error[qi,selected])}
    x,objective=regularized(A,md,centers[polygon_selected],alpha);row[f'polygon_a{alpha:g}']={'selected_rank':polygon_selected+1,'translation_error_m':float(np.linalg.norm(x[:3]-gtcenter)),'scale':float(x[3]),'objective':objective,'selected_radio_center_error_m':float(candidate_error[qi,polygon_selected])}
  results.append(row)
 usable=[x for x in results if x['usable']];summary={}
 for alpha in ALPHAS:
  for prefix in ('','polygon_'):
   key=f'{prefix}a{alpha:g}';t=np.asarray([x[key]['translation_error_m'] for x in usable]);r=np.asarray([x['rotation_error_deg'] for x in usable]);summary[key]={'median_translation_m':float(np.median(t)),'p90_translation_m':float(np.quantile(t,.9)),'recall_2m45':float(np.mean((t<=2)&(r<=45))),'recall_1m10':float(np.mean((t<=1)&(r<=10))),'median_selected_radio_center_error_m':float(np.median([x[key]['selected_radio_center_error_m'] for x in usable]))}
 report={'artifact_type':'goal_maplet_radio_basin_moge3_plane_hybrid_oracle_v1','query_count':len(results),'usable_count':len(usable),'candidate_zero_gt_anchor_removed':True,'radio_candidates_pose_free':True,'plane_correspondence_and_sign_oracle':True,'alphas_preregistered':list(ALPHAS),'scale_prior':SCALE_PRIOR,'production_eligible':False,'summary':summary,'rows':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
