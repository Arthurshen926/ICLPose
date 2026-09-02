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
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions,extract_query_plane_regions
from feature_extract.tools.vfm.evaluate_goal_maplet_moge3_planes_against_rendered_map import rotation_from_normals

ALPHAS=(.25,.04,.01)
SCALE_PRIOR=.04
def regularized(A,b,center,alpha):
 augmented=np.r_[A,np.c_[np.sqrt(alpha)*np.eye(3),np.zeros(3)],np.array([[0.,0.,0.,np.sqrt(SCALE_PRIOR)]])]
 target=np.r_[b,np.sqrt(alpha)*center,np.sqrt(SCALE_PRIOR)]
 x=np.linalg.lstsq(augmented,target,rcond=None)[0]
 residual=A@x-b;objective=float(np.mean(residual**2)+alpha*np.sum((x[:3]-center)**2)+SCALE_PRIOR*(x[3]-1)**2)
 return x,objective
def finite_member_world_boundaries(physical,plane,row,*,boundary_samples=20):
 lo,hi=map(int,plane.member_offsets[row:row+2]);members=np.asarray(plane.member_primitive_rows[lo:hi],np.int64)
 theta=np.arange(int(boundary_samples),dtype=np.float64)*(2*np.pi/int(boundary_samples));cosine=np.cos(theta);sine=np.sin(theta)
 world=(
  physical.primitive_centers[members,None,:]
  +cosine[None,:,None]*physical.primitive_scale1[members,None,None]*physical.primitive_tangent1[members,None,:]
  +sine[None,:,None]*physical.primitive_scale2[members,None,None]*physical.primitive_tangent2[members,None,:]
 )
 return world

def projected_finite_member_mask(
 physical,plane,row,image_shape,rotation_w2c,center,K,*,boundary_samples=20,world_boundaries=None,
):
 """Rasterize only the exact finite 2DGS members of one fused plane.

 ``boundary_uv`` is deliberately not read: it is a convex visualization
 summary and can bridge arbitrary holes and disconnected support.  Each member
 is instead rendered as its finite tangent ellipse.  This is still a coarse
 visibility diagnostic (there is no cross-plane z-buffer), but it cannot turn
 unobserved gaps into a filled facade.
 """
 mask=np.zeros(tuple(map(int,image_shape)),np.uint8)
 world=finite_member_world_boundaries(physical,plane,row,boundary_samples=boundary_samples) if world_boundaries is None else np.asarray(world_boundaries,np.float64)
 if world.shape[0]==0:return mask.astype(bool)
 camera=(world-center)@rotation_w2c.T
 visible=np.all(camera[:,:,2]>.1,axis=1)
 projected=np.empty((world.shape[0],world.shape[1],2),np.float64)
 projected[:,:,0]=K[0,0]*camera[:,:,0]/np.maximum(camera[:,:,2],1e-12)+K[0,2]-.5
 projected[:,:,1]=K[1,1]*camera[:,:,1]/np.maximum(camera[:,:,2],1e-12)+K[1,2]-.5
 polygons=np.rint(projected[visible]).astype(np.int32)
 if polygons.size:cv2.fillPoly(mask,list(polygons),1)
 return mask.astype(bool)

def projected_finite_member_dice(physical,plane,row,query_mask,rotation_w2c,center,K,*,world_boundaries=None):
 predicted=projected_finite_member_mask(physical,plane,row,query_mask.shape,rotation_w2c,center,K,world_boundaries=world_boundaries)
 intersection=np.sum(predicted&query_mask);return float(2*intersection/max(int(predicted.sum()+query_mask.sum()),1))

def load_query_planes(path):
 with np.load(path,allow_pickle=False) as data:
  if 'metadata_json' in data.files and 'labels' in data.files:
   return QueryPlaneRegions.load_npz(path)[0]
  return extract_query_plane_regions(data['points_camera'],data['normal_camera'],data['valid'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--direct_candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 physical=GoalMapletPhysicalMap.load_npz(a.physical_map);plane=GeometryNativePlanarMap.load_npz(a.planar_map,primitive_count=len(physical.primitive_ids));owner=np.full(len(physical.primitive_ids),-1,np.int64)
 for row in range(len(plane.plane_ids)):lo,hi=map(int,plane.member_offsets[row:row+2]);owner[plane.member_primitive_rows[lo:hi]]=row
 row_by_id=np.full(int(physical.primitive_ids.max())+1,-1,np.int64);row_by_id[physical.primitive_ids]=np.arange(len(physical.primitive_ids))
 with np.load(a.direct_candidates,allow_pickle=False) as d:image_ids=np.asarray(d['image_ids']);poses=np.asarray(d['candidate_poses_w2c'])[:,1:];candidate_error=np.asarray(d['translation_m'])[:,1:]
 results=[]
 for qi,image_id in enumerate(image_ids.tolist()):
  name=image_id.replace('/','__')+'.npz'
  q=load_query_planes(a.moge_dir/name)
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
   support_world={x[5]:finite_member_world_boundaries(physical,plane,x[5]) for x in values};polygon_scores=[]
   for center in centers:
    score=sum(projected_finite_member_dice(physical,plane,x[5],x[6],Rwc,center,K,world_boundaries=support_world[x[5]])*x[0][0] for x in values)/max(sum(x[0][0] for x in values),1);polygon_scores.append(score)
   polygon_selected=int(np.argmax(polygon_scores));row['finite_member_selected_rank']=polygon_selected+1;row['finite_member_score']=float(polygon_scores[polygon_selected]);row['finite_member_selected_radio_center_error_m']=float(candidate_error[qi,polygon_selected])
   for alpha in ALPHAS:
    candidates=[regularized(A,md,c,alpha) for c in centers];selected=int(np.argmin([x[1] for x in candidates]));x,objective=candidates[selected];row[f'a{alpha:g}']={'selected_rank':selected+1,'translation_error_m':float(np.linalg.norm(x[:3]-gtcenter)),'scale':float(x[3]),'objective':objective,'selected_radio_center_error_m':float(candidate_error[qi,selected])}
    x,objective=regularized(A,md,centers[polygon_selected],alpha);row[f'finite_member_a{alpha:g}']={'selected_rank':polygon_selected+1,'translation_error_m':float(np.linalg.norm(x[:3]-gtcenter)),'scale':float(x[3]),'objective':objective,'selected_radio_center_error_m':float(candidate_error[qi,polygon_selected])}
  results.append(row)
 usable=[x for x in results if x['usable']];summary={}
 for alpha in ALPHAS:
  for prefix in ('','finite_member_'):
   key=f'{prefix}a{alpha:g}';t=np.asarray([x[key]['translation_error_m'] for x in usable]);r=np.asarray([x['rotation_error_deg'] for x in usable]);summary[key]={'median_translation_m':float(np.median(t)),'p90_translation_m':float(np.quantile(t,.9)),'recall_2m45':float(np.mean((t<=2)&(r<=45))),'recall_1m10':float(np.mean((t<=1)&(r<=10))),'median_selected_radio_center_error_m':float(np.median([x[key]['selected_radio_center_error_m'] for x in usable]))}
 report={'artifact_type':'goal_maplet_radio_basin_moge3_plane_hybrid_oracle_v2_exact_finite_support','query_count':len(results),'usable_count':len(usable),'candidate_zero_gt_anchor_removed':True,'radio_candidates_pose_free':True,'plane_correspondence_and_sign_oracle':True,'convex_boundary_uv_consumed':False,'finite_support':'exact member primitive ellipses; no cross-plane z-buffer','alphas_preregistered':list(ALPHAS),'scale_prior':SCALE_PRIOR,'production_eligible':False,'summary':summary,'rows':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
