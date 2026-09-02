"""Oracle-correspondence gate for MoGe-3 planes and a rendered 2DGS plane map."""
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import extract_query_plane_regions
def rotation_from_normals(query,mapn):
 u,_,vt=np.linalg.svd(mapn.T@query);r=u@vt
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
 return r
def robust_translation_scale(A,b):
 combinations=list(itertools.combinations(range(len(b)),4));indices=np.linspace(0,len(combinations)-1,min(1024,len(combinations)),dtype=int);best=None
 for index in indices:
  subset=combinations[index]
  if np.linalg.matrix_rank(A[list(subset)])<4:continue
  x=np.linalg.lstsq(A[list(subset)],b[list(subset)],rcond=None)[0]
  if not .1<=x[3]<=10:continue
  residual=np.abs(A@x-b);key=(int(np.sum(residual<=.5)),-float(np.median(residual)),-float(np.quantile(residual,.8)))
  if best is None or key>best[0]:best=(key,x,residual)
 if best is None:return np.linalg.lstsq(A,b,rcond=None)[0],np.ones(len(b),bool)
 inlier=best[2]<=.5
 if np.sum(inlier)>=4 and np.linalg.matrix_rank(A[inlier])==4:return np.linalg.lstsq(A[inlier],b[inlier],rcond=None)[0],inlier
 return best[1],inlier
def main():
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 physical=GoalMapletPhysicalMap.load_npz(a.physical_map);plane=GeometryNativePlanarMap.load_npz(a.planar_map,primitive_count=physical.primitive_ids.size);owner=np.full(physical.primitive_ids.size,-1,np.int64)
 for row in range(len(plane.plane_ids)):lo,hi=map(int,plane.member_offsets[row:row+2]);owner[plane.member_primitive_rows[lo:hi]]=row
 row_by_id=np.full(int(physical.primitive_ids.max())+1,-1,np.int64);row_by_id[physical.primitive_ids]=np.arange(len(physical.primitive_ids));rows=[];paths=sorted(a.moge_dir.glob('*.npz'));paths=paths[:a.limit or None]
 for path in paths:
  with np.load(path,allow_pickle=False) as d:q=extract_query_plane_regions(d['points_camera'],d['normal_camera'],d['valid'])
  source=a.contributors/path.name
  with np.load(source,allow_pickle=False) as d:ids=np.asarray(d['topk_ids'][:,:,0],np.int64);gt=np.asarray(d['pose_w2c'],np.float64)
  primitive=np.full(ids.shape,-1,np.int64);v=(ids>=0)&(ids<row_by_id.size);primitive[v]=row_by_id[ids[v]];maplabel=np.full(ids.shape,-1,np.int64);v=primitive>=0;maplabel[v]=owner[primitive[v]]
  matches={}
  for region in range(len(q.normals_camera)):
   values=maplabel[q.labels==region];values=values[values>=0]
   if values.size<40:continue
   count=np.bincount(values,minlength=len(plane.plane_ids));match=int(count.argmax());fraction=float(count[match]/values.size)
   full_fraction=float(count[match]/np.sum(q.labels==region))
   if fraction<.5 or full_fraction<.25:continue
   nq=q.normals_camera[region].copy();nm=plane.normals_world[match].copy();# GT only resolves unoriented plane signs in this oracle.
   if nm@(gt[:3,:3].T@nq)<0:nm=-nm
   key=(int(count[match]),fraction,full_fraction);candidate=(key,nq,q.offsets_camera[region],nm,float(nm@plane.centers_world[match]),fraction)
   if match not in matches or key>matches[match][0]:matches[match]=candidate
  values=list(matches.values());result={'image':path.name,'query_plane_count':len(q.normals_camera),'unique_correspondence_count':len(values),'usable':False}
  if len(values)>=4:
   qn=np.asarray([x[1] for x in values]);qd=np.asarray([x[2] for x in values]);mn=np.asarray([x[3] for x in values]);md=np.asarray([x[4] for x in values]);purity=np.asarray([x[5] for x in values]);r_c2w=rotation_from_normals(qn,mn);A=np.c_[mn,qd];x,inlier=robust_translation_scale(A,md);center=x[:3];scale=float(x[3]);r_w2c=r_c2w.T;gtcenter=-gt[:3,:3].T@gt[:3,3];translation=float(np.linalg.norm(center-gtcenter));fixed_center=np.linalg.lstsq(mn,md-qd,rcond=None)[0];fixed_translation=float(np.linalg.norm(fixed_center-gtcenter));rotation=float(Rotation.from_matrix(r_w2c@gt[:3,:3].T).magnitude()*180/np.pi)
   valid_offset=np.abs(qd)>1e-6;implied=(md[valid_offset]-mn[valid_offset]@gtcenter)/qd[valid_offset];positive=implied[np.isfinite(implied)&(implied>0)]
   if positive.size:
    implied_median=float(np.median(positive));implied_p10=float(np.quantile(positive,.1));implied_p90=float(np.quantile(positive,.9));implied_spread=float((implied_p90-implied_p10)/max(abs(implied_median),1e-9));gt_scale_residual=np.abs(mn@gtcenter+implied_median*qd-md)
   else:
    implied_median=implied_p10=implied_p90=implied_spread=None;gt_scale_residual=np.full_like(md,np.nan)
   result.update(usable=True,translation_error_m=translation,fixed_metric_scale_translation_error_m=fixed_translation,rotation_error_deg=rotation,estimated_scale=scale,median_correspondence_purity=float(np.median(purity)),condition=float(np.linalg.cond(A)),normal_condition=float(np.linalg.cond(mn)),normal_rank=int(np.linalg.matrix_rank(mn)),robust_inlier_count=int(np.sum(inlier)),gt_center_implied_positive_scale_count=int(positive.size),gt_center_implied_scale_median=implied_median,gt_center_implied_scale_p10=implied_p10,gt_center_implied_scale_p90=implied_p90,gt_center_implied_scale_relative_p10_p90_span=implied_spread,gt_center_common_scale_plane_residual_median_m=float(np.nanmedian(gt_scale_residual)),gt_center_common_scale_plane_residual_p90_m=float(np.nanquantile(gt_scale_residual,.9)))
  rows.append(result)
 usable=[x for x in rows if x['usable']];report={'artifact_type':'moge3_sequential_planes_rendered_map_oracle_correspondence_scale_consistency_v2','query_count':len(rows),'usable_count':len(usable),'uses_gt_for_correspondence_and_normal_sign':True,'uses_gt_center_only_for_scale_consistency_diagnostic':True,'production_eligible':False,'median_translation_m':float(np.median([x['translation_error_m'] for x in usable])) if usable else None,'median_fixed_metric_scale_translation_m':float(np.median([x['fixed_metric_scale_translation_error_m'] for x in usable])) if usable else None,'median_rotation_deg':float(np.median([x['rotation_error_deg'] for x in usable])) if usable else None,'median_gt_center_implied_scale_relative_span':float(np.median([x['gt_center_implied_scale_relative_p10_p90_span'] for x in usable if x['gt_center_implied_scale_relative_p10_p90_span'] is not None])) if usable else None,'median_gt_center_common_scale_plane_residual_m':float(np.median([x['gt_center_common_scale_plane_residual_median_m'] for x in usable])) if usable else None,'recall_2m45':float(np.mean([x['translation_error_m']<=2 and x['rotation_error_deg']<=45 for x in usable])) if usable else 0.,'recall_1m10':float(np.mean([x['translation_error_m']<=1 and x['rotation_error_deg']<=10 for x in usable])) if usable else 0.,'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
