"""Fuse mapping-view plane masks into finite world-space plane instances."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from .geometry_native_planar_map import PrimitiveSurfaceTable,GeometryNativePlanarMap,SCHEMA,_frame
from .rendered_view_planes import RenderedPlaneObservations

def _fit_stats(count:int,total:np.ndarray,second:np.ndarray,reference:np.ndarray):
 center=total/count;covariance=second/count-np.outer(center,center);covariance=(covariance+covariance.T)*.5
 value,vector=np.linalg.eigh(covariance);normal=vector[:,0]
 if normal@reference<0:normal=-normal
 return center,normal,float(np.sqrt(max(value[0],0.0)))

def fuse_rendered_plane_observations(
 table:PrimitiveSurfaceTable,paths:list[Path],*,normal_degrees:float=8.,offset_m:float=.10,
 maximum_aggregate_rms_m:float=.07,minimum_members:int=6,minimum_views:int=1,
) -> tuple[GeometryNativePlanarMap,dict[str,np.ndarray]]:
 """Stable online fusion using shared primitive support plus aggregate refits.

 View masks are the connectivity authority.  Shared 2DGS primitive rows only
 propose cross-view associations; normal, offset, and aggregate rendered-point
 RMS must all pass before a view observation enters an instance.
 """
 table=table.validated();cosine_threshold=float(np.cos(np.deg2rad(normal_degrees)))
 groups=[];primitive_groups:dict[int,set[int]]={};global_observation=0
 for path in sorted(map(Path,paths)):
  observation,_=RenderedPlaneObservations.load_npz(path,table.primitive_ids.size)
  for row in range(observation.normals_world.shape[0]):
   lo,hi=map(int,observation.member_offsets[row:row+2]);members=observation.member_primitive_rows[lo:hi]
   candidate_count={}
   for primitive in members.tolist():
    for group in primitive_groups.get(primitive,()):candidate_count[group]=candidate_count.get(group,0)+1
   compatible=[]
   for group,overlap in candidate_count.items():
    state=groups[group];sign=1. if state['normal']@observation.normals_world[row]>=0 else -1.
    if abs(float(state['normal']@observation.normals_world[row]))<cosine_threshold:continue
    if abs(float(state['offset']-sign*observation.offsets_world[row]))>offset_m:continue
    count=state['count']+int(observation.pixel_counts[row]);total=state['sum']+observation.point_sum_world[row];second=state['second']+observation.point_second_moment_world[row]
    _,_,aggregate_rms=_fit_stats(count,total,second,state['normal'])
    if aggregate_rms<=maximum_aggregate_rms_m:compatible.append((overlap,-group,group,count,total,second,aggregate_rms))
   if compatible:
    *_,group,count,total,second,aggregate_rms=max(compatible);state=groups[group]
    center,normal,_=_fit_stats(count,total,second,state['normal']);state.update(count=count,sum=total,second=second,normal=normal,offset=float(normal@center),rms=aggregate_rms)
    state['members'].update(members.tolist());state['observations'].append(global_observation);state['p95']=max(state['p95'],float(observation.residual_p95_m[row]));state['views']+=1
   else:
    group=len(groups);normal=observation.normals_world[row].copy();center=observation.point_sum_world[row]/observation.pixel_counts[row]
    groups.append({'count':int(observation.pixel_counts[row]),'sum':observation.point_sum_world[row].copy(),'second':observation.point_second_moment_world[row].copy(),'normal':normal,'offset':float(normal@center),'rms':float(observation.residual_rms_m[row]),'p95':float(observation.residual_p95_m[row]),'members':set(members.tolist()),'observations':[global_observation],'views':1})
   for primitive in members.tolist():primitive_groups.setdefault(primitive,set()).add(group)
   global_observation+=1
 # Largest, most repeatedly observed instances own shared primitives first.
 order=sorted(range(len(groups)),key=lambda x:(-groups[x]['views'],-groups[x]['count'],x));claimed=set();accepted=[]
 for group in order:
  rows=np.asarray(sorted(groups[group]['members']-claimed),np.int64)
  if groups[group]['views']<minimum_views or rows.size<minimum_members:continue
  claimed.update(rows.tolist());accepted.append((group,rows))
 area=np.pi*table.scale1*table.scale2*table.opacity;signs=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],np.float64)
 normals=[];offsets=[];centers=[];frames=[];boundaries=[];boundary_area=[];members_all=[];support=[];rms=[];p95=[];cosine=[];observation_rows=[];observation_offsets=[0]
 for group,rows in accepted:
  state=groups[group];center,normal,aggregate_rms=_fit_stats(state['count'],state['sum'],state['second'],state['normal']);frame=_frame(normal)
  corners=(table.centers[rows,None,:]+signs[None,:,:1]*table.scale1[rows,None,None]*table.tangent1[rows,None,:]+signs[None,:,1:]*table.scale2[rows,None,None]*table.tangent2[rows,None,:]).reshape(-1,3)
  uv=(corners-center)@frame[:2].T
  try:hull=ConvexHull(uv);boundary=uv[hull.vertices];hull_area=float(hull.volume)
  except QhullError:boundary=uv[np.unique(uv,axis=0,return_index=True)[1]];hull_area=0.
  normals.append(normal);offsets.append(float(normal@center));centers.append(center);frames.append(frame);boundaries.append(boundary);boundary_area.append(hull_area);members_all.append(rows);support.append(float(area[rows].sum()));rms.append(aggregate_rms);p95.append(state['p95']);cosine.append(float(np.quantile(np.abs(table.normals[rows]@normal),.10)));observation_rows.extend(state['observations']);observation_offsets.append(len(observation_rows))
 boundary_offsets=np.r_[0,np.cumsum([x.shape[0] for x in boundaries])].astype(np.int64);member_offsets=np.r_[0,np.cumsum([x.size for x in members_all])].astype(np.int64);count=len(accepted)
 arrays=dict(plane_ids=np.arange(count,dtype=np.int64),normals_world=np.asarray(normals).reshape(-1,3),offsets_world=np.asarray(offsets),centers_world=np.asarray(centers).reshape(-1,3),frames_world=np.asarray(frames).reshape(-1,3,3),boundary_offsets=boundary_offsets,boundary_uv=np.concatenate(boundaries) if boundaries else np.zeros((0,2)),boundary_area_m2=np.asarray(boundary_area),member_offsets=member_offsets,member_primitive_rows=np.concatenate(members_all) if members_all else np.zeros(0,np.int64),member_counts=np.diff(member_offsets),support_area_m2=np.asarray(support),residual_rms_m=np.asarray(rms),residual_p95_m=np.asarray(p95),normal_cosine_p10=np.asarray(cosine))
 metadata={'artifact_type':SCHEMA,'representation':'finite_planes_from_2dgs_rendered_depth_sequential_ransac_cross_view_fusion','uses_parent_child_partition':False,'uses_voxel_identity_or_boundary':False,'uses_query_pose_or_ground_truth':False,'uses_mapping_rgb':False,'primitive_count':int(table.primitive_ids.size),'plane_count':count,'assigned_primitive_count':int(arrays['member_primitive_rows'].size),'observation_count':global_observation,'configuration':{'normal_degrees':normal_degrees,'offset_m':offset_m,'maximum_aggregate_rms_m':maximum_aggregate_rms_m,'minimum_members':minimum_members,'minimum_views':minimum_views},'boundary':'convex_summary_plus_exact_member_primitive_inventory'}
 lineage={'plane_observation_offsets':np.asarray(observation_offsets,np.int64),'plane_observation_rows':np.asarray(observation_rows,np.int64)}
 return GeometryNativePlanarMap(metadata=metadata,**arrays).validated(table.primitive_ids.size),lineage

__all__=['fuse_rendered_plane_observations']
