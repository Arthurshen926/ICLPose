import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import PrimitiveSurfaceTable
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import RenderedPlaneObservations, extract_rendered_plane_observations
def test_rendered_plane_observation_contract_shapes():
 o=RenderedPlaneObservations(labels=np.full((2,3),-1,np.int32),normals_world=np.zeros((0,3)),offsets_world=np.zeros(0),pixel_counts=np.zeros(0,np.int64),member_offsets=np.array([0]),member_primitive_rows=np.zeros(0,np.int64),residual_rms_m=np.zeros(0),residual_p95_m=np.zeros(0),point_sum_world=np.zeros((0,3)),point_second_moment_world=np.zeros((0,3,3)))
 assert o.labels.shape==(2,3) and o.member_offsets.tolist()==[0]

def test_sequential_ransac_recovers_one_complete_rendered_plane(tmp_path):
 h,w=20,30; yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');tile=(yy//5)*6+(xx//5);count=24
 centers=np.stack(((np.arange(count)%6-.5)*.8,(np.arange(count)//6-.5)*.8,np.full(count,5.)),axis=1)
 table=PrimitiveSurfaceTable(primitive_ids=np.arange(count),centers=centers,tangent1=np.tile([1.,0.,0.],(count,1)),tangent2=np.tile([0.,1.,0.],(count,1)),normals=np.tile([0.,0.,1.],(count,1)),scale1=np.full(count,.45),scale2=np.full(count,.45),opacity=np.ones(count)).validated()
 ids=np.full((h,w,4),-1,np.int32);ids[:,:,0]=tile;depth=np.full((h,w),5.,np.float32)
 path=tmp_path/'render.npz';np.savez(path,topk_ids=ids,topk_weights=np.ones((h,w,4),np.float16),dominant_depth=depth,pose_w2c=np.eye(4),camera_model_id=np.asarray(2,np.int32),camera_width=np.asarray(w,np.int32),camera_height=np.asarray(h,np.int32),camera_params=np.asarray([30.,15.,10.,0.]))
 result=extract_rendered_plane_observations(table,path,minimum_pixels=50,minimum_primitives=6,maximum_hypotheses=32)
 assert result.normals_world.shape==(1,3)
 assert result.pixel_counts[0] >= (h-2)*(w-2)*.9
 assert abs(result.offsets_world[0]-5.)<1e-8

def test_pinhole_fx_fy_camera_is_supported(tmp_path):
 h,w=20,30; yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');tile=(yy//5)*6+(xx//5);count=24
 centers=np.stack(((np.arange(count)%6-.5)*.8,(np.arange(count)//6-.5)*.8,np.full(count,5.)),axis=1)
 table=PrimitiveSurfaceTable(primitive_ids=np.arange(count),centers=centers,tangent1=np.tile([1.,0.,0.],(count,1)),tangent2=np.tile([0.,1.,0.],(count,1)),normals=np.tile([0.,0.,1.],(count,1)),scale1=np.full(count,.45),scale2=np.full(count,.45),opacity=np.ones(count)).validated()
 ids=np.full((h,w,4),-1,np.int32);ids[:,:,0]=tile
 path=tmp_path/'pinhole.npz';np.savez(path,topk_ids=ids,dominant_depth=np.full((h,w),5.,np.float32),pose_w2c=np.eye(4),camera_model_id=np.asarray(1,np.int32),camera_width=np.asarray(w,np.int32),camera_height=np.asarray(h,np.int32),camera_params=np.asarray([30.,30.,15.,10.]))
 result=extract_rendered_plane_observations(table,path,minimum_pixels=50,minimum_primitives=6,maximum_hypotheses=32)
 assert result.normals_world.shape==(1,3)
 assert abs(result.offsets_world[0]-5.)<1e-8
