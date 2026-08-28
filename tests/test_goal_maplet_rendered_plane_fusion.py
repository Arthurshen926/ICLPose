import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import PrimitiveSurfaceTable
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import RenderedPlaneObservations
from feature_extract.vfm.localization_goal_maplet.rendered_plane_fusion import fuse_rendered_plane_observations
def test_fuses_two_views_by_shared_primitive(tmp_path):
 t=PrimitiveSurfaceTable(np.arange(8),np.c_[np.arange(8),np.zeros(8),np.full(8,5.)],np.tile([1.,0,0],(8,1)),np.tile([0,1.,0],(8,1)),np.tile([0,0,1.],(8,1)),np.full(8,.1),np.full(8,.1),np.ones(8)).validated()
 for i,rows in enumerate((np.arange(7),np.arange(1,8))):
  points=t.centers[rows];o=RenderedPlaneObservations(np.zeros((2,4),np.int32),np.array([[0.,0,1.]]),np.array([5.]),np.array([len(rows)]),np.array([0,len(rows)]),rows,np.array([0.]),np.array([0.]),points.sum(0)[None],(points.T@points)[None]).validated(8);o.save_npz(tmp_path/f'{i}.npz',{'view':i})
 m,lineage=fuse_rendered_plane_observations(t,list(tmp_path.glob('*.npz')))
 assert m.plane_ids.size==1 and m.member_primitive_rows.size==8
 assert lineage['plane_observation_rows'].size==2
