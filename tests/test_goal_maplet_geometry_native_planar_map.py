import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    PrimitiveSurfaceTable, extract_geometry_native_planar_map,
)


def _table(groups):
    centers=[]; normals=[]
    for center, normal in groups:
        centers.append(center); normals.append(normal)
    centers=np.asarray(centers,np.float64); normals=np.asarray(normals,np.float64)
    tangent1=np.tile([1.,0.,0.],(len(centers),1))
    # Ensure tangents are valid for horizontal and vertical synthetic planes.
    for i,n in enumerate(normals):
        axis=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
        tangent1[i]=axis-n*np.dot(axis,n); tangent1[i]/=np.linalg.norm(tangent1[i])
    tangent2=np.cross(normals,tangent1)
    return PrimitiveSurfaceTable(
        primitive_ids=np.arange(len(centers)), centers=centers,
        tangent1=tangent1,tangent2=tangent2,normals=normals,
        scale1=np.full(len(centers),.3),scale2=np.full(len(centers),.3),
        opacity=np.ones(len(centers)),
    ).validated()


def test_complete_plane_crosses_any_implicit_voxel_boundary():
    rows=[]
    for x in np.arange(-2.4,2.5,.4):
        for y in np.arange(-1.2,1.3,.4):
            rows.append((np.array([x,y,0.]),np.array([0.,0.,1.])))
    result=extract_geometry_native_planar_map(_table(rows),minimum_members=5)
    assert result.plane_ids.size == 1
    assert result.member_counts[0] == len(rows)
    assert result.boundary_area_m2[0] > 10.0


def test_parallel_disconnected_surfaces_remain_distinct():
    rows=[]
    for z in (0.,2.):
        for x in np.arange(0.,1.7,.4):
            for y in np.arange(0.,1.7,.4):
                rows.append((np.array([x,y,z]),np.array([0.,0.,1.])))
    result=extract_geometry_native_planar_map(_table(rows),minimum_members=5)
    assert result.plane_ids.size == 2
    assert sorted(result.member_counts.tolist()) == [25,25]


def test_perpendicular_connected_surfaces_do_not_merge():
    rows=[]
    for x in np.arange(0.,1.7,.4):
        for y in np.arange(0.,1.7,.4):
            rows.append((np.array([x,y,0.]),np.array([0.,0.,1.])))
    for y in np.arange(0.,1.7,.4):
        for z in np.arange(.4,2.1,.4):
            rows.append((np.array([0.,y,z]),np.array([1.,0.,0.])))
    result=extract_geometry_native_planar_map(_table(rows),minimum_members=5)
    assert result.plane_ids.size == 2


def test_output_explicitly_rejects_voxel_identity_semantics(tmp_path):
    rows=[(np.array([x,y,0.]),np.array([0.,0.,1.])) for x in np.arange(0,1.7,.4) for y in np.arange(0,1.7,.4)]
    result=extract_geometry_native_planar_map(_table(rows),minimum_members=5)
    path=tmp_path/'plane.npz'; result.save_npz(path)
    loaded=type(result).load_npz(path,primitive_count=len(rows))
    assert loaded.metadata['uses_parent_child_partition'] is False
    assert loaded.metadata['uses_voxel_identity_or_boundary'] is False
