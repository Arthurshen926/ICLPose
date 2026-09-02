import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.large_plane_regions import (
    extract_large_plane_regions,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import (
    QueryPlaneRegions,
    SparseOcclusionCarrierConfig,
    merge_sparse_foreground_occluded_regions,
)


def test_large_plane_unites_support_across_a_hole_without_filling_it():
    yy, xx = np.mgrid[:32, :48]
    points = np.stack((xx * 0.05, yy * 0.05, np.full_like(xx, 4.0)), axis=2).astype(np.float64)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    valid = np.ones((32, 48), bool)
    valid[:, 21:27] = False
    planes = extract_large_plane_regions(points, normals, valid, minimum_pixels=100)
    assert len(planes.normals) == 1
    assert planes.component_counts.tolist() == [2]
    assert np.all(planes.labels[:, 21:27] == -1)
    assert int(planes.pixel_counts[0]) == int(valid.sum())


def test_large_plane_keeps_parallel_offsets_separate():
    yy, xx = np.mgrid[:30, :40]
    depth = np.where(xx < 20, 3.0, 4.0)
    points = np.stack((xx * 0.05, yy * 0.05, depth), axis=2)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    planes = extract_large_plane_regions(points, normals, np.ones((30, 40), bool), minimum_pixels=200)
    assert len(planes.normals) == 2
    assert sorted(np.round(np.abs(planes.offsets), 6).tolist()) == [3.0, 4.0]


def test_large_plane_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="incompatible"):
        extract_large_plane_regions(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), np.zeros((2, 3), bool))


def _split_plane(*,gap=8,foreground=True,offset_right=0.0):
    height,width=48,80
    yy,xx=np.mgrid[:height,:width]
    depth=np.full((height,width),6.0,np.float64)
    if foreground:depth[:,36:36+gap]=4.0
    depth[:,36+gap:]+=offset_right
    points=np.stack(((xx-39.5)*depth/80.0,(yy-23.5)*depth/80.0,depth),axis=2)
    normals=np.zeros_like(points);normals[...,2]=1.0
    valid=np.ones((height,width),bool)
    labels=np.full((height,width),-1,np.int32);labels[:,:36]=0;labels[:,36+gap:]=1
    base=QueryPlaneRegions(
        labels,np.asarray([[0,0,1],[0,0,1]],float),
        np.asarray([6.,6.+offset_right]),
        np.asarray([np.sum(labels==0),np.sum(labels==1)],np.int64),
        np.zeros(2),np.zeros(2),
    ).validated()
    return base,points,normals,valid


def test_sparse_occlusion_carrier_unites_visible_sides_without_filling_foreground():
    base,points,normals,valid=_split_plane(gap=8,foreground=True)
    assert len(base.normals_camera)==2
    carrier,audit=merge_sparse_foreground_occluded_regions(base,points,normals,valid)
    assert len(carrier.normals_camera)==1
    assert audit['merged_component_count']==1
    assert audit['hidden_pixel_count_added']==0
    assert np.array_equal(carrier.labels>=0,base.labels>=0)
    assert np.all(carrier.labels[:,36:44]==-1)


def test_sparse_occlusion_carrier_rejects_missing_foreground_depth_order():
    base,points,normals,valid=_split_plane(gap=8,foreground=False)
    carrier,audit=merge_sparse_foreground_occluded_regions(base,points,normals,valid)
    assert len(carrier.normals_camera)==2
    assert audit['merged_component_count']==0


def test_sparse_occlusion_carrier_rejects_wide_or_offset_occlusion():
    config=SparseOcclusionCarrierConfig(maximum_gap_pixels=12)
    base,points,normals,valid=_split_plane(gap=18,foreground=True)
    carrier,_=merge_sparse_foreground_occluded_regions(base,points,normals,valid,config=config)
    assert len(carrier.normals_camera)==2
    base,points,normals,valid=_split_plane(gap=8,foreground=True,offset_right=1.0)
    carrier,_=merge_sparse_foreground_occluded_regions(base,points,normals,valid,config=config)
    assert len(carrier.normals_camera)==2
