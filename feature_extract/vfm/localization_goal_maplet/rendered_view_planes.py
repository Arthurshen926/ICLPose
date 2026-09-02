"""Plane observations from a 2DGS depth/primitive-ID rendering.

The 2D image grid supplies the surface connectivity that an unordered set of
Gaussian surfels lacks.  No RGB or learned feature is used here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .geometry_native_planar_map import PrimitiveSurfaceTable, _fit_plane, _robust_group_quality
from .lineage import arrays_sha256, canonical_json_sha256

@dataclass(frozen=True)
class RenderedPlaneObservations:
    labels: np.ndarray
    normals_world: np.ndarray
    offsets_world: np.ndarray
    pixel_counts: np.ndarray
    member_offsets: np.ndarray
    member_primitive_rows: np.ndarray
    residual_rms_m: np.ndarray
    residual_p95_m: np.ndarray
    point_sum_world: np.ndarray
    point_second_moment_world: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in (
            'labels','normals_world','offsets_world','pixel_counts','member_offsets',
            'member_primitive_rows','residual_rms_m','residual_p95_m',
            'point_sum_world','point_second_moment_world')}

    def validated(self, primitive_count: int | None = None) -> 'RenderedPlaneObservations':
        n=int(np.asarray(self.normals_world).shape[0])
        if self.normals_world.shape!=(n,3) or self.offsets_world.shape!=(n,) or self.pixel_counts.shape!=(n,):
            raise ValueError('invalid rendered-plane leading shapes')
        if self.point_sum_world.shape!=(n,3) or self.point_second_moment_world.shape!=(n,3,3):
            raise ValueError('invalid rendered-plane sufficient statistics')
        if self.member_offsets.shape!=(n+1,) or self.member_offsets[0]!=0 or self.member_offsets[-1]!=self.member_primitive_rows.size or np.any(np.diff(self.member_offsets)<0):
            raise ValueError('invalid rendered-plane membership offsets')
        if self.residual_rms_m.shape!=(n,) or self.residual_p95_m.shape!=(n,):
            raise ValueError('invalid rendered-plane residual shapes')
        if not all(np.isfinite(x).all() for x in (self.normals_world,self.offsets_world,self.residual_rms_m,self.residual_p95_m,self.point_sum_world,self.point_second_moment_world)):
            raise ValueError('nonfinite rendered-plane observation')
        if np.any(self.pixel_counts<=0) or np.any(np.diff(self.member_offsets)<=0):
            raise ValueError('empty rendered-plane observation')
        if primitive_count is not None and np.any((self.member_primitive_rows<0)|(self.member_primitive_rows>=primitive_count)):
            raise ValueError('rendered-plane primitive row outside map')
        return self

    def save_npz(self, path: Path, metadata: dict[str, object]) -> dict[str, object]:
        arrays=self.arrays(); meta=dict(metadata)
        meta.update({'artifact_type':'goal_maplet_rendered_plane_observations_v1','arrays_sha256':arrays_sha256(arrays)})
        meta['content_sha256']=canonical_json_sha256(meta)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.temporary.npz')
        np.savez_compressed(temporary,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
        temporary.replace(path)
        return meta

    @classmethod
    def load_npz(cls,path:Path,primitive_count:int|None=None) -> tuple['RenderedPlaneObservations',dict[str,object]]:
        with np.load(Path(path),allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json'].item()))
            arrays={name:np.asarray(data[name]) for name in ('labels','normals_world','offsets_world','pixel_counts','member_offsets','member_primitive_rows','residual_rms_m','residual_p95_m','point_sum_world','point_second_moment_world')}
        if meta.get('artifact_type')!='goal_maplet_rendered_plane_observations_v1' or arrays_sha256(arrays)!=meta.get('arrays_sha256'):
            raise ValueError('rendered-plane observation lineage differs')
        return cls(**arrays).validated(primitive_count),meta

def extract_rendered_plane_observations(
    table: PrimitiveSurfaceTable, contributor_path: Path, *,
    normal_degrees: float=15., reciprocal_plane_distance_m: float=.06,
    minimum_pixels: int=80, minimum_primitives: int=6,
    maximum_rms_m: float=.05, maximum_p95_m: float=.10,
    maximum_planes: int=32, maximum_hypotheses: int=96,
) -> RenderedPlaneObservations:
    table=table.validated()
    with np.load(Path(contributor_path),allow_pickle=False) as d:
        ids=np.asarray(d['topk_ids'][:,:,0],np.int64);depth=np.asarray(d['dominant_depth'],np.float64)
        pose=np.asarray(d['pose_w2c'],np.float64);model=int(d['camera_model_id']);cw=int(d['camera_width']);ch=int(d['camera_height']);params=np.asarray(d['camera_params'],np.float64)
    if model == 1 and params.size >= 4:
        source_fx,source_fy,source_cx,source_cy=map(float,params[:4])
    elif model in (0,2,8) and params.size >= 3:
        source_fx=source_fy=float(params[0]);source_cx=float(params[1]);source_cy=float(params[2])
    else:raise ValueError('rendered plane extraction requires a supported centered COLMAP camera')
    h,w=ids.shape; maximum_id=int(np.max(table.primitive_ids));row_by_id=np.full(maximum_id+1,-1,np.int32);row_by_id[table.primitive_ids]=np.arange(table.primitive_ids.size,dtype=np.int32)
    valid=(ids>=0)&(ids<=maximum_id)&np.isfinite(depth)&(depth>0);rows=np.full(ids.shape,-1,np.int64);rows[valid]=row_by_id[ids[valid]];valid &= rows>=0
    fx=source_fx*w/cw;fy=source_fy*h/ch;cx=source_cx*w/cw;cy=source_cy*h/ch
    yy,xx=np.meshgrid(np.arange(h,dtype=np.float64)+.5,np.arange(w,dtype=np.float64)+.5,indexing='ij')
    camera=np.stack(((xx-cx)/fx*depth,(yy-cy)/fy*depth,depth),axis=-1)
    world=(camera-pose[:3,3])@pose[:3,:3]
    # Derive the observation normal from the rendered surface itself.  Raw
    # Gaussian local frames are noisy and may change abruptly inside a smooth
    # rendered facade; PlanaReLoc analogously fits the dense surface rather
    # than treating storage-element normals as region identities.
    normal=np.zeros((h,w,3),np.float64);normal_valid=np.zeros((h,w),bool)
    filtered_depth=cv2.bilateralFilter(depth.astype(np.float32),5,.25,3).astype(np.float64)
    filtered_camera=np.stack(((xx-cx)/fx*filtered_depth,(yy-cy)/fy*filtered_depth,filtered_depth),axis=-1)
    filtered_world=(filtered_camera-pose[:3,3])@pose[:3,:3]
    dx=filtered_world[1:-1,2:]-filtered_world[1:-1,:-2];dy=filtered_world[2:,1:-1]-filtered_world[:-2,1:-1]
    local=np.cross(dx,dy);length=np.linalg.norm(local,axis=2)
    central_valid=(valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&(length>1e-10))
    normal[1:-1,1:-1][central_valid]=local[central_valid]/length[central_valid,None]
    normal_valid[1:-1,1:-1]=central_valid
    valid &= normal_valid
    # Deterministic sequential RANSAC.  Each hypothesis selects all compatible
    # 3D points, but only the 2D connected component containing its seed can
    # become one primitive.  This keeps separate coplanar objects distinct.
    threshold=np.cos(np.deg2rad(max(float(normal_degrees),30.0)))
    remaining=valid.copy();labels=np.full((h,w),-1,np.int32);normals=[];offsets=[];counts=[];members=[];rms=[];p95=[];point_sum=[];point_second=[]
    flat_world=world.reshape(-1,3);flat_normal=normal.reshape(-1,3)
    for _plane in range(int(maximum_planes)):
        candidate=np.flatnonzero(remaining.reshape(-1))
        if candidate.size<minimum_pixels:break
        positions=np.linspace(0,candidate.size-1,min(int(maximum_hypotheses),candidate.size),dtype=np.int64)
        seeds=candidate[positions]
        best_mask=None;best_seed=-1;best_count=0
        for seed in seeds.tolist():
            seed_normal=flat_normal[seed];seed_point=flat_world[seed]
            distance=np.abs((flat_world-seed_point)@seed_normal).reshape(h,w)
            cosine=np.abs((flat_normal@seed_normal).reshape(h,w))
            inlier=remaining&(distance<=float(reciprocal_plane_distance_m))&(cosine>=threshold)
            count_component,component=cv2.connectedComponents(inlier.astype(np.uint8),connectivity=8)
            label=int(component.reshape(-1)[seed])
            if label==0:continue
            mask=component==label;size=int(mask.sum())
            if (size,-seed)>(best_count,-best_seed):best_mask,best_seed,best_count=mask,seed,size
        if best_mask is None or best_count<minimum_pixels:break
        mask=best_mask
        for _ in range(3):
            points=world[mask];center=np.mean(points,axis=0);covariance=(points-center).T@(points-center)/points.shape[0]
            _,vectors=np.linalg.eigh(covariance);plane_normal=vectors[:,0];reference=np.mean(normal[mask],axis=0)
            if float(plane_normal@reference)<0:plane_normal=-plane_normal
            distance=np.abs((flat_world-center)@plane_normal).reshape(h,w);cosine=np.abs((flat_normal@plane_normal).reshape(h,w))
            inlier=remaining&(distance<=float(reciprocal_plane_distance_m))&(cosine>=threshold)
            _,component=cv2.connectedComponents(inlier.astype(np.uint8),connectivity=8);component_label=int(component.reshape(-1)[best_seed])
            if component_label==0:break
            updated=component==component_label
            if np.array_equal(updated,mask):break
            mask=updated
        if int(mask.sum())<minimum_pixels:
            remaining.reshape(-1)[best_seed]=False;continue
        primitive_rows=np.unique(rows[mask])
        if primitive_rows.size<minimum_primitives:
            remaining[mask]=False;continue
        points=world[mask];center=np.mean(points,axis=0);covariance=(points-center).T@(points-center)/points.shape[0]
        _,vectors=np.linalg.eigh(covariance);plane_normal=vectors[:,0];reference=np.mean(normal[mask],axis=0)
        if float(plane_normal@reference)<0:plane_normal=-plane_normal
        residual=np.abs((points-center)@plane_normal);quality=(float(np.sqrt(np.mean(residual**2))),float(np.quantile(residual,.95)))
        if quality[0]>maximum_rms_m or quality[1]>maximum_p95_m:
            remaining[mask]=False;continue
        row=len(normals);labels[mask]=row;normals.append(plane_normal);offsets.append(float(plane_normal@center));counts.append(int(mask.sum()));members.append(primitive_rows);rms.append(quality[0]);p95.append(quality[1])
        point_sum.append(np.sum(points,axis=0));point_second.append(points.T@points)
        remaining[mask]=False
    member_offsets=np.r_[0,np.cumsum([x.size for x in members])].astype(np.int64)
    return RenderedPlaneObservations(labels=labels,normals_world=np.asarray(normals,np.float64).reshape(-1,3),offsets_world=np.asarray(offsets,np.float64),pixel_counts=np.asarray(counts,np.int64),member_offsets=member_offsets,member_primitive_rows=np.concatenate(members).astype(np.int64) if members else np.zeros(0,np.int64),residual_rms_m=np.asarray(rms),residual_p95_m=np.asarray(p95),point_sum_world=np.asarray(point_sum,np.float64).reshape(-1,3),point_second_moment_world=np.asarray(point_second,np.float64).reshape(-1,3,3))

__all__=['RenderedPlaneObservations','extract_rendered_plane_observations']
