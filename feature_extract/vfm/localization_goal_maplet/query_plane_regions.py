"""Sequential-RANSAC finite query planes from a monocular point map.

The sparse-occlusion carrier in this module deliberately merges only the
*observed* supports of compatible planes.  Pixels hidden by an occluder remain
unlabelled: downstream RADIO matching and PnP can group evidence on both sides
of a branch, but can never manufacture a descriptor or a correspondence in the
hidden strip.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .lineage import arrays_sha256, canonical_json_sha256
@dataclass(frozen=True)
class QueryPlaneRegions:
 labels:np.ndarray;normals_camera:np.ndarray;offsets_camera:np.ndarray;pixel_counts:np.ndarray;residual_rms:np.ndarray;residual_p95:np.ndarray
 def arrays(self):
  return {name:np.asarray(getattr(self,name)) for name in ('labels','normals_camera','offsets_camera','pixel_counts','residual_rms','residual_p95')}
 def validated(self):
  arrays=self.arrays();n=int(arrays['normals_camera'].shape[0])
  if arrays['labels'].ndim!=2 or arrays['normals_camera'].shape!=(n,3) or arrays['offsets_camera'].shape!=(n,) or arrays['pixel_counts'].shape!=(n,) or arrays['residual_rms'].shape!=(n,) or arrays['residual_p95'].shape!=(n,):raise ValueError('invalid query-plane shapes')
  if arrays['labels'].dtype.kind not in 'iu' or np.any((arrays['labels']<-1)|(arrays['labels']>=n)):raise ValueError('invalid query-plane labels')
  if not all(np.isfinite(arrays[x]).all() for x in ('normals_camera','offsets_camera','residual_rms','residual_p95')):raise ValueError('nonfinite query-plane values')
  if np.any(arrays['pixel_counts']<=0) or any(int(np.sum(arrays['labels']==row))!=int(arrays['pixel_counts'][row]) for row in range(n)):raise ValueError('query-plane pixel counts differ from labels')
  if n and np.max(np.abs(np.linalg.norm(arrays['normals_camera'],axis=1)-1))>1e-8:raise ValueError('query-plane normals are not unit')
  return self
 def save_npz(self,path:Path,metadata:dict):
  arrays=self.validated().arrays();meta=dict(metadata);meta.update(artifact_type='goal_maplet_query_plane_regions_v1',arrays_sha256=arrays_sha256(arrays));meta['content_sha256']=canonical_json_sha256(meta)
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.temporary.npz');np.savez_compressed(temporary,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)));temporary.replace(path);return meta
 @classmethod
 def load_npz(cls,path:Path):
  with np.load(Path(path),allow_pickle=False) as data:
   meta=json.loads(str(data['metadata_json'].item()));arrays={name:np.asarray(data[name]) for name in ('labels','normals_camera','offsets_camera','pixel_counts','residual_rms','residual_p95')}
  if meta.get('artifact_type')!='goal_maplet_query_plane_regions_v1' or arrays_sha256(arrays)!=meta.get('arrays_sha256'):raise ValueError('query-plane cache lineage differs')
  return cls(**arrays).validated(),meta
def extract_query_plane_regions(points:np.ndarray,normals:np.ndarray,valid:np.ndarray,*,minimum_pixels=80,maximum_planes=32,maximum_hypotheses=96,distance_ratio=.005,normal_degrees=30.):
 points=np.asarray(points,np.float64);normals=np.asarray(normals,np.float64);valid=np.asarray(valid,bool)&np.isfinite(points).all(2)&np.isfinite(normals).all(2)
 length=np.linalg.norm(normals,axis=2);valid&=length>1e-8;normals=normals/np.maximum(length[...,None],1e-15)
 h,w=valid.shape;flatp=points.reshape(-1,3);flatn=normals.reshape(-1,3);depth=np.linalg.norm(points[valid],axis=1);threshold=float(np.clip(distance_ratio*np.median(depth),.03,.20));cosine=float(np.cos(np.deg2rad(normal_degrees)))
 remaining=valid.copy();labels=np.full((h,w),-1,np.int32);outn=[];outd=[];counts=[];rms=[];p95=[]
 for _ in range(maximum_planes):
  candidate=np.flatnonzero(remaining.ravel())
  if candidate.size<minimum_pixels:break
  seeds=candidate[np.linspace(0,candidate.size-1,min(maximum_hypotheses,candidate.size),dtype=np.int64)];best=None
  for seed in seeds:
   n=flatn[seed];p=flatp[seed];mask=remaining&(np.abs((flatp-p)@n).reshape(h,w)<=threshold)&(np.abs(flatn@n).reshape(h,w)>=cosine);_,component=cv2.connectedComponents(mask.astype(np.uint8),8);owner=int(component.ravel()[seed]);region=component==owner if owner else None
   if region is not None and (best is None or int(region.sum())>int(best[0].sum())):best=(region,int(seed))
  if best is None or int(best[0].sum())<minimum_pixels:break
  mask,seed=best
  for _ in range(3):
   sample=points[mask];center=sample.mean(0);cov=(sample-center).T@(sample-center)/len(sample);_,v=np.linalg.eigh(cov);n=v[:,0];reference=normals[mask].mean(0);n*=1 if n@reference>=0 else -1
   proposal=remaining&(np.abs((flatp-center)@n).reshape(h,w)<=threshold)&(np.abs(flatn@n).reshape(h,w)>=cosine);_,component=cv2.connectedComponents(proposal.astype(np.uint8),8);owner=int(component.ravel()[seed]);updated=component==owner if owner else mask
   if np.array_equal(mask,updated):break
   mask=updated
  if int(mask.sum())<minimum_pixels:remaining.ravel()[seed]=False;continue
  sample=points[mask];center=sample.mean(0);cov=(sample-center).T@(sample-center)/len(sample);_,v=np.linalg.eigh(cov);n=v[:,0];n*=1 if n@normals[mask].mean(0)>=0 else -1;d=float(n@center)
  if d<0:n=-n;d=-d
  residual=np.abs(sample@n-d);labels[mask]=len(outn);outn.append(n);outd.append(d);counts.append(int(mask.sum()));rms.append(float(np.sqrt(np.mean(residual**2))));p95.append(float(np.quantile(residual,.95)));remaining[mask]=False
 return QueryPlaneRegions(labels,np.asarray(outn).reshape(-1,3),np.asarray(outd),np.asarray(counts,np.int64),np.asarray(rms),np.asarray(p95)).validated()


@dataclass(frozen=True)
class SparseOcclusionCarrierConfig:
 """Frozen, query-label-free limits for thin foreground occluders.

 ``maximum_gap_pixels`` is measured on the 256x144 MoGe grid.  A bridge must
 be supported by multiple separated image paths, so a single accidental near
 contact cannot join two regions.  The component cap prevents a dense canopy
 from being converted into one arbitrarily large carrier by transitivity.
 """

 maximum_gap_pixels:int=16
 minimum_bridge_paths:int=2
 minimum_path_separation_pixels:float=4.0
 maximum_group_components:int=6
 maximum_normal_degrees:float=12.0
 maximum_refit_p95_ratio:float=1.5
 minimum_corridor_valid_fraction:float=0.8
 minimum_foreground_fraction:float=0.6
 minimum_foreground_range_ratio:float=0.025
 maximum_other_plane_fraction:float=0.25
 distance_ratio:float=0.005

 def validated(self):
  if self.maximum_gap_pixels<2 or self.minimum_bridge_paths<1 or self.maximum_group_components<2:raise ValueError('invalid sparse-occlusion carrier size limit')
  if self.minimum_path_separation_pixels<0 or not 0<self.maximum_normal_degrees<90:raise ValueError('invalid sparse-occlusion carrier geometry limit')
  for value in (self.minimum_corridor_valid_fraction,self.minimum_foreground_fraction):
   if not 0<=value<=1:raise ValueError('invalid sparse-occlusion carrier fraction')
  if not 0<=self.maximum_other_plane_fraction<=1 or self.minimum_foreground_range_ratio<=0 or self.maximum_refit_p95_ratio<=0 or self.distance_ratio<=0:raise ValueError('invalid sparse-occlusion carrier threshold')
  return self

 def payload(self):return asdict(self.validated())


def _fit_carrier_plane(points:np.ndarray,reference:np.ndarray):
 center=np.mean(points,axis=0);covariance=(points-center).T@(points-center)/max(len(points),1);_,vectors=np.linalg.eigh(covariance);normal=vectors[:,0]
 if float(normal@reference)<0:normal=-normal
 normal/=max(float(np.linalg.norm(normal)),1e-15);offset=float(normal@center)
 if offset<0:normal=-normal;offset=-offset
 residual=np.abs(points@normal-offset)
 return normal,offset,residual


def _pixel_line(left:np.ndarray,right:np.ndarray)->np.ndarray:
 count=max(abs(int(left[0])-int(right[0])),abs(int(left[1])-int(right[1])))+1
 return np.unique(np.rint(np.linspace(left,right,count)).astype(np.int64),axis=0)


def _candidate_bridge_paths(left:np.ndarray,right:np.ndarray,config:SparseOcclusionCarrierConfig):
 # Query the smaller support for deterministic nearest pixels in the other
 # support.  Midpoint separation makes the decision depend on more than one
 # isolated contact.
 if len(left)>len(right):left,right=right,left
 distance,index=cKDTree(right).query(left,k=1)
 order=np.lexsort((left[:,1],left[:,0],distance))
 paths=[];midpoints=[]
 for row in order.tolist():
  if float(distance[row])>float(config.maximum_gap_pixels):break
  line=_pixel_line(left[row],right[int(index[row])])
  if len(line)<=2:continue
  midpoint=.5*(left[row].astype(np.float64)+right[int(index[row])].astype(np.float64))
  if any(float(np.linalg.norm(midpoint-prior))<config.minimum_path_separation_pixels for prior in midpoints):continue
  paths.append(line);midpoints.append(midpoint)
  if len(paths)>=config.minimum_bridge_paths:break
 return paths


def _bridge_path_is_foreground(line:np.ndarray,left:int,right:int,labels:np.ndarray,points:np.ndarray,valid:np.ndarray,normal:np.ndarray,offset:float,config:SparseOcclusionCarrierConfig):
 interior=np.asarray([yx for yx in line[1:-1] if int(labels[tuple(yx)]) not in (left,right)],np.int64).reshape(-1,2)
 if not len(interior):return False,{'corridor_pixels':0,'valid_fraction':0.0,'foreground_fraction':0.0,'other_plane_fraction':0.0}
 other=np.asarray([int(labels[tuple(yx)])>=0 for yx in interior],bool)
 other_fraction=float(np.mean(other))
 if other_fraction>config.maximum_other_plane_fraction:return False,{'corridor_pixels':int(len(interior)),'valid_fraction':0.0,'foreground_fraction':0.0,'other_plane_fraction':other_fraction}
 is_valid=np.asarray([bool(valid[tuple(yx)]) for yx in interior],bool)
 valid_fraction=float(np.mean(is_valid))
 if valid_fraction<config.minimum_corridor_valid_fraction:return False,{'corridor_pixels':int(len(interior)),'valid_fraction':valid_fraction,'foreground_fraction':0.0,'other_plane_fraction':other_fraction}
 sample=points[interior[is_valid,0],interior[is_valid,1]];distance=np.linalg.norm(sample,axis=1);direction=sample/np.maximum(distance[:,None],1e-15);denominator=direction@normal
 usable=denominator>1e-8
 if float(np.mean(usable))<config.minimum_corridor_valid_fraction:return False,{'corridor_pixels':int(len(interior)),'valid_fraction':valid_fraction,'foreground_fraction':0.0,'other_plane_fraction':other_fraction}
 predicted=offset/denominator[usable];foreground=(predicted-distance[usable])/np.maximum(predicted,1e-15)>=config.minimum_foreground_range_ratio;foreground_fraction=float(np.mean(foreground))
 return foreground_fraction>=config.minimum_foreground_fraction,{'corridor_pixels':int(len(interior)),'valid_fraction':valid_fraction,'foreground_fraction':foreground_fraction,'other_plane_fraction':other_fraction}


def merge_sparse_foreground_occluded_regions(base:QueryPlaneRegions,points:np.ndarray,normals:np.ndarray,valid:np.ndarray,*,config:SparseOcclusionCarrierConfig=SparseOcclusionCarrierConfig()):
 """Join coplanar visible islands across narrow, measured foreground gaps.

 The returned label support is a permutation/union of the original nonnegative
 pixels.  Hidden pixels are never filled.  The diagnostic dictionary is safe
 to produce before pose or ground-truth access.
 """
 base=base.validated();config=config.validated();points=np.asarray(points,np.float64);normals=np.asarray(normals,np.float64);valid=np.asarray(valid,bool)
 if points.ndim!=3 or points.shape[-1]!=3 or normals.shape!=points.shape or valid.shape!=points.shape[:2] or base.labels.shape!=valid.shape:raise ValueError('sparse-occlusion carrier inputs have incompatible shapes')
 valid=valid&np.isfinite(points).all(2)&np.isfinite(normals).all(2)
 observed=base.labels>=0;count=len(base.normals_camera)
 if not count:return base,{'config':config.payload(),'base_plane_count':0,'carrier_plane_count':0,'accepted_edges':[],'observed_pixel_count':0,'hidden_pixel_count_added':0}
 depth=np.linalg.norm(points[valid],axis=1);threshold=float(np.clip(config.distance_ratio*np.median(depth),.03,.20))
 masks=[base.labels==row for row in range(count)];pixels=[np.argwhere(mask) for mask in masks]
 candidates=[];cosine=float(np.cos(np.deg2rad(config.maximum_normal_degrees)))
 for left in range(count):
  for right in range(left+1,count):
   if float(base.normals_camera[left]@base.normals_camera[right])<cosine:continue
   paths=_candidate_bridge_paths(pixels[left],pixels[right],config)
   if len(paths)<config.minimum_bridge_paths:continue
   mask=masks[left]|masks[right];sample=points[mask];normal,offset,residual=_fit_carrier_plane(sample,base.normals_camera[left]+base.normals_camera[right]);ratio=float(np.quantile(residual,.95)/threshold)
   if ratio>config.maximum_refit_p95_ratio:continue
   path_rows=[]
   for path in paths:
    accepted,row=_bridge_path_is_foreground(path,left,right,base.labels,points,valid,normal,offset,config);row['accepted']=bool(accepted);path_rows.append(row)
   if sum(bool(row['accepted']) for row in path_rows)<config.minimum_bridge_paths:continue
   candidates.append((max(len(path) for path in paths),-min(row['foreground_fraction'] for row in path_rows),ratio,left,right,path_rows))
 parent=np.arange(count,dtype=np.int64);members={row:{row} for row in range(count)}
 def root(row):
  while int(parent[row])!=row:parent[row]=parent[int(parent[row])];row=int(parent[row])
  return int(row)
 accepted_edges=[]
 for _,_,pair_ratio,left,right,path_rows in sorted(candidates,key=lambda row:row[:5]):
  a,b=root(left),root(right)
  if a==b:continue
  union=members[a]|members[b]
  if len(union)>config.maximum_group_components:continue
  mask=np.isin(base.labels,np.asarray(sorted(union),np.int64));sample=points[mask];reference=np.sum(base.normals_camera[np.asarray(sorted(union),np.int64)],axis=0);normal,offset,residual=_fit_carrier_plane(sample,reference);group_ratio=float(np.quantile(residual,.95)/threshold)
  if group_ratio>config.maximum_refit_p95_ratio:continue
  keep,drop=min(a,b),max(a,b);parent[drop]=keep;members[keep]=union;del members[drop]
  accepted_edges.append({'left_region':int(left),'right_region':int(right),'pair_refit_p95_ratio':float(pair_ratio),'group_refit_p95_ratio':group_ratio,'bridge_paths':path_rows})
 groups=sorted((sorted(value) for value in members.values()),key=lambda value:value[0]);labels=np.full(base.labels.shape,-1,np.int32);outn=[];outd=[];counts=[];rms=[];p95=[]
 for group in groups:
  mask=np.isin(base.labels,np.asarray(group,np.int64));sample=points[mask];reference=np.sum(base.normals_camera[np.asarray(group,np.int64)],axis=0);normal,offset,residual=_fit_carrier_plane(sample,reference);labels[mask]=len(outn);outn.append(normal);outd.append(offset);counts.append(int(mask.sum()));rms.append(float(np.sqrt(np.mean(residual**2))));p95.append(float(np.quantile(residual,.95)))
 result=QueryPlaneRegions(labels,np.asarray(outn).reshape(-1,3),np.asarray(outd),np.asarray(counts,np.int64),np.asarray(rms),np.asarray(p95)).validated()
 if not np.array_equal(result.labels>=0,observed):raise AssertionError('sparse-occlusion carrier changed observed pixel support')
 diagnostics={'config':config.payload(),'distance_threshold':threshold,'base_plane_count':count,'carrier_plane_count':len(groups),'accepted_edges':accepted_edges,'merged_component_count':int(count-len(groups)),'multi_component_carrier_count':int(sum(len(group)>1 for group in groups)),'maximum_carrier_component_count':int(max(map(len,groups),default=0)),'observed_pixel_count':int(observed.sum()),'hidden_pixel_count_added':0,'observed_support_bit_exact':True}
 return result,diagnostics


__all__=['QueryPlaneRegions','SparseOcclusionCarrierConfig','extract_query_plane_regions','merge_sparse_foreground_occluded_regions']
