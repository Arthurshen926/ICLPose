"""Sequential-RANSAC finite query planes from a monocular point map."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
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
__all__=['QueryPlaneRegions','extract_query_plane_regions']
