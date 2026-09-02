"""Compact true mapping-view masks for fused finite plane instances."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .lineage import arrays_sha256,canonical_json_sha256
from .rendered_view_planes import RenderedPlaneObservations

@dataclass(frozen=True)
class PlaneVisibilityAtlas:
 plane_offsets:np.ndarray;plane_observation_rows:np.ndarray;view_names:np.ndarray;poses_w2c:np.ndarray;centers_world:np.ndarray;token_pixel_counts:np.ndarray
 def arrays(self):return {name:np.asarray(getattr(self,name)) for name in ('plane_offsets','plane_observation_rows','view_names','poses_w2c','centers_world','token_pixel_counts')}
 def validated(self):
  a=self.arrays();n=int(a['plane_observation_rows'].size);p=int(a['plane_offsets'].size-1)
  if a['plane_offsets'].shape!=(p+1,) or a['plane_offsets'][0]!=0 or a['plane_offsets'][-1]!=n or np.any(np.diff(a['plane_offsets'])<0):raise ValueError('invalid visibility plane offsets')
  if a['view_names'].shape!=(n,) or a['poses_w2c'].shape!=(n,4,4) or a['centers_world'].shape!=(n,3) or a['token_pixel_counts'].ndim!=3 or a['token_pixel_counts'].shape[0]!=n:raise ValueError('invalid visibility atlas shapes')
  if a['token_pixel_counts'].dtype!=np.uint8 or np.any(a['token_pixel_counts']>16):raise ValueError('invalid normalized token counts')
  if not np.isfinite(a['poses_w2c']).all() or not np.isfinite(a['centers_world']).all():raise ValueError('nonfinite visibility atlas')
  if len(set(a['plane_observation_rows'].tolist()))!=n:raise ValueError('duplicate visibility observations')
  return self
 def save_npz(self,path,metadata):
  arrays=self.validated().arrays();meta=dict(metadata);meta.update(artifact_type='goal_maplet_plane_visibility_atlas_v1',arrays_sha256=arrays_sha256(arrays));meta['content_sha256']=canonical_json_sha256(meta);path=Path(path);temporary=path.with_name(path.name+'.temporary.npz');np.savez_compressed(temporary,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)));temporary.replace(path);return meta
 @classmethod
 def load_npz(cls,path):
  with np.load(path,allow_pickle=False) as data:meta=json.loads(str(data['metadata_json'].item()));arrays={name:np.asarray(data[name]) for name in ('plane_offsets','plane_observation_rows','view_names','poses_w2c','centers_world','token_pixel_counts')}
  if meta.get('artifact_type')!='goal_maplet_plane_visibility_atlas_v1' or arrays_sha256(arrays)!=meta.get('arrays_sha256'):raise ValueError('visibility atlas lineage differs')
  if list(arrays['token_pixel_counts'].shape[1:])!=list(meta.get('token_grid',[36,64])):raise ValueError('visibility atlas token grid differs')
  return cls(**arrays).validated(),meta

def normalized_token_counts(mask,token_grid):
 mask=np.asarray(mask,np.uint8);height,width=map(int,token_grid)
 if mask.shape==(height*4,width*4):
  return mask.reshape(height,4,width,4).sum((1,3)).astype(np.uint8)
 import cv2
 fraction=cv2.resize(mask.astype(np.float32),(width,height),interpolation=cv2.INTER_AREA)
 return np.clip(np.rint(16.0*fraction),0,16).astype(np.uint8)

def build_plane_visibility_atlas(observation_paths,lineage_offsets,lineage_rows,contributors_dir,token_grid=(36,64)):
 paths=sorted(map(Path,observation_paths));lookup={};global_row=0
 for path in paths:
  observation,meta=RenderedPlaneObservations.load_npz(path)
  source=Path(contributors_dir)/str(meta['source_name'])
  with np.load(source,allow_pickle=False) as data:pose=np.asarray(data['pose_w2c'],np.float64)
  center=-pose[:3,:3].T@pose[:3,3]
  for local in range(len(observation.normals_world)):
   mask=(observation.labels==local).astype(np.uint8);counts=normalized_token_counts(mask,token_grid);lookup[global_row]=(meta['source_name'],pose,center,counts);global_row+=1
 rows=np.asarray(lineage_rows,np.int64);missing=[int(x) for x in rows if int(x) not in lookup]
 if missing:raise ValueError('lineage references missing rendered observations')
 records=[lookup[int(x)] for x in rows]
 return PlaneVisibilityAtlas(np.asarray(lineage_offsets,np.int64),rows,np.asarray([x[0] for x in records]),np.asarray([x[1] for x in records]),np.asarray([x[2] for x in records]),np.asarray([x[3] for x in records],np.uint8)).validated()

def mask_dice(query_mask_counts,template_counts):
 q=np.asarray(query_mask_counts,np.float64)/16.;t=np.asarray(template_counts,np.float64)/16.;return float(2*np.minimum(q,t).sum()/max(q.sum()+t.sum(),1e-12))

__all__=['PlaneVisibilityAtlas','build_plane_visibility_atlas','normalized_token_counts','mask_dice']
