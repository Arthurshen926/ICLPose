"""Diagnostic direct-placement chart atlas from pose-free MoGe-3 initializers."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import ExplicitChartAtlas,SCHEMA
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def distances(points,rows):
 out=np.full(len(points),np.inf)
 for chart in np.unique(rows):
  source=np.flatnonzero(rows==chart);target=np.flatnonzero(rows!=chart)
  if len(source)==0 or len(target)==0:continue
  out[source]=cKDTree(points[target]).query(points[source],k=1,workers=-1)[0]
 return out
def summary(x):
 f=x[np.isfinite(x)];return {'finite_fraction':float(len(f)/len(x)),'median_m':float(np.median(f)),'p90_m':float(np.quantile(f,.9)),'within_0p1m':float(np.mean(x<=.1)),'within_0p25m':float(np.mean(x<=.25)),'within_0p5m':float(np.mean(x<=.5))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--initializers',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=4);p.add_argument('--minimum_edge_m',type=float,default=.5);p.add_argument('--edge_relative_depth',type=float,default=.05);a=p.parse_args();camera=json.loads(a.cameras.read_text());names=[Path(x).name for x in camera['filepaths']];index={x:i for i,x in enumerate(names)};c2w=np.asarray(camera['cams2world'],np.float64)
 paths=sorted(x for x in a.initializers.glob('*.npz'));vertices=[];normals=[];uvs=[];confidence=[];faces=[];voff=[0];foff=[0];chart_names=[];chart_rows=[];valid_fraction=[]
 for chart,path in enumerate(paths):
  name=path.name[:-4];row=index[name]
  with np.load(path,allow_pickle=False) as data:pc=np.asarray(data['points_camera'],np.float64);nc=np.asarray(data['normal_camera'],np.float64);valid=np.asarray(data['valid'],bool)
  R=c2w[row,:3,:3];C=c2w[row,:3,3];world=pc@R.T+C;world_normal=nc@R.T;ys=np.arange(0,pc.shape[0],a.stride);xs=np.arange(0,pc.shape[1],a.stride);grid=world[ys[:,None],xs];grid_n=world_normal[ys[:,None],xs];q=valid[ys[:,None],xs]&np.isfinite(grid).all(2)&np.isfinite(grid_n).all(2);local=np.full(q.shape,-1,np.int64);local[q]=np.arange(q.sum())+voff[-1];v=grid[q];n=grid_n[q];n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-15);vertices.append(v);normals.append(n);uv=np.stack(np.meshgrid(xs/(pc.shape[1]-1),ys/(pc.shape[0]-1)),axis=-1);uvs.append(uv[q]);confidence.append(np.ones(len(v),np.float32));local_faces=[];metric_depth=pc[ys[:,None],xs,2]
  for y in range(len(ys)-1):
   for x in range(len(xs)-1):
    ids=np.asarray([local[y,x],local[y,x+1],local[y+1,x],local[y+1,x+1]])
    if np.any(ids<0):continue
    xyz=np.asarray([grid[y,x],grid[y,x+1],grid[y+1,x],grid[y+1,x+1]]);threshold=max(a.minimum_edge_m,a.edge_relative_depth*float(np.median([metric_depth[y,x],metric_depth[y,x+1],metric_depth[y+1,x],metric_depth[y+1,x+1]])))
    if max(np.linalg.norm(xyz[i]-xyz[j]) for i,j in ((0,1),(0,2),(1,3),(2,3),(0,3),(1,2)))<=threshold:local_faces.extend(((ids[0],ids[2],ids[1]),(ids[1],ids[2],ids[3])))
  f=np.asarray(local_faces,np.int64).reshape(-1,3);faces.append(f);voff.append(voff[-1]+len(v));foff.append(foff[-1]+len(f));chart_names.append(name);chart_rows.append(np.full(len(v),chart,np.int32));valid_fraction.append(float(q.mean()))
 metadata={'artifact_type':SCHEMA,'representation':'direct_placed_moge3_surface_charts_negative_control','source_initializer_manifest_file_sha256':sha(a.initializers/'manifest.json'),'source_cameras_file_sha256':sha(a.cameras),'chart_count':len(paths),'stride':a.stride,'uses_mapping_camera_pose_for_offline_placement':True,'uses_query_or_ground_truth':False,'multi_view_alignment_applied':False,'production_eligible':False,'control_only':True,'valid_fraction_mean':float(np.mean(valid_fraction))}
 atlas=ExplicitChartAtlas(np.asarray(chart_names),np.asarray(voff),np.concatenate(vertices),np.concatenate(normals),np.concatenate(uvs),np.concatenate(confidence),np.asarray(foff),np.concatenate(faces),metadata).validated();meta=atlas.save_npz(a.output);x=distances(atlas.vertices_world,np.concatenate(chart_rows));report={'artifact_type':'goal_maplet_moge3_direct_chart_atlas_audit_v1','output_file_sha256':sha(a.output),'content_sha256':meta['content_sha256'],'chart_count':len(paths),'vertex_count':len(atlas.vertices_world),'face_count':len(atlas.faces),'npz_mib':a.output.stat().st_size/2**20,'cross_chart_nearest':summary(x),'production_eligible':False,'blocker':'multi_view_chart_alignment_absent'};a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
