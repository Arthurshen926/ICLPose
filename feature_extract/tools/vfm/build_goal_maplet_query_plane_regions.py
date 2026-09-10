"""Cache deterministic finite query planes so downstream gates share geometry."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions,SparseOcclusionCarrierConfig,extract_query_plane_regions,merge_sparse_foreground_occluded_regions

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def _work(job):
 source,output,base_path,carrier,config_payload=job;source,output=map(Path,(source,output));base_path=None if base_path is None else Path(base_path);config=SparseOcclusionCarrierConfig(**config_payload)
 if output.exists():
  value,meta=QueryPlaneRegions.load_npz(output)
  if meta.get('source_file_sha256')!=_sha(source) or meta.get('uses_pose_or_ground_truth') is not False:raise ValueError('existing query-plane cache differs from MoGe source or pose-free contract')
  expected_base=None if base_path is None else _sha(base_path)
  if meta.get('base_query_plane_file_sha256')!=expected_base:raise ValueError('existing query-plane cache differs from base region source')
  if bool(meta.get('sparse_occlusion_carrier',False))!=bool(carrier) or (carrier and meta.get('carrier_diagnostics',{}).get('config')!=config.payload()):raise ValueError('existing query-plane cache has a different carrier contract')
  return output.name,len(value.normals_camera),int(value.pixel_counts.sum()),meta['content_sha256'],int(meta.get('carrier_diagnostics',{}).get('merged_component_count',0))
 with np.load(source,allow_pickle=False) as data:
  points=np.asarray(data['points_camera']);normals=np.asarray(data['normal_camera']);valid=np.asarray(data['valid']);diagnostics=None
  if base_path is None:value=extract_query_plane_regions(points,normals,valid);base_meta=None
  else:
   value,base_meta=QueryPlaneRegions.load_npz(base_path)
   if base_meta.get('source_file_sha256')!=_sha(source) or base_meta.get('uses_pose_or_ground_truth') is not False:raise ValueError('base query-plane cache differs from MoGe source')
  if carrier:value,diagnostics=merge_sparse_foreground_occluded_regions(value,points,normals,valid,config=config)
 metadata={'source_name':source.name,'source_file_sha256':_sha(source),'uses_pose_or_ground_truth':False,'extractor':'sequential_ransac_connected_finite_planes_v1','base_query_plane_file_sha256':None if base_path is None else _sha(base_path),'base_query_plane_content_sha256':None if base_meta is None else base_meta.get('content_sha256'),'sparse_occlusion_carrier':bool(carrier),'carrier_diagnostics':diagnostics}
 meta=value.save_npz(output,metadata)
 return output.name,len(value.normals_camera),int(value.pixel_counts.sum()),meta['content_sha256'],0 if diagnostics is None else int(diagnostics['merged_component_count'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--base_query_plane_dir',type=Path);p.add_argument('--workers',type=int,default=12);p.add_argument('--maximum_queries',type=int,default=0);p.add_argument('--selection',choices=('prefix','uniform'),default='prefix');p.add_argument('--sparse_occlusion_carrier',action='store_true');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);config=SparseOcclusionCarrierConfig()
 all_sources=sorted(x for x in a.moge_dir.glob('*.npz') if x.name!='manifest.json');sources=all_sources
 if 0<int(a.maximum_queries)<len(all_sources):
  if a.selection=='uniform':indices=np.rint(np.linspace(0,len(all_sources)-1,int(a.maximum_queries))).astype(np.int64);sources=[all_sources[int(i)] for i in indices]
  else:sources=all_sources[:int(a.maximum_queries)]
 jobs=[(str(x),str(a.output_dir/x.name),None if a.base_query_plane_dir is None else str(a.base_query_plane_dir/x.name),bool(a.sparse_occlusion_carrier),config.payload()) for x in sources]
 with mp.get_context('fork').Pool(a.workers) as pool:rows=sorted(pool.imap_unordered(_work,jobs,chunksize=1))
 report={'artifact_type':'goal_maplet_query_plane_region_cache_run_v3','source_dir':str(a.moge_dir),'base_query_plane_dir':None if a.base_query_plane_dir is None else str(a.base_query_plane_dir),'source_query_count':len(all_sources),'query_count':len(rows),'selection':a.selection,'maximum_queries':int(a.maximum_queries),'selected_names_in_order':[x.name for x in sources],'plane_count':sum(x[1] for x in rows),'covered_pixel_count':sum(x[2] for x in rows),'uses_pose_or_ground_truth':False,'sparse_occlusion_carrier':bool(a.sparse_occlusion_carrier),'carrier_config':config.payload() if a.sparse_occlusion_carrier else None,'merged_component_count':sum(x[4] for x in rows),'hidden_pixel_count_added':0,'observed_support_only':True,'rows':rows};(a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
