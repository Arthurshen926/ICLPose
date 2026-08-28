"""Cache deterministic finite query planes so downstream gates share geometry."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions,extract_query_plane_regions

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def _work(job):
 source,output=map(Path,job)
 if output.exists():
  value,meta=QueryPlaneRegions.load_npz(output);return output.name,len(value.normals_camera),int(value.pixel_counts.sum()),meta['content_sha256']
 with np.load(source,allow_pickle=False) as data:value=extract_query_plane_regions(data['points_camera'],data['normal_camera'],data['valid'])
 meta=value.save_npz(output,{'source_name':source.name,'source_file_sha256':_sha(source),'uses_pose_or_ground_truth':False,'extractor':'sequential_ransac_connected_finite_planes_v1'})
 return output.name,len(value.normals_camera),int(value.pixel_counts.sum()),meta['content_sha256']
def main():
 p=argparse.ArgumentParser();p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 sources=sorted(x for x in a.moge_dir.glob('*.npz') if x.name!='manifest.json');jobs=[(str(x),str(a.output_dir/x.name)) for x in sources]
 with mp.get_context('fork').Pool(a.workers) as pool:rows=sorted(pool.imap_unordered(_work,jobs,chunksize=1))
 report={'artifact_type':'goal_maplet_query_plane_region_cache_run_v1','source_dir':str(a.moge_dir),'query_count':len(rows),'plane_count':sum(x[1] for x in rows),'covered_pixel_count':sum(x[2] for x in rows),'uses_pose_or_ground_truth':False,'rows':rows};(a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
