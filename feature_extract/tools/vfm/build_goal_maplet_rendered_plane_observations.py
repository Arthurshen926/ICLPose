"""Build resumable per-mapping-view plane observations from 2DGS renderings."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os
from pathlib import Path
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import PrimitiveSurfaceTable
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import extract_rendered_plane_observations,RenderedPlaneObservations

_TABLE=None
def _sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def _work(item):
 source,output=item;source=Path(source);output=Path(output)
 if output.exists():
  observation,meta=RenderedPlaneObservations.load_npz(output,_TABLE.primitive_ids.size);return output.name,len(observation.normals_world),int(observation.pixel_counts.sum()),meta['content_sha256']
 observation=extract_rendered_plane_observations(_TABLE,source)
 meta=observation.save_npz(output,{'source_name':source.name,'source_file_sha256':_sha(source),'uses_rgb':False,'uses_radio':False,'uses_query_or_gt':False})
 return output.name,len(observation.normals_world),int(observation.pixel_counts.sum()),meta['content_sha256']
def main():
 global _TABLE
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--routes',nargs='+',default=['seq1','seq2','seq4','seq6','seq7','seq8','seq9','seq11']);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True);allowed=set(a.routes)
 sources=sorted(x for x in a.contributors.glob('*.npz') if x.name.split('__',1)[0] in allowed)
 if not sources:raise RuntimeError('no mapping contributors selected')
 _TABLE=PrimitiveSurfaceTable.from_physical_map(GoalMapletPhysicalMap.load_npz(a.physical_map))
 jobs=[(str(x),str(a.output_dir/(x.stem+'.planes.npz'))) for x in sources]
 context=mp.get_context('fork')
 with context.Pool(a.workers) as pool: rows=list(pool.imap_unordered(_work,jobs,chunksize=1))
 rows=sorted(rows);manifest={'artifact_type':'goal_maplet_rendered_plane_observation_run_v1','routes':sorted(allowed),'view_count':len(rows),'plane_count':sum(x[1] for x in rows),'covered_pixel_count':sum(x[2] for x in rows),'rows':rows,'uses_parent_child_partition':False,'uses_query_or_gt':False}
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},sort_keys=True))
if __name__=='__main__':main()
