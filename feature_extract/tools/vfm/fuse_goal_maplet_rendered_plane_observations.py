from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import PrimitiveSurfaceTable
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.rendered_plane_fusion import fuse_rendered_plane_observations
def sha(path):
 h=hashlib.sha256(path.read_bytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument('--physical_map',type=Path);source.add_argument('--surface_table',type=Path);p.add_argument('--observation_dir',type=Path,required=True);p.add_argument('--image_ids_file',type=Path);p.add_argument('--minimum_views',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 source_path=a.surface_table if a.surface_table is not None else a.physical_map
 table=PrimitiveSurfaceTable.load_npz(a.surface_table) if a.surface_table is not None else PrimitiveSurfaceTable.from_physical_map(GoalMapletPhysicalMap.load_npz(a.physical_map));paths=sorted(a.observation_dir.glob('*.planes.npz'))
 selected_ids=None
 if a.image_ids_file is not None:
  selected_ids={line.strip().replace('/','__') for line in a.image_ids_file.read_text().splitlines() if line.strip()}
  paths=[path for path in paths if path.name[:-len('.planes.npz')] in selected_ids]
  if len(paths)!=len(selected_ids):raise RuntimeError('selected observation inventory is incomplete')
 manifest=json.loads((a.observation_dir/'manifest.json').read_text());
 if ((selected_ids is None and len(paths)!=manifest['view_count']) or manifest.get('primitive_surface_table_file_sha256') not in (None,sha(source_path))):raise RuntimeError('observation run is incomplete or uses another surface table')
 result,lineage=fuse_rendered_plane_observations(table,paths,minimum_views=int(a.minimum_views));a.output.parent.mkdir(parents=True,exist_ok=True);result.save_npz(a.output)
 lineage_path=a.output.with_suffix('.lineage.npz');np.savez_compressed(lineage_path,**lineage)
 report={'artifact_type':'goal_maplet_rendered_plane_fusion_report_v1','primitive_surface_table_file_sha256':sha(source_path),'minimum_views':int(a.minimum_views),'planar_map':str(a.output),'planar_map_file_sha256':sha(a.output),'lineage_file_sha256':sha(lineage_path),'view_count':len(paths),'image_ids_file_sha256':None if a.image_ids_file is None else sha(a.image_ids_file),'observation_count':result.metadata['observation_count'],'plane_count':int(result.plane_ids.size),'assigned_primitive_count':int(result.member_primitive_rows.size),'assigned_primitive_fraction':float(result.member_primitive_rows.size/table.primitive_ids.size),'area_ge_1m2':int(np.sum(result.support_area_m2>=1)),'area_ge_5m2':int(np.sum(result.support_area_m2>=5)),'area_ge_10m2':int(np.sum(result.support_area_m2>=10)),'median_area_m2':float(np.median(result.support_area_m2)),'median_rms_m':float(np.median(result.residual_rms_m))}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
