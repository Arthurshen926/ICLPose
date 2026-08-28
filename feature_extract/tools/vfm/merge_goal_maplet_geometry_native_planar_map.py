"""Merge adjacent geometry-native plane fragments with full refit rollback."""
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap, merge_adjacent_coplanar_planes
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.summary.exists(): raise FileExistsError('merge outputs must be new')
    started=time.perf_counter();physical=GoalMapletPhysicalMap.load_npz(a.physical_map);source=GeometryNativePlanarMap.load_npz(a.input,primitive_count=physical.primitive_ids.size)
    merged=merge_adjacent_coplanar_planes(physical,source);a.output.parent.mkdir(parents=True,exist_ok=True);merged.save_npz(a.output);loaded=GeometryNativePlanarMap.load_npz(a.output,primitive_count=physical.primitive_ids.size)
    summary={'artifact_type':'goal_maplet_geometry_native_planar_map_merge_v1','physical_map_file_sha256':file_sha256(a.physical_map),'source_file_sha256':file_sha256(a.input),'output_file_sha256':file_sha256(a.output),'elapsed_seconds':time.perf_counter()-started,'premerge_plane_count':int(source.plane_ids.size),'plane_count':int(loaded.plane_ids.size),'accepted_merge_count':int(loaded.metadata['accepted_merge_count']),'rejected_merge_count':int(loaded.metadata['rejected_merge_count']),'area_gt_1m2':int(np.sum(loaded.support_area_m2>=1)),'area_gt_5m2':int(np.sum(loaded.support_area_m2>=5)),'area_gt_10m2':int(np.sum(loaded.support_area_m2>=10)),'metadata':loaded.metadata}
    summary['content_sha256']=hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
