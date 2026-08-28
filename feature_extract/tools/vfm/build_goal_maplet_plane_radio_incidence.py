"""Bridge physical planes to legacy RADIO parent/child retrieval indices.

The hierarchy remains an index only: a plane can overlap many cells and no
cell boundary is promoted to a physical plane boundary.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256
def _owner(offsets,rows,n):
 result=np.full(n,-1,np.int64)
 for owner,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):result[rows[int(lo):int(hi)]]=owner
 if np.any(result<0):raise ValueError('hierarchy does not own every primitive exactly once')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 physical=GoalMapletPhysicalMap.load_npz(a.physical_map);plane=GeometryNativePlanarMap.load_npz(a.planar_map,primitive_count=physical.primitive_ids.size)
 parent=_owner(physical.membership_offsets,physical.membership_primitive_rows,physical.primitive_ids.size);child=_owner(physical.child_member_offsets,physical.child_member_primitive_rows,physical.primitive_ids.size)
 offsets=[0];parents=[];children=[];weights=[]
 primitive_area=np.pi*physical.primitive_scale1*physical.primitive_scale2*physical.primitive_opacity
 for row in range(plane.plane_ids.size):
  lo,hi=map(int,plane.member_offsets[row:row+2]);members=plane.member_primitive_rows[lo:hi]
  pair=np.c_[parent[members],child[members]];unique,inverse=np.unique(pair,axis=0,return_inverse=True);mass=np.bincount(inverse,weights=primitive_area[members])
  parents.extend(unique[:,0]);children.extend(unique[:,1]);weights.extend(mass);offsets.append(len(parents))
 arrays={'plane_cell_offsets':np.asarray(offsets,np.int64),'parent_rows':np.asarray(parents,np.int64),'child_rows':np.asarray(children,np.int64),'surface_mass_m2':np.asarray(weights,np.float64)}
 metadata={'artifact_type':'goal_maplet_plane_radio_index_incidence_v1','semantics':'plane_to_legacy_retrieval_index_overlap_not_plane_identity','plane_count':int(plane.plane_ids.size),'parent_count':int(physical.maplet_ids.size),'child_count':int(physical.child_parent_rows.size),'arrays_sha256':arrays_sha256(arrays)};metadata['content_sha256']=canonical_json_sha256(metadata)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
 report={'plane_count':metadata['plane_count'],'incidence_rows':len(parents),'mean_cells_per_plane':float(np.mean(np.diff(offsets))),'content_sha256':metadata['content_sha256']};print(json.dumps(report,indent=2))
if __name__=='__main__':main()
