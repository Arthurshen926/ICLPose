"""Visualize plane masks extracted from one 2DGS mapping rendering."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import PrimitiveSurfaceTable
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import extract_rendered_plane_observations
def main():
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True);source.add_argument('--physical_map',type=Path);source.add_argument('--surface_table',type=Path);p.add_argument('--contributor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();table=PrimitiveSurfaceTable.load_npz(a.surface_table) if a.surface_table is not None else PrimitiveSurfaceTable.from_physical_map(GoalMapletPhysicalMap.load_npz(a.physical_map));o=extract_rendered_plane_observations(table,a.contributor)
 with np.load(a.contributor,allow_pickle=False) as d: depth=np.asarray(d['dominant_depth']);ids=np.asarray(d['topk_ids'][:,:,0])
 color=np.zeros((*o.labels.shape,3));rng=np.random.default_rng(5);palette=rng.uniform(.1,1,(max(1,len(o.normals_world)),3));valid=o.labels>=0;color[valid]=palette[o.labels[valid]]
 fig,ax=plt.subplots(1,3,figsize=(17,5),dpi=170);ax[0].imshow(depth,cmap='turbo',vmin=np.nanpercentile(depth[depth>0],2),vmax=np.nanpercentile(depth[depth>0],98));ax[0].set_title('2DGS rendered depth');ax[1].imshow(color);ax[1].set_title(f'geometry plane masks: {len(o.normals_world)}');ax[2].imshow(ids>=0,cmap='gray');ax[2].imshow(np.ma.masked_where(~valid,o.labels),cmap='turbo',alpha=.65);ax[2].set_title('accepted plane support over valid surface')
 for x in ax:x.axis('off')
 fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,bbox_inches='tight');print(json.dumps({'plane_count':len(o.normals_world),'covered_pixel_fraction':float(np.mean(valid)),'pixel_counts':o.pixel_counts.tolist()}))
if __name__=='__main__':main()
