"""Render the largest finite plane polygons instead of storage primitives."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
def main():
 p=argparse.ArgumentParser();p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--top',type=int,default=256);a=p.parse_args();m=GeometryNativePlanarMap.load_npz(a.planar_map)
 order=np.argsort(m.support_area_m2)[::-1][:a.top];polygons=[];colors=[];cmap=plt.get_cmap('turbo')
 for rank,row in enumerate(order):
  lo,hi=map(int,m.boundary_offsets[row:row+2]);uv=m.boundary_uv[lo:hi]
  if len(uv)<3:continue
  polygons.append(m.centers_world[row]+uv[:,:1]*m.frames_world[row,0]+uv[:,1:]*m.frames_world[row,1]);colors.append(cmap((rank%64)/63))
 fig=plt.figure(figsize=(16,8),dpi=180);ax=fig.add_subplot(121,projection='3d');collection=Poly3DCollection(polygons,facecolors=colors,edgecolors='k',linewidths=.12,alpha=.72);ax.add_collection3d(collection)
 xyz=np.concatenate(polygons);ax.set_xlim(xyz[:,0].min(),xyz[:,0].max());ax.set_ylim(xyz[:,2].min(),xyz[:,2].max());ax.set_zlim(xyz[:,1].min(),xyz[:,1].max());ax.view_init(22,-55);ax.set_xlabel('x');ax.set_ylabel('z');ax.set_zlabel('y');ax.set_title(f'Top {len(polygons)} finite plane polygons')
 ax=fig.add_subplot(122);collection=Poly3DCollection([np.c_[q[:,0],q[:,2],np.zeros(len(q))] for q in polygons],facecolors=colors,edgecolors='k',linewidths=.12,alpha=.72);ax3=fig.add_axes(ax.get_position(),projection='3d');ax.remove();ax3.add_collection3d(collection);ax3.set_xlim(xyz[:,0].min(),xyz[:,0].max());ax3.set_ylim(xyz[:,2].min(),xyz[:,2].max());ax3.set_zlim(-1,1);ax3.view_init(90,-90);ax3.set_title('Top view (finite boundaries)');ax3.set_axis_off()
 fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,bbox_inches='tight')
if __name__=='__main__':main()
