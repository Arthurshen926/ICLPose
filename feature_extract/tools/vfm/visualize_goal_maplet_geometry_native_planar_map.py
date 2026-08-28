"""Visual audit for a geometry-native planar map."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--physical_map',type=Path,required=True)
    parser.add_argument('--planar_map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    physical=GoalMapletPhysicalMap.load_npz(args.physical_map)
    planar=GeometryNativePlanarMap.load_npz(args.planar_map,primitive_count=physical.primitive_ids.size)
    owner=np.full(physical.primitive_ids.size,-1,np.int64)
    for plane in range(planar.plane_ids.size):
        lo,hi=int(planar.member_offsets[plane]),int(planar.member_offsets[plane+1])
        owner[planar.member_primitive_rows[lo:hi]]=plane
    rank=np.argsort(planar.support_area_m2)[::-1]
    display_rank=np.full(planar.plane_ids.size,-1,np.int64); display_rank[rank[:256]]=np.arange(min(256,rank.size))
    selected=owner>=0
    color=np.full((physical.primitive_ids.size,4),[.72,.72,.72,.08])
    cmap=plt.get_cmap('turbo')
    major=selected&(display_rank[owner.clip(min=0)]>=0)
    color[major]=cmap((display_rank[owner[major]]%64)/63.0); color[major,3]=.75
    minor=selected&~major; color[minor]=[.35,.55,.82,.13]
    rng=np.random.default_rng(0)
    sample=np.sort(rng.choice(physical.primitive_ids.size,min(180000,physical.primitive_ids.size),replace=False))
    p=physical.primitive_centers[sample]
    fig=plt.figure(figsize=(18,11),dpi=170)
    ax=fig.add_subplot(221,projection='3d')
    ax.scatter(p[:,0],p[:,2],p[:,1],c=color[sample],s=.25,depthshade=False)
    ax.set_title('Top-256 geometry-native planes (other surfaces faint)'); ax.set_xlabel('world x');ax.set_ylabel('world z');ax.set_zlabel('world y')
    ax.view_init(25,-55)
    ax=fig.add_subplot(222)
    ax.scatter(p[:,0],p[:,2],c=color[sample],s=.35)
    ax.set_aspect('equal');ax.set_title('Top view: finite connected plane instances');ax.set_xlabel('world x');ax.set_ylabel('world z')
    ax=fig.add_subplot(223)
    bins=np.geomspace(max(float(planar.support_area_m2.min()),1e-4),max(float(planar.support_area_m2.max()),1e-3),50)
    ax.hist(planar.support_area_m2,bins=bins,color='#4271ae');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('summed supported area (m²)');ax.set_ylabel('plane count')
    ax.set_title('Plane support-area distribution')
    ax=fig.add_subplot(224)
    ax.scatter(planar.boundary_area_m2,planar.residual_rms_m,s=np.clip(planar.member_counts/20,2,30),alpha=.25,c=planar.normal_cosine_p10,cmap='viridis')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('convex boundary area (m²)');ax.set_ylabel('plane RMS residual (m)');ax.set_title('Geometry quality (color = normal cosine p10)')
    stats={
        'planes':int(planar.plane_ids.size),'assigned_fraction':float(planar.member_primitive_rows.size/physical.primitive_ids.size),
        'area_gt_1m2':int(np.sum(planar.support_area_m2>=1.0)),'area_gt_5m2':int(np.sum(planar.support_area_m2>=5.0)),
        'area_gt_10m2':int(np.sum(planar.support_area_m2>=10.0)),'median_area_m2':float(np.median(planar.support_area_m2)),
        'median_rms_m':float(np.median(planar.residual_rms_m)),'p90_rms_m':float(np.quantile(planar.residual_rms_m,.9)),
    }
    fig.suptitle('St Mary’s Church 2DGS → geometry-native planar map\n'+json.dumps(stats),fontsize=12)
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,bbox_inches='tight')
    print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
