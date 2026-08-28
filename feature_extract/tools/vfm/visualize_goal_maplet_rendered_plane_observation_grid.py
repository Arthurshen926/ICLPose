from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from feature_extract.vfm.localization_goal_maplet.rendered_view_planes import RenderedPlaneObservations
def main():
 p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=sorted(a.observations.glob('*.planes.npz'));records=[]
 for path in paths:
  observation,_=RenderedPlaneObservations.load_npz(path);records.append((float(np.mean(observation.labels>=0)),path,observation))
 records.sort(key=lambda x:x[0]);indices=np.linspace(0,len(records)-1,12,dtype=int);rng=np.random.default_rng(5);palette=rng.uniform(.1,1,(64,3))
 fig,axes=plt.subplots(4,6,figsize=(18,11),dpi=160)
 for column,index in enumerate(indices):
  coverage,path,observation=records[index];source=a.contributors/path.name.replace('.planes.npz','.npz')
  with np.load(source,allow_pickle=False) as d:depth=np.asarray(d['dominant_depth'])
  color=np.zeros((*observation.labels.shape,3));valid=observation.labels>=0;color[valid]=palette[observation.labels[valid]%64]
  row=(column//6)*2;col=column%6;axes[row,col].imshow(depth,cmap='turbo',vmin=np.nanpercentile(depth[depth>0],2),vmax=np.nanpercentile(depth[depth>0],98));axes[row,col].set_title(path.name.split('.png')[0],fontsize=7)
  axes[row+1,col].imshow(color);axes[row+1,col].set_title(f'{len(observation.normals_world)} planes, cover {coverage:.1%}',fontsize=7)
 for ax in axes.flat:ax.axis('off')
 fig.suptitle('2DGS rendered depth and sequential-RANSAC planes (coverage quantiles, not cherry-picked)');fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,bbox_inches='tight')
if __name__=='__main__':main()
