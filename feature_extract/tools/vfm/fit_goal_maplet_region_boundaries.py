"""Fit physical radii/member references from mapping-only frozen unit outcomes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.vfm.localization_goal_maplet.learned_region_boundaries import optimize_boundaries,boundary_value


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['candidates','reference','model','small','large','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.candidates) as z:world=z['prototype_world']
 with np.load(a.reference) as z:sources=z['source_image'];ids=z['region_ids'];centers=z['centers']
 model=json.loads(a.model.read_text())
 if model['radius']!=6 or model['map_world_sha256']!=hashlib.sha256(world.tobytes()).hexdigest():raise ValueError('reference model map/radius differs')
 e=np.array(model['training_errors']);values=[None,np.exp(-e[:,0]/.5-e[:,1]/5),None]
 for k,path in [(0,a.small),(2,a.large)]:
  with np.load(path/'frozen.npz') as z:
   if float(z['radius'])!=[3.,6.,9.][k]:raise ValueError('boundary candidate radius differs')
   if not np.array_equal(sources,z['source_image']) or not np.array_equal(ids,z['region_ids']) or not np.array_equal(centers,z['centers']):raise ValueError('training inventory differs')
  with np.load(path/'labels.npz') as z:values[k]=z['utility']
 _,inv=np.unique(sources,return_inverse=True);q=np.zeros((len(np.unique(sources)),len(centers),3));observed=np.zeros(q.shape[:2],bool)
 for k,value in enumerate(values):q[inv,ids,k]=value
 observed[inv,ids]=True;radii=np.array([3.,6.,9.]);tree=cKDTree(world);members=[[np.asarray(g,int) for g in tree.query_ball_point(centers,r)] for r in radii];cost=np.array([[len(members[k][i]) for k in range(3)] for i in range(len(centers))]);budget=int(cost[:,1].sum());result={}
 choices={}
 for objective in ['independent','set']:
  choice,trace=optimize_boundaries(q,observed,cost,budget,objective);choices[objective]=choice;result[objective]={'trace':trace,'initial_objective':boundary_value(q,observed,np.ones(len(centers),int),objective),'final_objective':boundary_value(q,observed,choice,objective)}
 rng=np.random.default_rng(260915);active=np.flatnonzero(observed.any(0));permuted=choices['set'].copy()
 for _ in range(10000):
  permuted[active]=rng.permutation(choices['set'][active])
  if cost[np.arange(len(centers)),permuted].sum()<=budget:break
 else:raise RuntimeError('no budget-feasible shuffled control')
 choices['shuffled']=permuted
 for name,choice in choices.items():
  groups=[members[k][i] for i,k in enumerate(choice)];flat=np.concatenate(groups);offsets=np.r_[0,np.cumsum([len(g) for g in groups])]
  np.savez_compressed(a.output/(name+'.npz'),centers=centers,radii=radii[choice],prototype_rows=flat,offsets=offsets,world_sha256=np.asarray(hashlib.sha256(world.tobytes()).hexdigest()),training_images=np.asarray(model['training_images']))
  result.setdefault(name,{}).update(references=len(flat),budget=budget,radius_histogram={str(r):int(np.sum(radii[choice]==r)) for r in radii},changed_centers=int(np.sum(choice!=1)))
 np.savez_compressed(a.output/'training_proxy.npz',quality=q,observed=observed,cost=cost)
 (a.output/'report.json').write_text(json.dumps({'scope':__doc__,'models':result,'set_objective':'mean over mapping images of max frozen standalone regional pose utility; proxy pool upper bound, not final deployed set reward','unevaluated':'no utility evidence assigned; not labeled as negative; unsupervised centers kept at 6m','boundary_family':'fixed centers, per-center learned discrete metric radius 3/6/9m; members are all original points within learned radius','freeform_boundaries_learned':False},indent=2))

if __name__=='__main__':main()
