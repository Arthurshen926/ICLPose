"""Optimize full-pool final selected utility under the original member-reference cap."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.fit_goal_maplet_selection_aware_boundaries import selected_quality


def optimize(q,support,counts,residual,gc,gr,gq,cost,max_rounds=30):
 choice=np.ones(q.shape[1],int);budget=int(cost[:,1].sum());active=np.flatnonzero((support>0).any(axis=(0,2)));trace=[]
 def value(ch):return selected_quality(ch,q,support,counts,residual,gc,gr,gq)
 initial=value(choice)
 for iteration in range(max_rounds):
  used=int(cost[np.arange(len(choice)),choice].sum());best=value(choice)+1e-12;winner=None
  actions=[(int(i),k,int(cost[i,k]-cost[i,choice[i]])) for i in active for k in range(3) if k!=choice[i]]
  proposals=[(x,) for x in actions if used+x[2]<=budget]
  proposals.extend((x,y) for x in actions if x[2]>0 for y in actions if y[2]<0 and x[0]!=y[0] and used+x[2]+y[2]<=budget)
  for acts in proposals:
   new=choice.copy()
   for i,k,_ in acts:new[i]=k
   v=value(new)
   if v>best:best,winner=v,new
  if winner is None:break
  choice=winner;trace.append({'value':best,'references':int(cost[np.arange(len(choice)),choice].sum())});print('fit full pool',iteration,best,flush=True)
 return choice,{'initial_value':initial,'final_value':value(choice),'trace':trace,'budget':budget}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 with np.load(a.training) as z:d={k:z[k] for k in z.files}
 with np.load(a.candidates) as z:w=z['prototype_world']
 tree=cKDTree(w);members=[tree.query_ball_point(d['centers'],r) for r in [3.,6.,9.]];cost=np.array([[len(members[k][j]) for k in range(3)] for j in range(len(d['centers']))])
 # Even the all-small map has enough eligible regions: no omitted plane fallback.
 if np.any((d['support'][:,:,0]>0).sum(1)<4):raise ValueError('training requires explicit plane fallback cache')
 choice,report=optimize(d['quality'],d['support'],d['counts'],d['residual'],d['global_counts'],d['global_residual'],d['global_quality'],cost)
 choices={'learned':choice,'fixed':np.ones(len(choice),int)};active=np.flatnonzero((d['support']>0).any(axis=(0,2)))
 for seed in [260921,260922,260923]:
  rng=np.random.default_rng(seed);sh=choice.copy()
  for _ in range(10000):
   sh[active]=rng.permutation(choice[active])
   if cost[np.arange(len(sh)),sh].sum()<=report['budget']:break
  else:raise ValueError('no feasible shuffle')
  choices['shuffled'+str(seed)]=sh.copy()
 report['maps']={}
 for name,ch in choices.items():
  groups=[members[k][j] for j,k in enumerate(ch)];r=np.array([3.,6.,9.])[ch]
  np.savez_compressed(a.output/(name+'.npz'),centers=d['centers'],radii=r,prototype_rows=np.concatenate(groups),offsets=np.r_[0,np.cumsum([len(x) for x in groups])],world_sha256=np.asarray(hashlib.sha256(w.tobytes()).hexdigest()),training_images=d['training_images'])
  report['maps'][name]={'references':sum(map(len,groups)),'radius_histogram':{str(v):int(np.sum(r==v)) for v in [3.,6.,9.]}}
 report['scope']='full eligible mapping pool; two shared regional seeds; final global LM + MoGe utility; single mapping route, no 438 consensus';report['covered_centers']=len(active);report['training_samples']=len(d['quality']);report['eligible_per_seed']=d['eligible'].sum(axis=(1,2)).tolist()
 (a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
