"""Mapping-only explicit point membership from reliable candidate-independent labels."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

def member_evidence(source,prototype,labels,n):
 pos=np.zeros(n,int);neg=np.zeros(n,int)
 for s in np.unique(source):
  rows=np.flatnonzero(source==s);p=np.unique(prototype[rows[labels[rows]==1]]);q=np.setdiff1d(np.unique(prototype[rows[labels[rows]==0]]),p)
  pos[p]+=1;neg[q]+=1
 seen=(pos+neg)>0;upper_mass=beta.sf(.5,1+pos,1+neg)
 drop=seen&((pos+neg)>=3)&(upper_mass<.05)
 return drop,seen,pos,neg

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;a.output.mkdir(exist_ok=False);cp=b/'moge_full_input_v218.npz'
 with np.load(cp) as z:c={k:z[k] for k in z.files}
 with np.load(b/'moge_depth_extended_labels_v221.npz') as z:
  if str(z['frozen_candidate_sha256'])!=file_sha256(cp):raise ValueError('label lineage')
  labels=z['independent_metric_labels']
 with np.load(b/'full_pool_boundaries_v245/maps/fixed.npz') as z:m={k:z[k] for k in z.files}
 if str(m['world_sha256'])!=hashlib.sha256(c['prototype_world'].tobytes()).hexdigest():raise ValueError('map geometry lineage')
 if not all(str(n).startswith('seq9__') for n in m['training_images']):raise ValueError('mapping-only training required')
 keep=c['homography_keep'];drop,seen,pos,neg=member_evidence(c['source_image'][keep],c['prototype_rows'][keep],labels[keep],len(c['prototype_world']))
 groups=[m['prototype_rows'][m['offsets'][i]:m['offsets'][i+1]] for i in range(len(m['centers']))];learned=[g[~drop[g]] for g in groups];rng=np.random.default_rng(260925);random=[]
 for g,new in zip(groups,learned):
  remove=rng.choice(g[seen[g]],len(g)-len(new),replace=False);random.append(g[~np.isin(g,remove)])
 for name,gs in [('learned',learned),('shuffled',random)]:
  data=dict(m);data.update(prototype_rows=np.concatenate(gs),offsets=np.r_[0,np.cumsum([len(g) for g in gs])],member_policy=np.asarray('explicit_subset'));np.savez_compressed(a.output/(name+'.npz'),**data)
 np.savez_compressed(a.output/'evidence.npz',drop=drop,seen=seen,positive_views=pos,negative_views=neg)
 (a.output/'report.json').write_text(json.dumps({'scope':__doc__,'reliable_labels':'independent_metric_labels; unknown and ambiguous ignored','duplicate_rule':'at most one vote per prototype per mapping image; any positive overrides negative alternatives','decision':'Beta(1+positive_views,1+negative_views) mass above 0.5 below 0.05, at least 3 observed images','unobserved_members_preserved':True,'map_geometry_unchanged':True,'non_spherical_members_inside_fixed_6m_envelope':True,'training_route':'seq9','observed_prototype_fraction':float(seen.mean()),'removed_prototype_fraction':float(drop.mean()),'references_before':sum(map(len,groups)),'references_after':sum(map(len,learned)),'random_control_exact_per_region_capacity':all(len(x)==len(y) for x,y in zip(learned,random)),'limitations':['candidate-conditioned reliability; not causal pose-set utility','same-map pseudo-supervision','centers and region count fixed','not a learned free-boundary neural network']},indent=2))
if __name__=='__main__':main()
