"""Pose-free region appearance retrieval followed by new within-region correspondences.

Fixed 6m map; source-specific anonymous context modes; no query candidate pool
is consulted during retrieval. This is a restricted configuration-readout probe.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _records,_radio
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import sector_descriptors,normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['base','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--retrieval_policy',choices=['appearance','random'],default='appearance')
 a=p.parse_args();b=a.base;a.output.mkdir(exist_ok=False);old=b/'adaptive_memory_v234';candidate=b/'memory_transfer_candidates_v224.npz'
 with np.load(candidate) as z:c={k:z[k] for k in z.files}
 with np.load(old/'topology.npz') as z:keys=z['prototype_keys'];assert np.array_equal(c['prototype_world'],z['prototype_world']);mapping_names=z['source_names'].astype(str)
 with np.load(old/'multiscale/memory_transfer_candidates_v224.npz') as z:names=z['source_names'].astype(str);assert str(z['candidate_sha256'])==file_sha256(candidate)
 assert np.array_equal(names[:len(mapping_names)],mapping_names)
 with np.load(old/'radius1_map.npz') as z:desc=normalise(z['descriptors'].astype(np.float32).reshape(-1,256));assert int(z['radius'])==1
 projpath=b/'stmarys_chart_local_radio_projection_64d_v2.npz';assert file_sha256(projpath)==json.load(open(old/'radius1.json'))['input_sha256']['radio_projection']
 with np.load(projpath) as z:proj=z['weight']
 with np.load(b/'learned_boundaries_v241/maps/selection_aware.npz') as z:centers=z['centers']
 groups=cKDTree(c['prototype_world']).query_ball_point(centers,6.)
 modes=[];mode_regions=[];mode_members=[]
 for rid,rows in enumerate(groups):
  rows=np.asarray(rows,int);sources=np.unique(keys[rows,1]);source_groups=[rows[keys[rows,1]==s] for s in sources]
  # Deterministic visibility support, never query GT; retain four appearance modes.
  order=np.argsort([-len(np.unique(keys[g,0])) for g in source_groups],kind='stable')[:4]
  for j in order:
   g=source_groups[j]
   if len(g)<6:continue
   assert not any(mapping_names[s].split('__')[0] in ['seq9','seq12','seq14'] for s in keys[g,1])
   modes.append(normalise(desc[g].mean(0)));mode_regions.append(rid);mode_members.append(g)
 modes=np.asarray(modes,np.float32);mode_regions=np.asarray(mode_regions);device='cuda';td=torch.as_tensor(desc,device=device);tm=torch.as_tensor(modes.T,device=device)
 records=_records([Path('output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json')]);sources=np.unique(c['source_image']);srcout=[];tokout=[];prout=[];simout=[];activated=[];seconds=[]
 for i,s in enumerate(sources):
  start=time.perf_counter();raw=_radio(names[s],records);grid=normalise(raw@proj.T).reshape(36,64,64);qd=normalise(sector_descriptors(grid,1).reshape(2304,256));tq=torch.as_tensor(qd,device=device)
  # Four fixed image quadrants are context queries; preserve all token evidence.
  qi=np.arange(2304).reshape(36,64);patches=[qi[y:y+18,x:x+32].ravel() for y in [0,18] for x in [0,32]]
  aggregate=normalise(np.asarray([qd[g].mean(0) for g in patches]));rs=(torch.as_tensor(aggregate,device=device)@tm).max(0).values.cpu().numpy();rank=np.argsort(-rs,kind='stable');chosen=[];seen=set()
  if a.retrieval_policy=='random':rank=np.random.default_rng(260924+int(s)).permutation(len(modes))
  for j in rank:
   if int(mode_regions[j]) in seen:continue
   seen.add(int(mode_regions[j]));chosen.append(int(j))
   if len(chosen)==4:break
  triples=[]
  for j in chosen:
   g=mode_members[j];sim=tq@td[g].T;val,ind=sim.max(1);back=sim.argmax(0);tokens=torch.arange(2304,device=device);mutual=back[ind]==tokens
   t=tokens[mutual].cpu().numpy();pr=g[ind[mutual].cpu().numpy()];v=val[mutual].cpu().numpy()
   triples.extend(zip(t.tolist(),pr.tolist(),v.tolist()))
  triples.sort(key=lambda x:(-x[2],x[0],x[1]));triples=triples[:1024]
  srcout.extend([s]*len(triples));tokout.extend(t for t,_,_ in triples);prout.extend(pr for _,pr,_ in triples);simout.extend(v for _,_,v in triples);activated.append([int(mode_regions[j]) for j in chosen]);seconds.append(time.perf_counter()-start)
  if (i+1)%20==0:print('region frontend',i+1,'/',len(sources),flush=True)
 new={'source_image':np.asarray(srcout,int),'query_token':np.asarray(tokout,int),'prototype_rows':np.asarray(prout,int),'prototype_world':c['prototype_world'],'prototype_plane':c['prototype_plane'],'homography_keep':np.ones(len(srcout),bool),'association_features':np.asarray(simout,np.float32)[:,None]}
 # Freeze new candidates before any pose evaluation. Keep union as separate arm.
 for name,d in [('regional',new),('union',None),('control',None)]:
  if name=='control':d={k:c[k] for k in new}
  if name=='union':
   d={k:c[k] for k in ['prototype_world','prototype_plane']}
   keep=c['homography_keep'];d.update({k:np.r_[c[k][keep],new[k]] for k in ['source_image','query_token','prototype_rows']});d['homography_keep']=np.ones(len(d['source_image']),bool);d['association_features']=np.r_[c['association_features'][keep,0],new['association_features'][:,0]][:,None]
  path=a.output/(name+'.npz');np.savez_compressed(path,**d);sha=file_sha256(path)
  path.with_suffix('.json').write_text(json.dumps({'scope':__doc__,'query_image_names':names[sources].tolist(),'query_GT_used':False,'candidate_file_sha256':sha},indent=2))
  np.savez_compressed(a.output/(name+'_scores.npz'),logits=np.zeros((len(d['query_token']),1)),arm_names=np.asarray(['unused']),candidate_sha256=np.asarray(sha))
  np.savez_compressed(a.output/(name+'_features.npz'),source_names=names,candidate_sha256=np.asarray(sha))
 np.savez_compressed(a.output/'map.npz',descriptors=modes,mode_regions=mode_regions,member_rows=np.concatenate(mode_members),offsets=np.r_[0,np.cumsum([len(x) for x in mode_members])],centers=centers)
 (a.output/'audit.json').write_text(json.dumps({'query_GT_used':False,'retrieval_policy':a.retrieval_policy,'new_correspondences_generated':True,'regional_rows':len(srcout),'active_regions':activated,'source_names':names[sources].tolist(),'seconds_per_query':seconds,'mode_count':len(modes),'map_descriptor_bytes':modes.nbytes,'limitations':['fixed quadrants for query context','fixed source-specific mean descriptors, not learned configuration matching','new-only and union costs differ; fixed PnP scoring budget does not imply equal frontend work','single scene reused developer routes']},indent=2))
if __name__=='__main__':main()
