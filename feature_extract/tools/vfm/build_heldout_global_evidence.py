from pathlib import Path
import argparse,json,time
import numpy as np
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--mapping',action='store_true');p.add_argument('--proposal-root',type=Path,action='append',default=[]);p.add_argument('--output',type=Path);a=p.parse_args();s=a.split;b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');o=a.output or b/'heldout_evidence_v405/banks';o.mkdir(exist_ok=True,parents=True);dest=o/(('mapping_' if a.mapping else '')+s+'.npz');assert not dest.exists();maps,meta,sources=load_map(b)
cp=b/'native_hybrid_mapping_v286'/s/'corr.npz' if a.mapping else b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz'
with np.load(cp) as f:names=f['names'].astype(str);Ks=f['camera_matrices'];ks=f['radial_k1']
if a.mapping:
 mask=cp.with_name('mask.npz')
 with np.load(mask) as f:eligible=f['eligible'];assert json.loads(f['metadata_json'].item())['excluded_mapping_route']==s
 sources[str(mask)]=file_sha256(mask);md=b/'native_region_training_v276/moge'/s
else:
 eligible=np.ones(len(maps['world']),bool);cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1]);original=json.load(open(b/'overlap_lod_v402/lod_fixed'/f'{s}_mnn_audit.json'))
trigger=np.ones(len(names),bool)
if a.proposal_root:
 if a.mapping:raise ValueError('lazy proposal roots are query-only')
 trigger=np.zeros(len(names),bool)
 for root in a.proposal_root:
  pp=root/f'{s}_verified.npz'
  with np.load(pp) as f:
   if not np.array_equal(f['names'].astype(str),names):raise ValueError('proposal query mismatch')
   trigger|=f['accepted']
  sources[str(pp)]=file_sha256(pp)
ids=np.flatnonzero(eligible)
with np.load(b/'stmarys_chart_local_radio_projection_64d_v2.npz') as f:pm=json.loads(f['metadata_json'].item())['validation_learned_projection'];threshold=.5*(pm['positive_cosine_mean']+pm['same_plane_negative_cosine_mean'])
rows=[];tokens=[];pixels=[];offset=[0];audits=[]
for i,n in enumerate(names):
 if not trigger[i]:
  offset.append(len(rows));audits.append(dict(name=n,skipped_no_proposal=True,excluded_tokens=[],verification_tokens=[],retained_tokens=[],search_seconds=0.));continue
 cache=b/('adaptive_memory_v234/fine_cache' if a.mapping else 'native_fine_v264/query_cache')/n;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];_,_,qv=moge_tokens(md/n);q=normalise(grids[0].reshape(2304,64));used=diverse_tokens(q,qv)
 if not a.mapping:assert np.array_equal(used,np.array(original[i]['sampled_tokens']))
 allowed=qv.copy();allowed[used]=False;new=diverse_tokens(q,allowed);assert not np.intersect1d(used,new).size;start=time.perf_counter()
 if len(new):
  sim=q[new]@maps['coarse_map'][ids].T;rank=sim.argmax(1);mutual=sim.argmax(0)[rank]==np.arange(len(new));keep=mutual&(sim[np.arange(len(new)),rank]>=threshold);t=new[keep];pr=ids[rank[keep]];xy=np.c_[t%64*4+1.5,t//64*4+1.5];xy=fine_coordinates(grids,xy,pr[:,None],maps)[:,0]
 else:t=np.array([],int);pr=np.array([],int);xy=np.empty((0,2))
 rows.extend(pr);tokens.extend(t);pixels.extend(xy);offset.append(len(rows));audits.append(dict(name=n,excluded_tokens=used.tolist(),verification_tokens=new.tolist(),retained_tokens=t.tolist(),search_seconds=time.perf_counter()-start));sources[str(cache)]=file_sha256(cache);sources[str(md/n)]=file_sha256(md/n)
 if (i+1)%10==0:print(s,i+1,flush=True)
arrays=dict(names=names,offsets=np.array(offset),prototype_rows=np.array(rows,int),query_tokens=np.array(tokens,int),pixels=np.array(pixels).reshape(-1,2),camera_matrices=Ks,radial_k1=ks);sources[str(cp)]=file_sha256(cp);sources[str(Path(__file__))]=file_sha256(Path(__file__));metadata=dict(lazy_proposal_trigger=bool(a.proposal_root),trigger_count=int(trigger.sum()),query_pose_or_ground_truth_read=False,candidate_pose_used=False,proposal_audit_read_for_token_check=not a.mapping,excluded_proposal_tokens=True,independent_image_or_encoder=False,global_exact_mnn=True,threshold=threshold,source_sha256=sources,arrays_sha256=arrays_sha256(arrays));np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(metadata)));dest.with_suffix('.json').write_text(json.dumps(audits))
