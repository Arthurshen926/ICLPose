"""Freeze real v357-to-update pairs and reserved central-token evidence.

No pose labels are opened. Central token exclusion does not imply independent
encoder receptive fields or independent geometry. Full mapping replay is required.
"""
from pathlib import Path
import argparse,json,numpy as np
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.run_overlap_lod_frontend import fine_coordinates
from feature_extract.tools.vfm.local_precision_evidence import paired_features
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.refinement_source_contract import validate_crossroute_support
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domain',choices=['mapping','query'],required=True);p.add_argument('--split',required=True);a=p.parse_args();s=a.split;mapping=a.domain=='mapping'
 b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');root=b/'strong_local_precision_v414';out=root/a.domain/'pairs';out.mkdir(exist_ok=True,parents=True);dest=out/f'{s}_unlabelled.json';assert not dest.exists();rb=b/'strong_endpoint_replay_v413/replay_base' if mapping else b
 if mapping:
  seal=b/'strong_endpoint_replay_v413/endpoint_seal_before_labels.json';sm=json.load(open(seal));assert sm['full_mapping_counterpart_through_v357_complete']
 bp=rb/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';arms=['stage_plain_reg_all','stage_robust_reg_all'];candidates=[root/a.domain/'candidates'/f'{s}_{arm}.npz' for arm in arms]
 with np.load(bp) as z:names=z['names'].astype(str);base=z['pose_w2c'];usable=z['usable']
 if mapping:assert sm['routes'][s]['stages'][str(bp)]['file_sha256']==file_sha256(bp)
 proposed=[]
 for path in candidates:
  with np.load(path) as z:assert np.array_equal(names,z['names']);proposed.append(z['pose_w2c'])
 cp=[rb/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz',rb/'native_hybrid_fine_v301/readout'/f'{s}_reliability_rank32.npz', b/'strong_endpoint_replay_v412'/s/'surface/corr.npz' if mapping else b/f'learned64d_strict_{s}_h025_surfacecoord_homography_context_cell_v37_corr.npz']
 # Shard surface paths are obtained from their frozen consensus protocol.
 if not mapping:
  cmd=dict(json.load(open(b/'native_reliability_mainline_v275'/f'{s}_reliability_rank32_consensus/protocol.json'))['commands'])['sparse'];cp[2]=Path(cmd[cmd.index('--surface_correspondences')+1])
 corrs=[]
 for path in cp:
  with np.load(path) as z:c={k:z[k] for k in z.files if k!='metadata_json'};assert np.array_equal(names,c['names']);corrs.append(c)
 K=corrs[0]['camera_matrices'];rad=corrs[0]['radial_k1'];assert all(np.array_equal(K,c['camera_matrices']) and np.array_equal(rad,c['radial_k1']) for c in corrs)
 auditpaths=[rb/'direct_feature_consistency_v344'/f'{s}_multiscale_robust512_audit.json',rb/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all_audit.json']+[x.with_name(x.stem+'_audit.json') for x in candidates]
 audits=[json.load(open(x)) for x in auditpaths]
 guardpaths=[b/'strong_endpoint_replay_v413'/s/f'{kind}_guard'/f'{s}_refined_evidence.json' if mapping else b/folder/f'{s}_refined_evidence.json' for kind,folder in [('structure','projected_structure_v313'),('depth','depth_visibility_v319')]]
 guards=[json.load(open(x)) for x in guardpaths];maps,meta,sources=load_map(b)
 sources.update({str(x):file_sha256(x) for x in [bp,*candidates,*cp,*auditpaths,*guardpaths,Path(__file__),Path('feature_extract/tools/vfm/local_precision_evidence.py')]})
 eligible=np.ones(len(maps['world']),bool)
 if mapping:
  lp=b/'native_fine_v264/mapping_lineage.npz'
  with np.load(lp) as z:sn=z['source_names'].astype(str);owners=z['prototype_source_and_cell'][:,0]
  assert file_sha256(lp)==meta['lineage_sha256'];eligible=np.array([x.split('__')[0] for x in sn])[owners]!=s;sources[str(lp)]=file_sha256(lp);sources[str(seal)]=file_sha256(seal)
 ids=np.flatnonzero(eligible)
 with np.load(b/'stmarys_chart_local_radio_projection_64d_v2.npz') as z:pm=json.loads(z['metadata_json'].item())['validation_learned_projection'];threshold=.5*(pm['positive_cosine_mean']+pm['same_plane_negative_cosine_mean'])
 projection=b/'stmarys_chart_local_radio_projection_64d_v2.npz';sources[str(projection)]=file_sha256(projection)
 if mapping:md=b/'native_region_training_v276/moge'/s
 else:
  cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1])
 rows=[];tokens=[];pixels=[];offset=[0];pairs=[];exclusions=[]
 for i,n in enumerate(names):
  used=set()
  for c in corrs:
   lo,hi=c['correspondence_offsets'][i:i+2];used.update(c['query_tokens'][lo:hi].tolist())
  for audit in audits:assert audit[i]['name']==n;used.update(audit[i]['selected_tokens'])
  for audit in guards:assert audit[i]['name']==n;used.update(audit[i]['arms']['structure_guarded']['selected_tokens'])
  cache=b/('adaptive_memory_v234/fine_cache' if mapping else 'native_fine_v264/query_cache')/n;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];_,_,valid=moge_tokens(md/n);allowed=valid.copy();allowed[list(used)]=False;q=normalise(grids[0].reshape(2304,64));new=diverse_tokens(q,allowed);assert not used.intersection(new)
  if len(new):
   sim=q[new]@maps['coarse_map'][ids].T;rank=sim.argmax(1);keep=(sim.argmax(0)[rank]==np.arange(len(new)))&(sim[np.arange(len(new)),rank]>=threshold);t=new[keep];pr=ids[rank[keep]];xy=np.c_[t%64*4+1.5,t//64*4+1.5];xy=fine_coordinates(grids,xy,pr[:,None],maps)[:,0]
  else:t=np.array([],int);pr=np.array([],int);xy=np.empty((0,2))
  rows.extend(pr);tokens.extend(t);pixels.extend(xy);offset.append(len(rows));exclusions.append(dict(name=n,excluded_tokens=sorted(used),searched_tokens=new.tolist(),retained_tokens=t.tolist()))
  available=maps['available'][pr];pr=pr[available];t=t[available];xy=xy[available]
  for j,arm in enumerate(arms):
   candidate=proposed[j][i];valid_pair=bool(usable[i] and np.isfinite(candidate).all() and not separated_mode(base[i],candidate))
   f,count=paired_features(base[i],candidate,maps['world'][pr],xy,t,K[i],float(rad[i]),grids,[maps['coarse_map'][pr],maps['fine_map'][pr]]) if valid_pair else (np.zeros(12),0)
   pairs.append(dict(name=n,arm=arm,features=f.tolist(),support=int(count),valid_local_pair=valid_pair))
  sources[str(cache)]=file_sha256(cache);sources[str(md/n)]=file_sha256(md/n)
  if (i+1)%20==0:print(a.domain,s,i+1,flush=True)
 arr=dict(names=names,offsets=np.array(offset),prototype_rows=np.array(rows,int),query_tokens=np.array(tokens,int),pixels=np.array(pixels).reshape(-1,2),camera_matrices=K,radial_k1=rad)
 if mapping:validate_crossroute_support(names,arr['prototype_rows'],np.array([],int),owners,sn,s)
 bank=out/f'{s}_bank.npz';np.savez_compressed(bank,**arr,metadata_json=np.array(json.dumps(dict(arrays_sha256=arrays_sha256(arr),source_sha256=sources,query_pose_or_ground_truth_read=False,central_token_exclusion=True,independent_encoder_claim=False))))
 dest.write_text(json.dumps(dict(pairs=pairs,base_path=str(bp),candidate_paths=dict(zip(arms,map(str,candidates))),sources=sources,bank_sha256=file_sha256(bank),mapping_baseline_is_candidate_proxy=False,query_pose_or_ground_truth_read=False),indent=2));(out/f'{s}_token_exclusion.json').write_text(json.dumps(exclusions))


if __name__=='__main__':main()
