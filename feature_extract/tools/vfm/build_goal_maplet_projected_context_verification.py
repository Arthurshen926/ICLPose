"""Freeze projected RADIO verification controls on two existing candidate pairs."""
import argparse,json,time
from pathlib import Path
from feature_extract.tools.vfm.complete_scene_inputs import radio_manifests
import numpy as np
import torch
from feature_extract.tools.vfm.projected_surface_context import project_field,score_pair,phase_averaged_field,foreground_visibility
from feature_extract.tools.vfm.refine_goal_maplet_plane_uv_pose_by_view_geometry import _load_atlas
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _records,_radio
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--context-map',type=Path);p.add_argument('--observed_surface',action='store_true');p.add_argument('--phase_average',action='store_true');p.add_argument('--depth_visibility',choices=['none','observed','shuffled'],default='none');p.add_argument('--include_centered',action='store_true');p.add_argument('--splits',nargs='+',default=['seq10','shard0','shard1','shard2','shard3']);a=p.parse_args();
 if a.depth_visibility!='none' and a.phase_average:raise ValueError('depth visibility and phase average are separate experiments')
 b=a.base;a.output.mkdir(parents=True,exist_ok=False)
 ap=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';atlas,am=_load_atlas(ap);pp=b/'stmarys_chart_local_radio_projection_64d_v2.npz';mp=a.context_map or b/'native_context_v278/map.npz'
 with np.load(pp) as z:weight=torch.as_tensor(z['weight'],device=a.device)
 with np.load(mp) as z:mm=json.loads(z['metadata_json'].item());ma={k:z[k] for k in z.files if k!='metadata_json'}
 if arrays_sha256(ma)!=mm['arrays_sha256'] or canonical_json_sha256({k:v for k,v in mm.items() if k!='content_sha256'})!=mm['content_sha256'] or mm.get('query_pose_or_ground_truth_used_for_retrieval') is not False:raise ValueError('mapping-source authority differs')
 if mm['native_atlas_sha256']!=file_sha256(ap):raise ValueError('mapping/query-exclusion authority differs')
 used=set(mm.get('offline_mapping_source_names',[]));manifests=radio_manifests(b);records=_records(manifests)
 world=atlas['world_points'].astype(float);features=normalise(atlas['radio_features']);directions=normalise(atlas['prototype_view_direction_world']);source={str(x):file_sha256(x) for x in [ap,pp,mp,Path(__file__),Path(__file__).with_name('projected_surface_context.py')]+manifests}
 exclusion=None
 if mm.get('artifact_type')=='goal_maplet_native_crossroute_region_library_v1':
  route=mm['excluded_mapping_route']
  if a.splits!=[route]:raise ValueError('cross-route projection requires exactly its excluded route')
  lp=b/'native_fine_v264/mapping_lineage.npz';fp=b/'native_fine_v264/readout/map.npz'
  with np.load(fp) as z:fm=json.loads(z['metadata_json'].item())
  if fm['atlas_sha256']!=file_sha256(ap) or fm['lineage_sha256']!=file_sha256(lp):raise ValueError('projection source lineage differs')
  with np.load(lp) as z:sn=z['source_names'].astype(str);identity=z['prototype_source_and_cell'][:,0]
  from feature_extract.tools.vfm.refinement_source_contract import validate_crossroute_support
  keep=np.flatnonzero(np.asarray([n.split('__')[0] for n in sn])[identity]!=route)
  exclusion=validate_crossroute_support([route+'__mapping'],keep,ma['geometry_member_rows'],identity,sn,route)
  if len(identity)!=len(world):raise ValueError('projection atlas/source rows differ')
  world,features,directions=world[keep],features[keep],directions[keep]
  used=set(sn[identity[keep]]);source.update({str(lp):file_sha256(lp),str(fp):file_sha256(fp)})
 for split in a.splits:
  started=time.perf_counter();dirs={'pnp':b/'diverse_candidate_retention_v307'/f'{split}_diverse_support_consensus','refined':b/'diverse_refined_retention_v309'/f'{split}_diverse_refined_consensus'};pairs={};corrs={}
  for label,d in dirs.items():
   cmd=dict(json.loads((d/'protocol.json').read_text())['commands'])['selected'];paths=[Path(cmd[cmd.index('--'+key)+1]) for key in ['primary_pose','alternate_pose']];pairs[label]=[(path,*_load_pose_candidate(path)) for path in paths]
   source[str(d/'protocol.json')]=file_sha256(d/'protocol.json')
   for path,_,_ in pairs[label]:source[str(path)]=file_sha256(path)
  names=pairs['pnp'][0][1]['names'].astype(str)
  if set(names)&used:raise ValueError('query/source overlap')
  if exclusion and {n.split('__')[0] for n in names}!={mm['excluded_mapping_route']}:raise ValueError('query route differs')
  for pair in pairs.values():
   for path,arr,meta in pair:
    if not np.array_equal(names,arr['names'].astype(str)):raise ValueError('pose query alignment differs')
  # Exclude the union of every possible base and alternate generator token.
  lineage=json.loads((dirs['pnp']/'base_lineage.json').read_text());cpaths=[Path(lineage[k]) for k in ['primary_correspondence','alternate_correspondence']]+[b/'native_hybrid_transfer_v289/risk_stop'/f'{split}_appearance.npz']
  for cp in cpaths:
   c,cm=_load(cp);corrs[str(cp)]=c;source[str(cp)]=file_sha256(cp)
   if not np.array_equal(names,c['names'].astype(str)):raise ValueError('correspondence query alignment differs')
  cor=list(corrs.values());K=cor[0]['camera_matrices'];rad=cor[0]['radial_k1']
  if not all(np.array_equal(K,c['camera_matrices']) and np.array_equal(rad,c['radial_k1']) for c in cor):raise ValueError('camera mismatch')
  rows={label:[] for label in pairs};chosen={label:{} for label in pairs}
  if a.depth_visibility!='none':
   from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
   rcmd=dict(json.load(open(dirs['pnp']/'protocol.json'))['commands'])['alternate_render'];moge_dir=Path(rcmd[rcmd.index('--moge3_query')+1])
  if a.observed_surface:
   from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
   from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support
   pcmd=dict(json.load(open(b/'native_hybrid_mainline_v290'/f'{split}_risk_stop/protocol.json'))['commands'])['moge'];plane_dir=Path(pcmd[pcmd.index('--query_plane_dir')+1])
  def project(pose,camera,k):
   if a.phase_average:return phase_averaged_field(world,features,directions,pose,camera,k)
   if a.depth_visibility!='none':return project_field(world,features,directions,pose,camera,k,return_depth=True)
   f,m=project_field(world,features,directions,pose,camera,k);return f,m,None
  for i,n in enumerate(names):
   with torch.no_grad():q=normalise((torch.as_tensor(_radio(n,records),device=a.device)@weight.T).cpu().numpy()).reshape(36,64,-1)
   gener=np.unique(np.concatenate([c['query_tokens'][c['correspondence_offsets'][i]:c['correspondence_offsets'][i+1]] for c in cor]))
   query_mask=None
   if a.observed_surface:
    plane,pm=QueryPlaneRegions.load_npz(plane_dir/n)
    if pm.get('uses_pose_or_ground_truth') is not False:raise ValueError('query plane is not pose-free')
    source[str(plane_dir/n)]=file_sha256(plane_dir/n);query_mask=np.zeros(2304,bool)
    for rid in range(len(plane.pixel_counts)):
     ts,vs=_region_token_support(plane.labels,rid);query_mask[ts[vs>=.75]]=True
    query_mask=query_mask.reshape(36,64)
   if a.depth_visibility!='none':
    qp,qn,qv=moge_tokens(moge_dir/n);source[str(moge_dir/n)]=file_sha256(moge_dir/n);qd=qp[:,2].copy()
    if a.depth_visibility=='shuffled':
     ids=np.flatnonzero(qv);qd[ids]=qd[np.random.default_rng(314).permutation(ids)]
    qd=qd.reshape(36,64);qv=qv.reshape(36,64)
   field0,mask0,u0=project(pairs['pnp'][0][1]['pose_w2c'][i],K[i],float(rad[i]))
   for label,pair in pairs.items():
    if not np.array_equal(pair[0][1]['pose_w2c'][i],pairs['pnp'][0][1]['pose_w2c'][i],equal_nan=True):raise ValueError('base poses differ')
    f,m,u=project(pair[1][1]['pose_w2c'][i],K[i],float(rad[i]));vm=query_mask;visibility=None
    if a.depth_visibility!='none':
     vm,visibility=foreground_visibility([u0,u],[mask0,m],qd,qv)
     if query_mask is not None:vm &=query_mask
    result=score_pair(q,[field0,f],[mask0,m],gener,query_mask=vm,instability=[u0,u] if a.phase_average else None,include_centered=a.include_centered)
    rows[label].append(dict(name=n,depth_visibility=visibility,base_projected_fraction=float(mask0.mean()),alternate_projected_fraction=float(m.mean()),generator_token_fraction=float(len(gener)/2304),arms=result))
    for arm,r in result.items():chosen[label].setdefault(arm,[]).append(r['selected'])
   if (i+1)%20==0:print(split,i+1,flush=True)
  for label,pair in pairs.items():
   for arm,choices in chosen[label].items():
    ch=np.asarray(choices,np.int8);arr=dict(names=pair[0][1]['names'],pose_w2c=np.stack([v[1]['pose_w2c'] for v in pair],axis=1)[np.arange(len(ch)),ch],usable=np.stack([v[1]['usable'] for v in pair],axis=1)[np.arange(len(ch)),ch],selected_branch=ch)
    meta=dict(artifact_type='goal_maplet_projected_surface_context_selection_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,mapping_source_exclusion=exclusion,source_rgb_used=False,source_sha256=source,arm=arm,pair=label,token_budget=128,map_visibility='approximate atlas z-buffer; 0.25m band; geometric nearest token center then anonymous view direction; not full surface visibility',selection='strict higher mean cosine, ties/empty retain primary',not_independent_of_generator_backbone=True,depth_visibility=a.depth_visibility,depth_visibility_scope="common-domain per-pose median scale; fixed 1.5 foreground ratio; not calibrated",centered_structure_controls=a.include_centered,observed_query_surface_mask=a.observed_surface,phase_averaged=a.phase_average,phase_uncertainty_scope="five +/-1 pixel perturbations, sensitivity proxy only" if a.phase_average else None)
    meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(a.output/f'{split}_{label}_{arm}.npz',**arr,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
   (a.output/f'{split}_{label}_evidence.json').write_text(json.dumps(rows[label]))
  (a.output/f'{split}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,queries=len(names),scope='cached feature loading/projection/scoring, not RGB-to-pose runtime')))
 (a.output/'complete.json').write_text(json.dumps(dict(query_labels_used=False,source_sha256=source,controls=8 if a.include_centered else 5,pairs=2)))
if __name__=='__main__':main()
