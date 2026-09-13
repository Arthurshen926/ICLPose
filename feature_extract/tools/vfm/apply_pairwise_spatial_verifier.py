from pathlib import Path
import json,joblib,numpy as np
from scipy.special import expit
from feature_extract.tools.vfm.spatial_pose_evidence import evidence
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'spatial_verifier_v404';fit=joblib.load(r/'pairwise_verifier.joblib');model=fit['model'];meta=fit['metadata'];assert not meta['query_routes_read'];scale=meta['scale'];threshold=meta['threshold'];sources={str(r/'pairwise_verifier.joblib'):file_sha256(r/'pairwise_verifier.joblib')}
with np.load(b/'native_fine_v264/readout/map.npz') as f:world=f['world_points']
for folder,rawfolder,arm,label in [('overlap_lod_v402/lod_fixed_final','overlap_lod_v402/lod_fixed','mnn','mnn_seed1'),('overlap_lod_v402/lod_seed2_final','overlap_lod_v402/lod_seed2','mnn','mnn_seed2'),('overlap_weighted_v403/lod_final','overlap_weighted_v403/lod','weighted_mnn','soft_seed1'),('overlap_weighted_seed2_v403/lod_final','overlap_weighted_seed2_v403/lod','weighted_mnn','soft_seed2')]:
 o=r/(label+'_pairwise');o.mkdir(exist_ok=True)
 for s in ['seq10','shard0','shard1','shard2','shard3']:
  dest=o/f'{s}_verified.npz';assert not dest.exists()
  bp=b/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';ap=b/rawfolder/f'{s}_{arm}_audit.json';rp=b/folder/f'{s}_{arm}_audit.json'
  with np.load(bp) as f:names=f['names'];base=f['pose_w2c']
  with np.load(cp) as f:Ks=f['camera_matrices'];ks=f['radial_k1'];assert np.array_equal(names,f['names'])
  raw=json.load(open(ap));ref=json.load(open(rp));poses=[];accepted=[];audit=[]
  for i,name in enumerate(names.astype(str)):
   assert raw[i]['name']==ref[i]['name']==name
   regs=raw[i]['regions'];ids=np.concatenate([a['prototype_rows'] for a in regs]).astype(int) if regs else np.array([],int);t=np.concatenate([a['selected_tokens'] for a in regs]).astype(int) if regs else np.array([],int);xy=np.concatenate([np.array(a['query_pixels']).reshape(-1,2) for a in regs]) if regs else np.empty((0,2))
   pool=[base[i]]+[np.array(p) for p in ref[i]['refined_candidate_poses'] if p is not None];features=np.array([evidence(p,world[ids],t,xy,Ks[i],float(ks[i])) for p in pool]);prob=expit(scale*model.decision_function(features-features[:1]));j=int(prob.argmax());take=j>0 and prob[j]>=threshold and separated_mode(base[i],pool[j]);poses.append(pool[j] if take else base[i]);accepted.append(take);audit.append(dict(name=name,probabilities=prob.tolist(),features=features.tolist(),chosen=j,accepted=bool(take)))
  arrays=dict(names=names,pose_w2c=np.array(poses),accepted=np.array(accepted));m=dict(artifact_type='mapping_calibrated_pairwise_spatial_verifier_v404',arrays_sha256=arrays_sha256(arrays),query_pose_or_ground_truth_read=False,mode_separation_translation_m=.5,mode_separation_rotation_deg=3,source_sha256={**sources,**{str(p):file_sha256(p) for p in [bp,cp,ap,rp,Path(__file__),Path('feature_extract/tools/vfm/spatial_pose_evidence.py')]}},calibrated_on='mapping seq7 only; finite sample',model_metadata=meta);m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(m)));(o/f'{s}_audit.json').write_text(json.dumps(audit));print(label,s,flush=True)
