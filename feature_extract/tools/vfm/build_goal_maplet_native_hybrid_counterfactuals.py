"""Build mapping-only hybrid inventories; native identities and original prefixes stay fixed."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from feature_extract.tools.vfm.build_goal_maplet_native_region_augmentation import append_query_rows
from feature_extract.tools.vfm.build_goal_maplet_native_region_training_inventory import write
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import load_mapping_subtoken_head
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _region_token_support
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions
from feature_extract.tools.vfm.native_hybrid_region_value import pooled_additions,hybrid_features
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(exist_ok=False,parents=True)
 with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:atlas={k:z[k] for k in z.files if k!='metadata_json'}
 plane_ids=np.repeat(np.arange(len(atlas['plane_texel_offsets'])-1),np.diff(atlas['plane_texel_offsets']))
 with np.load(b/'native_fine_v264/readout/map.npz') as z:fm=json.loads(str(z['metadata_json']))
 head,hm=load_mapping_subtoken_head(b/'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz');head.eval();allnames=[];features=[]
 for route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']:
  d=o/route;d.mkdir();oldpath=b/'native_hybrid_mapping_v286'/route/'corr.npz';old,meta=_load(oldpath);extras=[[] for _ in range(9)]
  for i,name in enumerate(old['names'].astype(str)):
   with np.load(b/'native_counterfactual_v280/rows'/name) as z:raw={k:z[k] for k in z.files if k!='metadata_json'}
   off=raw['offsets'];rt=[raw['tokens'][lo:hi] for lo,hi in zip(off[:-1],off[1:])];rp=[raw['prototype_rows'][lo:hi] for lo,hi in zip(off[:-1],off[1:])];rs=[raw['scores'][lo:hi] for lo,hi in zip(off[:-1],off[1:])];lo,hi=old['correspondence_offsets'][i:i+2];ot=old['query_tokens'][lo:hi];op=old['prototype_atlas_row'][lo:hi];os=old['radio_match_score'][lo:hi]
   features.append(hybrid_features(ot,op,os,rt,rp,rs,raw['context_scores'],raw['centers']));allnames.append(name)
   grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/name,fm['projection_sha256']);assert ck==fm['checkpoint_sha256'];q=grids[0].reshape(2304,64).astype(np.float32)
   planes,pm=QueryPlaneRegions.load_npz(b/'native_region_training_v276/planes'/route/name);labels=np.full(2304,-1,int);visible=np.zeros(2304)
   for rid in range(len(planes.pixel_counts)):
    tt,vv=_region_token_support(planes.labels,rid);keep=vv>=.75;labels[tt[keep]]=rid;visible[tt[keep]]=vv[keep]
   for arm in range(9):
    selected=list(range(8))+([] if arm==0 else [7+arm]);tok,proto,score=pooled_additions(ot,op,rt,rp,rs,selected);extra={k:old[k][:0].copy() for k in old if k not in ['names','camera_matrices','radial_k1','correspondence_offsets']}
    if len(tok):
     with torch.no_grad():mean,var,logit=head(torch.from_numpy(q[tok]),torch.from_numpy(atlas['radio_features'][proto].astype(np.float32)),torch.from_numpy(tok))
     delta=mean.numpy().astype(float)*float(hm['coordinate_shrinkage']);assert hm.get('coordinate_affine_matrix') is None
     extra.update(world_points=atlas['world_points'][proto],query_tokens=tok,provenance_region_plane_atlas_row=np.c_[labels[tok],plane_ids[proto],atlas['texel_identity'][proto]],prototype_atlas_row=proto,query_plane_visible_fraction=visible[tok],radio_match_score=score,query_measurements_xy=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]+np.clip(delta,-2,2),query_measurement_variance_px2=float(hm.get('measurement_variance_scale',1.))*var.numpy().reshape(-1),correspondence_match_probability=torch.sigmoid(logit).numpy().reshape(-1))
     for k in ['prototype_world_covariance_m2','prototype_plane_pixel_purity','prototype_plane_depth_dispersion_m']:extra[k]=atlas[k][proto]
    extras[arm].append(extra)
  for arm in range(9):
   arr=append_query_rows(old,extras[arm]);mm={k:v for k,v in meta.items() if k not in ['content_sha256','arrays_sha256']};mm.update(correspondence_count=len(arr['query_tokens']),original_correspondences_sha256=file_sha256(oldpath),augmentation_scope='mapping cross-route native region MNN, original prefix retained, 1024 added-row cosine budget',counterfactual_added_rank=None if arm==0 else arm+8)
   write(d/f'arm{arm}_corr.npz',arr,mm);_load(d/f'arm{arm}_corr.npz')
  print(route,'hybrid inventories complete',flush=True)
 write(o/'features.npz',dict(names=np.asarray(allnames),features=np.asarray(features)),dict(query_pose_or_ground_truth_used=False,exclusion='seq9 omitted because fixed physical-region construction records seq9 mapping images',feature_contract='hybrid19',model_fitted=False))

if __name__=='__main__':main()
