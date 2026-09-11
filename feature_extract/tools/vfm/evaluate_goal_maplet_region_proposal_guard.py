"""Common original-observation guard; matched two-solve old-pose ensemble control."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import score_pose
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;a.output.mkdir(exist_ok=False)
 with np.load(b/'memory_transfer_candidates_v224.npz') as z:c={k:z[k] for k in z.files}
 old=[];new=[]
 for s in [0,1,2]:
  op=b/f'full_pool_boundaries_v245/fixed_s{s}_moge.npz' if s<2 else b/'region_frontend_v246/held_fixed_s2_moge.npz';npth=b/f'region_frontend_v246/strong_union_s{s}_moge.npz' if s<2 else b/'region_frontend_v246/held_union_s2_moge.npz'
  with np.load(op) as z:names=z['names'].astype(str);sources=z['source_image'];old.append(z['pose_w2c'])
  with np.load(npth) as z:
   if not np.array_equal(names,z['names']) or not np.array_equal(sources,z['source_image']):raise ValueError('query order')
   new.append(z['pose_w2c'])
 old=np.asarray(old);new=np.asarray(new);oldkeys=np.zeros((3,len(names),2));newkeys=oldkeys.copy()
 for i,(s,name) in enumerate(zip(sources,names)):
  rows=np.flatnonzero((c['source_image']==s)&c['homography_keep']);tok=c['query_token'][rows];w=c['prototype_world'][c['prototype_rows'][rows]];pixels=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];groups=[np.flatnonzero(tok==t) for t in np.unique(tok)]
  with np.load(Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/name) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
  for seed in [0,1,2]:
   for poses,keys in [(old,oldkeys),(new,newkeys)]:keys[seed,i]=(-1,-np.inf) if not np.isfinite(poses[seed,i]).all() else score_pose(poses[seed,i],w,pixels,groups,K,k1)[0]
 manifest={};decisions={}
 for s in [0,1,2]:
  for arm,proposal,key in [('guarded',new[s],newkeys[s]),('old_two_solve',old[(s+1)%3],oldkeys[(s+1)%3])]:
   accept=(key[:,0]>oldkeys[s,:,0])|((key[:,0]==oldkeys[s,:,0])&(key[:,1]>oldkeys[s,:,1]));out=np.where(accept[:,None,None],proposal,old[s]);name=f'{arm}_s{s}';path=a.output/(name+'.npz');np.savez_compressed(path,names=names,source_image=sources,pose_w2c=out);manifest[name]=str(path);decisions[name]=accept.tolist()
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'audit.json').write_text(json.dumps({'GT_opened':False,'decision':'strictly better original unique-token geometry key; ties retain first pose','original_observations_are_common_not_statistically_independent':True,'cost':'two full pose pipelines for both guarded proposal and old-two-solve control; not the one-solve baseline cost','seed_pairing':'s and (s+1)%3 for old control','decisions':decisions},indent=2))
if __name__=='__main__':main()
