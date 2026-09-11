"""Same frozen real/coarse features and 3x3 footprint, discrete vs continuous readout."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.vfm.localization_goal_maplet.subpixel_peak import quadratic_peak
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;old=b/'adaptive_memory_v234';a.output.mkdir(exist_ok=False)
 cp=b/'memory_transfer_candidates_v224.npz'
 with np.load(cp) as z:c={k:z[k] for k in z.files}
 with np.load(old/'fine_readout/map_readouts.npz') as z:
  assert np.array_equal(z['prototype_world'],c['prototype_world']);desc=[z['coarse'].astype(np.float32),z['fine'].astype(np.float32)];available=z['available']
 with np.load(old/'multiscale/memory_transfer_candidates_v224.npz') as z:names=z['source_names'].astype(str)
 with np.load(old/'roi_models/test_roi_pixels.npz') as z:mask=z['refinement_mask']&available[c['prototype_rows']];assert str(z['candidate_sha256'])==file_sha256(cp)
 xy=np.c_[(c['query_token']%64)*4+1.5,(c['query_token']//64)*4+1.5];discrete=np.repeat(xy[:,None],2,axis=1);continuous=discrete.copy();accepted=np.zeros((len(xy),2),bool);offset=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
 for count,s in enumerate(np.unique(c['source_image'])):
  rows=np.flatnonzero((c['source_image']==s)&mask);pr=c['prototype_rows'][rows]
  if not len(rows):continue
  with np.load(old/'fine_cache'/names[s]) as z:
   meta=json.loads(str(z['metadata_json']));assert meta['poses_or_labels_opened'] is False
   grids=[z['coarse_final'].astype(np.float32),z['fine_final'].astype(np.float32)]
  for k,g in enumerate(grids):
   probes=xy[rows,None]+offset;sim=np.sum(sample_grid(g,probes)*desc[k][pr,None],axis=-1);sub,valid,d=quadratic_peak(sim)
   actual=np.sum(sample_grid(g,xy[rows]+sub)*desc[k][pr],axis=-1);valid &= actual>=sim.max(1)
   discrete[rows,k]=xy[rows]+d;continuous[rows,k]=xy[rows]+np.where(valid[:,None],sub,d);accepted[rows,k]=valid
  if (count+1)%40==0:print('continuous readout',count+1,flush=True)
 for name,pixels in [('discrete',discrete),('continuous',continuous)]:np.savez_compressed(a.output/(name+'.npz'),refined_pixels=pixels.astype(np.float32),refinement_mask=mask,candidate_sha256=np.asarray(file_sha256(cp)))
 (a.output/'audit.json').write_text(json.dumps({'scope':__doc__,'query_GT_used':False,'continuous_accept_fraction_on_mask':accepted[mask].mean(0).tolist(),'refinement_mask_fraction':float(mask.mean()),'feature_source':'existing full-image matched fine cache; same frozen ROI mask for both controls','real_feature_reextraction':False,'acceptance':'negative definite quadratic, bounded stationary point, exact resampled cosine no worse than discrete peak'},indent=2))
if __name__=='__main__':main()
