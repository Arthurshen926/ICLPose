"""Preserve existing association scores; new uncalibrated candidates enter at a floor."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

def conservative_scores(old_source,old_plane,old_score,new_source,new_plane):
 if not len(old_score) or not np.isfinite(old_score).all():raise ValueError('finite nonempty base scores required')
 out=np.empty(len(new_source),float)
 for s in np.unique(new_source):
  orig=np.flatnonzero(old_source==s);fresh=np.flatnonzero(new_source==s)
  for p in np.unique(new_plane[fresh]):
   same=orig[old_plane[orig]==p]
   floor=np.min(old_score[same]) if len(same) else np.min(old_score[orig]) if len(orig) else np.min(old_score)
   out[fresh[new_plane[fresh]==p]]=floor
 return np.r_[old_score,out]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--frontend',type=Path,required=True);a=p.parse_args();b=a.base;f=a.frontend
 with np.load(b/'memory_transfer_candidates_v224.npz') as z:c={k:z[k] for k in z.files}
 with np.load(f/'union.npz') as z:u={k:z[k] for k in z.files}
 with np.load(b/'adaptive_memory_v234/adaptive_transfer_scores.npz') as z:
  if str(z['candidate_sha256'])!=file_sha256(b/'memory_transfer_candidates_v224.npz'):raise ValueError('base score lineage')
  old=z['logits'][:,z['arm_names'].tolist().index('fixed2')][c['homography_keep']]
 n=len(old)
 for key in ['source_image','query_token','prototype_rows']:
  if not np.array_equal(u[key][:n],c[key][c['homography_keep']]):raise ValueError('union prefix must preserve old kept rows')
 if not np.array_equal(u['prototype_world'],c['prototype_world']):raise ValueError('map changed')
 planes=u['prototype_plane'][u['prototype_rows']];scores=conservative_scores(u['source_image'][:n],planes[:n],old,u['source_image'][n:],planes[n:]);target=f/'union_strong_scores.npz'
 if target.exists():
  with np.load(target) as z:
   if not np.array_equal(scores,z['logits'][:,0]):raise ValueError('existing floor scores differ')
 else:np.savez_compressed(target,logits=scores[:,None],arm_names=np.array(['fixed2']),candidate_sha256=np.asarray(file_sha256(f/'union.npz')))
 (f/'conservative_score_audit.json').write_text(json.dumps({'old_rows_exactly_preserved':True,'old_logits_exactly_preserved':True,'query_GT_used':False,'new_logit_policy':'same query/plane minimum, query minimum if absent, global minimum if query has no base candidates','not_calibrated_probabilities':True,'union_sha256':file_sha256(f/'union.npz'),'score_sha256':file_sha256(target)},indent=2))
if __name__=='__main__':main()
