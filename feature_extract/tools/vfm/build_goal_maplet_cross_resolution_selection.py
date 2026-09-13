"""Symmetric coarse/fine feature checks between two frozen precision endpoints.

Same anchors and denominator for both candidates. Feature resolutions share a
backbone and are not independent evidence or calibrated likelihoods.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read,feature_loss
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def choose(losses,rule):
    scores=np.asarray(losses,float)
    if scores.shape!=(2,2) or not np.isfinite(scores).all():return 0
    if rule=='strict_both':return int(np.all(scores[1]<scores[0]-1e-10))
    if rule=='mean_loss':return int(scores[1].mean()<scores[0].mean()-1e-10)
    raise ValueError('unknown cross-resolution rule')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
 mp=b/'native_fine_v264/readout/map.npz'
 with np.load(mp) as z:maps={k:z[k] for k in z.files if k!='metadata_json'};mm=json.loads(z['metadata_json'].item())
 if arrays_sha256(maps)!=mm['arrays_sha256']:raise ValueError('map array authority differs')
 for s in ['seq10','shard0','shard1','shard2','shard3']:
  paths=[b/'crossfit_feature_step_v338'/f'{s}_robust.npz',b/'direct_feature_constrained_v340'/f'{s}_constrained_robust_direct512.npz'];poses=[]
  for path in paths:
   arr,m=read(path)
   if m['source_sha256'].get(str(mp))!=file_sha256(mp):raise ValueError('map source binding differs')
   poses.append(arr)
  cp=b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz';c,_=_load(cp);rp=b/'direct_feature_precision_v330'/f'{s}_fine_joint512_audit.json';rows=json.load(open(rp));names=poses[0]['names']
  if not np.array_equal(names,poses[1]['names']) or not np.array_equal(names,c['names']) or names.tolist()!=[r['name'] for r in rows]:raise ValueError('query order differs')
  sources={str(f):file_sha256(f) for f in [mp,cp,rp,*paths,Path(__file__),Path(__file__).with_name('direct_anonymous_feature_pose.py')]};evidence=[];choices={k:[] for k in ['strict_both','mean_loss']}
  for i,n in enumerate(names.astype(str)):
   ids=np.array(rows[i]['prototype_rows'],int);tokens=rows[i]['selected_tokens'];scores=np.full((2,2),np.nan)
   if len(ids)>=32:
    cache=b/'native_fine_v264/query_cache'/n;grids,ck=load_grids(cache,mm['projection_sha256']);sources[str(cache)]=file_sha256(cache)
    if ck!=mm['checkpoint_sha256']:raise ValueError('checkpoint differs')
    for j in range(2):
     for level,key in enumerate(['coarse','fine']):scores[j,level]=feature_loss(poses[j]['pose_w2c'][i],maps['world_points'][ids].astype(float),c['camera_matrices'][i].astype(float),float(c['radial_k1'][i]),grids[level],maps[key][ids].astype(float),True)
   for rule in choices:choices[rule].append(choose(scores,rule))
   evidence.append(dict(name=n,losses=scores.tolist() if np.isfinite(scores).all() else None,checked_tokens=tokens,insufficient=len(ids)<32))
  for rule,cs in choices.items():
   ch=np.array(cs,int);arr=dict(names=names,pose_w2c=np.stack([p['pose_w2c'] for p in poses],axis=1)[np.arange(len(ch)),ch],usable=np.stack([p['usable'] for p in poses],axis=1)[np.arange(len(ch)),ch],accepted=ch.astype(bool));m=dict(artifact_type='goal_maplet_cross_resolution_selection_v1',arrays_sha256=arrays_sha256(arr),query_pose_or_ground_truth_read=False,source_rgb_stored_or_consumed_at_runtime=False,source_sha256=sources,rule=rule,candidate_order='spatial_CV_then_constrained_fine',evidence_is_not_statistically_independent=True,not_calibrated_likelihood=True)
   m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(o/f'{s}_{rule}.npz',**arr,metadata_json=np.array(json.dumps(m,sort_keys=True)))
  (o/f'{s}_evidence.json').write_text(json.dumps(evidence));print(s,'done',flush=True)
if __name__=='__main__':main()
