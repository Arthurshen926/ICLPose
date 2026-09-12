"""Fit conditional fine coordinate/variance models on frozen mapping supervision."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.native_fine_reliability import reliability_features,fit,predict
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256,file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
 cal=json.load(open(b/'native_fine_calibration_v267/calibration.json'));datafile=b/'native_fine_calibration_v267/mapping_rows.npz'
 if file_sha256(datafile)!=cal['mapping_rows_sha256']:raise ValueError('mapping teacher binding differs')
 with np.load(datafile) as z:d={k:z[k] for k in z.files}
 mapfile=b/'native_fine_v264/readout/map.npz'
 if file_sha256(mapfile)!=cal['map_sha256']:raise ValueError('map binding differs')
 with np.load(mapfile) as z:maps=z['fine'];mm=json.loads(str(z['metadata_json']))
 features=np.zeros((len(d['source']),6));offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
 for src in np.unique(d['source']):
  rows=np.flatnonzero(d['source']==src);xy=d['target'][rows];tok=((xy[:,1]+.5)//4).astype(int)*64+((xy[:,0]+.5)//4).astype(int);center=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];probes=d['original'][rows,None]+offsets
  grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/str(d['source_names'][src]),mm['projection_sha256'])
  if ck!=mm['checkpoint_sha256']:raise ValueError('checkpoint differs')
  sim=np.sum(sample_grid(grids[1],probes)*maps[d['prototype'][rows],None].astype(np.float32),axis=-1);sim[(np.abs(probes-center[:,None])>2+1e-8).any(2)]=-np.inf;best=select_offsets(sim,[],False);update=probes[np.arange(len(rows)),best]
  if not np.array_equal(update,d['update'][rows]):raise ValueError('mapping readout replay differs')
  features[rows]=reliability_features(sim,update-d['original'][rows],d['variance'][rows])
 routes=np.array([str(d['source_names'][i]).split('__')[0] for i in d['source']]);train=np.isin(routes,cal['training_routes']);weights=np.zeros(len(train))
 for src in np.unique(d['source']):rows=d['source']==src;weights[rows]=1./rows.sum()
 model=fit(features[train],d['original'][train],d['update'][train],d['target'][train],d['variance'][train],weights[train],cal['alpha'],cal['variance_scale']);alpha,var=predict(model,features,d['variance']);report={}
 for split,mask in [('train',train),('heldout',~train)]:
  stats={}
  for arm,aa,vv in [('fixed',np.full(len(alpha),cal['alpha']),d['variance']*cal['variance_scale']),('conditional_mean',alpha,d['variance']*cal['variance_scale']),('conditional_both',alpha,var)]:
   residual=d['original'][mask]+aa[mask,None]*(d['update'][mask]-d['original'][mask])-d['target'][mask];error=np.linalg.norm(residual,axis=1);v=vv[mask]
   stats[arm]=dict(mean_pixel_error=float(error.mean()),median_pixel_error=float(np.median(error)),mean_gaussian_nll=float(np.mean(np.log(2*np.pi*v)+error**2/(2*v))))
  report[split]=stats
 payload=dict(artifact_type='goal_maplet_native_fine_reliability_v1',model=model,map_sha256=file_sha256(mapfile),source_mapping_rows_sha256=file_sha256(datafile),query_ground_truth_read=False,heldout_used_to_fit=False,training_routes=cal['training_routes'],heldout_routes=cal['heldout_routes'],conditional_correct_identity_only=True,report=report,fixed_alpha=cal['alpha'],fixed_variance_scale=cal['variance_scale'])
 payload['content_sha256']=canonical_json_sha256(payload);(o/'model.json').write_text(json.dumps(payload,indent=2));np.savez_compressed(o/'mapping_features.npz',features=features,source=d['source']);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
