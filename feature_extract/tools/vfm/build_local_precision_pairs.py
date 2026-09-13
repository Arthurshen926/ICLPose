from pathlib import Path
import argparse,json,numpy as np
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.local_precision_evidence import paired_features,FEATURE_NAMES
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.verification_bank_contract import validate_bank
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
p=argparse.ArgumentParser();p.add_argument('--route',required=True);a=p.parse_args();s=a.route
b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');r=b/'local_precision_v409';o=r/'mapping';o.mkdir(exist_ok=True);maps,meta,sources=load_map(b);mp=b/'native_fine_v264/readout/map.npz';bank=(r/'banks' if s in ['seq1','seq2','seq4','seq6'] else b/'heldout_evidence_v405/banks')/f'mapping_{s}.npz';up=b/'spatial_verifier_v404/mapping'/f'{s}_unlabelled.json';records=json.load(open(up));cp=b/'native_hybrid_mapping_v286'/s/'corr.npz'
with np.load(bank) as f:bk={k:f[k] for k in f.files if k!='metadata_json'};bm=json.loads(f['metadata_json'].item())
with np.load(cp) as f:validate_bank(bk,bm,maps['world'],mp,file_sha256(mp),meta['atlas_sha256'],f['names'],f['camera_matrices'],f['radial_k1'])
sources.update({str(x):file_sha256(x) for x in [bank,up,cp,Path(__file__),Path('feature_extract/tools/vfm/local_precision_evidence.py')]});pairs=[]
for i,n in enumerate(bk['names'].astype(str)):
 rows=[x for x in records if x['name']==n];cache=b/'adaptive_memory_v234/fine_cache'/n;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];sources[str(cache)]=file_sha256(cache);lo,hi=bk['offsets'][i:i+2];ids=bk['prototype_rows'][lo:hi];available=maps['available'][ids];ids=ids[available];xy=bk['pixels'][lo:hi][available];tokens=bk['query_tokens'][lo:hi][available]
 for j,x in enumerate(rows):
  for k,y in enumerate(rows[:j]):
   p0=np.array(y['pose']);p1=np.array(x['pose'])
   if separated_mode(p0,p1):continue
   f,count=paired_features(p0,p1,maps['world'][ids],xy,tokens,bk['camera_matrices'][i],float(bk['radial_k1'][i]),grids,[maps['coarse_map'][ids],maps['fine_map'][ids]])
   pairs.append(dict(name=n,base=k,candidate=j,features=f.tolist(),support=count))
(o/f'{s}_unlabelled.json').write_text(json.dumps(dict(pairs=pairs,sources=sources)))
# Labels join only after all pose/feature pairs have been frozen.
lp=b/'spatial_verifier_v404/mapping'/f'{s}.json';label=json.load(open(lp))['records'];lookup={n:[x for x in label if x['name']==n] for n in bk['names'].astype(str)}
for x in pairs:x.update(base_error=lookup[x['name']][x['base']]['error'],candidate_error=lookup[x['name']][x['candidate']]['error'])
(o/f'{s}.json').write_text(json.dumps(dict(pairs=pairs,query_routes_read=False,mapping_baseline_is_candidate_proxy=True,sources=sources,label_sha256=file_sha256(lp),unlabelled_sha256=file_sha256(o/f'{s}_unlabelled.json')),indent=2));print(s,len(pairs))
