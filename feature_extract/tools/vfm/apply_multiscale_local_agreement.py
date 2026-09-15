"""Parameter-free two-scale agreement on already frozen strong-local pairs."""
from pathlib import Path
import argparse,json,numpy as np
from feature_extract.tools.vfm.strong_local_selection import choose_multiscale_local
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256,arrays_sha256,canonical_json_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=['mapping','query'],required=True)
 p.add_argument('--base',type=Path,default=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1'))
 p.add_argument('--splits',nargs='+');p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--output',type=Path)
 args=p.parse_args();b=args.base;r=b/'strong_local_precision_v414'
 splits=args.splits or (['seq1','seq2','seq4','seq6','seq7','seq8','seq11'] if args.domain=='mapping' else ['seq10','shard0','shard1','shard2','shard3'])
 seeds=args.seeds or ([0] if args.domain=='mapping' else [1,2])
 if any(seed not in ([0] if args.domain=='mapping' else [1,2]) for seed in seeds):raise ValueError('seed does not belong to the requested domain')
 for seed in seeds:
  out=(args.output or r)/('mapping_agreement' if seed==0 else f'query_agreement_seed{seed}');out.mkdir(parents=True,exist_ok=False)
  for s in splits:
   up=r/args.domain/'pairs'/f'{s}_unlabelled.json';f=json.load(open(up));bp=Path(f['base_path']);assert file_sha256(bp)==f['sources'][str(bp)]
   with np.load(bp) as z:names=z['names'];base=z['pose_w2c']
   current=bp if seed==0 else b/f'eligible_verification_v413/seed{seed}/dual_support/uniform_verified'/f'{s}_confirmed.npz'
   with np.load(current) as z:np.testing.assert_array_equal(names,z['names']);cur=z['pose_w2c']
   candidates={}
   for arm,path in f['candidate_paths'].items():
    assert file_sha256(Path(path))==f['sources'][path]
    with np.load(path) as z:np.testing.assert_array_equal(names,z['names']);candidates[arm]=z['pose_w2c']
   groups={str(n):[] for n in names}
   for x in f['pairs']:groups[x['name']].append(x)
   poses=[];accepted=[];audit=[]
   for i,n in enumerate(names.astype(str)):
    rows=groups[n];bound=np.array_equal(base[i],cur[i]);j=choose_multiscale_local(rows) if bound else -1;poses.append(candidates[rows[j]['arm']][i] if j>=0 else cur[i]);accepted.append(j>=0);audit.append(dict(name=n,base_exactly_matches_current=bound,choice=rows[j]['arm'] if j>=0 else None))
   arr=dict(names=names,pose_w2c=np.array(poses),accepted=np.array(accepted));sources={str(p):file_sha256(p) for p in [up,bp,current,Path(__file__),Path('feature_extract/tools/vfm/strong_local_selection.py')]+[Path(v) for v in f['candidate_paths'].values()]};meta=dict(artifact_type='strong_multiscale_agreement_v414',arrays_sha256=arrays_sha256(arr),source_sha256=sources,query_pose_or_ground_truth_read=False,base_bound_to_v357=True,rule='positive coarse and fine mean gain; rank eligible candidates by fine mean',independent_evidence_claim=False);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(out/f'{s}_agreement.npz',**arr,metadata_json=np.array(json.dumps(meta)));(out/f'{s}_audit.json').write_text(json.dumps(audit))


if __name__=='__main__':main()
