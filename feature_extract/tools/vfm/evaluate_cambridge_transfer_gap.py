from pathlib import Path
import numpy as np,json,time
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=260901);args=p.parse_args();seed=args.seed;assert seed in [260901,260902]
r=Path('output/cambridge_gap_audit_v416');src=Path('output/cambridge_core_v415/StMarysChurch');b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1');names=sorted(n for n in json.load(open(src/'test_names.json')) if n.startswith('seq13__'));variants=['newmap_oldcamera','oldmap_newcamera','oldmap_oldcamera'];seals={}
for arm in variants:
 d=r/arm/f'StMarysChurch/readout_seed{seed}'
 while not all((d/n).exists() for n in names):time.sleep(5)
 seals[arm]={n:file_sha256(d/n) for n in names}
(r/f'endpoint_seal_seed{seed}.json').write_text(json.dumps(seals,indent=2));gt={x.image_id.replace('/','__')+'.npz':x.pose_w2c for x in parse_cambridge_pose_file(Path('/hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_test.txt'))};report={};arr={}
for variant in ['newmap_newcamera']+variants:
 d=(src if variant=='newmap_newcamera' else r/variant/'StMarysChurch')/f'readout_seed{seed}';errors=[]
 for n in names:
  with np.load(d/n) as z:errors.append(_pose_error(z['pooled_global_field'],gt[n]))
 arr[variant]=np.array(errors);report[variant]=metrics(arr[variant])
paths={'old_v275':'native_reliability_mainline_v275/{s}_reliability_rank32_consensus/selected.npz','old_v327':'sequential_depth_rescue_v327/{s}.npz','old_v344':'direct_feature_consistency_v344/{s}_multiscale_robust512.npz','old_v357':'regularized_stage_precision_v357/{s}_stage_plain_reg_all.npz','old_v414':'strong_local_precision_v414/query_agreement_seed1/{s}_agreement.npz'}
for label,template in paths.items():
 poses={}
 for i in range(4):
  with np.load(b/template.replace('query_agreement_seed1','query_agreement_seed'+str(seed-260900)).format(s=f'shard{i}')) as z:poses.update(zip(z['names'].astype(str),z['pose_w2c']))
 assert set(poses)==set(names);arr[label]=np.array([_pose_error(poses[n],gt[n]) for n in names]);report[label]=metrics(arr[label])
np.savez_compressed(r/f'same_query_errors_seed{seed}.npz',names=np.array(names),**arr);(r/f'same_query_metrics_seed{seed}.json').write_text(json.dumps(dict(scope='post-label same-query map-bundle x query-calibration factorial; not a blind generalization test',reports=report),indent=2));print(json.dumps(report,indent=2))
