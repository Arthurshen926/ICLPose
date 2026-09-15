"""Post-label diagnostics of frozen complete Cambridge runs; never deployment."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.tools.vfm.evaluate_goal_maplet_continuous_coordinate_pose_oracle import _pose_error,_oracle_candidate_rows,_solve_rows
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics

THRESHOLDS=[(.1,1),(.25,2),(.5,5)]

def candidates(z,names):
    """Read query-aligned poses and ragged pools, without mixing queries."""
    out=[[] for _ in names]
    if 'names' not in z or list(z['names'].astype(str))!=list(names):return out
    for k in z.files:
        if 'pose_w2c' not in k:continue
        a=z[k]
        if a.ndim==3 and a.shape[1:]==(4,4):
            if k=='candidate_pose_w2c' and 'candidate_offsets' in z:
                offsets=z['candidate_offsets']
                if len(offsets)!=len(names)+1 or offsets[-1]!=len(a):raise ValueError('Ragged pose offsets differ')
                for i in range(len(names)):out[i].extend(a[int(offsets[i]):int(offsets[i+1])])
            elif len(a)==len(names):
                for i,v in enumerate(a):
                    if k=='pose_w2c' and 'usable' in z and not z['usable'][i]:continue
                    out[i].append(v)
        elif a.ndim==4 and a.shape[0]==len(names) and a.shape[-2:]==(4,4):
            for i,v in enumerate(a):out[i].extend(v)
    return out

def hit(e,t,r):return bool(e[0]<=t and e[1]<=r)

def run(root,scene):
    s=root/scene;o=s/'postlabel_stage_audit';o.mkdir(exist_ok=True)
    if (o/'report.json').exists():return
    evaluation=s/'full_test_evaluation';seal=json.loads((evaluation/'endpoint_seal.json').read_text())
    for p,h in seal['sources'].items():
        if file_sha256(Path(p))!=h:raise ValueError('Frozen endpoint changed')
    ready=json.loads((s/'runtime_base/READY.json').read_text())
    files={}; batches={}
    for split in ready['splits']:
        front=s/'full_test'/split/'frontend';back=s/'full_test'/split/'backend/runtime_base'
        paths=sorted(set(list(front.rglob('*.npz'))+list(back.rglob('*.npz'))))
        paths=[p for p in paths if split in str(p.relative_to(s/'full_test'/split))]
        corr=[s/'runtime_base'/f'learned64d_strict_{split}_h025_{suffix}_corr.npz' for suffix in ['subtoken_mainline_v49','surfacecoord_homography_context_v50']]
        batches[split]=(paths,corr)
        for p in paths+corr:files[str(p)]=file_sha256(p)
    (o/'diagnostic_input_seal.json').write_text(json.dumps(dict(sources=files,role='POSTLABEL_ONLY_NO_SELECTION_OR_TRAINING'),indent=2))
    gt={v.image_id.replace('/','__')+'.npz':v.pose_w2c for v in parse_cambridge_pose_file(Path('/hy-tmp/Cambridge_stdloc')/scene/'dataset_test.txt')}
    rows=[]
    for split,(paths,corrpaths) in batches.items():
        with np.load(corrpaths[0]) as z:names=z['names'].astype(str)
        pool=[[] for _ in names];initial=[[] for _ in names]
        for path in paths:
            with np.load(path,allow_pickle=False) as z:values=candidates(z,names)
            for i,ps in enumerate(values):
                errors=[_pose_error(p,gt[names[i]]) for p in ps]
                pool[i].extend(errors)
                if path.name=='pnp.npz':initial[i].extend(errors)
        endpoints={}
        for seed in [1,2]:
            p=s/'full_test'/split/'backend/runtime_base/strong_local_precision_v414'/f'query_agreement_seed{seed}'/f'{split}_agreement.npz'
            with np.load(p) as z:
                if not np.array_equal(z['names'].astype(str),names):raise ValueError('Final order differs')
                endpoints[seed]=[_pose_error(p,gt[n]) for n,p in zip(names,z['pose_w2c'])]
        observed=[[] for _ in names];exact=[[] for _ in names];support=np.zeros(len(names),int)
        for cp in corrpaths:
            with np.load(cp) as z:a={k:z[k] for k in z.files if k!='metadata_json'}
            if not np.array_equal(a['names'].astype(str),names):raise ValueError('Correspondence order differs')
            for i,name in enumerate(names):
                lo,hi=map(int,a['correspondence_offsets'][i:i+2]);w=a['world_points'][lo:hi];tok=a['query_tokens'][lo:hi];pixel=a['query_measurements_xy'][lo:hi];K=a['camera_matrices'][i];rad=float(a['radial_k1'][i])
                if not len(w):continue
                chosen,projected,_=_oracle_candidate_rows(gt[name],w,tok,pixel,K,rad,maximum_error_px=4.)
                support[i]=max(support[i],len(chosen))
                if len(chosen)<6:continue
                for dest,px in [(observed,pixel),(exact,projected)]:
                    pose=_solve_rows(w[chosen],tok[chosen],px[chosen],K,rad)
                    dest[i].append(_pose_error(pose,gt[name]))
        for i,name in enumerate(names):
            row=dict(name=str(name),stages={'initial_pool':initial[i], 'all_saved_pose_pool':pool[i], 'gt_match_existing_pixels':observed[i], 'gt_match_exact_pixels':exact[i]},final={str(k):v[i] for k,v in endpoints.items()})
            row['attribution']={}
            for seed in [1,2]:
                for t,r in THRESHOLDS:
                    key=f'{seed}:{t}m_{r}deg';anyhit=lambda es:any(hit(e,t,r) for e in es)
                    if hit(endpoints[seed][i],t,r):category='final_success'
                    elif anyhit(pool[i]):category='saved_pose_available_selection_or_retention'
                    elif anyhit(observed[i]):category='gt_match_filtering_and_solver_can_recover'
                    elif anyhit(exact[i]):category='coordinate_or_geometry_sensitivity_under_gt_projection'
                    elif support[i]<6:category='retrieval_map_or_coordinate_support_insufficient'
                    else:category='geometry_or_remaining_solver_failure'
                    row['attribution'][key]=category
            rows.append(row)
        print(scene,split,'diagnostic complete',flush=True)
    if set(v['name'] for v in rows)!=set(seal['query_names']) or len(rows)!=len(seal['query_names']):raise ValueError('Diagnostic denominator differs')
    summary={}
    for stage in rows[0]['stages']:
        errors=[min(v['stages'][stage],key=lambda e:(e[0],e[1])) if v['stages'][stage] else (float('inf'),float('inf')) for v in rows]
        summary[stage]=dict(best_translation_candidate_metrics=metrics(np.array(errors)),any_candidate_joint_recall_percent={f'{t}m_{r}deg':100*np.mean([any(hit(e,t,r) for e in v['stages'][stage]) for v in rows]) for t,r in THRESHOLDS})
    attribution={key:{c:100*np.mean([v['attribution'][key]==c for v in rows]) for c in sorted(set(v['attribution'][key] for v in rows))} for key in rows[0]['attribution']}
    for p,h in files.items():
        if file_sha256(Path(p))!=h:raise ValueError('Diagnostic input changed')
    report=dict(scene=scene,queries=len(rows),role='POSTLABEL_ONLY_NO_SELECTION_OR_TRAINING',stages=summary,attribution_percent_all_queries=attribution,limitations=['Saved-pose pool combines both seeds and multiple earlier stages; upper bound, not deployable selector.', 'GT filtering of existing point/surface correspondences combines matching and solver effects.', 'Exact-pixel oracle uses GT projection; not a realistic accuracy result.', 'Map coverage, chart retrieval and within-chart coordinate support are not separately identifiable from these caches.', 'No result is used to train or select GreatCourt calibration.'],rows=rows)
    (o/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--root',type=Path,default=Path('output/cambridge_full_v418'));a=p.parse_args();run(a.root,a.scene)
