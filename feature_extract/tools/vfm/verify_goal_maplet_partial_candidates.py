"""Verify new regional poses using a separately frozen correspondence bank.

No query GT is opened. This is an uncalibrated conservative control: a proposal
must improve both old and new reprojection support. Optional MoGe verification
fits scale on one token parity and checks depth consistency on the other.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def depth_support(pose,world,tokens,query_points,reliable,pixels,groups,K,k1):
    _,rows=score_pose(pose,world,pixels,groups,K,k1)
    if not len(rows):return 0,None
    z=(world[rows]@pose[:3,:3].T+pose[:3,3])[:,2];tt=tokens[rows];qz=query_points[tt,2]
    valid=reliable[tt]&(qz>0)&(z>0);fit=valid&(tt%2==0);check=valid&(tt%2==1)
    if fit.sum()<6 or check.sum()<6:return 0,None
    scale=float(np.median(np.log(z[fit]/qz[fit])))
    residual=np.abs(np.log(z[check]/qz[check])-scale)
    return int((residual<=np.log(1.25)).sum()),scale


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','input','frontend','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--split',required=True);p.add_argument('--arms',nargs='+',default=['mnn','joint']);a=p.parse_args();b=a.base;s=a.split;a.output.mkdir(exist_ok=True,parents=True)
    bp=b/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';base,_=read(bp)
    cp=b/'native_hybrid_fine_v301/readout'/f'{s}_reliability_rank32.npz'
    with np.load(cp) as f:old={k:f[k] for k in ['names','world_points','query_tokens','query_measurements_xy','correspondence_offsets','camera_matrices','radial_k1']}
    assert np.array_equal(old['names'],base['names'])
    mp=b/'native_fine_v264/readout/map.npz'
    with np.load(mp) as f:mapworld=f['world_points']
    cmd=dict(json.load(open(b/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1])
    for arm in a.arms:
        rp=a.input/f'{s}_{arm}_audit.json';fp=a.frontend/f'{s}_{arm}_audit.json';records=json.load(open(rp));front=json.load(open(fp));result={k:[] for k in ['crossbank','crossbank_depth']};audits=[];sources={str(x):file_sha256(x) for x in [bp,cp,mp,rp,fp,Path(__file__),Path(__file__).with_name('token_hypothesis_ransac.py'),Path(__file__).with_name('build_goal_maplet_structured_memory_features.py')]}
        for i,name in enumerate(base['names'].astype(str)):
            assert records[i]['name']==front[i]['name']==name
            lo,hi=old['correspondence_offsets'][i:i+2];ow,ot,oxy=canonical_hypotheses(old['world_points'][lo:hi],old['query_tokens'][lo:hi],old['query_measurements_xy'][lo:hi]);og=[np.flatnonzero(ot==t) for t in np.unique(ot)];K=old['camera_matrices'][i];k1=float(old['radial_k1'][i])
            regions=front[i]['regions'];nt=np.concatenate([np.array(r['selected_tokens'],int) for r in regions]) if regions else np.empty(0,int);pr=np.concatenate([np.array(r['prototype_rows'],int) for r in regions]) if regions else np.empty(0,int);nxy=np.concatenate([np.array(r['query_pixels'],float).reshape(-1,2) for r in regions]) if regions else np.empty((0,2));nw,nt,nxy=canonical_hypotheses(mapworld[pr],nt,nxy);ng=[np.flatnonzero(nt==t) for t in np.unique(nt)]
            qp,qn,qv=moge_tokens(md/name);sources[str(md/name)]=file_sha256(md/name)
            poses=[base['pose_w2c'][i]]+[np.array(p) for p in records[i]['refined_candidate_poses'] if p is not None]
            scores=[(score_pose(p,ow,oxy,og,K,k1,return_selected=False)[0],score_pose(p,nw,nxy,ng,K,k1,return_selected=False)[0],depth_support(p,ow,ot,qp,qv,oxy,og,K,k1)) for p in poses]
            choices={}
            for kind in result:
                eligible=[j for j in range(1,len(poses)) if scores[j][0][0]>=24 and scores[j][0]>scores[0][0] and scores[j][1]>scores[0][1] and (kind=='crossbank' or scores[j][2][0]>scores[0][2][0])]
                chosen=max(eligible,key=lambda j:scores[j][0]) if eligible else 0;choices[kind]=chosen;result[kind].append(poses[chosen])
            audits.append(dict(name=name,choices=choices,scores=scores))
        for kind,poses in result.items():
            dest=a.output/f'{s}_{arm}_{kind}.npz'
            if dest.exists():raise FileExistsError(dest)
            arrays=dict(names=base['names'],pose_w2c=np.array(poses),accepted=np.array([r['choices'][kind]>0 for r in audits]));meta=dict(artifact_type='partial_visibility_crossbank_v1',arrays_sha256=arrays_sha256(arrays),source_sha256=sources,query_pose_or_ground_truth_read=False,selection=kind,not_independent_feature_backbone=True,calibrated_probability=False,depth_tolerance_relative=1.25,depth_scale_fit_even_tokens_check_odd=True);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
        (a.output/f'{s}_{arm}_audit.json').write_text(json.dumps(audits));print(s,arm,'verified',flush=True)
if __name__=='__main__':main()
