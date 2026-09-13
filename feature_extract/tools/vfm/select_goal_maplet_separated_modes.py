"""Keep the strong local endpoint; compare only separated regional pose modes.

The separation uses the frozen backend trust limits (0.5 m, 3 degrees), rather
than retuning an acceptance threshold against evaluation recalls.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def separated_mode(base,proposal):
    cb=-base[:3,:3].T@base[:3,3];cp=-proposal[:3,:3].T@proposal[:3,3]
    angle=np.rad2deg(np.arccos(np.clip((np.trace(proposal[:3,:3]@base[:3,:3].T)-1)/2,-1,1)))
    return bool(np.linalg.norm(cb-cp)>.5 or angle>3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','refined','verified','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--split',required=True);a=p.parse_args();s=a.split;a.output.mkdir(exist_ok=True,parents=True)
    bp=a.base/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';base,_=read(bp)
    for arm in ['mnn','joint']:
        rp=a.refined/f'{s}_{arm}_audit.json';vp=a.verified/f'{s}_{arm}_audit.json';rr=json.load(open(rp));vv=json.load(open(vp));values={k:[] for k in ['mode','mode_depth']};audit=[]
        for i,name in enumerate(base['names'].astype(str)):
            assert name==rr[i]['name']==vv[i]['name'];poses=[base['pose_w2c'][i]]+[np.array(p) for p in rr[i]['refined_candidate_poses'] if p is not None];scores=vv[i]['scores'];assert len(poses)==len(scores);choices={}
            for kind in values:
                eligible=[j for j in range(1,len(poses)) if separated_mode(poses[0],poses[j]) and scores[j][0][0]>=24 and tuple(scores[j][0])>tuple(scores[0][0]) and tuple(scores[j][1])>tuple(scores[0][1]) and (kind=='mode' or scores[j][2][0]>scores[0][2][0])]
                choice=max(eligible,key=lambda j:tuple(scores[j][0])) if eligible else 0;choices[kind]=choice;values[kind].append(poses[choice])
            audit.append(dict(name=name,choices=choices))
        for kind,poses in values.items():
            dest=a.output/f'{s}_{arm}_{kind}.npz'
            if dest.exists():raise FileExistsError(dest)
            arrays=dict(names=base['names'],pose_w2c=np.array(poses),accepted=np.array([r['choices'][kind]>0 for r in audit]));meta=dict(artifact_type='separated_partial_visibility_modes_v1',arrays_sha256=arrays_sha256(arrays),source_sha256={str(p):file_sha256(p) for p in [bp,rp,vp,Path(__file__)]},query_pose_or_ground_truth_read=False,translation_separation_m=.5,rotation_separation_deg=3,selection=kind,calibrated_probability=False);meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
        (a.output/f'{s}_{arm}_audit.json').write_text(json.dumps(audit))
if __name__=='__main__':main()
