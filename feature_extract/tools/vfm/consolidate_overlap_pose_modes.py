"""Keep the established fine endpoint inside its existing trust region.

A fixed secondary readout control, not an overlap/identity learner or calibrated
acceptance rule. Inputs have already passed the unchanged candidate scorer.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.select_goal_maplet_separated_modes import separated_mode
from feature_extract.tools.vfm.build_goal_maplet_crossfit_feature_pose import read
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','input','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['mnn','learned']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for s in ['seq10','shard0','shard1','shard2','shard3']:
        bp=a.base/'regularized_stage_precision_v357'/f'{s}_stage_plain_reg_all.npz';base,_=read(bp)
        for arm in a.arms:
            ip=a.input/f'{s}_{arm}_selected.npz';new,_=read(ip);assert np.array_equal(base['names'],new['names']);take=np.array([bool(accepted) and separated_mode(b,n) for accepted,b,n in zip(new['accepted'],base['pose_w2c'],new['pose_w2c'])]);poses=np.where(take[:,None,None],new['pose_w2c'],base['pose_w2c']);arrays=dict(names=base['names'],pose_w2c=poses,accepted=take);meta=dict(artifact_type='overlap_lod_local_mode_consolidation_v1',arrays_sha256=arrays_sha256(arrays),source_sha256={str(p):file_sha256(p) for p in [bp,ip,Path(__file__),Path(__file__).with_name('select_goal_maplet_separated_modes.py')]},query_pose_or_ground_truth_read=False,translation_separation_m=.5,rotation_separation_deg=3,rule='unchanged upstream acceptance AND separate from baseline trust region; otherwise baseline',calibrated_acceptance=False)
            dest=a.output/f'{s}_{arm}_consolidated.npz'
            if dest.exists():raise FileExistsError(dest)
            meta['content_sha256']=canonical_json_sha256(meta);np.savez_compressed(dest,**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
if __name__=='__main__':main()
