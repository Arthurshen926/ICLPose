"""Complete paired accounting for frozen memory, fine readout and MoGe trials."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.summarize_goal_maplet_structured_memory_results import compare_errors,THRESHOLDS
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    runs={};errors={};inventories={};gt={};sources={}
    for family in ['transfer','fine','roi']:
        for seed in [0,1]:
            for summary in sorted((a.root/f'{family}_seed{seed}').glob('*/summary.json')):
                report=json.loads(summary.read_text());arm=report['reports'][0]['arm']
                if report['reports'][0]['scored_hypotheses']!=224*5*256 or report['reports'][0]['budget_unreached_groups']!=0:
                    raise ValueError('unequal or incomplete PnP hypothesis budget')
                key=f'{family}_{summary.parent.name}_s{seed}';sources[key]=summary.parent/(arm+'.npz')
    moge_arms=['fixed2','adaptive','fixed25','adaptive25','adaptive_scale_fixed25','fixed_scale_learned25','coarse_readout','intermediate_readout','highres_readout','highres_pixels',
               'interpolated_pixels','full_window_readout','roi_readout','full_window_readout_pixels','roi_readout_pixels']
    for seed in [0,1]:
        for arm in moge_arms:sources[f'moge_{arm}_s{seed}']=a.root/f'moge_v2_{arm}_seed{seed}.npz'
    expected=16+10+8+30
    if len(sources)!=expected:raise ValueError(f'incomplete result inventory: {len(sources)}/{expected}')
    for key,path in sources.items():
        with np.load(path) as z:names=z['names'].astype(str);poses=z['pose_w2c']
        if len(names)!=224 or len(set(names))!=224:raise ValueError('incomplete/duplicate query population')
        e=[]
        for name,pose in zip(names,poses):
            if name not in gt:
                with np.load(a.contributors/name) as z:gt[name]=z['pose_w2c']
            e.append(_pose_error(pose,gt[name]))
        e=np.array(e);errors[key]=e;inventories[key]=names;routes=np.array([n.split('__')[0] for n in names]);by_route={}
        for route in ['all','seq12','seq14']:
            ix=np.ones(len(e),bool) if route=='all' else routes==route;x=e[ix]
            by_route[route]={'images':int(ix.sum()),'hits':[int(((x[:,0]<=t)&(x[:,1]<=r)).sum()) for t,r in THRESHOLDS],
                'translation_median':float(np.median(x[:,0]))}
        runs[key]={'path':str(path),'sha256':file_sha256(path),'by_route':by_route}
    comparisons=[('transfer_fixed2','transfer_adaptive'),('transfer_fixed2','transfer_fixed25'),('transfer_fixed25','transfer_adaptive25'),
        ('fine_coarse_readout','fine_intermediate_readout'),('fine_coarse_readout','fine_highres_readout'),
        ('fine_highres_readout_pixels0','fine_highres_readout_pixels1'),('roi_full_window_readout','roi_roi_readout'),
        ('moge_fixed2','moge_adaptive'),('moge_fixed2','moge_fixed25'),('moge_fixed25','moge_adaptive25'),
        ('moge_coarse_readout','moge_intermediate_readout'),('moge_coarse_readout','moge_highres_pixels'),
        ('moge_interpolated_pixels','moge_highres_pixels'),('moge_full_window_readout','moge_roi_readout'),
        ('moge_coarse_readout','moge_roi_readout'),('moge_roi_readout','moge_roi_readout_pixels'),
        ('transfer_fixed25','transfer_adaptive_scale_fixed25'),('transfer_fixed25','transfer_fixed_scale_learned25'),
        ('moge_fixed25','moge_adaptive_scale_fixed25'),('moge_fixed25','moge_fixed_scale_learned25'),
        ('moge_fixed2','moge_roi_readout')]
    paired={}
    for old,new in comparisons:
        for seed in [0,1]:
            akey=f'{old}_s{seed}';bkey=f'{new}_s{seed}'
            if not np.array_equal(inventories[akey],inventories[bkey]):raise ValueError('paired inventory differs')
            paired[f'{old}_to_{new}_s{seed}']=compare_errors(errors[akey],errors[bkey])
    a.output.write_text(json.dumps({'runs':runs,'paired':paired,'pose_run_count':len(sources),'unique_transfer_images':len(gt),
        'mapping_training_images':98,'repeats_do_not_increase_sample_size':True,'official_438_mainline_not_replaced':True,
        'same_scene_development_routes_not_new_scene_validation':True,
        'moge_v1_superseded':'v2 requires >=12/16 region support to avoid order-dependent boundary assignment',
        'all_correspondence_candidates_and_map_geometry_frozen_between_score_arms':True,
        'pnp_scored_hypotheses_per_run':224*5*256,'sampled_pose_hypotheses_may_differ_between_arms':True},indent=2))


if __name__=='__main__':main()
