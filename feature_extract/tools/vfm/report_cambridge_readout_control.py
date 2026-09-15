"""Evaluate the explicitly post-label common-readout diagnostic."""
import argparse,json
import numpy as np
from pathlib import Path
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT
from feature_extract.tools.vfm.report_cambridge_core_benchmark import SCENES,paired
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
ARMS=['regional_global','regional_global_field','pooled_global','pooled_global_field']

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();seal={};report={}
    for scene in SCENES:
        r=ROOT/scene;names=json.load(open(r/'test_names.json'));d=r/f'readout_seed{a.seed}';assert {p.name for p in d.glob('*.npz')}==set(names);seal[scene]={n:file_sha256(d/n) for n in names}
    (ROOT/f'readout_seal_seed{a.seed}.json').write_text(json.dumps(seal,indent=2))
    for scene in SCENES:
        r=ROOT/scene;names=json.load(open(r/'test_names.json'));gt={x.image_id:x.pose_w2c for x in parse_cambridge_pose_file(Path('/hy-tmp/Cambridge_stdloc')/scene/'dataset_test.txt')};errors={k:[] for k in ARMS}
        for n in names:
            path=r/f'readout_seed{a.seed}'/n;assert file_sha256(path)==seal[scene][n]
            with np.load(path) as z:
                for arm in ARMS:errors[arm].append(_pose_error(z[arm],gt[n[:-4].replace('__','/')]) if np.isfinite(z[arm]).all() else (np.inf,np.inf))
        errors={k:np.array(v) for k,v in errors.items()}
        with np.load(r/f'errors_seed{a.seed}.npz') as z:
            assert np.array_equal(z['names'],np.array(names));errors.update({k:z[k] for k in ['regional','pooled_multistart']})
        pairs=[('regional','regional_global'),('pooled_multistart','pooled_global'),('regional_global','regional_global_field'),('pooled_global','pooled_global_field'),('pooled_global','regional_global')]
        report[scene]=dict(metrics={k:metrics(v) for k,v in errors.items()},paired_comparisons={b+'_vs_'+a:paired(errors[a],errors[b]) for a,b in pairs});np.savez_compressed(r/f'readout_errors_seed{a.seed}.npz',names=np.array(names),**errors)
    macro={arm:{key:float(np.mean([report[s]['metrics'][arm]['recall_percent'][key] for s in SCENES])) for key in report[SCENES[0]]['metrics'][arm]['recall_percent']} for arm in ARMS}
    (ROOT/f'readout_report_seed{a.seed}.json').write_text(json.dumps(dict(post_label_exploratory_design=True,seed=a.seed,scenes=report,macro_scene_recall_percent=macro,preceding_report_sha256=file_sha256(ROOT/'report_seed260901.json')),indent=2));print(json.dumps(macro,indent=2))
if __name__=='__main__':main()
