"""Reproduce v415 portable core on all five official Cambridge test splits.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m
feature_extract.tools.vfm.run_cambridge_core_suite --stage inventory
Run stages in order: inventory, features, geometry, map, infer, control, evaluate.
Local pretrained assets and the original RGB datasets are prerequisites.
"""
import argparse,json,os,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
ROOT=Path('output/cambridge_core_v415')
SCENES=['GreatCourt','KingsCollege','OldHospital','ShopFacade','StMarysChurch']

def inventory():
    ROOT.mkdir(exist_ok=True,parents=True);report={}
    for scene in SCENES:
        root=Path('/hy-tmp/Cambridge_stdloc')/scene;lists={}
        for split in ['train','test']:
            lists[split]=sorted({l.split()[0] for l in (root/f'dataset_{split}.txt').read_text().splitlines() if len(l.split())==8 and l.split()[0].endswith('.png')})
        assert not set(lists['train'])&set(lists['test']);groups={}
        for n in lists['train']:groups.setdefault(n.split('/')[0],[]).append(n)
        allocations={s:0 for s in groups}
        for _ in range(min(120,len(lists['train']))):
            s=min((s for s in groups if allocations[s]<len(groups[s])),key=lambda s:(allocations[s]/len(groups[s]),s));allocations[s]+=1
        mapping=sorted(n for s,ns in groups.items() for n in np.array(ns)[np.linspace(0,len(ns)-1,allocations[s],dtype=int)])
        dest=ROOT/scene;dest.mkdir(exist_ok=True);allnames=[n.replace('/','__')+'.npz' for n in mapping+lists['test']];assert len(allnames)==len(set(allnames))
        for label,ns in [('mapping',mapping),('test',lists['test'])]:(dest/f'{label}_names.json').write_text(json.dumps([n.replace('/','__')+'.npz' for n in ns],indent=2))
        (dest/'feature_names.json').write_text(json.dumps(allnames));report[scene]=dict(train_total=len(lists['train']),map_images=len(mapping),test_images=len(lists['test']),mapping_allocation=allocations,split_sha256={s:file_sha256(root/f'dataset_{s}.txt') for s in lists})
    (ROOT/'dataset_inventory.json').write_text(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['inventory','features','geometry','map','infer','control','evaluate','readout','evaluate-readout','summarize']);p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=[260901,260902]);a=p.parse_args()
    if a.stage=='inventory':inventory();return
    env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def call(module,args,log):
        cmd=[os.sys.executable,'-m','feature_extract.tools.vfm.'+module,*map(str,args)]
        with open(log,'w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if a.stage=='summarize':
        for name in ['summarize_cambridge_core_benchmark','summarize_cambridge_readout_control']:call(name,[],ROOT/(name+'.log'))
        return
    if a.stage in ['evaluate','evaluate-readout']:
        for seed in a.seeds:call('report_cambridge_readout_control' if a.stage=='evaluate-readout' else 'report_cambridge_core_benchmark',['--seed',seed],ROOT/f'{a.stage}_seed{seed}.log')
        return
    for scene in a.scenes:
        r=ROOT/scene
        if a.stage=='features':call('extract_goal_maplet_fine_radio',['--names',r/'feature_names.json','--image_root',Path('/hy-tmp/Cambridge_stdloc')/scene,'--radio_projection',Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1/stmarys_chart_local_radio_projection_64d_v2.npz'),'--output',r/'features','--device',a.device],r/'reproduce_features.log')
        elif a.stage=='geometry':call('cambridge_core_prepare',['--scene',scene,'--device',a.device],r/'reproduce_geometry.log')
        elif a.stage=='map':call('cambridge_core_benchmark',['build','--scene',scene],r/'reproduce_map.log')
        else:
            for seed in a.seeds:
                def shard(i):
                    module={'control':'cambridge_core_multistart_control','readout':'cambridge_core_readout_control'}.get(a.stage,'cambridge_core_benchmark');prefix=[] if a.stage in ['control','readout'] else ['infer'];call(module,[*prefix,'--scene',scene,'--shard',i,'--shards',a.workers,'--seed',seed],r/f'reproduce_{a.stage}_seed{seed}_{i}.log')
                with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:list(pool.map(shard,range(a.workers)))
if __name__=='__main__':main()
