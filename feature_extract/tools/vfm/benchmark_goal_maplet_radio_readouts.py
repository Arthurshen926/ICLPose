"""Synchronized resident-model latency of the exact RGB readout contracts."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','image_root','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=[r for r in json.loads(a.plan.read_text())['records'] if r['name'].startswith(('seq12__','seq14__'))][::22][:10]
    checkpoint=Path('/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar')
    model=RADIOFeatureExtractor(version=str(checkpoint),device=a.device);results=[]
    for row in records:
        start=time.perf_counter();path=a.image_root/row['name'][:-4].replace('__','/')
        with Image.open(path) as im:
            im=im.convert('RGB');q=row['quadrants'][0];w,h=im.size
            crop=im.crop((q%2*w//2,q//2*h//2,(q%2+1)*w//2,(q//2+1)*h//2))
            def tensor(image,size):return torch.from_numpy(np.array(image.resize(size,Image.Resampling.LANCZOS),dtype=np.float32)/255).permute(2,0,1)[None]
            inputs={'coarse':tensor(im,(1024,576)),'fine':tensor(im,(1536,864)),'roi':tensor(crop,(768,432))}
        decode=time.perf_counter()-start
        if not results:
            for mode in ['coarse','fine','roi']:model.extract(inputs[mode])
            model.extract_dual(inputs['coarse']);torch.cuda.synchronize()
        for mode in ['coarse','coarse_plus_intermediate','fine','roi']:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            output=model.extract_dual(inputs['coarse']) if mode=='coarse_plus_intermediate' else model.extract(inputs[mode])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            results.append({'name':row['name'],'mode':mode,'seconds':elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'rgb_decode_and_three_resizes_seconds':decode})
            del output
    summary={}
    for mode in ['coarse','coarse_plus_intermediate','fine','roi']:
        times=[r['seconds'] for r in results if r['mode']==mode]
        summary[mode]={'median_seconds':float(np.median(times)),'p90_seconds':float(np.percentile(times,90)),'images':len(times)}
    a.output.write_text(json.dumps({'scope':__doc__,'summary':summary,'measurements':results,'checkpoint_sha256':file_sha256(checkpoint),
        'torch_version':torch.__version__,'warmup_per_shape':True,'poses_or_labels_opened':False,
        'timing_excludes':['model loading','MoGe inference','retrieval','PnP and refinement'],
        'not_full_localization_end_to_end_latency':True},indent=2))


if __name__=='__main__':main()
