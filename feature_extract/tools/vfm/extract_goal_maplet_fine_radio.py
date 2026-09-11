"""Matched map/query real-RGB RADIO readouts, without opening poses or labels.

The 1536x864 pass starts from native RGB, never from cached coarse tokens.
Intermediate features use a fixed data-independent orthogonal projection.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import _load_projection
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['names','image_root','radio_projection','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    names=json.loads(a.names.read_text())[a.shard::a.shards]
    if a.limit:names=names[:a.limit]
    projection,_=_load_projection(a.radio_projection)
    rng=np.random.default_rng(260910)
    intermediate=np.linalg.qr(rng.normal(size=(1280,128)))[0].astype(np.float32).T
    final_proj=torch.as_tensor(projection,device=a.device);mid_proj=torch.as_tensor(intermediate,device=a.device)
    extractor=RADIOFeatureExtractor(version=str(a.checkpoint.resolve()),device=a.device)
    contract={'checkpoint_sha256':file_sha256(a.checkpoint),'final_projection_sha256':file_sha256(a.radio_projection),
              'intermediate_index':-6,'intermediate_projection':'orthogonal_gaussian_seed260910_128d',
              'coarse_rgb_size':[1024,576],'fine_rgb_size':[1536,864],
              'poses_or_labels_opened':False,'native_rgb_resize':'PIL_LANCZOS','format':'goal_fine_radio_v2'}
    def project(x,proj):
        x=F.normalize(x.float(),dim=0)
        return F.normalize(torch.einsum('dc,chw->hwd',proj,x),dim=-1).cpu().numpy().astype(np.float16)
    timings=[]
    for i,name in enumerate(names):
        output=a.output/name
        if output.exists():
            with np.load(output) as z:
                meta=json.loads(str(z['metadata_json']))
                if any(meta.get(k)!=v for k,v in contract.items() if k!='format'):raise ValueError('existing extraction contract differs')
                complete=meta.get('format')==contract['format'] and 'coarse_final' in z.files
            if complete:continue
        path=a.image_root/name[:-4].replace('__','/')
        with Image.open(path) as im:
            im=im.convert('RGB');native=list(im.size)
            if im.width<1536 or im.height<864:raise ValueError('fine readout requires additional native RGB detail')
            def tensor(size):
                return torch.from_numpy(np.array(im.resize(size,Image.Resampling.LANCZOS),dtype=np.float32)/255).permute(2,0,1)[None]
            coarse=tensor((1024,576));fine=tensor((1536,864))
        torch.cuda.synchronize();t=time.perf_counter()
        dual=extractor.extract_dual(coarse,fine_intermediate_index=-6)
        coarse_mid=project(dual['fine'],mid_proj);coarse_final=project(dual['coarse'],final_proj);del dual
        torch.cuda.synchronize();tm=time.perf_counter()
        result=extractor.extract(fine);fine_final=project(result['local'],final_proj);del result
        torch.cuda.synchronize();tf=time.perf_counter()
        if coarse_mid.shape!=(36,64,128) or fine_final.shape!=(54,96,64):raise ValueError('RADIO unexpectedly resized RGB')
        meta={**contract,'image':name,'native_rgb_size':native,'rgb_sha256':file_sha256(path),
              'intermediate_seconds':tm-t,'fine_seconds':tf-tm}
        np.savez_compressed(output,coarse_final=coarse_final,coarse_intermediate=coarse_mid,fine_final=fine_final,metadata_json=np.asarray(json.dumps(meta)))
        timings.append([tm-t,tf-tm])
        if (i+1)%10==0 or i==0:print(json.dumps({'device':a.device,'done':i+1,'total':len(names),'seconds_mean':np.mean(timings,axis=0).tolist()}),flush=True)
    (a.output/f'shard{a.shard}.json').write_text(json.dumps({'contract':contract,'images':names,'timings':timings},indent=2))


if __name__=='__main__':main()
