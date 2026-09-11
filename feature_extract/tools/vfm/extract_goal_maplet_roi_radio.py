"""Actually extract only the frozen RGB ROIs; one quarter fine-image pixels/query."""
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
    for k in ['plan','image_root','projection','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());rows=plan['records'][a.shard::a.shards]
    proj,_=_load_projection(a.projection);proj=torch.as_tensor(proj,device=a.device)
    checkpoint=Path('/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar')
    checkpoint_hash=file_sha256(checkpoint);projection_hash=file_sha256(a.projection);plan_hash=file_sha256(a.plan)
    extractor=RADIOFeatureExtractor(version=str(checkpoint),device=a.device);times=[]
    for i,row in enumerate(rows):
        target=a.output/row['name']
        if target.exists():raise FileExistsError(target)
        path=a.image_root/row['name'][:-4].replace('__','/')
        with Image.open(path) as im:
            im=im.convert('RGB');w,h=im.size;crops=[]
            for q in row['quadrants']:
                x=q%2;y=q//2;crop=im.crop((x*w//2,y*h//2,(x+1)*w//2,(y+1)*h//2)).resize((768,432),Image.Resampling.LANCZOS)
                crops.append(torch.from_numpy(np.array(crop,dtype=np.float32)/255).permute(2,0,1))
        if not crops:continue
        torch.cuda.synchronize();start=time.perf_counter()
        result=extractor.extract_batch(torch.stack(crops));grid=F.normalize(result['local'].float(),dim=1)
        grid=F.normalize(torch.einsum('dc,bchw->bhwd',proj,grid),dim=-1).cpu().numpy().astype(np.float16)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;times.append([len(crops),elapsed])
        if grid.shape[1:]!=(27,48,64):raise ValueError('ROI resolution drift')
        np.savez_compressed(target,quadrants=np.array(row['quadrants']),features=grid,
            metadata_json=np.asarray(json.dumps({'plan_sha256':plan_hash,'rgb_sha256':file_sha256(path),
            'checkpoint_sha256':checkpoint_hash,'projection_sha256':projection_hash,
            'query_GT_opened':False,'seconds':elapsed,'input_size':[768,432]})))
        if (i+1)%50==0:print('ROI',a.device,i+1,'/',len(rows),flush=True)
    (a.output/f'shard{a.shard}.json').write_text(json.dumps({'timings':times,'plan_sha256':file_sha256(a.plan)},indent=2))


if __name__=='__main__':main()
