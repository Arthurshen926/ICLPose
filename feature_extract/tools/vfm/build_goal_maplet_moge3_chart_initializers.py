"""Pose-free MoGe-3 inference for a frozen MAtCha chart image inventory."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import cv2
import numpy as np
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def focal_for_source(camera_focal_px,camera_canvas_width,source_width):
 if camera_canvas_width<=0 or source_width<=0 or camera_focal_px<=0:raise ValueError('focal/canvas/source dimensions must be positive')
 return float(camera_focal_px)*float(source_width)/float(camera_canvas_width)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cameras',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--model_id',default='Ruicheng/moge-3-vitl');p.add_argument('--resolution_level',type=int,default=9);p.add_argument('--refine_steps',type=int,default=3);p.add_argument('--camera_focal_canvas_width',type=int,required=True,help='Pixel width on which cameras.json focals are defined (512 for the frozen MAtCha charts).');p.add_argument('--routes',nargs='+',default=['seq1','seq2','seq4','seq6','seq7','seq8','seq9','seq11']);a=p.parse_args()
 if a.camera_focal_canvas_width<=0:raise ValueError('camera focal canvas width must be positive')
 if a.output_dir.exists():raise FileExistsError('refusing to reuse MoGe-3 chart initializer directory')
 camera=json.loads(a.cameras.read_text());inventory=[]
 for source,focal in zip(camera['filepaths'],camera['focals']):
  path=Path(source);route=path.name.split('__',1)[0]
  if route in set(a.routes):inventory.append((path,float(focal)))
 inventory=sorted(inventory,key=lambda x:x[0].name)
 import torch
 import torch.nn.functional as F
 from huggingface_hub import snapshot_download
 from moge.model.v3 import MoGeModel
 model=MoGeModel.from_pretrained(a.model_id).to(a.device).eval();snapshot=Path(snapshot_download(a.model_id,local_files_only=True));weight=snapshot/'model.pt';a.output_dir.mkdir(parents=True);rows=[];torch.cuda.reset_peak_memory_stats(a.device)
 for index,(path,focal) in enumerate(inventory):
  bgr=cv2.imread(str(path));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];source_focal=focal_for_source(focal,a.camera_focal_canvas_width,w);fov=math.degrees(2*math.atan(w/(2*source_focal)));tensor=torch.as_tensor(rgb,dtype=torch.float32,device=a.device).permute(2,0,1)/255
  started=time.perf_counter()
  with torch.inference_mode():pred=model.infer(tensor,resolution_level=a.resolution_level,fov_x=fov,use_fp16=False,apply_mask=True,refine_steps=a.refine_steps)
  def resize(value,channels,mode='bilinear'):
   x=value.float();x=x.permute(2,0,1)[None] if channels else x[None,None];x=F.interpolate(x,size=(144,256),mode=mode,align_corners=False if mode!='nearest' else None);return x[0].permute(1,2,0).cpu().numpy() if channels else x[0,0].cpu().numpy()
  points=resize(pred['points'],True).astype(np.float32);depth=resize(pred['depth'],False).astype(np.float32);normal=resize(pred['normal'],True).astype(np.float32);normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-8);valid=resize(pred['mask'].float(),False,'nearest')>.5;valid&=np.isfinite(points).all(2)&np.isfinite(depth)&(depth>0);points[~valid]=np.nan;depth[~valid]=np.nan;normal[~valid]=0
  meta={'artifact_type':'goal_maplet_moge3_chart_initializer_v2','source_name':path.name,'source_image_file_sha256':sha(path),'camera_focal_canvas_px':focal,'camera_focal_canvas_width':a.camera_focal_canvas_width,'source_focal_px':source_focal,'source_width':w,'source_height':h,'fov_x_degrees':fov,'model_id':a.model_id,'resolution_level':a.resolution_level,'refine_steps':a.refine_steps,'uses_camera_pose':False,'uses_query_or_ground_truth':False};meta['content_sha256']=canonical(meta);out=a.output_dir/(path.name+'.npz');np.savez_compressed(out,points_camera=points,depth_camera=depth,normal_camera=normal,valid=valid,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)));rows.append({'name':path.name,'file_sha256':sha(out),'content_sha256':meta['content_sha256'],'valid_fraction':float(valid.mean()),'median_depth_m':float(np.nanmedian(depth)),'seconds':time.perf_counter()-started});print(f'{index+1}/{len(inventory)} {path.name}',flush=True)
 manifest={'artifact_type':'goal_maplet_moge3_chart_initializer_run_v2','cameras_file_sha256':sha(a.cameras),'consumed_camera_fields':['filepaths','focals'],'explicitly_not_consumed_camera_fields':['cams2world'],'camera_focal_canvas_width':a.camera_focal_canvas_width,'focal_rescaling_contract':'source_focal=camera_focal*source_width/camera_focal_canvas_width','allowed_routes':sorted(a.routes),'chart_count':len(rows),'model_id':a.model_id,'model_weight_file_sha256':sha(weight),'resolution_level':a.resolution_level,'refine_steps':a.refine_steps,'uses_camera_pose':False,'uses_query_or_ground_truth':False,'rows':rows,'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated(a.device))};manifest['content_sha256']=canonical(manifest);(a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
