"""Scene-neutral camera-only inventory and train-only rendered surface anchors."""
from pathlib import Path
import argparse,json,time,numpy as np
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView,load_gaussian_vfm_source_from_ply
from feature_extract.vfm.localization_v6.primitive_contributors import clean_primitive_surface_elements,render_primitive_contributors
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def calibration(scene,root):
    folder=Path('/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px')/scene/'empty_all'
    camera={}
    for line in (folder/'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        x=line.split();assert x[1]=='SIMPLE_RADIAL';camera[int(x[0])]=(int(x[2]),int(x[3]),list(map(float,x[4:])))
    names={}
    for line in (folder/'images.txt').read_text().splitlines():
        x=line.split()
        if len(x)==10 and not line.startswith('#'):
            # Read name and camera ID only; do not convert extrinsic fields.
            names[x[9]]=int(x[8])
    selected=json.load(open(root/'feature_names.json'));Ks=[];ks=[]
    for n in selected:
        image=n[:-4].replace('__','/');w,h,p=camera[names[image]];Ks.append([[p[0]*256/w,0,p[1]*256/w],[0,p[0]*144/h,p[2]*144/h],[0,0,1]]);ks.append(p[3])
    path=root/'camera_only.npz';np.savez_compressed(path,names=np.array(selected),camera_matrices=np.array(Ks),radial_k1=np.array(ks),metadata_json=np.array(json.dumps(dict(extrinsics_used=False,source_sha256={str(p):file_sha256(p) for p in [folder/'cameras.txt',folder/'images.txt']}))))
    return {n:(K,k) for n,K,k in zip(selected,Ks,ks)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);a=p.parse_args();root=Path('output/cambridge_core_v415')/a.scene;cams=calibration(a.scene,root);names=json.load(open(root/'mapping_names.json'));train=Path('/hy-tmp/Cambridge_stdloc')/a.scene/'dataset_train.txt';poses={x.image_id:x.pose_w2c for x in parse_cambridge_pose_file(train)}
    sources=Path('/hy-tmp/Cambridge_MAtCha_full_v1')/a.scene
    rel={'GreatCourt':'candidates/retained_512_appearance_3000/free_gaussians/point_cloud/iteration_33000/point_cloud.ply','KingsCollege':'candidates/retained_512_appearance_3000/free_gaussians/point_cloud/iteration_33000/point_cloud.ply','OldHospital':'free_gaussians/point_cloud/iteration_49000/point_cloud.ply','ShopFacade':'free_gaussians/point_cloud/iteration_30000/point_cloud.ply','StMarysChurch':'candidates/user_previous_appearance_3000/free_gaussians/point_cloud/iteration_33000/point_cloud.ply'};ply=sources/rel[a.scene];source=load_gaussian_vfm_source_from_ply(ply);elements=clean_primitive_surface_elements(source,np.flatnonzero(source.opacity>=.05));out=root/'geometry_v1';out.mkdir(exist_ok=True)
    meta=dict(scene=a.scene,mapping_pose_file_sha256=file_sha256(train),source_ply=str(ply),source_ply_sha256=file_sha256(ply),query_pose_or_labels_read=False,minimum_source_opacity=.05,minimum_dominant_weight=.05,minimum_incidence=.2,minimum_camera_depth_m=1.,maximum_disk_sigma=3.,coordinate_readout='ray intersection with dominant oriented disk plane at exact RADIO token center')
    if a.limit:names=names[:a.limit]
    for i,n in enumerate(names):
        dest=out/n
        if dest.exists():continue
        image=n[:-4].replace('__','/');pose=poses[image];K,k=cams[n];K=np.array(K);assert np.isfinite(pose).all();camera=ColmapCamera(1,1,256,144,(K[0,0],K[1,1],K[0,2],K[1,2]));start=time.perf_counter();depth=source.xyz@pose[2,:3]+pose[2,3];visible=clean_primitive_surface_elements(source,np.flatnonzero((source.opacity>=.05)&(depth>=1.)));buf=render_primitive_contributors(visible,GaussianVFMFeatureView(image,np.zeros((1,144,256),np.float32),pose,camera),width=256,height=144,top_k=1,device=a.device)
        t=np.arange(2304);xy=np.c_[t%64*4+1.5,t//64*4+1.5];pix=np.rint(xy).astype(int);ids=buf.dominant_ids[pix[:,1],pix[:,0]];weights=buf.dominant_weights[pix[:,1],pix[:,0]];keep=(ids>=0)&(weights>=.05);t,xy,ids=t[keep],xy[keep],ids[keep];C=-pose[:3,:3].T@pose[:3,3];rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(K).T@pose[:3,:3];normal=source.normal[ids];den=np.sum(normal*rays,1);distance=np.sum(normal*(source.xyz[ids]-C),1)/np.where(np.abs(den)>1e-8,den,1);incidence=np.abs(den)/np.maximum(np.linalg.norm(rays,axis=1),1e-8);world=C+distance[:,None]*rays;from feature_extract.vfm.vfm_2dgs_mapping import _surface_tangent_axes_and_scales
        u,v,su,sv=_surface_tangent_axes_and_scales(source,ids,normal);delta=world-source.xyz[ids];sigma2=(np.sum(delta*u,1)/su)**2+(np.sum(delta*v,1)/sv)**2;valid=(distance>=1.)&np.isfinite(world).all(1)&(incidence>=.2)&(sigma2<=9.)
        np.savez_compressed(dest,tokens=t[valid],world=world[valid].astype(np.float32),normals=normal[valid].astype(np.float32),primitive_ids=ids[valid],pose_w2c=pose,K=K,k1=k,metadata_json=np.array(json.dumps(meta)));print(a.scene,i+1,len(t[valid]),round(time.perf_counter()-start,2),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(**meta,completed_names=[n for n in names if (out/n).exists()]),indent=2))


if __name__=='__main__':main()
