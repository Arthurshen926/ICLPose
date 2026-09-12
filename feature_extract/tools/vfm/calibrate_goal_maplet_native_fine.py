"""Mapping-only cross-route conditional coordinate calibration for native anchors.

Visible identity teachers are projected into other mapping routes. This tests
coordinate readout conditional on a correct identity, not retrieval accuracy.
Independent mapping routes audit frozen coefficients; query GT is never read.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.native_fine_measurement import fit_update_scale
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import load_mapping_subtoken_head
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import _project
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=True)
    if (o/'calibration.json').exists():raise FileExistsError(o/'calibration.json')
    atlaspath=b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';mapfile=b/'native_fine_v264/readout/map.npz';lineagefile=b/'native_fine_v264/mapping_lineage.npz';headfile=b/'stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz'
    with np.load(mapfile) as z:maps={k:z[k] for k in ['world_points','fine','available']};meta=json.loads(str(z['metadata_json']))
    with np.load(atlaspath) as z:features=z['radio_features'].astype(np.float32)
    with np.load(lineagefile) as z:names=z['source_names'].astype(str);sources=z['prototype_source_and_cell'][:,0]
    routes=np.array([n.split('__')[0] for n in names]);train=['seq1','seq2','seq4','seq6'];held=['seq7','seq8','seq9','seq11'];head,hm=load_mapping_subtoken_head(headfile);head.eval();torch.set_num_threads(1)
    selected=[]
    for route in train+held:
        indices=np.flatnonzero(routes==route);indices=np.array([i for i in indices if (b/'adaptive_memory_v234/fine_cache'/names[i]).exists()])
        selected.extend(indices[np.linspace(0,len(indices)-1,min(20,len(indices)),dtype=int)])
    frozen=dict(training_routes=train,heldout_routes=held,maximum_views_per_route=20,maximum_rows_per_view=128,depth_tolerance='abs(map projected z - 2DGS dominant z)<=max(0.10m,0.01*z)',source_route_exclusion=True,identity_teacher='visible native projected anchor within token; highest coarse cosine identity per token',query_ground_truth_read=False,map_sha256=file_sha256(mapfile),head_sha256=file_sha256(headfile))
    (o/'protocol.json').write_text(json.dumps(frozen,indent=2));chunks={k:[] for k in ['original','update','target','variance','source','prototype']};ledger={}
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
    for num,src in enumerate(selected):
        name=names[src];path=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/name
        with np.load(path) as z:pose=z['pose_w2c'];K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']));depth=z['dominant_depth'].astype(float)
        xy,cam=_project(pose,maps['world_points'],K,k1)
        valid=maps['available']&(routes[sources]!=routes[src])&(cam[:,2]>0)&np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<255)&(xy[:,1]>=0)&(xy[:,1]<143)
        rows=np.flatnonzero(valid);pix=np.floor(xy[rows]+.5).astype(int);dz=depth[pix[:,1],pix[:,0]];rows=rows[np.isfinite(dz)&(dz>0)&(np.abs(dz-cam[rows,2])<=np.maximum(.1,.01*cam[rows,2]))]
        cache=b/'adaptive_memory_v234/fine_cache'/name;grids,ck=load_grids(cache,meta['projection_sha256'])
        if ck!=meta['checkpoint_sha256']:raise ValueError('mapping checkpoint mismatch')
        tok=((xy[rows,1]+.5)//4).astype(int)*64+((xy[rows,0]+.5)//4).astype(int)
        q=grids[0].reshape(-1,64);score=np.sum(q[tok]*features[rows],axis=1);order=np.lexsort((rows,-score,tok));rows=rows[order];tok=tok[order];_,first=np.unique(tok,return_index=True);rows=rows[first];tok=tok[first]
        if len(rows)>128:sel=np.linspace(0,len(rows)-1,128,dtype=int);rows=rows[sel];tok=tok[sel]
        if not len(rows):continue
        with torch.no_grad():mean,var,_=head(torch.from_numpy(q[tok]),torch.from_numpy(features[rows]),torch.from_numpy(tok))
        offset=mean.numpy().astype(float)
        if hm.get('coordinate_affine_matrix') is None:offset*=float(hm.get('coordinate_shrinkage',1.))
        else:offset=offset@np.asarray(hm['coordinate_affine_matrix'])+np.asarray(hm['coordinate_affine_bias_px'])
        center=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5];original=center+np.clip(offset,-2,2);probes=original[:,None]+offsets
        sim=np.sum(sample_grid(grids[1],probes)*maps['fine'][rows,None].astype(np.float32),axis=-1);sim[(np.abs(probes-center[:,None])>2+1e-8).any(2)]=-np.inf
        best=select_offsets(sim,[],False);update=probes[np.arange(len(rows)),best]
        for k,v in dict(original=original,update=update,target=xy[rows],variance=var.numpy().reshape(-1)*float(hm.get('measurement_variance_scale',1.)),source=np.full(len(rows),src),prototype=rows).items():chunks[k].append(v)
        ledger[name]=dict(camera_sha256=file_sha256(path),cache_sha256=file_sha256(cache))
        if (num+1)%20==0:print('mapping',num+1,flush=True)
    arr={k:np.concatenate(v) for k,v in chunks.items()};np.savez_compressed(o/'mapping_rows.npz',**arr,source_names=names)
    istrain=np.isin(routes[arr['source']],train);weights=np.zeros(len(arr['source']))
    for src in np.unique(arr['source']):mask=arr['source']==src;weights[mask]=1./mask.sum()
    coeff=fit_update_scale(*[arr[k][istrain] for k in ['original','update','target','variance']],weights[istrain]);report={}
    for split,mask in [('train',istrain),('heldout',~istrain)]:
        stats={}
        for label,alpha in [('head',0.),('fine',1.),('shrink',coeff['alpha'])]:
            residual=arr['original'][mask]+alpha*(arr['update'][mask]-arr['original'][mask])-arr['target'][mask];err=np.linalg.norm(residual,axis=1);v=arr['variance'][mask]*(coeff['variance_scale'] if label=='shrink' else 1)
            stats[label]=dict(mean_pixel_error=float(err.mean()),median_pixel_error=float(np.median(err)),normalized_residual_squared_mean=float(np.mean(err**2/v)),mean_isotropic_gaussian_nll=float(np.mean(np.log(2*np.pi*v)+err**2/(2*v))))
        report[split]=dict(rows=int(mask.sum()),views=len(np.unique(arr['source'][mask])),metrics=stats)
    payload=dict(frozen,**coeff,report=report,mapping_rows_sha256=file_sha256(o/'mapping_rows.npz'),source_ledger=ledger,conditional_correct_identity_only=True,heldout_used_to_fit=False)
    payload['content_sha256']=canonical_json_sha256(payload);(o/'calibration.json').write_text(json.dumps(payload,indent=2));print(json.dumps(dict(coefficients=coeff,report=report)),flush=True)

if __name__=='__main__':main()
