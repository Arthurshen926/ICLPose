"""Fit common-shift calibration and within-patch residual correlation on mapping."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.native_fine_measurement import fit_update_scale,compound_whiten
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids,select_offsets
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base;o=a.output;o.mkdir(parents=True,exist_ok=False)
    source=b/'native_fine_calibration_v267/mapping_rows.npz';basecal=json.load(open(b/'native_fine_calibration_v267/calibration.json'))
    with np.load(source) as z:data={k:z[k] for k in z.files};names=data.pop('source_names').astype(str)
    with np.load(b/'native_fine_v264/readout/map.npz') as z:maps=z['fine'];mm=json.loads(str(z['metadata_json']))
    with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as z:plane=np.repeat(np.arange(len(z['plane_texel_offsets'])-1),np.diff(z['plane_texel_offsets']))
    offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]]);groupids=np.full(len(data['source']),-1,int);updated=data['original'].copy();gid=0
    for src in np.unique(data['source']):
        rows=np.flatnonzero(data['source']==src);target=data['target'][rows];tok=((target[:,1]+.5)//4).astype(int)*64+((target[:,0]+.5)//4).astype(int)
        keys=np.c_[plane[data['prototype'][rows]],tok//64//8,tok%64//8];_,labels=np.unique(keys,axis=0,return_inverse=True);groups=[np.flatnonzero(labels==g) for g in np.unique(labels)];groups=[g for g in groups if len(g)>=3]
        if not groups:continue
        grids,ck=load_grids(b/'adaptive_memory_v234/fine_cache'/names[src],mm['projection_sha256'])
        if ck!=mm['checkpoint_sha256']:raise ValueError('checkpoint differs')
        probes=data['original'][rows,None]+offsets;center=np.c_[(tok%64)*4+1.5,(tok//64)*4+1.5]
        sim=np.sum(sample_grid(grids[1],probes)*maps[data['prototype'][rows],None].astype(np.float32),axis=-1);sim[(np.abs(probes-center[:,None])>2+1e-8).any(2)]=-np.inf
        best=select_offsets(sim,groups,True);updated[rows]=probes[np.arange(len(rows)),best]
        for group in groups:groupids[rows[group]]=gid;gid+=1
    keep=groupids>=0;data={k:v[keep] for k,v in data.items()};updated=updated[keep];groupids=groupids[keep];routes=np.array([names[i].split('__')[0] for i in data['source']]);train=np.isin(routes,basecal['training_routes']);weights=np.zeros(len(train))
    for src in np.unique(data['source']):mask=data['source']==src;weights[mask]=1./mask.sum()
    coeff=fit_update_scale(data['original'][train],updated[train],data['target'][train],data['variance'][train],weights[train]);res=data['original']+coeff['alpha']*(updated-data['original'])-data['target'];standard=res/np.sqrt(data['variance'][:,None]*coeff['variance_scale'])
    # View-balanced pair-moment estimator; zero mean is the explicit error model.
    numerator=denominator=0.
    for g in np.unique(groupids[train]):
        ix=np.flatnonzero(groupids==g);r=standard[ix];n=len(ix);w=weights[ix].mean();numerator+=w*(np.sum(r.sum(0)**2)-np.sum(r*r));denominator+=w*(n-1)*np.sum(r*r)
    rho=float(np.clip(numerator/max(denominator,1e-12),0,.95));report={}
    for split,mask in [('train',train),('heldout',~train)]:
        e=np.linalg.norm(res[mask],axis=1);rr=standard[mask];ids=groupids[mask];white=compound_whiten(rr,ids,rho)
        logdet=0.
        for g in np.unique(ids):
            n=int(np.sum(ids==g));logdet+=(n-1)*np.log(1-rho)+np.log(1+(n-1)*rho)
        logbase=np.log(2*np.pi*data['variance'][mask]*coeff['variance_scale']).sum()
        report[split]=dict(rows=int(mask.sum()),views=len(np.unique(data['source'][mask])),mean_pixel_error=float(e.mean()),median_pixel_error=float(np.median(e)),independent_nll_per_token=float((logbase+.5*np.sum(rr**2))/mask.sum()),correlated_nll_per_token=float((logbase+logdet+.5*np.sum(white**2))/mask.sum()))
    np.savez_compressed(o/'mapping_joint_rows.npz',**data,joint_update=updated,group_ids=groupids,source_names=names)
    result=dict(alpha=coeff['alpha'],variance_scale=coeff['variance_scale'],rho=rho,report=report,training_routes=basecal['training_routes'],heldout_routes=basecal['heldout_routes'],query_ground_truth_read=False,heldout_used_to_fit=False,conditional_correct_identity_only=True,source_mapping_rows_sha256=file_sha256(source),map_sha256=basecal['map_sha256'],rho_estimator='view-balanced uncentered pair residual moment; clamped [0,.95]',group_contract='same physical plane and 8x8 coarse token block, >=3; mapping has no query MoGe subdivision')
    result['content_sha256']=canonical_json_sha256(result);(o/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
