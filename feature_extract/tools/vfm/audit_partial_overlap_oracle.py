"""Post-evaluation ideal visibility/observability and exact token attrition.

GT and the manual tower outline are diagnostic only. None of these outputs may
be consumed by inference, training, threshold calibration, or candidate choice.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from matplotlib.path import Path as Polygon
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map,project
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.surface_configuration_matching import configuration_match
from feature_extract.tools.vfm.token_hypothesis_ransac import solve
from feature_extract.tools.vfm.report_goal_maplet_pose_metrics import metrics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=False);b=a.base;maps,meta,_=load_map(b);world=maps['world'];name='seq13__frame00142.png.npz';s='shard1'
    cmd=dict(json.load(open(b/'native_hybrid_mainline_v290'/f'{s}_risk_stop/protocol.json'))['commands'])['pnp'];gt=Path(cmd[cmd.index('--query_contributors')+1])
    with np.load(gt/name) as f:truth=f['pose_w2c'];depth=f['dominant_depth']
    with np.load(b/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz') as f:i=f['names'].astype(str).tolist().index(name);K=f['camera_matrices'][i];k1=float(f['radial_k1'][i])
    with np.load(b/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz') as f:texel=f['texel_identity']
    poly=np.array(json.load(open(b/'configuration_research_v361/occlusion_stage_diagnosis.json'))['manual_roi_diagnostic_only']);uv,z,inside=project(world,truth,K,k1);pix=np.clip(np.rint(np.nan_to_num(uv)).astype(int),[0,0],[255,143]);dep=depth[pix[:,1],pix[:,0]];visible=inside&np.isfinite(dep)&(dep>0)&(np.abs(dep-z)<=np.maximum(.1,.01*dep));tower=Polygon(poly).contains_points(uv)
    groups={}
    for key,mask in [('tower_only',visible&tower),('all_visible',visible)]:
        rows=np.flatnonzero(mask);_,first=np.unique(texel[rows],return_index=True);rows=rows[np.sort(first)];token=np.clip(np.floor(uv[rows]/4).astype(int),[0,0],[63,35]);token=token[:,1]*64+token[:,0];_,first=np.unique(token,return_index=True);rows=rows[np.sort(first)]
        if len(rows)>256:rows=rows[np.linspace(0,len(rows)-1,256,dtype=int)]
        groups[key]=rows
    # Same support count separates spatial/depth diversity from point count.
    candidates=groups['all_visible'];chosen=[int(np.argmin(np.linalg.norm(uv[candidates]-np.mean(uv[candidates],axis=0),axis=1)))];distance=np.linalg.norm(uv[candidates]-uv[candidates[chosen[0]]],axis=1)
    while len(chosen)<len(groups['tower_only']):
        distance[chosen]=-1;j=int(np.argmax(distance));chosen.append(j);distance=np.minimum(distance,np.linalg.norm(uv[candidates]-uv[candidates[j]],axis=1))
    groups['distributed_equal_count']=candidates[chosen]
    oracle={}
    for key,rows in groups.items():
        _,jac=cv2.projectPoints(world[rows],cv2.Rodrigues(truth[:3,:3])[0],truth[:3,3],K,np.array([k1,0.,0.,0.,0.]));J=jac[:,:6];scaled=J/np.maximum(np.linalg.norm(J,axis=0),1e-12);singular=np.linalg.svd(scaled,compute_uv=False);condition=float(singular[0]/max(singular[-1],1e-15));records={}
        for noise in [0.,.25,.5,1.]:
            errors=[]
            for seed in range(20):
                pixels=uv[rows]+np.random.default_rng(402+seed).normal(0,noise,(len(rows),2));pose=solve(world[rows],np.arange(len(rows)),K,k1,np.arange(len(rows)),pixels=pixels,iterations=1250,seed=402+seed,hypothesis_budget=250);errors.append(_pose_error(pose,truth))
            records[str(noise)]=dict(metrics=metrics(errors),errors=errors)
        oracle[key]=dict(anchors=len(rows),image_span_px=np.ptp(uv[rows],axis=0).tolist(),normalized_jacobian_condition=condition,noise_coarse_pixels=records)
    (o/'ideal_support.json').write_text(json.dumps(dict(query=name,diagnostic_only=True,supervision='shared-map rendered depth proxy',results=oracle),indent=2));print(oracle,flush=True)
    # Reproduce the old candidate selection exactly, including all 256 tokens.
    r=next(r for r in json.load(open(b/'partial_anchor_relative16_v384/shard1_joint_audit.json')) if r['name']==name);ts=np.array(r['sampled_tokens']);xy=np.c_[ts%64*4+1.5,ts//64*4+1.5];diagnostic=Polygon(poly).contains_points(xy)
    cmd=dict(json.load(open(b/'diverse_candidate_retention_v307/shard1_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1]);qp,qn,qv=moge_tokens(md/name);grids,_=load_grids(b/'native_fine_v264/query_cache'/name,meta['projection_sha256']);q=normalise(grids[0].reshape(2304,64));full=q[ts]@maps['coarse_map'].T;null=full.max(1)-.1;result=[]
    truth_masks={}
    for i in np.flatnonzero(diagnostic):
        dep=depth[int(round(xy[i,1])),int(round(xy[i,0]))];good=inside&(np.linalg.norm(uv-xy[i],axis=-1)<=4)&np.isfinite(dep)&(dep>0)&(np.abs(z-dep)<=max(.1,.01*dep));truth_masks[int(i)]=good;best=float(full[i,good].max()) if good.any() else None;result.append(dict(token=int(ts[i]),pixel=xy[i].tolist(),physical_proxy_available=bool(good.any()),best_positional_proxy_global_rank=int((full[i]>best).sum()+1) if best is not None else None,regions=[]))
    for rid in r['region_ids']:
        g=np.flatnonzero(np.linalg.norm(world-world[rid],axis=1)<=6)
        if len(g)<32:continue
        sim=full[:,g];rank=np.argsort(-sim,axis=1,kind='stable')[:,:32];ids=[]
        for candidates in g[rank]:
            _,first=np.unique(texel[candidates],return_index=True);ids.append(candidates[np.sort(first)[:4]])
        ids=np.array(ids);u=np.take_along_axis(full,ids,1);ind,iv,_,_=configuration_match(u,world[ids],maps['map_normals'][ids],qp[ts],qn[ts],xy,'independent',absolute_null=null);joint,jv,_,_=configuration_match(u,world[ids],maps['map_normals'][ids],qp[ts],qn[ts],xy,'joint',absolute_null=null,exclusive=True);mutual=sim.argmax(0)[rank[:,0]]==np.arange(len(ts))
        for out,i in zip(result,np.flatnonzero(diagnostic)):
            good=truth_masks[int(i)];out['regions'].append(dict(region=rid,positive_in_region=bool(good[g].any()),positive_in_top32=bool(good[g[rank[i]]].any()),positive_in_raw16=bool(good[g[rank[i,:16]]].any()),positive_in_distinct4=bool(good[ids[i]].any()),independent_positive=bool(iv[i] and good[ids[i,ind[i]]]),mnn_positive=bool(iv[i] and mutual[i] and good[ids[i,ind[i]]]),joint_positive=bool(jv[i] and good[ids[i,joint[i]]])))
    (o/'token_attrition.json').write_text(json.dumps(dict(query=name,diagnostic_only=True,positive_definition='<=4 coarse pixel reprojection AND query-depth-compatible atlas anchor, not independent physical identity truth',tokens=result),indent=2));print('attrition done',flush=True)
if __name__=='__main__':main()
