"""Four-factor controls: flat/LoD retrieval x MNN/trained partial matcher."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from scipy.special import expit,softmax
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_overlap_matcher import PartialOverlapMatcher,make_pair
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def fine_coordinates(grids,pixels,ids,maps):
    xy=np.repeat(pixels[:,None],ids.shape[1],axis=1).reshape(-1,2);pr=ids.ravel();offsets=np.array([(x,y) for y in [-4/3,0.,4/3] for x in [-4/3,0.,4/3]])
    samples=sample_grid(grids[1],(xy[:,None]+offsets).reshape(-1,2)).reshape(len(xy),9,64);scores=np.einsum('nkd,nd->nk',samples,maps['fine_map'][pr]);chosen=scores.argmax(1);chosen[(scores[:,4]>=scores.max(1))|~maps['available'][pr]]=4
    return (xy+offsets[chosen]).reshape(*ids.shape,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','model','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=['mnn','learned','overlap_mnn','weighted_mnn'],default=['mnn','learned']);p.add_argument('--split',required=True);p.add_argument('--lod',action='store_true');p.add_argument('--seed',type=int,default=260901);a=p.parse_args();o=a.output;o.mkdir(exist_ok=True,parents=True);s=a.split;torch.set_num_threads(1);start=time.perf_counter();arms=a.arms
    if any((o/f'{s}_{arm}.npz').exists() for arm in arms):raise FileExistsError(o)
    maps,meta,sources=load_map(a.base);world=maps['world'];lod=LocalizationLoD(world,maps['coarse_map']);tree=cKDTree(world);checkpoint=torch.load(a.model,map_location='cpu');model=PartialOverlapMatcher();model.load_state_dict(checkpoint['state_dict']);model.eval();mm=checkpoint['metadata'];assert mm['query_test_routes_opened'] is False;assert not {'seq10','seq13'}&set(mm['training_routes']+mm['calibration_routes']);calibration=mm['overlap_calibration']
    cp=a.base/'native_reliability_v274/readout'/f'{s}_reliability_rank32.npz'
    with np.load(cp) as f:names=f['names'].astype(str);Ks=f['camera_matrices'];ks=f['radial_k1']
    projection=a.base/'stmarys_chart_local_radio_projection_64d_v2.npz'
    with np.load(projection) as f:pm=json.loads(f['metadata_json'].item())
    vc=pm['validation_learned_projection'];threshold=.5*(vc['positive_cosine_mean']+vc['same_plane_negative_cosine_mean'])
    cmd=dict(json.load(open(a.base/'diverse_candidate_retention_v307'/f'{s}_diverse_support_consensus/protocol.json'))['commands'])['alternate_render'];md=Path(cmd[cmd.index('--moge3_query')+1]);sources.update({str(p):file_sha256(p) for p in [cp,projection,a.model,Path(__file__),Path(__file__).with_name('partial_overlap_matcher.py'),Path(__file__).with_name('localization_lod_memory.py'),Path(__file__).with_name('prepare_overlap_lod_training.py'),Path(__file__).with_name('token_hypothesis_ransac.py')]});values={arm:[] for arm in arms};audit={arm:[] for arm in arms};costs=[];regions={}
    np.savez_compressed(o/f'{s}_lod_index.npz',**lod.arrays())
    for i,name in enumerate(names):
        cache=a.base/'native_fine_v264/query_cache'/name;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];qp,qn,qv=moge_tokens(md/name);sources[str(cache)]=file_sha256(cache);sources[str(md/name)]=file_sha256(md/name);q=normalise(grids[0].reshape(2304,64));tokens=diverse_tokens(q,qv);t0=time.perf_counter();chosen,cost=lod.proposals(q,flat=not a.lod);cost['retrieval_seconds']=time.perf_counter()-t0;costs.append(cost);configs={arm:[] for arm in arms};detail={arm:[] for arm in arms};inventories=[]
        for rid in chosen:
            if rid not in regions:regions[rid]=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int)
            region=regions[rid]
            if len(region)<16 or len(tokens)<6:continue
            pair=make_pair(grids,qp,qn,qv,tokens,region,**maps);ids=pair['ids'];fine_xy=fine_coordinates(grids,pair['pixels'],ids,maps);inventories.append((world[ids].reshape(-1,3),np.repeat(tokens,16),fine_xy.reshape(-1,2)))
            with torch.no_grad():
                x={k:torch.from_numpy(pair[k])[None] for k in ['query','map','edges','similarity']};ol,il=model(**x);overlap=expit(calibration[0]*ol[0].numpy()+calibration[1]);identity=softmax(il[0].numpy(),axis=-1)
            for arm in arms:
                if arm in ['mnn','overlap_mnn','weighted_mnn']:
                    keep=pair['mutual']&(pair['similarity'][:,0,0]>=threshold)
                    if arm=='overlap_mnn':keep &= overlap>=.5
                    take=np.flatnonzero(keep);selected=np.zeros(len(take),int);weights=overlap[take] if arm=='weighted_mnn' else np.ones(len(take))
                else:
                    rank=np.argsort(-identity,axis=1,kind='stable')[:,:2];good=(overlap[:,None]>=.5)&(np.take_along_axis(identity,rank,1)>=.5*identity.max(1)[:,None]);take,column=np.where(good);selected=rank[take,column];weights=overlap[take]*identity[take,selected]
                pr=ids[take,selected];pixels=fine_xy[take,selected];stats={};pose=solve(world[pr],tokens[take],Ks[i],float(ks[i]),np.arange(len(pr)),pixels=pixels,iterations=1250,seed=a.seed+rid,hypothesis_budget=250,stats=stats,sampling_policy='overlap_prior' if arm=='weighted_mnn' else 'context_prior',scores=np.maximum(weights,1e-8))
                configs[arm].append(pose);detail[arm].append(dict(region=rid,selected_tokens=tokens[take].tolist(),prototype_rows=pr.tolist(),query_pixels=pixels.tolist(),match_weights=weights.tolist(),overlap_probabilities=overlap.tolist() if arm!='mnn' else None,pnp=stats))
        if inventories:
            cw,ct,cxy=canonical_hypotheses(*[np.concatenate([v[j] for v in inventories]) for j in range(3)]);groups=[np.flatnonzero(ct==t) for t in np.unique(ct)]
        for arm in arms:
            finite=[p for p in configs[arm] if p is not None];pose=max(finite,key=lambda p:score_pose(p,cw,cxy,groups,Ks[i],float(ks[i]),return_selected=False)[0]) if finite else np.full((4,4),np.nan);values[arm].append(pose);audit[arm].append(dict(name=name,accepted=bool(np.isfinite(pose).all()),regions=detail[arm],region_ids=chosen,candidate_poses=[p.tolist() if p is not None else None for p in configs[arm]],sampled_tokens=tokens.tolist(),valid_query_tokens=len(tokens)))
        if (i+1)%10==0:print(s,'lod' if a.lod else 'flat',i+1,flush=True)
    for arm in arms:
        arrays=dict(names=names,pose_w2c=np.array(values[arm]),usable=np.isfinite(values[arm]).all((1,2)));m=dict(artifact_type='overlap_lod_frontend_v1',arrays_sha256=arrays_sha256(arrays),source_sha256=sources,query_pose_or_ground_truth_read=False,arm=arm,lod=a.lod,seed=a.seed,geometry_anchors_unchanged=True,source_rgb_stored_or_consumed_at_runtime=False,subtoken_head_used=False,fine_grid_search_used=True,fine_identity_input_used=arm=='learned',map_members_learned=False,overlap_and_identity_trained=arm=='learned',overlap_prediction_used=arm!='mnn',identity_prediction_used=arm=='learned',max_regions=4,max_query_tokens=256,candidate_modes=16,maximum_output_modes_per_query_token=2 if arm=='learned' else 1,pnp_hypothesis_cap=1000,score_population='all 16 candidate identities with fixed fine-grid measurements; unique-token reprojection scoring',scope='four-factor cached-feature experiment; no out-of-core LoD streaming')
        m['content_sha256']=canonical_json_sha256(m);np.savez_compressed(o/f'{s}_{arm}.npz',**arrays,metadata_json=np.array(json.dumps(m,sort_keys=True)));(o/f'{s}_{arm}_audit.json').write_text(json.dumps(audit[arm]))
    (o/f'{s}_costs.json').write_text(json.dumps(costs));(o/f'{s}_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,controls=len(arms),coarse_representatives=len(lod.coarse_rows),fine_representatives=len(lod.fine_rows),index_bytes=sum(v.nbytes for v in lod.arrays().values()),scope='two matchers cached; includes shared data loading, excludes RGB encoders and reconstruction')))
if __name__=='__main__':main()
