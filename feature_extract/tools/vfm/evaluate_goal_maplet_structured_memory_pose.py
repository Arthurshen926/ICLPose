"""Fixed-budget pose test of image-crossfit association scores on full real inputs.

No map/query geometry changes; 50% uniform exploration; score/LM count each
query token once. Mapping poses are opened for evaluation only after freezing.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.token_hypothesis_ransac import solve,score_pose
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _top_groups
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics
from feature_extract.tools.vfm.audit_goal_maplet_pose_failure_stages import _pose_error
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def evaluation_images(candidate_sources, source_names, requested_names):
    """Zero-correspondence queries remain explicit localization failures."""
    if requested_names is None:return np.unique(candidate_sources)
    lookup={str(n):i for i,n in enumerate(source_names)}
    if any(n not in lookup for n in requested_names):raise ValueError('query inventory missing from source names')
    images=np.array(sorted({lookup[n] for n in requested_names}),np.int64)
    if not set(candidate_sources).issubset(set(images)):raise ValueError('candidates outside requested query inventory')
    return images


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','scores','features','contributors','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=260901)
    p.add_argument('--regional_seed_policy',choices=['group_order','shared'],default='group_order',
                   help='Shared regional seed supports exact frozen-set training; opt-in matched controls required.')
    p.add_argument('--arms',nargs='+',default=['uniform','cosine','base','combined','extended_base','extended_combined'])
    p.add_argument('--pixel_readout',type=Path,help='Optional matched fine readout; keeps original candidate geometry.')
    p.add_argument('--pixel_mode',type=int,choices=[0,1],default=1,help='0 coarse interpolation control; 1 real high-resolution RGB.')
    p.add_argument('--regional_radius',type=float,default=0.,help='Opt-in fixed metric region groups instead of four plane groups.')
    p.add_argument('--region_selection',choices=['support','coverage'],default='coverage')
    p.add_argument('--region_model',type=Path,help='Mapping-only learned regional initializer utility.')
    p.add_argument('--region_map',type=Path,help='Mapping-learned physical centers/radii/member inventory.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.candidates) as z:c={k:z[k] for k in z.files}
    with np.load(a.scores) as z:
        scores=z['logits'];arm_names=z['arm_names'].tolist()
        if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('score lineage differs')
    with np.load(a.features) as z:names=z['source_names'].astype(str)
    region_centers=None;region_radii=a.regional_radius;region_training_names=set();region_allowed_members=None
    region_model=json.loads(a.region_model.read_text()) if a.region_model else None
    if a.region_map:
        import hashlib,shutil
        from scipy.spatial import cKDTree
        from feature_extract.vfm.localization_goal_maplet.metric_region_memory import activate_regions
        if a.region_model:raise ValueError('separate boundary and online utility experiments')
        with np.load(a.region_map) as z:
            region_centers=z['centers'];region_radii=z['radii'];region_training_names=set(z['training_images'].astype(str))
            if str(z['world_sha256'])!=hashlib.sha256(c['prototype_world'].tobytes()).hexdigest():raise ValueError('boundary map world differs')
            offsets=z['offsets'];members=z['prototype_rows']
            member_policy=str(z['member_policy']) if 'member_policy' in z.files else 'sphere'
            if member_policy not in ['sphere','explicit_subset']:raise ValueError('unknown member policy')
        if region_centers.shape!=(len(region_radii),3) or not np.isfinite(region_centers).all() or not np.isfinite(region_radii).all() or np.any(region_radii<=0):raise ValueError('invalid learned boundaries')
        expected=cKDTree(c['prototype_world']).query_ball_point(region_centers,region_radii)
        if (len(offsets)!=len(expected)+1 or offsets[0]!=0 or offsets[-1]!=len(members)
                or np.any(np.diff(offsets)<0) or not np.issubdtype(offsets.dtype,np.integer)
                or not np.issubdtype(members.dtype,np.integer)):raise ValueError('member inventory differs')
        for i,rows in enumerate(expected):
            saved=members[offsets[i]:offsets[i+1]]
            if member_policy=='sphere':
                if not np.array_equal(np.sort(rows),np.sort(saved)):raise ValueError('learned members do not match boundaries')
            elif len(np.unique(saved))!=len(saved) or not np.isin(saved,rows).all():raise ValueError('explicit members must be a unique subset of their metric envelope')
        if member_policy=='explicit_subset':region_allowed_members=[members[offsets[i]:offsets[i+1]] for i in range(len(expected))]
        shutil.copyfile(a.region_map,a.output/'regions.npz')
    elif a.regional_radius:
        from feature_extract.vfm.localization_goal_maplet.metric_region_memory import build_regions,activate_regions
        region_centers,region_members=build_regions(c['prototype_world'],a.regional_radius)
        np.savez_compressed(a.output/'regions.npz',centers=region_centers,offsets=np.r_[0,np.cumsum([len(r) for r in region_members])],
            prototype_rows=np.concatenate(region_members),radius=np.asarray(a.regional_radius))
    refined_pixels=None;refinement_mask=None
    if a.pixel_readout is not None:
        with np.load(a.pixel_readout) as z:
            if str(z['candidate_sha256'])!=file_sha256(a.candidates):raise ValueError('pixel readout lineage differs')
            refined_pixels=z['refined_pixels'][:,a.pixel_mode]
            refinement_mask=z['refinement_mask'].astype(bool)
        if refined_pixels.shape!=(len(c['query_token']),2) or refinement_mask.shape!=(len(refined_pixels),):raise ValueError('pixel inventory differs')
        if not np.isfinite(refined_pixels).all():raise ValueError('nonfinite fine pixels')
    candidate_report=json.loads(a.candidates.with_suffix('.json').read_text())
    images=evaluation_images(c['source_image'],names,candidate_report.get('query_image_names'));records=[];all_errors={};source_hashes={}
    if region_training_names & set(names[images]):raise ValueError('boundary training/test overlap')
    if region_model is not None:
        import hashlib
        if set(names[images])&set(region_model['training_images']):raise ValueError('regional utility evaluation overlaps training')
        if region_model['radius']!=a.regional_radius or region_model['map_world_sha256']!=hashlib.sha256(c['prototype_world'].tobytes()).hexdigest():raise ValueError('regional utility map differs')
    for arm in a.arms:
        selected=[];pools=[];offsets=[0];stats=[]
        for count,s in enumerate(images):
            rows=np.flatnonzero((c['source_image']==s)&c['homography_keep'])
            tokens=c['query_token'][rows];proto=c['prototype_rows'][rows]
            world=c['prototype_world'][proto];planes=c['prototype_plane'][proto]
            priority=c['association_features'][rows,0] if arm in ['uniform','cosine'] else scores[rows,arm_names.index(arm)]
            path=a.contributors/names[s]
            # Only camera intrinsics; no pose or depth array is accessed here.
            with np.load(path) as z:K,k1=_scaled_intrinsics(int(z['camera_model_id']),z['camera_params'],int(z['camera_width']),int(z['camera_height']))
            source_hashes[str(path)]=file_sha256(path)
            seed_groups=[np.arange(len(rows))]+_top_groups(planes,4,tokens)
            if region_centers is not None:
                regional,region_ids=activate_regions(world,tokens,priority,region_centers,region_radii,4,a.region_selection=='coverage',region_model,
                    prototype_ids=proto if region_allowed_members is not None else None,allowed_members=region_allowed_members)
                seed_groups=[np.arange(len(rows))]+regional
                # Fill absent groups with existing plane controls, retaining the
                # same five-group budget even under incomplete region visibility.
                if len(seed_groups)<5:seed_groups+=_top_groups(planes,5-len(seed_groups),tokens)
            groups=[np.flatnonzero(tokens==t) for t in np.unique(tokens)]
            pixels=np.c_[(tokens%64)*4+1.5,(tokens//64)*4+1.5]
            if refined_pixels is not None:
                pixels=np.where(refinement_mask[rows,None],refined_pixels[rows],pixels)
            best=None;best_key=(-1,-np.inf);local=[]
            for gi,group in enumerate(seed_groups):
                st={'source':int(s),'group':gi,'arm':arm}
                group_seed=a.seed+(1 if gi>0 and a.regional_seed_policy=='shared' else gi)
                st['seed']=group_seed
                pose=solve(world,tokens,K,k1,group,pixels=pixels,iterations=2048,seed=group_seed,
                           sampling_policy='uniform' if arm=='uniform' else 'geometry_score',planes=planes,scores=priority,hypothesis_budget=256,stats=st)
                stats.append(st)
                if pose is None:continue
                key,_=score_pose(pose,world,pixels,groups,K,k1)
                local.append(pose)
                if key>best_key:best=pose;best_key=key
            pools.extend(local);offsets.append(len(pools));selected.append(np.full((4,4),np.nan) if best is None else best)
            if (count+1)%10==0:print(arm,count+1,'/',len(images),flush=True)
        path=a.output/(arm+'.npz')
        np.savez_compressed(path,source_image=images,names=names[images],pose_w2c=np.array(selected),candidate_offsets=np.array(offsets),candidate_pose_w2c=np.asarray(pools).reshape(-1,4,4))
        # Only now open held mapping poses to measure frozen outputs.
        errors=[];oracle=[]
        for i,s in enumerate(images):
            with np.load(a.contributors/names[s]) as z:gt=z['pose_w2c']
            errors.append(_pose_error(selected[i],gt));lo,hi=offsets[i:i+2]
            oracle.append([_pose_error(t,gt) for t in pools[lo:hi]])
        errors=np.array(errors);thresholds=[(.1,1),(.25,2),(.5,5),(1,10),(2,45)]
        hits=[int(((errors[:,0]<=t)&(errors[:,1]<=r)).sum()) for t,r in thresholds]
        pool_hits=[sum(any(e[0]<=t and e[1]<=r for e in group) for group in oracle) for t,r in thresholds]
        all_errors[arm]=errors.tolist()
        report={'arm':arm,'images':len(images),'hits':hits,'pool_oracle_hits':pool_hits,'translation_median':float(np.median(errors[:,0])),
                'scored_hypotheses':sum(s.get('scored_hypotheses',0) for s in stats),'budget_unreached_groups':sum(not s.get('budget_reached',False) for s in stats),
                'solver_seconds':sum(s.get('seconds',0) for s in stats),'pose_sha256':file_sha256(path),'stats':stats}
        records.append(report)
        (a.output/(arm+'.json')).write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='stats'}),flush=True)
    report={'scope':__doc__,'seed':a.seed,'regional_seed_policy':a.regional_seed_policy,'reports':[{k:v for k,v in r.items() if k!='stats'} for r in records],
            'errors':all_errors,'candidate_sha256':file_sha256(a.candidates),'score_sha256':file_sha256(a.scores),'camera_source_sha256':source_hashes,
            'pose_labels_used_for_selection':False,'candidate_rows_removed':False,'mapping_route_evaluation_not_official_test':True,'evaluation_routes':sorted(set(n.split('__')[0] for n in names[images])),'not_full_MoGe_refined_consensus':True,
            'pixel_readout_sha256':file_sha256(a.pixel_readout) if a.pixel_readout else None,'pixel_mode':a.pixel_mode if a.pixel_readout else None,
            'region_map_sha256':file_sha256(a.region_map) if a.region_map else None,'regional_radius':a.regional_radius if a.region_map is None else None,'region_selection':a.region_selection if region_centers is not None else None,
            'region_model_sha256':file_sha256(a.region_model) if a.region_model else None}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
