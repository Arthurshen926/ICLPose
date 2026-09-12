"""Freeze symmetric old/union-evidence verification controls without labels.

Both candidates see the same deterministic token budget and every alternative
for a selected token. Votes are unique image tokens, not correspondence rows.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import _load_selected
from feature_extract.tools.vfm.token_hypothesis_ransac import canonical_hypotheses,score_pose
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256,file_sha256


def select_supported(counts, usable, baseline, eligible):
    counts=np.asarray(counts);usable=np.asarray(usable,bool);baseline=np.asarray(baseline);eligible=np.asarray(eligible,bool)
    if counts.shape != usable.shape or counts.shape != (len(baseline),2) or eligible.shape!=baseline.shape:
        raise ValueError('shared selection shapes differ')
    out=baseline.copy()
    for i in range(len(out)):
        if not eligible[i]:continue
        valid=np.flatnonzero(usable[i])
        if len(valid)==1:out[i]=valid[0]
        elif len(valid)==2 and counts[i,0]!=counts[i,1]:out[i]=np.argmax(counts[i])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses',nargs=2,type=Path,required=True)
    p.add_argument('--old_correspondences',nargs=2,type=Path,required=True)
    p.add_argument('--augmented_correspondences',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--token_budget',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.token_budget<1:raise ValueError('positive token budget required')
    a.output.mkdir(parents=True,exist_ok=False)
    poses=[_load_pose_candidate(x) for x in a.poses];old=[_load(x)[0] for x in a.old_correspondences];aug,_=_load(a.augmented_correspondences);base,bm=_load_selected(a.baseline);names=base['names']
    for item in [x[0] for x in poses]+old+[aug]:
        if not np.array_equal(item['names'],names):raise ValueError('query identity differs')
    for key in ['primary_pose_file_sha256','alternate_pose_file_sha256']:
        if bm[key]!=file_sha256(a.poses[0 if key.startswith('primary') else 1]):raise ValueError('baseline candidate binding differs')
    for c in old:
        for k in ['camera_matrices','radial_k1']:
            if not np.array_equal(c[k],aug[k]):raise ValueError('camera inventory differs')
    for i in range(len(names)):
        l,h=old[0]['correspondence_offsets'][i:i+2];start=aug['correspondence_offsets'][i]
        for key in ['query_tokens','world_points','query_measurements_xy','prototype_atlas_row']:
            if not np.array_equal(old[0][key][l:h],aug[key][start:start+h-l]):raise ValueError('old evidence prefix differs')
    counts={k:np.zeros((len(names),2),np.int64) for k in ['old','union']};budget=np.zeros(len(names),int);observed={k:np.zeros(len(names),int) for k in counts}
    usable=np.stack([v[0]['usable'] for v in poses],axis=1)
    for i,name in enumerate(names.astype(str)):
        def rows(c):
            l,h=c['correspondence_offsets'][i:i+2]
            return c['world_points'][l:h],c['query_tokens'][l:h],c['query_measurements_xy'][l:h]
        full=[rows(aug),rows(old[1])];alltokens=np.unique(np.concatenate([x[1] for x in full]));rng=np.random.default_rng(260912+int(hashlib.sha256(name.encode()).hexdigest()[:8],16));selected=np.sort(rng.choice(alltokens,min(len(alltokens),a.token_budget),replace=False));budget[i]=len(selected)
        for label,items in [('old',[rows(c) for c in old]),('union',full)]:
            w,t,pix=map(np.concatenate,zip(*items));keep=np.isin(t,selected);w,t,pix=canonical_hypotheses(w[keep],t[keep],pix[keep]);observed[label][i]=len(np.unique(t))
            if not len(t):continue
            groups=[np.flatnonzero(t==token) for token in np.unique(t)]
            for j,(pose,_) in enumerate(poses):
                if usable[i,j]:counts[label][i,j]=score_pose(pose['pose_w2c'][i],w,pix,groups,aug['camera_matrices'][i],float(aug['radial_k1'][i]),return_selected=False)[0][0]
    missing=np.isposinf(base['candidate_plane_geometry_objective']).all(1)
    for label in counts:
        for scope in ['all','missing']:
            eligible=np.ones(len(names),bool) if scope=='all' else missing
            choice=select_supported(counts[label],usable,base['selected_branch'],eligible);idx=np.arange(len(names))
            arrays=dict(names=names,pose_w2c=np.stack([v[0]['pose_w2c'] for v in poses],axis=1)[idx,choice],usable=usable[idx,choice],selected_branch=choice,candidate_unique_token_support=counts[label],verification_token_budget=budget,evidence_token_count=observed[label],eligible=eligible)
            meta=dict(artifact_type='goal_maplet_shared_evidence_selection_v1',query_pose_or_ground_truth_read=False,arrays_sha256=arrays_sha256(arrays),evidence=label,scope=scope,selection='strictly greater positive-depth unique-token support at existing 4px threshold; ties retain baseline',token_budget=a.token_budget,source_sha256={str(x):file_sha256(x) for x in a.poses+a.old_correspondences+[a.augmented_correspondences,a.baseline]},not_independent_holdout=True)
            meta['content_sha256']=canonical_json_sha256(meta)
            np.savez_compressed(a.output/f'{label}_{scope}.npz',**arrays,metadata_json=np.array(json.dumps(meta,sort_keys=True)))


if __name__=='__main__':main()
