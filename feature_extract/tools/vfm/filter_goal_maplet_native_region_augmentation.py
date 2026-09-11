"""Pose-free selection of at most one added hypothesis per query token.

Every reference row is preserved exactly. Three policies retain identical
per-query token coverage and row counts, but choose different native prototypes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def select_added_rows(tokens, scores, policy, seed):
    tokens = np.asarray(tokens)
    scores = np.asarray(scores)
    if tokens.ndim != 1 or scores.shape != tokens.shape or not np.isfinite(scores).all():
        raise ValueError('invalid added token/score inventory')
    if policy not in ('cosine', 'probability', 'random'):
        raise ValueError('unknown selection policy')
    rng = np.random.default_rng(seed)
    selected = []
    for token in np.unique(tokens):
        rows = np.flatnonzero(tokens == token)
        chosen = rng.choice(rows) if policy == 'random' else rows[np.argmax(scores[rows])]
        selected.append(int(chosen))
    return np.asarray(sorted(selected), np.int64)


def filter_inventory(candidate, reference, policy):
    fixed = {'names','camera_matrices','radial_k1','correspondence_offsets'}
    for key in fixed - {'correspondence_offsets'}:
        if not np.array_equal(candidate[key],reference[key]):
            raise ValueError('reference query/camera inventory differs')
    if set(candidate) != set(reference):
        raise ValueError('reference schema differs')
    rows, offsets, audit = [], [0], []
    for i,name in enumerate(candidate['names'].astype(str)):
        lo,hi = map(int,candidate['correspondence_offsets'][i:i+2])
        a,b = map(int,reference['correspondence_offsets'][i:i+2])
        start = lo+b-a
        if hi < start:
            raise ValueError('reference prefix is missing')
        for key in set(reference)-fixed:
            if not np.array_equal(reference[key][a:b],candidate[key][lo:start]):
                raise ValueError('reference prefix differs: '+key)
        score_key = 'correspondence_match_probability' if policy=='probability' else 'radio_match_score'
        seed = 260928 + int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        added = select_added_rows(candidate['query_tokens'][start:hi],candidate[score_key][start:hi],policy,seed)+start
        selected = np.r_[np.arange(lo,start),added]
        rows.extend(selected.tolist())
        offsets.append(offsets[-1]+len(selected))
        audit.append(dict(name=name,old_rows=b-a,added_before=hi-start,added_after=len(added)))
    rows = np.asarray(rows,np.int64)
    result = {key:(value.copy() if key in fixed else value[rows]) for key,value in candidate.items()}
    result['correspondence_offsets'] = np.asarray(offsets,dtype=candidate['correspondence_offsets'].dtype)
    return result,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['candidate','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--policy',choices=['cosine','probability','random'],required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    candidate,meta=_load(a.candidate);reference,_=_load(a.reference)
    arrays,audit=filter_inventory(candidate,reference,a.policy)
    meta=dict(meta);meta.pop('content_sha256',None)
    maximum=max((int(np.unique(arrays['query_tokens'][lo:hi],return_counts=True)[1].max()) if hi>lo else 0) for lo,hi in zip(arrays['correspondence_offsets'][:-1],arrays['correspondence_offsets'][1:]))
    meta.update(arrays_sha256=arrays_sha256(arrays),correspondence_count=len(arrays['query_tokens']),
                maximum_hypotheses_per_token_observed=maximum,
                added_hypothesis_selection=dict(policy=a.policy,candidate_sha256=file_sha256(a.candidate),reference_sha256=file_sha256(a.reference),
                    rule='at most one added hypothesis per token; all original hypotheses preserved',
                    random_seed='260928 + first 8 hexadecimal SHA256 digits of query name',query_GT_used=False,
                    probability_caveat='ranking uses frozen head output; calibration on these new matches is not claimed'))
    meta['content_sha256']=canonical_json_sha256(meta)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
    _load(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(dict(policy=a.policy,query_GT_used=False,rows=audit),indent=2))

if __name__=='__main__':main()
