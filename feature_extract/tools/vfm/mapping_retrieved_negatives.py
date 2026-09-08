"""Mapping-only RADIO negative mining from label-independent source-mode pools."""
import numpy as np
import hashlib


def mine(query_rows, query_features, query_source, query_cell, query_plane, query_world,
         pool_features, pool_world, pool_source, pool_cell, pool_plane,
         maximum_modes, distance_threshold, policy, topk=16):
    from feature_extract.tools.vfm.train_goal_maplet_mapping_canonical_subtoken_head import _diverse_mode_indices
    if policy not in ('pool_far','radio_topk') or topk<1:
        raise ValueError('invalid mapping mining policy')
    output=np.zeros(len(query_rows),np.int64);valid=np.zeros(len(query_rows),bool)
    unique, inverse=np.unique(query_rows,return_inverse=True)
    first=np.unique(query_rows,return_index=True)[1]
    selected=np.zeros(len(unique),np.int64);ok=np.zeros(len(unique),bool)
    top1_wrong=[];cosines=[];distances=[];candidate_counts=[]
    for physical in np.unique(query_plane[first]):
        cells=np.unique(pool_cell[pool_plane==physical])
        cell_pools={int(c):np.flatnonzero((pool_plane==physical)&(pool_cell==c)) for c in cells}
        cached={}; present_sources=set(pool_source[pool_plane==physical].tolist())
        for source in np.unique(query_source[first][query_plane[first]==physical]):
            cache_key=int(source) if source in present_sources else None
            if cache_key not in cached:
                candidates=[]
                for raw in cell_pools.values():
                    rows=raw[pool_source[raw]!=source]
                    if len(rows)>=2:
                        candidates.extend(rows[_diverse_mode_indices(pool_features[rows],maximum_modes)].tolist())
                cached[cache_key]=np.asarray(candidates,np.int64)
            candidates=cached[cache_key]
            if not len(candidates):continue
            group=np.flatnonzero((query_plane[first]==physical)&(query_source[first]==source))
            for index in group:
                row=first[index]
                similarity=pool_features[candidates]@query_features[row]
                # Retrieval is frozen before opening per-query geometry for labeling.
                ranking=np.argsort(-similarity,kind='stable')[:topk]
                distance=np.linalg.norm(pool_world[candidates]-query_world[row],axis=1)
                safe=(pool_cell[candidates]!=query_cell[row])&(distance>distance_threshold)
                top1_wrong.append(bool(safe[ranking[0]]));candidate_counts.append(len(candidates))
                eligible=ranking[safe[ranking]] if policy=='radio_topk' else np.flatnonzero(safe)
                if not len(eligible):continue
                chosen=int(eligible[0] if policy=='radio_topk' else eligible[np.argmax(distance[eligible])])
                selected[index]=candidates[chosen];ok[index]=True
                cosines.append(float(similarity[chosen]));distances.append(float(distance[chosen]))
    output=selected[inverse];valid=ok[inverse]
    stats={'unique_query_count':len(unique),'valid_unique_query_count':int(ok.sum()),
           'valid_pair_count':int(valid.sum()),'topk':topk,'policy':policy,
           'top1_geometrically_wrong_fraction':float(np.mean(top1_wrong)) if top1_wrong else None,
           'negative_cosine_median':float(np.median(cosines)) if cosines else None,
           'negative_distance_median_m':float(np.median(distances)) if distances else None,
           'candidate_count_median':float(np.median(candidate_counts)) if candidate_counts else None,
           'pool_filtered_by_query_cell_or_world_labels':False,'query_physical_plane_known':True,
           'query_world_geometry_used_only_for_negative_labels':True,
           'scope':'same_physical_plane_not_full_plane_retrieval'}
    stats['selected_pool_rows_and_valid_sha256']=hashlib.sha256(output.astype('<i8').tobytes()+valid.tobytes()).hexdigest()
    return output,valid,stats
