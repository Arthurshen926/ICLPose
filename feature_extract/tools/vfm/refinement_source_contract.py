"""Route exclusion checks for mapping pseudoqueries in direct feature refinement."""
import numpy as np


def validate_crossroute_support(names, correspondence_rows, member_rows,
                                prototype_sources, source_names, excluded_route):
    """Validate actual source identities, not just an exclusion metadata claim.

Shared physical geometry is retained. This checks native feature provenance and
region membership, and does not claim a scene-disjoint reconstruction.
"""
    query_routes = {str(n).split('__')[0] for n in names}
    if query_routes != {excluded_route}:
        raise ValueError('pseudoquery route differs from excluded map route')
    sources = np.asarray(prototype_sources)
    if sources.ndim != 1 or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError('invalid prototype source identity')
    if np.any(sources < 0) or np.any(sources >= len(source_names)):
        raise ValueError('source identity outside lineage')
    members = np.asarray(member_rows)
    rows = np.asarray(correspondence_rows)
    for ids in (members, rows):
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('invalid native row identity')
        if np.any(ids < 0) or np.any(ids >= len(sources)):
            raise ValueError('native row outside lineage')
    routes = np.asarray([str(n).split('__')[0] for n in source_names])
    if np.any(routes[sources[members]] == excluded_route):
        raise ValueError('context map includes excluded route geometry members')
    if np.any(routes[sources[rows]] == excluded_route):
        raise ValueError('refinement support includes excluded route feature source')
    # The plane front-end may access valid native anchors outside regional balls;
    # require source exclusion, not containment in this particular retrieval map.
    return dict(excluded_mapping_route=excluded_route,
                context_and_refinement_feature_source_excluded=True,
                shared_scene_geometry=True)
