"""V8.1 non-redundant region-to-surface graph localization."""

from .region_surface_graph import (
    RegionEvidenceGraph,
    build_region_evidence_graph,
    refine_region_surface_pose,
    score_region_surface_pose,
)
from .virtual_pose_lattice import (
    VirtualPoseLattice,
    expand_virtual_pose_seeds,
    rank_pose_candidates_by_region_centres,
    rank_virtual_pose_lattice,
)

__all__ = [
    "RegionEvidenceGraph",
    "build_region_evidence_graph",
    "refine_region_surface_pose",
    "score_region_surface_pose",
    "VirtualPoseLattice",
    "expand_virtual_pose_seeds",
    "rank_pose_candidates_by_region_centres",
    "rank_virtual_pose_lattice",
]
