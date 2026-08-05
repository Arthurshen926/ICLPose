"""Goal-Maplet: physical, auditable VFM/2DGS localization contracts."""

from .physical_map import (
    DOUBLE_SIDED,
    SINGLE_SIDED,
    GoalMapletPhysicalMap,
    SurfacePrimitiveGeometry,
    build_goal_maplet_physical_map,
    load_surface_primitive_geometry,
)

__all__ = [
    "DOUBLE_SIDED",
    "SINGLE_SIDED",
    "GoalMapletPhysicalMap",
    "SurfacePrimitiveGeometry",
    "build_goal_maplet_physical_map",
    "load_surface_primitive_geometry",
]
