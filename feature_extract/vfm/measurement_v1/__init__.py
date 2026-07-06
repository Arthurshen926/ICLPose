"""Measurement-first localization primitives.

This package is intentionally opt-in.  The legacy MATCHA/RADIO pipelines keep
their existing entrypoints while measurement_v1 builds explicit 2D-3D
measurements around immutable render-surface anchors.
"""

from feature_extract.vfm.measurement_v1.types import PoseEstimate, QueryMeasurement, SurfaceAnchor

__all__ = ["PoseEstimate", "QueryMeasurement", "SurfaceAnchor"]
