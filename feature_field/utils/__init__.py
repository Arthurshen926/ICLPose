"""FeatureField utility package."""

from __future__ import annotations


__all__ = [
    "CameraData",
    "Feature3DGSProvider",
    "build_da3_image_order",
    "load_scene_colmap",
]


def __getattr__(name: str):
    if name == "Feature3DGSProvider":
        from feature_field.utils.feature_3dgs_provider import Feature3DGSProvider

        return Feature3DGSProvider
    if name in {"CameraData", "build_da3_image_order", "load_scene_colmap"}:
        from feature_field.utils.scene_colmap import CameraData, build_da3_image_order, load_scene_colmap

        return {
            "CameraData": CameraData,
            "build_da3_image_order": build_da3_image_order,
            "load_scene_colmap": load_scene_colmap,
        }[name]
    raise AttributeError(f"module 'feature_field.utils' has no attribute {name!r}")
