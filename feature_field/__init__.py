"""FeatureField system facade."""

from __future__ import annotations


_RUNTIME_EXPORTS = {
    "DCFFRuntime",
    "DepthGuidedRefiner",
    "FeatSharp",
    "build_dcff",
    "build_dcff_runtime",
    "intrinsics_to_K",
    "render_at_pose",
    "render_batch",
    "render_feature_bundle_at_pose",
    "render_feature_bundle_batch",
}

_DCFF_EXPORTS = {"DeferredCascadedRenderer", "SpatialHashGrid"}
_MODEL_EXPORTS = {
    "DualScaleTriplane",
    "FeatureRenderer",
    "TriPlaneDecoder",
    "TriPlaneFeatureModel",
    "TriplaneFeatureField",
}
_UTIL_EXPORTS = {"Feature3DGSProvider"}

__all__ = sorted(_RUNTIME_EXPORTS | _DCFF_EXPORTS | _MODEL_EXPORTS | _UTIL_EXPORTS)


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        from feature_field.runtime import (  # local import keeps package lightweight
            DCFFRuntime,
            DepthGuidedRefiner,
            FeatSharp,
            build_dcff,
            build_dcff_runtime,
            intrinsics_to_K,
            render_at_pose,
            render_batch,
            render_feature_bundle_at_pose,
            render_feature_bundle_batch,
        )

        return {
            "DCFFRuntime": DCFFRuntime,
            "DepthGuidedRefiner": DepthGuidedRefiner,
            "FeatSharp": FeatSharp,
            "build_dcff": build_dcff,
            "build_dcff_runtime": build_dcff_runtime,
            "intrinsics_to_K": intrinsics_to_K,
            "render_at_pose": render_at_pose,
            "render_batch": render_batch,
            "render_feature_bundle_at_pose": render_feature_bundle_at_pose,
            "render_feature_bundle_batch": render_feature_bundle_batch,
        }[name]

    if name in _DCFF_EXPORTS:
        from feature_field.dcff import DeferredCascadedRenderer, SpatialHashGrid

        return {
            "DeferredCascadedRenderer": DeferredCascadedRenderer,
            "SpatialHashGrid": SpatialHashGrid,
        }[name]

    if name in _MODEL_EXPORTS:
        from feature_field.models.feature_renderer import FeatureRenderer
        from feature_field.models.gsff_triplane import DualScaleTriplane, TriplaneFeatureField
        from feature_field.models.triplane_feature_model import TriPlaneDecoder, TriPlaneFeatureModel

        return {
            "DualScaleTriplane": DualScaleTriplane,
            "FeatureRenderer": FeatureRenderer,
            "TriPlaneDecoder": TriPlaneDecoder,
            "TriPlaneFeatureModel": TriPlaneFeatureModel,
            "TriplaneFeatureField": TriplaneFeatureField,
        }[name]

    if name in _UTIL_EXPORTS:
        from feature_field.utils.feature_3dgs_provider import Feature3DGSProvider

        return Feature3DGSProvider

    raise AttributeError(f"module 'feature_field' has no attribute {name!r}")
