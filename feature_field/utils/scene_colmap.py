"""FeatureField wrappers for COLMAP scene-loading helpers.

These helpers keep the active DCFF training path under the FeatureField
namespace while the remaining low-level 2DGS training script is still being
migrated out of the legacy feature_3dgs package.
"""

from feature_gaussian.legacy_3dgs.train_2dgs_joint_v2 import CameraData, build_da3_image_order, load_scene_colmap


__all__ = ["CameraData", "build_da3_image_order", "load_scene_colmap"]