"""PoseRefine system facade."""

from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.utils.geometry_solver import (
    compute_image_jacobian,
    diff_pose_solve,
    feature_metric_solve,
    pnp_ransac_solve,
)
from pose_refine.runtime import (
    apply_pose_delta,
    build_concat_pose_model,
    load_concat_pose_checkpoint,
    load_concat_pose_model,
    load_local_matcher_weights,
    run_model_refine_iteration,
)

try:
    from pose_refine.sparse_init import LoFTRInitializer, LoFTRResult
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise

    class LoFTRInitializer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "LoFTRInitializer requires OpenCV (cv2). Install opencv-python "
                "or opencv-python-headless to use sparse initialization."
            ) from exc

    LoFTRResult = None  # type: ignore[assignment]


__all__ = [
    "ConcatPoseNet",
    "LoFTRInitializer",
    "LoFTRResult",
    "apply_pose_delta",
    "build_concat_pose_model",
    "compute_image_jacobian",
    "diff_pose_solve",
    "feature_metric_solve",
    "load_concat_pose_checkpoint",
    "load_concat_pose_model",
    "load_local_matcher_weights",
    "pnp_ransac_solve",
    "run_model_refine_iteration",
]
