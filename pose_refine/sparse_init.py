"""LoFTR-based sparse pose initializer.

Bridges the gap between coarse retrieval (~7-11° rotation error) and the
refinement network's convergence range (<1°) by establishing 2D-2D
correspondences via LoFTR, unprojecting reference keypoints to 3D using
rendered depth, and solving PnP-RANSAC.

Two operating modes:
  * **Render-based** (`estimate_pose`): reference RGB + depth are both
    rendered from the Gaussian splat at the reference pose.
  * **Direct-image-based** (`estimate_pose_direct`): reference RGB is
    loaded from disk (actual training image); depth is still rendered.
    This avoids rendering artefacts in the matching image but requires
    that the depth render uses the exact same camera model.

Usage example::

    init = LoFTRInitializer(device="cuda")
    pose_w2c, info = init.estimate_pose(
        query_rgb, ref_rgb, ref_depth,
        ref_pose_w2c, intrinsics,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Result container
# ---------------------------------------------------------------------------


@dataclass
class LoFTRResult:
    """Result of a LoFTR initialisation attempt."""

    pose_w2c: Optional[np.ndarray] = None
    """Estimated query world-to-camera pose (4×4), or *None* on failure."""
    success: bool = False
    num_raw_matches: int = 0
    num_confident_matches: int = 0
    num_depth_valid: int = 0
    num_inliers: int = 0
    mean_confidence: float = 0.0
    failure_reason: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_gray_tensor(
    img: Union[np.ndarray, torch.Tensor],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Convert an image to grayscale float32 tensor (1, 1, H, W) in [0, 1].

    Accepts:
      * numpy (H, W, 3) uint8/float
      * numpy (H, W) grayscale
      * torch  (1, 3, H, W) or (1, 1, H, W)
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # assume RGB input
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"Unexpected numpy shape {img.shape}")
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32) / 255.0
        t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif isinstance(img, torch.Tensor):
        if img.dim() == 4 and img.shape[1] == 3:
            t = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        elif img.dim() == 4 and img.shape[1] == 1:
            t = img.float()
        else:
            raise ValueError(f"Unexpected tensor shape {img.shape}")
    else:
        raise TypeError(type(img))

    t = F.interpolate(t.to(device).float(), size=target_hw, mode="bilinear", align_corners=False)
    return t


def _scale_intrinsics(
    intrinsics: Dict[str, float],
    orig_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> Dict[str, float]:
    """Rescale intrinsics {fx, fy, cx, cy} from *orig_hw* to *target_hw*."""
    sy = target_hw[0] / orig_hw[0]
    sx = target_hw[1] / orig_hw[1]
    return {
        "fx": intrinsics["fx"] * sx,
        "fy": intrinsics["fy"] * sy,
        "cx": intrinsics["cx"] * sx,
        "cy": intrinsics["cy"] * sy,
    }


def _unproject_points(
    kpts: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: Dict[str, float],
    kpts_hw: Tuple[int, int],
    depth_hw: Tuple[int, int],
    min_depth: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject 2D keypoints to 3D in the camera frame.

    *kpts* are in the coordinate system of *kpts_hw*.
    *depth_map* has resolution *depth_hw*.
    *intrinsics* are at the *kpts_hw* resolution.

    Returns:
        pts_3d:  (M, 3) valid 3D points in camera frame
        kpts_valid: (M, 2) corresponding 2D keypoints (for indexing)
        valid_mask: (N,) bool mask over original kpts
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # Map keypoint coordinates to depth map pixel coords
    sx = depth_hw[1] / kpts_hw[1]
    sy = depth_hw[0] / kpts_hw[0]
    u_depth = (kpts[:, 0] * sx).astype(np.float64)
    v_depth = (kpts[:, 1] * sy).astype(np.float64)

    # Bilinear sample from depth map
    u0 = np.floor(u_depth).astype(int)
    v0 = np.floor(v_depth).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1
    dH, dW = depth_hw
    u0c = np.clip(u0, 0, dW - 1)
    v0c = np.clip(v0, 0, dH - 1)
    u1c = np.clip(u1, 0, dW - 1)
    v1c = np.clip(v1, 0, dH - 1)
    wu = u_depth - u0
    wv = v_depth - v0

    d00 = depth_map[v0c, u0c]
    d01 = depth_map[v0c, u1c]
    d10 = depth_map[v1c, u0c]
    d11 = depth_map[v1c, u1c]
    depth_vals = (
        (1 - wu) * (1 - wv) * d00
        + wu * (1 - wv) * d01
        + (1 - wu) * wv * d10
        + wu * wv * d11
    )

    valid = (depth_vals > min_depth) & np.isfinite(depth_vals)

    u_kpt = kpts[:, 0].astype(np.float64)
    v_kpt = kpts[:, 1].astype(np.float64)
    Z = depth_vals
    X = (u_kpt - cx) * Z / fx
    Y = (v_kpt - cy) * Z / fy
    pts_3d = np.stack([X, Y, Z], axis=-1)  # (N, 3)

    return pts_3d[valid], kpts[valid], valid


def _solve_pnp(
    pts_3d_world: np.ndarray,
    pts_2d: np.ndarray,
    intrinsics: Dict[str, float],
    reproj_threshold: float = 8.0,
    n_iters: int = 10_000,
    min_inliers: int = 4,
    use_magsac: bool = False,
    return_inliers: bool = False,
) -> Tuple[Optional[np.ndarray], int]:
    """Run PnP-RANSAC and return the query w2c pose.

    Args:
        pts_3d_world: (M, 3) 3D points in world frame.
        pts_2d:       (M, 2) corresponding query 2D keypoints.
        intrinsics:   {fx, fy, cx, cy} at *pts_2d*'s resolution.
        reproj_threshold: RANSAC inlier reprojection error (px).
        n_iters:      RANSAC iterations.
        min_inliers:  minimum inliers to accept the solution.
        use_magsac:   Use USAC_MAGSAC for more robust estimation.

    Returns:
        (pose_w2c (4×4) or None, num_inliers)
    """
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pts_3d = pts_3d_world.astype(np.float64)
    pts_2d = pts_2d.astype(np.float64)

    if use_magsac and hasattr(cv2, "USAC_MAGSAC"):
        try:
            params = cv2.UsacParams()
            params.confidence = 0.9999
            params.maxIterations = n_iters
            params.threshold = reproj_threshold
            params.loMethod = cv2.LOCAL_OPTIM_INNER_AND_ITER_LO
            params.loIterations = 10

            ret = cv2.solvePnPRansac(
                pts_3d, pts_2d, camera_matrix, None,
                params=params,
            )
            # OpenCV's USAC overload returns:
            #   (ok, cameraMatrix, rvec, tvec, inliers)
            if len(ret) == 5:
                success, _camera_matrix_out, rvec, tvec, inliers = ret
            else:
                success, rvec, tvec, inliers = ret
        except (cv2.error, TypeError):
            use_magsac = False

    if not use_magsac:
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                camera_matrix,
                None,
                iterationsCount=n_iters,
                reprojectionError=reproj_threshold,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            if return_inliers:
                return None, 0, np.zeros((len(pts_2d),), dtype=bool)
            return None, 0

    if not success or inliers is None or len(inliers) < min_inliers:
        count = 0 if inliers is None else len(inliers)
        if return_inliers:
            mask = np.zeros((len(pts_2d),), dtype=bool)
            if inliers is not None:
                mask[inliers.flatten()] = True
            return None, count, mask
        return None, count

    num_inliers = len(inliers)

    # Refine with inlier-only iterative solve
    if num_inliers >= 6:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                pts_3d[inliers.flatten()],
                pts_2d[inliers.flatten()],
                camera_matrix,
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                pass  # keep RANSAC solution
        except cv2.error:
            pass

    R_pnp, _ = cv2.Rodrigues(rvec)
    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = R_pnp
    pose_w2c[:3, 3] = tvec.flatten()
    if return_inliers:
        mask = np.zeros((len(pts_2d),), dtype=bool)
        mask[inliers.flatten()] = True
        return pose_w2c, num_inliers, mask
    return pose_w2c, num_inliers


def _compute_loftr_resolution(
    orig_hw: Tuple[int, int],
    long_edge: int,
) -> Tuple[int, int]:
    """Compute aspect-preserving LoFTR resolution.

    LoFTR internally pads to multiples of 8, so we round to multiples of 8
    for cleanliness.

    Args:
        orig_hw: (H, W) of original images.
        long_edge: target size for the longer edge.

    Returns:
        (H_loftr, W_loftr) rounded to multiples of 8.
    """
    H, W = orig_hw
    scale = long_edge / max(H, W)
    nH = int(round(H * scale / 8)) * 8
    nW = int(round(W * scale / 8)) * 8
    return max(nH, 8), max(nW, 8)


# ---------------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------------


class LoFTRInitializer:
    """LoFTR-based sparse pose initialiser.

    Matches query vs reference images to establish 2D–2D correspondences,
    then uses rendered depth to lift to 3D and solves PnP for the query pose.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        pretrained: str = "outdoor",
        loftr_long_edge: int = 840,
        confidence_threshold: float = 0.3,
        min_matches: int = 6,
        reproj_threshold: float = 8.0,
        pnp_iters: int = 10_000,
        use_magsac: bool = False,
    ):
        from kornia.feature import LoFTR as _LoFTR

        self.device = torch.device(device)
        self.loftr = _LoFTR(pretrained=pretrained).eval().to(self.device)
        self.loftr_long_edge = loftr_long_edge
        self.confidence_threshold = confidence_threshold
        self.min_matches = min_matches
        self.reproj_threshold = reproj_threshold
        self.pnp_iters = pnp_iters
        self.use_magsac = use_magsac

    # ------------------------------------------------------------------
    #  Core: single pair
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_pose(
        self,
        query_rgb: Union[np.ndarray, torch.Tensor],
        ref_rgb: Union[np.ndarray, torch.Tensor],
        ref_depth: Union[np.ndarray, torch.Tensor],
        ref_pose_w2c: np.ndarray,
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        *,
        confidence_threshold: Optional[float] = None,
        min_matches: Optional[int] = None,
        reproj_threshold: Optional[float] = None,
    ) -> LoFTRResult:
        """Estimate query w2c pose from a single query/reference pair.

        Both *query_rgb* and *ref_rgb* may be rendered or loaded from disk;
        they just need to be colour images.  *ref_depth* is the depth map at
        the reference viewpoint (camera-space Z in metres).

        All *intrinsics* are given at *orig_hw* resolution and are internally
        rescaled for LoFTR and depth unprojection.

        Args:
            query_rgb:    Query image — (H, W, 3) uint8/float or (1, 3, H, W).
            ref_rgb:      Reference image — same formats.
            ref_depth:    Reference depth map — (H_d, W_d) numpy or (1, 1, H, W) torch.
            ref_pose_w2c: (4, 4) w2c pose of the reference view.
            intrinsics:   {fx, fy, cx, cy} at *orig_hw* resolution.
            orig_hw:      (H, W) of the full-resolution images.
            confidence_threshold: override default.
            min_matches:  override default.
            reproj_threshold: override default.

        Returns:
            :class:`LoFTRResult` with the estimated pose (or failure info).
        """
        conf_thr = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        min_m = min_matches if min_matches is not None else self.min_matches
        reproj_thr = reproj_threshold if reproj_threshold is not None else self.reproj_threshold

        result = LoFTRResult()

        # --- Resolve LoFTR resolution (aspect-preserving) -----------------
        loftr_hw = _compute_loftr_resolution(orig_hw, self.loftr_long_edge)

        # --- Prepare grayscale tensors ------------------------------------
        gray_q = _to_gray_tensor(query_rgb, loftr_hw, self.device)
        gray_r = _to_gray_tensor(ref_rgb, loftr_hw, self.device)

        # --- LoFTR matching -----------------------------------------------
        match_input = {"image0": gray_q, "image1": gray_r}
        match_out = self.loftr(match_input)

        kpts0 = match_out["keypoints0"].cpu().numpy()  # (N, 2) in loftr coords
        kpts1 = match_out["keypoints1"].cpu().numpy()
        conf = match_out["confidence"].cpu().numpy()

        result.num_raw_matches = len(conf)
        if result.num_raw_matches == 0:
            result.failure_reason = "no_matches"
            return result

        # --- Filter by confidence -----------------------------------------
        keep = conf >= conf_thr
        kpts0, kpts1, conf = kpts0[keep], kpts1[keep], conf[keep]
        result.num_confident_matches = len(conf)
        result.mean_confidence = float(conf.mean()) if len(conf) > 0 else 0.0

        if result.num_confident_matches < min_m:
            result.failure_reason = f"too_few_confident_matches ({result.num_confident_matches} < {min_m})"
            return result

        # --- Resolve depth ------------------------------------------------
        if isinstance(ref_depth, torch.Tensor):
            d = ref_depth.detach().cpu().numpy()
            if d.ndim == 4:
                d = d[0, 0]
            elif d.ndim == 3:
                d = d[0]
        else:
            d = ref_depth
        depth_hw = d.shape[:2]

        # --- Intrinsics at LoFTR resolution (for unprojection of ref kpts)
        intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)

        # --- Unproject ref keypoints to 3D (camera frame) -----------------
        pts_3d_cam, kpts1_valid, valid_mask = _unproject_points(
            kpts1, d, intr_loftr, kpts_hw=loftr_hw, depth_hw=depth_hw,
        )
        kpts0_valid = kpts0[valid_mask]
        result.num_depth_valid = len(pts_3d_cam)

        if result.num_depth_valid < min_m:
            result.failure_reason = f"too_few_depth_valid ({result.num_depth_valid} < {min_m})"
            return result

        # --- Transform 3D points: ref camera → world ----------------------
        ref_w2c = np.asarray(ref_pose_w2c, dtype=np.float64)
        ref_c2w = np.linalg.inv(ref_w2c)
        pts_3d_world = (ref_c2w[:3, :3] @ pts_3d_cam.T + ref_c2w[:3, 3:4]).T

        # --- PnP RANSAC (world-3D vs query-2D at LoFTR resolution) --------
        intr_query_loftr = intr_loftr  # same camera → same intrinsics
        pose_w2c, num_inliers, inlier_mask = _solve_pnp(
            pts_3d_world,
            kpts0_valid,
            intr_query_loftr,
            reproj_threshold=reproj_thr,
            n_iters=self.pnp_iters,
            use_magsac=self.use_magsac,
            return_inliers=True,
        )
        result.num_inliers = num_inliers
        result.extra.update(
            {
                "query_keypoints": kpts0_valid.astype(np.float32),
                "ref_keypoints": kpts1_valid.astype(np.float32),
                "pts3d_world": pts_3d_world.astype(np.float32),
                "confidence": conf[valid_mask].astype(np.float32),
                "pnp_inlier_mask": inlier_mask.astype(bool),
                "loftr_hw": np.asarray(loftr_hw, dtype=np.int32),
            }
        )

        if pose_w2c is None:
            result.failure_reason = f"pnp_failed (inliers={num_inliers})"
            return result

        result.pose_w2c = pose_w2c.astype(np.float32)
        result.success = True
        return result

    # ------------------------------------------------------------------
    #  Convenience: load images from disk
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_pose_direct(
        self,
        query_img_path: Union[str, Path],
        ref_img_path: Union[str, Path],
        ref_depth: Union[np.ndarray, torch.Tensor],
        ref_pose_w2c: np.ndarray,
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        **kwargs,
    ) -> LoFTRResult:
        """Load query + reference images from disk and run estimation.

        Same semantics as :meth:`estimate_pose`.
        """
        query_bgr = cv2.imread(str(query_img_path))
        if query_bgr is None:
            res = LoFTRResult()
            res.failure_reason = f"cannot_read_query: {query_img_path}"
            return res
        query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

        ref_bgr = cv2.imread(str(ref_img_path))
        if ref_bgr is None:
            res = LoFTRResult()
            res.failure_reason = f"cannot_read_ref: {ref_img_path}"
            return res
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

        return self.estimate_pose(
            query_rgb, ref_rgb, ref_depth,
            ref_pose_w2c, intrinsics, orig_hw,
            **kwargs,
        )

    # ------------------------------------------------------------------
    #  Low-level: extract correspondences without solving PnP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_correspondences(
        self,
        query_rgb: Union[np.ndarray, torch.Tensor],
        ref_rgb: Union[np.ndarray, torch.Tensor],
        ref_depth: Union[np.ndarray, torch.Tensor],
        ref_pose_w2c: np.ndarray,
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        *,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 3D-2D correspondences from a query-reference pair.

        Returns:
            pts_3d_world: (M, 3) 3D points in world frame
            pts_2d_query: (M, 2) 2D query keypoints at LoFTR resolution
            confidences:  (M,) match confidences
        """
        conf_thr = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        loftr_hw = _compute_loftr_resolution(orig_hw, self.loftr_long_edge)

        gray_q = _to_gray_tensor(query_rgb, loftr_hw, self.device)
        gray_r = _to_gray_tensor(ref_rgb, loftr_hw, self.device)

        match_out = self.loftr({"image0": gray_q, "image1": gray_r})
        kpts0 = match_out["keypoints0"].cpu().numpy()
        kpts1 = match_out["keypoints1"].cpu().numpy()
        conf = match_out["confidence"].cpu().numpy()

        if len(conf) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0,))

        keep = conf >= conf_thr
        kpts0, kpts1, conf = kpts0[keep], kpts1[keep], conf[keep]

        if len(conf) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0,))

        # Resolve depth
        if isinstance(ref_depth, torch.Tensor):
            d = ref_depth.detach().cpu().numpy()
            if d.ndim == 4:
                d = d[0, 0]
            elif d.ndim == 3:
                d = d[0]
        else:
            d = ref_depth
        depth_hw = d.shape[:2]

        intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)

        pts_3d_cam, kpts1_valid, valid_mask = _unproject_points(
            kpts1, d, intr_loftr, kpts_hw=loftr_hw, depth_hw=depth_hw,
        )
        kpts0_valid = kpts0[valid_mask]
        conf_valid = conf[valid_mask]

        if len(pts_3d_cam) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0,))

        # Transform to world frame
        ref_w2c = np.asarray(ref_pose_w2c, dtype=np.float64)
        ref_c2w = np.linalg.inv(ref_w2c)
        pts_3d_world = (ref_c2w[:3, :3] @ pts_3d_cam.T + ref_c2w[:3, 3:4]).T

        return pts_3d_world, kpts0_valid, conf_valid

    # ------------------------------------------------------------------
    #  Multi-reference: match against K candidates, pick best
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_pose_multi(
        self,
        query_rgb: Union[np.ndarray, torch.Tensor],
        ref_entries: Sequence[Dict],
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        **kwargs,
    ) -> LoFTRResult:
        """Match *query_rgb* against multiple reference entries.

        Each entry in *ref_entries* is a dict with:
          * ``'rgb'``:      reference image (array or tensor)
          * ``'depth'``:    reference depth map
          * ``'pose_w2c'``: (4, 4) reference w2c pose

        The candidate with the most PnP inliers is returned.
        """
        best: Optional[LoFTRResult] = None
        for entry in ref_entries:
            r = self.estimate_pose(
                query_rgb,
                entry["rgb"],
                entry["depth"],
                entry["pose_w2c"],
                intrinsics,
                orig_hw,
                **kwargs,
            )
            if best is None or r.num_inliers > best.num_inliers:
                best = r
        return best if best is not None else LoFTRResult(failure_reason="no_ref_entries")

    # ------------------------------------------------------------------
    #  Multi-reference accumulated PnP
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_pose_accumulated(
        self,
        query_rgb: Union[np.ndarray, torch.Tensor],
        ref_entries: Sequence[Dict],
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        *,
        confidence_threshold: Optional[float] = None,
        reproj_threshold: Optional[float] = None,
    ) -> LoFTRResult:
        """Match query against K references and solve ONE PnP from all matches.

        Instead of solving PnP per-reference and picking the best, this
        accumulates 3D-2D correspondences from ALL references and runs a
        single robust PnP-RANSAC.  More correspondences from diverse
        viewpoints give stronger geometric constraints.

        Each entry in *ref_entries* is a dict with:
          * ``'rgb'`` or ``'img_path'``: reference image
          * ``'depth'``:    reference depth map
          * ``'pose_w2c'``: (4, 4) reference w2c pose

        Returns:
            :class:`LoFTRResult` with accumulated statistics.
        """
        reproj_thr = reproj_threshold if reproj_threshold is not None else self.reproj_threshold

        all_pts_3d = []
        all_pts_2d = []
        all_conf = []
        total_raw = 0
        total_confident = 0
        total_depth_valid = 0

        for entry in ref_entries:
            # Resolve reference image
            if "rgb" in entry:
                ref_rgb = entry["rgb"]
            elif "img_path" in entry:
                ref_bgr = cv2.imread(str(entry["img_path"]))
                if ref_bgr is None:
                    continue
                ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            else:
                continue

            pts_3d, pts_2d, conf = self.extract_correspondences(
                query_rgb, ref_rgb,
                entry["depth"], entry["pose_w2c"],
                intrinsics, orig_hw,
                confidence_threshold=confidence_threshold,
            )
            total_depth_valid += len(pts_3d)
            if len(pts_3d) > 0:
                all_pts_3d.append(pts_3d)
                all_pts_2d.append(pts_2d)
                all_conf.append(conf)

        result = LoFTRResult()
        result.num_depth_valid = total_depth_valid

        if total_depth_valid < self.min_matches:
            result.failure_reason = f"accumulated_too_few ({total_depth_valid})"
            return result

        pts_3d_all = np.concatenate(all_pts_3d, axis=0)
        pts_2d_all = np.concatenate(all_pts_2d, axis=0)
        conf_all = np.concatenate(all_conf, axis=0)

        result.num_confident_matches = len(pts_3d_all)
        result.mean_confidence = float(conf_all.mean())

        # Solve PnP on accumulated correspondences
        loftr_hw = _compute_loftr_resolution(orig_hw, self.loftr_long_edge)
        intr_loftr = _scale_intrinsics(intrinsics, orig_hw, loftr_hw)

        pose_w2c, num_inliers = _solve_pnp(
            pts_3d_all, pts_2d_all, intr_loftr,
            reproj_threshold=reproj_thr,
            n_iters=self.pnp_iters,
            use_magsac=self.use_magsac,
        )
        result.num_inliers = num_inliers

        if pose_w2c is None:
            result.failure_reason = f"accumulated_pnp_failed (inliers={num_inliers})"
            return result

        result.pose_w2c = pose_w2c.astype(np.float32)
        result.success = True
        result.extra["num_refs_used"] = len(all_pts_3d)
        result.extra["total_correspondences"] = len(pts_3d_all)
        return result

    @torch.no_grad()
    def estimate_pose_multi_direct(
        self,
        query_img_path: Union[str, Path],
        ref_entries: Sequence[Dict],
        intrinsics: Dict[str, float],
        orig_hw: Tuple[int, int] = (1080, 1920),
        **kwargs,
    ) -> LoFTRResult:
        """Match query against multiple reference entries loaded from disk.

        Each entry in *ref_entries* is a dict with:
          * ``'img_path'``: path to reference image
          * ``'depth'``:    reference depth map
          * ``'pose_w2c'``: (4, 4) reference w2c pose
        """
        query_bgr = cv2.imread(str(query_img_path))
        if query_bgr is None:
            return LoFTRResult(failure_reason=f"cannot_read_query: {query_img_path}")
        query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

        best: Optional[LoFTRResult] = None
        for entry in ref_entries:
            ref_bgr = cv2.imread(str(entry["img_path"]))
            if ref_bgr is None:
                continue
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            r = self.estimate_pose(
                query_rgb, ref_rgb,
                entry["depth"], entry["pose_w2c"],
                intrinsics, orig_hw,
                **kwargs,
            )
            if best is None or r.num_inliers > best.num_inliers:
                best = r
        return best if best is not None else LoFTRResult(failure_reason="no_valid_ref_entries")
