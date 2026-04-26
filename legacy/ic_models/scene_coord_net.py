"""
Scene Coordinate Regression (SCR) network.

Maps RADIO visual features directly to 3D world coordinates per pixel,
then estimates 6-DOF pose via PnP+RANSAC. Bypasses feature matching entirely.
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + residual


class SceneCoordNet(nn.Module):
    """
    Scene Coordinate Regression network.

    Input:  RADIO features (B, 64, 68, 120)
    Output: 3D world coordinates (B, 3, 68, 120) per pixel

    Architecture:
        - Positional encoding: normalized UV grid concatenated with input features
        - Stem: Conv3×3 lifting 66 → 128 channels
        - Body: 4 residual blocks at 128 channels for large receptive field
        - Coord head: Conv1×1 → 3 (unbounded X, Y, Z)
        - Confidence head: Conv1×1 → 1 → sigmoid (per-pixel reliability)
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_pos_encoding = use_pos_encoding

        stem_in = in_channels + 2 if use_pos_encoding else in_channels

        # Stem: lift to hidden dimension
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Residual body
        self.body = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # 3D coordinate head — no output activation (world coords are unbounded)
        self.coord_head = nn.Conv2d(hidden_channels, 3, 1)

        # Confidence head — sigmoid gives [0, 1] reliability per pixel
        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

        # Cache for the positional encoding grid (lazily created)
        self._pos_grid: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Coord head: small init so initial predictions are near the origin
        nn.init.normal_(self.coord_head.weight, std=0.01)
        nn.init.zeros_(self.coord_head.bias)

    # ------------------------------------------------------------------
    # Positional encoding
    # ------------------------------------------------------------------

    def _get_pos_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (1, 2, H, W) normalised UV grid in [-1, 1]."""
        if self._pos_grid is not None and self._pos_grid.shape[2:] == (H, W):
            return self._pos_grid.to(device)

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
        self._pos_grid = grid
        return grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, radio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radio_features: (B, 64, H, W) RADIO fine features.

        Returns:
            coords:     (B, 3, H, W) predicted 3D world coordinates per pixel.
            confidence: (B, 1, H, W) per-pixel reliability in [0, 1].
        """
        B, C, H, W = radio_features.shape

        x = radio_features
        if self.use_pos_encoding:
            pos = self._get_pos_grid(H, W, x.device).expand(B, -1, -1, -1)
            x = torch.cat([x, pos], dim=1)

        x = self.stem(x)
        x = self.body(x)

        coords = self.coord_head(x)
        confidence = self.confidence_head(x)

        return coords, confidence

    # ------------------------------------------------------------------
    # PnP + RANSAC pose solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_and_solve(
        self,
        radio_features: torch.Tensor,
        intrinsics: Dict[str, float],
        confidence_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full pipeline: features → 3D coords → PnP+RANSAC → pose.

        Args:
            radio_features: (B, 64, H, W) RADIO fine features.
            intrinsics: dict with 'fx', 'fy', 'cx', 'cy' at the feature-map
                        resolution (68×120).

        Returns:
            poses_w2c: (B, 4, 4) estimated world-to-camera poses.
                       Identity for samples where PnP fails.
            info: dict with:
                - 'coords':      (B, 3, H, W) predicted scene coordinates
                - 'confidence':   (B, 1, H, W) confidence maps
                - 'inlier_counts': list[int] per sample
                - 'success':      list[bool] per sample
        """
        coords, confidence = self.forward(radio_features)
        B, _, H, W = coords.shape

        # Build camera matrix at feature-map resolution
        K = np.array(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Build pixel coordinate grid (u, v) for the feature map
        us = torch.arange(W, device=coords.device, dtype=coords.dtype) + 0.5
        vs = torch.arange(H, device=coords.device, dtype=coords.dtype) + 0.5
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        pixel_coords = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

        coords_np = coords.cpu().float().numpy()          # (B, 3, H, W)
        conf_np = confidence.cpu().float().numpy()         # (B, 1, H, W)
        pixels_np = pixel_coords.cpu().numpy()             # (H, W, 2)

        poses_list = []
        inlier_counts = []
        successes = []

        for b in range(B):
            pose_b, n_inliers, ok = self._solve_pnp_single(
                coords_np[b], conf_np[b, 0], pixels_np, K, confidence_threshold
            )
            poses_list.append(pose_b)
            inlier_counts.append(n_inliers)
            successes.append(ok)

        poses_w2c = torch.from_numpy(np.stack(poses_list, axis=0)).float()
        poses_w2c = poses_w2c.to(radio_features.device)

        info = {
            "coords": coords,
            "confidence": confidence,
            "inlier_counts": inlier_counts,
            "success": successes,
        }
        return poses_w2c, info

    # ------------------------------------------------------------------
    # Single-sample PnP solver (runs on CPU / numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_pnp_single(
        coords_3hw: np.ndarray,
        conf_hw: np.ndarray,
        pixels_hw2: np.ndarray,
        K: np.ndarray,
        conf_thresh: float,
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Solve PnP+RANSAC for a single sample.

        Returns:
            pose_w2c: (4, 4) world-to-camera pose (identity on failure).
            n_inliers: number of RANSAC inliers.
            success: whether PnP succeeded.
        """
        identity = np.eye(4, dtype=np.float64)

        # Flatten spatial dims: (3, H, W) → (N, 3), (H, W) → (N,), (H, W, 2) → (N, 2)
        pts_3d = coords_3hw.transpose(1, 2, 0).reshape(-1, 3)   # (N, 3)
        confs = conf_hw.reshape(-1)                               # (N,)
        pts_2d = pixels_hw2.reshape(-1, 2)                        # (N, 2)

        # Filter by confidence and validity
        valid = (confs > conf_thresh) & np.all(np.isfinite(pts_3d), axis=1)
        pts_3d = pts_3d[valid].astype(np.float64)
        pts_2d = pts_2d[valid].astype(np.float64)

        if len(pts_3d) < 6:
            return identity, 0, False

        # RANSAC PnP
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=8.0,
            iterationsCount=1000,
            confidence=0.999,
        )

        if not ok or inliers is None or len(inliers) < 6:
            return identity, 0, False

        n_inliers = len(inliers)

        # Refine on inliers with iterative solver
        pts_3d_in = pts_3d[inliers.ravel()]
        pts_2d_in = pts_2d[inliers.ravel()]

        ok_ref, rvec_ref, tvec_ref = cv2.solvePnP(
            objectPoints=pts_3d_in,
            imagePoints=pts_2d_in,
            cameraMatrix=K,
            distCoeffs=None,
            rvec=rvec.copy(),
            tvec=tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if ok_ref:
            rvec, tvec = rvec_ref, tvec_ref

        # Convert to 4×4 w2c matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.ravel()

        return pose, n_inliers, True

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SceneCoordNet("
            f"in={self.in_channels}, hidden={self.hidden_channels}, "
            f"pos_enc={self.use_pos_encoding}, "
            f"params={self.param_count():,})"
        )


class SceneCoordNetV2(nn.Module):
    """
    Scene Coordinate Regression v2 with global context from coarse features.

    Inputs:
        fine_features:   (B, 64, 68, 120) — per-pixel detail from fine_geo
        coarse_features: (B, 64, 17, 30) — global scene context from coarse_sem
                         (downsampled at load time)

    Architecture:
        - Coarse branch: Conv encoder → AdaptiveAvgPool → 128-d global vector
        - Fine branch: fine_feat(64) + pos_enc(2) + global(128) = 194 → stem
        - Body: 6 ResBlocks with 256 channels
        - Heads: coord (3-ch) + confidence (1-ch sigmoid)
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 256,
        num_res_blocks: int = 6,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_pos_encoding = use_pos_encoding

        # Coarse branch: (B, 64, H_c, W_c) → (B, 128, 1, 1) global context
        self.coarse_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B, 128, 1, 1)
        )

        # Stem input: fine(64) + pos(2) + global(128) = 194
        stem_in = in_channels + 128
        if use_pos_encoding:
            stem_in += 2  # 194 total

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Deeper residual body
        self.body = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # 3D coordinate head — unbounded
        self.coord_head = nn.Conv2d(hidden_channels, 3, 1)

        # Confidence head — sigmoid [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        self._pos_grid: Optional[torch.Tensor] = None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small init for coord head so predictions start near origin
        nn.init.normal_(self.coord_head.weight, std=0.01)
        nn.init.zeros_(self.coord_head.bias)

    def _get_pos_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (1, 2, H, W) normalised UV grid in [-1, 1]."""
        if self._pos_grid is not None and self._pos_grid.shape[2:] == (H, W):
            return self._pos_grid.to(device)

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self._pos_grid = grid
        return grid

    def forward(
        self,
        fine_features: torch.Tensor,
        coarse_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fine_features:   (B, 64, H, W) fine_geo features.
            coarse_features: (B, 64, H_c, W_c) coarse_sem features (any resolution).

        Returns:
            coords:     (B, 3, H, W) predicted 3D world coordinates per pixel.
            confidence: (B, 1, H, W) per-pixel reliability in [0, 1].
        """
        B, _, H, W = fine_features.shape

        # Global context from coarse features
        global_ctx = self.coarse_encoder(coarse_features)  # (B, 128, 1, 1)
        global_ctx = global_ctx.expand(-1, -1, H, W)       # (B, 128, H, W)

        # Build input: fine + pos + global
        parts = [fine_features, global_ctx]
        if self.use_pos_encoding:
            pos = self._get_pos_grid(H, W, fine_features.device).expand(B, -1, -1, -1)
            parts.append(pos)

        x = torch.cat(parts, dim=1)  # (B, 194, H, W)

        x = self.stem(x)
        x = self.body(x)

        coords = self.coord_head(x)
        confidence = self.confidence_head(x)
        return coords, confidence

    @torch.no_grad()
    def predict_and_solve(
        self,
        fine_features: torch.Tensor,
        coarse_features: torch.Tensor,
        intrinsics: Dict[str, float],
        confidence_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full pipeline: features → 3D coords → PnP+RANSAC → pose.

        Args:
            fine_features:   (B, 64, H, W) fine_geo features.
            coarse_features: (B, 64, H_c, W_c) coarse_sem features.
            intrinsics: dict with 'fx', 'fy', 'cx', 'cy'.

        Returns:
            poses_w2c: (B, 4, 4) estimated world-to-camera poses.
            info: dict with coords, confidence, inlier_counts, success.
        """
        coords, confidence = self.forward(fine_features, coarse_features)
        B, _, H, W = coords.shape

        K = np.array(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        us = torch.arange(W, device=coords.device, dtype=coords.dtype) + 0.5
        vs = torch.arange(H, device=coords.device, dtype=coords.dtype) + 0.5
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        pixel_coords = torch.stack([grid_u, grid_v], dim=-1)

        coords_np = coords.cpu().float().numpy()
        conf_np = confidence.cpu().float().numpy()
        pixels_np = pixel_coords.cpu().numpy()

        poses_list = []
        inlier_counts = []
        successes = []

        for b in range(B):
            pose_b, n_inliers, ok = SceneCoordNet._solve_pnp_single(
                coords_np[b], conf_np[b, 0], pixels_np, K, confidence_threshold
            )
            poses_list.append(pose_b)
            inlier_counts.append(n_inliers)
            successes.append(ok)

        poses_w2c = torch.from_numpy(np.stack(poses_list, axis=0)).float()
        poses_w2c = poses_w2c.to(fine_features.device)

        info = {
            "coords": coords,
            "confidence": confidence,
            "inlier_counts": inlier_counts,
            "success": successes,
        }
        return poses_w2c, info

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SceneCoordNetV2("
            f"in={self.in_channels}, hidden={self.hidden_channels}, "
            f"pos_enc={self.use_pos_encoding}, "
            f"params={self.param_count():,})"
        )


# ------------------------------------------------------------------
# Default intrinsics for OldHospital at 68×120
# ------------------------------------------------------------------

OLDHOSPITAL_INTRINSICS_68x120 = {
    "fx": 104.6,
    "fy": 105.4,
    "cx": 60.0,
    "cy": 34.0,
}


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    net = SceneCoordNet()
    print(net)

    x = torch.randn(2, 64, 68, 120)
    coords, conf = net(x)
    print(f"coords: {coords.shape}  confidence: {conf.shape}")

    poses, info = net.predict_and_solve(x, OLDHOSPITAL_INTRINSICS_68x120)
    print(f"poses: {poses.shape}  success: {info['success']}")

    # V2 test
    net2 = SceneCoordNetV2()
    print(net2)

    fine = torch.randn(2, 64, 68, 120)
    coarse = torch.randn(2, 64, 17, 30)
    coords2, conf2 = net2(fine, coarse)
    print(f"v2 coords: {coords2.shape}  confidence: {conf2.shape}")

    poses2, info2 = net2.predict_and_solve(fine, coarse, OLDHOSPITAL_INTRINSICS_68x120)
    print(f"v2 poses: {poses2.shape}  success: {info2['success']}")
