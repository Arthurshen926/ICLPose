"""
RadioLocNet: Dual-Scale RADIO-DCFF Localization Network
========================================================
Matches RADIO query features against DCFF-rendered map features via
coarse-to-fine optical flow, then solves for 6-DOF pose via geometry.

Architecture:
  Stage 1 — Domain Alignment:
    2-layer MLP projection (64→64, GroupNorm, residual) for both query and
    rendered features, bridging the RADIO↔DCFF domain gap.  Features are
    L2-normalized before correlation so dot products equal cosine similarity.

  Stage 2 — Coarse Matching (H_c × W_c, e.g. 17×30):
    Global all-pairs correlation → single FlowRefinementHead step
    → flow_coarse + conf_coarse + hidden_coarse

  Stage 3 — Fine Matching (H_f × W_f, e.g. 68×120):
    Upsample coarse flow → 8 GRU iterations with warp-guided local
    correlation (radius=4) → flow_fine + conf_fine

  Stage 4 — Geometry Solver:
    Image Jacobian + weighted least-squares SE(3) solve → delta_xi

Input features:
  query:  64d fine (RADIO shallow PCA) + 64d coarse (RADIO deep PCA)
  render: 64d fine (DCFF explicit)     + 64d coarse (DCFF implicit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve, feature_metric_solve
from modules.lie_algebra import se3_exp


# ═══════════════════════════════════════════════════════════════════════════════
#  Correlation Functions
# ═══════════════════════════════════════════════════════════════════════════════

# Grid cache to avoid repeated meshgrid allocation
_GRID_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_base_grid(
    H: int, W: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cached (grid_x, grid_y) each shaped (1, H, W)."""
    key = (H, W, device)
    if key not in _GRID_CACHE:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        _GRID_CACHE[key] = (grid_x.unsqueeze(0), grid_y.unsqueeze(0))
    return _GRID_CACHE[key]


def global_correlation(
    fmap_q: torch.Tensor, fmap_r: torch.Tensor,
) -> torch.Tensor:
    """All-pairs dot-product correlation.
    (B, C, Hq, Wq) × (B, C, Hr, Wr) → (B, Hr*Wr, Hq, Wq)
    """
    B, C, Hq, Wq = fmap_q.shape
    Hr, Wr = fmap_r.shape[2:]
    f_q = fmap_q.reshape(B, C, Hq * Wq)
    f_r = fmap_r.reshape(B, C, Hr * Wr)
    corr = torch.einsum('bcn,bcm->bmn', f_q, f_r)
    return corr.reshape(B, Hr * Wr, Hq, Wq)


def local_correlation(
    fmap1: torch.Tensor, fmap2: torch.Tensor, radius: int = 4,
) -> torch.Tensor:
    """Local correlation in a (2r+1)² neighborhood. → (B, (2r+1)², H, W)"""
    B, C, H, W = fmap1.shape
    d = 2 * radius + 1
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    fmap2_unfold = fmap2_pad.unfold(2, d, 1).unfold(3, d, 1)
    fmap2_unfold = fmap2_unfold.reshape(B, C, H, W, d * d)
    corr = (fmap1.unsqueeze(-1) * fmap2_unfold).sum(dim=1)
    return corr.permute(0, 3, 1, 2).contiguous()


def guided_local_correlation(
    fmap_q: torch.Tensor, fmap_r: torch.Tensor,
    flow: torch.Tensor, radius: int = 4,
) -> torch.Tensor:
    """Warp-guided local correlation (RAFT-style).
    Warps reference features by current flow, then computes local (2r+1)²
    correlation. → (B, (2r+1)², H, W)
    """
    B, C, H, W = fmap_q.shape
    grid_x, grid_y = _get_base_grid(H, W, flow.device)
    grid_x = (grid_x + flow[:, 0]) / max(W - 1, 1) * 2.0 - 1.0
    grid_y = (grid_y + flow[:, 1]) / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped_r = F.grid_sample(
        fmap_r, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    return local_correlation(fmap_q, warped_r, radius=radius)


# ═══════════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class DomainProjection(nn.Module):
    """2-layer MLP projection with GroupNorm and optional residual.
    Handles domain gap between RADIO features and DCFF-rendered features.
    L2 normalization is applied externally before correlation.
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x
        return out


class ConvGRU(nn.Module):
    """Convolutional GRU with fused z+r gates."""

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
        self.conv_h = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = zr.chunk(2, dim=1)
        rh_x = torch.cat([r * h, x], dim=1)
        h_hat = torch.tanh(self.conv_h(rh_x))
        return (1 - z) * h + z * h_hat


class FlowRefinementHead(nn.Module):
    """Correlation → encoder → ConvGRU → flow delta + confidence.

    Args:
        corr_channels: number of correlation input channels
            - coarse (global): H_c * W_c
            - fine (local, r=4): (2*4+1)² = 81
        hidden_dim: GRU hidden state dimension
        context_dim: encoder output dimension fed to GRU
        conf_dim: confidence output channels (1 = scalar, 2 = directional)
    """

    def __init__(
        self,
        corr_channels: int,
        hidden_dim: int = 128,
        context_dim: int = 64,
        conf_dim: int = 1,
    ):
        super().__init__()
        self.conf_dim = conf_dim
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(corr_channels + 2 + conf_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, context_dim, 3, padding=1),
            nn.GELU(),
        )
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2 + conf_dim, 3, padding=1),
        )

    def forward(
        self,
        corr: torch.Tensor,
        hidden: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_flow: (B, 2, H, W)
            new_conf:   (B, conf_dim, H, W) ∈ (0, 1)
            new_hidden: (B, hidden_dim, H, W)
            flow_out:   (B, 2, H, W) = flow + delta_flow
        """
        inp = torch.cat([corr, flow, confidence], dim=1)
        inp_encoded = self.corr_encoder(inp)
        new_hidden = self.gru(hidden, inp_encoded)
        out = self.flow_head(new_hidden)
        delta_flow = out[:, :2]
        new_conf = torch.sigmoid(out[:, 2:])
        return delta_flow, new_conf, new_hidden, flow + delta_flow


class ContextAdapter(nn.Module):
    """Fuse upsampled hidden state with scale-specific query features."""

    def __init__(self, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + feat_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, h_up: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        return h_up + self.net(torch.cat([h_up, q_feat], dim=1))


class PoseRegressionHead(nn.Module):
    """Lightweight CNN: feature residual map → 6-DOF pose delta.

    Takes the concatenation of projected query + rendered features (or just
    the residual) at fine resolution and outputs a 6-DOF se(3) vector.
    Much more robust than analytical IC solver when features are noisy.

    Architecture:
      Conv stride-2 blocks to downsample → AdaptiveAvgPool → FC → 6D
    """

    def __init__(self, in_channels: int = 128, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        # Initialize last layer near-zero so initial predictions are conservative
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) concatenated or residual features
        Returns:
            delta_xi: (B, 6) se(3) pose update (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
        """
        feat = self.encoder(x).flatten(1)
        return self.fc(feat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Network
# ═══════════════════════════════════════════════════════════════════════════════

class RadioLocNet(nn.Module):
    """Dual-scale RADIO-DCFF localization network.

    Config parameters:
        feature_dim: input feature channels (default 64)
        match_dim:   internal matching dimension (default 64)
        hidden_dim:  GRU hidden state dimension (default 128)
        coarse_hw:   coarse resolution (H_c, W_c), default (17, 30)
        fine_hw:     fine resolution (H_f, W_f), default (68, 120)
        fine_iters:  number of fine GRU iterations (default 8)
        corr_radius: local correlation radius (default 4)
        damping:     LM damping for geometry solver (default 1e-3)
        irls_iters:  IRLS robust estimation iterations (default 0)
        robust_kernel: robust kernel type ('huber', 'gm', 'gnc_gm')
        img_hw:      original image resolution for intrinsics scaling
    """

    def __init__(
        self,
        feature_dim: int = 64,
        match_dim: int = 64,
        hidden_dim: int = 128,
        coarse_hw: Tuple[int, int] = (17, 30),
        fine_hw: Tuple[int, int] = (68, 120),
        fine_iters: int = 8,
        corr_radius: int = 4,
        damping: float = 1e-3,
        irls_iters: int = 0,
        robust_kernel: str = 'huber',
        intrinsics: Optional[Dict[str, float]] = None,
        img_hw: Tuple[int, int] = (1080, 1920),
    ):
        super().__init__()

        self.COARSE_HW = tuple(coarse_hw)
        self.FINE_HW = tuple(fine_hw)
        self.IMG_HW = tuple(img_hw)
        self.BASE_INTRINSICS = intrinsics or {
            'fx': 1670.0, 'fy': 1670.0, 'cx': 960.0, 'cy': 540.0,
        }
        self.fine_iters = fine_iters
        self.corr_radius = corr_radius
        self.damping = damping
        self.irls_iters = irls_iters
        self.robust_kernel = robust_kernel

        # Domain-alignment projections (separate for fine and coarse)
        self.proj_fine_q = DomainProjection(feature_dim, match_dim)
        self.proj_fine_r = DomainProjection(feature_dim, match_dim)
        self.proj_coarse_q = DomainProjection(feature_dim, match_dim)
        self.proj_coarse_r = DomainProjection(feature_dim, match_dim)

        # Coarse context encoder (query features → initial hidden state)
        self.context_net = nn.Sequential(
            nn.Conv2d(match_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # Coarse flow head: global correlation → flow + conf
        coarse_corr_ch = coarse_hw[0] * coarse_hw[1]
        self.coarse_head = FlowRefinementHead(
            coarse_corr_ch, hidden_dim, context_dim=match_dim,
        )

        # Fine flow head: guided local correlation → flow + conf
        fine_corr_ch = (2 * corr_radius + 1) ** 2
        self.fine_context = ContextAdapter(hidden_dim, match_dim)
        self.fine_head = FlowRefinementHead(
            fine_corr_ch, hidden_dim, context_dim=match_dim,
        )

        # Pose regression head: feature concat → 6-DOF delta
        # Input: projected query (match_dim) + projected rendered (match_dim)
        self.pose_head = PoseRegressionHead(
            in_channels=match_dim * 2,
            hidden=256,
        )

        self._print_summary()

    def _print_summary(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RadioLocNet] coarse={self.COARSE_HW}, fine={self.FINE_HW}, "
              f"iters={self.fine_iters}, radius={self.corr_radius}")
        print(f"  params={n_params:,} ({n_train:,} trainable)")

    def _scale_intrinsics(self, tH: int, tW: int) -> Dict[str, float]:
        """Scale base intrinsics to target resolution."""
        return {
            'fx': self.BASE_INTRINSICS['fx'] * tW / self.IMG_HW[1],
            'fy': self.BASE_INTRINSICS['fy'] * tH / self.IMG_HW[0],
            'cx': self.BASE_INTRINSICS['cx'] * tW / self.IMG_HW[1],
            'cy': self.BASE_INTRINSICS['cy'] * tH / self.IMG_HW[0],
        }

    def forward(
        self,
        q_fine: torch.Tensor,
        q_coarse: torch.Tensor,
        r_fine: torch.Tensor,
        r_coarse: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            q_fine:    (B, 64, H_f, W_f) query fine features (RADIO shallow PCA)
            q_coarse:  (B, 64, H_c, W_c) query coarse features (RADIO deep PCA)
            r_fine:    (B, 64, H_f, W_f) rendered fine features (DCFF explicit)
            r_coarse:  (B, 64, H_c, W_c) rendered coarse features (DCFF implicit)
            depth:     (B, H_f, W_f) rendered depth at current pose
            intrinsics: optional per-sample intrinsics dict {fx, fy, cx, cy}

        Returns:
            dict with: delta_xi, flow_coarse, flow_fine, conf_fine,
                       fine_flow_preds, fine_conf_preds
        """
        B = q_fine.shape[0]
        device = q_fine.device
        H_c, W_c = self.COARSE_HW
        H_f, W_f = self.FINE_HW

        # ── 1. Resize to canonical resolutions ──
        q_coarse_in = F.interpolate(q_coarse, (H_c, W_c), mode='bilinear', align_corners=False)
        r_coarse_in = F.interpolate(r_coarse, (H_c, W_c), mode='bilinear', align_corners=False)
        q_fine_in = F.interpolate(q_fine, (H_f, W_f), mode='bilinear', align_corners=False)
        r_fine_in = F.interpolate(r_fine, (H_f, W_f), mode='bilinear', align_corners=False)

        # ── 2. Domain alignment projection ──
        qc = self.proj_coarse_q(q_coarse_in)
        rc = self.proj_coarse_r(r_coarse_in)
        qf = self.proj_fine_q(q_fine_in)
        rf = self.proj_fine_r(r_fine_in)

        # ── 3. Coarse: global correlation → single GRU step ──
        # L2-normalize so dot-product correlation = cosine similarity
        qc = F.normalize(qc, p=2, dim=1)
        rc = F.normalize(rc, p=2, dim=1)
        coarse_corr = global_correlation(qc, rc)
        h_coarse = self.context_net(qc)
        flow_c = torch.zeros(B, 2, H_c, W_c, device=device)
        conf_c = torch.ones(B, 1, H_c, W_c, device=device) * 0.5

        _, conf_c, h_coarse, flow_c = self.coarse_head(
            coarse_corr, h_coarse, flow_c, conf_c,
        )

        # ── 4. Fine: upsample coarse → guided local correlation × N iters ──
        scale_x = W_f / W_c
        scale_y = H_f / H_c
        flow_f = F.interpolate(flow_c, (H_f, W_f), mode='bilinear', align_corners=False)
        flow_f[:, 0] *= scale_x
        flow_f[:, 1] *= scale_y
        conf_f = F.interpolate(conf_c, (H_f, W_f), mode='bilinear', align_corners=False)

        h_fine_up = F.interpolate(h_coarse, (H_f, W_f), mode='bilinear', align_corners=False)
        h_fine = self.fine_context(h_fine_up, qf)

        fine_flow_preds: List[torch.Tensor] = []
        fine_conf_preds: List[torch.Tensor] = []

        # L2-normalize fine features so local correlation = cosine similarity
        # (done after fine_context which needs un-normalized features)
        qf_norm = F.normalize(qf, p=2, dim=1)
        rf_norm = F.normalize(rf, p=2, dim=1)

        for _ in range(self.fine_iters):
            fine_corr = guided_local_correlation(
                qf_norm, rf_norm, flow_f, radius=self.corr_radius,
            )
            _, conf_f, h_fine, flow_f = self.fine_head(
                fine_corr, h_fine, flow_f, conf_f,
            )
            fine_flow_preds.append(flow_f)
            fine_conf_preds.append(conf_f)

        result: Dict[str, object] = {
            'flow_coarse': flow_c,
            'conf_coarse': conf_c,
            'flow_fine': flow_f,
            'conf_fine': conf_f,
            'fine_flow_preds': fine_flow_preds,
            'fine_conf_preds': fine_conf_preds,
        }

        # ── 5. Geometry Solver ──
        if depth is not None:
            with torch.cuda.amp.autocast(enabled=False):
                flow_f32 = flow_f.float()
                conf_f32 = conf_f.float()
                depth_f32 = depth.float()

                # Resize depth to fine resolution
                if depth_f32.ndim == 3:
                    depth_f32 = depth_f32.unsqueeze(1)
                depth_solve = F.interpolate(
                    depth_f32, (H_f, W_f),
                    mode='bilinear', align_corners=False,
                ).squeeze(1)

                # Image Jacobian at fine resolution
                solve_intrinsics = intrinsics or self._scale_intrinsics(H_f, W_f)
                Ju, Jv, valid = compute_image_jacobian(depth_solve, solve_intrinsics)

                # Detach confidence to prevent pose loss → confidence collapse
                conf_detached = conf_f32.detach()

                delta_xi = diff_pose_solve(
                    flow_f32, conf_detached, Ju, Jv, valid,
                    damping=self.damping,
                    irls_iters=self.irls_iters,
                    robust_kernel=self.robust_kernel,
                )
                result['delta_xi'] = delta_xi

        return result

    def forward_feature_metric(
        self,
        q_fine: torch.Tensor,
        r_fine: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Feature-metric pose solver (no flow, no correlation).

        Directly minimizes the feature residual between query and rendered
        features using the Inverse-Compositional Gauss-Newton method.
        Much more robust than flow-based matching when cos_sim is low (~0.35).

        Uses the domain-aligned (projected + normalized) features, so the
        projection layers are still trained/used.

        Args:
            q_fine:    (B, 64, H, W) query fine features
            r_fine:    (B, 64, H, W) rendered fine features
            depth:     (B, H, W) rendered depth
            intrinsics: camera intrinsics dict

        Returns:
            dict with: delta_xi (B, 6), feat_residual (B, C, H, W)
        """
        H_f, W_f = self.FINE_HW

        q = F.interpolate(q_fine, (H_f, W_f), mode='bilinear', align_corners=False)
        r = F.interpolate(r_fine, (H_f, W_f), mode='bilinear', align_corners=False)

        # Domain alignment
        qf = self.proj_fine_q(q)
        rf = self.proj_fine_r(r)

        # L2 normalize
        qf = F.normalize(qf, p=2, dim=1)
        rf = F.normalize(rf, p=2, dim=1)

        # Depth at fine resolution
        with torch.cuda.amp.autocast(enabled=False):
            depth_f = depth.float()
            if depth_f.ndim == 3:
                depth_f = depth_f.unsqueeze(1)
            depth_f = F.interpolate(depth_f, (H_f, W_f), mode='bilinear', align_corners=False).squeeze(1)

            solve_intr = intrinsics or self._scale_intrinsics(H_f, W_f)

            delta_xi, residual = feature_metric_solve(
                qf.float(), rf.float(), depth_f, solve_intr,
                damping=self.damping,
            )

        return {
            'delta_xi': delta_xi,
            'feat_residual': residual,
        }

    def forward_regression(
        self,
        q_fine: torch.Tensor,
        r_fine: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Learned pose regression from feature comparison.

        Concatenates projected+normalized query and rendered features
        and feeds them through PoseRegressionHead → 6-DOF delta_xi.
        No depth needed, no analytical solver. Fully learned.

        Args:
            q_fine: (B, 64, H, W) query fine features
            r_fine: (B, 64, H, W) rendered fine features

        Returns:
            dict with: delta_xi (B, 6)
        """
        H_f, W_f = self.FINE_HW
        q = F.interpolate(q_fine, (H_f, W_f), mode='bilinear', align_corners=False)
        r = F.interpolate(r_fine, (H_f, W_f), mode='bilinear', align_corners=False)

        # Domain alignment + L2 normalize
        qf = F.normalize(self.proj_fine_q(q), p=2, dim=1)
        rf = F.normalize(self.proj_fine_r(r), p=2, dim=1)

        # Concatenate and regress
        delta_xi = self.pose_head(torch.cat([qf, rf], dim=1))
        return {'delta_xi': delta_xi}

    def compute_gt_flow(
        self,
        pose_init: torch.Tensor,
        pose_gt: torch.Tensor,
        depth: torch.Tensor,
        target_hw: Tuple[int, int],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GT flow from pose difference and depth for supervision.

        Args:
            pose_init: (B, 4, 4) current w2c pose
            pose_gt:   (B, 4, 4) GT w2c pose
            depth:     (B, H, W) depth at pose_init viewpoint
            target_hw: (tH, tW) target flow resolution
            intrinsics: camera intrinsics dict

        Returns:
            flow_gt:    (B, 2, tH, tW) ground-truth optical flow
            valid_mask: (B, 1, tH, tW) validity mask
        """
        B, H, W = depth.shape
        device = depth.device
        intr = intrinsics or self._scale_intrinsics(H, W)
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']

        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth
        Y = (v_coords - cy) / fy * depth
        Z = depth

        pts = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = pose_gt @ torch.linalg.inv(pose_init)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)

        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([
            u_gt - u_coords.unsqueeze(1),
            v_gt - v_coords.unsqueeze(1),
        ], dim=1)

        valid = (
            (depth.unsqueeze(1) > 0.05)
            & (Z_gt_raw > 0.1)
            & (u_gt > -0.5) & (u_gt < W - 0.5)
            & (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()

        flow_gt = flow_gt * valid

        tH, tW = target_hw
        if tH != H or tW != W:
            sx, sy = tW / W, tH / H
            flow_gt = F.interpolate(flow_gt, (tH, tW), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= sx
            flow_gt[:, 1] *= sy
            valid = F.interpolate(valid, (tH, tW), mode='nearest')

        return flow_gt, valid
