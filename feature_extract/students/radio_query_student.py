import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = ConvNormAct(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.pw = ConvNormAct(channels, channels, kernel_size=1, stride=1)

    def forward(self, x):
        return x + self.pw(self.dw(x))


class DepthAwareLocalMatcher(nn.Module):
    """Small residual head that refines local correlation logits with geometry context."""

    def __init__(
        self,
        radius=4,
        hidden_dim=64,
        zero_init=True,
        residual_scale=1.0,
    ):
        super().__init__()
        self.radius = int(radius)
        self.corr_channels = (2 * self.radius + 1) ** 2
        context_channels = 5  # log-depth, inverse-depth, x/y pixel coords, valid/depth mask.
        self.refine = nn.Sequential(
            ConvNormAct(self.corr_channels + context_channels, hidden_dim, kernel_size=3),
            ResidualDepthwiseBlock(hidden_dim),
            nn.Conv2d(hidden_dim, self.corr_channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.refine[-1].weight)
            if self.refine[-1].bias is not None:
                nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _masked_standardize(value, mask):
        if mask is None:
            mean = value.mean(dim=(-2, -1), keepdim=True)
            std = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp(min=1e-6)
            return (value - mean) / std
        mask = mask.float()
        denom = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        mean = (value * mask).sum(dim=(-2, -1), keepdim=True) / denom
        var = ((value - mean).square() * mask).sum(dim=(-2, -1), keepdim=True) / denom
        return (value - mean) / torch.sqrt(var + 1e-6)

    def _build_context(self, corr, depth=None, valid_mask=None):
        B, _C, H, W = corr.shape
        device = corr.device
        dtype = corr.dtype
        if depth is None:
            depth_ch = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            inv_depth_ch = torch.zeros_like(depth_ch)
            depth_valid = torch.ones_like(depth_ch)
        else:
            depth_ch = depth.float()
            if depth_ch.ndim == 3:
                depth_ch = depth_ch.unsqueeze(1)
            if depth_ch.shape[-2:] != (H, W):
                depth_ch = F.interpolate(depth_ch, (H, W), mode="bilinear", align_corners=False)
            depth_valid = (depth_ch > 0.05).float()
            log_depth = torch.log(depth_ch.clamp(min=0.05))
            inv_depth = 1.0 / depth_ch.clamp(min=0.05)
            depth_ch = self._masked_standardize(log_depth, depth_valid).to(dtype=dtype)
            inv_depth_ch = self._masked_standardize(inv_depth, depth_valid).to(dtype=dtype)

        if valid_mask is not None:
            valid = valid_mask.float()
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            if valid.shape[-2:] != (H, W):
                valid = F.interpolate(valid, (H, W), mode="nearest")
            depth_valid = depth_valid * valid

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([depth_ch, inv_depth_ch, xy, depth_valid.to(dtype=dtype)], dim=1)

    def forward(self, corr, depth=None, valid_mask=None):
        if corr.shape[1] != self.corr_channels:
            raise ValueError(
                f"DepthAwareLocalMatcher expected {self.corr_channels} channels "
                f"for radius={self.radius}, got {corr.shape[1]}"
            )
        context = self._build_context(corr, depth=depth, valid_mask=valid_mask)
        residual = self.refine(torch.cat([corr.float(), context.float()], dim=1))
        return corr + self.residual_scale.to(dtype=corr.dtype) * residual.to(dtype=corr.dtype)


class DepthAwareLocalFlowHead(nn.Module):
    """Predict local pixel flow and confidence from correlation plus rendered geometry."""

    def __init__(
        self,
        radius=4,
        hidden_dim=64,
        zero_init=True,
        max_flow=None,
        base_flow_mode="none",
        base_temperature=0.05,
    ):
        super().__init__()
        self.radius = int(radius)
        self.corr_channels = (2 * self.radius + 1) ** 2
        self.max_flow = float(max_flow) if max_flow is not None else float(self.radius)
        self.base_flow_mode = str(base_flow_mode or "none").lower()
        if self.base_flow_mode not in {"none", "softargmax", "argmax"}:
            raise ValueError(
                "base_flow_mode must be one of {'none', 'softargmax', 'argmax'}, "
                f"got {base_flow_mode!r}"
            )
        self.base_temperature = float(base_temperature)
        context_channels = 8
        self.predict = nn.Sequential(
            ConvNormAct(self.corr_channels + context_channels, hidden_dim, kernel_size=3),
            ResidualDepthwiseBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
        )
        if zero_init:
            nn.init.zeros_(self.predict[-1].weight)
            if self.predict[-1].bias is not None:
                nn.init.zeros_(self.predict[-1].bias)

    _masked_standardize = staticmethod(DepthAwareLocalMatcher._masked_standardize)

    def _build_context(self, corr, depth=None, valid_mask=None):
        geometry_context = DepthAwareLocalMatcher._build_context(
            self,
            corr,
            depth=depth,
            valid_mask=valid_mask,
        )
        prior_flow = self._softargmax_flow(corr, temperature=self.base_temperature)
        top2 = torch.topk(corr.float(), k=min(2, corr.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            peak_margin = torch.zeros_like(top2[:, :1])
        else:
            peak_margin = top2[:, :1] - top2[:, 1:2]
        return torch.cat(
            [
                geometry_context,
                prior_flow.to(dtype=geometry_context.dtype),
                peak_margin.to(dtype=geometry_context.dtype),
            ],
            dim=1,
        )

    def _softargmax_flow(self, corr, *, temperature):
        window = 2 * self.radius + 1
        yy, xx = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            indexing="ij",
        )
        dx = xx.reshape(1, window * window, 1, 1)
        dy = yy.reshape(1, window * window, 1, 1)
        probs = torch.softmax(corr.float() / max(float(temperature), 1e-6), dim=1).to(dtype=corr.dtype)
        return torch.cat(
            [
                (probs * dx).sum(dim=1, keepdim=True),
                (probs * dy).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )

    def _base_flow(self, corr):
        if self.base_flow_mode == "none":
            return corr.new_zeros(corr.shape[0], 2, corr.shape[2], corr.shape[3])
        if self.base_flow_mode == "softargmax":
            return self._softargmax_flow(corr, temperature=self.base_temperature)
        window = 2 * self.radius + 1
        yy, xx = torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            torch.arange(-self.radius, self.radius + 1, device=corr.device, dtype=corr.dtype),
            indexing="ij",
        )
        dx = xx.reshape(1, window * window, 1, 1)
        dy = yy.reshape(1, window * window, 1, 1)
        idx = corr.argmax(dim=1, keepdim=True)
        dx_map = dx.expand(corr.shape[0], -1, corr.shape[2], corr.shape[3]).gather(1, idx)
        dy_map = dy.expand(corr.shape[0], -1, corr.shape[2], corr.shape[3]).gather(1, idx)
        return torch.cat([dx_map, dy_map], dim=1)

    def forward(self, corr, depth=None, valid_mask=None):
        if corr.shape[1] != self.corr_channels:
            raise ValueError(
                f"DepthAwareLocalFlowHead expected {self.corr_channels} channels "
                f"for radius={self.radius}, got {corr.shape[1]}"
            )
        context = self._build_context(corr, depth=depth, valid_mask=valid_mask)
        raw = self.predict(torch.cat([corr.float(), context.float()], dim=1))
        base_flow = self._base_flow(corr)
        residual_flow = torch.tanh(raw[:, :2]) * self.max_flow
        flow = (base_flow.float() + residual_flow).clamp(-self.max_flow, self.max_flow)
        confidence_logits = raw[:, 2:3]
        confidence = torch.sigmoid(confidence_logits)
        return {
            "flow": flow.to(dtype=corr.dtype),
            "confidence": confidence.to(dtype=corr.dtype),
            "confidence_logits": confidence_logits.to(dtype=corr.dtype),
        }


def _softplus_inverse(x: float) -> float:
    if x <= 0:
        raise ValueError(f"softplus inverse expects positive input, got {x}")
    return math.log(math.expm1(x))


class LocalCorrProjector(nn.Module):
    """Shared residual projection for map/query descriptors used by local correlation."""

    def __init__(
        self,
        feature_dim,
        hidden_dim=96,
        output_dim=None,
        zero_init=True,
        l2_normalize=True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim) if output_dim is not None else self.feature_dim
        self.l2_normalize = bool(l2_normalize)
        hidden_dim = int(hidden_dim)
        self.refine = nn.Sequential(
            ConvNormAct(self.feature_dim, hidden_dim, kernel_size=3),
            ResidualDepthwiseBlock(hidden_dim),
            nn.Conv2d(hidden_dim, self.output_dim, kernel_size=1),
        )
        self.skip = (
            nn.Identity()
            if self.output_dim == self.feature_dim
            else nn.Conv2d(self.feature_dim, self.output_dim, kernel_size=1, bias=False)
        )
        if zero_init:
            nn.init.zeros_(self.refine[-1].weight)
            if self.refine[-1].bias is not None:
                nn.init.zeros_(self.refine[-1].bias)

    def forward(self, feat):
        projected = self.skip(feat.float()) + self.refine(feat.float())
        if self.l2_normalize:
            projected = F.normalize(projected, dim=1)
        return projected


def rotation_6d_to_matrix(rot_6d):
    """Convert a 6D rotation representation to a valid rotation matrix."""
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(rotmat):
    return rotmat[..., :, 0:2].transpose(-1, -2).reshape(*rotmat.shape[:-2], 6)


class QueryChannelGate(nn.Module):
    """Query-side channel selector for localization-aware feature weighting."""

    def __init__(self, token_dim, fine_dim, coarse_dim, hidden_dim=None, zero_init=True):
        super().__init__()
        hidden_dim = int(hidden_dim or token_dim)
        self.fine_dim = int(fine_dim)
        self.coarse_dim = int(coarse_dim)
        self.net = nn.Sequential(
            nn.Linear(int(token_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.fine_dim + self.coarse_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, token):
        logits = self.net(token.float())
        fine_logits, coarse_logits = torch.split(logits, [self.fine_dim, self.coarse_dim], dim=-1)
        return {
            "fine": torch.sigmoid(fine_logits),
            "coarse": torch.sigmoid(coarse_logits),
            "fine_logits": fine_logits,
            "coarse_logits": coarse_logits,
        }


class AbsolutePoseInitHead(nn.Module):
    """Predict K absolute pose hypotheses from the query global token."""

    def __init__(self, token_dim, hidden_dim=None, hypotheses=3):
        super().__init__()
        self.hypotheses = int(hypotheses)
        if self.hypotheses <= 0:
            raise ValueError("pose_init_hypotheses must be positive")
        hidden_dim = int(hidden_dim or token_dim)
        self.net = nn.Sequential(
            nn.Linear(int(token_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.hypotheses * 12),
        )

    @staticmethod
    def _centers_rot_to_w2c(centers, rotmat):
        t = -(rotmat @ centers.unsqueeze(-1)).squeeze(-1)
        eye = torch.eye(4, device=centers.device, dtype=centers.dtype)
        pose = eye.view(1, 1, 4, 4).repeat(centers.shape[0], centers.shape[1], 1, 1)
        pose[:, :, :3, :3] = rotmat
        pose[:, :, :3, 3] = t
        return pose

    def forward(self, token):
        raw = self.net(token.float()).view(token.shape[0], self.hypotheses, 12)
        center = raw[..., 0:3]
        rot6d = raw[..., 3:9]
        scores = raw[..., 9]
        log_var = raw[..., 10:12].clamp(min=-8.0, max=8.0)
        rotmat = rotation_6d_to_matrix(rot6d)
        return {
            "center": center,
            "rot6d": rot6d,
            "rotmat": rotmat,
            "pose_w2c": self._centers_rot_to_w2c(center, rotmat),
            "scores": scores,
            "log_var": log_var,
        }


class AnchorPoseInitHead(nn.Module):
    """Predict anchor-ranked pose hypotheses with residual camera-center offsets."""

    def __init__(
        self,
        token_dim,
        anchor_centers,
        anchor_rotmats=None,
        hidden_dim=None,
        hypotheses=3,
        residual_scale=1.0,
    ):
        super().__init__()
        anchors = torch.as_tensor(anchor_centers, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError("pose_init_anchor_centers must have shape (num_anchors, 3)")
        if anchors.shape[0] <= 0:
            raise ValueError("pose_init_anchor_centers must contain at least one anchor")
        if anchor_rotmats is None:
            rotmats = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(anchors.shape[0], 1, 1)
        else:
            rotmats = torch.as_tensor(anchor_rotmats, dtype=torch.float32)
            if rotmats.shape != (anchors.shape[0], 3, 3):
                raise ValueError("pose_init_anchor_rotmats must have shape (num_anchors, 3, 3)")
        self.register_buffer("anchor_centers", anchors)
        self.register_buffer("anchor_rotmats", rotmats)
        self.register_buffer("anchor_rot6d", matrix_to_rotation_6d(rotmats))
        self.num_anchors = int(anchors.shape[0])
        self.hypotheses = min(int(hypotheses), self.num_anchors)
        if self.hypotheses <= 0:
            raise ValueError("pose_init_hypotheses must be positive")
        self.residual_scale = float(residual_scale)
        hidden_dim = int(hidden_dim or token_dim)
        self.trunk = nn.Sequential(
            nn.Linear(int(token_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.anchor_logits = nn.Linear(hidden_dim, self.num_anchors)
        self.pose_raw = nn.Linear(hidden_dim, self.num_anchors * 11)
        nn.init.zeros_(self.pose_raw.weight)
        nn.init.zeros_(self.pose_raw.bias)

    def forward(self, token):
        hidden = self.trunk(token.float())
        anchor_logits = self.anchor_logits(hidden)
        pose_raw = self.pose_raw(hidden).view(token.shape[0], self.num_anchors, 11)
        top_scores, top_idx = torch.topk(anchor_logits, k=self.hypotheses, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, pose_raw.shape[-1])
        selected_raw = pose_raw.gather(1, gather_idx)
        selected_anchor = self.anchor_centers.to(device=token.device, dtype=selected_raw.dtype)[top_idx]
        selected_rot6d = self.anchor_rot6d.to(device=token.device, dtype=selected_raw.dtype)[top_idx]
        residual = torch.tanh(selected_raw[..., 0:3]) * self.residual_scale
        center = selected_anchor + residual
        rot6d = selected_rot6d + selected_raw[..., 3:9]
        log_var = selected_raw[..., 9:11].clamp(min=-8.0, max=8.0)
        rotmat = rotation_6d_to_matrix(rot6d)
        return {
            "center": center,
            "rot6d": rot6d,
            "rotmat": rotmat,
            "pose_w2c": AbsolutePoseInitHead._centers_rot_to_w2c(center, rotmat),
            "scores": top_scores,
            "log_var": log_var,
            "anchor_logits": anchor_logits,
            "anchor_indices": top_idx,
            "anchor_centers": self.anchor_centers,
            "anchor_rotmats": self.anchor_rotmats,
            "anchor_residual": residual,
        }


class FeatureBankPoseInitHead(AnchorPoseInitHead):
    """Pose bank initializer driven by query-map/teacher feature similarity."""

    def __init__(
        self,
        token_dim,
        anchor_centers,
        anchor_descriptors,
        fine_dim,
        coarse_dim,
        anchor_rotmats=None,
        hidden_dim=None,
        hypotheses=10,
        residual_scale=0.5,
        temperature=0.05,
        projector="mlp",
        feature_source="fine_coarse",
    ):
        super().__init__(
            token_dim=token_dim,
            anchor_centers=anchor_centers,
            anchor_rotmats=anchor_rotmats,
            hidden_dim=hidden_dim,
            hypotheses=hypotheses,
            residual_scale=residual_scale,
        )
        descriptors = torch.as_tensor(anchor_descriptors, dtype=torch.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != self.num_anchors:
            raise ValueError("pose_init_anchor_descriptors must have shape (num_anchors, descriptor_dim)")
        self.register_buffer("anchor_descriptors", F.normalize(descriptors, dim=1))
        self.descriptor_dim = int(descriptors.shape[1])
        self.fine_dim = int(fine_dim)
        self.coarse_dim = int(coarse_dim)
        self.feature_source = str(feature_source or "fine_coarse").lower()
        if self.feature_source not in {"fine_coarse", "fine", "coarse"}:
            raise ValueError("FeatureBankPoseInitHead feature_source must be one of {'fine_coarse', 'fine', 'coarse'}")
        if self.feature_source == "fine":
            query_input_dim = self.fine_dim
        elif self.feature_source == "coarse":
            query_input_dim = self.coarse_dim
        else:
            query_input_dim = self.fine_dim + self.coarse_dim
        self.temperature = max(float(temperature), 1e-6)
        self.projector = str(projector or "mlp").lower()
        if self.projector == "identity":
            if self.descriptor_dim != query_input_dim:
                raise ValueError(
                    "FeatureBankPoseInitHead projector='identity' requires descriptor_dim to match feature_source"
                )
            self.query_descriptor = nn.Identity()
        elif self.projector == "mlp":
            self.query_descriptor = nn.Sequential(
                nn.Linear(query_input_dim, int(hidden_dim or token_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim or token_dim), self.descriptor_dim),
            )
        else:
            raise ValueError("FeatureBankPoseInitHead projector must be either 'identity' or 'mlp'")

    def _pool_query_features(self, fine, coarse):
        if self.feature_source in {"fine", "fine_coarse"}:
            if fine is None:
                raise ValueError("FeatureBankPoseInitHead.forward requires fine feature maps")
            fine_vec = fine.float().mean(dim=(-1, -2))
        else:
            fine_vec = None
        if self.feature_source in {"coarse", "fine_coarse"}:
            if coarse is None:
                raise ValueError("FeatureBankPoseInitHead.forward requires coarse feature maps")
            coarse_vec = coarse.float().mean(dim=(-1, -2))
        else:
            coarse_vec = None
        if self.feature_source == "fine":
            return fine_vec
        if self.feature_source == "coarse":
            return coarse_vec
        return torch.cat([fine_vec, coarse_vec], dim=1)

    def forward(self, token, *, fine=None, coarse=None):
        hidden = self.trunk(token.float())
        query_feat = self._pool_query_features(fine, coarse)
        query_desc = F.normalize(self.query_descriptor(query_feat), dim=1)
        anchor_logits = torch.matmul(
            query_desc,
            self.anchor_descriptors.to(device=query_desc.device, dtype=query_desc.dtype).t(),
        ) / self.temperature
        pose_raw = self.pose_raw(hidden).view(token.shape[0], self.num_anchors, 11)
        top_scores, top_idx = torch.topk(anchor_logits, k=self.hypotheses, dim=1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, pose_raw.shape[-1])
        selected_raw = pose_raw.gather(1, gather_idx)
        selected_anchor = self.anchor_centers.to(device=token.device, dtype=selected_raw.dtype)[top_idx]
        selected_rot6d = self.anchor_rot6d.to(device=token.device, dtype=selected_raw.dtype)[top_idx]
        residual = torch.tanh(selected_raw[..., 0:3]) * self.residual_scale
        center = selected_anchor + residual
        rot6d = selected_rot6d + selected_raw[..., 3:9]
        log_var = selected_raw[..., 9:11].clamp(min=-8.0, max=8.0)
        rotmat = rotation_6d_to_matrix(rot6d)
        return {
            "center": center,
            "rot6d": rot6d,
            "rotmat": rotmat,
            "pose_w2c": AbsolutePoseInitHead._centers_rot_to_w2c(center, rotmat),
            "scores": top_scores,
            "log_var": log_var,
            "anchor_logits": anchor_logits,
            "anchor_indices": top_idx,
            "anchor_centers": self.anchor_centers,
            "anchor_rotmats": self.anchor_rotmats,
            "anchor_residual": residual,
            "query_descriptor": query_desc,
        }


class RadioQueryStudent(nn.Module):
    """Minimal dual-head RGB encoder for RADIO-style query features."""

    def __init__(
        self,
        in_channels=3,
        feature_dim=64,
        fine_feature_dim=None,
        coarse_feature_dim=None,
        base_channels=32,
        stage_dims=(32, 64, 96, 128),
        output_hw=(68, 120),
        coarse_output_hw=None,
        input_hw=(1088, 1920),
        dropout=0.0,
        l2_normalize=True,
        predict_magnitude=False,
        fine_init_norm=1.0,
        coarse_init_norm=1.0,
        magnitude_min=1e-4,
        retrieval_dim=None,
        retrieval_hidden_dim=None,
        retrieval_dropout=0.0,
        retrieval_l2_normalize=True,
        fine_low_level_skip=False,
        fine_low_level_init=0.0,
        fine_highres_skip=False,
        fine_highres_source="stage2",
        fine_highres_init=0.0,
        fine_highres_zero_init=False,
        fine_loc_head=False,
        fine_loc_mode="residual",
        fine_loc_init=1.0,
        fine_loc_zero_init=True,
        fine_loc_detach_base=False,
        fine_loc_highres_source=None,
        fine_loc_highres_init=1.0,
        fine_loc_highres_zero_init=True,
        fine_loc_highres_detach=True,
        teacher_fine_condition=False,
        teacher_fine_init=1.0,
        teacher_fine_zero_init=True,
        teacher_fine_detach=True,
        scene_coord_head=False,
        scene_coord_zero_init=True,
        scene_coord_detach_base=False,
        scene_coord_use_pixel_grid=False,
        scene_coord_global_context=False,
        local_matcher_enabled=False,
        local_matcher_radius=4,
        local_matcher_hidden_dim=64,
        local_matcher_zero_init=True,
        local_matcher_residual_scale=1.0,
        local_flow_head_enabled=False,
        local_flow_head_radius=4,
        local_flow_head_hidden_dim=64,
        local_flow_head_zero_init=True,
        local_flow_head_max_flow=None,
        local_flow_head_base_flow_mode="none",
        local_flow_head_base_temperature=0.05,
        local_corr_projector_enabled=False,
        local_corr_projector_hidden_dim=96,
        local_corr_projector_output_dim=None,
        local_corr_projector_zero_init=True,
        local_corr_projector_l2_normalize=True,
        query_channel_gate_enabled=False,
        query_channel_gate_hidden_dim=None,
        query_channel_gate_zero_init=True,
        apply_query_channel_gate=False,
        pose_init_head=False,
        pose_init_hypotheses=3,
        pose_init_hidden_dim=None,
        pose_init_mode="direct",
        pose_init_anchor_centers=None,
        pose_init_anchor_rotmats=None,
        pose_init_anchor_descriptors=None,
        pose_init_residual_scale=1.0,
        pose_init_temperature=0.05,
        pose_init_feature_bank_projector="mlp",
        pose_init_feature_source="fine_coarse",
    ):
        super().__init__()
        stage_dims = tuple(stage_dims)
        if len(stage_dims) != 4:
            raise ValueError("stage_dims must have four entries")

        self.output_hw = tuple(output_hw) if output_hw is not None else None
        self.coarse_output_hw = tuple(coarse_output_hw) if coarse_output_hw is not None else self.output_hw
        self.input_hw = tuple(input_hw) if input_hw is not None else None
        self.feature_dim = int(feature_dim)
        self.fine_feature_dim = int(fine_feature_dim) if fine_feature_dim is not None else self.feature_dim
        self.coarse_feature_dim = int(coarse_feature_dim) if coarse_feature_dim is not None else self.feature_dim
        self.l2_normalize = l2_normalize
        self.predict_magnitude = bool(predict_magnitude)
        self.fine_init_norm = float(fine_init_norm)
        self.coarse_init_norm = float(coarse_init_norm)
        self.magnitude_min = float(magnitude_min)
        self.retrieval_dim = int(retrieval_dim) if retrieval_dim else 0
        self.retrieval_l2_normalize = retrieval_l2_normalize
        self.fine_low_level_skip = bool(fine_low_level_skip)
        self.fine_low_level_init = float(fine_low_level_init)
        self.fine_highres_skip = bool(fine_highres_skip)
        self.fine_highres_source = str(fine_highres_source).lower()
        self.fine_highres_init = float(fine_highres_init)
        self.fine_highres_zero_init = bool(fine_highres_zero_init)
        self.fine_loc_highres_source = (
            str(fine_loc_highres_source).lower()
            if fine_loc_highres_source is not None
            else None
        )
        self.fine_loc_mode = str(fine_loc_mode or "residual").lower()
        if self.fine_loc_mode not in {"residual", "direct"}:
            raise ValueError("fine_loc_mode must be either 'residual' or 'direct'")
        self.use_fine_loc_head = bool(fine_loc_head or self.fine_loc_highres_source is not None)
        self.fine_loc_init = float(fine_loc_init)
        self.fine_loc_zero_init = bool(fine_loc_zero_init)
        self.fine_loc_detach_base = bool(fine_loc_detach_base)
        self.fine_loc_highres_init = float(fine_loc_highres_init)
        self.fine_loc_highres_zero_init = bool(fine_loc_highres_zero_init)
        self.fine_loc_highres_detach = bool(fine_loc_highres_detach)
        self.teacher_fine_condition = bool(teacher_fine_condition)
        self.teacher_fine_init = float(teacher_fine_init)
        self.teacher_fine_zero_init = bool(teacher_fine_zero_init)
        self.teacher_fine_detach = bool(teacher_fine_detach)
        self.use_scene_coord_head = bool(scene_coord_head)
        self.scene_coord_zero_init = bool(scene_coord_zero_init)
        self.scene_coord_detach_base = bool(scene_coord_detach_base)
        self.scene_coord_use_pixel_grid = bool(scene_coord_use_pixel_grid)
        self.scene_coord_global_context = bool(scene_coord_global_context)
        self.local_matcher_enabled = bool(local_matcher_enabled)
        self.local_flow_head_enabled = bool(local_flow_head_enabled)
        self.local_corr_projector_enabled = bool(local_corr_projector_enabled)
        self.query_channel_gate_enabled = bool(query_channel_gate_enabled)
        self.query_channel_gate_zero_init = bool(query_channel_gate_zero_init)
        self.apply_query_channel_gate = bool(apply_query_channel_gate)
        self.pose_init_head_enabled = bool(pose_init_head)
        self.pose_init_mode = str(pose_init_mode or "direct").lower()
        if self.pose_init_mode not in {"direct", "anchor", "feature_bank"}:
            raise ValueError("pose_init_mode must be one of {'direct', 'anchor', 'feature_bank'}")
        self.pose_init_feature_source = str(pose_init_feature_source or "fine_coarse").lower()
        if self.pose_init_feature_source not in {"fine_coarse", "fine", "coarse"}:
            raise ValueError("pose_init_feature_source must be one of {'fine_coarse', 'fine', 'coarse'}")

        stem_dim = stage_dims[0] or base_channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_dim, kernel_size=5, stride=2),
            ResidualDepthwiseBlock(stem_dim),
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(stem_dim, stage_dims[1], stride=2),
            ResidualDepthwiseBlock(stage_dims[1]),
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(stage_dims[1], stage_dims[2], stride=2),
            ResidualDepthwiseBlock(stage_dims[2]),
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(stage_dims[2], stage_dims[3], stride=2),
            ResidualDepthwiseBlock(stage_dims[3]),
            ResidualDepthwiseBlock(stage_dims[3]),
        )

        self.fine_fuse = nn.Sequential(
            ConvNormAct(stage_dims[2] + stage_dims[3], stage_dims[3], kernel_size=1),
            ResidualDepthwiseBlock(stage_dims[3]),
        )
        self.fine_low_fuse = (
            nn.Sequential(
                ConvNormAct(stage_dims[1], stage_dims[3], kernel_size=1),
                ResidualDepthwiseBlock(stage_dims[3]),
            )
            if self.fine_low_level_skip
            else None
        )
        self.fine_low_scale = (
            nn.Parameter(torch.tensor(self.fine_low_level_init, dtype=torch.float32))
            if self.fine_low_level_skip
            else None
        )
        highres_channels = {
            "stage1": stage_dims[0],
            "stem": stage_dims[0],
            "stage2": stage_dims[1],
            "stage3": stage_dims[2],
        }
        if self.fine_highres_skip and self.fine_highres_source not in highres_channels:
            raise ValueError(
                f"fine_highres_source must be one of {sorted(highres_channels)}, "
                f"got {self.fine_highres_source!r}"
            )
        if self.fine_loc_highres_source is not None and self.fine_loc_highres_source not in highres_channels:
            raise ValueError(
                f"fine_loc_highres_source must be one of {sorted(highres_channels)}, "
                f"got {self.fine_loc_highres_source!r}"
            )
        if self.fine_highres_skip:
            highres_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            self.fine_highres_fuse = nn.Sequential(
                ConvNormAct(highres_channels[self.fine_highres_source], highres_hidden, kernel_size=3),
                ResidualDepthwiseBlock(highres_hidden),
                nn.Conv2d(highres_hidden, self.fine_feature_dim, kernel_size=1),
            )
            self.fine_highres_scale = nn.Parameter(
                torch.tensor(self.fine_highres_init, dtype=torch.float32)
            )
        else:
            self.fine_highres_fuse = None
            self.fine_highres_scale = None
        if self.teacher_fine_condition:
            teacher_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            self.teacher_fine_fuse = nn.Sequential(
                ConvNormAct(self.fine_feature_dim, teacher_hidden, kernel_size=3),
                ResidualDepthwiseBlock(teacher_hidden),
                nn.Conv2d(teacher_hidden, self.fine_feature_dim, kernel_size=1),
            )
            self.teacher_fine_scale = nn.Parameter(
                torch.tensor(self.teacher_fine_init, dtype=torch.float32)
            )
        else:
            self.teacher_fine_fuse = None
            self.teacher_fine_scale = None
        if self.use_scene_coord_head:
            scene_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            scene_in_channels = self.fine_feature_dim
            if self.scene_coord_use_pixel_grid:
                scene_in_channels += 2
            if self.scene_coord_global_context:
                scene_in_channels += scene_hidden
                self.scene_context_proj = nn.Conv2d(stage_dims[3], scene_hidden, kernel_size=1)
            else:
                self.scene_context_proj = None
            self.scene_coord_head = nn.Sequential(
                ConvNormAct(scene_in_channels, scene_hidden, kernel_size=3),
                ResidualDepthwiseBlock(scene_hidden),
                nn.Conv2d(scene_hidden, 3, kernel_size=1),
            )
        else:
            self.scene_context_proj = None
            self.scene_coord_head = None
        self.coarse_refine = nn.Sequential(
            ResidualDepthwiseBlock(stage_dims[3]),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.fine_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], self.fine_feature_dim, kernel_size=1),
        )
        self.coarse_head = nn.Sequential(
            ConvNormAct(stage_dims[3], stage_dims[2], kernel_size=3),
            nn.Conv2d(stage_dims[2], self.coarse_feature_dim, kernel_size=1),
        )
        if self.use_fine_loc_head:
            loc_hidden = max(16, min(stage_dims[2], self.fine_feature_dim))
            if fine_loc_head:
                self.fine_loc_head = nn.Sequential(
                    ConvNormAct(self.fine_feature_dim, loc_hidden, kernel_size=3),
                    ResidualDepthwiseBlock(loc_hidden),
                    nn.Conv2d(loc_hidden, self.fine_feature_dim, kernel_size=1),
                )
            else:
                self.fine_loc_head = None
            if self.fine_loc_highres_source is not None:
                self.fine_loc_highres_fuse = nn.Sequential(
                    ConvNormAct(highres_channels[self.fine_loc_highres_source], loc_hidden, kernel_size=3),
                    ResidualDepthwiseBlock(loc_hidden),
                    nn.Conv2d(loc_hidden, self.fine_feature_dim, kernel_size=1),
                )
                self.fine_loc_highres_scale = nn.Parameter(
                    torch.tensor(self.fine_loc_highres_init, dtype=torch.float32)
                )
            else:
                self.fine_loc_highres_fuse = None
                self.fine_loc_highres_scale = None
            self.fine_loc_scale = nn.Parameter(
                torch.tensor(self.fine_loc_init, dtype=torch.float32)
            )
        else:
            self.fine_loc_head = None
            self.fine_loc_scale = None
            self.fine_loc_highres_fuse = None
            self.fine_loc_highres_scale = None
        if self.predict_magnitude:
            self.fine_norm_head = nn.Conv2d(stage_dims[3], 1, kernel_size=1)
            self.coarse_norm_head = nn.Conv2d(stage_dims[3], 1, kernel_size=1)
        else:
            self.fine_norm_head = None
            self.coarse_norm_head = None

        if self.retrieval_dim > 0:
            retrieval_hidden_dim = int(retrieval_hidden_dim or stage_dims[3])
            self.retrieval_head = nn.Sequential(
                nn.Linear(stage_dims[3], retrieval_hidden_dim),
                nn.GELU(),
                nn.Dropout(retrieval_dropout) if retrieval_dropout > 0 else nn.Identity(),
                nn.Linear(retrieval_hidden_dim, self.retrieval_dim),
            )
        else:
            self.retrieval_head = None

        self.query_channel_gate = (
            QueryChannelGate(
                token_dim=stage_dims[3],
                fine_dim=self.fine_feature_dim,
                coarse_dim=self.coarse_feature_dim,
                hidden_dim=query_channel_gate_hidden_dim,
                zero_init=bool(query_channel_gate_zero_init),
            )
            if self.query_channel_gate_enabled
            else None
        )
        if self.pose_init_head_enabled and self.pose_init_mode == "feature_bank":
            self.pose_init_head = FeatureBankPoseInitHead(
                token_dim=stage_dims[3],
                hidden_dim=pose_init_hidden_dim,
                hypotheses=int(pose_init_hypotheses),
                anchor_centers=pose_init_anchor_centers,
                anchor_rotmats=pose_init_anchor_rotmats,
                anchor_descriptors=pose_init_anchor_descriptors,
                fine_dim=self.fine_feature_dim,
                coarse_dim=self.coarse_feature_dim,
                residual_scale=float(pose_init_residual_scale),
                temperature=float(pose_init_temperature),
                projector=str(pose_init_feature_bank_projector),
                feature_source=self.pose_init_feature_source,
            )
        elif self.pose_init_head_enabled and self.pose_init_mode == "anchor":
            self.pose_init_head = AnchorPoseInitHead(
                token_dim=stage_dims[3],
                hidden_dim=pose_init_hidden_dim,
                hypotheses=int(pose_init_hypotheses),
                anchor_centers=pose_init_anchor_centers,
                anchor_rotmats=pose_init_anchor_rotmats,
                residual_scale=float(pose_init_residual_scale),
            )
        else:
            self.pose_init_head = (
                AbsolutePoseInitHead(
                    token_dim=stage_dims[3],
                    hidden_dim=pose_init_hidden_dim,
                    hypotheses=int(pose_init_hypotheses),
                )
                if self.pose_init_head_enabled
                else None
            )

        self.local_matcher = (
            DepthAwareLocalMatcher(
                radius=int(local_matcher_radius),
                hidden_dim=int(local_matcher_hidden_dim),
                zero_init=bool(local_matcher_zero_init),
                residual_scale=float(local_matcher_residual_scale),
            )
            if self.local_matcher_enabled
            else None
        )
        self.local_flow_head = (
            DepthAwareLocalFlowHead(
                radius=int(local_flow_head_radius),
                hidden_dim=int(local_flow_head_hidden_dim),
                zero_init=bool(local_flow_head_zero_init),
                max_flow=local_flow_head_max_flow,
                base_flow_mode=local_flow_head_base_flow_mode,
                base_temperature=float(local_flow_head_base_temperature),
            )
            if self.local_flow_head_enabled
            else None
        )
        self.local_corr_projector = (
            LocalCorrProjector(
                feature_dim=self.fine_feature_dim,
                hidden_dim=int(local_corr_projector_hidden_dim),
                output_dim=local_corr_projector_output_dim,
                zero_init=bool(local_corr_projector_zero_init),
                l2_normalize=bool(local_corr_projector_l2_normalize),
            )
            if self.local_corr_projector_enabled
            else None
        )

        self._init_weights()
        if self.local_matcher is not None and bool(local_matcher_zero_init):
            nn.init.zeros_(self.local_matcher.refine[-1].weight)
            if self.local_matcher.refine[-1].bias is not None:
                nn.init.zeros_(self.local_matcher.refine[-1].bias)
        if self.local_flow_head is not None and bool(local_flow_head_zero_init):
            nn.init.zeros_(self.local_flow_head.predict[-1].weight)
            if self.local_flow_head.predict[-1].bias is not None:
                nn.init.zeros_(self.local_flow_head.predict[-1].bias)
        if self.local_corr_projector is not None and bool(local_corr_projector_zero_init):
            nn.init.zeros_(self.local_corr_projector.refine[-1].weight)
            if self.local_corr_projector.refine[-1].bias is not None:
                nn.init.zeros_(self.local_corr_projector.refine[-1].bias)
        if self.query_channel_gate is not None and self.query_channel_gate_zero_init:
            nn.init.zeros_(self.query_channel_gate.net[-1].weight)
            nn.init.zeros_(self.query_channel_gate.net[-1].bias)
        if self.fine_highres_zero_init and self.fine_highres_fuse is not None:
            nn.init.zeros_(self.fine_highres_fuse[-1].weight)
            if self.fine_highres_fuse[-1].bias is not None:
                nn.init.zeros_(self.fine_highres_fuse[-1].bias)
        if self.fine_loc_zero_init and self.fine_loc_head is not None:
            nn.init.zeros_(self.fine_loc_head[-1].weight)
            if self.fine_loc_head[-1].bias is not None:
                nn.init.zeros_(self.fine_loc_head[-1].bias)
        if self.fine_loc_highres_zero_init and self.fine_loc_highres_fuse is not None:
            nn.init.zeros_(self.fine_loc_highres_fuse[-1].weight)
            if self.fine_loc_highres_fuse[-1].bias is not None:
                nn.init.zeros_(self.fine_loc_highres_fuse[-1].bias)
        if self.teacher_fine_zero_init and self.teacher_fine_fuse is not None:
            nn.init.zeros_(self.teacher_fine_fuse[-1].weight)
            if self.teacher_fine_fuse[-1].bias is not None:
                nn.init.zeros_(self.teacher_fine_fuse[-1].bias)
        if self.scene_coord_zero_init and self.scene_coord_head is not None:
            nn.init.zeros_(self.scene_coord_head[-1].weight)
            if self.scene_coord_head[-1].bias is not None:
                nn.init.zeros_(self.scene_coord_head[-1].bias)
        if self.predict_magnitude:
            self._init_magnitude_head(self.fine_norm_head, self.fine_init_norm)
            self._init_magnitude_head(self.coarse_norm_head, self.coarse_init_norm)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_magnitude_head(self, head, init_norm):
        nn.init.zeros_(head.weight)
        init_mag = max(float(init_norm) - self.magnitude_min, 1e-4)
        nn.init.constant_(head.bias, _softplus_inverse(init_mag))

    def forward(self, x, teacher_fine=None):
        if self.input_hw is not None and tuple(x.shape[-2:]) != self.input_hw:
            x = F.interpolate(x, self.input_hw, mode="bilinear", align_corners=False)

        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        s3_to_s4 = F.avg_pool2d(s3, kernel_size=2, stride=2)
        fine_latent = self.fine_fuse(torch.cat([s4, s3_to_s4], dim=1))
        if self.fine_low_fuse is not None:
            s2_to_s4 = F.adaptive_avg_pool2d(s2, output_size=s4.shape[-2:])
            fine_latent = fine_latent + self.fine_low_scale * self.fine_low_fuse(s2_to_s4)
        coarse_latent = self.coarse_refine(s4)
        global_pose_token = coarse_latent.mean(dim=(-1, -2))

        fine = self.fine_head(fine_latent)
        coarse = self.coarse_head(coarse_latent)
        query_channel_weights = None
        if self.query_channel_gate is not None:
            query_channel_weights = self.query_channel_gate(global_pose_token)
            if self.apply_query_channel_gate:
                fine = fine * query_channel_weights["fine"].to(dtype=fine.dtype).unsqueeze(-1).unsqueeze(-1)
                coarse = coarse * query_channel_weights["coarse"].to(dtype=coarse.dtype).unsqueeze(-1).unsqueeze(-1)
        fine_mag = None
        coarse_mag = None
        if self.predict_magnitude:
            fine_mag = F.softplus(self.fine_norm_head(fine_latent)) + self.magnitude_min
            coarse_mag = F.softplus(self.coarse_norm_head(coarse_latent)) + self.magnitude_min

        if self.output_hw is not None:
            fine = F.interpolate(fine, self.output_hw, mode="bilinear", align_corners=False)
            if fine_mag is not None:
                fine_mag = F.interpolate(fine_mag, self.output_hw, mode="bilinear", align_corners=False)
        if self.fine_highres_fuse is not None:
            highres_sources = {
                "stage1": s1,
                "stem": s1,
                "stage2": s2,
                "stage3": s3,
            }
            fine_skip = self.fine_highres_fuse(highres_sources[self.fine_highres_source])
            if fine_skip.shape[-2:] != fine.shape[-2:]:
                fine_skip = F.interpolate(fine_skip, fine.shape[-2:], mode="bilinear", align_corners=False)
            fine = fine + self.fine_highres_scale * fine_skip
        if self.teacher_fine_fuse is not None and teacher_fine is not None:
            teacher = teacher_fine.float()
            if self.teacher_fine_detach:
                teacher = teacher.detach()
            if teacher.shape[-2:] != fine.shape[-2:]:
                teacher = F.interpolate(teacher, fine.shape[-2:], mode="bilinear", align_corners=False)
            fine = fine + self.teacher_fine_scale * self.teacher_fine_fuse(teacher)
        scene_coord = None
        if self.scene_coord_head is not None:
            scene_input = fine.detach() if self.scene_coord_detach_base else fine
            scene_inputs = [scene_input]
            if self.scene_coord_use_pixel_grid:
                B, _C, H, W = scene_input.shape
                yy, xx = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, H, device=scene_input.device, dtype=scene_input.dtype),
                    torch.linspace(-1.0, 1.0, W, device=scene_input.device, dtype=scene_input.dtype),
                    indexing="ij",
                )
                grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
                scene_inputs.append(grid)
            if self.scene_context_proj is not None:
                context = F.adaptive_avg_pool2d(s4, output_size=1)
                context = self.scene_context_proj(context)
                context = context.expand(-1, -1, scene_input.shape[-2], scene_input.shape[-1])
                scene_inputs.append(context)
            if len(scene_inputs) > 1:
                scene_input = torch.cat(scene_inputs, dim=1)
            scene_coord = self.scene_coord_head(scene_input)
        if self.coarse_output_hw is not None:
            coarse = F.interpolate(coarse, self.coarse_output_hw, mode="bilinear", align_corners=False)
            if coarse_mag is not None:
                coarse_mag = F.interpolate(coarse_mag, self.coarse_output_hw, mode="bilinear", align_corners=False)

        if self.predict_magnitude:
            fine = F.normalize(fine, dim=1) * fine_mag
            coarse = F.normalize(coarse, dim=1) * coarse_mag
        elif self.l2_normalize:
            fine = F.normalize(fine, dim=1)
            coarse = F.normalize(coarse, dim=1)

        fine_loc = None
        if self.use_fine_loc_head:
            fine_loc_base = fine.detach() if self.fine_loc_detach_base else fine
            loc_delta = torch.zeros_like(fine_loc_base)
            if self.fine_loc_head is not None:
                loc_delta = loc_delta + self.fine_loc_scale * self.fine_loc_head(fine_loc_base)
            if self.fine_loc_highres_fuse is not None:
                highres_sources = {
                    "stage1": s1,
                    "stem": s1,
                    "stage2": s2,
                    "stage3": s3,
                }
                loc_src = highres_sources[self.fine_loc_highres_source]
                if self.fine_loc_highres_detach:
                    loc_src = loc_src.detach()
                loc_skip = self.fine_loc_highres_fuse(loc_src)
                if loc_skip.shape[-2:] != fine_loc_base.shape[-2:]:
                    loc_skip = F.interpolate(
                        loc_skip,
                        fine_loc_base.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loc_delta = loc_delta + self.fine_loc_highres_scale * loc_skip
            fine_loc = loc_delta if self.fine_loc_mode == "direct" else fine_loc_base + loc_delta
            if self.predict_magnitude:
                fine_loc = F.normalize(fine_loc, dim=1) * fine_mag
            elif self.l2_normalize and not (
                self.fine_loc_mode == "direct" and loc_delta.abs().amax().detach() <= 0
            ):
                fine_loc = F.normalize(fine_loc, dim=1)

        outputs = {
            "fine": fine,
            "coarse": coarse,
            "backbone_features": {
                "stage2": s2,
                "stage3": s3,
                "stage4": s4,
            },
            "global_pose_token": global_pose_token,
        }
        if query_channel_weights is not None:
            outputs["query_channel_weights"] = query_channel_weights
        if self.pose_init_head is not None:
            if isinstance(self.pose_init_head, FeatureBankPoseInitHead):
                outputs["pose_init"] = self.pose_init_head(global_pose_token, fine=fine, coarse=coarse)
            else:
                outputs["pose_init"] = self.pose_init_head(global_pose_token)
        if fine_loc is not None:
            outputs["fine_loc"] = fine_loc
        if scene_coord is not None:
            outputs["scene_coord"] = scene_coord
        if self.predict_magnitude:
            outputs["magnitude"] = {
                "fine": fine_mag,
                "coarse": coarse_mag,
            }

        if self.retrieval_head is not None:
            retrieval = self.retrieval_head(global_pose_token)
            if self.retrieval_l2_normalize:
                retrieval = F.normalize(retrieval, dim=1)
            outputs["retrieval"] = retrieval

        return outputs
