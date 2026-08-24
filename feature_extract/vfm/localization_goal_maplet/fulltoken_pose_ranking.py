"""Compact full-layout evidence for candidate-conditioned pose ranking.

The representation keeps every RADIO token and both adjacent-token phase
directions.  It is deliberately computed after a candidate pose is rendered:
there are no keypoints, correspondences, PnP, or absolute-pose inputs.

Channels are split into signed empirical compatibility, rendered support, and
non-negative above-floor evidence.  Only the last group has the conservative
missing-evidence interpretation; a learned ranker that consumes signed
channels is an empirical ranking model and must not inherit that theorem.
"""

from __future__ import annotations

import numpy as np
import torch

from .lineage import arrays_sha256


FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS = (
    "full_36x64_token_and_hv_phase_candidate_evidence_v1"
)
FULLTOKEN_POSE_RANKING_CHANNELS = (
    "token_signed_cosine",
    "token_rendered_mass",
    "token_above_floor_evidence",
    "horizontal_signed_phase_cosine",
    "horizontal_edge_mass",
    "horizontal_above_floor_evidence",
    "vertical_signed_phase_cosine",
    "vertical_edge_mass",
    "vertical_above_floor_evidence",
)
TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS = (
    "full_36x64_token_phase_and_rendered_typed_geometry_candidate_evidence_v1"
)
TYPED_FULLTOKEN_POSE_RANKING_CHANNELS = FULLTOKEN_POSE_RANKING_CHANNELS + (
    "rendered_normal_axis_xx",
    "rendered_normal_axis_yy",
    "rendered_normal_axis_zz",
    "rendered_normal_axis_xy",
    "rendered_normal_axis_xz",
    "rendered_normal_axis_yz",
    "rendered_relative_log_depth",
    "rendered_log_depth_std",
    "rendered_boundary_probability",
)
QUERY_TYPED_FULLTOKEN_POSE_RANKING_FEATURE_SEMANTICS = (
    "full_36x64_query_predicted_and_rendered_typed_geometry_consistency_v1"
)
QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS = TYPED_FULLTOKEN_POSE_RANKING_CHANNELS + (
    "query_predicted_geometry_confidence",
    "query_rendered_normal_axis_evidence",
    "query_rendered_relative_depth_evidence",
    "query_rendered_depth_dispersion_evidence",
    "query_rendered_boundary_evidence",
)
FULLTOKEN_POSE_RANKER_SEMANTICS = (
    "shared_candidate_conditioned_spatial_cnn_listwise_ranker_v1"
)
FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS = (
    "shared_fulltoken_encoder_separate_location_orientation_and_joint_ranker_v1"
)


def compact_fulltoken_pose_ranking_features(
    query_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    gradient_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Return nine aligned ``[C,H,W]`` candidate evidence channels."""

    query = torch.as_tensor(query_descriptor)
    target = torch.as_tensor(target_descriptor, device=query.device, dtype=query.dtype)
    mass = torch.as_tensor(target_mass, device=query.device, dtype=query.dtype)
    valid = torch.as_tensor(target_valid, device=query.device, dtype=torch.bool)
    token_count = int(height) * int(width)
    if query.ndim != 2 or query.shape[0] != token_count:
        raise ValueError("query descriptor differs from the declared full token grid")
    if target.ndim != 3 or target.shape[0] != token_count or target.shape[2] != query.shape[1]:
        raise ValueError("target descriptor differs from the query feature grid")
    if target.shape[1] != 1 or mass.shape != (token_count, 1) or valid.shape != mass.shape:
        raise ValueError("compact direct ranking requires exactly one target descriptor per token")
    if any(
        not torch.is_floating_point(value) or not torch.isfinite(value).all()
        for value in (query, target, mass)
    ):
        raise ValueError("full-token ranking inputs must be finite floating tensors")
    if torch.any(mass < 0.0) or torch.any(mass > 1.0 + 2.0e-5):
        raise ValueError("rendered token mass is invalid")
    threshold = float(gradient_threshold)
    if not 0.0 < threshold < float("inf"):
        raise ValueError("gradient threshold must be finite and positive")

    q_norm = torch.linalg.vector_norm(query, dim=1)
    t = target[:, 0]
    t_norm = torch.linalg.vector_norm(t, dim=1)
    present = valid[:, 0] & (mass[:, 0] > 0.0) & (t_norm >= 1.0e-8)
    q_unit = query / q_norm[:, None].clamp_min(1.0e-8)
    t_unit = t / t_norm[:, None].clamp_min(1.0e-8)
    cosine = torch.sum(q_unit * t_unit, dim=1).clamp(-1.0, 1.0)
    rendered_mass = torch.where(present, mass[:, 0], torch.zeros_like(mass[:, 0]))
    signed = torch.where(present, cosine, torch.zeros_like(cosine))
    evidence = rendered_mass * (cosine + 1.0) * 0.5

    q_grid = q_unit.reshape(int(height), int(width), -1)
    t_grid = t_unit.reshape(int(height), int(width), -1)
    m_grid = rendered_mass.reshape(int(height), int(width))
    p_grid = present.reshape(int(height), int(width))

    def phase(vertical: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vertical:
            q0, q1 = q_grid[:-1], q_grid[1:]
            t0, t1 = t_grid[:-1], t_grid[1:]
            m0, m1 = m_grid[:-1], m_grid[1:]
            p0, p1 = p_grid[:-1], p_grid[1:]
            shape = (int(height) - 1, int(width))
            pad = (0, 0, 0, 1)
        else:
            q0, q1 = q_grid[:, :-1], q_grid[:, 1:]
            t0, t1 = t_grid[:, :-1], t_grid[:, 1:]
            m0, m1 = m_grid[:, :-1], m_grid[:, 1:]
            p0, p1 = p_grid[:, :-1], p_grid[:, 1:]
            shape = (int(height), int(width) - 1)
            pad = (0, 1, 0, 0)
        q_delta, t_delta = q1 - q0, t1 - t0
        qn = torch.linalg.vector_norm(q_delta, dim=2)
        tn = torch.linalg.vector_norm(t_delta, dim=2)
        informative = p0 & p1 & (qn >= threshold) & (tn >= threshold)
        phase_cosine = torch.sum(q_delta * t_delta, dim=2) / (qn * tn).clamp_min(1.0e-8)
        phase_cosine = phase_cosine.clamp(-1.0, 1.0)
        edge_mass = torch.where(informative, torch.minimum(m0, m1), torch.zeros(shape, device=query.device, dtype=query.dtype))
        phase_signed = torch.where(informative, phase_cosine, torch.zeros_like(phase_cosine))
        phase_evidence = edge_mass * (phase_cosine + 1.0) * 0.5
        return tuple(torch.nn.functional.pad(value, pad) for value in (
            phase_signed, edge_mass, phase_evidence,
        ))

    horizontal = phase(False)
    vertical = phase(True)
    return torch.stack([
        signed.reshape(int(height), int(width)),
        rendered_mass.reshape(int(height), int(width)),
        evidence.reshape(int(height), int(width)),
        *horizontal,
        *vertical,
    ], dim=0)


def compact_typed_fulltoken_pose_ranking_features(
    query_descriptor: torch.Tensor,
    target_descriptor: torch.Tensor,
    target_mass: torch.Tensor,
    target_valid: torch.Tensor,
    target_normal_axis_moment: torch.Tensor,
    target_relative_log_depth: torch.Tensor,
    target_log_depth_std: torch.Tensor,
    target_boundary: torch.Tensor,
    *,
    height: int = 36,
    width: int = 64,
    gradient_threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Append renderer-typed geometry without collapsing the token layout."""

    base = compact_fulltoken_pose_ranking_features(
        query_descriptor, target_descriptor, target_mass, target_valid,
        height=height, width=width, gradient_threshold=gradient_threshold,
    )
    device, dtype = base.device, base.dtype
    axis = torch.as_tensor(target_normal_axis_moment, device=device, dtype=dtype)
    relative = torch.as_tensor(target_relative_log_depth, device=device, dtype=dtype)
    depth_std = torch.as_tensor(target_log_depth_std, device=device, dtype=dtype)
    boundary = torch.as_tensor(target_boundary, device=device, dtype=dtype)
    valid = torch.as_tensor(target_valid, device=device, dtype=torch.bool).reshape(
        int(height), int(width), -1,
    )[..., 0]
    if (
        axis.shape != (int(height), int(width), 6)
        or relative.shape != (int(height), int(width))
        or depth_std.shape != relative.shape or boundary.shape != relative.shape
        or any(not torch.isfinite(value).all() for value in (axis, relative, depth_std, boundary))
        or torch.any(depth_std < 0.0) or torch.any(boundary < 0.0) or torch.any(boundary > 1.0)
    ):
        raise ValueError("typed full-token geometry arrays differ")
    trace = axis[..., :3].sum(dim=2)
    if torch.any(valid & (torch.abs(trace - 1.0) > 2.0e-4)):
        raise ValueError("typed normal-axis moment does not have unit trace")
    typed = torch.cat((
        axis.permute(2, 0, 1),
        relative[None], depth_std[None], boundary[None],
    ), dim=0)
    typed = torch.where(valid[None], typed, torch.zeros_like(typed))
    return torch.cat((base, typed), dim=0)


class FullTokenCandidatePoseRanker(torch.nn.Module):
    """Small shared spatial ranker; it never receives candidate pose values."""

    def __init__(self, input_channels: int = len(FULLTOKEN_POSE_RANKING_CHANNELS)) -> None:
        super().__init__()
        if int(input_channels) != len(FULLTOKEN_POSE_RANKING_CHANNELS):
            raise ValueError("ranker input channels differ from the frozen feature contract")
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 4 * 2, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1:] != (
            len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
        ):
            raise ValueError("ranker features must have shape [N,9,36,64]")
        encoded = self.encoder(value)
        average = torch.nn.functional.adaptive_avg_pool2d(encoded, (3, 4))
        maximum = torch.nn.functional.adaptive_max_pool2d(encoded, (3, 4))
        return self.head(torch.cat([average, maximum], dim=1).flatten(1))[:, 0]


class FactorizedFullTokenCandidatePoseRanker(torch.nn.Module):
    """Shared evidence encoder with explicitly separated pose-error heads.

    The separation is important for hierarchical search: a joint SE(3) score
    can hide useful translation evidence behind a large rotation error (and
    vice versa).  All heads still consume only the same rendered full-token
    evidence; candidate pose values are never model inputs.
    """

    HEAD_NAMES = ("location", "orientation", "joint")

    def __init__(self, input_channels: int = len(FULLTOKEN_POSE_RANKING_CHANNELS)) -> None:
        super().__init__()
        if int(input_channels) not in (
            len(FULLTOKEN_POSE_RANKING_CHANNELS),
            len(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS),
            len(QUERY_TYPED_FULLTOKEN_POSE_RANKING_CHANNELS),
        ):
            raise ValueError("ranker input channels differ from a frozen feature contract")
        self.input_channels = int(input_channels)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.location_head = self._new_head()
        self.orientation_head = self._new_head()
        self.joint_head = self._new_head()

    @staticmethod
    def _new_head() -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 4 * 2, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1:] != (
            self.input_channels, 36, 64,
        ):
            raise ValueError(
                f"ranker features must have shape [N,{self.input_channels},36,64]"
            )
        encoded = self.encoder(value)
        average = torch.nn.functional.adaptive_avg_pool2d(encoded, (3, 4))
        maximum = torch.nn.functional.adaptive_max_pool2d(encoded, (3, 4))
        return torch.cat([average, maximum], dim=1).flatten(1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(features)
        return {
            "location": self.location_head(encoded)[:, 0],
            "orientation": self.orientation_head(encoded)[:, 0],
            "joint": self.joint_head(encoded)[:, 0],
        }

    def initialize_from_single_ranker(
        self, reference: FullTokenCandidatePoseRanker,
    ) -> None:
        """Copy a trained joint ranker into all heads without aliasing state."""

        if not isinstance(reference, FullTokenCandidatePoseRanker):
            raise TypeError("factorized initialization requires a single-head ranker")
        self.encoder.load_state_dict(reference.encoder.state_dict(), strict=True)
        for head in (self.location_head, self.orientation_head, self.joint_head):
            head.load_state_dict(reference.head.state_dict(), strict=True)

    def initialize_from_factorized_ranker(
        self, reference: "FactorizedFullTokenCandidatePoseRanker",
    ) -> None:
        """Warm-start extra typed channels at zero while preserving old scores."""

        if not isinstance(reference, FactorizedFullTokenCandidatePoseRanker):
            raise TypeError("typed initialization requires a factorized ranker")
        if self.input_channels < reference.input_channels:
            raise ValueError("cannot drop factorized reference feature channels")
        current = self.encoder.state_dict()
        source = reference.encoder.state_dict()
        for name, value in source.items():
            if name == "0.weight" and current[name].shape[1] != value.shape[1]:
                current[name].zero_()
                current[name][:, : value.shape[1]].copy_(value)
            else:
                current[name].copy_(value)
        self.encoder.load_state_dict(current, strict=True)
        for own, other in (
            (self.location_head, reference.location_head),
            (self.orientation_head, reference.orientation_head),
            (self.joint_head, reference.joint_head),
        ):
            own.load_state_dict(other.state_dict(), strict=True)


def load_fulltoken_candidate_pose_ranker(
    path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FullTokenCandidatePoseRanker, dict[str, object]]:
    """Load and content-verify a trained diagnostic ranker."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("artifact_type") != (
        "goal_maplet_fulltoken_candidate_pose_ranker_v1"
    ):
        raise ValueError("not a full-token candidate pose ranker")
    if payload.get("model_semantics") != FULLTOKEN_POSE_RANKER_SEMANTICS:
        raise ValueError("full-token candidate pose ranker semantics differ")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("full-token candidate pose ranker lacks state")
    state_arrays = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("full-token candidate pose ranker state differs")
        state_arrays[name] = value.detach().cpu().numpy()
    if payload.get("model_content_sha256") != arrays_sha256(state_arrays):
        raise ValueError("full-token candidate pose ranker content differs")
    fusion = payload.get("retrieval_prior_fusion_weight")
    if not isinstance(fusion, (int, float)) or not np.isfinite(fusion) or fusion < 0.0:
        raise ValueError("full-token candidate pose ranker prior fusion differs")
    model = FullTokenCandidatePoseRanker()
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    metadata = {name: value for name, value in payload.items() if name != "state_dict"}
    return model, metadata


def load_factorized_fulltoken_candidate_pose_ranker(
    path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FactorizedFullTokenCandidatePoseRanker, dict[str, object]]:
    """Load and content-verify a three-head diagnostic ranker."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("artifact_type") != (
        "goal_maplet_factorized_fulltoken_candidate_pose_ranker_v1"
    ):
        raise ValueError("not a factorized full-token candidate pose ranker")
    if payload.get("model_semantics") != FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS:
        raise ValueError("factorized full-token candidate pose ranker semantics differ")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("factorized full-token candidate pose ranker lacks state")
    state_arrays = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("factorized full-token candidate pose ranker state differs")
        state_arrays[name] = value.detach().cpu().numpy()
    if payload.get("model_content_sha256") != arrays_sha256(state_arrays):
        raise ValueError("factorized full-token candidate pose ranker content differs")
    input_channels = int(payload.get("input_channels", len(FULLTOKEN_POSE_RANKING_CHANNELS)))
    model = FactorizedFullTokenCandidatePoseRanker(input_channels=input_channels)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    metadata = {name: value for name, value in payload.items() if name != "state_dict"}
    return model, metadata


def listwise_pose_ranking_loss(
    score: torch.Tensor,
    translation_m: torch.Tensor,
    rotation_deg: torch.Tensor,
    valid: torch.Tensor,
    *,
    soft_target_temperature: float = 0.5,
    ordering_margin: float = 0.05,
    translation_scale_m: float = 1.0,
    rotation_scale_deg: float = 10.0,
) -> torch.Tensor:
    """Listwise soft-target loss plus all-pairs ordinal supervision."""

    logits = torch.as_tensor(score)
    translation = torch.as_tensor(translation_m, device=logits.device, dtype=logits.dtype)
    rotation = torch.as_tensor(rotation_deg, device=logits.device, dtype=logits.dtype)
    mask = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or translation.shape != logits.shape or rotation.shape != logits.shape:
        raise ValueError("pose-ranking loss arrays must share shape [B,C]")
    if mask.shape != logits.shape or torch.any(torch.sum(mask, dim=1) < 2):
        raise ValueError("each ranking row requires at least two valid candidates")
    if any(not torch.isfinite(value).all() for value in (logits, translation, rotation)):
        raise ValueError("pose-ranking loss inputs must be finite")
    temperature = float(soft_target_temperature)
    margin = float(ordering_margin)
    translation_scale = float(translation_scale_m)
    rotation_scale = float(rotation_scale_deg)
    if (
        not 0.0 < temperature < float("inf")
        or not 0.0 <= margin < float("inf")
        or not 0.0 < translation_scale < float("inf")
        or not 0.0 < rotation_scale < float("inf")
    ):
        raise ValueError("pose-ranking loss constants are invalid")
    joint = torch.maximum(translation / translation_scale, rotation / rotation_scale)
    masked_score = torch.where(mask, logits, torch.full_like(logits, -1.0e9))
    target_logits = torch.where(mask, -joint / temperature, torch.full_like(joint, -1.0e9))
    target_probability = torch.softmax(target_logits, dim=1)
    listwise = -torch.sum(
        target_probability * torch.log_softmax(masked_score, dim=1), dim=1,
    ).mean()
    better = (
        mask[:, :, None] & mask[:, None, :]
        & (joint[:, :, None] + margin < joint[:, None, :])
    )
    difference = logits[:, :, None] - logits[:, None, :]
    ordinal_values = torch.nn.functional.softplus(-difference[better])
    ordinal = ordinal_values.mean() if ordinal_values.numel() else logits.sum() * 0.0
    return listwise + ordinal


def multiscale_pose_ranking_loss(
    score: torch.Tensor,
    translation_m: torch.Tensor,
    rotation_deg: torch.Tensor,
    valid: torch.Tensor,
    *,
    anchor_margin: float = 0.10,
    anchor_margin_weight: float = 0.5,
) -> torch.Tensor:
    """Give local curvature and broad discrimination equal supervision.

    A single 8 m / 45 degree normalization makes sub-metre pose differences
    almost indistinguishable in the target distribution.  This loss evaluates
    the same pose-free score at three nested local domains plus the complete
    candidate set.  It also requires the rendered GT anchor (row zero) to beat
    the hardest perturbation in every informative domain.  No pose value is an
    input to the ranker; errors are training labels only.
    """

    logits = torch.as_tensor(score)
    translation = torch.as_tensor(translation_m, device=logits.device, dtype=logits.dtype)
    rotation = torch.as_tensor(rotation_deg, device=logits.device, dtype=logits.dtype)
    mask = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
    if (
        logits.ndim != 2 or translation.shape != logits.shape
        or rotation.shape != logits.shape or mask.shape != logits.shape
        or not torch.all(mask[:, 0])
    ):
        raise ValueError("multiscale ranking arrays differ or lack GT anchors")
    margin = float(anchor_margin)
    margin_weight = float(anchor_margin_weight)
    if not 0.0 <= margin < float("inf") or not 0.0 <= margin_weight < float("inf"):
        raise ValueError("multiscale anchor constants are invalid")

    domains = (
        (0.5, 5.0, False),
        (1.0, 10.0, False),
        (2.0, 20.0, False),
        # Keep every frozen natural hard negative in the broad loss, including
        # poses just outside the nominal 8 m / 45 degree acquisition domain.
        (8.0, 45.0, True),
    )
    losses = []
    anchor_losses = []
    for maximum_translation_m, maximum_rotation_deg, include_all in domains:
        domain = mask if include_all else (
            mask
            & (translation <= maximum_translation_m + 1.0e-8)
            & (rotation <= maximum_rotation_deg + 1.0e-8)
        )
        informative = torch.sum(domain, dim=1) >= 2
        if not torch.any(informative):
            continue
        losses.append(listwise_pose_ranking_loss(
            logits[informative], translation[informative], rotation[informative],
            domain[informative],
            translation_scale_m=maximum_translation_m,
            rotation_scale_deg=maximum_rotation_deg,
        ))
        nonanchor = domain[informative].clone()
        nonanchor[:, 0] = False
        hardest = torch.where(
            nonanchor, logits[informative], torch.full_like(logits[informative], -1.0e9),
        ).max(dim=1).values
        anchor_losses.append(torch.nn.functional.softplus(
            margin - (logits[informative, 0] - hardest),
        ).mean())
    if not losses:
        raise ValueError("multiscale supervision has no informative domain")
    return torch.stack(losses).mean() + margin_weight * torch.stack(anchor_losses).mean()


def natural_hard_negative_pose_ranking_loss(
    score: torch.Tensor,
    translation_m: torch.Tensor,
    rotation_deg: torch.Tensor,
    valid: torch.Tensor,
    *,
    natural_candidate_start_row: int,
    anchor_margin: float = 0.20,
    anchor_margin_weight: float = 0.5,
) -> torch.Tensor:
    """Rank the GT anchor specifically against frozen natural retrieval rows.

    Natural candidates are sparse in the merged inventory and would otherwise
    be diluted by hundreds of deterministic local probes.  The loss keeps the
    anchor plus every natural row, combines medium-basin and broad ordering,
    and adds a hardest-natural anchor margin.  Pose errors remain labels only.
    """
    logits = torch.as_tensor(score)
    translation = torch.as_tensor(translation_m, device=logits.device, dtype=logits.dtype)
    rotation = torch.as_tensor(rotation_deg, device=logits.device, dtype=logits.dtype)
    mask = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
    start = int(natural_candidate_start_row)
    if (
        logits.ndim != 2 or translation.shape != logits.shape
        or rotation.shape != logits.shape or mask.shape != logits.shape
        or start <= 1 or start >= logits.shape[1] or not torch.all(mask[:, 0])
    ):
        raise ValueError("natural hard-negative arrays or start row differ")
    natural_mask = mask.clone()
    natural_mask[:, 1:start] = False
    if torch.any(torch.sum(natural_mask, dim=1) < 2):
        raise ValueError("each query requires at least one natural hard negative")
    losses = (
        listwise_pose_ranking_loss(
            logits, translation, rotation, natural_mask,
            translation_scale_m=2.0, rotation_scale_deg=20.0,
        ),
        listwise_pose_ranking_loss(
            logits, translation, rotation, natural_mask,
            translation_scale_m=8.0, rotation_scale_deg=45.0,
        ),
    )
    natural_only = natural_mask.clone()
    natural_only[:, 0] = False
    hardest = torch.where(
        natural_only, logits, torch.full_like(logits, -1.0e9),
    ).max(dim=1).values
    margin_loss = torch.nn.functional.softplus(
        float(anchor_margin) - (logits[:, 0] - hardest),
    ).mean()
    return torch.stack(losses).mean() + float(anchor_margin_weight) * margin_loss


def trajectory_monotonic_pose_ranking_loss(
    score: torch.Tensor,
    trajectory_seed_index: torch.Tensor,
    trajectory_alpha: torch.Tensor,
    valid: torch.Tensor,
    *,
    ordering_margin: float = 0.02,
) -> torch.Tensor:
    """Require every frozen seed-to-GT training path to rise monotonically."""
    logits = torch.as_tensor(score)
    seeds = torch.as_tensor(trajectory_seed_index, device=logits.device, dtype=torch.int64)
    alpha = torch.as_tensor(trajectory_alpha, device=logits.device, dtype=logits.dtype)
    mask = torch.as_tensor(valid, device=logits.device, dtype=torch.bool)
    if (
        logits.ndim != 2 or seeds.shape != logits.shape or alpha.shape != logits.shape
        or mask.shape != logits.shape or not torch.all(mask[:, 0])
        or not torch.all(seeds[:, 0] == -1) or not torch.all(alpha[:, 0] == 1.0)
        or torch.any(~torch.isfinite(logits)) or torch.any(~torch.isfinite(alpha))
        or torch.any((alpha < 0.0) | (alpha > 1.0))
    ):
        raise ValueError("trajectory monotonic loss arrays differ")
    margin = float(ordering_margin)
    if not 0.0 <= margin < float("inf"):
        raise ValueError("trajectory monotonic ordering margin is invalid")
    query_losses = []
    for query in range(logits.shape[0]):
        seed_losses = []
        unique = torch.unique(seeds[query, 1:])
        if unique.numel() == 0 or torch.any(unique < 0):
            raise ValueError("trajectory row lacks frozen natural seed groups")
        for seed in unique.tolist():
            rows = torch.nonzero(
                mask[query] & (seeds[query] == int(seed)), as_tuple=False,
            ).flatten()
            if rows.numel() < 2:
                raise ValueError("trajectory seed group requires at least two samples")
            order = rows[torch.argsort(alpha[query, rows], stable=True)]
            ordered_alpha = alpha[query, order]
            if torch.any(ordered_alpha[1:] <= ordered_alpha[:-1]):
                raise ValueError("trajectory seed alphas must be strictly increasing")
            ordered_score = logits[query, order]
            consecutive = torch.nn.functional.softplus(
                margin - (ordered_score[1:] - ordered_score[:-1]),
            ).mean()
            anchor = torch.nn.functional.softplus(
                margin - (logits[query, 0] - ordered_score[-1]),
            )
            seed_losses.append(consecutive + anchor)
        query_losses.append(torch.stack(seed_losses).mean())
    return torch.stack(query_losses).mean()


def greedy_distinct_pose_basin_order(
    score: np.ndarray,
    poses_w2c: np.ndarray,
    valid: np.ndarray,
    *,
    excluded_candidate_index: int | None = 0,
    translation_threshold_m: float = 0.5,
    rotation_threshold_deg: float = 5.0,
) -> np.ndarray:
    """Stable score order after deterministic physical-pose greedy NMS.

    Pairwise 0.5 m/5 degree proximity is not transitive, so this is explicitly
    an ordered evaluation NMS rather than an equivalence-class partition.
    """

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    poses = np.asarray(poses_w2c, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if poses.shape != (values.size, 4, 4) or mask.shape != values.shape:
        raise ValueError("pose-basin ranking arrays differ")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(poses)):
        raise ValueError("pose-basin ranking arrays must be finite")
    indices = np.flatnonzero(mask)
    if excluded_candidate_index is not None:
        indices = indices[indices != int(excluded_candidate_index)]
    order = indices[np.lexsort((indices, -values[indices]))]
    retained: list[int] = []
    centers = -np.swapaxes(poses[:, :3, :3], 1, 2) @ poses[:, :3, 3, None]
    centers = centers[:, :, 0]
    rotations = poses[:, :3, :3]
    for candidate in order.tolist():
        duplicate = False
        if retained:
            previous = np.asarray(retained, dtype=np.int64)
            translation = np.linalg.norm(
                centers[previous] - centers[candidate][None], axis=1,
            )
            # trace(R_candidate @ R_previous.T) is the Frobenius inner
            # product.  Vectorizing this exact expression removes the
            # report-time O(C^2) Python/matmul overhead without changing the
            # stable greedy NMS order or its non-transitive semantics.
            relative_trace = np.einsum(
                "ij,kij->k", rotations[candidate], rotations[previous],
            )
            cosine = np.clip((relative_trace - 1.0) / 2.0, -1.0, 1.0)
            rotation = np.degrees(np.arccos(cosine))
            duplicate = bool(np.any(
                (translation <= float(translation_threshold_m) + 1.0e-12)
                & (rotation <= float(rotation_threshold_deg) + 1.0e-10)
            ))
        if not duplicate:
            retained.append(candidate)
    return np.asarray(retained, dtype=np.int64)


def round_robin_union_score_rows(
    expert_score_rows: np.ndarray,
    poses_w2c: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """Encode an unweighted round-robin union of expert orders as scores.

    The first expert wins only exact same-depth ties.  No learned calibration
    weight is required, and duplicate physical poses are removed by the same
    deterministic greedy evaluation NMS used everywhere else.
    """

    experts = np.asarray(expert_score_rows, dtype=np.float64)
    poses = np.asarray(poses_w2c, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    if (
        experts.ndim != 3
        or poses.shape != (experts.shape[0], experts.shape[2], 4, 4)
        or valid.shape != (experts.shape[0], experts.shape[2])
        or experts.shape[1] < 2
    ):
        raise ValueError("round-robin expert ranking arrays differ")
    output = np.full(valid.shape, -1.0e9, dtype=np.float32)
    for query in range(experts.shape[0]):
        candidate_rows = np.flatnonzero(valid[query])
        candidate_rows = candidate_rows[candidate_rows != 0]
        orders = [
            candidate_rows[np.lexsort((candidate_rows, -row[candidate_rows]))]
            for row in experts[query]
        ]
        merged: list[int] = []
        seen: set[int] = set()
        for depth in range(max(map(len, orders), default=0)):
            for order in orders:
                if depth < len(order) and int(order[depth]) not in seen:
                    seen.add(int(order[depth]))
                    merged.append(int(order[depth]))
        preliminary = np.full((valid.shape[1],), -1.0e9, dtype=np.float64)
        for rank, candidate in enumerate(merged):
            preliminary[candidate] = float(len(merged) - rank)
        distinct = greedy_distinct_pose_basin_order(
            preliminary, poses[query], valid[query],
        )
        for rank, candidate in enumerate(distinct.tolist()):
            output[query, candidate] = float(len(distinct) - rank)
    return output


def distinct_pose_basin_recall_metrics(
    score_rows: np.ndarray,
    poses_w2c: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    valid_rows: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 4, 8, 16),
) -> dict[str, object]:
    """Evaluate strict/loose retention after duplicate-aware score ordering."""

    score = np.asarray(score_rows)
    poses = np.asarray(poses_w2c)
    translation = np.asarray(translation_m)
    rotation = np.asarray(rotation_deg)
    valid = np.asarray(valid_rows, dtype=bool)
    if (
        score.ndim != 2
        or poses.shape != score.shape + (4, 4)
        or translation.shape != score.shape
        or rotation.shape != score.shape
        or valid.shape != score.shape
        or score.shape[0] == 0
    ):
        raise ValueError("distinct pose-basin metric arrays differ")
    values = tuple(int(value) for value in topk)
    if not values or any(value <= 0 for value in values):
        raise ValueError("pose-basin TopK values must be positive")
    hits = {f"strict_recall_at_{value}": [] for value in values}
    hits.update({f"loose_recall_at_{value}": [] for value in values})
    distinct_count, raw_strict, raw_loose = [], [], []
    for query in range(score.shape[0]):
        nonanchor = valid[query].copy()
        nonanchor[0] = False
        raw_strict.append(bool(np.any(
            nonanchor & (translation[query] <= 0.5) & (rotation[query] <= 5.0)
        )))
        raw_loose.append(bool(np.any(
            nonanchor & (translation[query] <= 1.0) & (rotation[query] <= 10.0)
        )))
        order = greedy_distinct_pose_basin_order(
            score[query], poses[query], valid[query],
        )
        distinct_count.append(int(order.size))
        for value in values:
            selected = order[:value]
            hits[f"strict_recall_at_{value}"].append(bool(np.any(
                (translation[query, selected] <= 0.5)
                & (rotation[query, selected] <= 5.0)
            )))
            hits[f"loose_recall_at_{value}"].append(bool(np.any(
                (translation[query, selected] <= 1.0)
                & (rotation[query, selected] <= 10.0)
            )))
    return {
        "query_count": int(score.shape[0]),
        "raw_candidate_strict_0_5m_5deg": float(np.mean(raw_strict)),
        "raw_candidate_loose_1m_10deg": float(np.mean(raw_loose)),
        "mean_distinct_pose_basin_count": float(np.mean(distinct_count)),
        "minimum_distinct_pose_basin_count": int(np.min(distinct_count)),
        **{name: float(np.mean(result)) for name, result in hits.items()},
        "nms_semantics": "stable_score_order_greedy_0.5m_5deg_nontransitive_evaluation_nms",
    }


def continuous_seed_domain_recall_metrics(
    score_rows: np.ndarray,
    poses_w2c: np.ndarray,
    valid_rows: np.ndarray,
    *,
    translation_half_extent_m: float = 8.0,
    rotation_radius_deg: float = 45.0,
    topk: tuple[int, ...] = (1, 4, 8, 16),
) -> dict[str, object]:
    """Measure whether ranked seed domains retain the diagnostic target pose.

    Candidate zero supplies the diagnostic target and is always excluded from
    ranking.  A hit means the target lies inside, or within the stated
    strict/loose residual of, a continuous seed-centred domain.  It is not a
    claim that a search algorithm found the target pose.
    """

    score = np.asarray(score_rows)
    poses = np.asarray(poses_w2c, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    if (
        score.ndim != 2 or poses.shape != score.shape + (4, 4)
        or valid.shape != score.shape or score.shape[0] == 0
        or not np.all(valid[:, 0])
    ):
        raise ValueError("continuous seed-domain metric arrays differ")
    extent = float(translation_half_extent_m)
    radius = float(rotation_radius_deg)
    if not 0.0 < extent < float("inf") or not 0.0 < radius < 180.0:
        raise ValueError("continuous seed-domain bounds are invalid")
    values = tuple(int(value) for value in topk)
    result = {f"strict_domain_acquisition_at_{value}": [] for value in values}
    result.update({f"loose_domain_acquisition_at_{value}": [] for value in values})
    raw_strict, raw_loose = [], []
    for query in range(score.shape[0]):
        target_rotation = poses[query, 0, :3, :3]
        target_center = -target_rotation.T @ poses[query, 0, :3, 3]
        rotations = poses[query, :, :3, :3]
        centers = -np.swapaxes(rotations, 1, 2) @ poses[query, :, :3, 3, None]
        centers = centers[:, :, 0]
        delta = target_center[None] - centers
        translation_residual = np.linalg.norm(
            delta - np.clip(delta, -extent, extent), axis=1,
        )
        relative = rotations @ target_rotation.T
        cosine = np.clip(
            (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
        )
        rotation_residual = np.maximum(np.degrees(np.arccos(cosine)) - radius, 0.0)
        nonanchor = valid[query].copy()
        nonanchor[0] = False
        strict = nonanchor & (translation_residual <= 0.5 + 1.0e-6) & (
            rotation_residual <= 5.0 + 1.0e-5
        )
        loose = nonanchor & (translation_residual <= 1.0 + 1.0e-6) & (
            rotation_residual <= 10.0 + 1.0e-5
        )
        raw_strict.append(bool(np.any(strict)))
        raw_loose.append(bool(np.any(loose)))
        order = greedy_distinct_pose_basin_order(score[query], poses[query], valid[query])
        for value in values:
            selected = order[:value]
            result[f"strict_domain_acquisition_at_{value}"].append(bool(np.any(strict[selected])))
            result[f"loose_domain_acquisition_at_{value}"].append(bool(np.any(loose[selected])))
    return {
        "query_count": int(score.shape[0]),
        "translation_half_extent_m": extent,
        "rotation_radius_deg": radius,
        "raw_strict_domain_acquisition": float(np.mean(raw_strict)),
        "raw_loose_domain_acquisition": float(np.mean(raw_loose)),
        **{name: float(np.mean(rows)) for name, rows in result.items()},
        "acquisition_is_not_search_or_localization_success": True,
    }
