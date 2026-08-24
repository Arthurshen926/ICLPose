"""Pose-free RADIO prediction and candidate-conditioned typed geometry evidence."""

from __future__ import annotations

import torch

from .lineage import arrays_sha256


QUERY_TYPED_GEOMETRY_PREDICTOR_SEMANTICS = (
    "pose_free_fulltoken_radio_to_unsigned_normal_relative_depth_boundary_v1"
)


class QueryTypedGeometryPredictor(torch.nn.Module):
    """Predict camera-frame token geometry without pose or candidate inputs."""

    def __init__(self, input_channels: int = 128) -> None:
        super().__init__()
        if int(input_channels) != 128:
            raise ValueError("query typed geometry requires the frozen 128-D readout")
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 7, kernel_size=1),
        )

    def forward(self, query_descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        value = torch.as_tensor(query_descriptor)
        if value.ndim != 4 or value.shape[1:] != (128, 36, 64):
            raise ValueError("query descriptor must have shape [B,128,36,64]")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError("query descriptor must be finite floating point")
        raw = self.network(value)
        normal = raw[:, :3]
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(
            1.0e-8
        )
        x, y, z = normal.unbind(dim=1)
        moment = torch.stack((x * x, y * y, z * z, x * y, x * z, y * z), dim=1)
        return {
            "normal_axis_moment": moment,
            "relative_log_depth": 2.0 * torch.tanh(raw[:, 3]),
            "log_depth_std": torch.nn.functional.softplus(raw[:, 4]),
            "boundary": torch.sigmoid(raw[:, 5]),
            "confidence": torch.sigmoid(raw[:, 6]),
        }


def query_typed_geometry_supervision_loss(
    prediction: dict[str, torch.Tensor],
    target_geometry: torch.Tensor,
    target_mass: torch.Tensor,
) -> torch.Tensor:
    """Mass-weighted physical supervision at the target camera pose."""

    target = torch.as_tensor(target_geometry)
    mass = torch.as_tensor(target_mass, device=target.device, dtype=target.dtype)
    if target.ndim != 4 or target.shape[1:] != (9, 36, 64):
        raise ValueError("typed target geometry must have shape [B,9,36,64]")
    if mass.shape != (target.shape[0], 36, 64):
        raise ValueError("typed target mass shape differs")
    if not torch.isfinite(target).all() or not torch.isfinite(mass).all():
        raise ValueError("typed target supervision must be finite")
    weight = mass.clamp(0.0, 1.0)
    denominator = weight.sum().clamp_min(1.0)

    def weighted_smooth_l1(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.smooth_l1_loss(estimate, truth, reduction="none")
        if loss.ndim == 4:
            loss = loss.mean(dim=1)
        return torch.sum(weight * loss) / denominator

    geometry = (
        weighted_smooth_l1(prediction["normal_axis_moment"], target[:, :6])
        + weighted_smooth_l1(prediction["relative_log_depth"], target[:, 6])
        + weighted_smooth_l1(prediction["log_depth_std"], target[:, 7])
        + weighted_smooth_l1(prediction["boundary"], target[:, 8])
    )
    confidence = torch.nn.functional.binary_cross_entropy(
        prediction["confidence"], weight, reduction="mean",
    )
    return geometry + 0.25 * confidence


def append_query_typed_geometry_consistency(
    typed_candidate_features: torch.Tensor,
    query_prediction: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Append five fixed-layout, missing-conservative compatibility channels."""

    feature = torch.as_tensor(typed_candidate_features)
    if feature.ndim != 5 or feature.shape[2:] != (18, 36, 64):
        raise ValueError("typed candidate features must have shape [B,C,18,36,64]")
    batch, candidates = feature.shape[:2]
    candidate_moment = feature[:, :, 9:15]
    candidate_depth = feature[:, :, 15]
    candidate_std = feature[:, :, 16]
    candidate_boundary = feature[:, :, 17]
    candidate_mass = feature[:, :, 1].clamp(0.0, 1.0)

    query_moment = query_prediction["normal_axis_moment"][:, None]
    if query_moment.shape != (batch, 1, 6, 36, 64):
        raise ValueError("query typed prediction shape differs")
    confidence = query_prediction["confidence"][:, None].clamp(0.0, 1.0)
    if confidence.shape != (batch, 1, 36, 64):
        raise ValueError("query typed confidence shape differs")
    confidence = confidence.expand(-1, candidates, -1, -1)
    evidence_mass = confidence * candidate_mass
    # Frobenius inner product of symmetric axis moments; off diagonals occur twice.
    normal_similarity = (
        torch.sum(query_moment[:, :, :3] * candidate_moment[:, :, :3], dim=2)
        + 2.0 * torch.sum(query_moment[:, :, 3:] * candidate_moment[:, :, 3:], dim=2)
    ).clamp(0.0, 1.0)
    query_depth = query_prediction["relative_log_depth"][:, None]
    query_std = query_prediction["log_depth_std"][:, None]
    query_boundary = query_prediction["boundary"][:, None]
    depth_similarity = torch.exp(-torch.abs(query_depth - candidate_depth) / 0.5)
    std_similarity = torch.exp(-torch.abs(query_std - candidate_std) / 0.5)
    boundary_similarity = (1.0 - torch.abs(query_boundary - candidate_boundary)).clamp(0.0, 1.0)
    appended = torch.stack((
        confidence,
        evidence_mass * normal_similarity,
        evidence_mass * depth_similarity,
        evidence_mass * std_similarity,
        evidence_mass * boundary_similarity,
    ), dim=2)
    if not torch.isfinite(appended).all():
        raise ValueError("query typed consistency produced non-finite evidence")
    return torch.cat((feature, appended), dim=2)


def fixed_denominator_query_typed_geometry_score(
    query_prediction: dict[str, torch.Tensor],
    candidate_normal_axis_moment: torch.Tensor,
    candidate_relative_log_depth: torch.Tensor,
    candidate_log_depth_std: torch.Tensor,
    candidate_boundary: torch.Tensor,
    candidate_mass: torch.Tensor,
) -> torch.Tensor:
    """Return a missing-conservative geometry compatibility in ``[-1,1]``.

    Every one of the 2304 token locations remains in the denominator.  Query
    confidence and rendered mass are evidence factors, so disappearing
    candidate evidence cannot improve this score while descriptors are held
    fixed.
    """

    moment = torch.as_tensor(candidate_normal_axis_moment)
    depth = torch.as_tensor(candidate_relative_log_depth, device=moment.device)
    std = torch.as_tensor(candidate_log_depth_std, device=moment.device)
    boundary = torch.as_tensor(candidate_boundary, device=moment.device)
    mass = torch.as_tensor(candidate_mass, device=moment.device)
    if moment.ndim != 4 or moment.shape[1:] != (36, 64, 6):
        raise ValueError("candidate normal moment must have shape [B,36,64,6]")
    batch = int(moment.shape[0])
    if any(value.shape != (batch, 36, 64) for value in (depth, std, boundary, mass)):
        raise ValueError("candidate typed geometry arrays differ")
    query_moment = query_prediction["normal_axis_moment"].permute(0, 2, 3, 1)
    query_depth = query_prediction["relative_log_depth"]
    query_std = query_prediction["log_depth_std"]
    query_boundary = query_prediction["boundary"]
    confidence = query_prediction["confidence"].clamp(0.0, 1.0)
    if (
        query_moment.shape != moment.shape
        or any(value.shape != (batch, 36, 64) for value in (
            query_depth, query_std, query_boundary, confidence,
        ))
    ):
        raise ValueError("query typed geometry arrays differ")
    values = (
        moment, depth, std, boundary, mass, query_moment, query_depth,
        query_std, query_boundary, confidence,
    )
    if any(
        not torch.is_floating_point(value) or not torch.isfinite(value).all()
        for value in values
    ):
        raise ValueError("query/candidate typed geometry must be finite floating point")
    candidate_moment = moment.permute(0, 3, 1, 2)
    query_moment_cf = query_moment.permute(0, 3, 1, 2)
    normal_similarity = (
        torch.sum(query_moment_cf[:, :3] * candidate_moment[:, :3], dim=1)
        + 2.0 * torch.sum(
            query_moment_cf[:, 3:] * candidate_moment[:, 3:], dim=1,
        )
    ).clamp(0.0, 1.0)
    similarities = torch.stack((
        normal_similarity,
        torch.exp(-torch.abs(query_depth - depth) / 0.5),
        torch.exp(-torch.abs(query_std - std) / 0.5),
        (1.0 - torch.abs(query_boundary - boundary)).clamp(0.0, 1.0),
    ), dim=1)
    evidence = confidence * mass.clamp(0.0, 1.0)
    above_floor = torch.mean(evidence[:, None] * similarities, dim=(1, 2, 3))
    return -1.0 + 2.0 * above_floor


def conjunct_phase_and_typed_geometry_score(
    phase_score: torch.Tensor, geometry_score: torch.Tensor,
) -> torch.Tensor:
    """Conjoin two ``[-1,1]`` scores through their above-floor products."""

    phase = torch.as_tensor(phase_score)
    geometry = torch.as_tensor(geometry_score, device=phase.device, dtype=phase.dtype)
    if (
        phase.shape != geometry.shape
        or not torch.isfinite(phase).all()
        or not torch.isfinite(geometry).all()
    ):
        raise ValueError("phase and geometry score arrays differ")
    if torch.any((phase < -1.00001) | (phase > 1.00001)) or torch.any(
        (geometry < -1.00001) | (geometry > 1.00001)
    ):
        raise ValueError("phase and geometry scores must lie in [-1,1]")
    return -1.0 + 0.5 * (phase + 1.0) * (geometry + 1.0)


def load_query_typed_geometry_predictor(
    path, *, device: str | torch.device = "cpu",
) -> tuple[QueryTypedGeometryPredictor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_type")
        != "goal_maplet_query_typed_geometry_predictor_v1"
        or payload.get("model_semantics") != QUERY_TYPED_GEOMETRY_PREDICTOR_SEMANTICS
    ):
        raise ValueError("not a query typed geometry predictor")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("query typed geometry predictor lacks state")
    arrays = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("query typed geometry predictor state differs")
        arrays[name] = value.detach().cpu().numpy()
    if payload.get("model_content_sha256") != arrays_sha256(arrays):
        raise ValueError("query typed geometry predictor content differs")
    model = QueryTypedGeometryPredictor()
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}
