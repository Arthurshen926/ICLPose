"""Candidate-relative multimodal SE(3) correction from full-token evidence.

The model never sees an absolute pose matrix and never predicts world
coordinates.  It receives only the same query-versus-rendered full-layout
evidence used by the rankers and predicts a bounded mixture of left-tangent
corrections relative to that candidate.  This is a high-tolerance basin
transport head, not keypoint matching, PnP, or absolute pose regression.
"""

from __future__ import annotations

import numpy as np
import torch

from .fulltoken_pose_ranking import FULLTOKEN_POSE_RANKING_CHANNELS
from .lineage import arrays_sha256
from .se3_local_quadratic import left_retract_pose_w2c


RELATIVE_POSE_CORRECTION_SEMANTICS = (
    "fulltoken_candidate_relative_left_se3_multimodal_correction_v1"
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def left_pose_correction_coordinate(
    candidate_pose_w2c: np.ndarray,
    target_pose_w2c: np.ndarray,
    *,
    translation_scale_m: float = 8.0,
    rotation_scale_deg: float = 45.0,
) -> np.ndarray:
    """Return normalized ``xi`` satisfying ``target=Exp(xi) candidate``."""

    candidate = np.asarray(candidate_pose_w2c, dtype=np.float64).reshape(4, 4)
    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(target)):
        raise ValueError("relative correction poses must be finite")
    delta = target @ np.linalg.inv(candidate)
    rotation = delta[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < 1.0e-8:
        omega = 0.5 * np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        omega_hat = _skew(omega)
        inverse_jacobian = np.eye(3) - 0.5 * omega_hat + (1.0 / 12.0) * (omega_hat @ omega_hat)
    else:
        omega = theta / (2.0 * np.sin(theta)) * np.asarray([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        omega_hat = _skew(omega)
        coefficient = (
            1.0 / (theta * theta)
            - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
        inverse_jacobian = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
    rho = inverse_jacobian @ delta[:3, 3]
    translation_scale = float(translation_scale_m)
    rotation_scale = np.radians(float(rotation_scale_deg))
    if translation_scale <= 0.0 or rotation_scale <= 0.0:
        raise ValueError("relative correction scales must be positive")
    coordinate = np.concatenate([rho / translation_scale, omega / rotation_scale])
    if np.any(~np.isfinite(coordinate)):
        raise ValueError("relative correction coordinate is nonfinite")
    return coordinate


def apply_left_pose_correction_coordinate(
    candidate_pose_w2c: np.ndarray,
    coordinate: np.ndarray,
    *,
    translation_scale_m: float = 8.0,
    rotation_scale_deg: float = 45.0,
) -> np.ndarray:
    return left_retract_pose_w2c(
        candidate_pose_w2c, np.asarray(coordinate, dtype=np.float64),
        translation_step_m=float(translation_scale_m),
        rotation_step_degrees=float(rotation_scale_deg),
    )


class FullTokenRelativePoseCorrectionNet(torch.nn.Module):
    def __init__(self, *, mode_count: int = 4) -> None:
        super().__init__()
        if int(mode_count) <= 0:
            raise ValueError("relative correction mode count must be positive")
        self.mode_count = int(mode_count)
        channels = len(FULLTOKEN_POSE_RANKING_CHANNELS)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 48, 3, stride=2, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(48, 64, 3, stride=2, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1), torch.nn.SiLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 * 3 * 4, 192), torch.nn.SiLU(),
            torch.nn.Linear(192, self.mode_count * 7),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(features)
        if value.ndim != 4 or value.shape[1:] != (
            len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64,
        ):
            raise ValueError("relative correction features must have shape [N,9,36,64]")
        encoded = self.encoder(value)
        average = torch.nn.functional.adaptive_avg_pool2d(encoded, (3, 4))
        maximum = torch.nn.functional.adaptive_max_pool2d(encoded, (3, 4))
        raw = self.head(torch.cat([average, maximum], dim=1).flatten(1)).reshape(
            value.shape[0], self.mode_count, 7,
        )
        coordinate = torch.tanh(raw[:, :, :6])
        rotation = coordinate[:, :, 3:]
        rotation_norm = torch.linalg.vector_norm(rotation, dim=2, keepdim=True)
        coordinate = torch.cat([
            coordinate[:, :, :3],
            rotation / torch.clamp_min(rotation_norm, 1.0),
        ], dim=2)
        return coordinate, raw[:, :, 6]


def relative_pose_mixture_loss(
    predicted_coordinate: torch.Tensor,
    mode_logit: torch.Tensor,
    target_coordinate: torch.Tensor,
    *,
    error_temperature: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = torch.as_tensor(predicted_coordinate)
    logits = torch.as_tensor(mode_logit, device=prediction.device, dtype=prediction.dtype)
    target = torch.as_tensor(target_coordinate, device=prediction.device, dtype=prediction.dtype)
    if (
        prediction.ndim != 3 or prediction.shape[2] != 6
        or logits.shape != prediction.shape[:2]
        or target.shape != (prediction.shape[0], 6)
        or any(not torch.isfinite(value).all() for value in (prediction, logits, target))
    ):
        raise ValueError("relative correction loss arrays differ")
    temperature = float(error_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("relative correction temperature must be positive")
    error = torch.nn.functional.smooth_l1_loss(
        prediction, target[:, None, :].expand_as(prediction), reduction="none",
    ).mean(dim=2)
    log_mixture = torch.log_softmax(logits, dim=1)
    loss = -torch.logsumexp(log_mixture - error / temperature, dim=1).mean()
    best = torch.min(error, dim=1).values
    return loss, {
        "mean_best_mode_smooth_l1": float(best.detach().mean().cpu()),
        "mean_mixture_negative_log_likelihood": float(loss.detach().cpu()),
    }


def load_fulltoken_relative_pose_correction(
    path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FullTokenRelativePoseCorrectionNet, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_type")
        != "goal_maplet_fulltoken_relative_pose_correction_v1"
        or payload.get("model_semantics") != RELATIVE_POSE_CORRECTION_SEMANTICS
    ):
        raise ValueError("not a full-token relative pose correction model")
    mode_count = payload.get("mode_count")
    state = payload.get("state_dict")
    if not isinstance(mode_count, int) or mode_count <= 0 or not isinstance(state, dict):
        raise ValueError("relative pose correction model schema differs")
    state_arrays = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("relative pose correction state differs")
        state_arrays[name] = value.detach().cpu().numpy()
    if payload.get("model_content_sha256") != arrays_sha256(state_arrays):
        raise ValueError("relative pose correction content differs")
    model = FullTokenRelativePoseCorrectionNet(mode_count=mode_count)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}
