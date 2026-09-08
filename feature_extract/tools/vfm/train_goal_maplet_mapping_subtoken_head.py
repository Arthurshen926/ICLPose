"""Train a mapping-only RADIO-pair head for continuous query token coordinates.

The target is deliberately pair-specific.  For two mapping observations of the
same finite-plane metric cell, the map observation's metric world point is
projected into the query-like observation.  The head predicts that point's
offset from the 4x4 RADIO token centre, an isotropic measurement variance, and
a null/match probability.  The entire validation route is excluded from fit.

No RGB image, query image, query pose, query depth, or query label is read.
Mapping poses are used only to manufacture mapping-only correspondence labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.tools.vfm.build_goal_maplet_plane_pnp_observation_bank import (
    _load_contributor_geometry,
)
from feature_extract.tools.vfm.train_goal_maplet_chart_local_radio_projection import (
    _load_observation_bank,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import (
    PlaneVisibilityAtlas,
)


TOKEN_GRID = (36, 64)
TOKEN_SIZE_PX = 4.0
SUBTOKEN_HALF_EXTENT_PX = 2.0
MAXIMUM_POSITIVE_WORLD_DISTANCE_M = 0.35
MAXIMUM_VIEWS_PER_IDENTITY = 8
MINIMUM_VARIANCE_PX2 = 0.01
MINIMUM_UV_VARIANCE_M2 = 1e-5
MAXIMUM_CHART_UV_OFFSET_M = 0.5


def _load_projection(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"weight", "metadata_json"}:
            raise ValueError("chart-local RADIO projection schema differs")
        weight = np.asarray(data["weight"], np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
    content = dict(metadata); claimed = content.pop("content_sha256", None)
    if (
        metadata.get("artifact_type") != "goal_maplet_chart_local_radio_projection_v1"
        or metadata.get("query_pose_depth_or_ground_truth_read") is not False
        or arrays_sha256({"weight": weight}) != metadata.get("arrays_sha256")
        or canonical_json_sha256(content) != claimed
        or weight.ndim != 2
    ):
        raise ValueError("chart-local RADIO projection lineage differs")
    return weight, metadata


def _project_world_to_pixel(
    world: np.ndarray,
    pose_w2c: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(world, np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, np.float64).reshape(-1, 4, 4)
    if len(point) != len(pose):
        raise ValueError("world points and mapping poses differ")
    camera = np.einsum("nij,nj->ni", pose[:, :3, :3], point) + pose[:, :3, 3]
    z = camera[:, 2]
    ideal = camera[:, :2] / np.maximum(z[:, None], 1e-12)
    radial = np.asarray(radial_k1, np.float64)
    if radial.ndim == 0:
        radial = np.full(len(point), float(radial), np.float64)
    radial = radial.reshape(-1)
    matrix = np.asarray(camera_matrix, np.float64)
    if matrix.ndim == 2:
        matrix = np.broadcast_to(matrix, (len(point), 3, 3))
    if matrix.shape != (len(point), 3, 3) or radial.shape != (len(point),):
        raise ValueError("mapping intrinsics and world points differ")
    distorted = ideal * (1.0 + radial[:, None] * np.sum(ideal * ideal, axis=1, keepdims=True))
    pixel = distorted * matrix[:, (0, 1), (0, 1)] + matrix[:, (0, 1), (2, 2)]
    return pixel, z


def _token_centres(token_ids: np.ndarray) -> np.ndarray:
    token = np.asarray(token_ids, np.int64).reshape(-1)
    ty, tx = np.divmod(token, TOKEN_GRID[1])
    return np.c_[tx * TOKEN_SIZE_PX + 1.5, ty * TOKEN_SIZE_PX + 1.5]


def _representative_rows(
    identity: np.ndarray,
    observation: np.ndarray,
    uv: np.ndarray,
    cell: np.ndarray,
    selected_observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(np.asarray(selected_observations, bool)[observation])
    cell_centre = (cell[selected].astype(np.float64) + 0.5) * 0.5
    distance = np.sum((uv[selected] - cell_centre) ** 2, axis=1)
    order = selected[np.lexsort((selected, distance, observation[selected], identity[selected]))]
    first = np.r_[True, (identity[order][1:] != identity[order][:-1]) |
                  (observation[order][1:] != observation[order][:-1])]
    rows = order[first]
    return rows, identity[rows]


def _directed_pairs(
    representative_rows: np.ndarray,
    representative_identity: np.ndarray,
    *,
    maximum_views: int = MAXIMUM_VIEWS_PER_IDENTITY,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((representative_rows, representative_identity))
    rows = representative_rows[order]; identity = representative_identity[order]
    starts = np.flatnonzero(np.r_[True, identity[1:] != identity[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    query, mapping = [], []
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        group = rows[lo:hi]
        if len(group) < 2:
            continue
        if len(group) > int(maximum_views):
            take = np.unique(np.rint(np.linspace(0, len(group) - 1, int(maximum_views))).astype(np.int64))
            group = group[take]
        for index in range(len(group)):
            other = (index + 1) % len(group)
            query.extend((int(group[index]), int(group[other])))
            mapping.extend((int(group[other]), int(group[index])))
    return np.asarray(query, np.int64), np.asarray(mapping, np.int64)


def _pair_input(query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(query * mapping, dim=1, keepdim=True)
    ty = torch.div(token_ids, TOKEN_GRID[1], rounding_mode="floor").float()
    tx = torch.remainder(token_ids, TOKEN_GRID[1]).float()
    location = torch.stack((2.0 * tx / (TOKEN_GRID[1] - 1.0) - 1.0,
                            2.0 * ty / (TOKEN_GRID[0] - 1.0) - 1.0), dim=1)
    return torch.cat((query, mapping, query - mapping, query * mapping, cosine, location), dim=1)


class MappingSubtokenHead(torch.nn.Module):
    def __init__(self, feature_dimension: int, hidden_dimension: int = 96) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(4 * int(feature_dimension) + 3, int(hidden_dimension))
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.hidden(_pair_input(query, mapping, token_ids)))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        return mean, variance, self.match(hidden)


class MappingSurfaceCoordinateHead(torch.nn.Module):
    """Jointly refine query sub-token position and continuous chart UV."""

    def __init__(self, feature_dimension: int, hidden_dimension: int = 96) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(4 * int(feature_dimension) + 3, int(hidden_dimension))
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.hidden(_pair_input(query, mapping, token_ids)))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(self.chart_uv_mean(hidden))
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(self.chart_uv_log_variance(hidden))
        return mean, variance, chart_uv_mean, chart_uv_variance, self.match(hidden)


class MappingSurfaceCoordinateMixtureHead(torch.nn.Module):
    """Mapping-only mixture posterior for ambiguous within-cell chart coordinates."""

    def __init__(
        self, feature_dimension: int, hidden_dimension: int = 96, mixture_modes: int = 3,
    ) -> None:
        super().__init__()
        self.mixture_modes = int(mixture_modes)
        if self.mixture_modes < 2:
            raise ValueError("surface-coordinate mixture requires at least two modes")
        self.hidden = torch.nn.Linear(4 * int(feature_dimension) + 3, int(hidden_dimension))
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2 * self.mixture_modes)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), self.mixture_modes)
        self.chart_uv_logits = torch.nn.Linear(int(hidden_dimension), self.mixture_modes)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        hidden = F.gelu(self.hidden(_pair_input(query, mapping, token_ids)))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(
            self.chart_uv_mean(hidden).reshape(-1, self.mixture_modes, 2)
        )
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(
            self.chart_uv_log_variance(hidden)
        )
        return (
            mean, variance, chart_uv_mean, chart_uv_variance,
            self.chart_uv_logits(hidden), self.match(hidden),
        )


class MappingSurfaceCoordinateContextHead(torch.nn.Module):
    """Single-coordinate posterior conditioned on ray/plane/prototype geometry."""

    CONTEXT_DIMENSION = 5

    def __init__(self, feature_dimension: int, hidden_dimension: int = 96) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(
            4 * int(feature_dimension) + 3 + self.CONTEXT_DIMENSION,
            int(hidden_dimension),
        )
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
        geometric_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if geometric_context.ndim != 2 or geometric_context.shape != (
            len(query), self.CONTEXT_DIMENSION,
        ):
            raise ValueError("surface-coordinate geometric context differs")
        context = geometric_context
        if not torch.isfinite(context).all():
            raise ValueError("surface-coordinate geometric context differs")
        hidden = F.gelu(self.hidden(torch.cat((_pair_input(query, mapping, token_ids), context), dim=1)))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(self.chart_uv_mean(hidden))
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(self.chart_uv_log_variance(hidden))
        return mean, variance, chart_uv_mean, chart_uv_variance, self.match(hidden)


class MappingSurfaceCoordinateDeepContextHead(torch.nn.Module):
    """Two-layer mapping-only geometry-context coordinate posterior."""

    CONTEXT_DIMENSION = MappingSurfaceCoordinateContextHead.CONTEXT_DIMENSION

    def __init__(self, feature_dimension: int, hidden_dimension: int = 192) -> None:
        super().__init__()
        input_dimension = 4 * int(feature_dimension) + 3 + self.CONTEXT_DIMENSION
        self.hidden_1 = torch.nn.Linear(input_dimension, int(hidden_dimension))
        self.hidden_2 = torch.nn.Linear(int(hidden_dimension), int(hidden_dimension))
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
        geometric_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if geometric_context.ndim != 2 or geometric_context.shape != (
            len(query), self.CONTEXT_DIMENSION,
        ):
            raise ValueError("surface-coordinate geometric context differs")
        if not torch.isfinite(geometric_context).all():
            raise ValueError("surface-coordinate geometric context differs")
        hidden = F.gelu(self.hidden_1(
            torch.cat((_pair_input(query, mapping, token_ids), geometric_context), dim=1),
        ))
        hidden = F.gelu(self.hidden_2(hidden))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(self.chart_uv_mean(hidden))
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(self.chart_uv_log_variance(hidden))
        return mean, variance, chart_uv_mean, chart_uv_variance, self.match(hidden)


class MappingSurfaceCoordinateHomographyContextHead(torch.nn.Module):
    """Single-layer posterior with a frozen coarse-homography residual."""

    CONTEXT_DIMENSION = 8

    def __init__(self, feature_dimension: int, hidden_dimension: int = 96) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(
            4 * int(feature_dimension) + 3 + self.CONTEXT_DIMENSION,
            int(hidden_dimension),
        )
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
        geometric_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if geometric_context.ndim != 2 or geometric_context.shape != (
            len(query), self.CONTEXT_DIMENSION,
        ):
            raise ValueError("surface-coordinate homography context differs")
        if not torch.isfinite(geometric_context).all():
            raise ValueError("surface-coordinate homography context differs")
        hidden = F.gelu(self.hidden(
            torch.cat((_pair_input(query, mapping, token_ids), geometric_context), dim=1),
        ))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(self.chart_uv_mean(hidden))
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(self.chart_uv_log_variance(hidden))
        return mean, variance, chart_uv_mean, chart_uv_variance, self.match(hidden)


class MappingSurfaceCoordinateLocalCorrelationHead(torch.nn.Module):
    """Surface-coordinate posterior with geometry and a local RADIO cost volume.

    The first eight values retain the frozen V11 geometric/homography contract.
    The remaining 99 values are a candidate-conditioned 3x3 query by 3x3 map
    cosine volume followed by the two nine-element validity masks.  Candidate
    identity therefore remains explicit; the head only predicts a bounded
    residual inside that candidate's original metric cell.
    """

    GEOMETRIC_CONTEXT_DIMENSION = 8
    LOCAL_CORRELATION_DIMENSION = 9 * 9 + 9 + 9
    CONTEXT_DIMENSION = GEOMETRIC_CONTEXT_DIMENSION + LOCAL_CORRELATION_DIMENSION

    def __init__(self, feature_dimension: int, hidden_dimension: int = 96) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(
            4 * int(feature_dimension) + 3 + self.CONTEXT_DIMENSION,
            int(hidden_dimension),
        )
        self.mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.chart_uv_mean = torch.nn.Linear(int(hidden_dimension), 2)
        self.chart_uv_log_variance = torch.nn.Linear(int(hidden_dimension), 1)
        self.match = torch.nn.Linear(int(hidden_dimension), 1)

    def forward(
        self, query: torch.Tensor, mapping: torch.Tensor, token_ids: torch.Tensor,
        geometric_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if geometric_context.ndim != 2 or geometric_context.shape != (
            len(query), self.CONTEXT_DIMENSION,
        ):
            raise ValueError("surface-coordinate local-correlation context differs")
        if not torch.isfinite(geometric_context).all():
            raise ValueError("surface-coordinate local-correlation context differs")
        hidden = F.gelu(self.hidden(
            torch.cat((_pair_input(query, mapping, token_ids), geometric_context), dim=1),
        ))
        mean = SUBTOKEN_HALF_EXTENT_PX * torch.tanh(self.mean(hidden))
        variance = MINIMUM_VARIANCE_PX2 + F.softplus(self.log_variance(hidden))
        chart_uv_mean = MAXIMUM_CHART_UV_OFFSET_M * torch.tanh(self.chart_uv_mean(hidden))
        chart_uv_variance = MINIMUM_UV_VARIANCE_M2 + F.softplus(self.chart_uv_log_variance(hidden))
        return mean, variance, chart_uv_mean, chart_uv_variance, self.match(hidden)


def _metrics(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, np.float64); mean = np.asarray(mean, np.float64)
    error = np.linalg.norm(mean - target, axis=1)
    centre = np.linalg.norm(target, axis=1)
    sigma = np.sqrt(np.asarray(variance, np.float64).reshape(-1))
    order = np.argsort(sigma, kind="stable")
    quartile = np.array_split(order, 4)
    return {
        "pair_count": int(len(target)),
        "token_centre_error_px": {
            "median": float(np.median(centre)), "p90": float(np.quantile(centre, 0.9)),
            "mean": float(np.mean(centre)),
        },
        "predicted_error_px": {
            "median": float(np.median(error)), "p90": float(np.quantile(error, 0.9)),
            "mean": float(np.mean(error)),
        },
        "predicted_better_fraction": float(np.mean(error < centre)),
        "relative_median_improvement": float(1.0 - np.median(error) / max(np.median(centre), 1e-12)),
        "uncertainty_quantile_mean_error_px": [
            float(np.mean(error[part])) for part in quartile if len(part)
        ],
    }


def _state_arrays(model: MappingSubtokenHead) -> dict[str, np.ndarray]:
    return {
        name.replace(".", "__"): value.detach().cpu().numpy().astype(np.float32)
        for name, value in model.state_dict().items()
    }


def load_mapping_subtoken_head(
    path: Path,
) -> tuple[
    MappingSubtokenHead | MappingSurfaceCoordinateHead | MappingSurfaceCoordinateMixtureHead
    | MappingSurfaceCoordinateContextHead | MappingSurfaceCoordinateDeepContextHead
    | MappingSurfaceCoordinateHomographyContextHead | MappingSurfaceCoordinateLocalCorrelationHead,
    dict[str, object],
]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
    content = dict(metadata); claimed = content.pop("content_sha256", None)
    if (
        metadata.get("artifact_type") not in {
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v1",
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v2",
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v3",
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v4",
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v5",
            "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v6",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_head_v7",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_mixture_head_v8",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_context_head_v9",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_deep_context_head_v10",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11",
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_local_correlation_head_v12",
        }
        or metadata.get("query_rgb_pose_depth_or_ground_truth_read") is not False
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
        or canonical_json_sha256(content) != claimed
    ):
        raise ValueError("mapping subtoken head lineage differs")
    if metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_local_correlation_head_v12":
        model = MappingSurfaceCoordinateLocalCorrelationHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    elif metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11":
        model = MappingSurfaceCoordinateHomographyContextHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    elif metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_deep_context_head_v10":
        model = MappingSurfaceCoordinateDeepContextHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    elif metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_context_head_v9":
        model = MappingSurfaceCoordinateContextHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    elif metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_mixture_head_v8":
        model = MappingSurfaceCoordinateMixtureHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
            int(metadata.get("chart_uv_mixture_modes", -1)),
        )
    elif metadata.get("artifact_type") == "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_head_v7":
        model = MappingSurfaceCoordinateHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    else:
        model = MappingSubtokenHead(
            int(metadata["feature_dimension"]), int(metadata["hidden_dimension"]),
        )
    state = {name.replace("__", "."): torch.from_numpy(value) for name, value in arrays.items()}
    model.load_state_dict(state, strict=True); model.eval()
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_bank", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--radio_projection", type=Path, required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation_route", default="seq9")
    parser.add_argument("--hidden_dimension", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260903)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite mapping subtoken head")

    visibility, visibility_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    bank, bank_meta = _load_observation_bank(args.observation_bank)
    projection, projection_meta = _load_projection(args.radio_projection)
    if (
        bank_meta.get("visibility_atlas_content_sha256") != visibility_meta.get("content_sha256")
        or projection_meta.get("observation_bank_content_sha256") != bank_meta.get("content_sha256")
        or visibility_meta.get("planar_map_file_sha256") != file_sha256(args.planar_map)
    ):
        raise ValueError("mapping-only subtoken lineage differs")

    offsets = np.asarray(bank["observation_offsets"], np.int64)
    world = np.asarray(bank["world_points"], np.float64)
    token_ids = np.asarray(bank["token_ids"], np.int64)
    observation = np.repeat(np.arange(len(visibility.view_names)), np.diff(offsets))
    observation_plane = np.repeat(np.arange(len(planes.plane_ids)), np.diff(visibility.plane_offsets))
    plane = observation_plane[observation]
    uv = np.empty((len(world), 2), np.float64)
    for row in range(len(planes.plane_ids)):
        selected = np.flatnonzero(plane == row)
        uv[selected] = (world[selected] - planes.centers_world[row]) @ planes.frames_world[row, :2].T
    cell = np.floor(uv / 0.5).astype(np.int64)
    _, identity = np.unique(np.c_[plane, cell], axis=0, return_inverse=True)
    route_per_observation = np.asarray([
        str(name).split("__", 1)[0] for name in visibility.view_names.astype(str)
    ])
    all_routes = set(route_per_observation.tolist())
    fit_routes = all_routes - {str(args.validation_route)}
    if str(args.validation_route) not in all_routes or not fit_routes:
        raise ValueError("validation route does not form a mapping-only split")

    unique_names = sorted(set(visibility.view_names.astype(str).tolist()))
    camera_matrices = np.empty((len(visibility.view_names), 3, 3), np.float64)
    radial_coefficients = np.empty(len(visibility.view_names), np.float64)
    contributor_rows = []
    camera_contract = []
    for name in unique_names:
        path = args.mapping_contributors / name
        _, pose, matrix, k1 = _load_contributor_geometry(path)
        rows = np.flatnonzero(visibility.view_names.astype(str) == name)
        if not np.allclose(visibility.poses_w2c[rows], pose, atol=1e-12, rtol=0.0):
            raise ValueError("mapping contributor camera contract differs")
        camera_matrices[rows] = matrix
        radial_coefficients[rows] = float(k1)
        checksum = file_sha256(path)
        contributor_rows.append((name, checksum))
        camera_contract.append((name, checksum))
    contributor_inventory_sha = hashlib.sha256(
        json.dumps(contributor_rows, separators=(",", ":")).encode()
    ).hexdigest()

    def build_split(routes: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected_obs = np.isin(route_per_observation, sorted(routes))
        representatives, rep_identity = _representative_rows(identity, observation, uv, cell, selected_obs)
        query, mapping = _directed_pairs(representatives, rep_identity)
        pixel, depth = _project_world_to_pixel(
            world[mapping], visibility.poses_w2c[observation[query]],
            camera_matrices[observation[query]], radial_coefficients[observation[query]],
        )
        target = pixel - _token_centres(token_ids[query])
        distance = np.linalg.norm(world[query] - world[mapping], axis=1)
        valid = (
            (depth > 0.0) & np.isfinite(target).all(axis=1)
            & (np.max(np.abs(target), axis=1) < SUBTOKEN_HALF_EXTENT_PX)
            & (distance <= MAXIMUM_POSITIVE_WORLD_DISTANCE_M)
        )
        return query[valid], mapping[valid], target[valid].astype(np.float32)

    fit_q, fit_m, fit_target = build_split(fit_routes)
    val_q, val_m, val_target = build_split({str(args.validation_route)})
    if min(len(fit_q), len(val_q)) < int(args.batch_size):
        raise ValueError("insufficient mapping-only subtoken pairs")

    # Hard local null examples: next metric cell on the same plane, within the same split.
    def negative_rows(query_rows: np.ndarray, routes: set[str]) -> tuple[np.ndarray, np.ndarray]:
        selected_obs = np.isin(route_per_observation, sorted(routes))
        representatives, rep_identity = _representative_rows(identity, observation, uv, cell, selected_obs)
        unique_rep_identity, first = np.unique(rep_identity, return_index=True)
        representative = representatives[first]
        identity_plane = plane[representative]
        lookup: dict[int, int] = {}
        for p in np.unique(identity_plane):
            ids = unique_rep_identity[identity_plane == p]
            for index, value in enumerate(ids.tolist()):
                lookup[int(value)] = int(ids[(index + 1) % len(ids)]) if len(ids) > 1 else int(value)
        row_for_identity = {int(key): int(value) for key, value in zip(unique_rep_identity, representative)}
        negative = np.asarray([row_for_identity[lookup[int(identity[row])]] for row in query_rows], np.int64)
        keep = identity[negative] != identity[query_rows]
        return query_rows[keep], negative[keep]

    fit_nq, fit_nm = negative_rows(fit_q, fit_routes)
    val_nq, val_nm = negative_rows(val_q, {str(args.validation_route)})

    # Only materialise the already learned 64-D representation, not another RGB/view map.
    selected_rows = np.unique(np.concatenate((fit_q, fit_m, fit_nq, fit_nm, val_q, val_m, val_nq, val_nm)))
    projected_selected = np.asarray(bank["radio_features"][selected_rows], np.float32) @ projection.T
    projected_selected /= np.maximum(np.linalg.norm(projected_selected, axis=1, keepdims=True), 1e-8)
    feature = np.zeros((len(world), projection.shape[0]), np.float16)
    feature[selected_rows] = projected_selected.astype(np.float16)

    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MappingSubtokenHead(projection.shape[0], int(args.hidden_dimension)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = np.random.default_rng(int(args.seed))
    losses = []
    for step in range(int(args.steps)):
        pos = generator.integers(0, len(fit_q), size=int(args.batch_size))
        neg = generator.integers(0, len(fit_nq), size=int(args.batch_size))
        q = torch.from_numpy(feature[fit_q[pos]].astype(np.float32)).to(device)
        m = torch.from_numpy(feature[fit_m[pos]].astype(np.float32)).to(device)
        token = torch.from_numpy(token_ids[fit_q[pos]]).to(device)
        target = torch.from_numpy(fit_target[pos]).to(device)
        mean, variance, positive_logit = model(q, m, token)
        nq = torch.from_numpy(feature[fit_nq[neg]].astype(np.float32)).to(device)
        nm = torch.from_numpy(feature[fit_nm[neg]].astype(np.float32)).to(device)
        ntoken = torch.from_numpy(token_ids[fit_nq[neg]]).to(device)
        _, _, negative_logit = model(nq, nm, ntoken)
        error2 = torch.sum((mean - target) ** 2, dim=1, keepdim=True)
        nll = (0.5 * error2 / variance + torch.log(variance)).mean()
        robust = F.smooth_l1_loss(mean, target, beta=0.25)
        classification = 0.5 * (
            F.binary_cross_entropy_with_logits(positive_logit, torch.ones_like(positive_logit))
            + F.binary_cross_entropy_with_logits(negative_logit, torch.zeros_like(negative_logit))
        )
        loss = nll + 0.5 * robust + classification
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{args.steps} loss={np.mean(losses[-200:]):.6f}", flush=True)

    def evaluate(qrow: np.ndarray, mrow: np.ndarray, target: np.ndarray,
                 nqrow: np.ndarray, nmrow: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        model.eval(); means=[]; variances=[]; positive=[]; negative=[]
        with torch.no_grad():
            for start in range(0, len(qrow), 8192):
                stop = min(start + 8192, len(qrow))
                q = torch.from_numpy(feature[qrow[start:stop]].astype(np.float32)).to(device)
                m = torch.from_numpy(feature[mrow[start:stop]].astype(np.float32)).to(device)
                token = torch.from_numpy(token_ids[qrow[start:stop]]).to(device)
                mean, variance, logit = model(q, m, token)
                means.append(mean.cpu().numpy()); variances.append(variance.cpu().numpy()); positive.append(logit.cpu().numpy())
            for start in range(0, len(nqrow), 8192):
                stop = min(start + 8192, len(nqrow))
                q = torch.from_numpy(feature[nqrow[start:stop]].astype(np.float32)).to(device)
                m = torch.from_numpy(feature[nmrow[start:stop]].astype(np.float32)).to(device)
                token = torch.from_numpy(token_ids[nqrow[start:stop]]).to(device)
                negative.append(model(q, m, token)[2].cpu().numpy())
        mean = np.concatenate(means); variance = np.concatenate(variances)
        result = _metrics(target, mean, variance)
        result["positive_match_probability_mean"] = float(np.mean(torch.sigmoid(torch.from_numpy(np.concatenate(positive))).numpy()))
        result["negative_match_probability_mean"] = float(np.mean(torch.sigmoid(torch.from_numpy(np.concatenate(negative))).numpy()))
        return result, mean, variance

    fit_metrics, _, _ = evaluate(fit_q, fit_m, fit_target, fit_nq, fit_nm)
    validation_metrics, _, _ = evaluate(val_q, val_m, val_target, val_nq, val_nm)
    gate = bool(
        validation_metrics["relative_median_improvement"] >= 0.05
        and validation_metrics["predicted_error_px"]["p90"]
        <= validation_metrics["token_centre_error_px"]["p90"]
        and validation_metrics["predicted_better_fraction"] > 0.5
    )
    arrays = _state_arrays(model)
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v1",
        "feature_dimension": int(projection.shape[0]),
        "hidden_dimension": int(args.hidden_dimension),
        "coordinate_target": "map_observation_metric_point_projected_into_distinct_query_like_mapping_view_minus_RADIO_token_center",
        "coordinate_output": "continuous_raw_image_pixel_offset_bounded_to_original_4x4_RADIO_token",
        "uncertainty_output": "isotropic_centroid_measurement_variance_px2_not_map_surface_footprint",
        "null_output": "pair_match_probability_trained_with_same_plane_different_metric_cell_negatives",
        "fit_mapping_routes": sorted(fit_routes),
        "validation_mapping_route": str(args.validation_route),
        "fit_validation_route_disjoint": True,
        "query_rgb_pose_depth_or_ground_truth_read": False,
        "mapping_rgb_read_or_stored": False,
        "source_view_identity_retained_at_runtime": False,
        "metric_cell_size_m": 0.5,
        "maximum_positive_world_distance_m": MAXIMUM_POSITIVE_WORLD_DISTANCE_M,
        "subtoken_half_extent_px": SUBTOKEN_HALF_EXTENT_PX,
        "minimum_variance_px2": MINIMUM_VARIANCE_PX2,
        "fit_positive_pair_count": int(len(fit_q)),
        "validation_positive_pair_count": int(len(val_q)),
        "fit_null_pair_count": int(len(fit_nq)),
        "validation_null_pair_count": int(len(val_nq)),
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "mapping_validation_gate_definition": "median_relative_improvement>=5pct_AND_p90_nonincrease_AND_better_fraction>0.5",
        "mapping_validation_gate_pass": gate,
        "steps": int(args.steps), "batch_size": int(args.batch_size), "seed": int(args.seed),
        "observation_bank_file_sha256": file_sha256(args.observation_bank),
        "observation_bank_content_sha256": bank_meta.get("content_sha256"),
        "visibility_atlas_file_sha256": file_sha256(args.visibility_atlas),
        "visibility_atlas_content_sha256": visibility_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "radio_projection_file_sha256": file_sha256(args.radio_projection),
        "radio_projection_content_sha256": projection_meta.get("content_sha256"),
        "mapping_contributor_inventory_sha256": contributor_inventory_sha,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
