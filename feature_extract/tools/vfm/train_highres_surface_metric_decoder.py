"""Train a stride-4 RADIO+RGB metric decoder on cross-view 2DGS texels."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.tools.vfm.localize_2dgs_surface_queries import _load_raw_final
from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import load_gaussian_vfm_source_from_ply
from feature_extract.vfm.localization.continuous_surface_alignment import (
    _left_pose_vector_step,
    project_world_points,
)
from feature_extract.vfm.localization.highres_surface_metric_decoder import (
    HighresSurfaceMetricDecoder,
    HighresSurfaceMetricDecoderConfig,
    load_highres_surface_metric_decoder,
    save_highres_surface_metric_decoder,
)
from feature_extract.vfm.vfm_2dgs_mapping import _surface_tangent_axes_and_scales


class InsufficientGeometricOverlap(RuntimeError):
    """Raised when a nominal texel pair has too few mutually visible points."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_samples", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--mapping_camera_manifest", required=True)
    parser.add_argument("--mapping_depth_bank", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--initial_checkpoint", default="")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--samples_per_pair", type=int, default=64)
    parser.add_argument("--minimum_pair_overlap", type=int, default=8)
    parser.add_argument("--maximum_tangent_pair_distance", type=float, default=0.20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--validation_every", type=int, default=50)
    parser.add_argument("--validation_pairs", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _decode_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _sample(
    feature: torch.Tensor,
    xy: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    normalize: bool = True,
) -> torch.Tensor:
    grid = torch.stack(
        [
            2.0 * (xy[..., 0] + 0.5) / max(int(image_width), 1) - 1.0,
            2.0 * (xy[..., 1] + 0.5) / max(int(image_height), 1) - 1.0,
        ],
        dim=-1,
    )
    value = F.grid_sample(
        feature,
        grid.reshape(int(feature.shape[0]), 1, -1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[:, :, 0, :].transpose(1, 2)
    return F.normalize(value, dim=-1) if bool(normalize) else value


def _pair_index(
    cells: np.ndarray,
    views: np.ndarray,
    tangent_uv: np.ndarray,
    *,
    minimum_overlap: int,
    maximum_tangent_distance: float,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    by_cell: dict[int, list[int]] = defaultdict(list)
    for row, cell in enumerate(cells.tolist()):
        by_cell[int(cell)].append(int(row))
    by_pair: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for rows in by_cell.values():
        best_by_view: dict[int, int] = {}
        for row in rows:
            best_by_view.setdefault(int(views[row]), int(row))
        ordered = sorted(best_by_view)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                by_pair[(left, right)].append(
                    (best_by_view[left], best_by_view[right])
                )
    output = []
    for (left, right), pairs in sorted(by_pair.items()):
        if len(pairs) < int(minimum_overlap):
            continue
        array = np.asarray(
            [
                pair
                for pair in pairs
                if np.linalg.norm(tangent_uv[pair[0]] - tangent_uv[pair[1]])
                <= float(maximum_tangent_distance)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        if array.shape[0] < int(minimum_overlap):
            continue
        output.append((left, right, array[:, 0], array[:, 1]))
    return output


def _view_batch(
    view_ids: tuple[int, int],
    *,
    image_ids: list[str],
    token_paths: list[str],
    image_root: Path,
    cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for view in view_ids:
        if int(view) not in cache:
            cache[int(view)] = (
                _load_raw_final(Path(token_paths[int(view)]), "radio_final"),
                _decode_rgb(image_root / image_ids[int(view)]),
            )
            while len(cache) > 12:
                cache.popitem(last=False)
        cache.move_to_end(int(view))
        values.append(cache[int(view)])
    radio = torch.as_tensor(
        np.stack([value[0] for value in values]),
        dtype=torch.float32,
        device=device,
    )
    rgb = torch.as_tensor(
        np.stack([value[1].transpose(2, 0, 1) for value in values]),
        dtype=torch.float32,
        device=device,
    )
    return radio, rgb


def _sample_depth_map(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, depth.shape[0] - 1)
    return np.asarray(depth[y, x], dtype=np.float32)


def _loss_and_metrics(
    model: HighresSurfaceMetricDecoder,
    radio: torch.Tensor,
    rgb: torch.Tensor,
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    right_pose_xy: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(radio, rgb)
    fine = output["fine"]
    height, width = int(rgb.shape[2]), int(rgb.shape[3])
    left = _sample(
        fine[0:1], left_xy[None], image_width=width, image_height=height
    )[0]
    right = _sample(
        fine[1:2], right_xy[None], image_width=width, image_height=height
    )[0]
    identity_logits = left @ right.T / 0.07
    labels = torch.arange(left.shape[0], device=left.device)
    identity_loss = 0.5 * (
        F.cross_entropy(identity_logits, labels)
        + F.cross_entropy(identity_logits.T, labels)
    )
    offsets = left.new_tensor(
        [
            [0, 0],
            [-0.5, 0],
            [0.5, 0],
            [0, -0.5],
            [0, 0.5],
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
            [-4, 0],
            [4, 0],
            [0, -4],
            [0, 4],
            [-4, -4],
            [4, -4],
            [-4, 4],
            [4, 4],
            [-8, 0],
            [8, 0],
            [0, -8],
            [0, 8],
        ]
    )
    right_candidates = _sample(
        fine[1:2],
        (right_xy[:, None, :] + offsets[None]).reshape(1, -1, 2),
        image_width=width,
        image_height=height,
    )[0].reshape(right_xy.shape[0], offsets.shape[0], -1)
    left_candidates = _sample(
        fine[0:1],
        (left_xy[:, None, :] + offsets[None]).reshape(1, -1, 2),
        image_width=width,
        image_height=height,
    )[0].reshape(left_xy.shape[0], offsets.shape[0], -1)
    right_local = torch.sum(left[:, None, :] * right_candidates, dim=2) / 0.05
    left_local = torch.sum(right[:, None, :] * left_candidates, dim=2) / 0.05
    local_labels = torch.zeros(left.shape[0], dtype=torch.long, device=left.device)
    local_loss = 0.5 * (
        F.cross_entropy(right_local, local_labels)
        + F.cross_entropy(left_local, local_labels)
    )
    self_right_local = (
        torch.sum(right[:, None, :] * right_candidates, dim=2) / 0.05
    )
    self_left_local = (
        torch.sum(left[:, None, :] * left_candidates, dim=2) / 0.05
    )
    self_local_loss = 0.5 * (
        F.cross_entropy(self_right_local, local_labels)
        + F.cross_entropy(self_left_local, local_labels)
    )
    distances = torch.linalg.norm(offsets, dim=1)
    # A smooth near-pixel ramp is essential: supervising only whole stride-4
    # cells leaves no useful derivative inside a cell.
    required_margin = torch.clamp(distances / 8.0, max=1.0) * 0.25
    positive_right = right_local[:, :1] * 0.05
    positive_left = left_local[:, :1] * 0.05
    ranking_loss = 0.5 * (
        F.relu(required_margin[None, 1:] - positive_right + right_local[:, 1:] * 0.05).mean()
        + F.relu(required_margin[None, 1:] - positive_left + left_local[:, 1:] * 0.05).mean()
    )
    self_ranking_loss = 0.5 * (
        F.relu(
            required_margin[None, 1:]
            - self_right_local[:, :1] * 0.05
            + self_right_local[:, 1:] * 0.05
        ).mean()
        + F.relu(
            required_margin[None, 1:]
            - self_left_local[:, :1] * 0.05
            + self_left_local[:, 1:] * 0.05
        ).mean()
    )
    matchability = output["matchability"]
    match_left = _sample(
        matchability[0:1],
        left_xy[None],
        image_width=width,
        image_height=height,
        normalize=False,
    )[0, :, 0]
    match_right = _sample(
        matchability[1:2],
        right_xy[None],
        image_width=width,
        image_height=height,
        normalize=False,
    )[0, :, 0]
    match_loss = -0.5 * (
        torch.log(torch.clamp(match_left, min=1e-6)).mean()
        + torch.log(torch.clamp(match_right, min=1e-6)).mean()
    )
    pose_candidates = _sample(
        fine[1:2],
        right_pose_xy.reshape(1, -1, 2),
        image_width=width,
        image_height=height,
    )[0].reshape(right_pose_xy.shape[0], right_pose_xy.shape[1], -1)
    pose_logits = torch.mean(
        torch.sum(left[None] * pose_candidates, dim=2), dim=1
    ) / 0.05
    pose_labels = torch.zeros((1,), dtype=torch.long, device=left.device)
    pose_nll = F.cross_entropy(pose_logits[None], pose_labels)
    pose_similarity = pose_logits * 0.05
    pose_ranking = F.relu(
        0.10 - pose_similarity[:1] + pose_similarity[1:]
    ).mean()
    loss = (
        0.5 * identity_loss
        + 3.0 * local_loss
        + 1.0 * ranking_loss
        + 1.5 * self_local_loss
        + 0.5 * self_ranking_loss
        + 1.5 * pose_nll
        + 1.0 * pose_ranking
        + 0.05 * match_loss
    )
    right_order = torch.argsort(right_local, dim=1, descending=True)
    left_order = torch.argsort(left_local, dim=1, descending=True)
    right_rank = torch.argmax((right_order == 0).to(torch.int64), dim=1) + 1
    left_rank = torch.argmax((left_order == 0).to(torch.int64), dim=1) + 1
    metrics = {
        "loss": float(loss.detach().cpu()),
        "identity_r1": float(
            torch.mean((torch.argmax(identity_logits, dim=1) == labels).float())
            .detach()
            .cpu()
        ),
        "local_r1": float(
            0.5
            * (
                torch.mean((torch.argmax(right_local, dim=1) == 0).float())
                + torch.mean((torch.argmax(left_local, dim=1) == 0).float())
            ).detach().cpu()
        ),
        "local_mean_rank": float(
            0.5 * (right_rank.float().mean() + left_rank.float().mean())
        ),
        "local_center_probability": float(
            0.5
            * (
                torch.softmax(right_local, dim=1)[:, 0].mean()
                + torch.softmax(left_local, dim=1)[:, 0].mean()
            )
        ),
        "pose_r1": float(
            (torch.argmax(pose_logits) == 0).float().detach().cpu()
        ),
        "pose_center_probability": float(
            torch.softmax(pose_logits, dim=0)[0].detach().cpu()
        ),
        "self_local_center_probability": float(
            0.5
            * (
                torch.softmax(self_right_local, dim=1)[:, 0].mean()
                + torch.softmax(self_left_local, dim=1)[:, 0].mean()
            ).detach().cpu()
        ),
    }
    return loss, metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint = Path(args.output_checkpoint)
    summary_path = Path(args.summary_json)
    if (checkpoint.exists() or summary_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite decoder outputs")
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    with np.load(Path(args.training_samples), allow_pickle=False) as data:
        cells = np.asarray(data["surface_cell_ids"], dtype=np.int64)
        views = np.asarray(data["view_indices"], dtype=np.int64)
        observation_xy = np.asarray(data["observation_xy"], dtype=np.float32)
        source_indices = np.asarray(data["source_indices"], dtype=np.int64)
        tangent_uv = np.asarray(data["tangent_uv"], dtype=np.float32)
        image_ids = list(json.loads(str(data["view_image_ids_json"].item())))
        token_paths = list(json.loads(str(data["view_token_paths_json"].item())))
    pairs = _pair_index(
        cells,
        views,
        tangent_uv,
        minimum_overlap=int(args.minimum_pair_overlap),
        maximum_tangent_distance=float(args.maximum_tangent_pair_distance),
    )
    train_pairs = [
        pair for pair in pairs if pair[0] % 5 != 0 and pair[1] % 5 != 0
    ]
    validation_pairs = [
        pair for pair in pairs if pair[0] % 5 == 0 and pair[1] % 5 == 0
    ]
    if not train_pairs or not validation_pairs:
        raise ValueError("training samples do not produce disjoint view pairs")
    device = torch.device(
        str(args.device)
        if torch.cuda.is_available() or not str(args.device).startswith("cuda")
        else "cpu"
    )
    model = HighresSurfaceMetricDecoder(
        HighresSurfaceMetricDecoderConfig(
            hidden_dim=int(args.hidden_dim), output_dim=int(args.output_dim)
        )
    ).to(device)
    if str(args.initial_checkpoint):
        initial_model, _initial_metadata = load_highres_surface_metric_decoder(
            Path(args.initial_checkpoint), device=str(device)
        )
        if initial_model.config != model.config:
            raise ValueError("initial checkpoint decoder config does not match")
        model.load_state_dict(initial_model.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4
    )
    cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
    history = []
    best_validation = -1.0
    best_step = 0

    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply))
    normals = np.asarray(source.normal, dtype=np.float32)[source_indices]
    tangent1, tangent2, scale1, scale2 = _surface_tangent_axes_and_scales(
        source, source_indices, normals
    )
    observation_world = (
        np.asarray(source.xyz, dtype=np.float64)[source_indices]
        + tangent_uv[:, 0:1] * scale1[:, None] * tangent1
        + tangent_uv[:, 1:2] * scale2[:, None] * tangent2
    )
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    camera_by_image, _camera_audit = _load_query_camera_manifest(
        Path(args.mapping_camera_manifest)
    )
    depth_payload = json.loads(Path(args.mapping_depth_bank).read_text())
    depth_by_image = {
        str(record["image_id"]): Path(record["path"])
        for record in depth_payload["records"]
    }
    depth_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def depth_for(image_id: str) -> np.ndarray:
        if image_id not in depth_cache:
            depth_cache[image_id] = np.load(depth_by_image[image_id], mmap_mode="r")
            while len(depth_cache) > 12:
                depth_cache.popitem(last=False)
        depth_cache.move_to_end(image_id)
        return depth_cache[image_id]

    def run_pair(pair, *, training: bool) -> dict[str, float]:
        left_view, right_view, left_rows, right_rows = pair
        left_camera = camera_by_image[image_ids[left_view]]
        right_camera = camera_by_image[image_ids[right_view]]
        count = min(int(args.samples_per_pair), int(left_rows.size))
        chosen = rng.choice(left_rows.size, size=count, replace=False)
        shared_world = observation_world[left_rows[chosen]]
        left_projected, left_depth = project_world_points(
            shared_world,
            pose_by_image[image_ids[left_view]],
            camera_by_image[image_ids[left_view]],
        )
        right_projected, right_depth = project_world_points(
            shared_world,
            pose_by_image[image_ids[right_view]],
            camera_by_image[image_ids[right_view]],
        )
        valid = (
            (left_depth > 0.0)
            & (right_depth > 0.0)
            & (left_projected[:, 0] >= 8.0)
            & (left_projected[:, 0] < float(left_camera.width - 8))
            & (left_projected[:, 1] >= 8.0)
            & (left_projected[:, 1] < float(left_camera.height - 8))
            & (right_projected[:, 0] >= 8.0)
            & (right_projected[:, 0] < float(right_camera.width - 8))
            & (right_projected[:, 1] >= 8.0)
            & (right_projected[:, 1] < float(right_camera.height - 8))
        )
        left_render_depth = _sample_depth_map(
            depth_for(image_ids[left_view]), left_projected
        )
        right_render_depth = _sample_depth_map(
            depth_for(image_ids[right_view]), right_projected
        )
        valid &= (
            np.isfinite(left_render_depth)
            & np.isfinite(right_render_depth)
            & (np.abs(left_render_depth - left_depth) <= 0.05)
            & (np.abs(right_render_depth - right_depth) <= 0.05)
        )
        if np.sum(valid) < 4:
            shared_world = observation_world[right_rows[chosen]]
            left_projected, left_depth = project_world_points(
                shared_world,
                pose_by_image[image_ids[left_view]],
                camera_by_image[image_ids[left_view]],
            )
            right_projected, right_depth = project_world_points(
                shared_world,
                pose_by_image[image_ids[right_view]],
                camera_by_image[image_ids[right_view]],
            )
            left_render_depth = _sample_depth_map(
                depth_for(image_ids[left_view]), left_projected
            )
            right_render_depth = _sample_depth_map(
                depth_for(image_ids[right_view]), right_projected
            )
            valid = (
                np.isfinite(left_render_depth)
                & np.isfinite(right_render_depth)
                & (left_depth > 0.0)
                & (right_depth > 0.0)
                & (left_projected[:, 0] >= 8.0)
                & (left_projected[:, 0] < float(left_camera.width - 8))
                & (left_projected[:, 1] >= 8.0)
                & (left_projected[:, 1] < float(left_camera.height - 8))
                & (right_projected[:, 0] >= 8.0)
                & (right_projected[:, 0] < float(right_camera.width - 8))
                & (right_projected[:, 1] >= 8.0)
                & (right_projected[:, 1] < float(right_camera.height - 8))
                & (np.abs(left_render_depth - left_depth) <= 0.05)
                & (np.abs(right_render_depth - right_depth) <= 0.05)
            )
        if np.sum(valid) < 4:
            raise InsufficientGeometricOverlap(
                f"only {int(np.sum(valid))} mutually visible samples"
            )
        left_projected = left_projected[valid]
        right_projected = right_projected[valid]
        shared_world = shared_world[valid]
        right_pose_candidates = [right_projected]
        right_pose = pose_by_image[image_ids[right_view]]
        for axis in range(6):
            direction = np.zeros((6,), dtype=np.float64)
            direction[axis] = 1.0
            magnitudes = (
                (np.deg2rad(0.5), np.deg2rad(1.0))
                if axis < 3
                else (0.05, 0.10, 0.20)
            )
            for sign in (-1.0, 1.0):
                for magnitude in magnitudes:
                    perturbed = _left_pose_vector_step(
                        right_pose,
                        direction,
                        rotation_step=sign * float(magnitude) if axis < 3 else 0.0,
                        translation_step=sign * float(magnitude) if axis >= 3 else 0.0,
                    )
                    perturbed_xy, _perturbed_depth = project_world_points(
                        shared_world, perturbed, right_camera
                    )
                    right_pose_candidates.append(perturbed_xy.astype(np.float32))
        right_pose_xy = np.stack(right_pose_candidates, axis=0)
        radio, rgb = _view_batch(
            (left_view, right_view),
            image_ids=image_ids,
            token_paths=token_paths,
            image_root=Path(args.image_root),
            cache=cache,
            device=device,
        )
        loss, metrics = _loss_and_metrics(
            model,
            radio,
            rgb,
            torch.as_tensor(left_projected, dtype=torch.float32, device=device),
            torch.as_tensor(right_projected, dtype=torch.float32, device=device),
            torch.as_tensor(right_pose_xy, dtype=torch.float32, device=device),
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        return metrics

    rejected_train_pairs = 0
    rejected_validation_pairs = 0
    for step in range(1, int(args.steps) + 1):
        model.train()
        for _attempt in range(100):
            try:
                metrics = run_pair(
                    train_pairs[int(rng.integers(len(train_pairs)))], training=True
                )
                break
            except InsufficientGeometricOverlap:
                rejected_train_pairs += 1
        else:
            raise RuntimeError("could not find a geometrically valid training pair")
        if step == 1 or step % int(args.validation_every) == 0 or step == int(args.steps):
            model.eval()
            validation = []
            with torch.no_grad():
                chosen_pairs = rng.choice(
                    len(validation_pairs),
                    size=min(int(args.validation_pairs), len(validation_pairs)),
                    replace=False,
                )
                for pair_index in chosen_pairs.tolist():
                    try:
                        validation.append(
                            run_pair(validation_pairs[pair_index], training=False)
                        )
                    except InsufficientGeometricOverlap:
                        rejected_validation_pairs += 1
            if not validation:
                raise RuntimeError("no geometrically valid validation pairs")
            validation_summary = {
                key: float(np.mean([row[key] for row in validation]))
                for key in validation[0]
            }
            row = {
                "step": step,
                "train": metrics,
                "validation": validation_summary,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            selection = (
                validation_summary["local_r1"]
                + validation_summary["local_center_probability"]
                + validation_summary["pose_r1"]
                + validation_summary["pose_center_probability"]
                + 0.10 * validation_summary["identity_r1"]
            )
            if selection > best_validation:
                best_validation = selection
                best_step = step
                save_highres_surface_metric_decoder(
                    checkpoint,
                    model,
                    {
                        "vfm_layer": "radio_final",
                        "feature_branch": "raw_radio_final_plus_shallow_rgb",
                        "output_stride": 4,
                        "training_objectives": [
                            "cross_view_surface_texel_identity",
                            "local_displacement_nll",
                            "monotonic_displacement_ranking",
                            "within_view_spatial_sharpness",
                            "se3_pose_ranking",
                        ],
                        "uses_alike_descriptors": False,
                        "uses_radio_intermediate": False,
                        "uses_sfm_points": False,
                        "uses_sfm_tracks": False,
                        "stores_mapping_rgb": False,
                        "best_step": best_step,
                        "best_validation_selection": best_validation,
                    },
                )
    summary = {
        "stage": "train_highres_surface_metric_decoder",
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "best_step": best_step,
        "best_validation_selection": best_validation,
        "rejected_train_pairs": rejected_train_pairs,
        "rejected_validation_pairs": rejected_validation_pairs,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
