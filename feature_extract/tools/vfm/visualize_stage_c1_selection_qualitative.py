"""Generate Stage C1 qualitative comparisons for raw/compressed/selected VFM maps."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("matrix must have shape (N, C)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms.reshape(-1) > float(eps)
    normalized = values / np.maximum(norms, float(eps))
    return normalized.astype(np.float32, copy=False), valid


def compute_similarity_margin_map(
    query_feature_map: np.ndarray,
    landmark_features: np.ndarray,
    block_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Return top1-minus-top2 cosine margin for each query token."""

    feature_map = np.asarray(query_feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape (C, H, W)")
    channels, height, width = feature_map.shape
    landmarks = np.asarray(landmark_features, dtype=np.float32)
    if landmarks.ndim != 2:
        raise ValueError("landmark_features must have shape (N, C)")
    if landmarks.shape[1] != channels:
        raise ValueError("query and landmark feature dimensions must match")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    if landmarks.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)

    query = feature_map.reshape(channels, height * width).T.astype(np.float32, copy=False)
    query, valid_query = _normalize_rows(query)
    landmarks, valid_landmarks = _normalize_rows(landmarks)
    landmarks = landmarks[valid_landmarks]
    margins = np.zeros((query.shape[0],), dtype=np.float32)
    if query.shape[0] == 0 or landmarks.shape[0] < 2:
        return margins.reshape(height, width)

    requested = str(device).lower()
    if requested != "cpu":
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Torch is required for non-CPU margin computation") from exc
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested != "cpu":
            torch_device = torch.device(requested)
            landmark_tensor = torch.as_tensor(landmarks, dtype=torch.float32, device=torch_device).T.contiguous()
            for start in range(0, query.shape[0], int(block_size)):
                end = min(start + int(block_size), query.shape[0])
                query_tensor = torch.as_tensor(query[start:end], dtype=torch.float32, device=torch_device)
                scores = torch.matmul(query_tensor, landmark_tensor)
                top_scores = torch.topk(scores, k=2, dim=1, largest=True, sorted=True).values
                margins[start:end] = (top_scores[:, 0] - top_scores[:, 1]).detach().cpu().numpy().astype(np.float32)
            margins[~valid_query] = 0.0
            return margins.reshape(height, width)

    for start in range(0, query.shape[0], int(block_size)):
        end = min(start + int(block_size), query.shape[0])
        scores = query[start:end] @ landmarks.T
        top2_idx = np.argpartition(-scores, kth=1, axis=1)[:, :2]
        top2 = np.take_along_axis(scores, top2_idx, axis=1)
        top2.sort(axis=1)
        margins[start:end] = top2[:, 1] - top2[:, 0]
    margins[~valid_query] = 0.0
    return margins.reshape(height, width)


def group_energy_from_transform(transform_matrix: np.ndarray, group_size: int = 64) -> np.ndarray:
    """Sum squared projection weights over contiguous raw-channel groups."""

    matrix = np.asarray(transform_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("transform_matrix must have shape (input_dim, output_dim)")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    energies = []
    for start in range(0, matrix.shape[0], int(group_size)):
        group = matrix[start : start + int(group_size)]
        energies.append(float(np.sum(group * group)))
    return np.asarray(energies, dtype=np.float32)


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for Stage C1 qualitative visualization") from exc
    return cv2


def _draw_label(image: np.ndarray, label: str, y: int = 20) -> None:
    cv2 = _require_cv2()
    canvas = image
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, min(canvas.shape[0] - 1, y + 8)), (0, 0, 0), -1)
    cv2.putText(canvas, str(label), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def make_contact_sheet(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    columns: int,
    cell_width: int,
    cell_height: int,
    background: int = 18,
) -> np.ndarray:
    """Return an RGB contact sheet with labels drawn inside each fixed cell."""

    cv2 = _require_cv2()
    if int(columns) <= 0:
        raise ValueError("columns must be positive")
    if len(panels) != len(labels):
        raise ValueError("panels and labels must have the same length")
    if not panels:
        return np.full((int(cell_height), int(cell_width), 3), int(background), dtype=np.uint8)
    rows = int(np.ceil(len(panels) / float(columns)))
    sheet = np.full((rows * int(cell_height), int(columns) * int(cell_width), 3), int(background), dtype=np.uint8)
    for panel_idx, (panel, label) in enumerate(zip(panels, labels)):
        row = panel_idx // int(columns)
        col = panel_idx % int(columns)
        image = np.asarray(panel, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("each panel must have shape (H, W, 3)")
        resized = cv2.resize(image, (int(cell_width), int(cell_height)), interpolation=cv2.INTER_AREA)
        _draw_label(resized, str(label))
        y0 = row * int(cell_height)
        x0 = col * int(cell_width)
        sheet[y0 : y0 + int(cell_height), x0 : x0 + int(cell_width)] = resized
    return sheet


def render_margin_heatmap(
    margin_map: np.ndarray,
    image_rgb: np.ndarray,
    alpha: float = 0.55,
    percentile: float = 99.0,
) -> np.ndarray:
    cv2 = _require_cv2()
    image = np.asarray(image_rgb, dtype=np.uint8)
    margins = np.asarray(margin_map, dtype=np.float32)
    if margins.ndim != 2:
        raise ValueError("margin_map must have shape (H, W)")
    finite = margins[np.isfinite(margins)]
    if finite.size == 0:
        scaled = np.zeros_like(margins, dtype=np.uint8)
    else:
        high = float(np.percentile(finite, float(percentile)))
        if high <= 1e-8:
            high = float(np.max(finite)) if float(np.max(finite)) > 1e-8 else 1.0
        scaled = np.asarray(np.clip(margins / high, 0.0, 1.0) * 255.0, dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.resize(heat_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 1.0 - float(alpha), heat_rgb, float(alpha), 0.0)


def render_group_energy_heatmap(
    series: Sequence[tuple[str, np.ndarray]],
    group_size: int = 64,
    cell_width: int = 36,
) -> np.ndarray:
    cv2 = _require_cv2()
    if not series:
        return np.zeros((80, 400, 3), dtype=np.uint8)
    group_count = max(int(np.asarray(energy).size) for _name, energy in series)
    width = max(720, 180 + group_count * int(cell_width))
    row_height = 82
    height = 44 + len(series) * row_height
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"Stage C1 projection/channel-group energy (group size={int(group_size)})",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    global_max = max(float(np.max(np.asarray(energy))) for _name, energy in series if np.asarray(energy).size)
    global_max = max(global_max, 1e-6)
    for row_idx, (name, energy) in enumerate(series):
        values = np.asarray(energy, dtype=np.float32).reshape(-1)
        y0 = 50 + row_idx * row_height
        cv2.putText(canvas, name, (16, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
        if values.size == 0:
            continue
        local = values / global_max
        for group_idx, value in enumerate(local):
            x0 = 176 + group_idx * int(cell_width)
            x1 = x0 + int(cell_width) - 3
            color_value = int(np.clip(value, 0.0, 1.0) * 255.0)
            color = cv2.applyColorMap(np.asarray([[color_value]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
            color_rgb = tuple(int(v) for v in color[::-1].tolist())
            cv2.rectangle(canvas, (x0, y0), (x1, y0 + 32), color_rgb, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y0 + 32), (80, 80, 80), 1)
            if group_idx % 2 == 0:
                cv2.putText(
                    canvas,
                    str(group_idx),
                    (x0 + 2, y0 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (80, 80, 80),
                    1,
                    cv2.LINE_AA,
                )
        top = np.argsort(-values)[: min(4, values.size)]
        top_text = "top groups: " + ", ".join(f"{int(idx)}:{float(values[idx]):.2g}" for idx in top)
        cv2.putText(canvas, top_text, (176, y0 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _safe_query_name(query_id: str) -> str:
    safe = query_id.replace("/", "__").replace("\\", "__").replace(" ", "_")
    for suffix in (".png", ".jpg", ".jpeg"):
        if safe.lower().endswith(suffix):
            safe = safe[: -len(suffix)]
            break
    return safe.replace(".", "_")


def _read_image_rgb(path: Path) -> np.ndarray:
    cv2 = _require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise ValueError(f"failed to write image: {path}")


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class MethodSpec:
    name: str
    query_manifest: Path
    landmark_bank: Path
    transform: Path | None = None


@dataclass(frozen=True)
class ScenePreset:
    name: str
    track_observations: Path
    visibility_index: Path
    query_pose_file: Path
    candidate_bank: Path
    image_root: Path
    methods: tuple[MethodSpec, ...]


def _stage_c1_scene_preset(scene: str) -> ScenePreset:
    key = scene.lower()
    if key == "shopfacade":
        return ScenePreset(
            name="shopfacade",
            track_observations=Path("output/vfm/colmap_tracks/ShopFacade/model_train_tracks_min2_head100k_v1.jsonl"),
            visibility_index=Path("output/vfm/colmap_tracks/ShopFacade/model_train_full_visibility_min2_v1.npz"),
            query_pose_file=Path("/hy-tmp/Cambridge_stdloc/ShopFacade/dataset_test.txt"),
            candidate_bank=Path("output/vfm/candidate_banks/shopfacade_reference_pose_top10_fixed.jsonl"),
            image_root=Path("/hy-tmp/Cambridge_stdloc/ShopFacade"),
            methods=(
                MethodSpec(
                    "raw1280",
                    Path("output/vfm_tokens_radio/ShopFacade/test_manifest.json"),
                    Path("output/vfm/raw_landmark_banks/ShopFacade/head100k/shopfacade_radio_raw_mean_bilinear_view_head100k_l2.npz"),
                ),
                MethodSpec(
                    "random128",
                    Path("output/vfm/stage_c0_compression/shopfacade/random128/manifest.json"),
                    Path("output/vfm/stage_c0_compression/shopfacade/random128/landmarks.npz"),
                ),
                MethodSpec(
                    "learned128",
                    Path("output/vfm/stage_c1_patch_selector_final/shopfacade/learned128_seed0/test_manifest.json"),
                    Path("output/vfm/stage_c1_patch_selector_final/shopfacade/learned128_seed0/landmark_bank.npz"),
                    Path("output/vfm/stage_c1_patch_selector_final/shopfacade/learned128_seed0/transform.npz"),
                ),
                MethodSpec(
                    "group_gated128",
                    Path("output/vfm/stage_c1_group_gate/shopfacade/learned128_g64_keep50_lasso1e-4_seed0/test_manifest.json"),
                    Path("output/vfm/stage_c1_group_gate/shopfacade/learned128_g64_keep50_lasso1e-4_seed0/landmark_bank.npz"),
                    Path("output/vfm/stage_c1_group_gate/shopfacade/learned128_g64_keep50_lasso1e-4_seed0/transform.npz"),
                ),
            ),
        )
    if key == "oldhospital":
        return ScenePreset(
            name="oldhospital",
            track_observations=Path("output/vfm/colmap_tracks/OldHospital/model_train_tracks_min2_balanced300k_v2.jsonl"),
            visibility_index=Path("output/vfm/colmap_tracks/OldHospital/model_train_full_visibility_min2_v1.npz"),
            query_pose_file=Path("/hy-tmp/Cambridge_stdloc/OldHospital/dataset_test.txt"),
            candidate_bank=Path("output/vfm/candidate_banks/oldhospital_reference_pose_top10_fixed.jsonl"),
            image_root=Path("/hy-tmp/Cambridge_stdloc/OldHospital"),
            methods=(
                MethodSpec(
                    "raw1280",
                    Path("output/vfm_tokens_radio/OldHospital/test_manifest.json"),
                    Path("output/vfm/raw_landmark_banks/OldHospital/balanced300k/oldhospital_radio_raw_mean_bilinear_view_balanced300k_l2.npz"),
                ),
                MethodSpec(
                    "random128",
                    Path("output/vfm/stage_c0_compression/oldhospital/random128/manifest.json"),
                    Path("output/vfm/stage_c0_compression/oldhospital/random128/landmarks.npz"),
                ),
                MethodSpec(
                    "learned128",
                    Path("output/vfm/stage_c1_patch_selector_final/oldhospital/learned128_seed0/test_manifest.json"),
                    Path("output/vfm/stage_c1_patch_selector_final/oldhospital/learned128_seed0/landmark_bank.npz"),
                    Path("output/vfm/stage_c1_patch_selector_final/oldhospital/learned128_seed0/transform.npz"),
                ),
                MethodSpec(
                    "group_gated128",
                    Path("output/vfm/stage_c1_group_gate/oldhospital/learned128_g64_keep50_lasso1e-4_seed0/test_manifest.json"),
                    Path("output/vfm/stage_c1_group_gate/oldhospital/learned128_g64_keep50_lasso1e-4_seed0/landmark_bank.npz"),
                    Path("output/vfm/stage_c1_group_gate/oldhospital/learned128_g64_keep50_lasso1e-4_seed0/transform.npz"),
                ),
            ),
        )
    raise ValueError("scene preset must be one of: shopfacade, oldhospital")


def _parse_method_spec(value: str) -> MethodSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {3, 4} or not parts[0]:
        raise ValueError("--method must be name,query_manifest,landmark_bank[,transform]")
    transform = None if len(parts) == 3 or not parts[3] else Path(parts[3])
    return MethodSpec(parts[0], Path(parts[1]), Path(parts[2]), transform)


def _load_transform_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "matrix" not in data:
            raise ValueError(f"transform matrix not found in {path}")
        return np.asarray(data["matrix"], dtype=np.float32)


def _load_method_runtime(method: MethodSpec, xyz_by_track: dict[int, np.ndarray], reprojection_error_by_track: dict[int, float]):
    from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
    from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
    from feature_extract.vfm.tokens import TokenBankManifest

    manifest = TokenBankManifest.from_json(method.query_manifest)
    manifest.validate(verify_checksums=False)
    records = {record.image_id: record for record in manifest.records}
    bank = load_selected_track_bank_npz(method.landmark_bank)
    index = LandmarkMapIndex.from_track_bank(bank, xyz_by_track, reprojection_error_by_track)
    return records, index


def _method_label(name: str, metrics: dict[str, object]) -> str:
    translation = metrics.get("translation_error_m")
    rotation = metrics.get("rotation_error_deg")
    pose = "pose n/a" if translation is None or rotation is None else f"{float(translation):.2f}m/{float(rotation):.1f}deg"
    return (
        f"{name} | {pose} | P@1 {float(metrics.get('patch_at_1', 0.0)):.2f} | "
        f"inl {int(metrics.get('pnp_inlier_count', 0))}"
    )


def _submap_for_query(args, landmark_index, visibility_index, reference_submaps, query_id: str, gt, camera):
    from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import _limit_submap
    from feature_extract.vfm.landmark_visibility import filter_landmarks_by_visibility
    from feature_extract.vfm.patch_to_3d_matching import filter_landmarks_by_projected_visibility
    from feature_extract.vfm.query_to_3d_matching import filter_landmarks_by_reference_images

    submap = landmark_index
    if args.submap_mode == "gt_visible":
        submap = filter_landmarks_by_projected_visibility(landmark_index, gt.pose_w2c, camera)
    elif args.submap_mode == "reference_visibility":
        references = reference_submaps.get(query_id, [])
        if visibility_index is None:
            submap = filter_landmarks_by_reference_images(landmark_index, references)
        else:
            submap, _visibility_gate = filter_landmarks_by_visibility(landmark_index, visibility_index, references)
    return _limit_submap(submap, args.max_submap_landmarks)


def _visualize_query_method(args, method: MethodSpec, records, landmark_index, context, query_id: str) -> dict[str, object]:
    from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import _load_query_feature
    from feature_extract.vfm.patch_to_3d_matching import (
        PatchTo3DMatchingConfig,
        build_patch_positive_sets,
        evaluate_patch_matches,
        match_query_patches_to_landmarks,
        patch_positive_set_stats,
        patch_uncertainty_pnp_threshold,
    )
    from feature_extract.vfm.query_to_3d_matching import (
        LandmarkQualityConfig,
        estimate_pose_pnp_ransac,
        pnp_pose_error,
    )
    from feature_extract.vfm.query_to_3d_visualization import (
        render_patch_to_3d_match_overlay,
    )

    camera = context["camera"]
    gt = context["gt_by_query"].get(query_id)
    if gt is None:
        raise ValueError(f"query pose not found for {query_id}")
    if query_id not in records:
        raise ValueError(f"query_id {query_id!r} not found in manifest for {method.name}")
    record = records[query_id]
    submap = _submap_for_query(
        args,
        landmark_index,
        context["visibility_index"],
        context["reference_submaps"],
        query_id,
        gt,
        camera,
    )
    query_feature = _load_query_feature(record.token_path, args.layer_name)
    _channels, token_height, token_width = query_feature.shape
    stride_x = float(camera.width - 1) / max(float(token_width - 1), 1.0)
    stride_y = float(camera.height - 1) / max(float(token_height - 1), 1.0)
    stride = float(max(stride_x, stride_y))
    config = PatchTo3DMatchingConfig(
        top_k=args.top_k,
        mutual_top_k=args.mutual_top_k,
        match_mode=args.match_mode,
        ratio_threshold=None if args.disable_ratio_test else args.ratio_threshold,
        min_similarity=args.min_similarity,
        min_observation_count=args.min_observation_count,
        query_token_step=args.query_token_step,
        max_matches=args.max_matches,
        block_size=args.match_block_size,
        similarity_device=args.similarity_device,
        match_score_mode=args.match_score_mode,
        landmark_quality=LandmarkQualityConfig(enabled=False),
    )
    matches = match_query_patches_to_landmarks(
        query_feature,
        submap,
        config,
        image_width=int(camera.width),
        image_height=int(camera.height),
    )
    pnp = estimate_pose_pnp_ransac(
        matches,
        camera,
        reprojection_error_px=patch_uncertainty_pnp_threshold(stride, args.pnp_threshold_stride_multiplier),
        iterations=args.pnp_iterations,
    )
    positives = build_patch_positive_sets(
        submap,
        gt.pose_w2c,
        camera,
        token_width,
        token_height,
        patch_scale=args.patch_scale,
    )
    patch_stats = evaluate_patch_matches(
        matches,
        positives,
        gt.pose_w2c,
        camera,
        stride_px=stride,
        pnp_inlier_mask=pnp.inlier_mask,
        top_k=args.patch_at_k,
    )
    pose_error = pnp_pose_error(pnp.pose_w2c, gt.pose_w2c)
    image_rgb = _read_image_rgb(context["image_root"] / query_id)
    overlay, overlay_summary = render_patch_to_3d_match_overlay(
        image_rgb,
        matches,
        positives,
        pose_w2c=gt.pose_w2c,
        camera=camera,
        inlier_mask=pnp.inlier_mask,
        max_draw=args.max_draw_matches,
    )
    margin_map = compute_similarity_margin_map(
        query_feature,
        submap.features,
        block_size=args.margin_block_size,
        device=args.similarity_device,
    )
    margin_overlay = render_margin_heatmap(margin_map, image_rgb)
    metrics = {
        **patch_stats,
        "method": method.name,
        "query_id": query_id,
        "submap_landmark_count": int(len(submap)),
        "pnp_solve_rate_item": bool(pnp.success),
        "pnp_inlier_count": int(pnp.inlier_count),
        "pnp_inlier_ratio": float(pnp.inlier_ratio),
        "translation_error_m": _finite_or_none(pose_error.translation_m),
        "rotation_error_deg": _finite_or_none(pose_error.rotation_deg),
        "positive_set_stats": patch_positive_set_stats(positives),
        "overlay": overlay_summary,
        "margin": {
            "mean": float(np.mean(margin_map)),
            "median": float(np.median(margin_map)),
            "p90": float(np.percentile(margin_map, 90.0)),
            "p99": float(np.percentile(margin_map, 99.0)),
        },
    }
    return {
        "overlay_image": overlay,
        "margin_image": margin_overlay,
        "metrics": metrics,
    }


def _load_context(args, preset: ScenePreset):
    from feature_extract.tools.vfm.eval_query_to_3d_vfm_matching import (
        _infer_camera_model_dir,
        _load_camera_with_source,
        _load_reference_submaps,
        _load_track_stats,
        _parse_default_camera,
    )
    from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
    from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex

    xyz_by_track, reprojection_error_by_track = _load_track_stats(preset.track_observations)
    visibility_index = None
    if str(preset.visibility_index):
        visibility_index = LandmarkVisibilityIndex.load_npz(preset.visibility_index)
    camera_model_dir = _infer_camera_model_dir(str(preset.query_pose_file), args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    reference_submaps = _load_reference_submaps(str(preset.candidate_bank), args.submap_top_n)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(preset.query_pose_file)}
    return {
        "xyz_by_track": xyz_by_track,
        "reprojection_error_by_track": reprojection_error_by_track,
        "visibility_index": visibility_index,
        "camera": camera,
        "camera_source": camera_source,
        "reference_submaps": reference_submaps,
        "gt_by_query": gt_by_query,
        "image_root": preset.image_root,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Stage C1 qualitative selection comparisons")
    parser.add_argument("--scene", default="", choices=("", "shopfacade", "oldhospital"))
    parser.add_argument("--method", action="append", default=[], help="name,query_manifest,landmark_bank[,transform]")
    parser.add_argument("--track_observations", default="")
    parser.add_argument("--visibility_index", default="")
    parser.add_argument("--query_pose_file", default="")
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--query_id", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submap_mode", default="reference_visibility", choices=("reference_visibility", "gt_visible", "none"))
    parser.add_argument("--submap_top_n", type=int, default=10)
    parser.add_argument("--max_submap_landmarks", type=int, default=12000)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--query_token_step", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--mutual_top_k", type=int, default=1)
    parser.add_argument("--match_mode", default="mnn", choices=("nn", "mnn", "soft_mutual"))
    parser.add_argument("--ratio_threshold", type=float, default=0.95)
    parser.add_argument("--disable_ratio_test", action="store_true")
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--match_block_size", type=int, default=256)
    parser.add_argument("--margin_block_size", type=int, default=128)
    parser.add_argument("--similarity_device", default="cpu")
    parser.add_argument("--match_score_mode", default="similarity_quality", choices=("similarity", "landmark_quality", "similarity_quality"))
    parser.add_argument("--patch_scale", type=float, default=1.0)
    parser.add_argument("--patch_at_k", type=int, default=5)
    parser.add_argument("--pnp_threshold_stride_multiplier", type=float, default=1.5)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--max_draw_matches", type=int, default=180)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--cell_width", type=int, default=640)
    parser.add_argument("--cell_height", type=int, default=360)
    args = parser.parse_args(argv)

    if args.scene:
        preset = _stage_c1_scene_preset(args.scene)
    else:
        methods = tuple(_parse_method_spec(value) for value in args.method)
        if not methods:
            raise ValueError("provide either --scene preset or one or more --method specs")
        required = {
            "track_observations": args.track_observations,
            "query_pose_file": args.query_pose_file,
            "candidate_bank": args.candidate_bank,
            "image_root": args.image_root,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing required args without --scene: {missing}")
        preset = ScenePreset(
            name="custom",
            track_observations=Path(args.track_observations),
            visibility_index=Path(args.visibility_index) if args.visibility_index else Path(""),
            query_pose_file=Path(args.query_pose_file),
            candidate_bank=Path(args.candidate_bank),
            image_root=Path(args.image_root),
            methods=methods,
        )
    if args.method:
        preset = ScenePreset(
            name=preset.name,
            track_observations=preset.track_observations,
            visibility_index=preset.visibility_index,
            query_pose_file=preset.query_pose_file,
            candidate_bank=preset.candidate_bank,
            image_root=preset.image_root,
            methods=tuple(_parse_method_spec(value) for value in args.method),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(args, preset)
    method_runtime = {}
    for method in preset.methods:
        method_runtime[method.name] = _load_method_runtime(
            method,
            context["xyz_by_track"],
            context["reprojection_error_by_track"],
        )

    summary: dict[str, object] = {
        "stage": "stage_c1_selection_qualitative",
        "scene": preset.name,
        "query_ids": list(args.query_id),
        "methods": [method.name for method in preset.methods],
        "camera_source": context["camera_source"],
        "outputs": {},
        "queries": [],
    }
    for query_id in args.query_id:
        overlay_panels = []
        overlay_labels = []
        margin_panels = []
        margin_labels = []
        query_summary = {"query_id": query_id, "methods": []}
        query_dir = output_dir / _safe_query_name(query_id)
        query_dir.mkdir(parents=True, exist_ok=True)
        for method in preset.methods:
            records, index = method_runtime[method.name]
            result = _visualize_query_method(args, method, records, index, context, query_id)
            metrics = result["metrics"]
            label = _method_label(method.name, metrics)
            overlay_panels.append(result["overlay_image"])
            overlay_labels.append(label)
            margin_panels.append(result["margin_image"])
            margin_labels.append(
                f"{method.name} | margin p90 {float(metrics['margin']['p90']):.3f} | p99 {float(metrics['margin']['p99']):.3f}"
            )
            overlay_path = query_dir / f"{method.name}_match_overlay.png"
            margin_path = query_dir / f"{method.name}_margin_heatmap.png"
            _write_image_rgb(overlay_path, result["overlay_image"])
            _write_image_rgb(margin_path, result["margin_image"])
            query_summary["methods"].append(
                {
                    **metrics,
                    "match_overlay_png": str(overlay_path),
                    "margin_heatmap_png": str(margin_path),
                }
            )
        overlay_sheet = make_contact_sheet(
            overlay_panels,
            overlay_labels,
            columns=len(preset.methods),
            cell_width=args.cell_width,
            cell_height=args.cell_height,
        )
        margin_sheet = make_contact_sheet(
            margin_panels,
            margin_labels,
            columns=len(preset.methods),
            cell_width=args.cell_width,
            cell_height=args.cell_height,
        )
        overlay_sheet_path = output_dir / f"{_safe_query_name(query_id)}_match_comparison_contact_sheet.png"
        margin_sheet_path = output_dir / f"{_safe_query_name(query_id)}_margin_heatmap_contact_sheet.png"
        _write_image_rgb(overlay_sheet_path, overlay_sheet)
        _write_image_rgb(margin_sheet_path, margin_sheet)
        query_summary["match_comparison_contact_sheet_png"] = str(overlay_sheet_path)
        query_summary["margin_heatmap_contact_sheet_png"] = str(margin_sheet_path)
        summary["queries"].append(query_summary)

    energy_series = []
    for method in preset.methods:
        if method.transform is None:
            continue
        matrix = _load_transform_matrix(method.transform)
        energy_series.append((method.name, group_energy_from_transform(matrix, group_size=args.group_size)))
    if energy_series:
        energy_image = render_group_energy_heatmap(energy_series, group_size=args.group_size)
        energy_path = output_dir / "group_energy_heatmap.png"
        _write_image_rgb(energy_path, energy_image)
        summary["outputs"]["group_energy_heatmap_png"] = str(energy_path)
    summary_path = output_dir / "qualitative_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
