"""MATCHA-style sparse keypoint matching on rendered dense VFM features."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_query_to_render_vfm_matching import (
    _infer_camera_model_dir,
    _load_camera_with_source,
    _load_query_feature,
    _parse_default_camera,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.dense_gaussian_field_diagnostics import pca_feature_rgb
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMField,
    GaussianRGBSource,
    GaussianVFMRenderConfig,
    load_gaussian_rgb_source_from_ply,
    render_gaussian_vfm_feature_map,
    render_gaussian_vfm_feature_map_gsplat,
    render_gaussian_rgb_image_gsplat,
    render_gaussian_rgb_image_soft,
)
from feature_extract.vfm.query_to_3d_matching import (
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
    pnp_pose_error,
    reprojection_error_stats,
)
from feature_extract.vfm.rendered_pose_scoring import (
    annotate_measurement_uncertainty,
    score_pose_hypothesis,
)
from feature_extract.vfm.rendered_keypoint_matching import (
    bilinear_sample_feature_map,
    dual_softmax_keypoint_matches,
    keypoint_feature_matches_to_pnp_matches,
    mutual_nn_keypoint_matches,
    refine_render_keypoint_matches_by_local_correlation,
    render_anchor_topk_keypoint_matches,
)
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for image loading") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detect_orb(image_rgb: np.ndarray, max_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=max(int(max_keypoints), 1))
    keypoints = detector.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda kp: float(kp.response), reverse=True)[: int(max_keypoints)]
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    xy = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    scores = np.asarray([kp.response for kp in keypoints], dtype=np.float32)
    return xy, scores


def _detect_superpoint(image_rgb: np.ndarray, max_keypoints: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    from feature_extract.vfm.lowlevel_offset_sidecar import HLocSuperPointKeypointDetector

    detector = HLocSuperPointKeypointDetector(device=device, max_keypoints=int(max_keypoints))
    keypoints = detector.detect(image_rgb)
    return keypoints.xy.astype(np.float32, copy=False), keypoints.scores.astype(np.float32, copy=False)


def _detect_disk(_image_rgb: np.ndarray, _max_keypoints: int, _device: str) -> tuple[np.ndarray, np.ndarray]:
    raise RuntimeError("DISK detector entry is reserved; provide local DISK weights before enabling it")


def _detect_keypoints(image_rgb: np.ndarray, detector: str, max_keypoints: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    if detector == "orb":
        return _detect_orb(image_rgb, max_keypoints)
    if detector == "superpoint":
        return _detect_superpoint(image_rgb, max_keypoints, device)
    if detector == "disk":
        return _detect_disk(image_rgb, max_keypoints, device)
    raise ValueError(f"unsupported detector: {detector}")


def _make_keypoint_detector(detector: str, max_keypoints: int, device: str):
    if detector == "superpoint":
        from feature_extract.vfm.lowlevel_offset_sidecar import HLocSuperPointKeypointDetector

        model = HLocSuperPointKeypointDetector(device=device, max_keypoints=int(max_keypoints))

        def run(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            keypoints = model.detect(image_rgb)
            return keypoints.xy.astype(np.float32, copy=False), keypoints.scores.astype(np.float32, copy=False)

        return run

    def run(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _detect_keypoints(image_rgb, detector, int(max_keypoints), device)

    return run


def _scale_camera(camera: ColmapCamera, width: int, height: int) -> ColmapCamera:
    sx = float(width) / max(float(camera.width), 1.0)
    sy = float(height) / max(float(camera.height), 1.0)
    params = tuple(float(v) for v in camera.params)
    if camera.model_id == 1 and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        scaled = (fx * sx, fy * sy, cx * sx, cy * sy)
    elif camera.model_id in {0, 2, 8} and len(params) >= 3:
        if abs(sx - sy) > 1e-6:
            raise ValueError("SIMPLE camera models require isotropic render scaling")
        f, cx, cy = params[:3]
        scaled = (f * sx, cx * sx, cy * sy, *params[3:])
    else:
        scaled = params
    return ColmapCamera(camera_id=camera.camera_id, model_id=camera.model_id, width=int(width), height=int(height), params=scaled)


def _project_query_feature(feature_map: np.ndarray, selector_checkpoint: str, device: str) -> np.ndarray:
    if not selector_checkpoint:
        return np.asarray(feature_map, dtype=np.float32)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Torch is required for selector projection") from exc
    checkpoint = torch.load(Path(selector_checkpoint), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    state_dict = {
        (str(k)[len("module.") :] if str(k).startswith("module.") else str(k)): v
        for k, v in state_dict.items()
    }
    if "projection.weight" not in state_dict:
        raise ValueError("selector checkpoint is missing projection.weight")
    projection_weight = state_dict["projection.weight"]
    device_obj = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    if projection_weight.ndim == 4:
        selector = load_selector_from_checkpoint(Path(selector_checkpoint), device=str(device_obj))
        tensor = torch.as_tensor(feature_map[None], dtype=torch.float32, device=device_obj)
        with torch.no_grad():
            selected = selector(tensor).selected[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return selected

    from feature_extract.vfm.patch_selector_training import ResidualGatedPatchSelector

    output_dim, input_dim = int(projection_weight.shape[0]), int(projection_weight.shape[1])
    if int(feature_map.shape[0]) != input_dim:
        raise ValueError(f"selector expects {input_dim} input channels, got {feature_map.shape[0]}")
    group_count = int(state_dict.get("group_logits", torch.zeros((input_dim // 64,))).numel())
    group_size = input_dim // max(group_count, 1)
    hidden_dim = int(state_dict["residual.1.weight"].shape[0]) if "residual.1.weight" in state_dict else 256
    input_mean = state_dict.get("input_mean")
    selector = ResidualGatedPatchSelector(
        input_dim=input_dim,
        output_dim=output_dim,
        residual_hidden_dim=hidden_dim,
        group_size=group_size,
        input_mean=None if input_mean is None else input_mean.detach().cpu().numpy().reshape(-1),
    )
    selector.load_state_dict(state_dict, strict=False)
    selector.to(device_obj).eval()
    channels, height, width = feature_map.shape
    flat = np.asarray(feature_map, dtype=np.float32).reshape(channels, -1).T
    chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], 4096):
            tensor = torch.as_tensor(flat[start : start + 4096], dtype=torch.float32, device=device_obj)
            chunks.append(selector(tensor).detach().cpu().numpy().astype(np.float32, copy=False))
    selected_flat = np.concatenate(chunks, axis=0)
    return selected_flat.T.reshape(output_dim, height, width).astype(np.float32, copy=False)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_matches(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    matches,
    gt_errors: np.ndarray,
    render_xy_by_track: dict[int, np.ndarray],
    output_path: Path,
    *,
    max_draw: int = 80,
    good_threshold_px: float = 16.0,
) -> None:
    import cv2

    q = np.asarray(query_rgb, dtype=np.uint8)
    r = np.asarray(render_rgb, dtype=np.uint8)
    target_h = max(r.shape[0], 1)
    scale = target_h / max(float(q.shape[0]), 1.0)
    q_small = cv2.resize(q, (max(1, int(round(q.shape[1] * scale))), target_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, q_small.shape[1] + r.shape[1], 3), dtype=np.uint8)
    canvas[:, : q_small.shape[1]] = q_small
    canvas[:, q_small.shape[1] :] = r
    for idx, match in enumerate(matches[: int(max_draw)]):
        qxy = (np.asarray(match.xy, dtype=np.float64) * scale).astype(int)
        rxy = np.asarray(render_xy_by_track.get(int(match.track_id), np.zeros((2,), dtype=np.float64)))
        if rxy.shape != (2,):
            continue
        p1 = (int(qxy[0]), int(qxy[1]))
        p2 = (int(q_small.shape[1] + rxy[0]), int(rxy[1]))
        good = bool(idx < gt_errors.shape[0] and gt_errors[idx] <= float(good_threshold_px))
        color = (0, 220, 0) if good else (220, 40, 40)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.size == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    return np.clip(rgb.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)


def _render_keypoint_detector_image(
    rendered,
    *,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    config: GaussianVFMRenderConfig,
    rgb_source: GaussianRGBSource | None,
    renderer: str,
    device: str,
) -> np.ndarray:
    if rgb_source is None:
        return _to_uint8_rgb(pca_feature_rgb(rendered.feature_map, rendered.visibility_mask))
    if renderer == "gsplat":
        rgb, _alpha = render_gaussian_rgb_image_gsplat(
            rgb_source,
            pose_w2c=pose_w2c,
            camera=camera,
            config=config,
            device=device,
        )
    else:
        rgb, _alpha = render_gaussian_rgb_image_soft(
            rgb_source,
            pose_w2c=pose_w2c,
            camera=camera,
            config=config,
        )
    return _to_uint8_rgb(rgb)


def _pose_metrics(pose_w2c: np.ndarray | None, gt_pose_w2c: np.ndarray | None) -> tuple[float | None, float | None]:
    if pose_w2c is None or gt_pose_w2c is None:
        return None, None
    error = pnp_pose_error(pose_w2c, gt_pose_w2c)
    return float(error.translation_m), float(error.rotation_deg)


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if value is not None:
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(f):
                    out.append(f)
        return out

    t = vals("translation_error_m")
    r = vals("rotation_error_deg")
    summary = {
        "query_count": int(len(rows)),
        "pnp_solve_rate": float(np.mean([1.0 if row.get("pnp_success") else 0.0 for row in rows])) if rows else 0.0,
        "median_translation_error_m": None if not t else float(np.median(t)),
        "median_rotation_error_deg": None if not r else float(np.median(r)),
        "success_10cm_5deg": float(
            np.mean(
                [
                    1.0
                    if row.get("translation_error_m") is not None
                    and float(row["translation_error_m"]) <= 0.10
                    and row.get("rotation_error_deg") is not None
                    and float(row["rotation_error_deg"]) <= 5.0
                    else 0.0
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
        "success_25cm_10deg": float(
            np.mean(
                [
                    1.0
                    if row.get("translation_error_m") is not None
                    and float(row["translation_error_m"]) <= 0.25
                    and row.get("rotation_error_deg") is not None
                    and float(row["rotation_error_deg"]) <= 10.0
                    else 0.0
                    for row in rows
                ]
            )
        )
        if rows
        else 0.0,
        "mean_query_keypoints": None if not vals("query_keypoint_count") else float(np.mean(vals("query_keypoint_count"))),
        "mean_render_keypoints": None if not vals("render_keypoint_count") else float(np.mean(vals("render_keypoint_count"))),
        "mean_match_count": None if not vals("match_count") else float(np.mean(vals("match_count"))),
        "mean_pnp_inlier_count": None if not vals("pnp_inlier_count") else float(np.mean(vals("pnp_inlier_count"))),
    }
    for threshold in (5, 10, 16, 32):
        gt_values = vals(f"gt_precision_{threshold}px")
        inlier_values = vals(f"pnp_inlier_gt_precision_{threshold}px")
        summary[f"mean_gt_precision_{threshold}px"] = None if not gt_values else float(np.mean(gt_values))
        summary[f"mean_pnp_inlier_gt_precision_{threshold}px"] = (
            None if not inlier_values else float(np.mean(inlier_values))
        )
    return summary


def _geometry_row_fields(geometry: dict[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in geometry.items():
        if (
            key.startswith("gt_precision_")
            or key.startswith("pnp_inlier_gt_precision_")
            or key.startswith("gt_reproj_")
            or key.startswith("pnp_inlier_gt_reproj_")
        ):
            fields[key] = value
    return fields


def _pose_candidate_match_sets(matches, *, max_matches: int) -> list[tuple[str, list]]:
    values = list(matches)
    sets: list[tuple[str, list]] = [("all", values)]
    if len(values) > 80:
        sets.append(("top80", values[:80]))
    if len(values) > 60:
        sets.append(("top60", values[:60]))
    if any(match.pnp_soft_score is not None for match in values):
        conf_sorted = sorted(
            values,
            key=lambda match: (
                float(match.pnp_soft_score or 0.0),
                float(match.similarity),
            ),
            reverse=True,
        )
        sets.append(("conf_all", conf_sorted[: int(max_matches)]))
        if len(conf_sorted) > 80:
            sets.append(("conf80", conf_sorted[:80]))
    dedup: list[tuple[str, list]] = []
    seen: set[tuple[str, int]] = set()
    for label, subset in sets:
        if len(subset) < 4:
            continue
        key = (label, len(subset))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((label, subset[: int(max_matches)]))
    return dedup


def _estimate_pose_with_optional_rescore(
    matches,
    camera,
    *,
    enable_rescore: bool,
    pnp_reprojection_error_px: float,
    pnp_iterations: int,
    pnp_min_inliers: int,
    max_matches: int,
    rescore_margin: float,
):
    if not enable_rescore:
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(pnp_reprojection_error_px),
            iterations=int(pnp_iterations),
            min_inliers=int(pnp_min_inliers),
            refine_lm=True,
            refine_method="LM",
        )
        return pnp, "single", None, list(matches)
    candidate_rows = []
    for subset_label, subset in _pose_candidate_match_sets(matches, max_matches=max_matches):
        for threshold_scale in (0.75, 1.0, 1.25):
            threshold = float(pnp_reprojection_error_px) * float(threshold_scale)
            pnp = estimate_pose_pnp_ransac(
                subset,
                camera,
                reprojection_error_px=threshold,
                iterations=int(pnp_iterations),
                min_inliers=int(pnp_min_inliers),
                refine_lm=True,
                refine_method="LM",
            )
            score = score_pose_hypothesis(
                subset,
                pnp.pose_w2c if pnp.success else None,
                camera,
                inlier_threshold_px=threshold,
                inlier_mask=pnp.inlier_mask,
            )
            candidate_rows.append(
                {
                    "label": f"{subset_label}@{threshold_scale:.2f}",
                    "subset": subset,
                    "pnp": pnp,
                    "score": score,
                    "threshold_px": threshold,
                }
            )
    if not candidate_rows:
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(pnp_reprojection_error_px),
            iterations=int(pnp_iterations),
            min_inliers=int(pnp_min_inliers),
            refine_lm=True,
            refine_method="LM",
        )
        return pnp, "fallback", None, list(matches)
    best = max(candidate_rows, key=lambda row: float(row["score"].score))
    canonical = next((row for row in candidate_rows if str(row["label"]) == "all@1.00"), candidate_rows[0])
    if float(best["score"].score) < float(canonical["score"].score) + float(rescore_margin):
        best = dict(canonical)
        best["label"] = f"guarded:{best['label']}"
    return best["pnp"], str(best["label"]), best["score"], list(best["subset"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--selector_checkpoint", default="")
    parser.add_argument("--gaussian_rgb_ply", default="")
    parser.add_argument("--rgb_max_gaussians", type=int, default=0)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_pose_mode", default="gt", choices=("gt",))
    parser.add_argument("--render_width", type=int, default=320)
    parser.add_argument("--render_height", type=int, default=180)
    parser.add_argument("--render_radius_px", type=float, default=2.0)
    parser.add_argument("--render_depth_epsilon", type=float, default=0.02)
    parser.add_argument("--renderer", default="soft", choices=("soft", "gsplat"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel_chunk", type=int, default=32)
    parser.add_argument("--detector", default="orb", choices=("orb", "superpoint", "disk"))
    parser.add_argument("--max_keypoints", type=int, default=1000)
    parser.add_argument("--match_mode", default="mnn", choices=("mnn", "mnn_dual_filter", "dual_softmax", "render_anchor_topk"))
    parser.add_argument("--ratio_threshold", type=float, default=0.9)
    parser.add_argument("--render_anchor_top_l", type=int, default=5)
    parser.add_argument("--dual_softmax_logit_scale", type=float, default=10.0)
    parser.add_argument("--min_dual_softmax_confidence", type=float, default=0.0)
    parser.add_argument("--min_similarity", type=float, default=0.0)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--fine_render_search_radius_px", type=float, default=0.0)
    parser.add_argument("--fine_render_search_step_px", type=float, default=1.0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=12.0)
    parser.add_argument("--pnp_iterations", type=int, default=2000)
    parser.add_argument("--pnp_min_inliers", type=int, default=6)
    parser.add_argument("--measurement_sigma_px", type=float, default=0.0)
    parser.add_argument("--enable_pose_rescore", action="store_true")
    parser.add_argument("--pose_rescore_margin", type=float, default=0.03)
    parser.add_argument("--max_queries", type=int, default=20)
    parser.add_argument("--visualize_limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    manifest.validate(verify_checksums=False)
    gt_by_query = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    camera_model_dir = _infer_camera_model_dir(args.query_pose_file, args.camera_model_dir)
    camera, camera_source = _load_camera_with_source(camera_model_dir, _parse_default_camera(args.default_camera))
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    field = GaussianVFMField.load_npz(Path(args.field))
    rgb_source = (
        load_gaussian_rgb_source_from_ply(Path(args.gaussian_rgb_ply), max_gaussians=int(args.rgb_max_gaussians))
        if args.gaussian_rgb_ply
        else None
    )
    render_config = GaussianVFMRenderConfig(
        width=int(args.render_width),
        height=int(args.render_height),
        radius_px=float(args.render_radius_px),
        depth_epsilon=float(args.render_depth_epsilon),
        l2_normalize_pixels=True,
    )
    detector_fn = _make_keypoint_detector(args.detector, int(args.max_keypoints), args.device)

    output_dir = Path(args.output_dir)
    rows = []
    records = list(manifest.records)
    if int(args.max_queries) > 0:
        records = records[: int(args.max_queries)]
    for vis_idx, record in enumerate(records):
        gt = gt_by_query.get(record.image_id)
        if gt is None:
            continue
        query_image = _read_rgb(Path(args.image_root) / record.image_id)
        query_feature = _load_query_feature(record.token_path, args.layer_name)
        query_feature = _project_query_feature(query_feature, args.selector_checkpoint, args.device)
        if int(query_feature.shape[0]) != int(field.feature_dim):
            rows.append(
                {
                    "query_id": record.image_id,
                    "status": "descriptor_dim_mismatch",
                    "query_dim": int(query_feature.shape[0]),
                    "field_dim": int(field.feature_dim),
                    "pnp_success": False,
                }
            )
            continue
        pose_w2c = gt.pose_w2c
        if args.renderer == "gsplat":
            rendered = render_gaussian_vfm_feature_map_gsplat(
                field,
                pose_w2c=pose_w2c,
                camera=camera,
                config=render_config,
                device=args.device,
                channel_chunk=args.channel_chunk,
            )
        else:
            rendered = render_gaussian_vfm_feature_map(field, pose_w2c=pose_w2c, camera=camera, config=render_config)
        render_rgb = _render_keypoint_detector_image(
            rendered,
            pose_w2c=pose_w2c,
            camera=camera,
            config=render_config,
            rgb_source=rgb_source,
            renderer=args.renderer,
            device=args.device,
        )
        query_xy, _query_scores = detector_fn(query_image)
        render_xy, _render_scores = detector_fn(render_rgb)
        qdesc, qvalid = bilinear_sample_feature_map(
            query_feature,
            query_xy,
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
        rdesc, rvalid = bilinear_sample_feature_map(
            rendered.feature_map,
            render_xy,
            image_width=int(render_config.width),
            image_height=int(render_config.height),
        )
        query_xy_valid = query_xy[qvalid]
        render_xy_valid = render_xy[rvalid]
        qdesc = qdesc[qvalid]
        rdesc = rdesc[rvalid]
        if args.match_mode == "dual_softmax":
            kp_matches = dual_softmax_keypoint_matches(
                query_xy_valid,
                qdesc,
                render_xy_valid,
                rdesc,
                logit_scale=float(args.dual_softmax_logit_scale),
                min_confidence=float(args.min_dual_softmax_confidence),
                min_similarity=float(args.min_similarity),
                max_matches=args.max_matches,
            )
        elif args.match_mode == "render_anchor_topk":
            kp_matches = render_anchor_topk_keypoint_matches(
                query_xy_valid,
                qdesc,
                render_xy_valid,
                rdesc,
                top_l=int(args.render_anchor_top_l),
                min_similarity=float(args.min_similarity),
                dual_softmax_logit_scale=float(args.dual_softmax_logit_scale),
                min_dual_softmax_confidence=float(args.min_dual_softmax_confidence),
                max_matches=args.max_matches,
            )
        else:
            kp_matches = mutual_nn_keypoint_matches(
                query_xy_valid,
                qdesc,
                render_xy_valid,
                rdesc,
                ratio_threshold=float(args.ratio_threshold),
                min_similarity=float(args.min_similarity),
                dual_softmax_logit_scale=(
                    float(args.dual_softmax_logit_scale) if args.match_mode == "mnn_dual_filter" else None
                ),
                min_dual_softmax_confidence=float(args.min_dual_softmax_confidence),
                max_matches=args.max_matches,
            )
        if float(args.fine_render_search_radius_px) > 0.0:
            kp_matches = refine_render_keypoint_matches_by_local_correlation(
                kp_matches,
                qdesc,
                rendered.feature_map,
                image_width=int(render_config.width),
                image_height=int(render_config.height),
                search_radius_px=float(args.fine_render_search_radius_px),
                step_px=float(args.fine_render_search_step_px),
            )
        pnp_matches = keypoint_feature_matches_to_pnp_matches(
            kp_matches,
            rendered.depth,
            render_camera,
            pose_w2c,
            image_width=int(render_config.width),
            image_height=int(render_config.height),
        )
        if float(args.measurement_sigma_px) > 0.0:
            pnp_matches = annotate_measurement_uncertainty(
                pnp_matches,
                base_sigma_px=float(args.measurement_sigma_px),
                fine_refined=float(args.fine_render_search_radius_px) > 0.0,
            )
        render_xy_by_match = {int(match.render_index): np.asarray(match.render_xy, dtype=np.float64) for match in kp_matches}
        pnp, pose_candidate_label, pose_candidate_score, pose_pnp_matches = _estimate_pose_with_optional_rescore(
            pnp_matches,
            camera,
            enable_rescore=bool(args.enable_pose_rescore),
            pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
            pnp_iterations=int(args.pnp_iterations),
            pnp_min_inliers=int(args.pnp_min_inliers),
            max_matches=int(args.max_matches),
            rescore_margin=float(args.pose_rescore_margin),
        )
        translation_error, rotation_error = _pose_metrics(pnp.pose_w2c if pnp.success else None, gt.pose_w2c)
        geometry = (
            reprojection_error_stats(
                pose_pnp_matches,
                gt.pose_w2c,
                camera,
                thresholds_px=(5.0, 10.0, 16.0, 32.0),
                pnp_inlier_mask=pnp.inlier_mask,
            )
            if pose_pnp_matches
            else {}
        )
        gt_errors = (
            match_reprojection_errors(pose_pnp_matches, gt.pose_w2c, camera)
            if pose_pnp_matches
            else np.zeros((0,), dtype=np.float64)
        )
        if vis_idx < int(args.visualize_limit):
            _draw_matches(
                query_image,
                render_rgb,
                pose_pnp_matches,
                gt_errors,
                render_xy_by_match,
                output_dir / "visualizations" / f"{record.image_id.replace('/', '__')}_matches.png",
            )
        rows.append(
            {
                "query_id": record.image_id,
                "status": "ok",
                "query_keypoint_count": int(query_xy.shape[0]),
                "render_keypoint_count": int(render_xy.shape[0]),
                "valid_query_descriptor_count": int(qdesc.shape[0]),
                "valid_render_descriptor_count": int(rdesc.shape[0]),
                "render_visible_fraction": float(np.mean(rendered.visibility_mask)),
                "match_count": int(len(kp_matches)),
                "depth_valid_match_count": int(len(pnp_matches)),
                "pose_candidate_match_count": int(len(pose_pnp_matches)),
                "pnp_success": bool(pnp.success),
                "pnp_inlier_count": int(pnp.inlier_count),
                "pnp_inlier_ratio": float(pnp.inlier_ratio),
                "pose_candidate_label": pose_candidate_label,
                "pose_candidate_score": None if pose_candidate_score is None else float(pose_candidate_score.score),
                "pose_candidate_weighted_residual": None
                if pose_candidate_score is None
                else float(pose_candidate_score.weighted_residual),
                "pose_candidate_coverage": None if pose_candidate_score is None else float(pose_candidate_score.coverage),
                "measurement_sigma_px_mean": None
                if not pnp_matches or pnp_matches[0].measurement_sigma_px is None
                else float(np.mean([float(match.measurement_sigma_px or 0.0) for match in pnp_matches])),
                "mean_dual_softmax_confidence": (
                    None
                    if not kp_matches or kp_matches[0].dual_softmax_confidence is None
                    else float(np.mean([float(match.dual_softmax_confidence or 0.0) for match in kp_matches]))
                ),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                **_geometry_row_fields(geometry),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(rows_path, rows)
    summary = {
        "stage": "rendered_feature_keypoint_pose",
        "elapsed_sec": float(time.perf_counter() - started),
        "inputs": {
            "query_manifest": str(args.query_manifest),
            "query_pose_file": str(args.query_pose_file),
            "image_root": str(args.image_root),
            "field": str(args.field),
            "selector_checkpoint": str(args.selector_checkpoint),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        },
        "camera": {
            "source": camera_source,
            "width": int(camera.width),
            "height": int(camera.height),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
        },
        "config": {
            "detector": args.detector,
            "match_mode": args.match_mode,
            "max_keypoints": int(args.max_keypoints),
            "ratio_threshold": float(args.ratio_threshold),
            "render_anchor_top_l": int(args.render_anchor_top_l),
            "dual_softmax_logit_scale": float(args.dual_softmax_logit_scale),
            "min_dual_softmax_confidence": float(args.min_dual_softmax_confidence),
            "min_similarity": float(args.min_similarity),
            "fine_render_search_radius_px": float(args.fine_render_search_radius_px),
            "fine_render_search_step_px": float(args.fine_render_search_step_px),
            "measurement_sigma_px": float(args.measurement_sigma_px),
            "enable_pose_rescore": bool(args.enable_pose_rescore),
            "pose_rescore_margin": float(args.pose_rescore_margin),
            "renderer": args.renderer,
        },
        "metrics": _summary(rows),
        "outputs": {"rows": str(rows_path), "summary": str(summary_path), "visualizations": str(output_dir / "visualizations")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
