"""RADIO large-plane retrieval followed by LoFTR finite-support Sim(3).

RADIO is used only to shortlist plane families.  LoFTR supplies the pixel-level
within-plane correspondences expected by a PlanarReloc-style backend.  Every
pair, homography, and SE(3)/Sim(3) estimate is frozen before held pose/reference
geometry is opened for the historical diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from feature_extract.tools.vfm.evaluate_goal_maplet_chart_plane_pose_oracle import (
    _robust_point_pose,
    _rotation_error_deg,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_large_plane_radio_pose import (
    MAXIMUM_FAMILY_CANDIDATES,
    _family_scores,
    _map_local_regions,
    _regions_from_grid,
    build_plane_families,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_planar_support_pose import _radio_records, _radio_tokens
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import StrictHeldRayInventory
from feature_extract.vfm.localization_goal_maplet.large_plane_regions import extract_large_plane_regions
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import _load_source_reference_dense


SCHEMA = "goal_maplet_large_plane_radio_loftr_sim3_historical_control_v6_final"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 432
LOFTR_CONFIDENCE_MINIMUM = 0.20
RADIO_FAMILY_SCORE_MINIMUM = 0.35
HOMOGRAPHY_RANSAC_THRESHOLD_PX = 4.0
HOMOGRAPHY_MINIMUM_INLIERS = 8
HOMOGRAPHY_MINIMUM_INLIER_RATIO = 0.35
POSE_MINIMUM_CORRESPONDENCES = 12
MAXIMUM_MEMBER_VIEWS_PER_FAMILY = 4
MASK_DILATION_CELLS = 0
CROPPED_PLANE_PATCH_MATCHING = False
CROPPED_PATCH_SIZE = 384
MULTIVIEW_MINIMUM_RAW_MATCHES = 8
MULTIVIEW_MINIMUM_SUPPORT_VIEWS = 2
MULTIVIEW_MINIMUM_SIM3_INLIERS = 6
MULTIVIEW_MINIMUM_SIM3_INLIER_RATIO = 0.30


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def _load_reference_vertices(
    authority: ProjectiveExactFaceSeamAuthority, source_root: Path,
) -> tuple[np.ndarray, dict[str, str]]:
    source_root = Path(source_root).resolve()
    if str(source_root) != str(authority.metadata.get("source_root")):
        raise ValueError("source-reference root differs from authority")
    inventory = authority.metadata.get("source_selected_pointmap_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("authority lacks source pointmap inventory")
    vertices, hashes = [], {}
    for chart, name in enumerate(authority.chart_names.astype(str)):
        path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        hashes[name] = file_sha256(path)
        if hashes[name] != inventory.get(name):
            raise ValueError(f"source pointmap differs for {name}")
        dense, _ = _load_source_reference_dense(path, output_height=144, output_width=256)
        lo, hi = map(int, authority.chart_vertex_offsets[chart:chart + 2])
        pixel = authority.sampled_vertex_pixel_indices[lo:hi]
        sample = dense.reshape(-1, 3)[pixel]
        if not np.isfinite(sample).all():
            raise ValueError(f"source reference is nonfinite on authority vertices for {name}")
        vertices.append(sample)
    return np.concatenate(vertices), hashes


def _load_source_calibrated_moge_vertices(
    authority: ProjectiveExactFaceSeamAuthority,
    source_root: Path,
    initializer_root: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    source_root = Path(source_root).resolve()
    initializer_root = Path(initializer_root).resolve()
    if str(source_root) != str(authority.metadata.get("source_root")):
        raise ValueError("source calibrated MoGe root differs from authority")
    manifest_path = initializer_root / "manifest.json"
    if file_sha256(manifest_path) != authority.metadata.get("moge3_initializer_manifest_file_sha256"):
        raise ValueError("source calibrated MoGe manifest differs from authority")
    cameras = json.loads((source_root / "cameras.json").read_text())
    camera_rows = {Path(path).name: row for row, path in enumerate(cameras["filepaths"])}
    pointmap_inventory = authority.metadata.get("source_selected_pointmap_inventory")
    initializer_inventory = authority.metadata.get("moge3_initializer_file_sha256")
    if not isinstance(pointmap_inventory, dict) or not isinstance(initializer_inventory, dict):
        raise ValueError("source calibrated MoGe authority inventory is incomplete")
    vertices, scales, pointmap_hashes, initializer_hashes = [], {}, {}, {}
    for chart, name in enumerate(authority.chart_names.astype(str)):
        pointmap_path = source_root / "pointmaps" / f"{Path(name).stem}.json"
        initializer_path = initializer_root / f"{name}.npz"
        pointmap_hashes[name] = file_sha256(pointmap_path)
        initializer_hashes[name] = file_sha256(initializer_path)
        if pointmap_hashes[name] != pointmap_inventory.get(name) or initializer_hashes[name] != initializer_inventory.get(name):
            raise ValueError(f"source calibrated MoGe inputs differ for {name}")
        reference, _ = _load_source_reference_dense(pointmap_path, output_height=144, output_width=256)
        with np.load(initializer_path, allow_pickle=False) as data:
            camera_points = np.asarray(data["points_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        row = camera_rows[name]
        c2w = np.asarray(cameras["cams2world"][row], np.float64)
        reference_camera = (reference - c2w[:3, 3]) @ c2w[:3, :3]
        common = valid & np.isfinite(reference_camera).all(2) & np.isfinite(camera_points).all(2) & (reference_camera[..., 2] > 0) & (camera_points[..., 2] > 0)
        if int(common.sum()) < 256:
            raise ValueError(f"insufficient source scale overlap for {name}")
        scale = float(np.median(reference_camera[..., 2][common] / camera_points[..., 2][common]))
        scales[name] = scale
        world = scale * (camera_points @ c2w[:3, :3].T) + c2w[:3, 3]
        lo, hi = map(int, authority.chart_vertex_offsets[chart:chart + 2])
        pixel = authority.sampled_vertex_pixel_indices[lo:hi]
        sample = world.reshape(-1, 3)[pixel]
        if not np.isfinite(sample).all():
            raise ValueError(f"source calibrated MoGe is nonfinite on authority vertices for {name}")
        vertices.append(sample)
    return np.concatenate(vertices), {
        "scale_by_chart": scales,
        "pointmap_file_sha256": pointmap_hashes,
        "initializer_file_sha256": initializer_hashes,
        "initializer_manifest_file_sha256": file_sha256(manifest_path),
    }


class _LoFTR:
    def __init__(self, device: str) -> None:
        import torch
        from kornia.feature import LoFTR
        self.torch = torch
        self.device = torch.device(device)
        torch.manual_seed(260831)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(260831)
        self.model = LoFTR(pretrained="outdoor").eval().to(self.device)

    def __call__(self, query: np.ndarray, mapping: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        q = torch.from_numpy(query)[None, None].to(self.device, torch.float32) / 255.0
        m = torch.from_numpy(mapping)[None, None].to(self.device, torch.float32) / 255.0
        with torch.no_grad():
            output = self.model({"image0": q, "image1": m})
        confidence = output["confidence"].detach().cpu().numpy().astype(np.float64)
        keep = confidence >= LOFTR_CONFIDENCE_MINIMUM
        return {
            "query_xy": output["keypoints0"].detach().cpu().numpy().astype(np.float64)[keep],
            "map_xy": output["keypoints1"].detach().cpu().numpy().astype(np.float64)[keep],
            "confidence": confidence[keep],
        }


def _plane_crop(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    mask_image = cv2.resize(np.asarray(mask, np.uint8), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
    yy, xx = np.nonzero(mask_image)
    if not len(xx):
        raise ValueError("cannot crop an empty plane support")
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    pad = max(8, int(round(.05 * max(x1 - x0, y1 - y0))))
    x0, x1 = max(0, x0 - pad), min(IMAGE_WIDTH, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(IMAGE_HEIGHT, y1 + pad)
    crop = image[y0:y1, x0:x1]
    crop_mask = mask_image[y0:y1, x0:x1]
    background = int(np.median(crop[crop_mask])) if crop_mask.any() else 127
    masked = np.full(crop.shape, background, np.uint8)
    masked[crop_mask] = crop[crop_mask]
    scale = min(CROPPED_PATCH_SIZE / masked.shape[1], CROPPED_PATCH_SIZE / masked.shape[0])
    width, height = max(1, int(round(masked.shape[1] * scale))), max(1, int(round(masked.shape[0] * scale)))
    resized = cv2.resize(masked, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((CROPPED_PATCH_SIZE, CROPPED_PATCH_SIZE), background, np.uint8)
    ox, oy = (CROPPED_PATCH_SIZE - width) // 2, (CROPPED_PATCH_SIZE - height) // 2
    canvas[oy:oy + height, ox:ox + width] = resized
    return canvas, (float(scale), float(ox - scale * x0), float(oy - scale * y0))


def _cropped_plane_pair(
    matcher: _LoFTR, query_image: np.ndarray, map_image: np.ndarray,
    query_mask: np.ndarray, map_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    query_crop, query_transform = _plane_crop(query_image, query_mask)
    map_crop, map_transform = _plane_crop(map_image, map_mask)
    pair = matcher(query_crop, map_crop)
    for key, transform in (("query_xy", query_transform), ("map_xy", map_transform)):
        scale, offset_x, offset_y = transform
        pair[key] = (pair[key] - np.asarray([offset_x, offset_y])) / scale
    return pair


def _grid_rows(xy: np.ndarray, *, factor: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = (np.asarray(xy, np.float64) + 0.5) / factor - 0.5
    xx = np.rint(grid[:, 0]).astype(np.int64)
    yy = np.rint(grid[:, 1]).astype(np.int64)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    return yy, xx, valid


def _masked_plane_points(
    pair: dict[str, np.ndarray], query: dict[str, object], mapping: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qy, qx, qvalid = _grid_rows(pair["query_xy"], factor=IMAGE_WIDTH / 256.0, height=144, width=256)
    my, mx, mvalid = _grid_rows(pair["map_xy"], factor=IMAGE_WIDTH / 128.0, height=72, width=128)
    keep = qvalid & mvalid
    query_mask = np.asarray(query["grid_mask"], np.uint8)
    map_mask = np.asarray(mapping["grid_mask"], np.uint8)
    if MASK_DILATION_CELLS:
        kernel = np.ones((2 * MASK_DILATION_CELLS + 1, 2 * MASK_DILATION_CELLS + 1), np.uint8)
        query_mask = cv2.dilate(query_mask, kernel).astype(bool)
        map_mask = cv2.dilate(map_mask, kernel).astype(bool)
    else:
        query_mask = query_mask.astype(bool)
        map_mask = map_mask.astype(bool)
    rows = np.flatnonzero(keep)
    if len(rows):
        keep[rows] &= query_mask[qy[rows], qx[rows]]
        keep[rows] &= map_mask[my[rows], mx[rows]]
        qpoint = np.asarray(query["grid_points"])[qy[rows], qx[rows]]
        mpoint = np.asarray(mapping["grid_points"])[my[rows], mx[rows]]
        keep[rows] &= np.isfinite(qpoint).all(1) & np.isfinite(mpoint).all(1)
    rows = np.flatnonzero(keep)
    return (
        np.asarray(query["grid_points"])[qy[rows], qx[rows]],
        np.asarray(mapping["grid_points"])[my[rows], mx[rows]],
        pair["query_xy"][rows], pair["map_xy"][rows], pair["confidence"][rows],
    )


def _plane_pair(
    pair: dict[str, np.ndarray], query: dict[str, object], mapping: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source, target, query_xy, map_xy, confidence = _masked_plane_points(pair, query, mapping)
    rows = np.arange(len(source), dtype=np.int64)
    inlier = np.zeros((len(rows),), bool)
    if len(rows) >= HOMOGRAPHY_MINIMUM_INLIERS:
        cv2.setRNGSeed(260831)
        _, mask = cv2.findHomography(
            query_xy, map_xy, cv2.RANSAC,
            HOMOGRAPHY_RANSAC_THRESHOLD_PX, maxIters=3000, confidence=0.999,
        )
        if mask is not None:
            inlier = mask.reshape(-1).astype(bool)
    accepted = (
        int(inlier.sum()) >= HOMOGRAPHY_MINIMUM_INLIERS
        and float(np.mean(inlier)) >= HOMOGRAPHY_MINIMUM_INLIER_RATIO
    )
    if not accepted:
        return np.zeros((0, 3)), np.zeros((0, 3)), {
            "accepted": False, "masked_match_count": int(len(rows)),
            "homography_inlier_count": int(inlier.sum()),
            "homography_inlier_ratio": float(np.mean(inlier)) if len(inlier) else 0.0,
        }
    selected = rows[inlier]
    return (
        source[selected], target[selected],
        {
            "accepted": True, "masked_match_count": int(len(rows)),
            "homography_inlier_count": int(len(selected)),
            "homography_inlier_ratio": float(np.mean(inlier)),
            "confidence_median": float(np.median(confidence[selected])),
        },
    )


def _pose(source: np.ndarray, target: np.ndarray, *, scale: bool) -> dict[str, object]:
    if len(source) < POSE_MINIMUM_CORRESPONDENCES:
        return {"usable": False, "correspondence_count": int(len(source))}
    rotation, translation, estimated_scale, inlier, residual = _robust_point_pose(
        source, target, estimate_scale=scale,
    )
    return {
        "usable": bool(np.isfinite(rotation).all() and np.isfinite(translation).all()),
        "correspondence_count": int(len(source)), "rotation_c2w": rotation.tolist(),
        "translation_world": translation.tolist(), "estimated_scale": float(estimated_scale),
        "inlier_count": int(inlier.sum()),
        "residual_median_m": float(np.median(residual[inlier])) if inlier.any() else None,
        "residual_p90_m": float(np.quantile(residual[inlier], .9)) if inlier.any() else None,
    }


def _summary(rows: list[dict[str, object]], branch: str) -> dict[str, object]:
    usable = [row[branch] for row in rows if row[branch]["usable"]]
    translation = np.asarray([row["translation_error_m"] for row in usable])
    rotation = np.asarray([row["rotation_error_deg"] for row in usable])
    return {
        "query_count": len(rows), "usable_count": len(usable),
        "translation_error_median_m": float(np.median(translation)) if len(translation) else None,
        "translation_error_p90_m": float(np.quantile(translation, .9)) if len(translation) else None,
        "rotation_error_median_deg": float(np.median(rotation)) if len(rotation) else None,
        "rotation_error_p90_deg": float(np.quantile(rotation, .9)) if len(rotation) else None,
        "full_query_recall_2m45": float(np.sum((translation <= 2) & (rotation <= 45)) / len(rows)),
        "full_query_recall_1m10": float(np.sum((translation <= 1) & (rotation <= 10)) / len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--source_reference_root", type=Path)
    parser.add_argument("--source_moge3_initializer_root", type=Path)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--moge3_query", type=Path, required=True)
    parser.add_argument("--radio_manifest", type=Path, required=True)
    parser.add_argument("--source_image_root", type=Path, required=True)
    parser.add_argument("--query_image_root", type=Path, required=True)
    parser.add_argument("--loftr_weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite large-plane LoFTR control")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("large-plane LoFTR authority differs from pin")
    if file_sha256(args.loftr_weight) == "":
        raise ValueError("LoFTR weight hash is empty")
    if (args.alignment is None) == (args.source_reference_root is None):
        raise ValueError("provide exactly one of alignment or source_reference_root")
    source_reference_hashes = None
    if args.alignment is not None:
        vertices = _load_aligned_vertices(args.alignment, authority)
        map_geometry_mode = "aligned_MoGe3_charts"
    elif args.source_moge3_initializer_root is None:
        vertices, source_reference_hashes = _load_reference_vertices(authority, args.source_reference_root)
        map_geometry_mode = "source_only_MASt3R_reference_surface_control"
    else:
        vertices, source_reference_hashes = _load_source_calibrated_moge_vertices(
            authority, args.source_reference_root, args.source_moge3_initializer_root,
        )
        map_geometry_mode = "source_reference_scale_calibrated_MoGe3_per_view_control"
    records = _radio_records(args.radio_manifest)
    local_regions, map_token_hash = _map_local_regions(authority, vertices, records)
    families = build_plane_families(local_regions)
    by_chart_name = {str(name): [] for name in authority.chart_names.astype(str)}
    for region in local_regions:
        by_chart_name[str(region["chart_name"])].append(region)
    query_manifest_path = args.moge3_query / "manifest.json"
    query_manifest = json.loads(query_manifest_path.read_text())
    names = [str(row["name"]) for row in query_manifest.get("rows", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("query MoGe3 inventory is empty or duplicated")
    matcher = _LoFTR(args.device)
    source_images = {
        name: _load_gray(args.source_image_root / name) for name in by_chart_name
    }
    source_image_hashes = {
        name: file_sha256(args.source_image_root / name) for name in by_chart_name
    }
    frozen = []
    for name in names:
        query_image_path = args.query_image_root / name
        query_image = _load_gray(query_image_path)
        query_feature, query_radio_hash = _radio_tokens(name, records)
        with np.load(args.moge3_query / f"{name}.npz", allow_pickle=False) as data:
            query_points = np.asarray(data["points_camera"], np.float64)
            query_normals = np.asarray(data["normal_camera"], np.float64)
            query_valid = np.asarray(data["valid"], bool)
        planes = extract_large_plane_regions(
            query_points, query_normals, query_valid,
            minimum_pixels=160, maximum_planes=16, maximum_hypotheses=128,
        )
        query_regions = _regions_from_grid(planes, query_points, query_feature, query=True)
        candidate_by_region = []
        required_charts: set[str] = set()
        for query_row, query in enumerate(query_regions):
            score = _family_scores(np.asarray(query["descriptor"]), families)
            order = np.argsort(-score, kind="stable")[:MAXIMUM_FAMILY_CANDIDATES]
            order = order[score[order] >= RADIO_FAMILY_SCORE_MINIMUM]
            candidates = []
            for rank, family in enumerate(order):
                members = sorted(
                    families[int(family)]["member_regions"],
                    key=lambda row: (-float(np.asarray(row["descriptor"]) @ np.asarray(query["descriptor"])), str(row["chart_name"]), int(row["local_region"])),
                )[:MAXIMUM_MEMBER_VIEWS_PER_FAMILY]
                candidates.append((int(family), rank + 1, float(score[family]), members))
                required_charts.update(str(row["chart_name"]) for row in members)
            candidate_by_region.append(candidates)
        pair_cache: dict[object, dict[str, np.ndarray]] = {}
        if not CROPPED_PLANE_PATCH_MATCHING:
            pair_cache.update({
                chart_name: matcher(query_image, source_images[chart_name])
                for chart_name in sorted(required_charts)
            })
        source_points, target_points, multiview_source_points, multiview_target_points, decisions = [], [], [], [], []
        for query_row, (query, candidates) in enumerate(zip(query_regions, candidate_by_region)):
            evaluated = []
            multiview_trials = []
            for family, rank, radio_score, members in candidates:
                family_source, family_target, family_views = [], [], 0
                for member in members:
                    pair_key: object = (
                        query_row, str(member["chart_name"]), int(member["local_region"]),
                    ) if CROPPED_PLANE_PATCH_MATCHING else str(member["chart_name"])
                    if pair_key not in pair_cache:
                        pair_cache[pair_key] = _cropped_plane_pair(
                            matcher, query_image, source_images[str(member["chart_name"])],
                            np.asarray(query["grid_mask"]), np.asarray(member["grid_mask"]),
                        )
                    pair = pair_cache[pair_key]
                    raw_source, raw_target, _, _, _ = _masked_plane_points(
                        pair, query, member,
                    )
                    if len(raw_source):
                        family_views += 1
                        family_source.extend(raw_source)
                        family_target.extend(raw_target)
                    source, target, audit = _plane_pair(
                        pair, query, member,
                    )
                    evaluated.append({
                        "family_id": family, "family_rank": rank,
                        "radio_family_score": radio_score,
                        "map_chart_name": str(member["chart_name"]),
                        "map_local_region": int(member["local_region"]),
                        "source": source, "target": target, **audit,
                    })
                family_source_array = np.asarray(family_source, np.float64).reshape(-1, 3)
                family_target_array = np.asarray(family_target, np.float64).reshape(-1, 3)
                local_inlier = np.zeros((len(family_source_array),), bool)
                local_scale = None
                local_residual_median = None
                if (
                    len(family_source_array) >= MULTIVIEW_MINIMUM_RAW_MATCHES
                    and family_views >= MULTIVIEW_MINIMUM_SUPPORT_VIEWS
                ):
                    _, _, local_scale_value, local_inlier, local_residual = _robust_point_pose(
                        family_source_array, family_target_array, estimate_scale=True,
                    )
                    local_scale = float(local_scale_value)
                    if local_inlier.any():
                        local_residual_median = float(np.median(local_residual[local_inlier]))
                local_accepted = bool(
                    int(local_inlier.sum()) >= MULTIVIEW_MINIMUM_SIM3_INLIERS
                    and float(np.mean(local_inlier)) >= MULTIVIEW_MINIMUM_SIM3_INLIER_RATIO
                )
                multiview_trials.append({
                    "family_id": family, "family_rank": rank,
                    "radio_family_score": radio_score, "support_view_count": family_views,
                    "raw_match_count": int(len(family_source_array)),
                    "local_sim3_inlier_count": int(local_inlier.sum()),
                    "local_sim3_inlier_ratio": float(np.mean(local_inlier)) if len(local_inlier) else 0.0,
                    "local_sim3_scale": local_scale,
                    "local_sim3_residual_median_m": local_residual_median,
                    "accepted": local_accepted,
                    "source": family_source_array[local_inlier] if local_accepted else np.zeros((0, 3)),
                    "target": family_target_array[local_inlier] if local_accepted else np.zeros((0, 3)),
                })
            accepted = [row for row in evaluated if row["accepted"]]
            best = max(
                accepted,
                key=lambda row: (row["homography_inlier_count"], row["homography_inlier_ratio"], row["confidence_median"], row["radio_family_score"], -row["family_rank"]),
                default=None,
            )
            if best is not None:
                source_points.extend(best["source"])
                target_points.extend(best["target"])
            accepted_multiview = [row for row in multiview_trials if row["accepted"]]
            best_multiview = max(
                accepted_multiview,
                key=lambda row: (row["local_sim3_inlier_count"], row["local_sim3_inlier_ratio"], row["radio_family_score"], -row["family_rank"]),
                default=None,
            )
            if best_multiview is not None:
                multiview_source_points.extend(best_multiview["source"])
                multiview_target_points.extend(best_multiview["target"])
            decisions.append({
                "query_region": query_row, "query_token_count": int(len(query["token_ids"])),
                "radio_candidate_family_count": len(candidates),
                "evaluated_member_pair_count": len(evaluated),
                "loftr_homography_accepted": best is not None,
                "selected_family_id": None if best is None else int(best["family_id"]),
                "selected_map_chart_name": None if best is None else str(best["map_chart_name"]),
                "selected_inlier_count": 0 if best is None else int(best["homography_inlier_count"]),
                "multiview_family_accepted": best_multiview is not None,
                "multiview_selected_family_id": None if best_multiview is None else int(best_multiview["family_id"]),
                "multiview_selected_inlier_count": 0 if best_multiview is None else int(best_multiview["local_sim3_inlier_count"]),
                "candidate_audits": [{k: v for k, v in row.items() if k not in ("source", "target")} for row in evaluated],
                "multiview_candidate_audits": [{k: v for k, v in row.items() if k not in ("source", "target")} for row in multiview_trials],
            })
        source = np.asarray(source_points, np.float64).reshape(-1, 3)
        target = np.asarray(target_points, np.float64).reshape(-1, 3)
        multiview_source = np.asarray(multiview_source_points, np.float64).reshape(-1, 3)
        multiview_target = np.asarray(multiview_target_points, np.float64).reshape(-1, 3)
        frozen.append({
            "name": name, "query_image_file_sha256": file_sha256(query_image_path),
            "query_radio_file_sha256": query_radio_hash,
            "query_plane_count": len(query_regions), "queried_source_chart_count": len(required_charts),
            "loftr_pair_inference_count": len(pair_cache),
            "loftr_match_count": int(sum(len(row["confidence"]) for row in pair_cache.values())),
            "token_correspondence_count": int(len(source)), "decisions": decisions,
            "metric_pose": _pose(source, target, scale=False),
            "scale_aware_pose": _pose(source, target, scale=True),
            "multiview_metric_pose": _pose(multiview_source, multiview_target, scale=False),
            "multiview_scale_aware_pose": _pose(multiview_source, multiview_target, scale=True),
            # Phase-two-only geometry, removed before JSON serialization.
            "_query_points": query_points,
            "_query_valid": query_valid,
            "_query_regions": query_regions,
        })
    # Phase two: target pose is opened only after all pair geometry and poses.
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    held_row = {name: row for row, name in enumerate(held.view_names.astype(str))}
    if set(held_row) != set(names):
        raise ValueError("held inventory differs from frozen LoFTR query inventory")
    rows = []
    family_trees = [cKDTree(np.asarray(row["points_world"])) for row in families]
    physical_audit = {
        "homography_selected": 0, "homography_selected_correct": 0,
        "multiview_selected": 0, "multiview_selected_correct": 0,
        "matchable": 0, "plane_count": 0,
    }
    for frozen_row in frozen:
        held_index = held_row[frozen_row["name"]]
        target_pose = held.camera_to_world[held_index]
        common = held.reference_valid[held_index] & np.asarray(frozen_row["_query_valid"])
        metric_scale = float(np.median(
            held.reference_depth_m[held_index][common] / np.asarray(frozen_row["_query_points"])[..., 2][common]
        ))
        audited_decisions = []
        for query, decision in zip(frozen_row["_query_regions"], frozen_row["decisions"]):
            world = metric_scale * (np.asarray(query["token_points"]) @ target_pose[:3, :3].T) + target_pose[:3, 3]
            normal = np.asarray(query["normal"]) @ target_pose[:3, :3].T
            compatible = []
            for family, family_row in enumerate(families):
                angle = float(np.degrees(np.arccos(np.clip(abs(float(normal @ family_row["normal_world"])), -1.0, 1.0))))
                if angle > 30.0:
                    continue
                distance = family_trees[family].query(world, k=1)[0]
                close = distance <= 1.50
                if int(close.sum()) >= 3 and float(np.mean(close)) >= 0.10 and float(np.median(distance[close])) <= 0.75:
                    compatible.append(family)
            homography_selected = decision["selected_family_id"]
            homography_correct = homography_selected is not None and int(homography_selected) in compatible
            multiview_selected = decision["multiview_selected_family_id"]
            multiview_correct = multiview_selected is not None and int(multiview_selected) in compatible
            physical_audit["plane_count"] += 1
            physical_audit["matchable"] += int(bool(compatible))
            physical_audit["homography_selected"] += int(homography_selected is not None)
            physical_audit["homography_selected_correct"] += int(homography_correct)
            physical_audit["multiview_selected"] += int(multiview_selected is not None)
            physical_audit["multiview_selected_correct"] += int(multiview_correct)
            audited_decisions.append({
                **decision, "physically_compatible_families": compatible,
                "physically_matchable": bool(compatible),
                "homography_selected_family_physically_correct": bool(homography_correct),
                "multiview_selected_family_physically_correct": bool(multiview_correct),
            })
        row = {key: value for key, value in frozen_row.items() if not key.startswith("_")}
        row["decisions"] = audited_decisions
        row["moge_metric_scale_postlabel"] = metric_scale
        for branch in ("metric_pose", "scale_aware_pose", "multiview_metric_pose", "multiview_scale_aware_pose"):
            pose = dict(row[branch])
            if pose["usable"]:
                rotation = np.asarray(pose.pop("rotation_c2w"), np.float64)
                translation = np.asarray(pose.pop("translation_world"), np.float64)
                pose["rotation_error_deg"] = _rotation_error_deg(rotation, target_pose)
                pose["translation_error_m"] = float(np.linalg.norm(translation - target_pose[:3, 3]))
            row[branch] = pose
        rows.append(row)
    physical_summary = {
        **physical_audit,
        "matchable_fraction": physical_audit["matchable"] / max(physical_audit["plane_count"], 1),
        "homography_selected_precision": physical_audit["homography_selected_correct"] / max(physical_audit["homography_selected"], 1),
        "homography_selected_matchable_recall": physical_audit["homography_selected_correct"] / max(physical_audit["matchable"], 1),
        "multiview_selected_precision": physical_audit["multiview_selected_correct"] / max(physical_audit["multiview_selected"], 1),
        "multiview_selected_matchable_recall": physical_audit["multiview_selected_correct"] / max(physical_audit["matchable"], 1),
    }
    payload = {
        "artifact_type": SCHEMA,
        "authority_file_sha256": file_sha256(args.authority),
        "authority_content_sha256": args.expected_authority_content_sha256,
        "alignment_manifest_file_sha256": file_sha256(args.alignment / "manifest.json") if args.alignment is not None else None,
        "map_geometry_mode": map_geometry_mode,
        "source_reference_pointmap_inventory_sha256": canonical_json_sha256(source_reference_hashes) if source_reference_hashes is not None else None,
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "moge3_query_manifest_file_sha256": file_sha256(query_manifest_path),
        "radio_manifest_file_sha256": file_sha256(args.radio_manifest),
        "map_radio_file_inventory_sha256": canonical_json_sha256(map_token_hash),
        "source_image_file_inventory_sha256": canonical_json_sha256(source_image_hashes),
        "loftr_weight_file_sha256": file_sha256(args.loftr_weight),
        "map_local_plane_count": len(local_regions), "map_large_plane_family_count": len(families),
        "matching_contract": {
            "RADIO_role": "top4_large_plane_family_retrieval_only",
            "local_matcher": "Kornia_LoFTR_outdoor_on_finite_plane_crops_letterboxed_to_384x384",
            "cropped_plane_patch_matching": CROPPED_PLANE_PATCH_MATCHING,
            "cropped_patch_size": CROPPED_PATCH_SIZE,
            "crop_background": "median_plane_intensity_to_avoid_artificial_mask_boundary",
            "loftr_confidence_minimum": LOFTR_CONFIDENCE_MINIMUM,
            "homography_minimum_inliers": HOMOGRAPHY_MINIMUM_INLIERS,
            "homography_minimum_inlier_ratio": HOMOGRAPHY_MINIMUM_INLIER_RATIO,
            "homography_ransac_threshold_px": HOMOGRAPHY_RANSAC_THRESHOLD_PX,
            "finite_plane_mask_dilation_cells": MASK_DILATION_CELLS,
            "finite_plane_mask_dilation_reason": "one_native_geometry_lattice_cell_boundary_uncertainty_control",
            "no_match_rule": "reject_query_plane_unless_a_retrieved_family_member_passes_LoFTR_homography",
            "pose_minimum_correspondences": POSE_MINIMUM_CORRESPONDENCES,
            "multiview_no_match_rule": "aggregate_same_family_across_source_views_then_require_local_Sim3_consensus",
            "multiview_minimum_raw_matches": MULTIVIEW_MINIMUM_RAW_MATCHES,
            "multiview_minimum_support_views": MULTIVIEW_MINIMUM_SUPPORT_VIEWS,
            "multiview_minimum_sim3_inliers": MULTIVIEW_MINIMUM_SIM3_INLIERS,
            "multiview_minimum_sim3_inlier_ratio": MULTIVIEW_MINIMUM_SIM3_INLIER_RATIO,
        },
        "phase_separation": {
            "query_pose_or_reference_used_by_matching": False,
            "all_RADIO_rankings_LoFTR_pairs_homographies_and_poses_frozen_before_held_open": True,
            "held_used_only_for_pose_error": True,
        },
        "metric_pose_summary": _summary(rows, "metric_pose"),
        "scale_aware_pose_summary": _summary(rows, "scale_aware_pose"),
        "multiview_metric_pose_summary": _summary(rows, "multiview_metric_pose"),
        "multiview_scale_aware_pose_summary": _summary(rows, "multiview_scale_aware_pose"),
        "held_postlabel_plane_selection_audit": physical_summary,
        "historical_held_control": True, "blind_or_preregistered_claim": False,
        "production_eligible": False, "promotion_eligible": False,
        "rows": rows,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        "metric": payload["metric_pose_summary"],
        "scale_aware": payload["scale_aware_pose_summary"],
        "multiview_metric": payload["multiview_metric_pose_summary"],
        "multiview_scale_aware": payload["multiview_scale_aware_pose_summary"],
        "plane_selection": payload["held_postlabel_plane_selection_audit"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
