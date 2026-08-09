"""G17.2 visual and quantitative audit of exact surface pose evidence.

This script is deliberately diagnostic: it freezes the candidate pool and
does not train, tune or refine a pose.  It checks the clean-2DGS round trip,
compares oracle/selected/phase-near-miss evidence, exposes readout attention,
and scans the raw surface score around the ground-truth pose.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import _camera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    encode_physical_instance_regions,
    load_physical_instance_readout,
    physical_instance_attention_weights,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    FEATURE_NAMES,
    extract_surface_likelihood_features,
)
from feature_extract.vfm.localization_goal_maplet.surface_refiner import canonical_alignment_score
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    dominant_child_owner,
    render_canonical_surface_field,
    render_surface_identity,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error
from feature_extract.vfm.official_2dgs_renderer import (
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)


def _pose_key(value: object) -> tuple[float, ...]:
    return tuple(np.asarray(value, dtype=np.float64).round(9).reshape(-1).tolist())


def _camera_center(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return -value[:3, :3].T @ value[:3, 3]


def _translate_camera(pose: np.ndarray, camera_axis: int, distance: float) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    center = _camera_center(value)
    direction = value[:3, :3].T[:, int(camera_axis)]
    center = center + float(distance) * direction
    value[:3, 3] = -value[:3, :3] @ center
    return value


def _rotate_camera(pose: np.ndarray, camera_axis: int, degrees: float) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
    angle = np.deg2rad(float(degrees))
    axis = np.zeros(3, dtype=np.float64)
    axis[int(camera_axis)] = 1.0
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    center = _camera_center(value)
    value[:3, :3] = delta @ value[:3, :3]
    value[:3, 3] = -value[:3, :3] @ center
    return value


def _pca_rgb(maps: list[np.ndarray]) -> list[np.ndarray]:
    channels = maps[0].shape[0]
    stacked = np.concatenate(
        [value.reshape(channels, -1).T for value in maps], axis=0,
    ).astype(np.float64)
    stacked -= np.mean(stacked, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(stacked[:: max(stacked.shape[0] // 12000, 1)], full_matrices=False)
    basis = vt[:3].T
    result = []
    projected_all = np.concatenate(
        [(value.reshape(channels, -1).T @ basis) for value in maps], axis=0,
    )
    low, high = np.percentile(projected_all, [2.0, 98.0], axis=0)
    for value in maps:
        projected = value.reshape(channels, -1).T @ basis
        projected = (projected - low) / np.maximum(high - low, 1.0e-8)
        result.append(np.clip(projected, 0.0, 1.0).reshape(value.shape[1], value.shape[2], 3))
    return result


def _render_role_field(physical, field, readout, pose, camera, query_shape, token_xy, device):
    rendered = render_canonical_surface_field(
        physical,
        field,
        pose,
        camera,
        width=int(query_shape[2]),
        height=int(query_shape[1]),
        device=str(device),
    )
    flat = encode_physical_instance_regions(
        readout,
        np.asarray(rendered.feature, dtype=np.float32),
        token_xy,
        role="context",
        device=str(device),
        spatial_valid_mask=np.asarray(rendered.mask, dtype=bool),
    )
    feature = flat.reshape(query_shape[1], query_shape[2], query_shape[0]).transpose(2, 0, 1)
    return replace(
        rendered,
        feature=feature,
        mode_feature=np.asarray(rendered.feature, dtype=np.float32),
    )


def _round_trip(labels, contributor_path, identity, rendered, physical):
    truth_ids = np.asarray(labels.topk_primitive_ids, dtype=np.int64)
    truth_valid = truth_ids[..., 0] >= 0
    rendered_valid = np.asarray(identity.mask, dtype=bool)
    rendered_id = np.full(rendered_valid.shape, -1, dtype=np.int64)
    rendered_id[rendered_valid] = physical.primitive_ids[
        np.asarray(identity.primitive_rows, dtype=np.int64)[rendered_valid]
    ]
    joint = truth_valid & rendered_valid
    identity_match = joint & (rendered_id == truth_ids[..., 0])
    top1 = float(np.mean(rendered_id[joint] == truth_ids[..., 0][joint])) if np.any(joint) else 0.0
    topk = (
        float(np.mean(np.any(rendered_id[joint][:, None] == truth_ids[joint], axis=1)))
        if np.any(joint) else 0.0
    )
    union = np.sum(truth_valid | rendered_valid)
    mask_iou = float(np.sum(joint) / max(int(union), 1))
    with np.load(contributor_path, allow_pickle=False) as data:
        truth_depth = np.asarray(data["dominant_depth"], dtype=np.float32)
    # Contributor depth is the gsplat primitive-center z, whereas the runtime
    # surface renderer deliberately reports the ray/primitive-plane
    # intersection.  The latter is useful geometry evidence but is not the
    # same quantity.  Evaluate both only where the dominant primitive agrees,
    # otherwise a small ID-disagreement tail dominates the RMS.
    plane_depth_valid = identity_match & (truth_depth > 0.0) & (np.asarray(rendered.depth) > 0.0)
    matched_plane_vs_center_depth_rms = (
        float(np.sqrt(np.mean(np.square(
            np.asarray(rendered.depth)[plane_depth_valid] - truth_depth[plane_depth_valid]
        ))))
        if np.any(plane_depth_valid) else None
    )
    pose = np.asarray(labels.pose_w2c, dtype=np.float64)
    identity_rows = np.asarray(identity.primitive_rows, dtype=np.int64)
    center_depth = np.zeros(rendered_valid.shape, dtype=np.float64)
    centers_camera = (
        physical.primitive_centers[identity_rows[rendered_valid]] @ pose[:3, :3].T
        + pose[:3, 3]
    )
    center_depth[rendered_valid] = centers_camera[:, 2]
    center_depth_valid = identity_match & (truth_depth > 0.0) & (center_depth > 0.0)
    matched_center_depth_rms = (
        float(np.sqrt(np.mean(np.square(center_depth[center_depth_valid] - truth_depth[center_depth_valid]))))
        if np.any(center_depth_valid) else None
    )
    row_by_id = np.full((int(np.max(physical.primitive_ids)) + 1,), -1, dtype=np.int64)
    row_by_id[physical.primitive_ids] = np.arange(physical.primitive_ids.size, dtype=np.int64)
    truth_top1 = truth_ids[..., 0]
    normal_valid = joint & (truth_top1 >= 0) & (truth_top1 < row_by_id.size)
    truth_rows = row_by_id[np.maximum(truth_top1[normal_valid], 0)]
    normal_valid_rows = truth_rows >= 0
    normal_error = None
    if np.any(normal_valid_rows):
        render_rows = identity_rows[normal_valid][normal_valid_rows]
        cosine_normal = np.abs(np.sum(
            physical.primitive_normals[render_rows]
            * physical.primitive_normals[truth_rows[normal_valid_rows]],
            axis=1,
        ))
        normal_error = float(np.median(np.degrees(np.arccos(np.clip(cosine_normal, -1.0, 1.0)))))
    return {
        "primitive_top1_agreement": top1,
        "primitive_topk_agreement": topk,
        "mask_iou": mask_iou,
        "matched_primitive_center_depth_rms_m": matched_center_depth_rms,
        "matched_primitive_plane_vs_center_depth_rms_m": matched_plane_vs_center_depth_rms,
        "depth_definition": (
            "contributor=dominant_primitive_center_z; "
            "runtime=ray_primitive_plane_intersection_z"
        ),
        "normal_angular_error_deg": normal_error,
        "feature_coverage": float(np.mean(np.asarray(rendered.mask, dtype=bool))),
        "truth_surface_coverage": float(np.mean(truth_valid)),
    }


def _plot_triptych(path: Path, query: np.ndarray, named_rendered: list[tuple[str, object]]) -> None:
    rgb = _pca_rgb([query] + [np.asarray(item.feature) for _name, item in named_rendered])
    columns = 5
    figure, axes = plt.subplots(len(named_rendered), columns, figsize=(15, 2.9 * len(named_rendered)))
    axes = np.asarray(axes).reshape(len(named_rendered), columns)
    for row, ((name, rendered), render_rgb) in enumerate(zip(named_rendered, rgb[1:])):
        cosine = np.sum(
            query / np.maximum(np.linalg.norm(query, axis=0, keepdims=True), 1.0e-8)
            * np.asarray(rendered.feature), axis=0,
        )
        difference = np.mean(np.abs(query - np.asarray(rendered.feature)), axis=0)
        images = (rgb[0], render_rgb, cosine, difference, np.asarray(rendered.field_missing, dtype=float))
        titles = ("query context PCA", f"{name} render PCA", "same-pixel cosine", "absolute difference", "field missing")
        for column, (image, title) in enumerate(zip(images, titles)):
            cmap = None if image.ndim == 3 else ("coolwarm" if column == 2 else "magma")
            axes[row, column].imshow(image, cmap=cmap, vmin=(-1 if column == 2 else None), vmax=(1 if column == 2 else None))
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_attention(path: Path, query_attention, render_attention, shape) -> None:
    center = query_attention["base_weight"].shape[1] // 2
    items = (
        ("query legacy center", query_attention["base_weight"][:, center]),
        ("query learned delta", query_attention["learned_delta"][:, center]),
        ("query final center", query_attention["final_weight"][:, center]),
        ("render legacy center", render_attention["base_weight"][:, center]),
        ("render learned delta", render_attention["learned_delta"][:, center]),
        ("render final center", render_attention["final_weight"][:, center]),
    )
    figure, axes = plt.subplots(2, 3, figsize=(10, 4.5))
    for axis, (title, value) in zip(axes.reshape(-1), items):
        axis.imshow(value.reshape(shape), cmap="viridis")
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _identity_rgb(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    identity = np.asarray(value, dtype=np.int64)
    color = np.zeros(identity.shape + (3,), dtype=np.float32)
    safe = np.maximum(identity, 0).astype(np.uint64)
    color[..., 0] = ((safe * np.uint64(2654435761)) & np.uint64(255)).astype(np.float32) / 255.0
    color[..., 1] = ((safe * np.uint64(2246822519) >> np.uint64(8)) & np.uint64(255)).astype(np.float32) / 255.0
    color[..., 2] = ((safe * np.uint64(3266489917) >> np.uint64(16)) & np.uint64(255)).astype(np.float32) / 255.0
    color[~np.asarray(valid, dtype=bool)] = 0.0
    return color


def _plot_geometry(path: Path, rendered, feature_rgb: np.ndarray, rgb: np.ndarray | None) -> None:
    mask = np.asarray(rendered.visibility, dtype=bool)
    normal_rgb = np.clip(0.5 * (np.asarray(rendered.normal) + 1.0), 0.0, 1.0)
    normal_rgb[~mask] = 0.0
    depth = np.asarray(rendered.depth, dtype=np.float32)
    panels = (
        ("clean 2DGS RGB", np.zeros(mask.shape + (3,), dtype=np.float32) if rgb is None else rgb, None),
        ("depth", np.where(mask, depth, np.nan), "viridis"),
        ("normal", normal_rgb, None),
        ("primitive ID", _identity_rgb(rendered.primitive_id, mask), None),
        ("parent ID", _identity_rgb(rendered.maplet_id, mask), None),
        ("child ID", _identity_rgb(rendered.child_id, mask), None),
        ("canonical feature PCA", feature_rgb, None),
        ("render valid", np.asarray(rendered.mask, dtype=float), "gray"),
        ("field missing", np.asarray(rendered.field_missing, dtype=float), "magma"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11, 7))
    for axis, (title, value, cmap) in zip(axes.reshape(-1), panels):
        axis.imshow(value, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_pose_landscape(path: Path, landscape: dict[str, dict[str, object]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (name, values) in zip(axes.reshape(-1), landscape.items()):
        offsets = np.asarray(values["offsets"], dtype=np.float64)
        scores = np.asarray(values["scores"], dtype=np.float64)
        coverage = np.asarray(values["coverage"], dtype=np.float64)
        axis.plot(offsets, scores, "o-", color="#2563eb", label="surface score")
        axis.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, label="GT")
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_xlabel("pose offset")
        axis.set_ylabel("score", color="#2563eb")
        coverage_axis = axis.twinx()
        coverage_axis.plot(offsets, coverage, "s:", color="#d97706", label="coverage")
        coverage_axis.set_ylabel("coverage", color="#d97706")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_candidate_cloud(path: Path, details: list[dict[str, object]], gt_pose: np.ndarray) -> None:
    centers = np.stack([_camera_center(np.asarray(item["pose_w2c"])) for item in details])
    forward = np.stack([
        np.asarray(item["pose_w2c"], dtype=np.float64).reshape(4, 4)[:3, :3].T[:, 2]
        for item in details
    ])
    scores = np.asarray([
        float(item.get("surface_alignment_score", item.get("score", 0.0)))
        for item in details
    ])
    gt_center = _camera_center(gt_pose)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2], c=scores,
        cmap="viridis", s=45, depthshade=False,
    )
    axis.quiver(
        centers[:, 0], centers[:, 1], centers[:, 2],
        forward[:, 0], forward[:, 1], forward[:, 2],
        length=0.35, normalize=True, color="#374151", linewidth=0.7,
    )
    for index, center in enumerate(centers):
        axis.text(center[0], center[1], center[2], str(index + 1), fontsize=7)
    axis.scatter(
        [gt_center[0]], [gt_center[1]], [gt_center[2]],
        marker="*", s=180, color="#dc2626", label="GT", depthshade=False,
    )
    axis.set_xlabel("world x")
    axis.set_ylabel("world y")
    axis.set_zlabel("world z")
    axis.legend(loc="best")
    figure.colorbar(points, ax=axis, shrink=0.7, label="frozen surface score")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_instance_readout", required=True)
    parser.add_argument("--gaussian_rgb_ply", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--maximum_queries", type=int, default=6)
    parser.add_argument("--scan_steps_m", nargs="+", type=float, default=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--scan_steps_deg", nargs="+", type=float, default=(-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0))
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite G17.2 diagnostic")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout, readout_metadata = load_physical_instance_readout(
        Path(args.physical_instance_readout), device=str(args.device),
    )
    if readout_metadata.get("physical_map_sha256") != physical.content_sha256:
        raise ValueError("readout and physical map differ")
    if readout_metadata.get("canonical_field_sha256") != field.content_sha256:
        raise ValueError("readout and canonical field differ")
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    rgb_source = (
        load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
        if args.gaussian_rgb_ply else None
    )
    pool = json.loads(Path(args.candidate_pool).read_text())
    if pool.get("physical_instance_readout_sha256") != file_sha256(Path(args.physical_instance_readout)):
        raise ValueError("candidate/readout lineage differs")
    contributor_by_image = {}
    for path in Path(args.contributors).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        contributor_by_image[str(metadata["image_id"])] = path
    source_rows = sorted(pool.get("rows", []), key=lambda row: str(row["image_id"]))
    source_rows = source_rows[int(args.shard_index) :: int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        source_rows = source_rows[: int(args.maximum_queries)]
    child_owner = dominant_child_owner(physical)
    rows = []
    translation_steps = [float(value) for value in args.scan_steps_m]
    angle_steps = [float(value) for value in args.scan_steps_deg]
    for row in source_rows:
        image_id = str(row["image_id"])
        contributor = contributor_by_image.get(image_id)
        if contributor is None:
            raise ValueError(f"missing diagnostic contributor: {image_id}")
        labels = ContributorLabels.load_npz(contributor)
        camera = _camera(contributor)
        with np.load(contributor, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        height, width = mapped.shape[1:]
        grid_y, grid_x = np.mgrid[:height, :width]
        token_xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        query_flat = encode_physical_instance_regions(
            readout, mapped, token_xy, role="context", device=str(args.device),
        )
        query = query_flat.reshape(height, width, mapped.shape[0]).transpose(2, 0, 1)
        details = list(row.get("mode_details", {}).get(str(args.mode_name), []))[:16]
        if not details:
            raise ValueError(f"no frozen pose candidates: {image_id}")
        oracle_index = int(np.argmin([
            float(item["translation_m"]) / 0.5 + float(item["rotation_deg"]) / 5.0
            for item in details
        ]))
        selected_index = 0
        near_candidates = [
            index for index, item in enumerate(details)
            if index != oracle_index and float(item["rotation_deg"]) <= 10.0
        ]
        near_index = min(
            near_candidates,
            key=lambda index: abs(float(details[index]["translation_m"]) - 0.75),
            default=(1 if len(details) > 1 else 0),
        )
        named_pose = [
            ("gt", labels.pose_w2c),
            ("oracle", np.asarray(details[oracle_index]["pose_w2c"], dtype=np.float64)),
            ("selected", np.asarray(details[selected_index]["pose_w2c"], dtype=np.float64)),
            ("phase_near_miss", np.asarray(details[near_index]["pose_w2c"], dtype=np.float64)),
        ]
        rendered_by_name = {
            name: _render_role_field(
                physical, field, readout, pose, camera, query.shape, token_xy, args.device,
            )
            for name, pose in named_pose
        }
        truth_height, truth_width = labels.topk_primitive_ids.shape[:2]
        identity = render_surface_identity(
            physical,
            labels.pose_w2c,
            camera,
            width=int(truth_width),
            height=int(truth_height),
            device=str(args.device),
        )
        round_trip_render = render_canonical_surface_field(
            physical,
            field,
            labels.pose_w2c,
            camera,
            width=int(truth_width),
            height=int(truth_height),
            device=str(args.device),
        )
        round_trip = _round_trip(
            labels, contributor, identity, round_trip_render, physical,
        )
        slug = image_id.replace("/", "__").replace(".png", "")
        _plot_triptych(
            output_dir / f"{slug}_feature_triptych.png",
            query,
            [(name, rendered_by_name[name]) for name in ("gt", "oracle", "selected", "phase_near_miss")],
        )
        gt_feature_rgb = _pca_rgb([query, np.asarray(rendered_by_name["gt"].feature)])[1]
        rendered_rgb = None
        if rgb_source is not None:
            rendered_rgb, _rgb_depth, _rgb_alpha = render_official_2dgs_rgb_depth(
                rgb_source,
                pose_w2c=labels.pose_w2c,
                camera=camera,
                width=width,
                height=height,
                device=str(args.device),
            )
        _plot_geometry(
            output_dir / f"{slug}_renderer_roundtrip.png",
            rendered_by_name["gt"],
            gt_feature_rgb,
            rendered_rgb,
        )
        query_attention = physical_instance_attention_weights(
            readout, mapped, token_xy, role="context", device=str(args.device),
        )
        selected_render = rendered_by_name["selected"]
        render_attention = physical_instance_attention_weights(
            readout,
            np.asarray(selected_render.mode_feature, dtype=np.float32),
            token_xy,
            role="context",
            device=str(args.device),
            spatial_valid_mask=np.asarray(selected_render.mask, dtype=bool),
        )
        _plot_attention(
            output_dir / f"{slug}_attention.png", query_attention, render_attention, (height, width),
        )
        candidate_report = []
        for name, pose in named_pose:
            rendered = rendered_by_name[name]
            token_feature, typed_target, _summary = extract_surface_likelihood_features(query, rendered, pose)
            cosine = token_feature[:, FEATURE_NAMES.index("cosine")].reshape(height, width)
            np.savez_compressed(
                output_dir / f"{slug}_{name}_token_evidence.npz",
                cosine=cosine.astype(np.float16),
                contribution=(cosine * np.asarray(rendered.mask) / float(height * width)).astype(np.float16),
                valid=np.asarray(rendered.mask, dtype=bool),
                field_missing=np.asarray(rendered.field_missing, dtype=bool),
                depth=np.asarray(rendered.depth, dtype=np.float16),
                normal=np.asarray(rendered.normal, dtype=np.float16),
                incidence=np.asarray(rendered.incidence, dtype=np.float16),
                uncertainty=np.asarray(rendered.uncertainty, dtype=np.float16),
                maplet_id=np.asarray(rendered.maplet_id, dtype=np.int32),
                child_row=np.where(
                    np.asarray(rendered.surface_id) >= 0,
                    child_owner[np.maximum(np.asarray(rendered.surface_id), 0)],
                    -1,
                ).astype(np.int32),
                typed_target=typed_target.reshape(height, width),
            )
            error = pnp_pose_error(pose, labels.pose_w2c)
            candidate_report.append({
                "name": name,
                "surface_score": canonical_alignment_score(rendered, query),
                "translation_m": float(error.translation_m),
                "rotation_deg": float(error.rotation_deg),
                "coverage": float(np.mean(rendered.mask)),
                "field_missing_fraction": float(np.mean(rendered.field_missing)),
                "camera_center": _camera_center(pose).tolist(),
            })
        landscape = {}
        scan_specs = {
            "facade_lateral_m": [(value, _translate_camera(labels.pose_w2c, 0, value)) for value in translation_steps],
            "facade_normal_m": [(value, _translate_camera(labels.pose_w2c, 2, value)) for value in translation_steps],
            "yaw_deg": [(value, _rotate_camera(labels.pose_w2c, 1, value)) for value in angle_steps],
            "pitch_deg": [(value, _rotate_camera(labels.pose_w2c, 0, value)) for value in angle_steps],
        }
        for scan_name, values in scan_specs.items():
            scores, coverages = [], []
            for value, pose in values:
                key = _pose_key(pose)
                reused = None
                for name, known_pose in named_pose:
                    if _pose_key(known_pose) == key:
                        reused = rendered_by_name[name]
                        break
                rendered = reused or _render_role_field(
                    physical, field, readout, pose, camera, query.shape, token_xy, args.device,
                )
                scores.append(canonical_alignment_score(rendered, query))
                coverages.append(float(np.mean(rendered.mask)))
            zero = int(np.argmin(np.abs(np.asarray([item[0] for item in values]))))
            landscape[scan_name] = {
                "offsets": [float(item[0]) for item in values],
                "scores": scores,
                "coverage": coverages,
                "gt_is_global_axis_peak": bool(int(np.argmax(scores)) == zero),
                "gt_is_local_axis_peak": bool(
                    scores[zero] >= scores[max(zero - 1, 0)]
                    and scores[zero] >= scores[min(zero + 1, len(scores) - 1)]
                ),
            }
        _plot_pose_landscape(
            output_dir / f"{slug}_pose_score_landscape.png", landscape,
        )
        _plot_candidate_cloud(
            output_dir / f"{slug}_candidate_pose_cloud.png", details, labels.pose_w2c,
        )
        rows.append({
            "image_id": image_id,
            "round_trip": round_trip,
            "candidate_evidence": candidate_report,
            "pose_score_landscape": landscape,
            "candidate_cloud": [
                {
                    "rank": int(item.get("rank", index + 1)),
                    "camera_center": item["camera_center"],
                    "translation_m": float(item["translation_m"]),
                    "rotation_deg": float(item["rotation_deg"]),
                    "score": float(item.get("surface_alignment_score", item["score"])),
                    "supporting_region_count": int(item.get("supporting_region_count", 0)),
                }
                for index, item in enumerate(details)
            ],
            "feature_triptych": str(output_dir / f"{slug}_feature_triptych.png"),
            "attention_visualization": str(output_dir / f"{slug}_attention.png"),
            "renderer_roundtrip_visualization": str(output_dir / f"{slug}_renderer_roundtrip.png"),
            "pose_score_landscape_visualization": str(output_dir / f"{slug}_pose_score_landscape.png"),
            "candidate_pose_cloud_visualization": str(output_dir / f"{slug}_candidate_pose_cloud.png"),
        })
        print(json.dumps({"image_id": image_id, "round_trip": round_trip}), flush=True)
    metrics = {
        key: [float(row["round_trip"][key]) for row in rows if row["round_trip"][key] is not None]
        for key in (
            "primitive_top1_agreement", "primitive_topk_agreement", "mask_iou",
            "matched_primitive_center_depth_rms_m",
            "matched_primitive_plane_vs_center_depth_rms_m",
            "normal_angular_error_deg", "feature_coverage",
        )
    }
    landscape_axes = [axis for row in rows for axis in row["pose_score_landscape"].values()]
    landscape_score = np.asarray([
        score for value in landscape_axes for score in value["scores"]
    ], dtype=np.float64)
    landscape_coverage = np.asarray([
        coverage for value in landscape_axes for coverage in value["coverage"]
    ], dtype=np.float64)
    score_coverage_correlation = (
        float(np.corrcoef(landscape_score, landscape_coverage)[0, 1])
        if landscape_score.size > 1
        and float(np.std(landscape_score)) > 0.0
        and float(np.std(landscape_coverage)) > 0.0
        else 0.0
    )
    phase_margins = []
    for row in rows:
        evidence = {value["name"]: value for value in row["candidate_evidence"]}
        phase_margins.append(
            float(evidence["gt"]["surface_score"] - evidence["phase_near_miss"]["surface_score"])
        )
    result = {
        "stage": "g17_2_exact_surface_visual_diagnostic",
        "candidate_pool_sha256": file_sha256(Path(args.candidate_pool)),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "physical_instance_readout_sha256": file_sha256(Path(args.physical_instance_readout)),
        "query_count": len(rows),
        "trajectory_ids": sorted({row["image_id"].split("/", 1)[0] for row in rows}),
        "summary": {
            key: {
                "median": float(np.median(values)) if values else None,
                "p90": float(np.percentile(values, 90.0)) if values else None,
            }
            for key, values in metrics.items()
        },
        "landscape_summary": {
            "landscape_local_peak_fraction": float(np.mean([
                value["gt_is_local_axis_peak"] for value in landscape_axes
            ])) if landscape_axes else 0.0,
            "landscape_global_peak_fraction": float(np.mean([
                value["gt_is_global_axis_peak"] for value in landscape_axes
            ])) if landscape_axes else 0.0,
            "score_coverage_correlation": score_coverage_correlation,
            "gt_minus_phase_near_miss_margin_median": (
                float(np.median(phase_margins)) if phase_margins else None
            ),
        },
        "diagnostic_contract": {
            "candidate_set_frozen_before_scoring": True,
            "uses_gt_for_diagnostic_only": True,
            "continuous_pose_updates": False,
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
