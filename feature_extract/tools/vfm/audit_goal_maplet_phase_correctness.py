"""Try to falsify Goal-Maplet phase evidence with rendering controls.

The audit expands near-phase negatives, applies common image roll, compares
1x/2x/4x mask-aware rendering, perturbs primitive density/opacity, and splits
phase evidence between primitive boundaries and interiors.  It stores only
scalar diagnostics and the existing poses, never query or mapping images.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import (
    directional_phase_statistics,
    dual_band_phase_evidence,
    load_phase_readout_policy,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.surface_renderer import render_canonical_surface_field


BANDS = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.5))


def _camera(path: Path) -> ColmapCamera:
    with np.load(path, allow_pickle=False) as data:
        return ColmapCamera(0, int(data["camera_model_id"]), int(data["camera_width"]), int(data["camera_height"]), tuple(np.asarray(data["camera_params"], dtype=np.float64).tolist()))


def _metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _query(path: Path, mapper) -> np.ndarray:
    metadata = _metadata(path)
    with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
        raw = np.asarray(data["radio_final"], dtype=np.float32)
    return np.asarray(mapper.project(raw).measurement_context, dtype=np.float32)


def _rotate_grid(feature: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = feature.shape
    matrix = cv2.getRotationMatrix2D(((width - 1.0) / 2.0, (height - 1.0) / 2.0), float(angle_deg), 1.0)
    source = np.asarray(feature, dtype=np.float32).transpose(1, 2, 0)
    rotated = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if rotated.ndim == 2:
        rotated = rotated[..., None]
    rotated_mask = cv2.warpAffine(np.asarray(mask, dtype=np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    return rotated.transpose(2, 0, 1), rotated_mask


def _phase_score(query: np.ndarray, rendered, policy) -> tuple[float, dict[str, float]]:
    evidence = dual_band_phase_evidence(query, np.asarray(rendered.feature, dtype=np.float32), query, np.asarray(rendered.feature, dtype=np.float32), np.asarray(rendered.mask, dtype=bool))
    return float(policy.score(evidence)), evidence.as_dict()


def _boundary_mask(rendered) -> np.ndarray:
    primitive = np.asarray(rendered.primitive_id, dtype=np.int64)
    valid = np.asarray(rendered.mask, dtype=bool) & (primitive >= 0)
    boundary = np.zeros(valid.shape, dtype=bool)
    changed = valid[:, 1:] & valid[:, :-1] & (primitive[:, 1:] != primitive[:, :-1])
    boundary[:, 1:] |= changed
    boundary[:, :-1] |= changed
    changed = valid[1:, :] & valid[:-1, :] & (primitive[1:, :] != primitive[:-1, :])
    boundary[1:, :] |= changed
    boundary[:-1, :] |= changed
    # Include the immediate support transition but do not let invalid pixels
    # become evidence.  The complement is a conservative primitive interior.
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool) & valid


def _region_phase(query: np.ndarray, rendered, selector: np.ndarray) -> dict[str, float]:
    values = []
    for dy, dx in ((0, 1), (1, 0)):
        values.append(directional_phase_statistics(query, np.asarray(rendered.feature), np.asarray(rendered.mask), delta_y=dy, delta_x=dx, edge_selector=selector))
    return {
        "conditional_phase": float(np.mean([value.conditional_score for value in values])),
        "observability": float(np.mean([value.observability for value in values])),
        "fixed_grid_phase": float(np.mean([value.fixed_grid_score for value in values])),
        "informative_edge_count": int(sum(value.informative_edge_count for value in values)),
        "grid_edge_count": int(sum(value.grid_edge_count for value in values)),
    }


def _perturbed_maps(physical: GoalMapletPhysicalMap, field: CanonicalSurfaceField) -> dict[str, tuple[GoalMapletPhysicalMap, CanonicalSurfaceField]]:
    ids = np.asarray(physical.primitive_ids, dtype=np.uint64)
    hashed = ids * np.uint64(11400714819323198485) + np.uint64(7046029254386353131)
    opacity = np.asarray(physical.primitive_opacity, dtype=np.float64)
    prune = opacity.copy()
    prune[(hashed % np.uint64(20)) == 0] = 0.0
    jitter_unit = ((hashed >> np.uint64(16)) % np.uint64(1001)).astype(np.float64) / 1000.0
    jitter = np.clip(opacity * (0.95 + 0.10 * jitter_unit), 0.0, 1.0)
    result = {}
    metadata = {key: value for key, value in dict(physical.metadata or {}).items() if key != "content_sha256"}
    for name, values in (("prune_5pct", prune), ("opacity_jitter_5pct", jitter)):
        changed_physical = replace(physical, primitive_opacity=values, metadata={**metadata, "diagnostic_perturbation": name})
        changed_field = replace(field, physical_map_sha256=changed_physical.content_sha256)
        result[name] = (changed_physical, changed_field)
    return result


def _select_candidates(source: dict[str, object], per_band: int) -> list[dict[str, object]]:
    details = list(source.get("mode_details", {}).get("actual_parent_actual_child", []))
    selected = []
    for low, high in BANDS:
        band = [row for row in details if low <= float(row["translation_m"]) < high]
        # Frozen proposal order is the only deterministic tie-break; GT is
        # used solely to assign evaluation bands, never by a runtime policy.
        for row in band[:max(int(per_band), 0)]:
            selected.append({**row, "translation_band_m": f"{low:g}-{high:g}"})
    return selected


def _pairwise_agreement(first: list[float], second: list[float]) -> float | None:
    values = []
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            delta_a = first[left] - first[right]
            delta_b = second[left] - second[right]
            if delta_a != 0.0 and delta_b != 0.0:
                values.append(np.sign(delta_a) == np.sign(delta_b))
    return float(np.mean(values)) if values else None


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    negatives = [row for row in rows if row["kind"] == "negative"]
    gt_by_image = {str(row["image_id"]): row for row in rows if row["kind"] == "ground_truth"}
    band_summary = {}
    for low, high in BANDS:
        name = f"{low:g}-{high:g}"
        selected = [row for row in negatives if row["translation_band_m"] == name]
        band_summary[name] = {
            "pair_count": len(selected),
            "gt_score_greater_fraction": float(np.mean([gt_by_image[str(row["image_id"])]["scores"]["supersample_1x"] > row["scores"]["supersample_1x"] for row in selected])) if selected else None,
            "phase_visible_margin_median": float(np.median([gt_by_image[str(row["image_id"])]["components"]["supersample_1x"]["phase_visible"] - row["components"]["supersample_1x"]["phase_visible"] for row in selected])) if selected else None,
            "observability_margin_median": float(np.median([gt_by_image[str(row["image_id"])]["components"]["supersample_1x"]["phase_observability"] - row["components"]["supersample_1x"]["phase_observability"] for row in selected])) if selected else None,
        }
    query_stability = []
    for image_id in sorted(gt_by_image):
        selected = [row for row in rows if str(row["image_id"]) == image_id]
        base = [float(row["scores"]["supersample_1x"]) for row in selected]
        item = {"image_id": image_id}
        for protocol in ("supersample_2x", "supersample_4x", "prune_5pct", "opacity_jitter_5pct"):
            other = [float(row["scores"][protocol]) for row in selected]
            item[f"pairwise_agreement_{protocol}"] = _pairwise_agreement(base, other)
            item[f"top1_preserved_{protocol}"] = int(np.argmax(base)) == int(np.argmax(other))
        for angle in (-20, -10, -5, 5, 10, 20):
            other = [float(row["roll_scores"][f"{angle:+d}deg"]) for row in selected]
            item[f"pairwise_agreement_roll_{angle:+d}deg"] = _pairwise_agreement(base, other)
            item[f"top1_preserved_roll_{angle:+d}deg"] = int(np.argmax(base)) == int(np.argmax(other))
        query_stability.append(item)
    scalar_keys = sorted({key for item in query_stability for key in item if key != "image_id"})
    stability = {key: float(np.mean([item[key] for item in query_stability if item.get(key) is not None])) for key in scalar_keys}
    gt_rows = list(gt_by_image.values())
    boundary_fraction = np.asarray([row["boundary"]["informative_edge_count"] for row in gt_rows], dtype=np.float64) / np.maximum(np.asarray([row["boundary"]["informative_edge_count"] + row["interior"]["informative_edge_count"] for row in gt_rows], dtype=np.float64), 1.0)
    return {
        "expanded_phase_pairs": band_summary,
        "candidate_order_stability": stability,
        "ground_truth_boundary_informative_fraction_median": float(np.median(boundary_fraction)),
        "ground_truth_boundary_conditional_phase_median": float(np.median([row["boundary"]["conditional_phase"] for row in gt_rows])),
        "ground_truth_interior_conditional_phase_median": float(np.median([row["interior"]["conditional_phase"] for row in gt_rows])),
        "per_query_stability": query_stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--phase_readout_model", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--per_band", type=int, default=2)
    parser.add_argument("--maximum_queries", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite phase-correctness report")
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    policy = load_phase_readout_policy(Path(args.phase_readout_model))
    pool = json.loads(Path(args.candidate_pool).read_text())
    if str(pool.get("physical_map_sha256", "")) != physical.content_sha256 or str(pool.get("canonical_field_sha256", "")) != field.content_sha256:
        raise ValueError("candidate pool lineage differs")
    contributor_by_image = {str(_metadata(path)["image_id"]): path for path in Path(args.contributors).glob("*.npz")}
    sources = sorted(pool.get("rows", []), key=lambda row: str(row["image_id"]))[int(args.shard_index)::int(args.shard_count)]
    if int(args.maximum_queries) > 0:
        sources = sources[:int(args.maximum_queries)]
    perturbations = _perturbed_maps(physical, field)
    rows = []
    for source in sources:
        image_id = str(source["image_id"])
        path = contributor_by_image[image_id]
        labels = ContributorLabels.load_npz(path)
        camera = _camera(path)
        query = _query(path, mapper)
        candidates = [{"pose_w2c": labels.pose_w2c, "kind": "ground_truth", "translation_m": 0.0, "translation_band_m": "ground_truth"}]
        candidates.extend({**value, "kind": "negative"} for value in _select_candidates(source, int(args.per_band)))
        for candidate_index, candidate in enumerate(candidates):
            pose = np.asarray(candidate["pose_w2c"], dtype=np.float64)
            record = {"image_id": image_id, "candidate_index": candidate_index, "kind": candidate["kind"], "translation_m": float(candidate["translation_m"]), "rotation_deg": float(candidate.get("rotation_deg", 0.0)), "translation_band_m": candidate["translation_band_m"], "scores": {}, "components": {}, "roll_scores": {}}
            rendered_by_factor = {}
            for factor in (1, 2, 4):
                rendered = render_canonical_surface_field(physical, field, pose, camera, width=int(query.shape[2]), height=int(query.shape[1]), supersample_factor=factor, device=str(args.device))
                score, components = _phase_score(query, rendered, policy)
                key = f"supersample_{factor}x"
                record["scores"][key] = score
                record["components"][key] = components
                rendered_by_factor[factor] = rendered
            rendered = rendered_by_factor[1]
            for angle in (-20, -10, -5, 5, 10, 20):
                rotated_query, _ = _rotate_grid(query, np.ones(query.shape[1:], dtype=bool), angle)
                rotated_render, rotated_mask = _rotate_grid(np.asarray(rendered.feature), np.asarray(rendered.mask), angle)
                evidence = dual_band_phase_evidence(rotated_query, rotated_render, rotated_query, rotated_render, rotated_mask)
                record["roll_scores"][f"{angle:+d}deg"] = float(policy.score(evidence))
            for name, (changed_physical, changed_field) in perturbations.items():
                changed = render_canonical_surface_field(changed_physical, changed_field, pose, camera, width=int(query.shape[2]), height=int(query.shape[1]), device=str(args.device))
                score, components = _phase_score(query, changed, policy)
                record["scores"][name] = score
                record["components"][name] = components
            boundary = _boundary_mask(rendered)
            interior = np.asarray(rendered.mask, dtype=bool) & ~boundary
            record["boundary"] = _region_phase(query, rendered, boundary)
            record["interior"] = _region_phase(query, rendered, interior)
            rows.append(record)
        print(json.dumps({"image_id": image_id, "candidate_count": len(candidates)}), flush=True)
    result = {"stage": "goal_maplet_phase_correctness_g19_c", "query_count": len(sources), "physical_map_sha256": physical.content_sha256, "canonical_field_sha256": field.content_sha256, "negative_bands_m": BANDS, "summary": _summary(rows), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
