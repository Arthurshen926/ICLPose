from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation, load_colmap_track_observations
from feature_extract.vfm.track_feature_sampling import load_colmap_track_observations_jsonl


REAL_REAL_MEASUREMENT_FIELDNAMES = [
    "query_id",
    "support_image_id",
    "track_id",
    "support_track_id",
    "track_length",
    "support_x",
    "support_y",
    "render_x",
    "render_y",
    "center_x",
    "center_y",
    "query_gt_x",
    "query_gt_y",
    "requested_residual_px",
    "target_is_dustbin",
    "support_reprojection_error",
    "query_reprojection_error",
    "support_frame_gap",
    "support_view_angle_deg",
]


def _within_margin(x: float, y: float, *, width: int, height: int, margin_px: float) -> bool:
    margin = float(margin_px)
    return margin <= float(x) < float(width) - margin and margin <= float(y) < float(height) - margin


def _image_size(obs: ColmapTrackObservation, *, image_width: int | None, image_height: int | None) -> tuple[int, int]:
    width = int(image_width if image_width is not None else (obs.image_width or 0))
    height = int(image_height if image_height is not None else (obs.image_height or 0))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be provided either as arguments or observation metadata")
    return width, height


def _scaled_xy(obs: ColmapTrackObservation, *, image_width: int | None, image_height: int | None) -> tuple[float, float]:
    target_w, target_h = _image_size(obs, image_width=image_width, image_height=image_height)
    source_w = int(obs.image_width or target_w)
    source_h = int(obs.image_height or target_h)
    if source_w <= 0 or source_h <= 0:
        raise ValueError("observation image dimensions must be positive")
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    return float(obs.xy[0]) * scale_x, float(obs.xy[1]) * scale_y


def _view_angle_deg(a: ColmapTrackObservation, b: ColmapTrackObservation) -> float | None:
    if a.viewing_ray is None or b.viewing_ray is None:
        return None
    ray_a = a.viewing_ray
    ray_b = b.viewing_ray
    norm_a = math.sqrt(float((ray_a * ray_a).sum()))
    norm_b = math.sqrt(float((ray_b * ray_b).sum()))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    cosine = max(-1.0, min(1.0, float((ray_a * ray_b).sum()) / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def _sequence_and_frame(image_id: str) -> tuple[str, int] | None:
    text = str(image_id)
    match = re.search(r"(^|/)(?P<seq>seq[^/]+)/frame(?P<frame>\d+)\.[^.]+$", text)
    if match is None:
        return None
    return str(match.group("seq")), int(match.group("frame"))


def _passes_sequence_frame_filter(
    support: ColmapTrackObservation,
    query: ColmapTrackObservation,
    *,
    same_sequence_only: bool,
    max_frame_gap: int | None,
) -> str | None:
    if not same_sequence_only and max_frame_gap is None:
        return None
    support_key = _sequence_and_frame(str(support.image_id))
    query_key = _sequence_and_frame(str(query.image_id))
    if support_key is None or query_key is None:
        return "unparseable_image_id"
    support_seq, support_frame = support_key
    query_seq, query_frame = query_key
    if same_sequence_only and support_seq != query_seq:
        return "different_sequence"
    if max_frame_gap is not None and abs(int(support_frame) - int(query_frame)) > int(max_frame_gap):
        return "large_frame_gap"
    return None


def _sample_offset(*, radius: float, rng: random.Random, outside_window: bool) -> tuple[float, float]:
    if bool(outside_window):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        if rng.random() < 0.5:
            return sign * float(radius), 0.0
        return 0.0, sign * float(radius)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return float(radius) * math.cos(angle), float(radius) * math.sin(angle)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REAL_REAL_MEASUREMENT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in REAL_REAL_MEASUREMENT_FIELDNAMES})


def _filter_observations_by_image_allowlist(
    observations: Sequence[ColmapTrackObservation],
    image_id_allowlist: set[str] | None,
) -> tuple[list[ColmapTrackObservation], int | None]:
    if image_id_allowlist is None:
        return list(observations), None
    allowed = {str(value).strip() for value in image_id_allowlist if str(value).strip()}
    return [obs for obs in observations if str(obs.image_id) in allowed], int(len(allowed))


def build_real_real_measurement_rows_from_observations(
    observations: Sequence[ColmapTrackObservation],
    *,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    dustbin_residual_bins_px: Sequence[float] = (),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    wrong_support_rows_per_positive: int = 0,
    max_rows: int | None = None,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    """Create real-real measurement rows from shared SfM track observations.

    Each row uses one real image observation as the immutable template/support
    patch and another observation of the same 3D track as the query target. The
    query center is synthetically offset from the query observation so the
    measurement branch must recover a known local residual flow.
    """

    if float(search_radius_px) <= 0.0 or float(context_radius_px) < 0.0:
        raise ValueError("search_radius_px must be positive and context_radius_px must be non-negative")
    bins = [float(value) for value in residual_bins_px if float(value) > 0.0]
    if not bins:
        raise ValueError("residual_bins_px must contain at least one positive value")
    max_residual = max(bins)
    if max_residual > float(search_radius_px) + 1e-8:
        raise ValueError("residual bins cannot exceed search_radius_px")
    dustbin_bins = [float(value) for value in dustbin_residual_bins_px if float(value) > 0.0]
    if any(value <= float(search_radius_px) for value in dustbin_bins):
        raise ValueError("dustbin residual bins must exceed search_radius_px")
    max_center_residual = max(bins + dustbin_bins)

    raw_observation_count = int(len(observations))
    observations, image_allowlist_count = _filter_observations_by_image_allowlist(observations, image_id_allowlist)

    grouped: dict[int, list[ColmapTrackObservation]] = defaultdict(list)
    skipped: dict[str, int] = {}
    wrong_support_index: dict[str, list[tuple[ColmapTrackObservation, float, float, int, int]]] = defaultdict(list)
    support_margin = float(search_radius_px) + float(context_radius_px)
    for obs in observations:
        if int(obs.track_length) < int(min_track_length):
            skipped["short_track"] = skipped.get("short_track", 0) + 1
            continue
        if max_reprojection_error is not None and float(obs.reprojection_error) > float(max_reprojection_error):
            skipped["high_reprojection_error"] = skipped.get("high_reprojection_error", 0) + 1
            continue
        grouped[int(obs.track_id)].append(obs)
        if int(wrong_support_rows_per_positive) > 0:
            width, height = _image_size(obs, image_width=image_width, image_height=image_height)
            sx, sy = _scaled_xy(obs, image_width=image_width, image_height=image_height)
            if _within_margin(sx, sy, width=width, height=height, margin_px=support_margin):
                wrong_support_index[str(obs.image_id)].append((obs, sx, sy, width, height))

    rng = random.Random(int(seed))
    candidates: list[dict[str, Any]] = []
    margin = float(search_radius_px) + float(context_radius_px) + max_center_residual
    candidate_pairs = 0
    wrong_support_rows = 0
    for track_id in sorted(grouped):
        track_obs = sorted(grouped[track_id], key=lambda item: (str(item.image_id), int(item.point2d_idx)))
        if len(track_obs) < 2:
            continue
        for support in track_obs:
            support_w, support_h = _image_size(support, image_width=image_width, image_height=image_height)
            sx, sy = _scaled_xy(support, image_width=image_width, image_height=image_height)
            if not _within_margin(sx, sy, width=support_w, height=support_h, margin_px=margin):
                skipped["support_near_boundary"] = skipped.get("support_near_boundary", 0) + 1
                continue
            for query in track_obs:
                if str(query.image_id) == str(support.image_id):
                    continue
                candidate_pairs += 1
                sequence_skip = _passes_sequence_frame_filter(
                    support,
                    query,
                    same_sequence_only=bool(same_sequence_only),
                    max_frame_gap=max_frame_gap,
                )
                if sequence_skip is not None:
                    skipped[sequence_skip] = skipped.get(sequence_skip, 0) + 1
                    continue
                if max_view_angle_deg is not None:
                    angle = _view_angle_deg(support, query)
                    if angle is not None and angle > float(max_view_angle_deg):
                        skipped["large_view_angle"] = skipped.get("large_view_angle", 0) + 1
                        continue
                query_w, query_h = _image_size(query, image_width=image_width, image_height=image_height)
                qx, qy = _scaled_xy(query, image_width=image_width, image_height=image_height)
                if not _within_margin(qx, qy, width=query_w, height=query_h, margin_px=margin):
                    skipped["query_near_boundary"] = skipped.get("query_near_boundary", 0) + 1
                    continue
                wrong_support_candidates: list[tuple[ColmapTrackObservation, float, float]] = []
                if int(wrong_support_rows_per_positive) > 0:
                    for wrong_support, wx, wy, wrong_w, wrong_h in wrong_support_index.get(str(support.image_id), []):
                        if int(wrong_support.track_id) == int(track_id):
                            continue
                        if wrong_w != support_w or wrong_h != support_h:
                            continue
                        wrong_support_candidates.append((wrong_support, wx, wy))
                offset_specs = [(value, False) for value in bins] + [(value, True) for value in dustbin_bins]
                for radius, target_is_dustbin in offset_specs:
                    dx, dy = _sample_offset(radius=float(radius), rng=rng, outside_window=bool(target_is_dustbin))
                    cx = qx - dx
                    cy = qy - dy
                    if not _within_margin(cx, cy, width=query_w, height=query_h, margin_px=float(search_radius_px) + float(context_radius_px)):
                        skipped["center_near_boundary"] = skipped.get("center_near_boundary", 0) + 1
                        continue
                    positive_row = {
                        "query_id": str(query.image_id),
                        "support_image_id": str(support.image_id),
                        "track_id": int(track_id),
                        "support_track_id": int(track_id),
                        "track_length": int(min(query.track_length, support.track_length)),
                        "support_x": sx,
                        "support_y": sy,
                        "render_x": sx,
                        "render_y": sy,
                        "center_x": cx,
                        "center_y": cy,
                        "query_gt_x": qx,
                        "query_gt_y": qy,
                        "requested_residual_px": float(radius),
                        "target_is_dustbin": bool(target_is_dustbin),
                        "support_reprojection_error": float(support.reprojection_error),
                        "query_reprojection_error": float(query.reprojection_error),
                    }
                    candidates.append(positive_row)
                    if not bool(target_is_dustbin) and wrong_support_candidates and int(wrong_support_rows_per_positive) > 0:
                        sampled_wrong_supports = rng.sample(
                            wrong_support_candidates,
                            k=min(int(wrong_support_rows_per_positive), len(wrong_support_candidates)),
                        )
                        for wrong_support, wx, wy in sampled_wrong_supports:
                            wrong_row = dict(positive_row)
                            wrong_row.update(
                                {
                                    "support_track_id": int(wrong_support.track_id),
                                    "track_length": int(min(query.track_length, wrong_support.track_length)),
                                    "support_x": wx,
                                    "support_y": wy,
                                    "render_x": wx,
                                    "render_y": wy,
                                    "target_is_dustbin": True,
                                    "support_reprojection_error": float(wrong_support.reprojection_error),
                                }
                            )
                            candidates.append(wrong_row)
                            wrong_support_rows += 1

    rng.shuffle(candidates)
    if max_rows is not None:
        candidates = candidates[: int(max_rows)]
    _write_rows(Path(output_rows_csv), candidates)
    return {
        "input_observations": int(len(observations)),
        "raw_input_observations": int(raw_observation_count),
        "image_allowlist_count": image_allowlist_count,
        "track_count": int(len(grouped)),
        "candidate_pairs": int(candidate_pairs),
        "output_rows": int(len(candidates)),
        "output_rows_csv": str(output_rows_csv),
        "skipped": dict(sorted(skipped.items())),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "residual_bins_px": bins,
        "dustbin_residual_bins_px": dustbin_bins,
        "dustbin_rows": int(sum(1 for row in candidates if bool(row.get("target_is_dustbin", False)))),
        "wrong_support_rows": int(wrong_support_rows),
        "wrong_support_rows_per_positive": int(wrong_support_rows_per_positive),
        "max_view_angle_deg": None if max_view_angle_deg is None else float(max_view_angle_deg),
        "same_sequence_only": bool(same_sequence_only),
        "max_frame_gap": None if max_frame_gap is None else int(max_frame_gap),
    }


def build_real_same_image_measurement_rows_from_observations(
    observations: Sequence[ColmapTrackObservation],
    *,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    dustbin_residual_bins_px: Sequence[float] = (),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    wrong_support_rows_per_positive: int = 0,
    max_rows: int | None = None,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    """Create same-image local-offset rows from SfM observations.

    This is a lower-bound sanity protocol: the template/support patch and the
    query target come from the same real image observation. It isolates whether
    the local cost-volume branch can recover subpixel residuals on real image
    texture before introducing cross-view or render-real domain shift.
    """

    bins = [float(value) for value in residual_bins_px if float(value) > 0.0]
    if not bins:
        raise ValueError("residual_bins_px must contain at least one positive value")
    if max(bins) > float(search_radius_px) + 1e-8:
        raise ValueError("residual bins cannot exceed search_radius_px")
    dustbin_bins = [float(value) for value in dustbin_residual_bins_px if float(value) > 0.0]
    if any(value <= float(search_radius_px) for value in dustbin_bins):
        raise ValueError("dustbin residual bins must exceed search_radius_px")
    raw_observation_count = int(len(observations))
    observations, image_allowlist_count = _filter_observations_by_image_allowlist(observations, image_id_allowlist)
    rng = random.Random(int(seed))
    margin = float(search_radius_px) + float(context_radius_px) + max(bins + dustbin_bins)
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    wrong_support_rows = 0
    wrong_support_index: dict[str, list[tuple[ColmapTrackObservation, float, float, int, int]]] = defaultdict(list)
    if int(wrong_support_rows_per_positive) > 0:
        support_margin = float(search_radius_px) + float(context_radius_px)
        for support in observations:
            if int(support.track_length) < int(min_track_length):
                continue
            if max_reprojection_error is not None and float(support.reprojection_error) > float(max_reprojection_error):
                continue
            swidth, sheight = _image_size(support, image_width=image_width, image_height=image_height)
            sx, sy = _scaled_xy(support, image_width=image_width, image_height=image_height)
            if not _within_margin(sx, sy, width=swidth, height=sheight, margin_px=support_margin):
                continue
            wrong_support_index[str(support.image_id)].append((support, sx, sy, swidth, sheight))
    for obs in observations:
        if int(obs.track_length) < int(min_track_length):
            skipped["short_track"] = skipped.get("short_track", 0) + 1
            continue
        if max_reprojection_error is not None and float(obs.reprojection_error) > float(max_reprojection_error):
            skipped["high_reprojection_error"] = skipped.get("high_reprojection_error", 0) + 1
            continue
        width, height = _image_size(obs, image_width=image_width, image_height=image_height)
        qx, qy = _scaled_xy(obs, image_width=image_width, image_height=image_height)
        if not _within_margin(qx, qy, width=width, height=height, margin_px=margin):
            skipped["near_boundary"] = skipped.get("near_boundary", 0) + 1
            continue
        wrong_support_candidates: list[tuple[ColmapTrackObservation, float, float]] = []
        if int(wrong_support_rows_per_positive) > 0:
            for support, sx, sy, swidth, sheight in wrong_support_index.get(str(obs.image_id), []):
                if int(support.track_id) == int(obs.track_id):
                    continue
                if swidth != width or sheight != height:
                    continue
                wrong_support_candidates.append((support, sx, sy))
        offset_specs = [(value, False) for value in bins] + [(value, True) for value in dustbin_bins]
        for radius, target_is_dustbin in offset_specs:
            dx, dy = _sample_offset(radius=float(radius), rng=rng, outside_window=bool(target_is_dustbin))
            positive_row = {
                "query_id": str(obs.image_id),
                "support_image_id": str(obs.image_id),
                "track_id": int(obs.track_id),
                "support_track_id": int(obs.track_id),
                "track_length": int(obs.track_length),
                "support_x": qx,
                "support_y": qy,
                "render_x": qx,
                "render_y": qy,
                "center_x": qx - dx,
                "center_y": qy - dy,
                "query_gt_x": qx,
                "query_gt_y": qy,
                "requested_residual_px": float(radius),
                "target_is_dustbin": bool(target_is_dustbin),
                "support_reprojection_error": float(obs.reprojection_error),
                "query_reprojection_error": float(obs.reprojection_error),
            }
            rows.append(positive_row)
            if not bool(target_is_dustbin) and wrong_support_candidates and int(wrong_support_rows_per_positive) > 0:
                sampled_wrong_supports = rng.sample(
                    wrong_support_candidates,
                    k=min(int(wrong_support_rows_per_positive), len(wrong_support_candidates)),
                )
                for wrong_support, sx, sy in sampled_wrong_supports:
                    rows.append(
                        {
                            "query_id": str(obs.image_id),
                            "support_image_id": str(wrong_support.image_id),
                            "track_id": int(obs.track_id),
                            "support_track_id": int(wrong_support.track_id),
                            "track_length": int(min(obs.track_length, wrong_support.track_length)),
                            "support_x": sx,
                            "support_y": sy,
                            "render_x": sx,
                            "render_y": sy,
                            "center_x": qx - dx,
                            "center_y": qy - dy,
                            "query_gt_x": qx,
                            "query_gt_y": qy,
                            "requested_residual_px": float(radius),
                            "target_is_dustbin": True,
                            "support_reprojection_error": float(wrong_support.reprojection_error),
                            "query_reprojection_error": float(obs.reprojection_error),
                        }
                    )
                    wrong_support_rows += 1
                    if max_rows is not None and len(rows) >= int(max_rows):
                        break
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    _write_rows(Path(output_rows_csv), rows)
    return {
        "input_observations": int(len(observations)),
        "raw_input_observations": int(raw_observation_count),
        "image_allowlist_count": image_allowlist_count,
        "output_rows": int(len(rows)),
        "output_rows_csv": str(output_rows_csv),
        "skipped": dict(sorted(skipped.items())),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "residual_bins_px": bins,
        "dustbin_residual_bins_px": dustbin_bins,
        "dustbin_rows": int(sum(1 for row in rows if bool(row.get("target_is_dustbin", False)))),
        "wrong_support_rows": int(wrong_support_rows),
        "wrong_support_rows_per_positive": int(wrong_support_rows_per_positive),
        "max_view_angle_deg": None if max_view_angle_deg is None else float(max_view_angle_deg),
        "same_image": True,
    }


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(int(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return 0.5 * float(sorted_values[mid - 1] + sorted_values[mid])


def _frame_gap(a: str, b: str) -> int | None:
    key_a = _sequence_and_frame(a)
    key_b = _sequence_and_frame(b)
    if key_a is None or key_b is None:
        return None
    seq_a, frame_a = key_a
    seq_b, frame_b = key_b
    if seq_a != seq_b:
        return None
    return abs(int(frame_a) - int(frame_b))


def build_real_real_query_measurement_rows_from_observations(
    observations: Sequence[ColmapTrackObservation],
    *,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (1.0,),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    max_tracks_per_query: int = 256,
    emit_all_residual_bins: bool = False,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    """Create one positive local-measurement row per query/track for PnP proxy coverage.

    The training-row builder intentionally creates many offset bins per image
    pair. That is inappropriate for pose proxy evaluation because PnP needs
    broad unique-track coverage per query, not repeated rows for a few sampled
    pairs. This builder fixes that protocol: each output row is one fixed SfM
    track anchor for one query image, paired with a single support observation
    of the same track and a synthetic near-GT center offset.
    """

    if float(search_radius_px) <= 0.0 or float(context_radius_px) < 0.0:
        raise ValueError("search_radius_px must be positive and context_radius_px must be non-negative")
    bins = [float(value) for value in residual_bins_px if float(value) > 0.0]
    if not bins:
        raise ValueError("residual_bins_px must contain at least one positive value")
    if max(bins) > float(search_radius_px) + 1e-8:
        raise ValueError("residual bins cannot exceed search_radius_px")
    if int(max_tracks_per_query) <= 0:
        raise ValueError("max_tracks_per_query must be positive")

    raw_observation_count = int(len(observations))
    observations, image_allowlist_count = _filter_observations_by_image_allowlist(observations, image_id_allowlist)

    filtered: list[ColmapTrackObservation] = []
    skipped: dict[str, int] = {}
    for obs in observations:
        if int(obs.track_length) < int(min_track_length):
            skipped["short_track"] = skipped.get("short_track", 0) + 1
            continue
        if max_reprojection_error is not None and float(obs.reprojection_error) > float(max_reprojection_error):
            skipped["high_reprojection_error"] = skipped.get("high_reprojection_error", 0) + 1
            continue
        filtered.append(obs)

    grouped: dict[int, list[ColmapTrackObservation]] = defaultdict(list)
    for obs in filtered:
        grouped[int(obs.track_id)].append(obs)

    rng = random.Random(int(seed))
    support_margin = float(search_radius_px) + float(context_radius_px)
    query_margin = support_margin + max(bins)
    candidates_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_query_tracks = 0
    for track_id in sorted(grouped):
        track_obs = sorted(grouped[track_id], key=lambda item: (str(item.image_id), int(item.point2d_idx)))
        if len(track_obs) < 2:
            continue
        for query in track_obs:
            query_w, query_h = _image_size(query, image_width=image_width, image_height=image_height)
            qx, qy = _scaled_xy(query, image_width=image_width, image_height=image_height)
            if not _within_margin(qx, qy, width=query_w, height=query_h, margin_px=query_margin):
                skipped["query_near_boundary"] = skipped.get("query_near_boundary", 0) + 1
                continue
            support_candidates: list[tuple[ColmapTrackObservation, float, float, float | None, int | None]] = []
            for support in track_obs:
                if str(support.image_id) == str(query.image_id):
                    continue
                sequence_skip = _passes_sequence_frame_filter(
                    support,
                    query,
                    same_sequence_only=bool(same_sequence_only),
                    max_frame_gap=max_frame_gap,
                )
                if sequence_skip is not None:
                    skipped[sequence_skip] = skipped.get(sequence_skip, 0) + 1
                    continue
                angle = _view_angle_deg(support, query)
                if max_view_angle_deg is not None and angle is not None and angle > float(max_view_angle_deg):
                    skipped["large_view_angle"] = skipped.get("large_view_angle", 0) + 1
                    continue
                support_w, support_h = _image_size(support, image_width=image_width, image_height=image_height)
                sx, sy = _scaled_xy(support, image_width=image_width, image_height=image_height)
                if not _within_margin(sx, sy, width=support_w, height=support_h, margin_px=support_margin):
                    skipped["support_near_boundary"] = skipped.get("support_near_boundary", 0) + 1
                    continue
                support_candidates.append((support, sx, sy, angle, _frame_gap(str(support.image_id), str(query.image_id))))
            if not support_candidates:
                skipped["no_support_candidate"] = skipped.get("no_support_candidate", 0) + 1
                continue
            candidate_query_tracks += 1
            support, sx, sy, angle, gap = min(
                support_candidates,
                key=lambda item: (
                    10**9 if item[4] is None else int(item[4]),
                    10**9 if item[3] is None else float(item[3]),
                    float(item[0].reprojection_error),
                    str(item[0].image_id),
                    int(item[0].point2d_idx),
                ),
            )
            candidates_by_query[str(query.image_id)].append(
                {
                    "query_id": str(query.image_id),
                    "support_image_id": str(support.image_id),
                    "track_id": int(track_id),
                    "support_track_id": int(track_id),
                    "track_length": int(min(query.track_length, support.track_length)),
                    "support_x": sx,
                    "support_y": sy,
                    "render_x": sx,
                    "render_y": sy,
                    "query_width": int(query_w),
                    "query_height": int(query_h),
                    "query_gt_x": qx,
                    "query_gt_y": qy,
                    "target_is_dustbin": False,
                    "support_reprojection_error": float(support.reprojection_error),
                    "query_reprojection_error": float(query.reprojection_error),
                    "support_frame_gap": "" if gap is None else int(gap),
                    "support_view_angle_deg": "" if angle is None else float(angle),
                }
            )

    output_rows: list[dict[str, Any]] = []
    rows_per_query: list[int] = []
    unique_tracks_per_query: list[int] = []
    for query_id in sorted(candidates_by_query):
        query_rows = sorted(
            candidates_by_query[query_id],
            key=lambda row: (
                -int(row["track_length"]),
                float(row["query_reprojection_error"]) + float(row["support_reprojection_error"]),
                int(row["track_id"]),
                str(row["support_image_id"]),
            ),
        )[: int(max_tracks_per_query)]
        materialized_rows: list[dict[str, Any]] = []
        for base_row in query_rows:
            radius_values = bins if bool(emit_all_residual_bins) else [float(rng.choice(bins))]
            for radius in radius_values:
                dx, dy = _sample_offset(radius=float(radius), rng=rng, outside_window=False)
                cx = float(base_row["query_gt_x"]) - dx
                cy = float(base_row["query_gt_y"]) - dy
                if not _within_margin(
                    cx,
                    cy,
                    width=int(base_row["query_width"]),
                    height=int(base_row["query_height"]),
                    margin_px=support_margin,
                ):
                    skipped["center_near_boundary"] = skipped.get("center_near_boundary", 0) + 1
                    continue
                row = dict(base_row)
                row.pop("query_width", None)
                row.pop("query_height", None)
                row.update(
                    {
                        "center_x": cx,
                        "center_y": cy,
                        "requested_residual_px": float(radius),
                    }
                )
                materialized_rows.append(row)
        rows_per_query.append(len(materialized_rows))
        unique_tracks_per_query.append(len(query_rows))
        output_rows.extend(materialized_rows)

    _write_rows(Path(output_rows_csv), output_rows)
    return {
        "input_observations": int(len(observations)),
        "raw_input_observations": int(raw_observation_count),
        "image_allowlist_count": image_allowlist_count,
        "filtered_observations": int(len(filtered)),
        "track_count": int(len(grouped)),
        "candidate_query_tracks": int(candidate_query_tracks),
        "query_count": int(len(rows_per_query)),
        "output_rows": int(len(output_rows)),
        "output_rows_csv": str(output_rows_csv),
        "rows_per_query_min": int(min(rows_per_query)) if rows_per_query else 0,
        "rows_per_query_median": float(_median(rows_per_query)),
        "rows_per_query_max": int(max(rows_per_query)) if rows_per_query else 0,
        "unique_tracks_per_query_min": int(min(unique_tracks_per_query)) if unique_tracks_per_query else 0,
        "unique_tracks_per_query_median": float(_median(unique_tracks_per_query)),
        "unique_tracks_per_query_max": int(max(unique_tracks_per_query)) if unique_tracks_per_query else 0,
        "max_tracks_per_query": int(max_tracks_per_query),
        "emit_all_residual_bins": bool(emit_all_residual_bins),
        "skipped": dict(sorted(skipped.items())),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "residual_bins_px": bins,
        "dustbin_rows": 0,
        "wrong_support_rows": 0,
        "max_view_angle_deg": None if max_view_angle_deg is None else float(max_view_angle_deg),
        "same_sequence_only": bool(same_sequence_only),
        "max_frame_gap": None if max_frame_gap is None else int(max_frame_gap),
    }


def build_real_real_query_measurement_rows_from_colmap_model(
    *,
    model_dir: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (1.0,),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    max_tracks_per_query: int = 256,
    emit_all_residual_bins: bool = False,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations(Path(model_dir), min_track_length=int(min_track_length))
    return build_real_real_query_measurement_rows_from_observations(
        observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        residual_bins_px=residual_bins_px,
        min_track_length=int(min_track_length),
        max_reprojection_error=max_reprojection_error,
        max_view_angle_deg=max_view_angle_deg,
        same_sequence_only=bool(same_sequence_only),
        max_frame_gap=max_frame_gap,
        max_tracks_per_query=int(max_tracks_per_query),
        emit_all_residual_bins=bool(emit_all_residual_bins),
        seed=int(seed),
        image_id_allowlist=image_id_allowlist,
    )


def build_real_real_query_measurement_rows_from_jsonl(
    *,
    track_observations_jsonl: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (1.0,),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    max_tracks_per_query: int = 256,
    emit_all_residual_bins: bool = False,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations_jsonl(Path(track_observations_jsonl))
    return build_real_real_query_measurement_rows_from_observations(
        observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        residual_bins_px=residual_bins_px,
        min_track_length=int(min_track_length),
        max_reprojection_error=max_reprojection_error,
        max_view_angle_deg=max_view_angle_deg,
        same_sequence_only=bool(same_sequence_only),
        max_frame_gap=max_frame_gap,
        max_tracks_per_query=int(max_tracks_per_query),
        emit_all_residual_bins=bool(emit_all_residual_bins),
        seed=int(seed),
        image_id_allowlist=image_id_allowlist,
    )


def build_real_real_measurement_rows_from_colmap_model(
    *,
    model_dir: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    dustbin_residual_bins_px: Sequence[float] = (),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    wrong_support_rows_per_positive: int = 0,
    max_rows: int | None = None,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations(Path(model_dir), min_track_length=int(min_track_length))
    return build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        residual_bins_px=residual_bins_px,
        dustbin_residual_bins_px=dustbin_residual_bins_px,
        min_track_length=int(min_track_length),
        max_reprojection_error=max_reprojection_error,
        max_view_angle_deg=max_view_angle_deg,
        same_sequence_only=bool(same_sequence_only),
        max_frame_gap=max_frame_gap,
        wrong_support_rows_per_positive=int(wrong_support_rows_per_positive),
        max_rows=max_rows,
        seed=int(seed),
        image_id_allowlist=image_id_allowlist,
    )


def build_real_same_image_measurement_rows_from_jsonl(
    *,
    track_observations_jsonl: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    dustbin_residual_bins_px: Sequence[float] = (),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    wrong_support_rows_per_positive: int = 0,
    max_rows: int | None = None,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations_jsonl(Path(track_observations_jsonl))
    return build_real_same_image_measurement_rows_from_observations(
        observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        residual_bins_px=residual_bins_px,
        dustbin_residual_bins_px=dustbin_residual_bins_px,
        min_track_length=int(min_track_length),
        max_reprojection_error=max_reprojection_error,
        max_view_angle_deg=max_view_angle_deg,
        wrong_support_rows_per_positive=int(wrong_support_rows_per_positive),
        max_rows=max_rows,
        seed=int(seed),
        image_id_allowlist=image_id_allowlist,
    )


def build_real_real_measurement_rows_from_jsonl(
    *,
    track_observations_jsonl: Path,
    output_rows_csv: Path,
    image_width: int | None = None,
    image_height: int | None = None,
    search_radius_px: float,
    context_radius_px: float,
    residual_bins_px: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    dustbin_residual_bins_px: Sequence[float] = (),
    min_track_length: int = 2,
    max_reprojection_error: float | None = None,
    max_view_angle_deg: float | None = None,
    same_sequence_only: bool = False,
    max_frame_gap: int | None = None,
    wrong_support_rows_per_positive: int = 0,
    max_rows: int | None = None,
    seed: int = 0,
    image_id_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    observations = load_colmap_track_observations_jsonl(Path(track_observations_jsonl))
    return build_real_real_measurement_rows_from_observations(
        observations,
        output_rows_csv=Path(output_rows_csv),
        image_width=image_width,
        image_height=image_height,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        residual_bins_px=residual_bins_px,
        dustbin_residual_bins_px=dustbin_residual_bins_px,
        min_track_length=int(min_track_length),
        max_reprojection_error=max_reprojection_error,
        max_view_angle_deg=max_view_angle_deg,
        same_sequence_only=bool(same_sequence_only),
        max_frame_gap=max_frame_gap,
        wrong_support_rows_per_positive=int(wrong_support_rows_per_positive),
        max_rows=max_rows,
        seed=int(seed),
        image_id_allowlist=image_id_allowlist,
    )
