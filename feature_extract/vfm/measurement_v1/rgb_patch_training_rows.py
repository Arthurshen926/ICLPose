from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(default) if value == "" else float(value)


def _str(row: Mapping[str, object], key: str, default: str = "") -> str:
    value = str(row.get(key, "")).strip()
    return value if value else str(default)


def _inside_margin(x: float, y: float, *, image_width: int, image_height: int, margin_px: float) -> bool:
    margin = float(margin_px)
    return margin <= float(x) <= float(image_width - 1) - margin and margin <= float(y) <= float(image_height - 1) - margin


def build_rgb_patch_measurement_rows(
    *,
    anchor_rows_csv: Path,
    output_rows_csv: Path,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    offsets_per_anchor: int = 1,
    max_rows: int | None = None,
    seed: int = 0,
    min_quality: float = 0.0,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Build query-side residual-delta training rows from fixed render anchors.

    For GT-render / near-GT basin training, the immutable anchor target in the
    query is the anchor render pixel. We perturb only the query-side center, so
    the supervised delta is `query_gt_xy - center_xy` while `render_xy` and the
    3D surface anchor remain fixed.
    """

    source_rows = _read_csv(Path(anchor_rows_csv))
    rng = random.Random(int(seed))
    crop_radius = float(search_radius_px) + float(context_radius_px)
    rows: list[dict[str, object]] = []
    usable_anchor_count = 0
    skipped_low_quality = 0
    skipped_boundary = 0
    for anchor in source_rows:
        quality = _float(anchor, "quality", default=1.0)
        if quality < float(min_quality):
            skipped_low_quality += 1
            continue
        render_x = _float(anchor, "render_x")
        render_y = _float(anchor, "render_y")
        if not _inside_margin(render_x, render_y, image_width=int(image_width), image_height=int(image_height), margin_px=crop_radius):
            skipped_boundary += 1
            continue
        usable_anchor_count += 1
        for offset_idx in range(int(offsets_per_anchor)):
            recorded_offset_idx = offset_idx if bool(include_identity) else offset_idx + 1
            # The first offset is deterministic center identity; later offsets
            # sample the near-GT basin. Keeping identity rows prevents the
            # branch from learning a mandatory nonzero correction.
            if bool(include_identity) and recorded_offset_idx == 0:
                dx = 0.0
                dy = 0.0
            else:
                dx = rng.uniform(-float(search_radius_px), float(search_radius_px))
                dy = rng.uniform(-float(search_radius_px), float(search_radius_px))
            center_x = render_x - dx
            center_y = render_y - dy
            if not _inside_margin(center_x, center_y, image_width=int(image_width), image_height=int(image_height), margin_px=crop_radius):
                continue
            rows.append(
                {
                    "query_id": _str(anchor, "query_id"),
                    "anchor_id": _str(anchor, "anchor_id"),
                    "candidate_id": _str(anchor, "candidate_id", "gt"),
                    "render_x": render_x,
                    "render_y": render_y,
                    "center_x": center_x,
                    "center_y": center_y,
                    "query_gt_x": render_x,
                    "query_gt_y": render_y,
                    "delta_x": render_x - center_x,
                    "delta_y": render_y - center_y,
                    "anchor_quality": quality,
                    "offset_index": recorded_offset_idx,
                }
            )
            if max_rows is not None and len(rows) >= int(max_rows):
                break
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    fieldnames = [
        "query_id",
        "anchor_id",
        "candidate_id",
        "render_x",
        "render_y",
        "center_x",
        "center_y",
        "query_gt_x",
        "query_gt_y",
        "delta_x",
        "delta_y",
        "anchor_quality",
        "offset_index",
    ]
    _write_csv(Path(output_rows_csv), rows, fieldnames)
    summary = {
        "stage": "measurement_v1_rgb_patch_training_rows",
        "anchor_rows_csv": str(anchor_rows_csv),
        "output_rows_csv": str(output_rows_csv),
        "source_anchor_count": int(len(source_rows)),
        "usable_anchor_count": int(usable_anchor_count),
        "row_count": int(len(rows)),
        "skipped_low_quality_count": int(skipped_low_quality),
        "skipped_boundary_count": int(skipped_boundary),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "offsets_per_anchor": int(offsets_per_anchor),
        "seed": int(seed),
        "min_quality": float(min_quality),
        "include_identity": bool(include_identity),
    }
    summary_path = Path(output_rows_csv).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
