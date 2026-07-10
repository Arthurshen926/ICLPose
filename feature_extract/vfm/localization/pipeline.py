"""Real-image RADIO selector + coarse + measurement pipeline helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from feature_extract.vfm.localization.model import SelectorCoarseMeasurementModel
from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _first_text(row: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _optional_xy(row: Mapping[str, object], x_names: Sequence[str], y_names: Sequence[str]) -> np.ndarray | None:
    x_text = _first_text(row, *x_names)
    y_text = _first_text(row, *y_names)
    if not x_text or not y_text:
        return None
    return np.asarray([float(x_text), float(y_text)], dtype=np.float32)


def _image_stem(image_id: str) -> str:
    text = str(image_id).replace("\\", "/").strip("/")
    stem = str(Path(text).with_suffix(""))
    return stem.replace("/", "_")


def _image_token(image_id: str) -> str:
    return str(image_id).replace("\\", "/").strip("/").replace("/", "__")


def _feature_path_from_template(template: str, *, image_id: str) -> Path:
    if not str(template).strip():
        raise ValueError("feature path missing and no feature_path_template was provided")
    return Path(
        str(template).format(
            image_id=str(image_id),
            image_stem=_image_stem(str(image_id)),
            image_token=_image_token(str(image_id)),
        )
    )


def _load_feature_map(path: Path, *, key: str = "") -> np.ndarray:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(value)
    if value.suffix == ".npy":
        arr = np.load(value)
    else:
        with np.load(value) as data:
            if str(key):
                if str(key) not in data:
                    raise KeyError(f"{value} does not contain feature key {key!r}")
                arr = data[str(key)]
            else:
                keys = list(data.files)
                preferred = [name for name in ("radio_dual", "feature_map", "feature", "arr_0") if name in data]
                if len(keys) == 1:
                    arr = data[keys[0]]
                elif preferred:
                    arr = data[preferred[0]]
                else:
                    raise ValueError(f"{value} contains multiple arrays; pass --feature_key")
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3:
        raise ValueError(f"feature map at {value} must have shape (C,H,W)")
    return out


def _load_rgb_chw(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(Path(path)).convert("RGB"), dtype=np.float32) / 255.0
    return np.moveaxis(arr, -1, 0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RealRadioGroundTruth:
    query_xy: np.ndarray
    reference_xy: np.ndarray

    def __post_init__(self) -> None:
        query = np.asarray(self.query_xy, dtype=np.float32).reshape(2)
        reference = np.asarray(self.reference_xy, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(query)) or not np.all(np.isfinite(reference)):
            raise ValueError("ground-truth coordinates must be finite")
        object.__setattr__(self, "query_xy", query)
        object.__setattr__(self, "reference_xy", reference)


@dataclass(frozen=True)
class RealRadioLocalizationPair:
    query_id: str
    reference_image_id: str
    query_feature_path: Path
    reference_feature_path: Path
    query_gt_xy: np.ndarray | None = None
    reference_gt_xy: np.ndarray | None = None
    ground_truth: tuple[RealRadioGroundTruth, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.query_id).strip():
            raise ValueError("query_id is required")
        if not str(self.reference_image_id).strip():
            raise ValueError("reference_image_id is required")
        gt = list(self.ground_truth)
        if self.query_gt_xy is not None or self.reference_gt_xy is not None:
            if self.query_gt_xy is None or self.reference_gt_xy is None:
                raise ValueError("query_gt_xy and reference_gt_xy must be provided together")
            gt.append(RealRadioGroundTruth(self.query_gt_xy, self.reference_gt_xy))
        object.__setattr__(self, "query_id", str(self.query_id))
        object.__setattr__(self, "reference_image_id", str(self.reference_image_id))
        object.__setattr__(self, "query_feature_path", Path(self.query_feature_path))
        object.__setattr__(self, "reference_feature_path", Path(self.reference_feature_path))
        object.__setattr__(self, "ground_truth", tuple(gt))
        object.__setattr__(self, "metadata", dict(self.metadata))


def load_real_radio_localization_pairs_csv(
    path: Path,
    *,
    feature_path_template: str = "",
) -> list[RealRadioLocalizationPair]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(_read_csv(Path(path))):
        query_id = _first_text(row, "query_id")
        reference_id = _first_text(row, "reference_image_id", "support_image_id")
        if not query_id or not reference_id:
            raise ValueError("pairs_csv rows must contain query_id and reference_image_id/support_image_id")
        query_feature = _first_text(row, "query_feature_path")
        reference_feature = _first_text(row, "reference_feature_path", "support_feature_path")
        if not query_feature:
            query_feature = str(_feature_path_from_template(str(feature_path_template), image_id=query_id))
        if not reference_feature:
            reference_feature = str(_feature_path_from_template(str(feature_path_template), image_id=reference_id))
        key = (query_id, reference_id, query_feature, reference_feature)
        item = grouped.setdefault(
            key,
            {
                "query_id": query_id,
                "reference_image_id": reference_id,
                "query_feature_path": Path(query_feature),
                "reference_feature_path": Path(reference_feature),
                "ground_truth": [],
                "metadata": {"row_indices": []},
            },
        )
        item["metadata"]["row_indices"].append(int(row_index))
        query_gt = _optional_xy(row, ("query_gt_x", "gt_query_x"), ("query_gt_y", "gt_query_y"))
        reference_gt = _optional_xy(
            row,
            ("reference_gt_x", "support_x", "render_x", "gt_reference_x"),
            ("reference_gt_y", "support_y", "render_y", "gt_reference_y"),
        )
        if query_gt is not None and reference_gt is not None:
            item["ground_truth"].append(RealRadioGroundTruth(query_gt, reference_gt))
    return [
        RealRadioLocalizationPair(
            query_id=str(item["query_id"]),
            reference_image_id=str(item["reference_image_id"]),
            query_feature_path=Path(item["query_feature_path"]),
            reference_feature_path=Path(item["reference_feature_path"]),
            ground_truth=tuple(item["ground_truth"]),
            metadata=dict(item["metadata"]),
        )
        for item in grouped.values()
    ]


def _proposal_key(proposal: CoarseProposal) -> tuple[int, int, int]:
    return (int(proposal.query_index), int(proposal.reference_index), -1 if proposal.rank is None else int(proposal.rank))


def _nearest_ground_truth(
    pair: RealRadioLocalizationPair,
    proposal: CoarseProposal,
    *,
    max_reference_distance_px: float,
) -> tuple[RealRadioGroundTruth | None, float | None]:
    if not pair.ground_truth:
        return None, None
    distances = [
        float(np.linalg.norm(np.asarray(proposal.reference_xy, dtype=np.float32) - np.asarray(gt.reference_xy, dtype=np.float32)))
        for gt in pair.ground_truth
    ]
    best = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    distance = float(distances[best])
    if distance > float(max_reference_distance_px):
        return None, distance
    return pair.ground_truth[best], distance


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return None if not finite else float(np.median(np.asarray(finite, dtype=np.float32)))


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return None if not finite else float(np.mean(np.asarray(finite, dtype=np.float32)))


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "query_id",
        "reference_image_id",
        "proposal_index",
        "query_index",
        "reference_index",
        "rank",
        "coarse_score",
        "coarse_confidence",
        "query_x",
        "query_y",
        "reference_x",
        "reference_y",
        "measured_query_x",
        "measured_query_y",
        "measured_reference_x",
        "measured_reference_y",
        "measurement_confidence",
        "measurement_uncertainty_px",
        "gt_query_x",
        "gt_query_y",
        "gt_reference_x",
        "gt_reference_y",
        "gt_reference_distance_px",
        "coarse_query_epe_px",
        "measurement_query_epe_px",
    ]
    seen = set(preferred)
    out = list(preferred)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                out.append(str(key))
    return out


def run_real_radio_localization_pairs(
    pairs: Sequence[RealRadioLocalizationPair],
    *,
    image_root: Path,
    feature_root: Path,
    output_dir: Path,
    feature_mapper,
    coarse_matcher,
    measurement_branch,
    feature_key: str = "",
    max_pairs: int | None = None,
    gt_reference_radius_px: float = 4.0,
) -> dict[str, Any]:
    selected_pairs = list(pairs)
    if max_pairs is not None:
        selected_pairs = selected_pairs[: int(max_pairs)]
    model = SelectorCoarseMeasurementModel(
        feature_mapper=feature_mapper,
        coarse_matcher=coarse_matcher,
        measurement_branch=measurement_branch,
    )
    rows: list[dict[str, Any]] = []
    coarse_epe: list[float] = []
    paired_coarse_epe: list[float] = []
    measurement_epe: list[float] = []
    measured_count = 0
    gt_count = 0
    for pair in selected_pairs:
        query_rgb = _load_rgb_chw(Path(image_root) / pair.query_id)
        reference_rgb = _load_rgb_chw(Path(image_root) / pair.reference_image_id)
        query_feature = _load_feature_map(_resolve_path(pair.query_feature_path, base_dir=Path(feature_root)), key=str(feature_key))
        reference_feature = _load_feature_map(
            _resolve_path(pair.reference_feature_path, base_dir=Path(feature_root)),
            key=str(feature_key),
        )
        result = model.match_pair(
            query_feature,
            reference_feature,
            query_image_size=(int(query_rgb.shape[2]), int(query_rgb.shape[1])),
            reference_image_size=(int(reference_rgb.shape[2]), int(reference_rgb.shape[1])),
            query_rgb=query_rgb,
            reference_rgb=reference_rgb,
        )
        measurements = {_proposal_key(item.proposal): item for item in result.measurements}
        measured_count += int(len(result.measurements))
        for proposal_index, proposal in enumerate(result.coarse_proposals):
            measurement = measurements.get(_proposal_key(proposal))
            row: dict[str, Any] = {
                "query_id": str(pair.query_id),
                "reference_image_id": str(pair.reference_image_id),
                "proposal_index": int(proposal_index),
                "query_index": int(proposal.query_index),
                "reference_index": int(proposal.reference_index),
                "rank": "" if proposal.rank is None else int(proposal.rank),
                "coarse_score": float(proposal.score),
                "coarse_confidence": "" if proposal.confidence is None else float(proposal.confidence),
                "query_x": float(proposal.query_xy[0]),
                "query_y": float(proposal.query_xy[1]),
                "reference_x": float(proposal.reference_xy[0]),
                "reference_y": float(proposal.reference_xy[1]),
                "measured_query_x": "",
                "measured_query_y": "",
                "measured_reference_x": "",
                "measured_reference_y": "",
                "measurement_confidence": "",
                "measurement_uncertainty_px": "",
            }
            if measurement is not None:
                row.update(
                    {
                        "measured_query_x": float(measurement.measured_query_xy[0]),
                        "measured_query_y": float(measurement.measured_query_xy[1]),
                        "measured_reference_x": float(measurement.measured_reference_xy[0]),
                        "measured_reference_y": float(measurement.measured_reference_xy[1]),
                        "measurement_confidence": "" if measurement.confidence is None else float(measurement.confidence),
                        "measurement_uncertainty_px": (
                            "" if measurement.uncertainty_px is None else float(measurement.uncertainty_px)
                        ),
                    }
                )
            gt, gt_distance = _nearest_ground_truth(
                pair,
                proposal,
                max_reference_distance_px=float(gt_reference_radius_px),
            )
            if gt is not None:
                gt_count += 1
                coarse_value = float(np.linalg.norm(np.asarray(proposal.query_xy, dtype=np.float32) - gt.query_xy))
                row.update(
                    {
                        "gt_query_x": float(gt.query_xy[0]),
                        "gt_query_y": float(gt.query_xy[1]),
                        "gt_reference_x": float(gt.reference_xy[0]),
                        "gt_reference_y": float(gt.reference_xy[1]),
                        "gt_reference_distance_px": "" if gt_distance is None else float(gt_distance),
                        "coarse_query_epe_px": coarse_value,
                    }
                )
                coarse_epe.append(coarse_value)
                if measurement is not None:
                    measured_value = float(np.linalg.norm(np.asarray(measurement.measured_query_xy, dtype=np.float32) - gt.query_xy))
                    row["measurement_query_epe_px"] = measured_value
                    paired_coarse_epe.append(coarse_value)
                    measurement_epe.append(measured_value)
            elif gt_distance is not None:
                row["gt_reference_distance_px"] = float(gt_distance)
            rows.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "proposals.csv", rows, _fieldnames(rows))
    _write_jsonl(output / "proposals.jsonl", rows)
    improve_ratio = None
    if paired_coarse_epe and measurement_epe:
        pairs_for_ratio = min(len(paired_coarse_epe), len(measurement_epe))
        improve_ratio = float(
            np.mean(
                np.asarray(measurement_epe[:pairs_for_ratio], dtype=np.float32)
                < np.asarray(paired_coarse_epe[:pairs_for_ratio], dtype=np.float32)
            )
        )
    summary = {
        "stage": "real_radio_closed_loop_localization",
        "pair_count": int(len(selected_pairs)),
        "proposal_count": int(len(rows)),
        "measurement_count": int(measured_count),
        "gt_matched_count": int(gt_count),
        "gt_reference_radius_px": float(gt_reference_radius_px),
        "coarse_epe_median_px": _median(coarse_epe),
        "coarse_epe_mean_px": _mean(coarse_epe),
        "measurement_epe_median_px": _median(measurement_epe),
        "measurement_epe_mean_px": _mean(measurement_epe),
        "measurement_improve_ratio": improve_ratio,
        "outputs": {
            "proposals_csv": str(output / "proposals.csv"),
            "proposals_jsonl": str(output / "proposals.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
