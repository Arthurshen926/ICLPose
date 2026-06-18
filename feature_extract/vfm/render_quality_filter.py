"""Filter VPR candidates using render visibility and quality sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


@dataclass(frozen=True)
class RenderQualityRecord:
    image_id: str
    alpha_coverage: float | None = None
    visible_ratio: float | None = None
    overlap_score: float | None = None
    view_angle_deg: float | None = None
    render_valid: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RenderQualityRecord":
        return cls(
            image_id=str(data["image_id"]),
            alpha_coverage=None if data.get("alpha_coverage") is None else float(data["alpha_coverage"]),
            visible_ratio=None if data.get("visible_ratio") is None else float(data["visible_ratio"]),
            overlap_score=None if data.get("overlap_score") is None else float(data["overlap_score"]),
            view_angle_deg=None if data.get("view_angle_deg") is None else float(data["view_angle_deg"]),
            render_valid=bool(data.get("render_valid", True)),
        )


def load_render_quality_sidecar(path: Path) -> dict[str, RenderQualityRecord]:
    records: dict[str, RenderQualityRecord] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = RenderQualityRecord.from_mapping(json.loads(line))
        records[record.image_id] = record
    if not records:
        raise ValueError("render quality sidecar is empty")
    return records


def _passes(
    record: RenderQualityRecord,
    *,
    min_alpha_coverage: float | None,
    min_visible_ratio: float | None,
    min_overlap_score: float | None,
    max_view_angle_deg: float | None,
    require_render_valid: bool,
) -> bool:
    if require_render_valid and not bool(record.render_valid):
        return False
    if min_alpha_coverage is not None and (
        record.alpha_coverage is None or float(record.alpha_coverage) < float(min_alpha_coverage)
    ):
        return False
    if min_visible_ratio is not None and (
        record.visible_ratio is None or float(record.visible_ratio) < float(min_visible_ratio)
    ):
        return False
    if min_overlap_score is not None and (
        record.overlap_score is None or float(record.overlap_score) < float(min_overlap_score)
    ):
        return False
    if max_view_angle_deg is not None and (
        record.view_angle_deg is None or float(record.view_angle_deg) > float(max_view_angle_deg)
    ):
        return False
    return True


def _quality_metadata(record: RenderQualityRecord) -> dict[str, object]:
    metadata: dict[str, object] = {
        "render_quality_filter_passed": True,
        "render_valid": bool(record.render_valid),
    }
    if record.alpha_coverage is not None:
        metadata["render_alpha_coverage"] = float(record.alpha_coverage)
    if record.visible_ratio is not None:
        metadata["render_visible_ratio"] = float(record.visible_ratio)
    if record.overlap_score is not None:
        metadata["render_overlap_score"] = float(record.overlap_score)
    if record.view_angle_deg is not None:
        metadata["render_view_angle_deg"] = float(record.view_angle_deg)
    return metadata


def filter_candidate_bank_by_render_quality(
    bank: CandidateHypothesisBank,
    quality_by_image_id: Mapping[str, RenderQualityRecord],
    *,
    min_alpha_coverage: float | None = None,
    min_visible_ratio: float | None = None,
    min_overlap_score: float | None = None,
    max_view_angle_deg: float | None = None,
    require_render_valid: bool = True,
    drop_missing: bool = True,
) -> CandidateHypothesisBank:
    kept = []
    for candidate in bank.candidates:
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        record = quality_by_image_id.get(candidate.reference_image)
        if record is None:
            if drop_missing:
                continue
            kept.append(candidate)
            continue
        if not _passes(
            record,
            min_alpha_coverage=min_alpha_coverage,
            min_visible_ratio=min_visible_ratio,
            min_overlap_score=min_overlap_score,
            max_view_angle_deg=max_view_angle_deg,
            require_render_valid=bool(require_render_valid),
        ):
            continue
        metadata = dict(candidate.metadata)
        metadata.update(_quality_metadata(record))
        kept.append(replace(candidate, metadata=metadata))
    return CandidateHypothesisBank.from_candidates(
        protocol_name=f"{bank.protocol_name}_render_quality",
        protocol_kind=bank.protocol_kind,
        candidates=kept,
        protocol_fingerprint=bank.protocol_fingerprint,
    )
