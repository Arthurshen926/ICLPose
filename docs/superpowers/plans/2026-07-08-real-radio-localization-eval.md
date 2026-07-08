# Real RADIO Localization Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-image absolute-pose evaluation for `JointFeatureMapper + MatchaTopKCoarseMatcher + RGBPatchMeasurementAdapter`.

**Architecture:** Add a focused `feature_extract/vfm/localization/pose_eval.py` module that converts closed-loop real-image proposals into `QueryTo3DMatch` records through nearest COLMAP support observations, deduplicates and spatially filters them, solves PnP per query, and writes summary metrics. Add a CLI wrapper that reuses the existing real-radio model loading and closed-loop pair execution path. Keep render paths out of the implementation.

**Tech Stack:** Python, NumPy, PIL, PyTorch model loading, existing COLMAP binary readers, existing Cambridge pose parser, existing PnP utilities in `feature_extract.vfm.query_to_3d_matching`, pytest.

---

## File Structure

- Create `feature_extract/vfm/localization/pose_eval.py`
  - Owns COLMAP support-observation indexing, nearest support-track association, proposal-to-2D3D conversion, per-query PnP, summary metrics, and artifact writing.
- Modify `feature_extract/vfm/localization/__init__.py`
  - Exports the new pose-eval helpers used by tests and scripts.
- Create `feature_extract/tools/vfm/eval_real_radio_pose_localization.py`
  - CLI that loads checkpoints, runs real closed-loop matching, converts proposals to 2D-3D, solves PnP, and writes `proposals`, `matches_2d3d`, `pose_rows`, and `summary`.
- Create `tests/test_real_radio_pose_eval.py`
  - Unit tests for indexing, proposal conversion, deduplication, and pose summary.
- Create `tests/test_real_radio_pose_eval_cli.py`
  - CLI argument parsing and runtime-device behavior tests.

## Task 1: COLMAP Observation Bridge

**Files:**
- Create: `feature_extract/vfm/localization/pose_eval.py`
- Modify: `feature_extract/vfm/localization/__init__.py`
- Test: `tests/test_real_radio_pose_eval.py`

- [ ] **Step 1: Write failing tests for nearest support observation lookup**

Add this test file:

```python
from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.pose_eval import build_support_observation_index


def _obs(image_id: str, track_id: int, xy: tuple[float, float]) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=xy,
        xyz=np.asarray([float(track_id), 0.0, 4.0], dtype=np.float64),
        track_length=3,
        reprojection_error=0.25,
        camera_id=1,
        image_width=100,
        image_height=80,
    )


def test_support_observation_index_chooses_nearest_within_radius() -> None:
    index = build_support_observation_index(
        [
            _obs("seq/r.png", 11, (10.0, 10.0)),
            _obs("seq/r.png", 12, (15.0, 10.0)),
            _obs("seq/other.png", 99, (10.0, 10.0)),
        ]
    )

    match = index.nearest("seq/r.png", np.asarray([14.0, 10.0], dtype=np.float32), max_distance_px=2.0)

    assert match is not None
    assert match.observation.track_id == 12
    assert match.distance_px == 1.0


def test_support_observation_index_respects_radius_and_image_id() -> None:
    index = build_support_observation_index([_obs("seq/r.png", 11, (10.0, 10.0))])

    assert index.nearest("seq/r.png", np.asarray([30.0, 30.0], dtype=np.float32), max_distance_px=4.0) is None
    assert index.nearest("seq/missing.png", np.asarray([10.0, 10.0], dtype=np.float32), max_distance_px=4.0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py::test_support_observation_index_chooses_nearest_within_radius tests/test_real_radio_pose_eval.py::test_support_observation_index_respects_radius_and_image_id -q
```

Expected: FAIL because `feature_extract.vfm.localization.pose_eval` does not exist.

- [ ] **Step 3: Implement observation indexing**

Create `feature_extract/vfm/localization/pose_eval.py` with:

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation


@dataclass(frozen=True)
class NearestSupportObservation:
    observation: ColmapTrackObservation
    distance_px: float


class SupportObservationIndex:
    def __init__(self, observations: Sequence[ColmapTrackObservation]) -> None:
        by_image: dict[str, list[ColmapTrackObservation]] = {}
        for observation in observations:
            by_image.setdefault(str(observation.image_id), []).append(observation)
        self._by_image = {image_id: tuple(items) for image_id, items in by_image.items()}
        self._xy_by_image = {
            image_id: np.asarray([item.xy for item in items], dtype=np.float64)
            for image_id, items in self._by_image.items()
        }

    def nearest(
        self,
        image_id: str,
        xy: np.ndarray | Sequence[float],
        *,
        max_distance_px: float,
    ) -> NearestSupportObservation | None:
        items = self._by_image.get(str(image_id))
        if not items:
            return None
        query_xy = np.asarray(xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(query_xy)):
            return None
        distances = np.linalg.norm(self._xy_by_image[str(image_id)] - query_xy[None, :], axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance > float(max_distance_px):
            return None
        return NearestSupportObservation(observation=items[best], distance_px=distance)


def build_support_observation_index(observations: Sequence[ColmapTrackObservation]) -> SupportObservationIndex:
    return SupportObservationIndex(observations)
```

Export from `feature_extract/vfm/localization/__init__.py`:

```python
from feature_extract.vfm.localization.pose_eval import (
    NearestSupportObservation,
    SupportObservationIndex,
    build_support_observation_index,
)
```

Add names to `__all__` if this module uses explicit exports.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py::test_support_observation_index_chooses_nearest_within_radius tests/test_real_radio_pose_eval.py::test_support_observation_index_respects_radius_and_image_id -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add feature_extract/vfm/localization/pose_eval.py feature_extract/vfm/localization/__init__.py tests/test_real_radio_pose_eval.py
git commit -m "feat: add real radio support observation index"
```

## Task 2: Proposal To 2D-3D Conversion

**Files:**
- Modify: `feature_extract/vfm/localization/pose_eval.py`
- Test: `tests/test_real_radio_pose_eval.py`

- [ ] **Step 1: Write failing tests for proposal conversion and deduplication**

Append:

```python
from feature_extract.vfm.localization import CoarseProposal, MeasurementResult
from feature_extract.vfm.localization.pose_eval import (
    ClosedLoopProposalRecord,
    convert_proposals_to_query_3d_matches,
    deduplicate_query_3d_matches,
)


def _proposal(query_xy=(3.0, 4.0), reference_xy=(10.0, 10.0), score=0.7) -> CoarseProposal:
    return CoarseProposal(
        query_index=0,
        reference_index=0,
        query_xy=np.asarray(query_xy, dtype=np.float32),
        reference_xy=np.asarray(reference_xy, dtype=np.float32),
        score=score,
        confidence=score,
        rank=0,
    )


def test_convert_proposals_uses_measurement_coordinates_and_support_xyz() -> None:
    index = build_support_observation_index([_obs("seq/r.png", 42, (20.0, 21.0))])
    proposal = _proposal(reference_xy=(10.0, 10.0))
    measurement = MeasurementResult(
        proposal=proposal,
        measured_query_xy=np.asarray([7.0, 8.0], dtype=np.float32),
        measured_reference_xy=np.asarray([20.0, 21.0], dtype=np.float32),
        confidence=0.9,
        uncertainty_px=0.5,
    )

    rows, matches = convert_proposals_to_query_3d_matches(
        [
            ClosedLoopProposalRecord(
                query_id="seq/q.png",
                reference_image_id="seq/r.png",
                proposal=proposal,
                measurement=measurement,
                proposal_index=0,
            )
        ],
        observation_index=index,
        max_support_distance_px=2.0,
    )

    assert len(rows) == 1
    assert len(matches["seq/q.png"]) == 1
    match = matches["seq/q.png"][0]
    assert match.track_id == 42
    np.testing.assert_allclose(match.xy, [7.0, 8.0])
    np.testing.assert_allclose(match.xyz, [42.0, 0.0, 4.0])
    assert match.pnp_soft_score == 0.9
    assert rows[0]["association_status"] == "matched"


def test_convert_proposals_falls_back_to_coarse_coordinates() -> None:
    index = build_support_observation_index([_obs("seq/r.png", 43, (10.0, 10.0))])
    proposal = _proposal(query_xy=(3.0, 4.0), reference_xy=(10.0, 10.0), score=0.6)

    rows, matches = convert_proposals_to_query_3d_matches(
        [
            ClosedLoopProposalRecord(
                query_id="seq/q.png",
                reference_image_id="seq/r.png",
                proposal=proposal,
                measurement=None,
                proposal_index=0,
            )
        ],
        observation_index=index,
        max_support_distance_px=1.0,
    )

    assert rows[0]["association_status"] == "matched"
    np.testing.assert_allclose(matches["seq/q.png"][0].xy, [3.0, 4.0])
    assert matches["seq/q.png"][0].pnp_soft_score == 0.6


def test_deduplicate_query_3d_matches_keeps_best_confidence() -> None:
    index = build_support_observation_index([_obs("seq/r.png", 42, (10.0, 10.0))])
    low = _proposal(query_xy=(1.0, 1.0), reference_xy=(10.0, 10.0), score=0.1)
    high = _proposal(query_xy=(2.0, 2.0), reference_xy=(10.0, 10.0), score=0.9)
    _, matches = convert_proposals_to_query_3d_matches(
        [
            ClosedLoopProposalRecord("seq/q.png", "seq/r.png", low, None, 0),
            ClosedLoopProposalRecord("seq/q.png", "seq/r.png", high, None, 1),
        ],
        observation_index=index,
        max_support_distance_px=1.0,
    )

    deduped = deduplicate_query_3d_matches(matches["seq/q.png"])

    assert len(deduped) == 1
    np.testing.assert_allclose(deduped[0].xy, [2.0, 2.0])
    assert deduped[0].pnp_soft_score == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py -q
```

Expected: FAIL because conversion helpers are not implemented.

- [ ] **Step 3: Implement conversion helpers**

Add to `pose_eval.py`:

```python
from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


@dataclass(frozen=True)
class ClosedLoopProposalRecord:
    query_id: str
    reference_image_id: str
    proposal: CoarseProposal
    measurement: MeasurementResult | None
    proposal_index: int


def _score_for_pnp(record: ClosedLoopProposalRecord) -> float:
    if record.measurement is not None and record.measurement.confidence is not None:
        return float(record.measurement.confidence)
    if record.proposal.confidence is not None:
        return float(record.proposal.confidence)
    return float(record.proposal.score)


def _measured_query_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_query_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.query_xy, dtype=np.float64).reshape(2)


def _measured_reference_xy(record: ClosedLoopProposalRecord) -> np.ndarray:
    if record.measurement is not None:
        return np.asarray(record.measurement.measured_reference_xy, dtype=np.float64).reshape(2)
    return np.asarray(record.proposal.reference_xy, dtype=np.float64).reshape(2)


def convert_proposals_to_query_3d_matches(
    records: Sequence[ClosedLoopProposalRecord],
    *,
    observation_index: SupportObservationIndex,
    max_support_distance_px: float,
) -> tuple[list[dict[str, Any]], dict[str, list[QueryTo3DMatch]]]:
    rows: list[dict[str, Any]] = []
    matches_by_query: dict[str, list[QueryTo3DMatch]] = {}
    for record in records:
        query_xy = _measured_query_xy(record)
        reference_xy = _measured_reference_xy(record)
        nearest = observation_index.nearest(
            record.reference_image_id,
            reference_xy,
            max_distance_px=float(max_support_distance_px),
        )
        base_row: dict[str, Any] = {
            "query_id": str(record.query_id),
            "reference_image_id": str(record.reference_image_id),
            "proposal_index": int(record.proposal_index),
            "query_x": float(query_xy[0]),
            "query_y": float(query_xy[1]),
            "reference_x": float(reference_xy[0]),
            "reference_y": float(reference_xy[1]),
            "score": float(_score_for_pnp(record)),
        }
        if nearest is None:
            rows.append({**base_row, "association_status": "missing_support_observation"})
            continue
        observation = nearest.observation
        match = QueryTo3DMatch(
            token_index=int(record.proposal_index),
            xy=query_xy.astype(np.float64, copy=False),
            track_id=int(observation.track_id),
            xyz=np.asarray(observation.xyz, dtype=np.float64).reshape(3),
            similarity=float(_score_for_pnp(record)),
            ratio=1.0,
            landmark_variance=0.0,
            source="real_radio_closed_loop",
            observation_count=int(observation.track_length),
            landmark_reprojection_error=float(observation.reprojection_error),
            pnp_soft_score=float(_score_for_pnp(record)),
            patch_offset_confidence=None if record.measurement is None else record.measurement.confidence,
            measurement_sigma_px=None if record.measurement is None else record.measurement.uncertainty_px,
            coarse_rank=record.proposal.rank,
            coarse_score=float(record.proposal.score),
        )
        matches_by_query.setdefault(str(record.query_id), []).append(match)
        rows.append(
            {
                **base_row,
                "association_status": "matched",
                "track_id": int(observation.track_id),
                "support_observation_distance_px": float(nearest.distance_px),
                "support_track_length": int(observation.track_length),
                "support_reprojection_error": float(observation.reprojection_error),
            }
        )
    return rows, matches_by_query


def deduplicate_query_3d_matches(matches: Sequence[QueryTo3DMatch]) -> list[QueryTo3DMatch]:
    best: dict[int, QueryTo3DMatch] = {}
    for match in matches:
        key = int(match.track_id)
        existing = best.get(key)
        current_score = 0.0 if match.pnp_soft_score is None else float(match.pnp_soft_score)
        existing_score = -float("inf") if existing is None or existing.pnp_soft_score is None else float(existing.pnp_soft_score)
        if existing is None or current_score > existing_score:
            best[key] = match
    return [best[key] for key in sorted(best)]
```

Export `ClosedLoopProposalRecord`, `convert_proposals_to_query_3d_matches`, and `deduplicate_query_3d_matches` from `localization/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add feature_extract/vfm/localization/pose_eval.py feature_extract/vfm/localization/__init__.py tests/test_real_radio_pose_eval.py
git commit -m "feat: convert real radio proposals to 2d3d matches"
```

## Task 3: Query-Level PnP And Summary Metrics

**Files:**
- Modify: `feature_extract/vfm/localization/pose_eval.py`
- Test: `tests/test_real_radio_pose_eval.py`

- [ ] **Step 1: Write failing tests for summary denominator and failure accounting**

Append:

```python
from feature_extract.vfm.localization.pose_eval import summarize_pose_rows


def test_summarize_pose_rows_counts_failures_in_recall_denominator() -> None:
    summary = summarize_pose_rows(
        [
            {
                "query_id": "ok.png",
                "success": True,
                "translation_error_m": 0.2,
                "rotation_error_deg": 1.0,
                "match_count": 10,
                "inlier_count": 7,
                "failure_reason": "",
            },
            {
                "query_id": "fail.png",
                "success": False,
                "translation_error_m": float("inf"),
                "rotation_error_deg": float("inf"),
                "match_count": 3,
                "inlier_count": 0,
                "failure_reason": "insufficient_matches",
            },
        ]
    )

    assert summary["query_count"] == 2
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["recall_0_25m_2deg"] == 0.5
    assert summary["recall_0_5m_5deg"] == 0.5
    assert summary["failure_counts"]["insufficient_matches"] == 1
    assert summary["median_translation_error_m"] == 0.2
    assert summary["median_rotation_error_deg"] == 1.0
```

- [ ] **Step 2: Write failing test for PnP evaluation using a tiny synthetic camera**

Append:

```python
from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch
from feature_extract.vfm.localization.pose_eval import evaluate_query_poses


def test_evaluate_query_poses_solves_simple_pnp() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    pose = np.eye(4, dtype=np.float64)
    gt = CambridgePoseRecord(
        image_id="q.png",
        camera_center=np.zeros(3, dtype=np.float64),
        rotation_w2c=np.eye(3, dtype=np.float64),
        pose_w2c=pose,
    )
    xyz_values = [
        np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        np.asarray([1.0, -1.0, 4.0], dtype=np.float64),
        np.asarray([1.0, 1.0, 4.0], dtype=np.float64),
        np.asarray([-1.0, 1.0, 4.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 6.0], dtype=np.float64),
        np.asarray([0.5, -0.25, 5.0], dtype=np.float64),
    ]
    matches = []
    for idx, xyz in enumerate(xyz_values):
        xy = np.asarray([80.0 * xyz[0] / xyz[2] + 50.0, 80.0 * xyz[1] / xyz[2] + 50.0], dtype=np.float64)
        matches.append(
            QueryTo3DMatch(
                token_index=idx,
                xy=xy,
                track_id=idx,
                xyz=xyz,
                similarity=1.0,
                ratio=1.0,
                landmark_variance=0.0,
                pnp_soft_score=1.0,
            )
        )

    rows = evaluate_query_poses(
        {"q.png": matches},
        cameras_by_query={"q.png": camera},
        gt_poses_by_query={"q.png": gt},
        pnp_reprojection_error_px=2.0,
        pnp_iterations=200,
        pnp_min_inliers=4,
    )

    assert len(rows) == 1
    assert rows[0]["success"] is True
    assert rows[0]["translation_error_m"] < 1e-4
    assert rows[0]["rotation_error_deg"] < 1e-4
    assert rows[0]["inlier_count"] >= 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py -q
```

Expected: FAIL because `summarize_pose_rows` and `evaluate_query_poses` are missing.

- [ ] **Step 4: Implement PnP and summary helpers**

Add to `pose_eval.py`:

```python
from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.rgb_patch_pose_proxy import scaled_colmap_camera
from feature_extract.vfm.query_to_3d_matching import (
    SpatialDiversityPnPConfig,
    estimate_pose_pnp_ransac,
    match_spatial_distribution_stats,
    pnp_pose_error,
    pnp_reprojection_residual_stats,
    select_pnp_matches_by_spatial_diversity,
)


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [float(row[key]) for row in rows if np.isfinite(float(row.get(key, float("inf"))))]
    return np.asarray(values, dtype=np.float64)


def summarize_pose_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    success_rows = [row for row in values if bool(row.get("success", False))]
    translations = _finite_values(success_rows, "translation_error_m")
    rotations = _finite_values(success_rows, "rotation_error_deg")
    failure_counts: dict[str, int] = {}
    for row in values:
        if bool(row.get("success", False)):
            continue
        reason = str(row.get("failure_reason", "") or "unknown")
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    def percentile(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("inf")

    def recall(max_t: float, max_r: float) -> float:
        if not values:
            return 0.0
        passed = sum(
            1
            for row in success_rows
            if float(row.get("translation_error_m", float("inf"))) <= max_t
            and float(row.get("rotation_error_deg", float("inf"))) <= max_r
        )
        return float(passed / len(values))

    match_counts = np.asarray([float(row.get("match_count", 0)) for row in values], dtype=np.float64)
    inlier_counts = np.asarray([float(row.get("inlier_count", 0)) for row in values], dtype=np.float64)
    return {
        "query_count": int(len(values)),
        "success_count": int(len(success_rows)),
        "success_rate": float(len(success_rows) / len(values)) if values else 0.0,
        "median_translation_error_m": percentile(translations, 50.0),
        "translation_error_p90_m": percentile(translations, 90.0),
        "median_rotation_error_deg": percentile(rotations, 50.0),
        "rotation_error_p90_deg": percentile(rotations, 90.0),
        "recall_0_25m_2deg": recall(0.25, 2.0),
        "recall_0_5m_5deg": recall(0.5, 5.0),
        "recall_5m_10deg": recall(5.0, 10.0),
        "median_match_count": float(np.median(match_counts)) if match_counts.size else 0.0,
        "median_inlier_count": float(np.median(inlier_counts)) if inlier_counts.size else 0.0,
        "failure_counts": failure_counts,
    }


def evaluate_query_poses(
    matches_by_query: Mapping[str, Sequence[QueryTo3DMatch]],
    *,
    cameras_by_query: Mapping[str, ColmapCamera],
    gt_poses_by_query: Mapping[str, CambridgePoseRecord],
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
    spatial_diversity: SpatialDiversityPnPConfig | None = None,
) -> list[dict[str, Any]]:
    pose_rows: list[dict[str, Any]] = []
    for query_id in sorted(matches_by_query):
        raw_matches = deduplicate_query_3d_matches(matches_by_query[query_id])
        camera = cameras_by_query.get(query_id)
        gt = gt_poses_by_query.get(query_id)
        if camera is None:
            pose_rows.append({"query_id": query_id, "success": False, "match_count": 0, "inlier_count": 0, "translation_error_m": float("inf"), "rotation_error_deg": float("inf"), "failure_reason": "missing_camera"})
            continue
        if gt is None:
            pose_rows.append({"query_id": query_id, "success": False, "match_count": 0, "inlier_count": 0, "translation_error_m": float("inf"), "rotation_error_deg": float("inf"), "failure_reason": "missing_gt_pose"})
            continue
        matches = raw_matches
        if spatial_diversity is not None:
            matches = select_pnp_matches_by_spatial_diversity(matches, int(camera.width), int(camera.height), spatial_diversity)
        if len(matches) < int(pnp_min_inliers):
            pose_rows.append({"query_id": query_id, "success": False, "match_count": int(len(matches)), "inlier_count": 0, "translation_error_m": float("inf"), "rotation_error_deg": float("inf"), "failure_reason": "insufficient_matches"})
            continue
        pnp = estimate_pose_pnp_ransac(
            matches,
            camera,
            reprojection_error_px=float(pnp_reprojection_error_px),
            confidence=float(pnp_confidence),
            iterations=int(pnp_iterations),
            min_inliers=int(pnp_min_inliers),
            refine_method="LM",
        )
        error = pnp_pose_error(pnp.pose_w2c if pnp.success else None, gt.pose_w2c)
        spatial_all = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height))
        spatial_inliers = match_spatial_distribution_stats(matches, int(camera.width), int(camera.height), pnp.inlier_mask)
        residuals = pnp_reprojection_residual_stats(matches, pnp.pose_w2c, camera, inlier_mask=pnp.inlier_mask)
        pose_rows.append(
            {
                "query_id": query_id,
                "success": bool(pnp.success),
                "match_count": int(len(matches)),
                "inlier_count": int(pnp.inlier_count),
                "inlier_ratio": float(pnp.inlier_ratio),
                "translation_error_m": float(error.translation_m),
                "rotation_error_deg": float(error.rotation_deg),
                "failure_reason": "" if pnp.success else "pnp_failed",
                "all_grid_4x4_occupancy_frac": spatial_all.get("grid_4x4_occupancy_frac"),
                "inlier_grid_4x4_occupancy_frac": spatial_inliers.get("grid_4x4_occupancy_frac"),
                **residuals,
            }
        )
    return pose_rows
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add feature_extract/vfm/localization/pose_eval.py tests/test_real_radio_pose_eval.py
git commit -m "feat: add real radio query pose metrics"
```

## Task 4: Real Closed-Loop Pose Eval CLI

**Files:**
- Create: `feature_extract/tools/vfm/eval_real_radio_pose_localization.py`
- Modify: `feature_extract/vfm/localization/pose_eval.py`
- Test: `tests/test_real_radio_pose_eval_cli.py`

- [ ] **Step 1: Write failing CLI parse tests**

Create `tests/test_real_radio_pose_eval_cli.py`:

```python
from __future__ import annotations

from feature_extract.tools.vfm.eval_real_radio_pose_localization import parse_args, resolve_runtime_device


def test_eval_real_radio_pose_localization_cli_args() -> None:
    args = parse_args(
        [
            "--pairs_csv",
            "pairs.csv",
            "--image_root",
            "images",
            "--feature_root",
            "features",
            "--colmap_model_dir",
            "sparse/0",
            "--query_pose_file",
            "dataset_test.txt",
            "--matcha_joint_checkpoint",
            "joint.pt",
            "--output_dir",
            "out",
            "--feature_key",
            "radio_final",
            "--feature_path_template",
            "{image_token}.npz",
            "--k_per_query",
            "2",
            "--max_support_distance_px",
            "6",
            "--pnp_reprojection_error_px",
            "8",
            "--pnp_min_inliers",
            "6",
        ]
    )

    assert args.colmap_model_dir == "sparse/0"
    assert args.query_pose_file == "dataset_test.txt"
    assert args.feature_key == "radio_final"
    assert args.k_per_query == 2
    assert args.max_support_distance_px == 6.0
    assert args.pnp_min_inliers == 6


def test_eval_real_radio_pose_runtime_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert str(resolve_runtime_device("cuda")) == "cpu"
    assert str(resolve_runtime_device("cpu")) == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_real_radio_pose_eval_cli.py -q
```

Expected: FAIL because the CLI file does not exist.

- [ ] **Step 3: Add artifact helpers**

Add to `pose_eval.py`:

```python
def write_mapping_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_mapping_rows_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
```

- [ ] **Step 4: Implement CLI skeleton and args**

Create `feature_extract/tools/vfm/eval_real_radio_pose_localization.py`:

```python
"""Run real-image RADIO selector + coarse + measurement absolute-pose evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import (
    load_colmap_track_observations,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.coarse_matcher import MatchaTopKCoarseMatcher
from feature_extract.vfm.localization.feature_mapper import JointFeatureMapper
from feature_extract.vfm.localization.measurement import RGBPatchMeasurementAdapter
from feature_extract.vfm.localization.pipeline import load_real_radio_localization_pairs_csv
from feature_extract.vfm.matcha_joint_training import load_matcha_joint_model
from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import load_rgb_patch_measurement_branch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--measurement_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="")
    parser.add_argument("--feature_path_template", default="{image_id}.npz")
    parser.add_argument("--k_per_query", type=int, default=1)
    parser.add_argument("--mutual_mode", default="annotate", choices=("none", "filter", "annotate"))
    parser.add_argument("--logit_scale", type=float, default=10.0)
    parser.add_argument("--min_similarity", type=float, default=-1.0)
    parser.add_argument("--max_matches", type=int, default=0)
    parser.add_argument("--prediction_head", default="gated", choices=("likelihood_mean", "mean", "direct", "gated"))
    parser.add_argument("--measurement_batch_size", type=int, default=128)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_support_distance_px", type=float, default=6.0)
    parser.add_argument("--pnp_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=1000)
    parser.add_argument("--pnp_confidence", type=float, default=0.999)
    parser.add_argument("--pnp_min_inliers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def resolve_runtime_device(device: str) -> torch.device:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested
```

- [ ] **Step 5: Implement CLI orchestration function**

Append:

```python
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from feature_extract.vfm.localization.pose_eval import run_real_radio_pose_localization_eval

    runtime_device = resolve_runtime_device(str(args.device))
    device_text = str(runtime_device)
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    observations = load_colmap_track_observations(model_dir)
    gt_poses = {record.image_id: record for record in parse_cambridge_pose_file(Path(args.query_pose_file))}
    pairs = load_real_radio_localization_pairs_csv(
        Path(args.pairs_csv),
        feature_path_template=str(args.feature_path_template),
    )
    joint_run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=device_text)
    measurement_checkpoint = Path(args.measurement_checkpoint) if str(args.measurement_checkpoint) else Path(args.matcha_joint_checkpoint)
    measurement_branch = load_rgb_patch_measurement_branch(measurement_checkpoint, device=runtime_device)
    summary = run_real_radio_pose_localization_eval(
        pairs,
        image_root=Path(args.image_root),
        feature_root=Path(args.feature_root),
        output_dir=Path(args.output_dir),
        feature_mapper=JointFeatureMapper(joint_run.model, device=device_text),
        coarse_matcher=MatchaTopKCoarseMatcher(
            k_per_query=int(args.k_per_query),
            mutual_mode=str(args.mutual_mode),
            logit_scale=float(args.logit_scale),
            min_similarity=float(args.min_similarity),
            max_matches=int(args.max_matches) if int(args.max_matches) > 0 else None,
            anchor_side="query",
        ),
        measurement_branch=RGBPatchMeasurementAdapter(
            branch=measurement_branch,
            device=device_text,
            prediction_head=str(args.prediction_head),
            batch_size=int(args.measurement_batch_size),
        ),
        cameras=cameras,
        colmap_images=images,
        colmap_observations=observations,
        gt_poses_by_query=gt_poses,
        feature_key=str(args.feature_key),
        max_pairs=int(args.max_pairs) if int(args.max_pairs) > 0 else None,
        max_support_distance_px=float(args.max_support_distance_px),
        pnp_reprojection_error_px=float(args.pnp_reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
        pnp_confidence=float(args.pnp_confidence),
        pnp_min_inliers=int(args.pnp_min_inliers),
    )
    summary.update(
        {
            "pairs_csv": str(args.pairs_csv),
            "image_root": str(args.image_root),
            "feature_root": str(args.feature_root),
            "colmap_model_dir": str(args.colmap_model_dir),
            "query_pose_file": str(args.query_pose_file),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "measurement_checkpoint": str(measurement_checkpoint),
            "feature_key": str(args.feature_key),
            "feature_path_template": str(args.feature_path_template),
        }
    )
    Path(summary["outputs"]["summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 6: Run CLI parse tests**

Run:

```bash
pytest tests/test_real_radio_pose_eval_cli.py -q
```

Expected: PASS because parsing imports only the CLI module; the runner is imported inside `main()`.

- [ ] **Step 7: Commit Task 4**

```bash
git add feature_extract/tools/vfm/eval_real_radio_pose_localization.py feature_extract/vfm/localization/pose_eval.py tests/test_real_radio_pose_eval_cli.py
git commit -m "feat: add real radio pose eval cli"
```

## Task 5: End-To-End Eval Runner

**Files:**
- Modify: `feature_extract/vfm/localization/pose_eval.py`
- Test: `tests/test_real_radio_pose_eval.py`

- [ ] **Step 1: Write failing unit test using fake mapper/matcher/measurement**

Append a compact runner test:

```python
from pathlib import Path
from PIL import Image

from feature_extract.vfm.localization import MappedFeatureMap
from feature_extract.vfm.localization.pose_eval import run_real_radio_pose_localization_eval
from feature_extract.vfm.localization.pipeline import RealRadioLocalizationPair
from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation


def _write_rgb(path: Path, size=(100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8), mode="RGB").save(path)


def test_run_real_radio_pose_localization_eval_writes_artifacts(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    feature_root = tmp_path / "features"
    _write_rgb(image_root / "q.png")
    _write_rgb(image_root / "r.png")
    feature_root.mkdir()
    np.save(feature_root / "q.npy", np.zeros((2, 1, 1), dtype=np.float32))
    np.save(feature_root / "r.npy", np.zeros((2, 1, 1), dtype=np.float32))

    class FakeFeatureMapper:
        def project(self, feature_map):
            return MappedFeatureMap(np.asarray(feature_map, dtype=np.float32), np.asarray(feature_map, dtype=np.float32))

    class FakeCoarseMatcher:
        def match(self, query_descriptors, reference_descriptors, *, query_image_size, reference_image_size):
            return [_proposal(query_xy=(50.0, 50.0), reference_xy=(50.0, 50.0), score=1.0)]

    class FakeMeasurement:
        def measure(self, query_rgb, reference_rgb, proposals, *, mapped_query=None, mapped_reference=None):
            return []

    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    colmap_image = ColmapImageObservation(
        image_id=1,
        image_name="q.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.zeros(3, dtype=np.float64),
        xys=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    gt = CambridgePoseRecord("q.png", np.zeros(3, dtype=np.float64), np.eye(3), np.eye(4))
    summary = run_real_radio_pose_localization_eval(
        [RealRadioLocalizationPair("q.png", "r.png", Path("q.npy"), Path("r.npy"))],
        image_root=image_root,
        feature_root=feature_root,
        output_dir=tmp_path / "out",
        feature_mapper=FakeFeatureMapper(),
        coarse_matcher=FakeCoarseMatcher(),
        measurement_branch=FakeMeasurement(),
        cameras={1: camera},
        colmap_images={1: colmap_image},
        colmap_observations=[_obs("r.png", 1, (50.0, 50.0))],
        gt_poses_by_query={"q.png": gt},
        max_support_distance_px=2.0,
    )

    assert summary["pose"]["query_count"] == 1
    assert summary["bridge"]["associated_match_count"] == 1
    assert Path(summary["outputs"]["matches_2d3d_csv"]).exists()
    assert Path(summary["outputs"]["pose_rows_csv"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py::test_run_real_radio_pose_localization_eval_writes_artifacts -q
```

Expected: FAIL because runner is missing.

- [ ] **Step 3: Implement runner**

Add to `pose_eval.py`:

```python
from PIL import Image
from feature_extract.vfm.colmap_tracks import ColmapImageObservation
from feature_extract.vfm.localization.model import SelectorCoarseMeasurementModel
from feature_extract.vfm.localization.pipeline import (
    RealRadioLocalizationPair,
    _load_feature_map,
    _load_rgb_chw,
    _resolve_path,
)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _cameras_by_query_from_colmap(
    *,
    cameras: Mapping[int, ColmapCamera],
    colmap_images: Mapping[int, ColmapImageObservation],
    image_root: Path,
) -> dict[str, ColmapCamera]:
    by_name = {image.image_name: image for image in colmap_images.values()}
    out: dict[str, ColmapCamera] = {}
    for image_name, image in by_name.items():
        camera = cameras.get(int(image.camera_id))
        if camera is None:
            continue
        image_path = Path(image_root) / image_name
        if image_path.exists():
            width, height = _image_size(image_path)
            out[image_name] = scaled_colmap_camera(camera, image_width=width, image_height=height)
        else:
            out[image_name] = camera
    return out


def _bridge_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    matched = [row for row in rows if row.get("association_status") == "matched"]
    distances = _finite_values(matched, "support_observation_distance_px")
    return {
        "proposal_count": int(total),
        "associated_match_count": int(len(matched)),
        "association_rate": float(len(matched) / total) if total else 0.0,
        "support_observation_distance_median_px": float(np.median(distances)) if distances.size else None,
        "support_observation_distance_p90_px": float(np.percentile(distances, 90.0)) if distances.size else None,
    }


def run_real_radio_pose_localization_eval(
    pairs: Sequence[RealRadioLocalizationPair],
    *,
    image_root: Path,
    feature_root: Path,
    output_dir: Path,
    feature_mapper,
    coarse_matcher,
    measurement_branch,
    cameras: Mapping[int, ColmapCamera],
    colmap_images: Mapping[int, ColmapImageObservation],
    colmap_observations: Sequence[ColmapTrackObservation],
    gt_poses_by_query: Mapping[str, CambridgePoseRecord],
    feature_key: str = "",
    max_pairs: int | None = None,
    max_support_distance_px: float = 6.0,
    pnp_reprojection_error_px: float = 8.0,
    pnp_iterations: int = 1000,
    pnp_confidence: float = 0.999,
    pnp_min_inliers: int = 4,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_pairs = list(pairs)
    if max_pairs is not None:
        selected_pairs = selected_pairs[: int(max_pairs)]
    model = SelectorCoarseMeasurementModel(feature_mapper, coarse_matcher, measurement_branch)
    records: list[ClosedLoopProposalRecord] = []
    proposal_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        query_rgb = _load_rgb_chw(Path(image_root) / pair.query_id)
        reference_rgb = _load_rgb_chw(Path(image_root) / pair.reference_image_id)
        query_feature = _load_feature_map(_resolve_path(pair.query_feature_path, base_dir=Path(feature_root)), key=str(feature_key))
        reference_feature = _load_feature_map(_resolve_path(pair.reference_feature_path, base_dir=Path(feature_root)), key=str(feature_key))
        result = model.match_pair(
            query_feature,
            reference_feature,
            query_image_size=(int(query_rgb.shape[2]), int(query_rgb.shape[1])),
            reference_image_size=(int(reference_rgb.shape[2]), int(reference_rgb.shape[1])),
            query_rgb=query_rgb,
            reference_rgb=reference_rgb,
        )
        measurements = {(id(item.proposal), int(item.proposal.query_index), int(item.proposal.reference_index)): item for item in result.measurements}
        for proposal_index, proposal in enumerate(result.coarse_proposals):
            measurement = measurements.get((id(proposal), int(proposal.query_index), int(proposal.reference_index)))
            records.append(ClosedLoopProposalRecord(pair.query_id, pair.reference_image_id, proposal, measurement, proposal_index))
            proposal_rows.append(
                {
                    "query_id": pair.query_id,
                    "reference_image_id": pair.reference_image_id,
                    "proposal_index": int(proposal_index),
                    "query_x": float(proposal.query_xy[0]),
                    "query_y": float(proposal.query_xy[1]),
                    "reference_x": float(proposal.reference_xy[0]),
                    "reference_y": float(proposal.reference_xy[1]),
                    "coarse_score": float(proposal.score),
                    "measurement_confidence": "" if measurement is None or measurement.confidence is None else float(measurement.confidence),
                }
            )
    observation_index = build_support_observation_index(colmap_observations)
    match_rows, matches_by_query = convert_proposals_to_query_3d_matches(
        records,
        observation_index=observation_index,
        max_support_distance_px=float(max_support_distance_px),
    )
    cameras_by_query = _cameras_by_query_from_colmap(cameras=cameras, colmap_images=colmap_images, image_root=Path(image_root))
    pose_rows = evaluate_query_poses(
        matches_by_query,
        cameras_by_query=cameras_by_query,
        gt_poses_by_query=gt_poses_by_query,
        pnp_reprojection_error_px=float(pnp_reprojection_error_px),
        pnp_iterations=int(pnp_iterations),
        pnp_confidence=float(pnp_confidence),
        pnp_min_inliers=int(pnp_min_inliers),
    )
    write_mapping_rows_csv(output / "proposals.csv", proposal_rows)
    write_mapping_rows_jsonl(output / "proposals.jsonl", proposal_rows)
    write_mapping_rows_csv(output / "matches_2d3d.csv", match_rows)
    write_mapping_rows_jsonl(output / "matches_2d3d.jsonl", match_rows)
    write_mapping_rows_csv(output / "pose_rows.csv", pose_rows)
    write_mapping_rows_jsonl(output / "pose_rows.jsonl", pose_rows)
    summary = {
        "stage": "real_radio_pose_localization",
        "pair_count": int(len(selected_pairs)),
        "bridge": _bridge_summary(match_rows),
        "pose": summarize_pose_rows(pose_rows),
        "max_support_distance_px": float(max_support_distance_px),
        "pnp_reprojection_error_px": float(pnp_reprojection_error_px),
        "pnp_iterations": int(pnp_iterations),
        "pnp_confidence": float(pnp_confidence),
        "pnp_min_inliers": int(pnp_min_inliers),
        "outputs": {
            "proposals_csv": str(output / "proposals.csv"),
            "proposals_jsonl": str(output / "proposals.jsonl"),
            "matches_2d3d_csv": str(output / "matches_2d3d.csv"),
            "matches_2d3d_jsonl": str(output / "matches_2d3d.jsonl"),
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "pose_rows_jsonl": str(output / "pose_rows.jsonl"),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
```

- [ ] **Step 4: Run focused runner tests**

Run:

```bash
pytest tests/test_real_radio_pose_eval.py tests/test_real_radio_pose_eval_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Run existing localization regressions**

Run:

```bash
pytest tests/test_real_radio_closed_loop_pipeline.py tests/test_real_radio_localization_scripts.py tests/test_localization_refactor.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add feature_extract/vfm/localization/pose_eval.py tests/test_real_radio_pose_eval.py
git commit -m "feat: run real radio pose localization eval"
```

## Task 6: Cambridge Smoke Run

**Files:**
- No source changes expected unless the smoke run exposes a bug.

- [ ] **Step 1: Run a tiny smoke evaluation**

Use the full-epoch checkpoint and cap pairs to keep runtime short:

```bash
python feature_extract/tools/vfm/eval_real_radio_pose_localization.py \
  --pairs_csv output/vfm/stage_r_matcha_joint/oldhospital/measurement_v1_real_real_sfm_tracks_v1/split_q90/val_rows.csv \
  --image_root /hy-tmp/Cambridge_stdloc/OldHospital/processed \
  --feature_root output/vfm_tokens_radio/OldHospital/train \
  --colmap_model_dir /hy-tmp/Cambridge_stdloc/OldHospital/sparse/0 \
  --track_observations_jsonl output/vfm/colmap_tracks/OldHospital/model_train_tracks_min2_balanced300k_v2.jsonl \
  --query_pose_file /hy-tmp/Cambridge_stdloc/OldHospital/dataset_train.txt \
  --matcha_joint_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/full_epoch_14144_prefetch_accum4_joint.pt \
  --measurement_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/full_epoch_14144_prefetch_accum4_adapter.pt \
  --output_dir output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/eval_pose_val_smoke \
  --feature_key radio_final \
  --feature_path_template '{image_token}.npz' \
  --max_pairs 5 \
  --k_per_query 2 \
  --max_matches 128 \
  --max_support_distance_px 6 \
  --pnp_reprojection_error_px 8 \
  --pnp_min_inliers 4 \
  --device cuda
```

Expected: command completes, writes `summary.json`, and reports nonzero proposal count. Pose success may be low on a 5-pair smoke run; failures must be explicit in `pose.failure_counts`.

- [ ] **Step 2: Run a larger validation sample if smoke succeeds**

```bash
python feature_extract/tools/vfm/eval_real_radio_pose_localization.py \
  --pairs_csv output/vfm/stage_r_matcha_joint/oldhospital/measurement_v1_real_real_sfm_tracks_v1/split_q90/val_rows.csv \
  --image_root /hy-tmp/Cambridge_stdloc/OldHospital/processed \
  --feature_root output/vfm_tokens_radio/OldHospital/train \
  --colmap_model_dir /hy-tmp/Cambridge_stdloc/OldHospital/sparse/0 \
  --track_observations_jsonl output/vfm/colmap_tracks/OldHospital/model_train_tracks_min2_balanced300k_v2.jsonl \
  --query_pose_file /hy-tmp/Cambridge_stdloc/OldHospital/dataset_train.txt \
  --matcha_joint_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/full_epoch_14144_prefetch_accum4_joint.pt \
  --measurement_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/full_epoch_14144_prefetch_accum4_adapter.pt \
  --output_dir output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/eval_pose_val_200pairs \
  --feature_key radio_final \
  --feature_path_template '{image_token}.npz' \
  --max_pairs 200 \
  --k_per_query 2 \
  --max_matches 128 \
  --max_support_distance_px 6 \
  --pnp_reprojection_error_px 8 \
  --pnp_min_inliers 4 \
  --device cuda
```

Expected: command completes and prints headline pose metrics from `summary["pose"]`.

- [ ] **Step 3: Report results**

Summarize:

- output directory,
- query count and pair count,
- association rate,
- PnP success rate,
- median/p90 translation and rotation,
- threshold recalls,
- dominant failure reasons.

Do not claim full benchmark quality if only a capped sample was run.

## Self-Review Checklist

- Spec coverage:
  - Coarse conditional rank remains a separate diagnostic and is not conflated with pose metrics.
  - Proposal-to-2D3D conversion uses nearest support COLMAP observation, not CSV `track_id` alone.
  - PnP metrics include failures in recall denominators.
  - No render input is introduced.
- Placeholder scan:
  - The plan contains exact files, tests, commands, and expected outcomes.
- Type consistency:
  - `ClosedLoopProposalRecord`, `SupportObservationIndex`, `convert_proposals_to_query_3d_matches`, `evaluate_query_poses`, `summarize_pose_rows`, and `run_real_radio_pose_localization_eval` are defined before use in tests or CLI.
