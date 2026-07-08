from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from feature_extract.vfm.cambridge_pose_lattice import CambridgePoseRecord
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.colmap_tracks import ColmapImageObservation
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization import CoarseProposal, MappedFeatureMap, MeasurementResult
from feature_extract.vfm.localization.pipeline import RealRadioLocalizationPair
from feature_extract.vfm.localization.pose_eval import build_support_observation_index
from feature_extract.vfm.localization.pose_eval import (
    ClosedLoopProposalRecord,
    convert_proposals_to_query_3d_matches,
    deduplicate_query_3d_matches,
    evaluate_query_poses,
    run_real_radio_pose_localization_eval,
    summarize_pose_rows,
)
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


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


def test_support_observation_index_scales_observations_to_target_image_size() -> None:
    index = build_support_observation_index(
        [_obs("seq/r.png", 11, (10.0, 5.0))],
        target_image_sizes={"seq/r.png": (200, 160)},
    )

    match = index.nearest("seq/r.png", np.asarray([20.0, 10.0], dtype=np.float32), max_distance_px=0.01)

    assert match is not None
    assert match.observation.track_id == 11


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
            del query_descriptors, reference_descriptors, query_image_size, reference_image_size
            return [_proposal(query_xy=(50.0, 50.0), reference_xy=(50.0, 50.0), score=1.0)]

    class FakeMeasurement:
        def measure(self, query_rgb, reference_rgb, proposals, *, mapped_query=None, mapped_reference=None):
            del query_rgb, reference_rgb, proposals, mapped_query, mapped_reference
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
        colmap_observations=[_obs("r.png", 1, (50.0, 40.0))],
        gt_poses_by_query={"q.png": gt},
        max_support_distance_px=2.0,
    )

    assert summary["pose"]["query_count"] == 1
    assert summary["bridge"]["associated_match_count"] == 1
    assert Path(summary["outputs"]["matches_2d3d_csv"]).exists()
    assert Path(summary["outputs"]["pose_rows_csv"]).exists()
