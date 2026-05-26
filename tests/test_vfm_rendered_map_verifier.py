import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.score_candidate_bank_rendered_map import main as rendered_map_cli_main
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import (
    SelectedTrackFeatureBank,
    TrackFeature,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.rendered_map_verifier import (
    build_track_visibility_index,
    build_track_xyz_index,
    project_xyz_to_image,
    projected_selected_map_token_grid,
    sparse_rendered_map_evidence,
    rendered_selected_map_token_grid,
    score_candidate_bank_by_rendered_selected_map,
    score_candidate_bank_by_projected_rendered_selected_map,
    score_candidate_bank_by_sparse_rendered_selected_map,
)
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _selector_checkpoint(path):
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.bias.zero_()
        selector.uncertainty_head.weight.zero_()
        selector.uncertainty_head.bias.zero_()
    torch.save(selector.state_dict(), path)


def _query_manifest(tmp_path):
    token_path = tmp_path / "q_tokens.npz"
    np.savez_compressed(
        token_path,
        radio_final=np.asarray([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    )
    return TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="query.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 14),),
                split="test",
                scene="synthetic",
            ),
        )
    )


def _two_query_manifest(tmp_path):
    first = _query_manifest(tmp_path).records[0]
    token_path = tmp_path / "q2_tokens.npz"
    np.savez_compressed(
        token_path,
        radio_final=np.asarray([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    )
    second = TokenBankRecord(
        image_id="query2.png",
        token_path=token_path,
        layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 14),),
        split="test",
        scene="synthetic",
    )
    return TokenBankManifest(records=(first, second))


def _candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="rendered_selected_map_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="wrong",
                candidate_type="reference_pose",
                reference_image="wrong_ref.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="correct",
                candidate_type="reference_pose",
                reference_image="correct_ref.png",
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    )


def _two_query_candidate_bank():
    candidates = []
    for query_id in ("query.png", "query2.png"):
        candidates.extend(
            [
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{query_id}:wrong",
                    candidate_type="reference_pose",
                    reference_image="wrong_ref.png",
                    pose_error=PoseCost(1.0, 20.0),
                ),
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{query_id}:correct",
                    candidate_type="reference_pose",
                    reference_image="correct_ref.png",
                    pose_error=PoseCost(0.05, 1.0),
                ),
            ]
        )
    return CandidateHypothesisBank.from_candidates(
        protocol_name="rendered_selected_map_two_query_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=candidates,
    )


def _track_bank():
    return SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("correct_ref.png", "support_ref.png"),
            ),
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([-1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("wrong_ref.png", "support_ref.png"),
            ),
        },
        feature_dim=2,
    )


def _write_track_observations(path, extra_reference=False):
    records = [
        {"track_id": 1, "image_id": "correct_ref.png"},
        {"track_id": 2, "image_id": "wrong_ref.png"},
    ]
    if extra_reference:
        records.append({"track_id": 99, "image_id": "missing_ref.png"})
    lines = []
    for idx, record in enumerate(records):
        item = {
            "track_id": record["track_id"],
            "image_id": record["image_id"],
            "point2d_idx": idx,
            "xy": [1.0, 1.0],
            "xyz": [0.0, 0.0, 1.0],
            "track_length": 2,
            "reprojection_error": 0.5,
            "camera_id": 1,
            "image_width": 10,
            "image_height": 10,
        }
        lines.append(json.dumps(item, sort_keys=True))
    path.write_text("\n".join(lines) + "\n")


def _spatial_query_manifest(tmp_path):
    token_path = tmp_path / "spatial_q_tokens.npz"
    tokens = np.zeros((2, 3, 3), dtype=np.float32)
    tokens[:, 1, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    tokens[:, 0, 0] = np.asarray([-1.0, 0.0], dtype=np.float32)
    np.savez_compressed(token_path, radio_final=tokens)
    return TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="query.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 1),),
                split="test",
                scene="synthetic",
            ),
        )
    )


def _spatial_candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="sparse_rendered_map_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="wrong_spatial",
                candidate_type="reference_pose",
                reference_image="wrong_spatial_ref.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="correct_spatial",
                candidate_type="reference_pose",
                reference_image="correct_spatial_ref.png",
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    )


def _spatial_track_bank():
    return SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("correct_spatial_ref.png", "wrong_spatial_ref.png"),
            ),
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([-1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("correct_spatial_ref.png", "wrong_spatial_ref.png"),
            ),
        },
        feature_dim=2,
    )


def _obs(track_id, image_id, xy):
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=xy,
        xyz=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        track_length=2,
        reprojection_error=0.5,
        camera_id=1,
        image_width=3,
        image_height=3,
    )


def _spatial_observation_index():
    return {
        "correct_spatial_ref.png": [
            _obs(1, "correct_spatial_ref.png", (1.0, 1.0)),
            _obs(2, "correct_spatial_ref.png", (0.0, 0.0)),
        ],
        "wrong_spatial_ref.png": [
            _obs(1, "wrong_spatial_ref.png", (0.0, 0.0)),
            _obs(2, "wrong_spatial_ref.png", (1.0, 1.0)),
        ],
    }


def test_rendered_selected_map_token_grid_uses_track_observation_xy():
    grid, mask, count, visibility_fraction = rendered_selected_map_token_grid(
        reference_image="correct_spatial_ref.png",
        track_bank=_spatial_track_bank(),
        observation_index=_spatial_observation_index(),
        token_height=3,
        token_width=3,
    )

    assert count == 2
    assert visibility_fraction == pytest.approx(1.0)
    assert mask[1, 1]
    assert mask[0, 0]
    assert grid[:, 1, 1].tolist() == pytest.approx([1.0, 0.0])
    assert grid[:, 0, 0].tolist() == pytest.approx([-1.0, 0.0])


def test_sparse_rendered_selected_map_scoring_uses_spatial_layout_not_global_mean(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)

    rows = score_candidate_bank_by_sparse_rendered_selected_map(
        bank=_spatial_candidate_bank(),
        query_manifest=_spatial_query_manifest(tmp_path),
        track_bank=_spatial_track_bank(),
        observation_index=_spatial_observation_index(),
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="sparse_rendered_selected_map",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
        local_radius=0,
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)


def test_sparse_rendered_map_evidence_counts_local_feature_inliers():
    query_map = np.zeros((2, 2, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 1, 1] = np.asarray([0.0, 1.0], dtype=np.float32)
    rendered_grid = np.zeros((2, 2, 2), dtype=np.float32)
    rendered_grid[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    rendered_grid[:, 1, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    rendered_mask = np.zeros((2, 2), dtype=bool)
    rendered_mask[0, 0] = True
    rendered_mask[1, 1] = True

    evidence = sparse_rendered_map_evidence(
        query_map,
        rendered_grid,
        rendered_mask,
        local_radius=0,
        inlier_threshold=0.5,
        inlier_weight=0.25,
    )

    assert evidence.match_count == 2
    assert evidence.mean_similarity == pytest.approx(0.5)
    assert evidence.inlier_fraction == pytest.approx(0.5)
    assert evidence.score == pytest.approx(0.625)


def test_sparse_rendered_map_evidence_uses_cosine_not_feature_magnitude():
    query_map = np.zeros((2, 1, 3), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 2] = np.asarray([20.0, 1.0], dtype=np.float32)
    rendered_grid = np.zeros((2, 1, 3), dtype=np.float32)
    rendered_grid[:, 0, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    rendered_mask = np.zeros((1, 3), dtype=bool)
    rendered_mask[0, 1] = True

    evidence = sparse_rendered_map_evidence(
        query_map,
        rendered_grid,
        rendered_mask,
        local_radius=1,
    )

    assert evidence.mean_similarity == pytest.approx(1.0)


def test_project_xyz_to_image_uses_world_to_camera_pose():
    camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    pose_w2c = np.eye(4, dtype=np.float64)

    projected = project_xyz_to_image(np.asarray([0.0, 0.0, 1.0]), pose_w2c, camera)

    assert projected == pytest.approx((1.0, 1.0))


def test_projected_selected_map_token_grid_uses_candidate_pose():
    camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    pose_w2c = np.eye(4, dtype=np.float64)
    track_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("ref.png",),
            )
        },
        feature_dim=2,
    )

    grid, mask, count, visibility_fraction = projected_selected_map_token_grid(
        track_bank=track_bank,
        track_xyz_index={1: np.asarray([0.0, 0.0, 1.0], dtype=np.float64)},
        pose_w2c=pose_w2c,
        camera=camera,
        token_height=3,
        token_width=3,
    )

    assert count == 1
    assert visibility_fraction == pytest.approx(1.0)
    assert mask[1, 1]
    assert grid[:, 1, 1].tolist() == pytest.approx([1.0, 0.0])


def test_projected_selected_map_token_grid_uses_track_utility_weights():
    camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    track_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("ref.png",),
            ),
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([-1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=0.0,
                observation_image_ids=("ref.png",),
            ),
        },
        feature_dim=2,
    )

    grid, mask, count, visibility_fraction = projected_selected_map_token_grid(
        track_bank=track_bank,
        track_xyz_index={
            1: np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            2: np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        },
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        token_height=3,
        token_width=3,
    )

    assert count == 2
    assert visibility_fraction == pytest.approx(1.0)
    assert mask[1, 1]
    assert grid[:, 1, 1].tolist() == pytest.approx([1.0, 0.0])


def test_projected_selected_map_token_grid_can_filter_visible_tracks():
    camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    track_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("ref.png",),
            ),
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([-1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("distractor.png",),
            ),
        },
        feature_dim=2,
    )

    grid, mask, count, visibility_fraction = projected_selected_map_token_grid(
        track_bank=track_bank,
        track_xyz_index={
            1: np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            2: np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        },
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=camera,
        token_height=3,
        token_width=3,
        visible_track_ids={1},
    )

    assert count == 1
    assert visibility_fraction == pytest.approx(1.0)
    assert mask[1, 1]
    assert grid[:, 1, 1].tolist() == pytest.approx([1.0, 0.0])


def test_projected_rendered_selected_map_scoring_uses_candidate_pose(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    good_pose = np.eye(4, dtype=np.float64)
    bad_pose = np.eye(4, dtype=np.float64)
    bad_pose[0, 3] = 1.0
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="projected_rendered_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="bad_pose",
                candidate_type="reference_pose",
                reference_image="ref.png",
                pose=bad_pose.tolist(),
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="good_pose",
                candidate_type="reference_pose",
                reference_image="ref.png",
                pose=good_pose.tolist(),
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    )
    track_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("ref.png",),
            )
        },
        feature_dim=2,
    )

    rows = score_candidate_bank_by_projected_rendered_selected_map(
        bank=bank,
        query_manifest=_spatial_query_manifest(tmp_path),
        track_bank=track_bank,
        track_xyz_index={1: np.asarray([0.0, 0.0, 1.0], dtype=np.float64)},
        camera_by_image={"query.png": camera},
        default_camera=camera,
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="projected_rendered_selected_map",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
        local_radius=0,
    )

    assert rows[1].score > rows[0].score
    assert rows[1].mean_similarity == pytest.approx(rows[1].score)
    assert rows[1].inlier_fraction == pytest.approx(1.0)
    assert rows[1].match_count == 1
    assert rows[1].visibility_fraction == pytest.approx(1.0)
    assert evaluate_score_table(rows).mean_top1_acc == pytest.approx(1.0)


def test_projected_rendered_selected_map_scoring_uses_query_camera_not_reference(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    query_camera = ColmapCamera(camera_id=1, model_id=2, width=3, height=3, params=(1.0, 1.0, 1.0, 0.0))
    reference_camera = ColmapCamera(camera_id=2, model_id=2, width=3, height=3, params=(1.0, 0.0, 0.0, 0.0))
    pose = np.eye(4, dtype=np.float64)
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="query_camera_projection_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="candidate",
                candidate_type="reference_pose",
                reference_image="ref.png",
                pose=pose.tolist(),
                pose_error=PoseCost(0.05, 1.0),
            )
        ],
    )
    track_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("ref.png",),
            )
        },
        feature_dim=2,
    )

    rows = score_candidate_bank_by_projected_rendered_selected_map(
        bank=bank,
        query_manifest=_spatial_query_manifest(tmp_path),
        track_bank=track_bank,
        track_xyz_index={1: np.asarray([0.0, 0.0, 1.0], dtype=np.float64)},
        camera_by_image={"query.png": query_camera, "ref.png": reference_camera},
        default_camera=reference_camera,
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="projected_rendered_selected_map",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
        local_radius=0,
    )

    assert rows[0].score > 0.9


def test_rendered_selected_map_scoring_prefers_correct_reference_and_reports_top1(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)

    rows = score_candidate_bank_by_rendered_selected_map(
        bank=_candidate_bank(),
        query_manifest=_query_manifest(tmp_path),
        track_bank=_track_bank(),
        visibility_index={
            "correct_ref.png": {1},
            "wrong_ref.png": {2},
        },
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="rendered_selected_map_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)


def test_rendered_selected_map_missing_reference_tracks_gets_low_score(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="missing_track_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="missing",
                candidate_type="reference_pose",
                reference_image="missing_ref.png",
                pose_error=PoseCost(1.0, 20.0),
            )
        ],
    )

    rows = score_candidate_bank_by_rendered_selected_map(
        bank=bank,
        query_manifest=_query_manifest(tmp_path),
        track_bank=_track_bank(),
        visibility_index={"missing_ref.png": {99}},
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="rendered_selected_map_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
    )

    assert rows[0].score == pytest.approx(-1.0)
    assert rows[0].risk == pytest.approx(1.0)


def test_rendered_selected_map_scoring_rejects_query_feature_provenance(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    contaminated_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("query.png", "correct_ref.png"),
            )
        },
        feature_dim=2,
    )

    with pytest.raises(ValueError, match="forbidden image"):
        score_candidate_bank_by_rendered_selected_map(
            bank=_candidate_bank(),
            query_manifest=_query_manifest(tmp_path),
            track_bank=contaminated_bank,
            visibility_index={"correct_ref.png": {1}},
            selector=load_selector_from_checkpoint(checkpoint),
            layer_name="radio_final",
            method="rendered_selected_map_cosine",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
            device="cpu",
        )


def test_rendered_selected_map_scoring_requires_track_provenance(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    no_provenance_bank = SelectedTrackFeatureBank(
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros(2, dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
            )
        },
        feature_dim=2,
    )

    with pytest.raises(ValueError, match="missing observation provenance"):
        score_candidate_bank_by_rendered_selected_map(
            bank=_candidate_bank(),
            query_manifest=_query_manifest(tmp_path),
            track_bank=no_provenance_bank,
            visibility_index={"correct_ref.png": {1}},
            selector=load_selector_from_checkpoint(checkpoint),
            layer_name="radio_final",
            method="rendered_selected_map_cosine",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
            device="cpu",
        )


def test_rendered_selected_map_scoring_validates_feature_dim_before_visibility(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    wrong_dim_bank = SelectedTrackFeatureBank(tracks={}, feature_dim=3)

    with pytest.raises(ValueError, match="track bank feature_dim"):
        score_candidate_bank_by_rendered_selected_map(
            bank=_candidate_bank(),
            query_manifest=_query_manifest(tmp_path),
            track_bank=wrong_dim_bank,
            visibility_index={},
            selector=load_selector_from_checkpoint(checkpoint),
            layer_name="radio_final",
            method="rendered_selected_map_cosine",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
            device="cpu",
        )


def test_rendered_selected_map_scoring_validates_query_manifest_once(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    validate_count = 0
    original_validate = TokenBankManifest.validate

    def counted_validate(self, verify_checksums=True):
        nonlocal validate_count
        validate_count += 1
        original_validate(self, verify_checksums=verify_checksums)

    monkeypatch.setattr(TokenBankManifest, "validate", counted_validate)

    rows = score_candidate_bank_by_rendered_selected_map(
        bank=_two_query_candidate_bank(),
        query_manifest=_two_query_manifest(tmp_path),
        track_bank=_track_bank(),
        visibility_index={
            "correct_ref.png": {1},
            "wrong_ref.png": {2},
        },
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="rendered_selected_map_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
    )

    assert len(rows) == 4
    assert validate_count == 1


def test_rendered_selected_map_cli_writes_rows_report_and_md(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    manifest_path = tmp_path / "query_manifest.json"
    track_bank_path = tmp_path / "tracks.npz"
    observations_path = tmp_path / "observations.jsonl"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    _selector_checkpoint(checkpoint)
    _candidate_bank().to_jsonl(bank_path)
    _query_manifest(tmp_path).to_json(manifest_path)
    save_selected_track_bank_npz(_track_bank(), track_bank_path)
    _write_track_observations(observations_path)

    rendered_map_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(manifest_path),
            "--track_bank",
            str(track_bank_path),
            "--track_observations",
            str(observations_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
            "--output_md",
            str(md_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[1]["score"] > rows[0]["score"]
    assert rows[1]["risk"] == pytest.approx(0.0)
    assert report["mean_top1_acc"] == pytest.approx(1.0)
    assert report["rendered_map_parameters"]["mode"] == "global_descriptor"
    assert report["rendered_map_parameters"]["local_radius"] == 0
    assert "Rendered Selected Map Cosine" in md_path.read_text()


def test_rendered_selected_map_cli_supports_sparse_grid_mode(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    manifest_path = tmp_path / "query_manifest.json"
    track_bank_path = tmp_path / "tracks.npz"
    observations_path = tmp_path / "observations.jsonl"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _selector_checkpoint(checkpoint)
    _spatial_candidate_bank().to_jsonl(bank_path)
    _spatial_query_manifest(tmp_path).to_json(manifest_path)
    save_selected_track_bank_npz(_spatial_track_bank(), track_bank_path)
    lines = []
    for observations in _spatial_observation_index().values():
        for obs in observations:
            lines.append(
                json.dumps(
                    {
                        "track_id": obs.track_id,
                        "image_id": obs.image_id,
                        "point2d_idx": obs.point2d_idx,
                        "xy": list(obs.xy),
                        "xyz": obs.xyz.tolist(),
                        "track_length": obs.track_length,
                        "reprojection_error": obs.reprojection_error,
                        "camera_id": obs.camera_id,
                        "image_width": obs.image_width,
                        "image_height": obs.image_height,
                    },
                    sort_keys=True,
                )
            )
    observations_path.write_text("\n".join(lines) + "\n")

    rendered_map_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(manifest_path),
            "--track_bank",
            str(track_bank_path),
            "--track_observations",
            str(observations_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--mode",
            "sparse_grid",
            "--method",
            "sparse_rendered_selected_map",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[1]["score"] > rows[0]["score"]
    assert report["mean_top1_acc"] == pytest.approx(1.0)
    assert report["rendered_map_parameters"]["mode"] == "sparse_grid"


def test_rendered_selected_map_cli_supports_projected_grid_mode(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    manifest_path = tmp_path / "query_manifest.json"
    track_bank_path = tmp_path / "tracks.npz"
    observations_path = tmp_path / "observations.jsonl"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _selector_checkpoint(checkpoint)
    _spatial_query_manifest(tmp_path).to_json(manifest_path)
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(
                    track_id=1,
                    mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                    variance=np.zeros(2, dtype=np.float32),
                    observation_count=2,
                    mean_utility=1.0,
                    observation_image_ids=("ref.png",),
                )
            },
            feature_dim=2,
        ),
        track_bank_path,
    )
    good_pose = np.eye(4, dtype=np.float64)
    bad_pose = np.eye(4, dtype=np.float64)
    bad_pose[0, 3] = 1.0
    CandidateHypothesisBank.from_candidates(
        protocol_name="projected_grid_cli_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="bad",
                candidate_type="reference_pose",
                reference_image="ref.png",
                pose=bad_pose.tolist(),
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="good",
                candidate_type="reference_pose",
                reference_image="ref.png",
                pose=good_pose.tolist(),
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    ).to_jsonl(bank_path)
    observations_path.write_text(
        json.dumps(
            {
                "track_id": 1,
                "image_id": "ref.png",
                "point2d_idx": 1,
                "xy": [1.0, 1.0],
                "xyz": [0.0, 0.0, 1.0],
                "track_length": 2,
                "reprojection_error": 0.5,
                "camera_id": 1,
                "image_width": 3,
                "image_height": 3,
            },
            sort_keys=True,
        )
        + "\n"
    )

    rendered_map_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(manifest_path),
            "--track_bank",
            str(track_bank_path),
            "--track_observations",
            str(observations_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--mode",
            "projected_grid",
            "--default_camera",
            "2,3,3,1,1,1,0",
            "--method",
            "projected_rendered_selected_map",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[1]["score"] > rows[0]["score"]
    assert report["mean_top1_acc"] == pytest.approx(1.0)
    assert report["rendered_map_parameters"]["mode"] == "projected_grid"
    assert report["rendered_map_parameters"]["default_camera"] == "2,3,3,1,1,1,0"


def test_projected_grid_cli_can_filter_tracks_by_reference_visibility(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    manifest_path = tmp_path / "query_manifest.json"
    track_bank_path = tmp_path / "tracks.npz"
    observations_path = tmp_path / "observations.jsonl"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _selector_checkpoint(checkpoint)
    _spatial_query_manifest(tmp_path).to_json(manifest_path)
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(
                    track_id=1,
                    mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                    variance=np.zeros(2, dtype=np.float32),
                    observation_count=2,
                    mean_utility=1.0,
                    observation_image_ids=("ref.png",),
                ),
                2: TrackFeature(
                    track_id=2,
                    mean_feature=np.asarray([-1.0, 0.0], dtype=np.float32),
                    variance=np.zeros(2, dtype=np.float32),
                    observation_count=2,
                    mean_utility=1.0,
                    observation_image_ids=("other.png",),
                ),
            },
            feature_dim=2,
        ),
        track_bank_path,
    )
    CandidateHypothesisBank.from_candidates(
        protocol_name="projected_grid_visibility_filter_smoke",
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="candidate",
                candidate_type="rendered_pose_lattice",
                reference_image="ref.png",
                pose=np.eye(4, dtype=np.float64).tolist(),
                pose_error=PoseCost(0.05, 1.0),
            )
        ],
    ).to_jsonl(bank_path)
    lines = []
    for track_id, image_id in ((1, "ref.png"), (2, "other.png")):
        lines.append(
            json.dumps(
                {
                    "track_id": track_id,
                    "image_id": image_id,
                    "point2d_idx": track_id,
                    "xy": [1.0, 1.0],
                    "xyz": [0.0, 0.0, 1.0],
                    "track_length": 2,
                    "reprojection_error": 0.5,
                    "camera_id": 1,
                    "image_width": 3,
                    "image_height": 3,
                },
                sort_keys=True,
            )
        )
    observations_path.write_text("\n".join(lines) + "\n")

    rendered_map_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(manifest_path),
            "--track_bank",
            str(track_bank_path),
            "--track_observations",
            str(observations_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--mode",
            "projected_grid",
            "--default_camera",
            "2,3,3,1,1,1,0",
            "--projected_visibility_filter",
            "reference_image",
            "--method",
            "projected_rendered_selected_map_refvis",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[0]["score"] > 0.9
    assert rows[0]["mean_similarity"] > 0.9
    assert rows[0]["inlier_fraction"] == pytest.approx(1.0)
    assert rows[0]["match_count"] == 1
    assert rows[0]["visibility_fraction"] == pytest.approx(1.0)
    assert rows[0]["risk"] == pytest.approx(0.0)
    assert report["rendered_map_parameters"]["projected_visibility_filter"] == "reference_image"
    assert report["evidence_summary"]["mean_match_count"] == pytest.approx(1.0)
    assert report["evidence_summary"]["mean_visibility_fraction"] == pytest.approx(1.0)
    assert report["evidence_summary"]["empty_evidence_fraction"] == pytest.approx(0.0)


def test_projected_grid_cli_rejects_camera_track_model_mismatch_when_summary_exists(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    summary_path = tmp_path / "observations_summary.json"
    observations_path.write_text("")
    summary_path.write_text(json.dumps({"model_dir": "/expected/model"}))

    with pytest.raises(ValueError, match="camera_model_dir does not match"):
        rendered_map_cli_main(
            [
                "--bank",
                str(tmp_path / "missing_bank.jsonl"),
                "--query_manifest",
                str(tmp_path / "missing_manifest.json"),
                "--track_bank",
                str(tmp_path / "missing_tracks.npz"),
                "--track_observations",
                str(observations_path),
                "--selector_checkpoint",
                str(tmp_path / "missing_selector.pt"),
                "--layer_name",
                "radio_final",
                "--mode",
                "projected_grid",
                "--camera_model_dir",
                "/other/model",
                "--output_rows",
                str(tmp_path / "rows.json"),
                "--output_report",
                str(tmp_path / "report.json"),
            ]
        )


def test_build_track_visibility_index_uses_observation_jsonl(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    _write_track_observations(observations_path, extra_reference=True)

    visibility = build_track_visibility_index(observations_path)

    assert visibility["correct_ref.png"] == {1}
    assert visibility["wrong_ref.png"] == {2}
    assert visibility["missing_ref.png"] == {99}
