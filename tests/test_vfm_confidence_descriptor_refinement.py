import json
from pathlib import Path
from typing import Tuple

import numpy as np

from feature_extract.vfm.confidence_descriptor_refinement import (
    ConfidenceDescriptorRefinementConfig,
    ConfidenceRefinementTrainingSet,
    ConfidenceDescriptorRefiner,
    DiagonalDescriptorSelectionConfig,
    DiagonalDescriptorSelector,
    _descriptor_loss,
    build_confidence_refinement_samples,
    train_diagonal_descriptor_selector,
    train_confidence_descriptor_refiner,
)
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _write_fixture(tmp_path: Path) -> Tuple[Path, Path, Path]:
    query_map = np.zeros((4, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    token_path = tmp_path / "q.npz"
    np.savez_compressed(token_path, radio_final=query_map)
    manifest = TokenBankManifest(
        records=(
            TokenBankRecord(
                image_id="q.png",
                token_path=token_path,
                layers=(TokenLayerSpec("radio_final", "synthetic", "final", 4, 16),),
                split="train",
                scene="Synthetic",
            ),
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.to_json(manifest_path)
    bank_path = tmp_path / "bank.npz"
    save_selected_track_bank_npz(
        SelectedTrackFeatureBank(
            tracks={
                1: TrackFeature(1, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), np.zeros((4,), dtype=np.float32), 6, 1.0, ("r",)),
                2: TrackFeature(2, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), np.zeros((4,), dtype=np.float32), 6, 1.0, ("r",)),
                3: TrackFeature(3, np.asarray([0.7, 0.7, 0.0, 0.0], dtype=np.float32), np.zeros((4,), dtype=np.float32), 1, 1.0, ("r",)),
            },
            feature_dim=4,
        ),
        bank_path,
    )
    match_path = tmp_path / "matches.jsonl"
    rows = [
        {"query_id": "q.png", "token_index": 0, "track_id": 1, "strong_positive_label": True, "weak_positive_label": False, "hard_negative_label": False, "ignore_label": False, "gt_reproj_error_stride": 0.1, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 3, "strong_positive_label": False, "weak_positive_label": False, "hard_negative_label": True, "ignore_label": False, "gt_reproj_error_stride": 4.0, "similarity": 0.8},
        {"query_id": "q.png", "token_index": 1, "track_id": 2, "strong_positive_label": True, "weak_positive_label": False, "hard_negative_label": False, "ignore_label": False, "gt_reproj_error_stride": 0.2, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 1, "track_id": 3, "strong_positive_label": False, "weak_positive_label": False, "hard_negative_label": True, "ignore_label": False, "gt_reproj_error_stride": 3.5, "similarity": 0.8},
    ]
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return manifest_path, bank_path, match_path


def test_build_confidence_refinement_samples_groups_matches_by_query_token(tmp_path: Path) -> None:
    manifest_path, bank_path, match_path = _write_fixture(tmp_path)

    samples, meta = build_confidence_refinement_samples(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        max_candidates_per_token=4,
    )

    assert samples.sample_count == 2
    assert samples.feature_dim == 4
    assert samples.candidate_features.shape == (2, 4, 4)
    assert samples.candidate_mask[:, :2].all()
    assert samples.labels[:, 0].tolist() == [1.0, 1.0]
    assert samples.labels[:, 1].tolist() == [0.0, 0.0]
    assert meta["hard_negative_count"] == 2


def test_build_confidence_refinement_samples_can_reserve_negative_candidates(tmp_path: Path) -> None:
    manifest_path, bank_path, _match_path = _write_fixture(tmp_path)
    rows = [
        {"query_id": "q.png", "token_index": 0, "track_id": 1, "strong_positive_label": True, "ignore_label": False, "gt_reproj_error_stride": 0.1, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 2, "strong_positive_label": True, "ignore_label": False, "gt_reproj_error_stride": 0.2, "similarity": 0.8},
        {"query_id": "q.png", "token_index": 0, "track_id": 3, "hard_negative_label": True, "ignore_label": False, "gt_reproj_error_stride": 4.0, "similarity": 0.7},
    ]
    match_path = tmp_path / "many_positive_matches.jsonl"
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    samples, meta = build_confidence_refinement_samples(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        max_candidates_per_token=2,
        min_negative_candidates=1,
    )

    assert samples.sample_count == 1
    assert samples.labels[0, :2].tolist() == [1.0, 0.0]
    assert meta["hard_negative_count"] == 1


def test_build_confidence_refinement_samples_prefers_semi_hard_gt_far_negatives(tmp_path: Path) -> None:
    manifest_path, bank_path, _match_path = _write_fixture(tmp_path)
    rows = [
        {"query_id": "q.png", "token_index": 0, "track_id": 1, "strong_positive_label": True, "ignore_label": False, "gt_reproj_error_stride": 0.1, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 2, "strong_positive_label": False, "ignore_label": True, "gt_reproj_error_stride": 1.4, "similarity": 0.95},
        {"query_id": "q.png", "token_index": 0, "track_id": 3, "hard_negative_label": False, "ignore_label": False, "gt_reproj_error_stride": 3.0, "similarity": 0.85},
    ]
    match_path = tmp_path / "semi_hard_matches.jsonl"
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    samples, meta = build_confidence_refinement_samples(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        max_candidates_per_token=2,
        min_negative_candidates=1,
        negative_sampling_mode="semi_hard",
    )

    assert samples.sample_count == 1
    assert samples.labels[0, :2].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(samples.candidate_features[0, 1], np.asarray([0.7, 0.7, 0.0, 0.0], dtype=np.float32))
    assert meta["semi_hard_negative_count"] == 1


def test_build_confidence_refinement_samples_can_weight_positives_by_reprojection_distance(tmp_path: Path) -> None:
    manifest_path, bank_path, _match_path = _write_fixture(tmp_path)
    rows = [
        {"query_id": "q.png", "token_index": 0, "track_id": 1, "strong_positive_label": True, "ignore_label": False, "gt_reproj_error_stride": 0.0, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 2, "strong_positive_label": True, "ignore_label": False, "gt_reproj_error_stride": 1.0, "similarity": 0.9},
        {"query_id": "q.png", "token_index": 0, "track_id": 3, "hard_negative_label": True, "ignore_label": False, "gt_reproj_error_stride": 4.0, "similarity": 0.8},
    ]
    match_path = tmp_path / "precision_weighted_matches.jsonl"
    match_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    samples, meta = build_confidence_refinement_samples(
        match_jsonl=match_path,
        query_manifest=manifest_path,
        landmark_bank=bank_path,
        max_candidates_per_token=3,
        positive_reprojection_weight=1.0,
        positive_reprojection_scale_stride=1.0,
    )

    assert samples.sample_count == 1
    np.testing.assert_allclose(samples.weights[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(samples.weights[0, 1], np.exp(-1.0), atol=1e-6)
    assert samples.weights[0, 0] > samples.weights[0, 1]
    assert meta["positive_reprojection_weighted_count"] == 2


def test_confidence_descriptor_refiner_improves_training_top1() -> None:
    query = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    candidates = np.asarray(
        [
            [[0.5, 1.0], [1.0, 0.5]],
            [[0.5, 1.0], [1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    weights = np.ones((2, 2), dtype=np.float32)
    teacher = labels * 0.9 + 0.05
    samples = ConfidenceRefinementTrainingSet(
        query_features=query,
        candidate_features=candidates,
        candidate_mask=mask,
        labels=labels,
        weights=weights,
        teacher_scores=teacher,
        reprojection_strides=np.asarray([[4.0, 0.1], [4.0, 0.1]], dtype=np.float32),
    )

    run = train_confidence_descriptor_refiner(
        samples,
        ConfidenceDescriptorRefinementConfig(output_dim=2, steps=120, batch_size=2, lr=0.2, distill_loss_weight=0.2),
    )

    assert run.summary.raw_train_top1_acc == 0.0
    assert run.summary.train_top1_acc == 1.0
    assert np.isfinite(run.summary.final_loss)


def test_score_anchor_loss_penalizes_descriptor_geometry_drift() -> None:
    query = np.asarray([[1.0, 1.0]], dtype=np.float32)
    candidates = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    labels = np.asarray([[1.0, 0.0]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    weights = np.ones((1, 2), dtype=np.float32)
    teacher = labels * 0.9 + 0.05
    model = ConfidenceDescriptorRefiner(input_dim=2, output_dim=2, hidden_dim=4)

    import torch

    with torch.no_grad():
        model.projection.weight.copy_(torch.asarray([[1.0, 0.0], [0.0, 0.1]], dtype=torch.float32))
    tensors = (
        torch.as_tensor(query),
        torch.as_tensor(candidates),
        torch.as_tensor(mask),
        torch.as_tensor(labels),
        torch.as_tensor(weights),
        torch.as_tensor(teacher),
    )

    unanchored = _descriptor_loss(
        model,
        *tensors,
        ConfidenceDescriptorRefinementConfig(
            output_dim=2,
            steps=1,
            score_anchor_loss_weight=0.0,
        ),
    )
    anchored = _descriptor_loss(
        model,
        *tensors,
        ConfidenceDescriptorRefinementConfig(
            output_dim=2,
            steps=1,
            score_anchor_loss_weight=5.0,
        ),
    )

    assert anchored > unanchored


def test_confidence_descriptor_refiner_starts_as_identity_when_square() -> None:
    model = ConfidenceDescriptorRefiner(input_dim=4, output_dim=4, hidden_dim=8)
    rows = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    import torch

    with torch.no_grad():
        encoded = model(torch.as_tensor(rows)).numpy()
    expected = rows / np.linalg.norm(rows, axis=1, keepdims=True)

    np.testing.assert_allclose(encoded, expected, atol=1e-6)


def test_diagonal_descriptor_selector_starts_as_identity() -> None:
    model = DiagonalDescriptorSelector(feature_dim=4, max_log_scale=0.5)
    rows = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    import torch

    with torch.no_grad():
        encoded = model(torch.as_tensor(rows)).numpy()
    expected = rows / np.linalg.norm(rows, axis=1, keepdims=True)

    np.testing.assert_allclose(encoded, expected, atol=1e-6)
    np.testing.assert_allclose(
        model.channel_scales().detach().numpy(),
        np.ones((4,), dtype=np.float32),
        atol=1e-6,
    )


def test_diagonal_descriptor_selector_improves_top1_by_reweighting_channels() -> None:
    query = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    candidates = np.asarray(
        [
            [[0.5, 1.0], [1.0, 0.5]],
            [[0.5, 1.0], [1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    weights = np.ones((2, 2), dtype=np.float32)
    teacher = labels * 0.9 + 0.05
    samples = ConfidenceRefinementTrainingSet(
        query_features=query,
        candidate_features=candidates,
        candidate_mask=mask,
        labels=labels,
        weights=weights,
        teacher_scores=teacher,
        reprojection_strides=np.asarray([[4.0, 0.1], [4.0, 0.1]], dtype=np.float32),
    )

    run = train_diagonal_descriptor_selector(
        samples,
        DiagonalDescriptorSelectionConfig(
            steps=160,
            batch_size=2,
            lr=0.25,
            margin_loss_weight=0.4,
            distill_loss_weight=0.1,
            anchor_loss_weight=0.0,
            scale_regularization_weight=0.0,
            max_log_scale=1.0,
        ),
    )

    scales = run.model.channel_scales().detach().numpy()
    assert run.summary.raw_train_top1_acc == 0.0
    assert run.summary.train_top1_acc == 1.0
    assert scales[0] > scales[1]


def test_stage_c216_diagonal_selector_cli_exports_descriptor_artifacts(tmp_path: Path) -> None:
    from feature_extract.tools.vfm.train_stage_c216_diagonal_descriptor_selection import main

    manifest_path, bank_path, match_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"

    main(
        [
            "--match_jsonl",
            str(match_path),
            "--query_manifest",
            str(manifest_path),
            "--landmark_bank",
            str(bank_path),
            "--steps",
            "4",
            "--batch_size",
            "2",
            "--output_model",
            str(output_dir / "selector.pt"),
            "--summary_json",
            str(output_dir / "summary.json"),
            "--output_query_dir",
            str(output_dir / "query_tokens"),
            "--output_query_manifest",
            str(output_dir / "query_manifest.json"),
            "--output_landmark_bank",
            str(output_dir / "landmark_bank.npz"),
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    exported_manifest = TokenBankManifest.from_json(output_dir / "query_manifest.json")
    with np.load(output_dir / "landmark_bank.npz") as data:
        feature_dim = int(data["feature_dim"])

    assert summary["stage"] == "stage_c216_diagonal_descriptor_selection"
    assert summary["training"]["output_dim"] == 4
    assert len(exported_manifest.records) == 1
    assert feature_dim == 4
