import json
from pathlib import Path
from typing import Tuple

import numpy as np

from feature_extract.vfm.confidence_descriptor_refinement import (
    ConfidenceDescriptorRefinementConfig,
    ConfidenceDescriptorRefiner,
    build_confidence_refinement_samples,
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
    from feature_extract.vfm.confidence_descriptor_refinement import ConfidenceRefinementTrainingSet

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


def test_confidence_descriptor_refiner_starts_as_identity_when_square() -> None:
    model = ConfidenceDescriptorRefiner(input_dim=4, output_dim=4, hidden_dim=8)
    rows = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    import torch

    with torch.no_grad():
        encoded = model(torch.as_tensor(rows)).numpy()
    expected = rows / np.linalg.norm(rows, axis=1, keepdims=True)

    np.testing.assert_allclose(encoded, expected, atol=1e-6)
