from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import feature_extract.vfm.measurement_v1.candidate_rgb_inference as inference_module
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.measurement_v1.candidate_rgb_inference import (
    INFERENCE_EVIDENCE_ARRAYS,
    INFERENCE_ROW_FIELDS,
    CandidateRGBInferenceData,
    materialize_candidate_rgb_inference_inputs,
    prepare_candidate_rgb_inference_batch,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import TensorImageLRUCache


def _write_source_rows(
    path: Path,
    *,
    split: str,
    query_id: str,
    support_id: str,
    source_query_row: int,
    track_id: int,
    candidate_hash: str,
) -> None:
    fields = [
        *[key for key in INFERENCE_ROW_FIELDS if key != "supervision_source_row_index"],
        "query_gt_x",
        "query_gt_y",
        "target_gt_projected_x",
        "target_gt_projected_y",
        "target_geometry_correct_2px",
        "geometry_supervision_weight",
    ]
    row = {key: "" for key in fields}
    row.update(
        {
            "query_id": query_id,
            "support_image_id": support_id,
            "track_id": str(track_id),
            "support_track_id": str(track_id),
            "track_length": "4",
            "support_x": "50",
            "support_y": "50",
            "render_x": "50",
            "render_y": "50",
            "center_x": "50",
            "center_y": "50",
            "support_reprojection_error": "0.2",
            "support_frame_gap": "1",
            "candidate_identity_key": f"identity-{split}",
            "support_view_set_id": f"views-{split}",
            "candidate_measurement_rank": "1",
            "candidate_score_rank": "1",
            "candidate_role": "identity_rank_1",
            "candidate_prototype_id": "0",
            "candidate_bank_row": "10",
            "candidate_assignment_probability": "0.4",
            "candidate_retrieval_similarity": "0.8",
            "source_query_row": str(source_query_row),
            "split": split,
            "support_view_rank": "0",
            "support_view_probability": "1.0",
            "query_gt_x": "51",
            "query_gt_y": "49",
            "target_gt_projected_x": "50.5",
            "target_gt_projected_y": "49.5",
            "target_geometry_correct_2px": "True",
            "geometry_supervision_weight": "1.0",
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    summary = {
        "stage": "candidate_specific_real_rgb_measurement_rows",
        "split": split,
        "coordinate_space": {"coordinate_space_id": "test"},
        "protocol": {
            "pose_used_for_candidate_or_support_selection": False,
            "pose_derived_features_exposed_to_rgb_scorer": False,
        },
        "inputs": {"selection_artifact_sha256": candidate_hash},
        "outputs": {"rows_csv_sha256": file_sha256_short(path)},
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary))


def _write_inputs(root: Path) -> tuple[Path, dict[str, Path]]:
    evidence = root / "candidate_evidence.npz"
    n, candidates, views = 3, 2, 2
    arrays = {
        "selected_rows": np.asarray([10, 20, 30], dtype=np.int64),
        "query_ids": np.asarray(["q_train.png", "q_validation.png", "q_test.png"]),
        "query_xy": np.full((n, 2), 50.0, dtype=np.float32),
        "split_names": np.asarray(["train", "validation", "test"]),
        "candidate_compact_columns": np.tile(
            np.asarray([[0, 1]], dtype=np.int64), (n, 1)
        ),
        "candidate_source_columns": np.tile(
            np.asarray([[0, 1]], dtype=np.int64), (n, 1)
        ),
        "candidate_roles": np.tile(
            np.asarray([["rank1", "rank2"]]), (n, 1)
        ),
        "candidate_valid": np.ones((n, candidates), dtype=bool),
        "candidate_track_ids": np.asarray(
            [[101, 102], [201, 202], [301, 302]], dtype=np.int64
        ),
        "candidate_prototype_ids": np.zeros((n, candidates), dtype=np.int64),
        "candidate_bank_rows": np.tile(
            np.asarray([[10, 11]], dtype=np.int64), (n, 1)
        ),
        "candidate_prior_probabilities": np.tile(
            np.asarray([[0.4, 0.2]], dtype=np.float32), (n, 1)
        ),
        "candidate_score_ranks": np.tile(
            np.asarray([[1, 2]], dtype=np.int64), (n, 1)
        ),
        "candidate_coarse_similarities": np.tile(
            np.asarray([[0.8, 0.7]], dtype=np.float32), (n, 1)
        ),
        "candidate_support_view_probabilities": np.full(
            (n, candidates, views), 0.5, dtype=np.float32
        ),
        "source_set_dustbin_probability": np.full((n,), 0.1, dtype=np.float32),
        "retained_candidate_probability_mass": np.full((n,), 0.6, dtype=np.float32),
        "omitted_candidate_probability_mass": np.full((n,), 0.3, dtype=np.float32),
        "unknown_probability": np.full((n,), 0.4, dtype=np.float32),
        "candidate_target_gt_residuals_px": np.zeros(
            (n, candidates), dtype=np.float32
        ),
    }
    metadata = {
        "format": "candidate_evidence_v3",
        "ground_truth_used_for_selection": False,
        "pose_used_for_selection": False,
        "image_retrieval": False,
        "submap": False,
        "render": False,
        "colmap_model_dir": str(root / "model"),
        "colmap_cameras_sha256": "camera",
    }
    np.savez(
        evidence,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    evidence_hash = file_sha256_short(evidence)
    rows: dict[str, Path] = {}
    for split, query_row, track in (
        ("train", 10, 101),
        ("validation", 20, 201),
        ("test", 30, 301),
    ):
        path = root / f"{split}.csv"
        _write_source_rows(
            path,
            split=split,
            query_id=f"q_{split}.png",
            support_id=f"s_{split}.png",
            source_query_row=query_row,
            track_id=track,
            candidate_hash=evidence_hash,
        )
        rows[split] = path
    return evidence, rows


def test_materialization_strips_all_target_fields(tmp_path: Path) -> None:
    evidence, rows = _write_inputs(tmp_path)
    output = tmp_path / "inference"
    materialize_candidate_rgb_inference_inputs(
        candidate_evidence=evidence,
        rows_by_split=rows,
        output_dir=output,
    )
    with np.load(
        output / "candidate_rgb_inference_evidence_v1.npz", allow_pickle=False
    ) as payload:
        assert set(payload.files) == {*INFERENCE_EVIDENCE_ARRAYS, "metadata_json"}
        assert "candidate_target_gt_residuals_px" not in payload.files
        metadata = json.loads(str(payload["metadata_json"].item()))
        assert metadata["contains_ground_truth"] is False
    with (output / "validation.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        sanitized = list(reader)
    assert tuple(reader.fieldnames or ()) == INFERENCE_ROW_FIELDS
    assert len(sanitized) == 1
    assert "query_gt_x" not in sanitized[0]
    assert sanitized[0]["supervision_source_row_index"] == "0"


def test_inference_data_rejects_extra_target_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, rows = _write_inputs(tmp_path)
    output = tmp_path / "inference"
    materialize_candidate_rgb_inference_inputs(
        candidate_evidence=evidence,
        rows_by_split=rows,
        output_dir=output,
    )
    monkeypatch.setattr(
        inference_module,
        "coordinate_space_from_evidence",
        lambda _metadata: {
            "coordinate_space_id": "test",
            "image_width": 100,
            "image_height": 100,
        },
    )
    validation = output / "validation.csv"
    with validation.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    with validation.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*INFERENCE_ROW_FIELDS, "query_gt_x"]
        )
        writer.writeheader()
        writer.writerow({**source[0], "query_gt_x": "51"})
    summary_path = validation.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["outputs"]["rows_csv_sha256"] = file_sha256_short(validation)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="unapproved columns"):
        CandidateRGBInferenceData(
            inference_evidence=output / "candidate_rgb_inference_evidence_v1.npz",
            rows_by_split={
                split: output / f"{split}.csv"
                for split in ("train", "validation", "test")
            },
            max_views=2,
        )


def test_inference_batch_contains_no_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, rows = _write_inputs(tmp_path)
    output = tmp_path / "inference"
    materialize_candidate_rgb_inference_inputs(
        candidate_evidence=evidence,
        rows_by_split=rows,
        output_dir=output,
    )
    monkeypatch.setattr(
        inference_module,
        "coordinate_space_from_evidence",
        lambda _metadata: {
            "coordinate_space_id": "test",
            "image_width": 100,
            "image_height": 100,
        },
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    for split in ("train", "validation", "test"):
        Image.new("RGB", (100, 100), color=(80, 100, 120)).save(
            image_root / f"q_{split}.png"
        )
        Image.new("RGB", (100, 100), color=(120, 100, 80)).save(
            image_root / f"s_{split}.png"
        )
    data = CandidateRGBInferenceData(
        inference_evidence=output / "candidate_rgb_inference_evidence_v1.npz",
        rows_by_split={
            split: output / f"{split}.csv"
            for split in ("train", "validation", "test")
        },
        max_views=2,
    )
    batch = prepare_candidate_rgb_inference_batch(
        data,
        data.indices_by_split["validation"].tolist(),
        image_root=image_root,
        image_width=100,
        image_height=100,
        image_cache=TensorImageLRUCache(),
        image_cache_device=torch.device("cpu"),
        crop_radius_px=2.0,
        step_px=1.0,
    )
    assert batch["query_patches_by_group"].shape == (1, 3, 5, 5)
    assert batch["support_patches"].shape == (1, 3, 5, 5)
    assert not any("target" in key or "supervision" in key for key in batch)


def test_empty_frozen_support_view_mass_is_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, rows = _write_inputs(tmp_path)
    output = tmp_path / "inference"
    materialize_candidate_rgb_inference_inputs(
        candidate_evidence=evidence,
        rows_by_split=rows,
        output_dir=output,
    )
    monkeypatch.setattr(
        inference_module,
        "coordinate_space_from_evidence",
        lambda _metadata: {
            "coordinate_space_id": "test",
            "image_width": 100,
            "image_height": 100,
        },
    )
    validation = output / "validation.csv"
    with validation.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    source[0]["support_view_probability"] = ""
    with validation.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INFERENCE_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(source)
    summary_path = validation.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    summary["outputs"]["rows_csv_sha256"] = file_sha256_short(validation)
    summary_path.write_text(json.dumps(summary))
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("q_validation.png", "s_validation.png"):
        Image.new("RGB", (100, 100), color=(80, 100, 120)).save(image_root / name)
    data = CandidateRGBInferenceData(
        inference_evidence=output / "candidate_rgb_inference_evidence_v1.npz",
        rows_by_split={
            split: output / f"{split}.csv"
            for split in ("train", "validation", "test")
        },
        max_views=2,
    )
    batch = prepare_candidate_rgb_inference_batch(
        data,
        data.indices_by_split["validation"].tolist(),
        image_root=image_root,
        image_width=100,
        image_height=100,
        image_cache=TensorImageLRUCache(),
        image_cache_device=torch.device("cpu"),
        crop_radius_px=2.0,
        step_px=1.0,
    )
    assert torch.equal(batch["pair_view_probabilities"], torch.zeros(1))
