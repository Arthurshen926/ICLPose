from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm import train_real_radio_joint_localization
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineTrainingSet
from feature_extract.vfm.matcha_joint_training import MatchaJointTrainingSet


def _coarse_rows() -> MatchaCoarseFineTrainingSet:
    return MatchaCoarseFineTrainingSet(
        query_features=np.ones((1, 4), dtype=np.float32),
        render_features=np.ones((1, 4), dtype=np.float32),
        query_offset_labels=np.asarray([0], dtype=np.int64),
        render_offset_labels=np.asarray([0], dtype=np.int64),
        negative_render_features=np.zeros((1, 1, 4), dtype=np.float32),
        roundtrip_errors_px=np.zeros((1,), dtype=np.float32),
    )


def _full_joint_set() -> MatchaJointTrainingSet:
    return MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_sample_pair_indices=np.asarray([0], dtype=np.int64),
        fine_query_cell_indices=np.asarray([0], dtype=np.int64),
        fine_render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_query_offset_labels=np.asarray([0], dtype=np.int64),
        fine_render_offset_labels=np.asarray([0], dtype=np.int64),
        fine_query_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        fine_render_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
    )


def test_train_real_radio_joint_localization_cli_defaults_to_full_joint_training() -> None:
    args = train_real_radio_joint_localization.parse_args(
        [
            "--joint_cache",
            "train.npz",
            "--output_model",
            "adapter.pt",
            "--output_joint_model",
            "joint.pt",
            "--summary_json",
            "summary.json",
            "--steps",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert args.joint_cache == "train.npz"
    assert args.model_type == "residual_adapter"
    assert args.measurement_patch_loss_weight == pytest.approx(1.0)
    assert args.local_window_fine_loss_weight > 0.0
    assert not hasattr(args, "sample_cache")
    assert not hasattr(args, "render_cache_manifest_csv")


def test_validate_joint_localization_training_set_rejects_row_only_samples() -> None:
    row_only = MatchaJointTrainingSet(coarse_fine_samples=_coarse_rows())

    with pytest.raises(ValueError, match="full-map"):
        train_real_radio_joint_localization._validate_joint_localization_training_set(row_only, source="train")


def test_validate_joint_localization_training_set_requires_fine_supervision_for_measurement_loss() -> None:
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="fine.*measurement"):
        train_real_radio_joint_localization._validate_joint_localization_training_set(
            samples,
            source="train",
            require_measurement_supervision=True,
        )


def test_validate_joint_localization_training_set_allows_different_query_reference_grid_sizes() -> None:
    samples = MatchaJointTrainingSet(
        coarse_fine_samples=_coarse_rows(),
        query_feature_maps=np.ones((1, 4, 2, 2), dtype=np.float32),
        render_feature_maps=np.ones((1, 4, 1, 2), dtype=np.float32),
        query_heatmap_targets=np.zeros((1, 2, 2), dtype=np.float32),
        render_heatmap_targets=np.zeros((1, 1, 2), dtype=np.float32),
        sample_pair_indices=np.asarray([0], dtype=np.int64),
        query_cell_indices=np.asarray([0], dtype=np.int64),
        render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_sample_pair_indices=np.asarray([0], dtype=np.int64),
        fine_query_cell_indices=np.asarray([0], dtype=np.int64),
        fine_render_cell_indices=np.asarray([0], dtype=np.int64),
        fine_query_offset_labels=np.asarray([0], dtype=np.int64),
        fine_render_offset_labels=np.asarray([0], dtype=np.int64),
        fine_query_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        fine_render_xy=np.asarray([[4.0, 4.0]], dtype=np.float32),
        query_rgb_images=np.zeros((1, 3, 16, 16), dtype=np.float32),
        render_rgb_images=np.zeros((1, 3, 8, 16), dtype=np.float32),
    )

    audit = train_real_radio_joint_localization._validate_joint_localization_training_set(
        samples,
        source="train",
        require_measurement_supervision=True,
    )

    assert audit["query_feature_map_shape"] == [1, 4, 2, 2]
    assert audit["reference_feature_map_shape"] == [1, 4, 1, 2]


def test_train_real_radio_joint_localization_main_uses_joint_backend(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    full_set = _full_joint_set()

    def fake_load_npz(path):
        captured["loaded_path"] = Path(path)
        return full_set, {"format": "fake_real_joint_cache"}

    def fake_train(samples, config, **kwargs):
        captured["samples"] = samples
        captured["config"] = config
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(model=SimpleNamespace(), summary={"sample_count": 1, "final_loss": 0.0})

    monkeypatch.setattr(train_real_radio_joint_localization, "load_matcha_joint_training_set_npz", fake_load_npz)
    monkeypatch.setattr(train_real_radio_joint_localization, "train_matcha_joint_model", fake_train)
    monkeypatch.setattr(train_real_radio_joint_localization, "joint_run_as_coarse_fine_adapter_run", lambda run: run)
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_coarse_fine_adapter", lambda run, path: captured.setdefault("adapter_path", Path(path)))
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_joint_model", lambda run, path: captured.setdefault("joint_path", Path(path)))

    summary_json = tmp_path / "summary.json"
    train_real_radio_joint_localization.main(
        [
            "--joint_cache",
            "train.npz",
            "--output_model",
            str(tmp_path / "adapter.pt"),
            "--output_joint_model",
            str(tmp_path / "joint.pt"),
            "--summary_json",
            str(summary_json),
            "--steps",
            "3",
            "--device",
            "cpu",
        ]
    )

    assert captured["loaded_path"] == Path("train.npz")
    assert captured["samples"] is full_set
    assert captured["config"].model_type == "residual_adapter"
    assert captured["config"].measurement_patch_loss_weight == pytest.approx(1.0)
    assert captured["config"].steps == 3
    assert captured["adapter_path"] == tmp_path / "adapter.pt"
    assert captured["joint_path"] == tmp_path / "joint.pt"
    summary = json.loads(summary_json.read_text())
    assert summary["stage"] == "real_radio_joint_localization_training"
    assert summary["joint_training_contract"]["requires_full_feature_maps"] is True
    assert summary["joint_training_contract"]["requires_rgb_measurement_images"] is True


def test_train_real_radio_joint_localization_main_uses_referenced_lazy_provider(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    full_set = _full_joint_set()
    manifest = tmp_path / "referenced_manifest.json"
    manifest.write_text(json.dumps({"format": "vfm_real_radio_joint_referenced_manifest_v1", "records": [{}]}) + "\n")

    class FakeProvider:
        def __init__(self, path, **kwargs):
            captured.setdefault("provider_paths", []).append(Path(path))
            captured.setdefault("provider_kwargs", []).append(dict(kwargs))

        def __len__(self):
            return 3

        def get(self, index):
            captured.setdefault("provider_get_indices", []).append(int(index))
            return full_set

    def fake_train_from_provider(sample_count, get_sample, config, **kwargs):
        captured["provider_sample_count"] = int(sample_count)
        captured["provider_callable"] = get_sample
        captured["config"] = config
        captured["train_kwargs"] = kwargs
        assert get_sample(1) is full_set
        return SimpleNamespace(model=SimpleNamespace(), summary={"sample_count": 3, "provider_lazy_training": True})

    monkeypatch.setattr(train_real_radio_joint_localization, "RealRadioReferencedJointSampleProvider", FakeProvider)
    monkeypatch.setattr(train_real_radio_joint_localization, "train_matcha_joint_model_from_sample_provider", fake_train_from_provider)
    monkeypatch.setattr(train_real_radio_joint_localization, "joint_run_as_coarse_fine_adapter_run", lambda run: run)
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_coarse_fine_adapter", lambda run, path: captured.setdefault("adapter_path", Path(path)))
    monkeypatch.setattr(train_real_radio_joint_localization, "save_matcha_joint_model", lambda run, path: captured.setdefault("joint_path", Path(path)))

    summary_json = tmp_path / "summary.json"
    train_real_radio_joint_localization.main(
        [
            "--joint_cache_manifest",
            str(manifest),
            "--output_model",
            str(tmp_path / "adapter.pt"),
            "--output_joint_model",
            str(tmp_path / "joint.pt"),
            "--summary_json",
            str(summary_json),
            "--steps",
            "2",
            "--referenced_feature_cache_size",
            "256",
            "--referenced_rgb_cache_size",
            "128",
            "--provider_prefetch_workers",
            "2",
            "--provider_prefetch_depth",
            "8",
            "--provider_gradient_accumulation_pairs",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert captured["provider_paths"] == [manifest]
    assert captured["provider_kwargs"] == [{"feature_cache_size": 256, "rgb_cache_size": 128}]
    assert captured["provider_get_indices"] == [0, 1]
    assert captured["provider_sample_count"] == 3
    assert captured["config"].model_type == "residual_adapter"
    assert captured["train_kwargs"]["provider_name"] == "real_radio_referenced_manifest"
    assert captured["train_kwargs"]["provider_prefetch_workers"] == 2
    assert captured["train_kwargs"]["provider_prefetch_depth"] == 8
    assert captured["train_kwargs"]["provider_gradient_accumulation_pairs"] == 4
    assert captured["adapter_path"] == tmp_path / "adapter.pt"
    assert captured["joint_path"] == tmp_path / "joint.pt"
    summary = json.loads(summary_json.read_text())
    assert summary["joint_cache"]["metadata"]["format"] == "vfm_real_radio_joint_referenced_manifest_v1"
    assert "records" not in summary["joint_cache"]["metadata"]
    assert summary["joint_cache"]["metadata"]["record_count"] == 1
    assert summary["training"]["provider_lazy_training"] is True
