from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from feature_extract.tools.vfm.calibrate_candidate_highres_rgb_prior_temperature import (
    CALIBRATION_FORMAT,
)
from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
)
from feature_extract.tools.vfm.score_candidate_highres_rgb_multiscale_likelihood import (
    _validate_calibration_for_scoring,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _calibration(*, checkpoint: Path, layout: Path, passed: bool) -> dict[str, object]:
    return {
        "format": CALIBRATION_FORMAT,
        "runtime_safe": True,
        "contains_target_fields": False,
        "calibration_uses_train_only_targets": True,
        "appearance_control_semantics": "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2",
        "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "checkpoint_inner_validation_used_for_model_selection": False,
        "checkpoint_sha256": file_sha256_short(checkpoint),
        "candidate_prior_temperature": 0.35,
        "source": "fine",
        "safety": {
            "heldout_evaluation_allowed": passed,
            "pnp_integration_allowed": False,
            "promotion_allowed": False,
        },
        "inner_validation": {"gate": {"passed": passed}},
        "lineage": {
            "layout_sha256": file_sha256_short(layout),
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "source_image_manifest_sha256": "images",
        },
    }


def test_highres_scorer_requires_a_passed_calibration_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    layout_path = tmp_path / "layout.npz"
    calibration_path = tmp_path / "calibration.json"
    checkpoint.write_bytes(b"checkpoint")
    layout_path.write_bytes(b"layout")
    calibration_path.write_text("{}")
    layout = SimpleNamespace(metadata={"projection_space_id": "projection", "descriptor_space_id": "descriptor"})

    temperature, source = _validate_calibration_for_scoring(
        calibration=_calibration(checkpoint=checkpoint, layout=layout_path, passed=True),
        calibration_path=calibration_path,
        checkpoint_path=checkpoint,
        layout_path=layout_path,
        layout=layout,
        source_image_manifest_sha256="images",
    )
    assert temperature == 0.35
    assert source == "fine"

    with pytest.raises(ValueError, match="not eligible"):
        _validate_calibration_for_scoring(
            calibration=_calibration(checkpoint=checkpoint, layout=layout_path, passed=False),
            calibration_path=calibration_path,
            checkpoint_path=checkpoint,
            layout_path=layout_path,
            layout=layout,
            source_image_manifest_sha256="images",
        )


def test_highres_scorer_rejects_calibration_from_validation_selected_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    layout_path = tmp_path / "layout.npz"
    checkpoint.write_bytes(b"checkpoint")
    layout_path.write_bytes(b"layout")
    calibration = _calibration(checkpoint=checkpoint, layout=layout_path, passed=True)
    calibration["checkpoint_inner_validation_used_for_model_selection"] = True
    layout = SimpleNamespace(metadata={"projection_space_id": "projection", "descriptor_space_id": "descriptor"})
    with pytest.raises(ValueError, match="not eligible"):
        _validate_calibration_for_scoring(
            calibration=calibration,
            calibration_path=tmp_path / "calibration.json",
            checkpoint_path=checkpoint,
            layout_path=layout_path,
            layout=layout,
            source_image_manifest_sha256="images",
        )
