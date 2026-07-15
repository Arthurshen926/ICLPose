from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    contract_compatibility_signature,
    image_root_manifest,
    inference_contract_compatibility_signature,
    require_compatible_contracts,
    require_inference_compatible_contracts,
    validate_sampling_dimensions,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 6), color=color).save(path)


def test_image_manifest_is_content_sensitive_and_root_location_independent(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _write_image(first_root / "frame.png", (10, 20, 30))
    _write_image(second_root / "frame.png", (10, 20, 30))

    first = image_root_manifest(first_root, ["frame.png"])
    second = image_root_manifest(second_root, ["frame.png"])
    assert first["sampled_content_manifest_sha256"] == second[
        "sampled_content_manifest_sha256"
    ]
    assert first["resolved_image_root"] != second["resolved_image_root"]

    _write_image(second_root / "frame.png", (30, 20, 10))
    changed = image_root_manifest(second_root, ["frame.png"])
    assert changed["sampled_content_manifest_sha256"] != first[
        "sampled_content_manifest_sha256"
    ]


def test_coordinate_dimensions_and_runtime_contract_are_fail_closed() -> None:
    coordinate = {
        "coordinate_space_id": "space",
        "image_width": 1024,
        "image_height": 576,
    }
    validate_sampling_dimensions(
        coordinate, image_width=1024, image_height=576
    )
    with pytest.raises(ValueError, match="expected 1024x576"):
        validate_sampling_dimensions(
            coordinate, image_width=1920, image_height=1080
        )

    expected = {
        "version": 1,
        "coordinate_space": coordinate,
        "image_source": {
            "image_count": 1,
            "image_ids_sha256": "ids",
            "sampled_content_manifest_sha256": "pixels",
            "source_image_dimensions": {"1920x1080": 1},
        },
        "candidate_evidence_sha256": "candidate",
        "availability_evidence_sha256": "availability",
        "rows_sha256": {"train": "rows"},
    }
    relocated = {
        **expected,
        "image_source": {
            **expected["image_source"],
            "resolved_image_root": "/a/different/copy",
        },
    }
    assert contract_compatibility_signature(expected) == (
        contract_compatibility_signature(relocated)
    )
    require_compatible_contracts(expected, relocated, context="test")

    changed = {
        **relocated,
        "image_source": {
            **relocated["image_source"],
            "sampled_content_manifest_sha256": "different",
        },
    }
    with pytest.raises(ValueError, match="data contract mismatch"):
        require_compatible_contracts(expected, changed, context="test")


def test_inference_contract_allows_new_candidates_but_not_new_rgb_inputs() -> None:
    expected = {
        "version": 1,
        "coordinate_space": {
            "coordinate_space_id": "space",
            "image_width": 1024,
            "image_height": 576,
        },
        "image_source": {
            "image_count": 2,
            "image_ids_sha256": "ids",
            "sampled_content_manifest_sha256": "pixels",
            "source_image_dimensions": {"1920x1080": 2},
        },
        "candidate_evidence_sha256": "training-candidates",
        "availability_evidence_sha256": "training-availability",
        "rows_sha256": {"train": "training-rows"},
    }
    runtime = {
        **expected,
        "candidate_evidence_sha256": "inferred-candidates",
        "availability_evidence_sha256": "inferred-availability",
        "rows_sha256": {
            "train": "new-train-rows",
            "validation": "new-validation-rows",
            "test": "new-test-rows",
        },
    }

    assert inference_contract_compatibility_signature(expected) == (
        inference_contract_compatibility_signature(runtime)
    )
    require_inference_compatible_contracts(expected, runtime, context="test")
    with pytest.raises(ValueError, match="data contract mismatch"):
        require_compatible_contracts(expected, runtime, context="test")

    changed_pixels = {
        **runtime,
        "image_source": {
            **runtime["image_source"],
            "sampled_content_manifest_sha256": "different-pixels",
        },
    }
    with pytest.raises(ValueError, match="RGB inference contract mismatch"):
        require_inference_compatible_contracts(
            expected, changed_pixels, context="test"
        )
