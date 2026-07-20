"""Coordinate and real-image provenance contracts for RGB measurement."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary


def _short_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()[:16]


def coordinate_space_contract(
    cameras: Mapping[int, Any], *, colmap_model_dir: Path, cameras_sha256: str
) -> dict[str, Any]:
    sizes = sorted({(int(camera.width), int(camera.height)) for camera in cameras.values()})
    if len(sizes) != 1:
        raise ValueError(
            "RGB measurement requires one shared COLMAP coordinate size; "
            f"found {sizes}"
        )
    width, height = sizes[0]
    identity = {
        "version": 1,
        "source": "colmap_camera_pixel_coordinates",
        "colmap_cameras_sha256": str(cameras_sha256),
        "camera_count": int(len(cameras)),
        "image_width": int(width),
        "image_height": int(height),
        "rgb_sampling_policy": "coordinate_space_scaled_to_loaded_rgb_tensor",
    }
    return {
        **identity,
        "colmap_model_dir": str(Path(colmap_model_dir).resolve()),
        "coordinate_space_id": _short_json_hash(identity),
    }


def coordinate_space_from_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    model_dir_text = str(metadata.get("colmap_model_dir", "")).strip()
    expected_hash = str(metadata.get("colmap_cameras_sha256", "")).strip()
    if not model_dir_text or not expected_hash:
        raise ValueError(
            "candidate evidence must identify its COLMAP model and cameras hash"
        )
    model_dir = Path(model_dir_text)
    cameras_path = model_dir / "cameras.bin"
    actual_hash = file_sha256_short(cameras_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "candidate evidence COLMAP camera artifact is stale: "
            f"metadata={expected_hash}, actual={actual_hash}"
        )
    cameras = read_colmap_cameras_binary(cameras_path)
    return coordinate_space_contract(
        cameras,
        colmap_model_dir=model_dir,
        cameras_sha256=actual_hash,
    )


def validate_sampling_dimensions(
    coordinate_space: Mapping[str, Any], *, image_width: int, image_height: int
) -> None:
    expected = (
        int(coordinate_space.get("image_width", -1)),
        int(coordinate_space.get("image_height", -1)),
    )
    actual = (int(image_width), int(image_height))
    if actual != expected:
        raise ValueError(
            "RGB sampling dimensions must equal the frozen SfM/candidate coordinate "
            f"space: expected {expected[0]}x{expected[1]}, got {actual[0]}x{actual[1]}"
        )


def _sampled_file_digest(path: Path, *, sample_bytes: int = 32768) -> str:
    size = int(path.stat().st_size)
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(int(sample_bytes)))
        if size > int(sample_bytes):
            handle.seek(max(0, size - int(sample_bytes)))
            digest.update(handle.read(int(sample_bytes)))
    return digest.hexdigest()[:16]


def image_root_manifest(
    image_root: Path, image_ids: Sequence[str]
) -> dict[str, Any]:
    root = Path(image_root).resolve(strict=True)
    unique_ids = sorted({str(value).strip() for value in image_ids if str(value).strip()})
    if not unique_ids:
        raise ValueError("RGB image manifest requires at least one image id")
    rows: list[str] = []
    dimensions: Counter[str] = Counter()
    for image_id in unique_ids:
        relative = Path(image_id)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"RGB image id escapes image_root: {image_id!r}")
        path = (root / relative).resolve(strict=True)
        if root != path and root not in path.parents:
            raise ValueError(f"RGB image id escapes image_root: {image_id!r}")
        with Image.open(path) as image:
            width, height = image.size
        dimensions[f"{int(width)}x{int(height)}"] += 1
        rows.append(
            f"{image_id}:{path.stat().st_size}:{width}x{height}:"
            f"{_sampled_file_digest(path)}"
        )
    return {
        "version": 1,
        "resolved_image_root": str(root),
        "image_count": int(len(unique_ids)),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(unique_ids).encode("utf8")
        ).hexdigest()[:16],
        "sampled_content_manifest_sha256": hashlib.sha256(
            "\n".join(rows).encode("utf8")
        ).hexdigest()[:16],
        "source_image_dimensions": dict(sorted(dimensions.items())),
        "sampled_bytes_per_file_end": 32768,
    }


def image_source_contract_signature(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable real-image identity for a spatial feature cache.

    This is intentionally separate from a candidate/measurement data contract:
    image-context caches are shared by multiple target-free probes, but their
    descriptors cease to be meaningful as soon as either image IDs or source
    pixels change.
    """

    value = dict(contract)
    required = {
        "version",
        "resolved_image_root",
        "image_count",
        "image_ids_sha256",
        "sampled_content_manifest_sha256",
        "source_image_dimensions",
        "sampled_bytes_per_file_end",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"image source contract lacks fields: {missing}")
    signature = {
        "version": int(value["version"]),
        "resolved_image_root": str(value["resolved_image_root"]),
        "image_count": int(value["image_count"]),
        "image_ids_sha256": str(value["image_ids_sha256"]),
        "sampled_content_manifest_sha256": str(
            value["sampled_content_manifest_sha256"]
        ),
        "source_image_dimensions": dict(value["source_image_dimensions"]),
        "sampled_bytes_per_file_end": int(value["sampled_bytes_per_file_end"]),
    }
    if (
        signature["version"] != 1
        or signature["image_count"] <= 0
        or not signature["resolved_image_root"]
        or not signature["image_ids_sha256"]
        or not signature["sampled_content_manifest_sha256"]
        or not signature["source_image_dimensions"]
        or signature["sampled_bytes_per_file_end"] <= 0
    ):
        raise ValueError("image source contract is invalid")
    return signature


def require_compatible_image_source_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, context: str
) -> None:
    """Fail closed when two cached image descriptor sources differ."""

    expected_signature = image_source_contract_signature(expected)
    actual_signature = image_source_contract_signature(actual)
    if expected_signature != actual_signature:
        differences = {
            key: {
                "expected": expected_signature.get(key),
                "actual": actual_signature.get(key),
            }
            for key in sorted(set(expected_signature) | set(actual_signature))
            if expected_signature.get(key) != actual_signature.get(key)
        }
        raise ValueError(
            f"{context} image source contract mismatch: "
            f"{json.dumps(differences, sort_keys=True)}"
        )


def contract_compatibility_signature(contract: Mapping[str, Any]) -> dict[str, Any]:
    coordinate = dict(contract.get("coordinate_space", {}))
    images = dict(contract.get("image_source", {}))
    return {
        "version": int(contract.get("version", -1)),
        "coordinate_space_id": str(coordinate.get("coordinate_space_id", "")),
        "coordinate_width": int(coordinate.get("image_width", -1)),
        "coordinate_height": int(coordinate.get("image_height", -1)),
        "image_count": int(images.get("image_count", -1)),
        "image_ids_sha256": str(images.get("image_ids_sha256", "")),
        "image_sampled_content_manifest_sha256": str(
            images.get("sampled_content_manifest_sha256", "")
        ),
        "source_image_dimensions": dict(images.get("source_image_dimensions", {})),
        "candidate_evidence_sha256": str(
            contract.get("candidate_evidence_sha256", "")
        ),
        "availability_evidence_sha256": str(
            contract.get("availability_evidence_sha256", "")
        ),
        "rows_sha256": dict(contract.get("rows_sha256", {})),
    }


def inference_contract_compatibility_signature(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable RGB-input part of a training data contract.

    Candidate evidence and measurement-row artifacts are query-specific model
    inputs.  Requiring their hashes to equal the training artifacts would make
    a trained verifier unusable for a newly inferred candidate posterior.  The
    coordinate system and real-image source remain fail-closed because changing
    either changes the meaning or pixels of every sampled patch.
    """

    signature = contract_compatibility_signature(contract)
    for key in (
        "candidate_evidence_sha256",
        "availability_evidence_sha256",
        "rows_sha256",
    ):
        signature.pop(key)
    return signature


def require_compatible_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, context: str
) -> None:
    expected_signature = contract_compatibility_signature(expected)
    actual_signature = contract_compatibility_signature(actual)
    if expected_signature != actual_signature:
        differences = {
            key: {"expected": expected_signature.get(key), "actual": actual_signature.get(key)}
            for key in sorted(set(expected_signature) | set(actual_signature))
            if expected_signature.get(key) != actual_signature.get(key)
        }
        raise ValueError(
            f"{context} RGB data contract mismatch: "
            f"{json.dumps(differences, sort_keys=True)}"
        )


def require_inference_compatible_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, context: str
) -> None:
    """Validate deployment inputs without pinning query-specific candidates."""

    expected_signature = inference_contract_compatibility_signature(expected)
    actual_signature = inference_contract_compatibility_signature(actual)
    if expected_signature != actual_signature:
        differences = {
            key: {
                "expected": expected_signature.get(key),
                "actual": actual_signature.get(key),
            }
            for key in sorted(set(expected_signature) | set(actual_signature))
            if expected_signature.get(key) != actual_signature.get(key)
        }
        raise ValueError(
            f"{context} RGB inference contract mismatch: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
