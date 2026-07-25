"""Fail-closed audit for full frozen LoFTR candidate-anchor evidence.

The LoFTR anchor residual may only be fitted after every expected frozen S0
query has exactly one aligned target-free evidence artifact.  This audit
checks that one-to-one relation, all immutable candidate arrays, the full
790-image mapping bank, source-image manifests, and the checkpoint / maplet /
geometry lineage.  It never reads pose or target fields.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_loftr_candidate_anchor_evidence import (
    FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT,
    LOFTR_ANCHOR_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_loftr_pair_cache import (
    load_frozen_loftr_pair_cache,
)
from feature_extract.vfm.localization.frozen_loftr_coordinate_contract import (
    validate_frozen_loftr_colmap_coordinate_metadata,
)
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_ARTIFACT_FORMAT,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import image_root_manifest
from feature_extract.tools.vfm.build_frozen_loftr_pair_cache import (
    kornia_pretrained_checkpoint_path,
)


FROZEN_LOFTR_ANCHOR_MANIFEST_AUDIT_FORMAT = "frozen_loftr_anchor_manifest_audit_v1"
_BASE_FIELDS = (
    "verification_query_ids",
    "split_names",
    "verification_source_row_indices",
    "verification_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "candidate_view_weights",
)


@dataclass(frozen=True)
class FrozenQueryArtifact:
    """One complete query evidence artifact with only provenance-safe fields."""

    path: Path
    sha256: str
    query_id: str
    split_name: str
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-artifact-glob", required=True)
    parser.add_argument("--anchor-artifact-glob", required=True)
    parser.add_argument("--mapping-support-manifest", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--loftr-checkpoint", required=True)
    parser.add_argument("--hloc-root", default="third_party/Hierarchical-Localization")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--expected-train-query-count", type=int, default=63)
    parser.add_argument("--expected-validation-query-count", type=int, default=21)
    parser.add_argument("--expected-mapping-image-count", type=int, default=790)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _paths(pattern: str, *, label: str) -> tuple[Path, ...]:
    paths = tuple(Path(value) for value in sorted(glob.glob(str(pattern))))
    if not paths or len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError(f"{label} glob must resolve unique existing files")
    return paths


def _json_metadata(payload: Mapping[str, np.ndarray], *, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload:
        raise ValueError(f"{path}: artifact lacks metadata_json")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata is not an object")
    return metadata


def _required_contract(*, metadata: Mapping[str, Any], anchor: bool, path: Path) -> None:
    strict_name = (
        "strict_frozen_loftr_anchor_contract"
        if anchor
        else "strict_frozen_appearance_contract"
    )
    strict = metadata.get(strict_name)
    expected = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if anchor:
        expected.update(
            {
                "support_view_descriptor_averaging": False,
                "all_mapping_images_pair_cached": True,
                "pair_cache_image_level_selection": False,
                "anchor_evidence_pose_free": True,
                "source_to_colmap_coordinate_contract": True,
                "legacy_coordinate_ambiguous_evidence_rejected": True,
            }
        )
    if (
        metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or not isinstance(strict, Mapping)
        or any(strict.get(key) is not value for key, value in expected.items())
    ):
        raise ValueError(f"{path}: frozen target-free contract is invalid")


def _load_query_artifact(path: Path, *, anchor: bool) -> FrozenQueryArtifact:
    required = (*_BASE_FIELDS, "metadata_json")
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: artifact lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in _BASE_FIELDS}
        metadata = _json_metadata(payload, path=Path(path))
        if anchor:
            anchor_required = (
                "feature_names",
                "candidate_view_usable",
                "candidate_view_features",
                "candidate_view_pair_match_counts",
                "candidate_usable_view_weight_mass",
            )
            missing = sorted(set(anchor_required).difference(payload.files))
            if missing:
                raise ValueError(f"{path}: anchor evidence lacks {missing}")
            arrays.update({name: np.asarray(payload[name]).copy() for name in anchor_required})
    expected_format = (
        FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT
        if anchor
        else FROZEN_APPEARANCE_ARTIFACT_FORMAT
    )
    if metadata.get("format") != expected_format:
        raise ValueError(f"{path}: artifact format differs from expected frozen schema")
    _required_contract(metadata=metadata, anchor=anchor, path=Path(path))
    if anchor:
        validate_frozen_loftr_colmap_coordinate_metadata(metadata)
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
    if (
        len(query_ids) != 192
        or len(set(query_ids.tolist())) != 1
        or split_names.shape != (192,)
        or len(set(split_names.tolist())) != 1
        or rows.shape != (192,)
        or xy.shape != (192, 2)
        or tracks.shape != (192, 20)
        or probabilities.shape != tracks.shape
        or null.shape != (192,)
        or weights.ndim != 3
        or weights.shape[:2] != tracks.shape
        or np.any(~np.isfinite(xy))
        or np.any(~np.isfinite(probabilities))
        or np.any(~np.isfinite(null))
        or np.any(~np.isfinite(weights))
        or np.any(probabilities < 0.0)
        or np.any(null <= 0.0)
        or np.any(weights < 0.0)
        or np.max(np.abs(probabilities.sum(axis=1, dtype=np.float64) + null - 1.0)) > 2e-4
        or str(metadata.get("query_id", "")) != str(query_ids[0])
        or str(metadata.get("split_name", "")) != str(split_names[0])
        or int(metadata.get("row_count", -1)) != 192
    ):
        raise ValueError(f"{path}: query identity or fixed top-20 tensors are invalid")
    supported = weights.sum(axis=2, dtype=np.float64)
    if (
        np.any((probabilities > 0.0) & ~np.isclose(supported, 1.0, atol=2e-4))
        or np.any((probabilities <= 0.0) & (supported > 2e-4))
    ):
        raise ValueError(f"{path}: fixed support-view mixture does not conserve candidate mass")
    if anchor:
        names = tuple(np.asarray(arrays["feature_names"]).astype(str).reshape(-1).tolist())
        usable = np.asarray(arrays["candidate_view_usable"], dtype=bool)
        values = np.asarray(arrays["candidate_view_features"], dtype=np.float32)
        counts = np.asarray(arrays["candidate_view_pair_match_counts"], dtype=np.int32)
        mass = np.asarray(arrays["candidate_usable_view_weight_mass"], dtype=np.float32)
        expected_shape = (*weights.shape, len(LOFTR_ANCHOR_FEATURE_NAMES))
        if (
            names != LOFTR_ANCHOR_FEATURE_NAMES
            or usable.shape != weights.shape
            or values.shape != expected_shape
            or counts.shape != weights.shape
            or mass.shape != tracks.shape
            or np.any(counts[usable] <= 0)
            or np.any(~np.isfinite(values[usable]))
            or np.any(~np.isfinite(mass))
            or np.any(mass < 0.0)
            or np.any(mass > 1.0001)
            or not np.allclose(
                mass, (weights * usable.astype(np.float32)).sum(axis=2), atol=2e-5
            )
        ):
            raise ValueError(f"{path}: LoFTR anchor tensors are invalid")
        implementation = metadata.get("implementation")
        if (
            not isinstance(implementation, Mapping)
            or not str(implementation.get("source_sha256", ""))
        ):
            raise ValueError(f"{path}: LoFTR anchor implementation lineage is missing")
    return FrozenQueryArtifact(
        path=Path(path),
        sha256=file_sha256_short(Path(path)),
        query_id=str(query_ids[0]),
        split_name=str(split_names[0]),
        metadata=metadata,
        arrays=arrays,
    )


def assert_frozen_anchor_aligned(
    *, direct: FrozenQueryArtifact, anchor: FrozenQueryArtifact
) -> None:
    """Reject an anchor artifact whose immutable S0 rows changed in any way."""

    if direct.query_id != anchor.query_id or direct.split_name != anchor.split_name:
        raise ValueError("LoFTR anchor query identity differs from its direct S0 artifact")
    for field in _BASE_FIELDS:
        if not np.array_equal(direct.arrays[field], anchor.arrays[field]):
            raise ValueError(f"LoFTR anchor changed immutable direct S0 field: {field}")
    inputs = anchor.metadata.get("inputs")
    source = inputs.get("appearance_artifact") if isinstance(inputs, Mapping) else None
    if not isinstance(source, Mapping) or str(source.get("sha256", "")) != direct.sha256:
        raise ValueError("LoFTR anchor appearance-artifact lineage differs from direct S0")
    if anchor.metadata.get("appearance_source_contract") != direct.metadata.get(
        "strict_frozen_appearance_contract"
    ):
        raise ValueError("LoFTR anchor stored a different direct S0 appearance contract")


def _input_sha(metadata: Mapping[str, Any], name: str, expected: str, *, path: Path) -> None:
    inputs = metadata.get("inputs")
    value = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(value, Mapping) or str(value.get("sha256", "")) != str(expected):
        raise ValueError(f"{path}: {name} lineage differs from the declared input")


def _mapping_support_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("mapping support manifest records are invalid")
    image_ids = tuple(str(item.get("image_id", "")) for item in records if isinstance(item, Mapping))
    if (
        len(image_ids) != len(records)
        or not image_ids
        or any(not value for value in image_ids)
        or len(set(image_ids)) != len(image_ids)
        or tuple(sorted(image_ids)) != image_ids
    ):
        raise ValueError("mapping support manifest image ids are invalid")
    return image_ids


def _validate_pair_cache(
    *,
    anchor: FrozenQueryArtifact,
    expected_mapping_ids: tuple[str, ...],
    mapping_manifest_sha: str,
    maplet_sha: str,
    checkpoint_sha: str,
    image_root: Path,
    hloc_root: Path,
) -> dict[str, Any]:
    inputs = anchor.metadata.get("inputs")
    cache_input = inputs.get("loftr_pair_cache") if isinstance(inputs, Mapping) else None
    if not isinstance(cache_input, Mapping):
        raise ValueError(f"{anchor.path}: LoFTR pair-cache lineage is missing")
    cache_path = Path(str(cache_input.get("path", "")))
    if not cache_path.is_file() or str(cache_input.get("sha256", "")) != file_sha256_short(cache_path):
        raise ValueError(f"{anchor.path}: LoFTR pair-cache file is stale or missing")
    cache = load_frozen_loftr_pair_cache(cache_path)
    metadata = dict(cache.metadata)
    mapping = metadata.get("mapping_support_manifest")
    maplet = metadata.get("maplet_support_index")
    matcher = metadata.get("matcher")
    source = metadata.get("image_source_contract")
    wrapper = Path(hloc_root) / "hloc" / "matchers" / "loftr.py"
    try:
        from kornia.feature.loftr.loftr import urls
        import torch
    except ImportError as error:  # pragma: no cover - LoFTR cache depends on Kornia.
        raise RuntimeError("LoFTR manifest audit requires Kornia and torch") from error
    expected_loader_checkpoint = kornia_pretrained_checkpoint_path(
        weights=str(matcher.get("weights", "")) if isinstance(matcher, Mapping) else "",
        hub_dir=Path(torch.hub.get_dir()),
        urls=urls,
    ).resolve(strict=True)
    if (
        cache.query_id != anchor.query_id
        or tuple(cache.support_image_ids.tolist()) != expected_mapping_ids
        or not isinstance(mapping, Mapping)
        or str(mapping.get("sha256", "")) != mapping_manifest_sha
        or not isinstance(maplet, Mapping)
        or str(maplet.get("sha256", "")) != maplet_sha
        or not isinstance(matcher, Mapping)
        or str(matcher.get("checkpoint_sha256", "")) != checkpoint_sha
        or not wrapper.is_file()
        or str(matcher.get("hloc_wrapper_sha256", "")) != file_sha256_short(wrapper)
        or expected_loader_checkpoint != Path(str(matcher.get("checkpoint", ""))).resolve(
            strict=True
        )
        or file_sha256_short(expected_loader_checkpoint) != checkpoint_sha
        or not isinstance(source, Mapping)
        or not isinstance(source.get("query"), Mapping)
        or not isinstance(source.get("mapping_support"), Mapping)
        or dict(source["query"]) != image_root_manifest(Path(image_root), [anchor.query_id])
    ):
        raise ValueError(f"{anchor.path}: LoFTR pair-cache provenance is incompatible")
    if anchor.metadata.get("pair_cache_contract") != metadata.get("strict_global_pair_contract"):
        raise ValueError(f"{anchor.path}: stored LoFTR pair-cache contract differs from cache")
    return {
        "path": str(cache_path),
        "sha256": str(cache_input["sha256"]),
        "match_count": int(len(cache.match_confidence)),
        "implementation_source_sha256": str(
            metadata.get("implementation", {}).get("source_sha256", "")
        ),
        "matcher_signature": {
            key: matcher.get(key)
            for key in (
                "weights",
                "checkpoint_sha256",
                "hloc_wrapper_sha256",
                "kornia_version",
                "torch_version",
                "dtype",
                "max_num_matches",
                "match_threshold",
            )
        },
        "mapping_source_manifest": dict(source["mapping_support"]),
    }


def audit_frozen_loftr_anchor_manifest(
    *,
    direct_artifacts: Sequence[Path],
    anchor_artifacts: Sequence[Path],
    mapping_support_manifest: Path,
    maplet_support_index: Path,
    support_geometry_index: Path,
    loftr_checkpoint: Path,
    image_root: Path,
    hloc_root: Path,
    expected_train_query_count: int,
    expected_validation_query_count: int,
    expected_mapping_image_count: int,
    output_json: Path,
) -> dict[str, Any]:
    """Validate exact S0-to-LoFTR coverage before any train-only residual fit."""

    output = Path(output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    expected_counts = {
        "train": int(expected_train_query_count),
        "validation": int(expected_validation_query_count),
    }
    if (
        any(value <= 0 for value in expected_counts.values())
        or int(expected_mapping_image_count) <= 0
        or not direct_artifacts
        or not anchor_artifacts
    ):
        raise ValueError("frozen LoFTR manifest audit arguments are invalid")
    mapping_ids = _mapping_support_ids(Path(mapping_support_manifest))
    if len(mapping_ids) != int(expected_mapping_image_count):
        raise ValueError("mapping support manifest count differs from the frozen protocol")
    mapping_manifest_sha = file_sha256_short(Path(mapping_support_manifest))
    maplet_sha = file_sha256_short(Path(maplet_support_index))
    geometry_sha = file_sha256_short(Path(support_geometry_index))
    checkpoint_sha = file_sha256_short(Path(loftr_checkpoint))
    mapping_source_manifest = image_root_manifest(Path(image_root), mapping_ids)
    direct_by_query: dict[str, Path] = {}
    direct_split_counts = {key: 0 for key in expected_counts}
    for path in direct_artifacts:
        direct = _load_query_artifact(Path(path), anchor=False)
        if direct.query_id in direct_by_query:
            raise ValueError("direct S0 artifacts repeat a query id")
        if direct.split_name not in direct_split_counts:
            raise ValueError("direct S0 artifact has unsupported split")
        _input_sha(direct.metadata, "maplet_support_index", maplet_sha, path=direct.path)
        _input_sha(direct.metadata, "support_geometry_index", geometry_sha, path=direct.path)
        direct_by_query[direct.query_id] = direct.path
        direct_split_counts[direct.split_name] += 1
    if direct_split_counts != expected_counts:
        raise ValueError(
            f"direct S0 split coverage differs: expected={expected_counts}, got={direct_split_counts}"
        )
    records = []
    anchor_implementation_hashes: set[str] = set()
    cache_implementation_hashes: set[str] = set()
    matcher_signatures: set[str] = set()
    anchor_split_counts = {key: 0 for key in expected_counts}
    seen_queries: set[str] = set()
    for path in anchor_artifacts:
        anchor = _load_query_artifact(Path(path), anchor=True)
        if anchor.query_id in seen_queries:
            raise ValueError("LoFTR anchor artifacts repeat a query id")
        if anchor.query_id not in direct_by_query:
            raise ValueError("LoFTR anchor artifact has no direct S0 source query")
        if anchor.split_name not in anchor_split_counts:
            raise ValueError("LoFTR anchor artifact has unsupported split")
        direct = _load_query_artifact(direct_by_query[anchor.query_id], anchor=False)
        assert_frozen_anchor_aligned(direct=direct, anchor=anchor)
        _input_sha(anchor.metadata, "maplet_support_index", maplet_sha, path=anchor.path)
        _input_sha(anchor.metadata, "support_geometry_index", geometry_sha, path=anchor.path)
        _input_sha(anchor.metadata, "loftr_checkpoint", checkpoint_sha, path=anchor.path)
        cache_record = _validate_pair_cache(
            anchor=anchor,
            expected_mapping_ids=mapping_ids,
            mapping_manifest_sha=mapping_manifest_sha,
            maplet_sha=maplet_sha,
            checkpoint_sha=checkpoint_sha,
            image_root=Path(image_root),
            hloc_root=Path(hloc_root),
        )
        if cache_record["mapping_source_manifest"] != mapping_source_manifest:
            raise ValueError(f"{anchor.path}: mapping image source manifest is stale")
        anchor_implementation_hashes.add(
            str(anchor.metadata["implementation"]["source_sha256"])
        )
        cache_implementation_hashes.add(str(cache_record["implementation_source_sha256"]))
        matcher_signatures.add(json.dumps(cache_record["matcher_signature"], sort_keys=True))
        seen_queries.add(anchor.query_id)
        anchor_split_counts[anchor.split_name] += 1
        records.append(
            {
                "query_id": anchor.query_id,
                "split_name": anchor.split_name,
                "direct_s0": {"path": str(direct.path), "sha256": direct.sha256},
                "anchor": {"path": str(anchor.path), "sha256": anchor.sha256},
                "pair_cache": cache_record,
            }
        )
    if anchor_split_counts != expected_counts or seen_queries != set(direct_by_query):
        raise ValueError(
            "LoFTR anchor coverage differs from direct S0: "
            f"expected={expected_counts}, got={anchor_split_counts}"
        )
    if (
        len(anchor_implementation_hashes) != 1
        or len(cache_implementation_hashes) != 1
        or "" in cache_implementation_hashes
        or len(matcher_signatures) != 1
    ):
        raise ValueError("LoFTR anchor artifact set mixes implementation or matcher configurations")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": FROZEN_LOFTR_ANCHOR_MANIFEST_AUDIT_FORMAT,
        "protocol": {
            "target_free": True,
            "full_mapping_pair_cache": True,
            "fixed_global_top_l": 20,
            "image_level_selection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "candidate_identity_fixed": True,
            "source_image_manifest_revalidated": True,
        },
        "expected_query_counts": expected_counts,
        "actual_query_counts": anchor_split_counts,
        "mapping_support_image_count": int(len(mapping_ids)),
        "implementation": {
            "anchor_evidence_source_sha256": next(iter(anchor_implementation_hashes)),
            "pair_cache_source_sha256": next(iter(cache_implementation_hashes)),
            "matcher_signature": json.loads(next(iter(matcher_signatures))),
        },
        "inputs": {
            "mapping_support_manifest": {
                "path": str(Path(mapping_support_manifest)),
                "sha256": mapping_manifest_sha,
            },
            "maplet_support_index": {
                "path": str(Path(maplet_support_index)),
                "sha256": maplet_sha,
            },
            "support_geometry_index": {
                "path": str(Path(support_geometry_index)),
                "sha256": geometry_sha,
            },
            "loftr_checkpoint": {
                "path": str(Path(loftr_checkpoint)),
                "sha256": checkpoint_sha,
            },
            "mapping_image_source_manifest": mapping_source_manifest,
        },
        "records": sorted(records, key=lambda item: (item["split_name"], item["query_id"])),
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_frozen_loftr_anchor_manifest(
        direct_artifacts=_paths(args.direct_artifact_glob, label="direct S0 artifact"),
        anchor_artifacts=_paths(args.anchor_artifact_glob, label="LoFTR anchor artifact"),
        mapping_support_manifest=Path(args.mapping_support_manifest),
        maplet_support_index=Path(args.maplet_support_index),
        support_geometry_index=Path(args.support_geometry_index),
        loftr_checkpoint=Path(args.loftr_checkpoint),
        image_root=Path(args.image_root),
        hloc_root=Path(args.hloc_root),
        expected_train_query_count=int(args.expected_train_query_count),
        expected_validation_query_count=int(args.expected_validation_query_count),
        expected_mapping_image_count=int(args.expected_mapping_image_count),
        output_json=Path(args.output_json),
    )
    print(
        json.dumps(
            {
                "format": summary["format"],
                "actual_query_counts": summary["actual_query_counts"],
                "mapping_support_image_count": summary["mapping_support_image_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
