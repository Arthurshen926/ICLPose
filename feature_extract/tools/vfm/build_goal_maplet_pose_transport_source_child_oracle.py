"""Replace pose-transport query child evidence with contributor-derived oracle evidence.

This is a diagnostic intervention, not a deployable localization input.  It keeps
the sealed candidate poses, rendered target evidence, RADIO tensor, reliability,
and evaluation labels byte-for-byte unchanged.  Only the query token -> physical
child distribution is replaced, using the coordinate-correct contributor raster.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_extract.tools.vfm.audit_goal_maplet_soft_child_renderer_contract import (
    _contributor_token_children,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _load_dataset,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)


SCHEMA = "goal_maplet_real_sparse_pose_transport_dataset_v2"
SOURCE_SEMANTICS = (
    "diagnostic_query_contributor_coordinate_correct_token_child_oracle_v1"
)


def _contributor_inventory(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in sorted(path.glob("*.npz")):
        with np.load(source, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                continue
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        image_id = str(metadata.get("image_id", ""))
        if not image_id:
            continue
        if image_id in result:
            raise ValueError(f"duplicate contributor image ID: {image_id}")
        result[image_id] = source.resolve()
    return result


def _replace_source_evidence(
    arrays: dict[str, np.ndarray],
    *,
    source_rows: np.ndarray,
    source_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(source_rows, dtype=np.int32)
    probabilities = np.asarray(source_probabilities, dtype=np.float32)
    expected = arrays["source_child_rows"].shape
    if rows.shape != expected or probabilities.shape != expected:
        raise ValueError("oracle source evidence shape differs from sealed dataset")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("oracle source probabilities are invalid")
    if np.any(np.sum(probabilities, axis=2, dtype=np.float64) > 1.0 + 2.0e-6):
        raise ValueError("oracle source probabilities exceed token capacity")
    result = {name: np.asarray(value).copy() for name, value in arrays.items()}
    result["source_child_rows"] = rows
    result["source_child_probabilities"] = probabilities
    return result


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dataset", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_dataset).resolve()
    physical_path = Path(args.physical_map).resolve()
    contributor_root = Path(args.contributors).resolve()
    output_path = Path(args.output_npz).resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError("refusing to overwrite source-child oracle dataset")
    if input_path == output_path:
        raise ValueError("oracle dataset must not overwrite its sealed parent")

    arrays, parent_metadata = _load_dataset(input_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    if (
        parent_metadata.get("physical_map_sha256") != physical.content_sha256
        or parent_metadata.get("physical_map_file_sha256") != file_sha256(physical_path)
    ):
        raise ValueError("live physical map differs from sealed dataset")
    image_ids = [str(value) for value in arrays["image_ids"].tolist()]
    inventory = _contributor_inventory(contributor_root)
    missing = sorted(set(image_ids) - set(inventory))
    if missing:
        raise ValueError(f"contributors missing dataset queries: {missing[:3]}")

    slot_count = int(arrays["source_child_rows"].shape[2])
    rows_out: list[np.ndarray] = []
    probabilities_out: list[np.ndarray] = []
    null_out: list[np.ndarray] = []
    coordinate_audits: list[dict[str, object]] = []
    contributor_files: list[dict[str, str]] = []
    expected_hashes = [str(value) for value in arrays["contributor_file_sha256"].tolist()]
    for query_index, image_id in enumerate(image_ids):
        contributor = inventory[image_id]
        live_hash = file_sha256(contributor)
        if live_hash != expected_hashes[query_index]:
            raise ValueError(f"contributor bytes differ for {image_id}")
        rows, probabilities, null, coordinate = _contributor_token_children(
            physical,
            contributor,
            token_height=36,
            token_width=64,
            top_l=slot_count,
        )
        rows_out.append(rows.astype(np.int32))
        probabilities_out.append(probabilities.astype(np.float32))
        null_out.append(null.astype(np.float32))
        coordinate_audits.append(coordinate)
        contributor_files.append(
            {"image_id": image_id, "path": str(contributor), "file_sha256": live_hash}
        )

    result = _replace_source_evidence(
        arrays,
        source_rows=np.stack(rows_out),
        source_probabilities=np.stack(probabilities_out),
    )
    result_hash = arrays_sha256(result)
    unchanged_names = sorted(
        name for name in arrays
        if name not in {"source_child_rows", "source_child_probabilities"}
    )
    if not all(np.array_equal(arrays[name], result[name]) for name in unchanged_names):
        raise AssertionError("oracle intervention changed non-source arrays")
    observed = np.sum(np.stack(probabilities_out), axis=2, dtype=np.float64)
    null = np.stack(null_out).astype(np.float64)
    coordinate_hash = canonical_json_sha256({"rows": coordinate_audits})
    metadata = {
        **parent_metadata,
        "content_sha256": result_hash,
        "source_child_semantics": SOURCE_SEMANTICS,
        "source_child_oracle_diagnostic": True,
        "diagnostic_uses_query_contributor_geometry": True,
        "diagnostic_uses_query_pose_upstream_of_contributor": True,
        "production_eligible": False,
        "end_to_end_localization_claim_eligible": False,
        "parent_dataset_path": str(input_path),
        "parent_dataset_file_sha256": file_sha256(input_path),
        "parent_dataset_content_sha256": parent_metadata["content_sha256"],
        "source_evidence_intervention_only": True,
        "unchanged_array_names": unchanged_names,
        "contributor_files": contributor_files,
        "contributor_coordinate_audits_sha256": coordinate_hash,
        "mean_oracle_observed_mass": float(np.mean(observed)),
        "minimum_oracle_observed_mass": float(np.min(observed)),
        "maximum_oracle_mass_conservation_error": float(
            np.max(np.abs(observed + null - 1.0), initial=0.0)
        ),
    }
    _atomic_savez(output_path, result, metadata)
    reopened_arrays, reopened_metadata = _load_dataset(output_path)
    if (
        reopened_metadata.get("content_sha256") != result_hash
        or any(not np.array_equal(result[name], reopened_arrays[name]) for name in result)
    ):
        raise AssertionError("oracle dataset replay differs after save")
    sidecar = output_path.with_suffix(".json")
    summary = {
        "artifact_type": "goal_maplet_pose_transport_source_child_oracle_build_v1",
        "output_npz": str(output_path),
        "output_file_sha256": file_sha256(output_path),
        "output_content_sha256": result_hash,
        "source_child_semantics": SOURCE_SEMANTICS,
        "query_count": len(image_ids),
        "source_slots": slot_count,
        "mean_oracle_observed_mass": float(np.mean(observed)),
        "minimum_oracle_observed_mass": float(np.min(observed)),
        "parent_dataset_content_sha256": parent_metadata["content_sha256"],
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    summary["content_sha256"] = canonical_json_sha256(summary)
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
