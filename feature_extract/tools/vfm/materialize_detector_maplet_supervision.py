"""Attach GT residual labels to a complete inference-only maplet artifact.

The expensive candidate features must be built without supervision.  This tool
only adds labels after requiring exact full-row coverage, so GT pose cannot
affect which query tokens or candidate identities enter the training pool.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


_REQUIRED_INFERENCE_FIELDS = {
    "selected_rows",
    "selected_columns",
    "features",
    "valid_edges",
}
_LEGACY_AUDIT_FIELD = "selected_from_pose_keep"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_feature_artifact", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive_threshold_px", type=float, default=2.0)
    return parser.parse_args(argv)


def materialize_supervision(
    inference_feature_artifact: Path,
    proposals: Path,
    *,
    positive_threshold_px: float,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, object]]:
    threshold = float(positive_threshold_px)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("positive_threshold_px must be finite and positive")

    inference_path = Path(inference_feature_artifact)
    proposal_path = Path(proposals)
    with np.load(inference_path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files or "feature_names" not in payload.files:
            raise ValueError("inference feature artifact has no schema metadata")
        data_fields = set(payload.files) - {"metadata_json", "feature_names"}
        allowed_fields = _REQUIRED_INFERENCE_FIELDS | {_LEGACY_AUDIT_FIELD}
        if not _REQUIRED_INFERENCE_FIELDS.issubset(data_fields) or not data_fields.issubset(
            allowed_fields
        ):
            raise ValueError(
                "inference feature fields differ from the target-free contract: "
                f"{sorted(data_fields)}"
            )
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in _REQUIRED_INFERENCE_FIELDS
        }
        legacy_pose_keep_field_present = _LEGACY_AUDIT_FIELD in data_fields
        feature_names = tuple(np.asarray(payload["feature_names"]).astype(str).tolist())
        metadata = json.loads(str(payload["metadata_json"].item()))

    if metadata.get("format") != "detector_maplet_geometry_features_v1":
        raise ValueError("unsupported detector-maplet feature format")
    if metadata.get("supervision_mode") != "none_inference_only":
        raise ValueError("source feature artifact is not inference-only")
    source_threshold = metadata.get("positive_threshold_px")
    if source_threshold is not None and not math.isclose(
        float(source_threshold), threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("requested threshold differs from the source feature contract")
    actual_proposal_hash = file_sha256_short(proposal_path)
    if metadata.get("proposals_sha256") != actual_proposal_hash:
        raise ValueError("source feature artifact references different proposals")

    with np.load(proposal_path, allow_pickle=False) as payload:
        if "candidate_gt_residuals_px" not in payload.files:
            raise ValueError("proposal artifact has no candidate GT residuals")
        residuals = np.asarray(payload["candidate_gt_residuals_px"], dtype=np.float32)
        proposal_rows = len(np.asarray(payload["query_ids"]))
    if residuals.ndim != 2 or residuals.shape[0] != proposal_rows:
        raise ValueError("proposal residual tensor is not row-aligned")
    if np.any(np.isnan(residuals)) or np.any(residuals < 0.0):
        raise ValueError("proposal residuals must be non-negative or positive infinity")

    selected_rows = np.asarray(arrays["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(arrays["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(arrays["valid_edges"], dtype=bool)
    features = np.asarray(arrays["features"])
    expected_rows = np.arange(proposal_rows, dtype=np.int64)
    if not np.array_equal(selected_rows, expected_rows):
        raise ValueError(
            "supervised materialization requires exact full proposal-row coverage"
        )
    if selected_columns.ndim != 2 or selected_columns.shape != valid_edges.shape:
        raise ValueError("selected columns and valid-edge mask differ")
    if features.ndim != 3 or features.shape[:2] != selected_columns.shape:
        raise ValueError("feature tensor is not candidate-row aligned")
    if len(feature_names) != features.shape[2]:
        raise ValueError("feature names and tensor dimension differ")
    expected_valid = selected_columns >= 0
    if not np.array_equal(valid_edges, expected_valid):
        raise ValueError("valid-edge mask disagrees with selected columns")
    if np.any(selected_columns[valid_edges] >= residuals.shape[1]):
        raise ValueError("selected candidate column exceeds proposal width")
    safe_columns = np.maximum(selected_columns, 0)
    selected_residuals = np.take_along_axis(
        residuals[selected_rows], safe_columns, axis=1
    )
    labels = valid_edges & (selected_residuals <= threshold)
    supervised_arrays = {**arrays, "labels": labels}
    supervised_metadata = {
        **metadata,
        "supervision_mode": "candidate_gt_reprojection_residual_threshold_v1",
        "positive_threshold_px": threshold,
        "source_inference_feature_artifact": str(inference_path),
        "source_inference_feature_artifact_sha256": file_sha256_short(inference_path),
        "supervision_proposals": str(proposal_path),
        "supervision_proposals_sha256": actual_proposal_hash,
        "supervision_full_proposal_row_coverage": True,
        "legacy_pose_keep_audit_field_removed": bool(
            legacy_pose_keep_field_present
        ),
        "supervision_selected_row_count": int(len(selected_rows)),
        "supervision_positive_edge_count": int(np.sum(labels)),
        "supervision_valid_edge_count": int(np.sum(valid_edges)),
    }
    return supervised_arrays, feature_names, supervised_metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    arrays, feature_names, metadata = materialize_supervision(
        Path(args.inference_feature_artifact),
        Path(args.proposals),
        positive_threshold_px=float(args.positive_threshold_px),
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite supervised artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **arrays,
        feature_names=np.asarray(feature_names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    summary = {
        "stage": "materialize_detector_maplet_supervision",
        "output": str(output),
        "output_sha256": file_sha256_short(output),
        "source_inference_feature_artifact_sha256": metadata[
            "source_inference_feature_artifact_sha256"
        ],
        "proposals_sha256": metadata["supervision_proposals_sha256"],
        "row_count": int(len(arrays["selected_rows"])),
        "candidate_top_k": int(arrays["selected_columns"].shape[1]),
        "positive_threshold_px": float(metadata["positive_threshold_px"]),
        "positive_edge_count": int(np.sum(arrays["labels"])),
        "positive_edge_rate": float(np.mean(arrays["labels"])),
        "full_row_coverage": True,
    }
    (output.parent / "materialize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
