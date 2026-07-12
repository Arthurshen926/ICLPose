"""Build inference-only identity/geometry candidate-score fusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


def geometric_probability_fusion(
    identity: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    identity_values = np.asarray(identity, dtype=np.float32)
    geometry_values = np.asarray(geometry, dtype=np.float32)
    if identity_values.shape != geometry_values.shape:
        raise ValueError("identity and geometry score arrays must have equal shapes")
    valid = np.isfinite(identity_values) & np.isfinite(geometry_values)
    for name, values in (("identity", identity_values), ("geometry", geometry_values)):
        if np.any((values[valid] < -1e-6) | (values[valid] > 1.0 + 1e-6)):
            raise ValueError(f"{name} scores are not probabilities")
    output = np.full(identity_values.shape, -np.inf, dtype=np.float32)
    output[valid] = np.sqrt(
        np.clip(identity_values[valid], 0.0, 1.0)
        * np.clip(geometry_values[valid], 0.0, 1.0)
    ).astype(np.float32)
    return output


def _comma_list(value: str) -> tuple[str, ...]:
    output = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument("--prefixes", type=_comma_list, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.score_artifact)
    source_summary_path = source_path.parent / "summary.json"
    if not source_summary_path.exists():
        raise ValueError("source score artifact requires a sibling summary.json")
    source_summary = json.loads(source_summary_path.read_text())
    source_outputs = source_summary.get("outputs")
    recorded_hash = (
        None
        if not isinstance(source_outputs, dict)
        else source_outputs.get("scores_sha256")
    )
    source_hash = file_sha256_short(source_path)
    if not recorded_hash or str(recorded_hash) != source_hash:
        raise ValueError("source score artifact hash differs from its summary")
    data_manifest = source_summary.get("data_manifest")
    if not isinstance(data_manifest, dict):
        raise ValueError("source score summary is missing data_manifest")
    source_protocol = source_summary.get("protocol")
    if not isinstance(source_protocol, dict) or not source_protocol.get(
        "baseline_strategy"
    ):
        raise ValueError("source score summary is missing baseline strategy identity")

    with np.load(source_path, allow_pickle=False) as payload:
        source_scores = {key: np.asarray(payload[key]) for key in payload.files}
    if "baseline_scores" not in source_scores:
        raise ValueError("source score artifact has no baseline_scores")
    output_scores: dict[str, np.ndarray] = {
        "baseline_scores": np.asarray(
            source_scores["baseline_scores"], dtype=np.float32
        )
    }
    fusions: list[dict[str, str]] = []
    for prefix in tuple(args.prefixes):
        identity_key = f"{prefix}__set_candidate_probability"
        geometry_key = f"{prefix}__geometry_p05px"
        missing = {
            identity_key,
            geometry_key,
        } - set(source_scores)
        if missing:
            raise ValueError(f"source score arrays are missing: {sorted(missing)}")
        output_scores[identity_key] = np.asarray(
            source_scores[identity_key], dtype=np.float32
        )
        for suffix in (
            "geometry_p01px",
            "geometry_p02px",
            "geometry_p05px",
        ):
            auxiliary_key = f"{prefix}__{suffix}"
            if auxiliary_key not in source_scores:
                raise ValueError(f"source score array is missing: {auxiliary_key}")
            output_scores[auxiliary_key] = np.asarray(
                source_scores[auxiliary_key], dtype=np.float32
            )
        support_view_keys = sorted(
            key
            for key in source_scores
            if key.startswith(f"{prefix}__support_view_probability_")
        )
        if not support_view_keys:
            raise ValueError(f"source scores have no support-view posterior: {prefix}")
        for auxiliary_key in support_view_keys:
            output_scores[auxiliary_key] = np.asarray(
                source_scores[auxiliary_key], dtype=np.float32
            )
        output_key = f"{prefix}__identity_geometry_geomean"
        output_scores[output_key] = geometric_probability_fusion(
            source_scores[identity_key], source_scores[geometry_key]
        )
        fusions.append(
            {
                "output_key": output_key,
                "identity_key": identity_key,
                "geometry_key": geometry_key,
                "formula": "sqrt(clamp(identity,0,1)*clamp(geometry,0,1))",
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "identity_geometry_scores.npz"
    np.savez(output_path, **output_scores)
    summary = {
        "stage": "candidate_identity_geometry_score_fusion",
        "protocol": {
            "inference_only": True,
            "ground_truth_pose_read": False,
            "score_strategy_selected": False,
            "baseline_strategy": str(source_protocol["baseline_strategy"]),
            "measurement": False,
            "render": False,
            "image_retrieval": False,
            "submap": False,
        },
        "data_manifest": data_manifest,
        "source": {
            "score_artifact": str(source_path),
            "score_artifact_sha256": source_hash,
            "summary": str(source_summary_path),
            "summary_sha256": file_sha256_short(source_summary_path),
        },
        "prefixes": list(args.prefixes),
        "fusions": fusions,
        "outputs": {
            "scores": str(output_path),
            "scores_sha256": file_sha256_short(output_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
