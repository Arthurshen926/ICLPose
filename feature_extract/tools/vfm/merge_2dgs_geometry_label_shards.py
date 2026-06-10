"""Merge sharded 2DGS geometry-label manifests without copying NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    token_manifest = TokenBankManifest.from_json(Path(args.token_manifest))
    image_order = [record.image_id for record in token_manifest.records]
    shard_dirs = [Path(path) for path in args.shard_dirs]

    record_by_image: dict[str, dict] = {}
    summaries = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "geometry_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
        payload = _load_json(manifest_path)
        summaries.append(_load_json(shard_dir / "geometry_label_summary.json"))
        for record in payload.get("records", []):
            image_id = str(record["image_id"])
            if image_id in record_by_image:
                raise ValueError(f"Duplicate image_id across shards: {image_id}")
            record_by_image[image_id] = dict(record)

    merged_records = [record_by_image[image_id] for image_id in image_order if image_id in record_by_image]
    missing = [image_id for image_id in image_order if image_id not in record_by_image]

    valid_ratios = [float(record.get("valid_ratio", float("nan"))) for record in merged_records]
    alpha_means = [float(record.get("mean_alpha", float("nan"))) for record in merged_records]
    depth_medians = [float(record.get("median_depth_m", float("nan"))) for record in merged_records]
    valid_ratios = [value for value in valid_ratios if np.isfinite(value)]
    alpha_means = [value for value in alpha_means if np.isfinite(value)]
    depth_medians = [value for value in depth_medians if np.isfinite(value)]

    first_summary = summaries[0] if summaries else {}
    merged_payload = {
        "records": merged_records,
        "inputs": first_summary.get("inputs", {}),
        "camera": first_summary.get("camera", {}),
        "render": first_summary.get("render", {}),
        "surface": first_summary.get("surface", {}),
        "shards": {
            "dirs": [str(path) for path in shard_dirs],
            "summaries": summaries,
            "missing_image_count": int(len(missing)),
            "missing_images": missing[:50],
        },
    }
    summary = {
        **{key: value for key, value in merged_payload.items() if key != "records"},
        "record_count": int(len(merged_records)),
        "mean_valid_ratio": _mean(valid_ratios),
        "median_valid_ratio": _median(valid_ratios),
        "mean_alpha": _mean(alpha_means),
        "median_depth_m": _median(depth_medians),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry_manifest.json").write_text(json.dumps(merged_payload, indent=2))
    (output_dir / "geometry_label_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
