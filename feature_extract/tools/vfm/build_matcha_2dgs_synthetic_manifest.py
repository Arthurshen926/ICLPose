"""Build a thin MATCHA streaming manifest for synthetic 2DGS source/target pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.matcha_streaming_manifest import (
    MatchaStreamingPairManifest,
    build_streaming_pair_records,
)
from feature_extract.vfm.matcha_synthetic_pairs import (
    DEFAULT_MIN_OVERLAP,
    DEFAULT_MIN_SUPERVISION_COUNT,
    DEFAULT_SOURCE_ROTATION_RANGE_DEG,
    DEFAULT_SOURCE_TRANSLATION_RANGE_M,
    DEFAULT_TARGET_ROTATION_RANGE_DEG,
    DEFAULT_TARGET_TRANSLATION_RANGE_M,
    SYNTHETIC_PAIR_SOURCE,
    SYNTHETIC_RANDOM_PAIR_TYPE,
    SYNTHETIC_RANDOM_PAIR_TYPE_ID,
)
from feature_extract.vfm.tokens import TokenBankManifest


POSE_BIN_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "micro": {
        "source_translation_range_m": (0.0, 0.03),
        "source_rotation_range_deg": (0.0, 1.0),
        "target_translation_range_m": (0.0, 0.03),
        "target_rotation_range_deg": (0.0, 1.0),
    },
    "small": {
        "source_translation_range_m": (0.0, 0.03),
        "source_rotation_range_deg": (0.0, 1.0),
        "target_translation_range_m": (0.03, 0.10),
        "target_rotation_range_deg": (1.0, 3.0),
    },
    "medium": {
        "source_translation_range_m": (0.0, 0.03),
        "source_rotation_range_deg": (0.0, 1.0),
        "target_translation_range_m": (0.10, 0.25),
        "target_rotation_range_deg": (3.0, 6.0),
    },
    "large": {
        "source_translation_range_m": (0.0, 0.05),
        "source_rotation_range_deg": (0.0, 2.0),
        "target_translation_range_m": (0.25, 0.50),
        "target_rotation_range_deg": (6.0, 10.0),
    },
    "reference_like": {
        "source_translation_range_m": (0.0, 0.10),
        "source_rotation_range_deg": (0.0, 5.0),
        "target_translation_range_m": (0.50, 4.50),
        "target_rotation_range_deg": (10.0, 35.0),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--pair_type", default=SYNTHETIC_RANDOM_PAIR_TYPE, choices=(SYNTHETIC_RANDOM_PAIR_TYPE,))
    parser.add_argument("--pose_bin", default="", choices=("", *POSE_BIN_PRESETS.keys()))
    parser.add_argument("--source_translation_range_m", default=_format_range_default(DEFAULT_SOURCE_TRANSLATION_RANGE_M))
    parser.add_argument("--source_rotation_range_deg", default=_format_range_default(DEFAULT_SOURCE_ROTATION_RANGE_DEG))
    parser.add_argument("--target_translation_range_m", default=_format_range_default(DEFAULT_TARGET_TRANSLATION_RANGE_M))
    parser.add_argument("--target_rotation_range_deg", default=_format_range_default(DEFAULT_TARGET_ROTATION_RANGE_DEG))
    parser.add_argument("--min_supervision_count", type=int, default=DEFAULT_MIN_SUPERVISION_COUNT)
    parser.add_argument("--min_overlap", type=float, default=DEFAULT_MIN_OVERLAP)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--target_pair_count", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--seed", type=int, default=0)
    explicit = set()
    if argv is not None:
        items = list(argv)
        for name in (
            "--source_translation_range_m",
            "--source_rotation_range_deg",
            "--target_translation_range_m",
            "--target_rotation_range_deg",
        ):
            if any(str(item) == name or str(item).startswith(f"{name}=") for item in items):
                explicit.add(name)
    args = parser.parse_args(argv)
    if str(args.pose_bin):
        preset = POSE_BIN_PRESETS[str(args.pose_bin)]
        if "--source_translation_range_m" not in explicit:
            args.source_translation_range_m = _format_range_default(preset["source_translation_range_m"])
        if "--source_rotation_range_deg" not in explicit:
            args.source_rotation_range_deg = _format_range_default(preset["source_rotation_range_deg"])
        if "--target_translation_range_m" not in explicit:
            args.target_translation_range_m = _format_range_default(preset["target_translation_range_m"])
        if "--target_rotation_range_deg" not in explicit:
            args.target_rotation_range_deg = _format_range_default(preset["target_rotation_range_deg"])
    _validate_args(args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = TokenBankManifest.from_json(Path(args.query_manifest))
    records = _select_records(
        source.records,
        int(args.max_queries),
        str(args.view_selection),
        start_index=int(args.start_index),
    )
    query_ids = [str(record.image_id) for record in records]
    pair_query_ids = _cycle_query_ids_to_target_pair_count(query_ids, int(args.target_pair_count))
    pair_records = build_streaming_pair_records(
        pair_query_ids,
        split=str(args.split_name),
        pair_types=(str(args.pair_type),),
        pair_type_ids={SYNTHETIC_RANDOM_PAIR_TYPE: SYNTHETIC_RANDOM_PAIR_TYPE_ID},
        seed=int(args.seed),
    )
    manifest = MatchaStreamingPairManifest(
        records=pair_records,
        metadata={
            "pair_source": SYNTHETIC_PAIR_SOURCE,
            "source_query_manifest": str(args.query_manifest),
            "view_selection": str(args.view_selection),
            "start_index": int(args.start_index),
            "max_queries": int(args.max_queries),
            "target_pair_count": int(args.target_pair_count),
            "seed": int(args.seed),
            "synthetic_pose_bin_preset": str(args.pose_bin),
            "synthetic_source_translation_range_m": list(_parse_range(args.source_translation_range_m, "source_translation_range_m")),
            "synthetic_source_rotation_range_deg": list(_parse_range(args.source_rotation_range_deg, "source_rotation_range_deg")),
            "synthetic_target_translation_range_m": list(_parse_range(args.target_translation_range_m, "target_translation_range_m")),
            "synthetic_target_rotation_range_deg": list(_parse_range(args.target_rotation_range_deg, "target_rotation_range_deg")),
            "synthetic_min_supervision_count": int(args.min_supervision_count),
            "synthetic_min_overlap": float(args.min_overlap),
        },
    )
    manifest.to_json(Path(args.output_manifest))
    summary = manifest.to_dict()
    summary.pop("records", None)
    summary["stage"] = "matcha_2dgs_synthetic_manifest_builder"
    summary["outputs"] = {"manifest": str(args.output_manifest)}
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _validate_args(args: argparse.Namespace) -> None:
    _parse_range(args.source_translation_range_m, "source_translation_range_m")
    _parse_range(args.source_rotation_range_deg, "source_rotation_range_deg")
    _parse_range(args.target_translation_range_m, "target_translation_range_m")
    _parse_range(args.target_rotation_range_deg, "target_rotation_range_deg")
    if int(args.min_supervision_count) < 0:
        raise ValueError("min_supervision_count must be non-negative")
    if float(args.min_overlap) < 0.0 or float(args.min_overlap) > 1.0:
        raise ValueError("min_overlap must be in [0, 1]")
    if int(args.target_pair_count) < 0:
        raise ValueError("target_pair_count must be non-negative")


def _select_records(records: Sequence[object], max_queries: int, mode: str, start_index: int = 0) -> list[object]:
    values = list(records)[max(0, int(start_index)) :]
    if int(max_queries) <= 0 or len(values) <= int(max_queries):
        return values
    if mode == "uniform":
        indices = np.linspace(0, len(values) - 1, int(max_queries), dtype=np.int64)
        return [values[int(idx)] for idx in indices]
    return values[: int(max_queries)]


def _cycle_query_ids_to_target_pair_count(query_ids: Sequence[str], target_pair_count: int) -> list[str]:
    values = [str(query_id) for query_id in query_ids]
    if int(target_pair_count) <= 0 or len(values) >= int(target_pair_count):
        return values
    output = []
    for index in range(int(target_pair_count)):
        output.append(values[index % len(values)])
    return output


def _parse_range(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be formatted as min,max")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{name} must be formatted as min,max")
    minimum = float(parts[0])
    maximum = float(parts[1])
    if minimum < 0.0 or maximum < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if minimum > maximum:
        raise ValueError(f"{name} must be ordered min,max")
    return minimum, maximum


def _format_range_default(value: tuple[float, float]) -> str:
    return f"{float(value[0])},{float(value[1])}"


if __name__ == "__main__":
    main()
