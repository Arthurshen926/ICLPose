#!/usr/bin/env python3
"""Export a guarded refinement init cache with identity fallback."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries  # noqa: E402
from feature_extract.localizability.refinement_policy import build_guarded_refinement_entries  # noqa: E402


def _row_scalar(data: np.ndarray, index: int, default: float | bool) -> float | bool:
    if data.ndim == 0:
        value = data.item()
    elif data.ndim == 1:
        value = data[index]
    else:
        row = np.asarray(data[index]).reshape(-1)
        if row.dtype == np.bool_:
            value = bool(np.any(row))
        elif row.size == 0:
            value = default
        else:
            finite = row[np.isfinite(row)]
            value = float(np.max(finite)) if finite.size else default
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _load_refinement_metadata(cache_path: str | Path) -> dict[str, dict[str, object]]:
    data = np.load(str(cache_path), allow_pickle=True)
    if "query_image_names" not in data.files:
        raise ValueError(f"{cache_path} does not contain query_image_names")
    names = [str(name) for name in data["query_image_names"]]
    success_arr = np.asarray(data["refine_success"]) if "refine_success" in data.files else None
    inlier_arr = np.asarray(data["refine_num_inliers"]) if "refine_num_inliers" in data.files else None
    metadata: dict[str, dict[str, object]] = {}
    for idx, name in enumerate(names):
        metadata[name] = {
            "refine_success": True if success_arr is None else bool(_row_scalar(success_arr, idx, True)),
            "refine_num_inliers": math.inf if inlier_arr is None else float(_row_scalar(inlier_arr, idx, math.inf)),
        }
    return metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-cache", required=True)
    parser.add_argument("--refined-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics-json", default=None)
    parser.add_argument("--min-inliers", type=int, default=100)
    parser.add_argument("--max-delta-trans-m", type=float, default=0.35)
    parser.add_argument("--max-delta-rot-deg", type=float, default=5.0)
    parser.add_argument("--accepted-source", default="guarded_render_loftr_refine")
    parser.add_argument("--fallback-source", default="guarded_identity_fallback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity_entries, identity_stats = load_retrieval_init_entries(args.identity_cache)
    refined_entries, refined_stats = load_retrieval_init_entries(args.refined_cache)
    metadata = _load_refinement_metadata(args.refined_cache)
    guarded_entries, stats, diagnostics = build_guarded_refinement_entries(
        identity_entries,
        refined_entries,
        refinement_metadata=metadata,
        min_inliers=int(args.min_inliers),
        max_delta_trans_m=float(args.max_delta_trans_m),
        max_delta_rot_deg=float(args.max_delta_rot_deg),
        accepted_source=str(args.accepted_source),
        fallback_source=str(args.fallback_source),
    )
    stats["identity_cache"] = str(args.identity_cache)
    stats["refined_cache"] = str(args.refined_cache)
    stats["identity_cache_stats"] = identity_stats
    stats["refined_cache_stats"] = refined_stats
    save_retrieval_init_entries(guarded_entries, stats, args.output)

    diagnostics_json = args.diagnostics_json
    if diagnostics_json is None:
        diagnostics_json = str(Path(args.output).with_suffix(".diagnostics.json"))
    diagnostics_payload = {"stats": stats, "diagnostics": diagnostics}
    diagnostics_path = Path(diagnostics_json)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(_jsonable(diagnostics_payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({"output": str(args.output), "diagnostics_json": str(diagnostics_path), "stats": stats}), indent=2))


if __name__ == "__main__":
    main()
