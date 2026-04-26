#!/usr/bin/env python3
"""
Generate a compact localization report bundle with metrics and optional visuals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.loc_reporting import create_report_bundle, ensure_dir


DEFAULT_BASE_DIR = Path("/root/result/loc")


def _parse_metric_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _assign_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Metrics JSON must contain an object at the top level: {path}")
    return loaded


def _merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable localization report bundles.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name used in report title/output path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory. Defaults to /root/result/loc/<exp_name> when exp_name is provided.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        action="append",
        default=[],
        help="Path to a JSON file containing metrics. Can be provided multiple times and will be merged in order.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        action="append",
        default=[],
        help='Inline metric assignment such as "rot.median=1.24" or "success.rate=0.82".',
    )
    parser.add_argument(
        "--summary-line",
        type=str,
        action="append",
        default=[],
        help="Compact summary line to include near the top of the report.",
    )
    parser.add_argument(
        "--note",
        type=str,
        action="append",
        default=[],
        help="Free-form note appended to the report.",
    )
    parser.add_argument(
        "--panel-image",
        type=str,
        action="append",
        default=[],
        help="Image path to include in a single side-by-side qualitative panel. Repeat for multiple images.",
    )
    parser.add_argument(
        "--panel-caption",
        type=str,
        action="append",
        default=[],
        help="Caption corresponding to each --panel-image entry.",
    )
    parser.add_argument("--panel-title", type=str, default=None, help="Optional title for the qualitative panel.")
    parser.add_argument(
        "--panel-name",
        type=str,
        default="qualitative_panel.png",
        help="Filename for the saved qualitative panel inside the output directory.",
    )
    return parser.parse_args()


def resolve_output_dir(exp_name: str, requested_output_dir: str | None) -> Path:
    if requested_output_dir:
        return Path(requested_output_dir)
    if exp_name:
        return DEFAULT_BASE_DIR / exp_name
    return DEFAULT_BASE_DIR / "loc_report"


def build_metrics(args: argparse.Namespace) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for metrics_json in args.metrics_json:
        metrics = _merge_dicts(metrics, _load_metrics_json(Path(metrics_json)))

    for metric_arg in args.metric:
        if "=" not in metric_arg:
            raise ValueError(f"Invalid --metric value (expected key=value): {metric_arg}")
        key, raw_value = metric_arg.split("=", 1)
        _assign_nested(metrics, key.strip(), _parse_metric_value(raw_value.strip()))

    return metrics


def main() -> None:
    args = parse_args()
    metrics = build_metrics(args)
    if not metrics:
        raise ValueError("No metrics were provided. Use --metrics-json and/or --metric.")

    exp_name = args.exp_name
    if not exp_name:
        if args.output_dir:
            exp_name = Path(args.output_dir).name
        elif args.metrics_json:
            exp_name = Path(args.metrics_json[0]).stem
        else:
            exp_name = "loc_report"

    output_dir = ensure_dir(resolve_output_dir(exp_name, args.output_dir))

    if args.panel_caption and len(args.panel_caption) != len(args.panel_image):
        raise ValueError("The number of --panel-caption values must match --panel-image values.")

    bundle = create_report_bundle(
        exp_name=exp_name,
        output_dir=output_dir,
        metrics=metrics,
        summary_lines=args.summary_line,
        notes=args.note,
        panel_images=args.panel_image or None,
        panel_captions=args.panel_caption or None,
        panel_title=args.panel_title,
        panel_name=args.panel_name,
    )

    print(f"Report bundle created: {bundle['output_dir']}")
    print(f"  metrics.json: {bundle['metrics']['json']}")
    print(f"  metrics.txt: {bundle['metrics']['text']}")
    print(f"  report.md: {bundle['report']['markdown']}")
    print(f"  report.txt: {bundle['report']['text']}")
    for panel_path in bundle["panels"]:
        print(f"  panel: {panel_path}")


if __name__ == "__main__":
    main()
