"""Gate any target-free visual evidence family against its matched control.

The gate is deliberately evidence-family agnostic.  It only consumes
sequence-grouped train OOF candidate audits, checks that their immutable
baseline is identical, and requires visual hard-repeat evidence to exceed its
matched non-visual control.  A pass authorizes at most one frozen validation
candidate audit; it never trains or selects a pose solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from feature_extract.tools.vfm.evaluate_frozen_fulltrack_spatial_pyramid_control_gate import (
    evaluate_paired_visual_control_gate,
)


ARTIFACT_FORMAT = "frozen_fulltrack_paired_visual_control_gate_v1"
STAGE = "evaluate_frozen_fulltrack_paired_visual_control_gate"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-audit-summary", required=True)
    parser.add_argument("--control-audit-summary", required=True)
    parser.add_argument("--visual-family", required=True)
    parser.add_argument("--control-family", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--minimum-hard-pairs", type=int, default=50)
    parser.add_argument("--minimum-visual-hard-wilson-lower", type=float, default=0.50)
    parser.add_argument(
        "--minimum-visual-over-control-wilson-lower",
        type=float,
        default=0.02,
    )
    return parser.parse_args(argv)


def evaluate_frozen_fulltrack_paired_visual_control_gate(
    *,
    visual_audit_summary: Path,
    control_audit_summary: Path,
    visual_family: str,
    control_family: str,
    output_json: Path,
    minimum_hard_pairs: int = 50,
    minimum_visual_hard_wilson_lower: float = 0.50,
    minimum_visual_over_control_wilson_lower: float = 0.02,
) -> dict[str, Any]:
    return evaluate_paired_visual_control_gate(
        visual_audit_summary=Path(visual_audit_summary),
        control_audit_summary=Path(control_audit_summary),
        visual_family=str(visual_family),
        control_family=str(control_family),
        output_json=Path(output_json),
        artifact_format=ARTIFACT_FORMAT,
        stage=STAGE,
        minimum_hard_pairs=int(minimum_hard_pairs),
        minimum_visual_hard_wilson_lower=float(minimum_visual_hard_wilson_lower),
        minimum_visual_over_control_wilson_lower=float(
            minimum_visual_over_control_wilson_lower
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate_frozen_fulltrack_paired_visual_control_gate(
        visual_audit_summary=Path(args.visual_audit_summary),
        control_audit_summary=Path(args.control_audit_summary),
        visual_family=str(args.visual_family),
        control_family=str(args.control_family),
        output_json=Path(args.output_json),
        minimum_hard_pairs=int(args.minimum_hard_pairs),
        minimum_visual_hard_wilson_lower=float(args.minimum_visual_hard_wilson_lower),
        minimum_visual_over_control_wilson_lower=float(
            args.minimum_visual_over_control_wilson_lower
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
