"""Score fixed candidates after descriptor-bank control corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from feature_extract.vfm.artifacts import attach_report_inputs, file_sha256_short
from feature_extract.vfm.descriptor_controls import apply_descriptor_control, mask_descriptor_channels_by_utility
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    score_candidate_bank_by_descriptor_cosine,
)


def _parse_channel_utility(text: str) -> np.ndarray:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--channel_utility must contain at least one value")
    return np.asarray(values, dtype=np.float32)


def _load_selector_channel_utility(path: Path) -> np.ndarray:
    import torch

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if "utility_head.weight" not in state:
        raise ValueError("selector checkpoint is missing utility_head.weight")
    return np.abs(np.asarray(state["utility_head.weight"].detach().cpu(), dtype=np.float32).reshape(-1))


def _resolve_channel_utility(channel_utility: str, selector_checkpoint: str) -> np.ndarray:
    if channel_utility and selector_checkpoint:
        raise ValueError("use either --channel_utility or --selector_checkpoint, not both")
    if channel_utility:
        return _parse_channel_utility(channel_utility)
    if selector_checkpoint:
        return _load_selector_channel_utility(Path(selector_checkpoint))
    raise ValueError("utility-channel controls require --channel_utility or --selector_checkpoint")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score candidates with descriptor control corruptions")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_descriptors", required=True)
    parser.add_argument("--map_descriptors", required=True)
    parser.add_argument(
        "--control",
        required=True,
        choices=[
            "identity",
            "query_shuffle",
            "map_shuffle",
            "wrong_scene_query",
            "wrong_scene_map",
            "high_utility_channels",
            "low_utility_channels",
        ],
    )
    parser.add_argument("--replacement_descriptors", default="")
    parser.add_argument("--channel_utility", default="")
    parser.add_argument("--selector_checkpoint", default="")
    parser.add_argument("--mask_fraction", type=float, default=0.25)
    parser.add_argument("--mask_target", choices=["query", "map", "both"], default="both")
    parser.add_argument("--no_renormalize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", required=True)
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    query_descriptors = TokenDescriptorBank.from_npz(Path(args.query_descriptors))
    map_descriptors = TokenDescriptorBank.from_npz(Path(args.map_descriptors))
    replacement = (
        TokenDescriptorBank.from_npz(Path(args.replacement_descriptors))
        if args.replacement_descriptors
        else None
    )

    control = args.control
    if control == "query_shuffle":
        query_descriptors = apply_descriptor_control(query_descriptors, "query_shuffle", seed=args.seed)
    elif control == "map_shuffle":
        map_descriptors = apply_descriptor_control(map_descriptors, "map_shuffle", seed=args.seed)
    elif control == "wrong_scene_query":
        query_descriptors = apply_descriptor_control(
            query_descriptors,
            "wrong_scene",
            seed=args.seed,
            replacement_bank=replacement,
        )
    elif control == "wrong_scene_map":
        map_descriptors = apply_descriptor_control(
            map_descriptors,
            "wrong_scene",
            seed=args.seed,
            replacement_bank=replacement,
        )
    elif control in {"high_utility_channels", "low_utility_channels"}:
        remove = "high" if control == "high_utility_channels" else "low"
        utility = _resolve_channel_utility(args.channel_utility, args.selector_checkpoint)
        if args.mask_target in {"query", "both"}:
            query_descriptors = mask_descriptor_channels_by_utility(
                query_descriptors,
                utility=utility,
                fraction=args.mask_fraction,
                remove=remove,
                renormalize=not args.no_renormalize,
            )
        if args.mask_target in {"map", "both"}:
            map_descriptors = mask_descriptor_channels_by_utility(
                map_descriptors,
                utility=utility,
                fraction=args.mask_fraction,
                remove=remove,
                renormalize=not args.no_renormalize,
            )

    rows = score_candidate_bank_by_descriptor_cosine(
        bank=bank,
        query_descriptors=query_descriptors,
        map_descriptors=map_descriptors,
        method=args.method,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
    )
    row_payload = []
    for row in rows:
        item = dict(row.__dict__)
        item["protocol_kind"] = row.protocol_kind.value
        row_payload.append(item)
    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(row_payload, indent=2, sort_keys=True) + "\n")

    report = evaluate_score_table(rows).to_dict()
    report_with_inputs = attach_report_inputs(
        report,
        bank,
        Path(args.bank),
        {
            "query_descriptors": args.query_descriptors,
            "map_descriptors": args.map_descriptors,
            "replacement_descriptors": args.replacement_descriptors,
            "selector_checkpoint": args.selector_checkpoint,
        },
    )
    report_with_inputs["inputs"]["control"] = control
    report_with_inputs["inputs"]["control_seed"] = int(args.seed)
    report_with_inputs["inputs"]["mask_fraction"] = float(args.mask_fraction)
    report_with_inputs["inputs"]["mask_target"] = args.mask_target
    report_with_inputs["inputs"]["renormalized_after_mask"] = not args.no_renormalize
    if args.replacement_descriptors:
        report_with_inputs["inputs"]["replacement_descriptors_sha256"] = file_sha256_short(
            Path(args.replacement_descriptors)
        )
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report_with_inputs, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Descriptor Control", [report]) + "\n")


if __name__ == "__main__":
    main()
