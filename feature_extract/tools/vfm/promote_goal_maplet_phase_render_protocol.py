"""Freeze a validated raster protocol into an existing phase policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import load_phase_readout_policy


def _report(path: Path, expected_policy_sha256: str, factor: int) -> dict[str, object]:
    report = json.loads(path.read_text())
    contract = dict(report.get("surface_verification_contract") or {})
    if str(contract.get("phase_readout_policy_sha256", "")) != expected_policy_sha256:
        raise ValueError("render-protocol report used a different phase policy")
    if int(contract.get("render_supersample_factor", 1)) != int(factor):
        raise ValueError("render-protocol report used a different supersample factor")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_policy", required=True)
    parser.add_argument("--selection_report", required=True)
    parser.add_argument("--confirmation_report", required=True)
    parser.add_argument("--render_supersample_factor", type=int, required=True)
    parser.add_argument("--output_policy", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_policy)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite promoted phase policy")
    factor = int(args.render_supersample_factor)
    if factor <= 1:
        raise ValueError("promotion requires a mask-aware supersampled render")
    input_path = Path(args.input_policy)
    policy = load_phase_readout_policy(input_path)
    input_sha256 = file_sha256(input_path)
    selection_path = Path(args.selection_report)
    confirmation_path = Path(args.confirmation_report)
    selection = _report(selection_path, input_sha256, factor)
    confirmation = _report(confirmation_path, input_sha256, factor)
    selection_ids = {str(row["image_id"]).split("/", 1)[0] for row in selection.get("rows", [])}
    confirmation_ids = {str(row["image_id"]).split("/", 1)[0] for row in confirmation.get("rows", [])}
    if selection_ids & confirmation_ids:
        raise ValueError("render selection and confirmation trajectories overlap")
    payload = dict(policy.metadata)
    payload.update({
        "required_render_supersample_factor": factor,
        "render_protocol": "complete_clean_2dgs_mask_aware_supersample_pool_v1",
        "base_phase_policy_sha256": input_sha256,
        "render_protocol_selection_report_sha256": file_sha256(selection_path),
        "render_protocol_confirmation_report_sha256": file_sha256(confirmation_path),
        "render_protocol_selection_trajectory_ids": sorted(selection_ids),
        "render_protocol_confirmation_trajectory_ids": sorted(confirmation_ids),
        "confirmation_used_for_protocol_selection": False,
        "promotion_note": "same single stored canonical field and frozen two-coefficient phase readout; only mask-aware render sampling changed",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    # Reload the final artifact so malformed policies fail before promotion.
    promoted = load_phase_readout_policy(output)
    print(json.dumps({
        "output_policy": str(output),
        "output_policy_sha256": file_sha256(output),
        "component_names": list(promoted.component_names),
        "required_render_supersample_factor": factor,
        "selection_trajectory_ids": sorted(selection_ids),
        "confirmation_trajectory_ids": sorted(confirmation_ids),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
