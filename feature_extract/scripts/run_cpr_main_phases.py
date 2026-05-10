"""Run the staged CPR mainline with explicit checkpoint handoff.

The CPR plan is intentionally staged: query alignment first, then fine WLS,
then coarse local pose-lattice scoring, and only then map-side fine-tuning.
This helper keeps that ordering concrete by warmstarting each phase from the
previous phase's best checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Phase:
    name: str
    config: Path
    warmstart_from: str | None = None


PHASES: tuple[Phase, ...] = (
    Phase(
        "phase2",
        REPO_ROOT / "feature_extract/configs/cpr_main_phase2_query_align_frozenmap.yaml",
    ),
    Phase(
        "phase3",
        REPO_ROOT / "feature_extract/configs/cpr_main_phase3_fine_wls_frozenmap.yaml",
        warmstart_from="phase2",
    ),
    Phase(
        "phase4",
        REPO_ROOT / "feature_extract/configs/cpr_main_phase4_coarse_lattice_frozenmap.yaml",
        warmstart_from="phase3",
    ),
    Phase(
        "phase5",
        REPO_ROOT / "feature_extract/configs/cpr_main_phase5_map_finetune_lattice.yaml",
        warmstart_from="phase4",
    ),
    Phase(
        "phase6",
        REPO_ROOT / "feature_extract/configs/cpr_main_phase6_joint_low_lr.yaml",
        warmstart_from="phase5",
    ),
)

PHASE_BY_NAME = {phase.name: phase for phase in PHASES}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_config = cfg.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        return deep_merge(load_yaml_config(base_path), cfg)
    return cfg


def checkpoint_path(phase: Phase) -> Path:
    cfg = load_yaml_config(phase.config)
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    return output_dir / cfg["exp_name"] / "checkpoints" / "best.pth"


def selected_phases(start: str, end: str) -> list[Phase]:
    names = [phase.name for phase in PHASES]
    start_idx = names.index(start)
    end_idx = names.index(end)
    if start_idx > end_idx:
        raise ValueError(f"--from-phase {start} must not come after --to-phase {end}")
    return list(PHASES[start_idx : end_idx + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged CPR mainline training.")
    parser.add_argument("--from-phase", choices=PHASE_BY_NAME, default="phase2")
    parser.add_argument("--to-phase", choices=PHASE_BY_NAME, default="phase6")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-warmstart",
        action="store_true",
        help="Run a later phase without its expected previous best checkpoint.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for phase in selected_phases(args.from_phase, args.to_phase):
        cmd = [
            args.python,
            "-m",
            "feature_extract.train",
            "--config",
            str(phase.config),
        ]
        if phase.warmstart_from is not None:
            warmstart = checkpoint_path(PHASE_BY_NAME[phase.warmstart_from])
            if warmstart.is_file():
                cmd.extend(["--warmstart", str(warmstart)])
            elif not args.allow_missing_warmstart:
                raise FileNotFoundError(
                    f"{phase.name} expects warmstart checkpoint from "
                    f"{phase.warmstart_from}: {warmstart}"
                )
        if args.smoke_test:
            cmd.append("--smoke-test")

        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
