"""Evaluate frozen projective seam correspondences on explicit atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    evaluate_projective_exact_face_seam_geometry,
)


def _arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must have NAME=PATH form")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("arm must have NAME=PATH form")
    return name, Path(path)


def _load_arm(
    path: Path, authority: ProjectiveExactFaceSeamAuthority
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        required = (
            "chart_names",
            "chart_vertex_offsets",
            "chart_face_offsets",
            "faces",
            "vertices_world",
        )
        if any(name not in data.files for name in required):
            raise ValueError(f"{path}: explicit atlas array inventory is incomplete")
        if not np.array_equal(data["chart_names"].astype(str), authority.chart_names.astype(str)):
            raise ValueError(f"{path}: chart order differs from projective authority")
        if not np.array_equal(data["chart_vertex_offsets"], authority.chart_vertex_offsets) or not np.array_equal(data["chart_face_offsets"], authority.chart_face_offsets) or not np.array_equal(data["faces"], authority.faces):
            raise ValueError(f"{path}: exact topology differs from projective authority")
        return np.asarray(data["vertices_world"], np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_arm, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite projective seam report")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    arms = {name: _load_arm(path, authority) for name, path in args.arm}
    if len(arms) != len(args.arm):
        raise ValueError("projective seam arm names are duplicated")
    report = evaluate_projective_exact_face_seam_geometry(authority, arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_sha256": report["content_sha256"],
                "formal_arm_gate_eligible": report["formal_arm_gate_eligible"],
                "m0_formal_decision": report["m0_source_reference"][
                    "formal_decision"
                ],
                "arm_decisions": {
                    name: row["formal_decision"]
                    for name, row in report["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
