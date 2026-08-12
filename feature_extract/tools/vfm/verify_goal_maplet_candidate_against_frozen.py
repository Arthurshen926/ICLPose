"""Verify deployment candidate reports match the train-OOF frozen recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    RUN_SOURCE_IDENTITY_KEYS,
    _candidate_configuration,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--candidate_reports", nargs="+", required=True)
    args = parser.parse_args(argv)
    frozen = json.loads(Path(args.frozen).read_text())
    expected = frozen["candidate_configuration"]
    for value in args.candidate_reports:
        report = json.loads(Path(value).read_text())
        actual = _candidate_configuration(report)
        if actual != expected:
            raise ValueError(f"deployment candidate configuration drift: {value}")
        manifest = report.get("run_manifest", {})
        source_identity = {
            key: manifest.get(key) for key in RUN_SOURCE_IDENTITY_KEYS
        }
        if source_identity != frozen.get("implementation_source_identity"):
            raise ValueError(f"deployment candidate implementation drift: {value}")
    print(json.dumps({
        "artifact_type": "goal_maplet_frozen_candidate_verification_v1",
        "verified_report_count": len(args.candidate_reports), "verified": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
