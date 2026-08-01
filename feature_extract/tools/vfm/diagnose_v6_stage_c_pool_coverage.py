"""Attach pose-error diagnostics to a frozen V6 Stage-C candidate pool.

The runtime sparse screen and exact/local ranking are never changed by this
tool.  It deterministically rebuilds Stage B from the serialized query-side
RADIO frame sufficient statistics, then uses ground truth only to measure
coverage of the already-frozen complete, sparse-screened and refined pools.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _predicted_pose_hypotheses,
)
from feature_extract.tools.vfm.replay_v6_stage_c import (
    _frame_matches_from_query,
    _query_row,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    factorized_pose_distribution_modes,
    pose_distribution_consensus_modes,
)
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


TRANSLATION_RADIUS_M = 0.85
ROTATION_RADIUS_DEG = 5.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_report", required=True)
    parser.add_argument("--stage_c_report", required=True)
    parser.add_argument("--radio_atlas", required=True)
    parser.add_argument("--query_contributor_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--regenerated_pose_hypotheses", type=int, default=4096)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error_row(hypothesis: object, target_pose: np.ndarray) -> dict[str, object]:
    error = pnp_pose_error(hypothesis.pose_w2c, target_pose)
    return {
        "translation_m": float(error.translation_m),
        "rotation_deg": float(error.rotation_deg),
        "source_chart_ids": [
            int(value) for value in hypothesis.source_chart_ids
        ],
        "control_model": str(hypothesis.control_model),
        "seed_model": str(hypothesis.seed_model),
        "coarse_score": float(hypothesis.score),
    }


def _oracle(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    return dict(
        min(
            rows,
            key=lambda row: (
                float(row["translation_m"]) / 0.30
                + float(row["rotation_deg"]) / 3.0
            ),
        )
    )


def _coverage(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "count": len(rows),
        "oracle": _oracle(rows),
        "recall_20cm_3deg": bool(
            any(
                float(row["translation_m"]) <= 0.20
                and float(row["rotation_deg"]) <= 3.0
                for row in rows
            )
        ),
        "recall_30cm_3deg": bool(
            any(
                float(row["translation_m"]) <= 0.30
                and float(row["rotation_deg"]) <= 3.0
                for row in rows
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_path = Path(args.source_report)
    stage_c_path = Path(args.stage_c_report)
    atlas_path = Path(args.radio_atlas)
    source = json.loads(source_path.read_text())
    stage_c = json.loads(stage_c_path.read_text())
    query = _query_row(source, str(stage_c["image_id"]), 0)
    image_id = str(query["image_id"])
    expected_source_hashes = set(stage_c.get("source_report_sha256s", ()))
    if expected_source_hashes and _sha256(source_path) not in expected_source_hashes:
        raise ValueError("Stage-C report does not reference the source report")

    atlas = MapletFeatureAtlasBank.load_npz(atlas_path)
    views = _load_views(
        Path(args.query_contributor_dir),
        atlas,
        Path(args.image_root),
        trajectory_ids=(image_id.split("/", 1)[0],),
    )
    matches = [value for value in views if value.image_id == image_id]
    if len(matches) != 1:
        raise ValueError(f"expected one contributor view for {image_id!r}")
    view = matches[0]
    frame_matches = _frame_matches_from_query(query)
    generated = _predicted_pose_hypotheses(
        frame_matches,
        atlas,
        view,
        maximum_charts=len(frame_matches),
        maximum_pose_hypotheses=int(args.regenerated_pose_hypotheses),
    )
    raw = [
        replace(value, seed_model=f"proposal_source_0:{value.seed_model}")
        for value in generated
    ]
    consensus = list(
        pose_distribution_consensus_modes(
            raw,
            translation_radius_m=TRANSLATION_RADIUS_M,
            rotation_radius_deg=ROTATION_RADIUS_DEG,
        )
    )
    factorized = list(factorized_pose_distribution_modes(raw, consensus))
    pool = [
        *[
            value for value in raw if len(set(value.source_chart_ids)) >= 2
        ],
        *[
            replace(
                value,
                seed_model=f"proposal_source_0:{value.seed_model}",
            )
            for value in consensus[:32]
        ],
        *[
            replace(
                value,
                seed_model=f"proposal_source_0:{value.seed_model}",
            )
            for value in factorized[:32]
        ],
    ]
    if len(pool) != int(stage_c["pool_size_reconstructed"]):
        raise ValueError("reconstructed pool size differs from Stage-C report")
    errors = [_error_row(value, view.pose_w2c) for value in pool]
    broad_indices = [
        int(value)
        for value in stage_c.get("stage_c_audit", {}).get(
            "broad_screen_selected_source_ranks", ()
        )
    ]
    refined_indices = [
        int(value) for value in stage_c.get("refined_pool_indices", ())
    ]
    broad_rows = [
        {"pool_index": index, **errors[index]} for index in broad_indices
    ]
    refined_rows = [
        {"pool_index": index, **errors[index]} for index in refined_indices
    ]
    payload = {
        "stage": "v6_stage_c_frozen_pool_coverage_diagnostic",
        "deployable_result": False,
        "ground_truth_used_for_selection": False,
        "image_id": image_id,
        "source_report": str(source_path),
        "source_report_sha256": _sha256(source_path),
        "stage_c_report": str(stage_c_path),
        "stage_c_report_sha256": _sha256(stage_c_path),
        "radio_atlas": str(atlas_path),
        "radio_atlas_sha256": _sha256(atlas_path),
        "complete_pool": _coverage(errors),
        "broad_screen_pool": _coverage(broad_rows),
        "refined_pool_initial_poses": _coverage(refined_rows),
        "broad_screen_rows": broad_rows,
        "refined_pool_rows": refined_rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "complete_pool": payload["complete_pool"],
                "broad_screen_pool": payload["broad_screen_pool"],
                "refined_pool_initial_poses": payload[
                    "refined_pool_initial_poses"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
