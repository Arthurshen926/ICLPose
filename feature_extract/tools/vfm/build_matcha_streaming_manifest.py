"""Build a thin full-train MATCHA streaming pair manifest.

The output stores only query ids, split names, pair types, candidate ids and
sampling seeds. Dense RADIO/render tensors are intentionally not serialized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _select_records
from feature_extract.tools.vfm.build_matcha_joint_cache import PAIR_TYPE_IDS, _parse_pair_types
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.matcha_streaming_manifest import (
    MatchaStreamingPairManifest,
    build_streaming_pair_records,
)
from feature_extract.vfm.render_pose_protocol import group_top_reference_poses
from feature_extract.vfm.tokens import TokenBankManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--pair_types", default="A_gt")
    parser.add_argument("--candidate_bank", default="")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    pair_types = _parse_pair_types(str(args.pair_types))
    if "D_reference" in pair_types and not str(args.candidate_bank):
        raise ValueError("--candidate_bank is required when pair_types includes D_reference")
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
    pair_types = _parse_pair_types(str(args.pair_types))
    candidate_ids: dict[str, str] = {}
    if "D_reference" in pair_types:
        top1 = group_top_reference_poses(CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)).candidates)
        candidate_ids = {
            str(query_id): str(candidate.candidate_id)
            for query_id, candidate in top1.items()
            if candidate is not None
        }
    query_ids = [str(record.image_id) for record in records]
    pair_records = build_streaming_pair_records(
        query_ids,
        split=str(args.split_name),
        pair_types=pair_types,
        pair_type_ids=PAIR_TYPE_IDS,
        seed=int(args.seed),
        candidate_ids=candidate_ids,
    )
    manifest = MatchaStreamingPairManifest(
        records=pair_records,
        metadata={
            "source_query_manifest": str(args.query_manifest),
            "candidate_bank": str(args.candidate_bank),
            "view_selection": str(args.view_selection),
            "start_index": int(args.start_index),
            "max_queries": int(args.max_queries),
            "seed": int(args.seed),
        },
    )
    manifest.to_json(Path(args.output_manifest))
    summary = manifest.to_dict()
    summary.pop("records", None)
    summary["stage"] = "matcha_streaming_pair_manifest_builder"
    summary["outputs"] = {"manifest": str(args.output_manifest)}
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
