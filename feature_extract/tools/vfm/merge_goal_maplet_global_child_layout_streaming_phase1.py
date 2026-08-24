"""Exact-validate and merge all hierarchy-layout Phase-1 shard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.global_child_layout_streaming_contract import (
    RUN_SCHEMA, SHARD_RUN_SCHEMA, load_hierarchy_score,
)
from feature_extract.vfm.localization_goal_maplet.global_parent_layout_streaming_contract import (
    EXPECTED_SEQ10_QUERY_COUNT, load_json_no_duplicate_keys,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256, file_sha256,
)


def _load_shard(path: Path) -> dict:
    value = load_json_no_duplicate_keys(path)
    unhashed = dict(value)
    content = unhashed.pop("content_sha256", "")
    if (
        value.get("artifact_type") != SHARD_RUN_SCHEMA
        or content != canonical_json_sha256(unhashed)
        or value.get("query_route") != "seq10"
        or value.get("control_only") is not True
        or value.get("production_eligible") is not False
        or value.get("phase2_labels_opened") is not False
        or value.get("uses_query_pose") is not False
        or value.get("uses_query_ground_truth") is not False
    ):
        raise ValueError("hierarchy Phase-1 shard contract differs")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite hierarchy Phase-1 run")
    shard_paths = [Path(value).resolve() for value in args.shard]
    shards = [_load_shard(path) for path in shard_paths]
    if len(shards) != int(shards[0].get("shard_count", -1)):
        raise ValueError("hierarchy Phase-1 shard set is incomplete")
    if sorted(int(value["shard_index"]) for value in shards) != list(range(len(shards))):
        raise ValueError("hierarchy Phase-1 shard indices differ")
    binding_keys = (
        "proposal_file_sha256", "proposal_content_sha256",
        "retrieval_summary_file_sha256", "physical_map_file_sha256",
        "physical_map_content_sha256", "camera_manifest_file_sha256",
        "gpu_gate_file_sha256", "gpu_gate_content_sha256", "position_count",
        "orientation_count", "total_factor_pair_count_per_query",
        "maximum_query_parents", "maximum_scene_children",
        "returned_topk_per_query", "position_chunk_size", "torch_dtype",
    )
    reference = shards[0]
    if any(
        any(shard.get(key) != reference.get(key) for key in binding_keys)
        for shard in shards[1:]
    ):
        raise ValueError("hierarchy Phase-1 shard bindings differ")
    rows = []
    for shard_path, shard in zip(shard_paths, shards):
        declared = shard.get("rows")
        if not isinstance(declared, list) or len(declared) != int(shard.get("query_count", -1)):
            raise ValueError("hierarchy Phase-1 shard rows differ")
        for row in declared:
            artifact = Path(str(row.get("artifact", ""))).resolve()
            if file_sha256(artifact) != row.get("artifact_file_sha256"):
                raise ValueError("hierarchy score file binding differs")
            arrays, metadata = load_hierarchy_score(artifact)
            if (
                metadata.get("query_index") != row.get("query_index")
                or metadata.get("image_id") != row.get("image_id")
                or metadata.get("content_sha256") != row.get("content_sha256")
            ):
                raise ValueError("hierarchy score row binding differs")
            rows.append(dict(row))
    rows.sort(key=lambda row: int(row["query_index"]))
    if [int(row["query_index"]) for row in rows] != list(range(EXPECTED_SEQ10_QUERY_COUNT)):
        raise ValueError("hierarchy Phase-1 query inventory differs")
    if [str(row["image_id"]) for row in rows] != sorted(str(row["image_id"]) for row in rows):
        raise ValueError("hierarchy Phase-1 image order differs")
    report = {
        "artifact_type": RUN_SCHEMA,
        "query_route": "seq10",
        "query_count": len(rows),
        "rows": rows,
        **{key: reference[key] for key in binding_keys},
        "proposal": reference["proposal"],
        "retrieval_summary": reference["retrieval_summary"],
        "physical_map": reference["physical_map"],
        "camera_manifest": reference["camera_manifest"],
        "gpu_gate": reference["gpu_gate"],
        "shards": [
            {"path": str(path), "file_sha256": file_sha256(path),
             "content_sha256": shard["content_sha256"]}
            for path, shard in zip(shard_paths, shards)
        ],
        "score_before_label_contract": {
            "all_score_files_exact_loaded_before_label_open": True,
            "phase2_labels_opened": False,
        },
        "control_only": True,
        "production_eligible": False,
        "phase2_labels_opened": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_contributor_artifact": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "query_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
