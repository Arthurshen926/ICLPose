"""Re-select frozen independent pose scores with a saved robust statistic.

This is an inference-only artifact transform.  It never loads a COLMAP pose,
target residual, image feature, or candidate artifact: all candidate evidence
and all hypotheses were frozen by the source score artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_independent_landmark_pose_scores import (
    _merge_score_artifacts,
    _paths,
)
from feature_extract.tools.vfm.score_independent_landmark_pose_hypotheses import (
    SCORE_ARTIFACT_FORMAT,
    SELECTION_STATISTICS,
)
from feature_extract.vfm.artifacts import file_sha256_short


PROFILE_FORMAT = "independent_landmark_pose_score_profile_v1"


def _selection_source_field(
    metadata: Mapping[str, object], statistic: str, score_fields: set[str]
) -> str:
    selection = metadata.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("source score artifact lacks a profile selection manifest")
    mappings = selection.get("statistic_score_fields")
    if not isinstance(mappings, Mapping):
        raise ValueError("source score artifact lacks statistic score fields")
    source_field = mappings.get(str(statistic))
    if not isinstance(source_field, str) or source_field not in score_fields:
        raise ValueError(
            f"source score artifact cannot select statistic {statistic!r}"
        )
    return source_field


def reselect_score_profile(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    *,
    statistic: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return the same frozen score rows with a deterministic new top-1 flag."""

    if str(statistic) not in SELECTION_STATISTICS:
        raise ValueError(f"unsupported selection statistic: {statistic}")
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "independent_score_top1",
        "independent_selection_scores",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"source score arrays lack {missing}")
    source_field = _selection_source_field(metadata, str(statistic), set(arrays))
    values = np.asarray(arrays[source_field], dtype=np.float64).reshape(-1)
    count = len(values)
    if np.any(~np.isfinite(values)):
        raise ValueError("selection statistic contains non-finite values")
    key_arrays = (
        np.asarray(arrays["split_names"]).astype(str),
        np.asarray(arrays["evaluation_labels"]).astype(str),
        np.asarray(arrays["query_ids"]).astype(str),
    )
    if any(len(value) != count for value in key_arrays):
        raise ValueError("score rows are not aligned with selection values")
    selected = np.zeros((count,), dtype=bool)
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row, key in enumerate(zip(*key_arrays)):
        groups.setdefault(tuple(key), []).append(row)
    for rows in groups.values():
        indices = np.asarray(rows, dtype=np.int64)
        # np.argmax intentionally preserves the source artifact's deterministic
        # first-row tie break for identical frozen profile values.
        selected[int(indices[np.argmax(values[indices])])] = True

    output_arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
    output_arrays["independent_selection_scores"] = values.copy()
    output_arrays["independent_score_top1"] = selected
    output_metadata = dict(metadata)
    source_selection = metadata.get("selection")
    assert isinstance(source_selection, Mapping)
    output_metadata["selection"] = {
        **dict(source_selection),
        "statistic": str(statistic),
        "score_field": "independent_selection_scores",
        "source_score_field": str(source_field),
        "tie_break": "first_frozen_merged_score_row_v1",
    }
    output_metadata["row_count"] = int(count)
    output_metadata["query_count"] = int(len(groups))
    output_metadata["profile_transform"] = {
        "format": PROFILE_FORMAT,
        "target_free": True,
        "source_score_rows_frozen": True,
        "candidate_or_hypothesis_regeneration": False,
    }
    return output_arrays, output_metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_artifacts", required=True)
    parser.add_argument("--selection_statistic", choices=SELECTION_STATISTICS, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score_paths = _paths(args.score_artifacts)
    arrays, source_metadata, compatibility_hash = _merge_score_artifacts(score_paths)
    output_arrays, metadata = reselect_score_profile(
        arrays, source_metadata[0], statistic=str(args.selection_statistic)
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata["profile_transform"] = {
        **dict(metadata["profile_transform"]),
        "source_score_artifacts": [str(path) for path in score_paths],
        "source_score_artifact_sha256": [
            file_sha256_short(path) for path in score_paths
        ],
        "source_score_compatibility_sha256": compatibility_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **output_arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "stage": PROFILE_FORMAT,
                "output": str(output),
                "selection": metadata["selection"],
                "row_count": metadata["row_count"],
                "query_count": metadata["query_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
