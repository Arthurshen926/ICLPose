"""Build a target-only rerun manifest for lifted-LoFTR tail attribution.

This tool is deliberately downstream of frozen target-free scores.  It is
allowed to identify a selected bad pose and its best frozen oracle pose, but
it never computes a visual score, changes a score, or produces an artifact
that can be consumed by the formal P1 audit.  The manifest feeds only the
explicit diagnostic subset options of the scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_lifted_loftr_map_to_query_pose_evidence import (
    _STATISTIC_FIELDS,
    _load_score,
    _validate_target_lineage,
)
from feature_extract.tools.vfm.audit_v5_dynamic_absolute_context_pose_evidence import (
    _load_targets,
    _row_keys,
)
from feature_extract.vfm.artifacts import file_sha256_short


MANIFEST_FORMAT = "frozen_lifted_loftr_tail_diagnostic_manifest_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-score-artifacts", required=True)
    parser.add_argument("--target-artifact", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument(
        "--score-statistic",
        choices=tuple(_STATISTIC_FIELDS),
        default="mean",
    )
    parser.add_argument("--tail-threshold-m", type=float, default=1.0)
    parser.add_argument("--oracle-threshold-m", type=float, default=0.10)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("tail manifest score paths must be non-empty and unique")
    return paths


def _rank_descending(scores: np.ndarray, hypothesis_indices: np.ndarray) -> np.ndarray:
    """Deterministic descending ranks using immutable hypothesis indices for ties."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    hypotheses = np.asarray(hypothesis_indices, dtype=np.int64).reshape(-1)
    if (
        len(values) == 0
        or values.shape != hypotheses.shape
        or not np.isfinite(values).all()
        or len(np.unique(hypotheses)) != len(hypotheses)
    ):
        raise ValueError("tail diagnostic score rows are malformed")
    order = np.lexsort((hypotheses, -values))
    ranks = np.empty((len(order),), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return ranks


def _source_inputs(metadata: Mapping[str, object]) -> dict[str, dict[str, str]]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("score artifact lacks its target-free input manifest")
    output: dict[str, dict[str, str]] = {}
    for name, item in inputs.items():
        if not isinstance(item, Mapping):
            raise ValueError("score artifact input manifest is malformed")
        path = str(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        if not path or not digest:
            raise ValueError("score artifact input manifest is incomplete")
        output[str(name)] = {"path": path, "sha256": digest}
    return output


def build_tail_manifest(
    *,
    visual_score_artifacts: Sequence[Path],
    target_artifact: Path,
    profile_name: str,
    score_statistic: str,
    tail_threshold_m: float,
    oracle_threshold_m: float,
    max_queries: int,
) -> dict[str, Any]:
    """Select frozen tail/oracle pairs after target-free scoring has completed."""

    if (
        str(score_statistic) not in _STATISTIC_FIELDS
        or not str(profile_name)
        or float(tail_threshold_m) <= 0.0
        or float(oracle_threshold_m) <= 0.0
        or int(max_queries) <= 0
    ):
        raise ValueError("tail manifest configuration is invalid")
    target, target_metadata = _load_targets(Path(target_artifact))
    target_keys = _row_keys(target)
    target_positions = {key: row for row, key in enumerate(target_keys)}
    if len(target_positions) != len(target_keys):
        raise ValueError("target artifact repeats an immutable hypothesis key")
    loaded: list[tuple[Path, dict[str, np.ndarray], dict[str, object]]] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    for path in visual_score_artifacts:
        arrays, metadata = _load_score(Path(path), expected_variant="visual")
        _validate_target_lineage(score_metadata=[metadata], target_metadata=target_metadata)
        names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
        if str(profile_name) not in names.tolist():
            raise ValueError(f"{path}: requested profile is absent")
        keys = _row_keys(arrays)
        duplicate = next((key for key in keys if key in seen_keys), None)
        if duplicate is not None:
            raise ValueError(f"visual scores repeat frozen hypothesis key {duplicate!r}")
        if any(key not in target_positions for key in keys):
            raise ValueError(f"{path}: target artifact does not cover all frozen rows")
        seen_keys.update(keys)
        loaded.append((Path(path), arrays, metadata))
    if not loaded:
        raise ValueError("tail manifest needs at least one visual score artifact")

    selected_rows: list[dict[str, Any]] = []
    field = _STATISTIC_FIELDS[str(score_statistic)]
    for path, arrays, metadata in loaded:
        names = np.asarray(arrays["profile_names"]).astype(str).reshape(-1)
        profile_index = int(np.flatnonzero(names == str(profile_name))[0])
        keys = _row_keys(arrays)
        groups: dict[tuple[str, str, str], list[int]] = {}
        for row, key in enumerate(keys):
            groups.setdefault(key[:3], []).append(row)
        for (split_name, evaluation_label, query_id), group_rows in sorted(groups.items()):
            rows = np.asarray(group_rows, dtype=np.int64)
            hypotheses = np.asarray(arrays["hypothesis_indices"], dtype=np.int64)[rows]
            scores = np.asarray(arrays[field], dtype=np.float64)[rows, profile_index]
            target_rows = np.asarray([target_positions[keys[int(row)]] for row in rows])
            translation = np.asarray(target["translation_errors_m"], dtype=np.float64)[target_rows]
            rotation = np.asarray(target["rotation_errors_deg"], dtype=np.float64)[target_rows]
            ranks = _rank_descending(scores, hypotheses)
            selected_local = int(np.lexsort((hypotheses, -scores))[0])
            oracle_local = int(np.lexsort((hypotheses, rotation, translation))[0])
            selected_error = float(translation[selected_local])
            oracle_error = float(translation[oracle_local])
            if (
                selected_error < float(tail_threshold_m)
                or oracle_error > float(oracle_threshold_m)
            ):
                continue
            requested = [int(hypotheses[selected_local])]
            if int(hypotheses[oracle_local]) != requested[0]:
                requested.append(int(hypotheses[oracle_local]))
            selected_rows.append(
                {
                    "query_id": str(query_id),
                    "split_name": str(split_name),
                    "evaluation_label": str(evaluation_label),
                    "profile_name": str(profile_name),
                    "score_statistic": str(score_statistic),
                    "source_visual_score_artifact": {
                        "path": str(path),
                        "sha256": file_sha256_short(path),
                    },
                    "source_score_inputs_target_free": _source_inputs(metadata),
                    "selected_hypothesis_index_TARGET_ONLY": int(hypotheses[selected_local]),
                    "selected_translation_error_m_TARGET_ONLY": selected_error,
                    "selected_rotation_error_deg_TARGET_ONLY": float(rotation[selected_local]),
                    "selected_score_target_free": float(scores[selected_local]),
                    "oracle_hypothesis_index_TARGET_ONLY": int(hypotheses[oracle_local]),
                    "oracle_translation_error_m_TARGET_ONLY": oracle_error,
                    "oracle_rotation_error_deg_TARGET_ONLY": float(rotation[oracle_local]),
                    "oracle_score_target_free": float(scores[oracle_local]),
                    "oracle_score_rank_TARGET_ONLY": int(ranks[oracle_local]),
                    "selected_minus_oracle_score_target_free": float(
                        scores[selected_local] - scores[oracle_local]
                    ),
                    "diagnostic_hypothesis_indices": requested,
                }
            )
    selected_rows.sort(
        key=lambda item: (
            -float(item["selected_translation_error_m_TARGET_ONLY"]),
            str(item["query_id"]),
        )
    )
    selected_rows = selected_rows[: int(max_queries)]
    return {
        "format": MANIFEST_FORMAT,
        "stage": "target_only_frozen_lifted_loftr_tail_manifest",
        "target_only": True,
        "must_not_be_used_for_inference_or_formal_p1_audit": True,
        "score_inputs_were_target_free_before_this_join": True,
        "profile_name": str(profile_name),
        "score_statistic": str(score_statistic),
        "thresholds": {
            "tail_threshold_m": float(tail_threshold_m),
            "oracle_threshold_m": float(oracle_threshold_m),
        },
        "source_target_artifact": {
            "path": str(target_artifact),
            "sha256": file_sha256_short(Path(target_artifact)),
        },
        "source_visual_score_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in visual_score_artifacts
        ],
        "tail_pairs_TARGET_ONLY": selected_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite tail manifest: {output}")
    manifest = build_tail_manifest(
        visual_score_artifacts=_paths(args.visual_score_artifacts),
        target_artifact=Path(args.target_artifact),
        profile_name=str(args.profile_name),
        score_statistic=str(args.score_statistic),
        tail_threshold_m=float(args.tail_threshold_m),
        oracle_threshold_m=float(args.oracle_threshold_m),
        max_queries=int(args.max_queries),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "tail_pair_count": len(manifest["tail_pairs_TARGET_ONLY"]),
                "profile_name": manifest["profile_name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
