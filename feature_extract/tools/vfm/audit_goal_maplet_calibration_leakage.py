"""Invalidate retrieval descendants of a protected-query calibration fit.

This audit is intentionally post-hoc and fail-closed.  It starts from the
content/file/path identity of one validity-calibration artifact, discovers
JSON consumers, and then follows cryptographic/path bindings transitively.
The output does not delete historical artifacts; it makes their admissible
claim scope machine-readable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--scan_root", action="append", required=True)
    parser.add_argument("--protected_query_route", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _json_without_duplicates(path: Path) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _trajectory(image_id: str) -> str:
    value = str(image_id)
    return value.split("/", 1)[0] if "/" in value else ""


def _node_identifiers(path: Path, payload: dict[str, object]) -> set[str]:
    resolved = path.resolve()
    identifiers = {
        str(path),
        str(resolved),
        resolved.name,
        file_sha256(resolved),
    }
    content = payload.get("content_sha256")
    if isinstance(content, str) and len(content) == 64:
        identifiers.add(content)
    return identifiers


def _classify(payload: dict[str, object]) -> str:
    value = str(payload.get("artifact_type", "unknown_json_artifact"))
    if value == "goal_maplet_pure_radio_retrieval_run_v1":
        return "retrieval_run"
    if "candidate_pool" in value:
        return "pose_free_candidate_pool"
    if "candidate_dataset" in value or "candidate_labels" in value:
        return "candidate_label_dataset"
    if "coverage" in value or "evaluation" in value:
        return "evaluation"
    if "surface_mapper" in value or "ranker" in value:
        return "trained_or_evaluated_model"
    if "calibration" in value:
        return "calibration"
    return "other_transitive_consumer"


def _owned_materializations(
    path: Path, payload: dict[str, object]
) -> list[dict[str, object]]:
    """Return outputs owned by a contaminated JSON, not its input bindings."""

    candidates: dict[str, dict[str, object]] = {}

    def add(candidate: Path, *, source: str, expected_sha256: str = "") -> None:
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_file() or resolved == path.resolve():
            return
        key = str(resolved)
        row = candidates.setdefault(key, {"path": key, "ownership_evidence": []})
        evidence = row["ownership_evidence"]
        assert isinstance(evidence, list)
        if source not in evidence:
            evidence.append(source)
        if expected_sha256:
            row["declared_file_sha256"] = expected_sha256

    # Retrieval summaries are manifests: every row artifact is an output of
    # that run and therefore inherits its invalid claim scope.
    if payload.get("artifact_type") == "goal_maplet_pure_radio_retrieval_run_v1":
        for row in payload.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("artifact"), str):
                add(
                    Path(str(row["artifact"])),
                    source="retrieval_manifest_row",
                    expected_sha256=str(row.get("artifact_sha256", "")),
                )

    # Builders consistently expose their materialized outputs under output_*.
    for key, value in payload.items():
        if key.startswith("output_") and isinstance(value, str):
            add(Path(value), source=f"json_field:{key}")

    # Paired JSON/NPZ, JSON/NPY and JSON/PT artifacts are common in this tree.
    for suffix in (".npz", ".npy", ".pt"):
        add(path.with_suffix(suffix), source="same_stem_companion")

    for row in candidates.values():
        candidate = Path(str(row["path"]))
        size = candidate.stat().st_size
        row["size_bytes"] = int(size)
        # Query shards and label files are small enough to hash.  Huge
        # historical tensors remain bound by path/size and their owner JSON;
        # hashing gigabytes is deliberately not required for invalidation.
        if size <= 128 * 1024 * 1024:
            row["file_sha256"] = file_sha256(candidate)
            expected = str(row.get("declared_file_sha256", ""))
            if expected:
                row["declared_hash_matches"] = expected == row["file_sha256"]
    return [candidates[key] for key in sorted(candidates)]


def build_invalidation_report(
    *,
    calibration_path: Path,
    scan_roots: Iterable[Path],
    protected_query_routes: Iterable[str],
    output_path: Path | None = None,
) -> dict[str, object]:
    calibration_path = Path(calibration_path).resolve()
    calibration = _json_without_duplicates(calibration_path)
    metadata = calibration.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("calibration metadata is not an object")
    fit_routes = sorted({str(value) for value in metadata.get("fit_trajectory_ids", [])})
    protected = sorted({str(value) for value in protected_query_routes})
    overlap = sorted(set(fit_routes) & set(protected))
    if not fit_routes or not overlap:
        raise ValueError(
            "calibration does not demonstrate protected-query route overlap"
        )
    calibration_content = str(calibration.get("content_sha256", ""))
    if len(calibration_content) != 64:
        raise ValueError("calibration content_sha256 is missing")

    paths: list[Path] = []
    roots = [Path(value).resolve() for value in scan_roots]
    for root in roots:
        paths.extend(path.resolve() for path in root.rglob("*.json"))
    excluded = Path(output_path).resolve() if output_path is not None else None
    paths = sorted(set(path for path in paths if path != excluded))
    payloads = {path: _json_without_duplicates(path) for path in paths}
    identifiers = {
        path: _node_identifiers(path, payload) for path, payload in payloads.items()
    }
    strings = {path: set(_strings(payload)) for path, payload in payloads.items()}

    source_identifiers = _node_identifiers(calibration_path, calibration)
    affected: dict[Path, dict[str, object]] = {}
    frontier: dict[Path, set[str]] = {}
    for path in paths:
        if path == calibration_path:
            continue
        matched = strings[path] & source_identifiers
        if matched:
            affected[path] = {
                "dependency_depth": 1,
                "upstream_paths": [str(calibration_path)],
                "matched_identifiers": sorted(matched),
            }
            frontier[path] = identifiers[path]

    depth = 1
    while frontier:
        union: dict[str, list[str]] = {}
        for upstream, values in frontier.items():
            for value in values:
                union.setdefault(value, []).append(str(upstream))
        next_frontier: dict[Path, set[str]] = {}
        for path in paths:
            if path == calibration_path or path in affected:
                continue
            matched = strings[path] & set(union)
            if not matched:
                continue
            upstream = sorted(
                {source for value in matched for source in union[value]}
            )
            affected[path] = {
                "dependency_depth": depth + 1,
                "upstream_paths": upstream,
                "matched_identifiers": sorted(matched),
            }
            next_frontier[path] = identifiers[path]
        frontier = next_frontier
        depth += 1

    rows: list[dict[str, object]] = []
    materialized: dict[str, dict[str, object]] = {}
    direct_retrieval_query_routes: set[str] = set()
    for path in sorted(affected):
        payload = payloads[path]
        record = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "content_sha256": str(payload.get("content_sha256", "")),
            "artifact_type": str(payload.get("artifact_type", "unknown")),
            "category": _classify(payload),
            "historical_only": True,
            "production_eligible": False,
            **affected[path],
        }
        if payload.get("artifact_type") == "goal_maplet_pure_radio_retrieval_run_v1":
            query_routes = sorted(
                {
                    _trajectory(str(row.get("image_id", "")))
                    for row in payload.get("rows", [])
                    if isinstance(row, dict)
                }
                - {""}
            )
            record["query_routes"] = query_routes
            record["query_count"] = int(payload.get("query_count", 0))
            direct_retrieval_query_routes.update(query_routes)
        rows.append(record)
        for owned in _owned_materializations(path, payload):
            owned_path = str(owned["path"])
            previous = materialized.setdefault(owned_path, owned)
            owners = previous.setdefault("invalid_owner_jsons", [])
            assert isinstance(owners, list)
            owners.append(str(path))
            previous["historical_only"] = True
            previous["production_eligible"] = False

    if not any(row["category"] == "retrieval_run" for row in rows):
        raise ValueError("no retrieval summary consumes the leaky calibration")

    category_counts = Counter(str(row["category"]) for row in rows)
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_calibration_leak_invalidation_audit_v1",
        "status": "invalid_protected_query_calibration_overlap",
        "passed": False,
        "historical_only": True,
        "production_eligible": False,
        "authority": (
            "must_not_be_cited_as_strict_retrieval_candidate_coverage_or_pose_"
            "estimation_evidence"
        ),
        "leaky_calibration": {
            "path": str(calibration_path),
            "file_sha256": file_sha256(calibration_path),
            "content_sha256": calibration_content,
            "fit_trajectory_ids": fit_routes,
            "fit_image_count": len(metadata.get("fit_image_ids", [])),
            "protected_query_routes": protected,
            "overlapping_routes": overlap,
        },
        "scan": {
            "roots": [str(value) for value in roots],
            "json_file_count": len(paths),
            "closure_semantics": (
                "forward_path_file_sha256_or_content_sha256_dependency_closure_v1"
            ),
            "closure_is_conservative": True,
            "unbound_artifacts_may_require_manual_invalidation": True,
        },
        "direct_retrieval_query_routes": sorted(direct_retrieval_query_routes),
        "affected_json_count": len(rows),
        "affected_json_category_counts": dict(sorted(category_counts.items())),
        "affected_json_artifacts": rows,
        "owned_materialized_artifact_count": len(materialized),
        "owned_materialized_artifacts": [
            materialized[key] for key in sorted(materialized)
        ],
        "replacement_protocol": {
            "validity_calibration_fit_routes": ["seq10"],
            "strict_query_routes": ["seq12", "seq14"],
            "required_disjointness": [
                "calibration_routes_disjoint_from_mapper_fit_routes",
                "calibration_routes_disjoint_from_mapper_validation_routes",
                "calibration_routes_disjoint_from_canonical_mapping_routes",
                "query_routes_disjoint_from_calibration_routes",
            ],
            "cli_default": "reject_overlap_without_explicit_historical_control_override",
        },
        "failure_reasons": [
            "seq12_protected_query_route_was_used_to_fit_validity_calibration",
            "seq12_and_seq14_retrieval_runs_consumed_that_calibration",
            "candidate_pool_pose_label_and_pose_backend_descendants_inherit_the_leak",
        ],
        "files_deleted": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite calibration invalidation audit")
    report = build_invalidation_report(
        calibration_path=Path(args.calibration),
        scan_roots=[Path(value) for value in args.scan_root],
        protected_query_routes=args.protected_query_route,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "content_sha256": report["content_sha256"],
                "affected_json_count": report["affected_json_count"],
                "owned_materialized_artifact_count": report[
                    "owned_materialized_artifact_count"
                ],
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
