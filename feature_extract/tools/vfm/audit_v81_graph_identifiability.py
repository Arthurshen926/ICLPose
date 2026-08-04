"""Audit rooted 1/2/3-hop ambiguity of the physical V8 maplet graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_graph", required=True)
    parser.add_argument("--feature_bank", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _class_statistics(signature: list[str]) -> dict[str, object]:
    count = Counter(signature)
    sizes = np.asarray([count[value] for value in signature], dtype=np.float64)
    histogram = Counter(count.values())
    return {
        "unique_node_fraction": float(np.mean(sizes == 1)),
        "median_equivalence_class_size": float(np.median(sizes)),
        "p90_equivalence_class_size": float(np.quantile(sizes, 0.9)),
        "maximum_equivalence_class_size": int(np.max(sizes)),
        "equivalence_class_count": int(len(count)),
        "class_size_histogram": {
            str(size): int(classes) for size, classes in sorted(histogram.items())
        },
    }


def _rooted_signatures(graph: PhysicalMapletGraph) -> list[list[str]]:
    count = int(graph.maplet_ids.size)
    extent = np.maximum(np.asarray(graph.extents[:, :2], dtype=np.float64), 1e-5)
    normal = np.asarray(graph.normals, dtype=np.float64)
    initial = [
        _digest(
            (
                tuple(np.rint(normal[index] / 0.25).astype(int).tolist()),
                tuple(np.rint(np.log(extent[index]) / 0.35).astype(int).tolist()),
            )
        )
        for index in range(count)
    ]
    adjacency: list[dict[int, tuple[int, ...]]] = [dict() for _ in range(count)]
    for source, target, features in zip(
        graph.edge_source.tolist(),
        graph.edge_target.tolist(),
        np.asarray(graph.edge_features, dtype=np.float64),
    ):
        first, second = min(int(source), int(target)), max(int(source), int(target))
        direction = features[:3] if int(source) == first else -features[:3]
        attribute = (
            *np.rint(direction / 0.25).astype(int).tolist(),
            int(round(float(features[3]) / 0.35)),
            int(round(float(features[4]) / 0.20)),
            int(round(abs(float(features[5])) / 0.40)),
            int(round(float(features[6]))),
        )
        previous = adjacency[first].get(second)
        if previous is None or attribute[-1] < previous[-1]:
            adjacency[first][second] = attribute
            reverse = (-attribute[0], -attribute[1], -attribute[2], *attribute[3:])
            adjacency[second][first] = reverse
    result = [initial]
    signature = initial
    for _hop in range(1, 4):
        updated = []
        for node in range(count):
            neighbourhood = sorted(
                (attribute, signature[other])
                for other, attribute in adjacency[node].items()
            )
            updated.append(_digest((signature[node], neighbourhood)))
        signature = updated
        result.append(signature)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite graph-identifiability audit")
    graph = PhysicalMapletGraph.load_npz(Path(args.physical_graph))
    feature_hash = hashlib.sha256(Path(args.feature_bank).read_bytes()).hexdigest()
    if feature_hash != graph.feature_bank_sha256:
        raise ValueError("physical graph and canonical feature-bank lineage differ")
    signatures = _rooted_signatures(graph)
    long_endpoints = np.unique(
        np.r_[
            graph.edge_source[np.rint(graph.edge_features[:, 6]).astype(int) == 2],
            graph.edge_target[np.rint(graph.edge_features[:, 6]).astype(int) == 2],
        ]
    )
    hops = {}
    for hop in range(4):
        statistics = _class_statistics(signatures[hop])
        counts = Counter(signatures[hop])
        statistics["long_edge_endpoint_unique_fraction"] = (
            float(np.mean([counts[signatures[hop][int(row)]] == 1 for row in long_endpoints]))
            if long_endpoints.size
            else None
        )
        hops[str(hop)] = statistics
    undirected_pairs = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(graph.edge_source, graph.edge_target)
    }
    report = {
        "stage": "v81_physical_graph_rooted_identifiability",
        "physical_graph": str(Path(args.physical_graph).resolve()),
        "canonical_feature_bank_sha256": feature_hash,
        "node_count": int(graph.maplet_ids.size),
        "stored_directed_edge_count": int(graph.edge_source.size),
        "unique_undirected_edge_count": int(len(undirected_pairs)),
        "directed_edge_duplication_factor": float(
            graph.edge_source.size / max(len(undirected_pairs), 1)
        ),
        "rooted_hops": hops,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
