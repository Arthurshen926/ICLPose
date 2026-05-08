"""Canonical FeatureRetrieval index-building entrypoint."""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("feature_retrieval.retrievers.build_index", run_name="__main__")


if __name__ == "__main__":
    main()
