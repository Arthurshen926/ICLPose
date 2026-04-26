"""Canonical FeatureGaussian training entrypoint."""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("feature_gaussian.train_impl", run_name="__main__")


if __name__ == "__main__":
    main()
