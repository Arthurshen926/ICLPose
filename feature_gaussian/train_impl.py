"""FeatureGaussian training compatibility entrypoint.

The 2DGS feature-training implementation has not been fully migrated yet, but
the canonical entrypoint now stays under the FeatureGaussian namespace.
"""

from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("feature_gaussian.legacy_3dgs.train_2dgs_joint_v3", run_name="__main__")


if __name__ == "__main__":
    main()