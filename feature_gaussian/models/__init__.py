"""FeatureGaussian model implementations."""

from __future__ import annotations


__all__ = [
    "GaussianFeatureModel",
    "MultiScaleGaussianModel",
    "RawCombinedGaussianModel",
    "RawScaleGaussianModel",
]


def __getattr__(name: str):
    if name == "GaussianFeatureModel":
        from feature_gaussian.models.gaussian_feature_model import GaussianFeatureModel

        return GaussianFeatureModel
    if name == "MultiScaleGaussianModel":
        from feature_gaussian.models.multiscale_gaussian_model import MultiScaleGaussianModel

        return MultiScaleGaussianModel
    if name in {"RawCombinedGaussianModel", "RawScaleGaussianModel"}:
        from feature_gaussian.models.raw_gaussian_model import (
            RawCombinedGaussianModel,
            RawScaleGaussianModel,
        )

        return {
            "RawCombinedGaussianModel": RawCombinedGaussianModel,
            "RawScaleGaussianModel": RawScaleGaussianModel,
        }[name]
    raise AttributeError(f"module 'feature_gaussian.models' has no attribute {name!r}")
