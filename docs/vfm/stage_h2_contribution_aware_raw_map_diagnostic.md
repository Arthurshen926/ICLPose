# Stage H2 Contribution-Aware Raw Gaussian Map Diagnostic

## Scope

This diagnostic tests contribution-aware Gaussian sampling and raw VFM aggregation without selector training.

Implemented:

- approximate token-grid Gaussian contribution maps;
- contribution-based saliency voting;
- contribution top-token raw feature aggregation;
- contribution visibility-gated center sampling;
- q32 no-selector evaluation against existing owner-visible baselines.

Code:

- `feature_extract/vfm/gaussian_raw_landmarks.py`
- `feature_extract/tools/vfm/build_stage_h2_gaussian_token_contribution_maps.py`
- `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py`

## Contribution Map Export

Artifact:

`output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/contribution_maps_v96_r15/`

Configuration:

- OldHospital train reference views: 96 uniform
- Gaussian source: 504,352 2DGS Gaussians
- token contribution radius: `1.5 px`
- depth epsilon: `0.05`
- opacity threshold: `0.05`
- view-angle weight: disabled

Diagnostics:

| metric | value |
|---|---:|
| mean valid token fraction | 0.990 |
| mean top-alpha | 0.603 |
| mean alpha entropy | 0.595 |

So the approximate contribution maps are dense, but not necessarily correct for localization.

## Raw Map Variants

All variants use raw RADIO 1280D and q32 OldHospital GT-visible evaluation.

| Variant | Sampling | Aggregation | Anchors |
|---|---|---|---:|
| owner v96 baseline | owner-projection vote | owner-visible center sample | existing |
| owner vote + token contribution agg | owner-projection vote | top-contributor token average | 10,279 |
| contribution vote + token contribution agg, alpha0 | contribution vote | top-contributor token average | 14,872 |
| contribution vote + token contribution agg, alpha0.25 | contribution vote | top-contributor token average | 11,489 |
| owner vote + contribution visibility center sample | owner-projection vote | contribution-visible center sample | 2,604 |
| contribution vote + contribution visibility center sample | contribution vote | contribution-visible center sample | 4,314 |

## q32 No-Selector Results

| Variant | S@25 | S@50 | median t | PnP-inlier Patch@1 | inliers | nonempty patch frac | visible landmarks |
|---|---:|---:|---:|---:|---:|---:|---:|
| owner v96 baseline | 0.219 | 0.781 | 0.371m | 0.369 | 114.2 | 0.388 | 13,024 |
| owner vote + token contribution agg | 0.156 | 0.500 | 0.544m | 0.346 | 64.9 | 0.313 | 7,848 |
| contribution vote + token contribution agg, alpha0 | 0.062 | 0.406 | 0.557m | 0.333 | 68.6 | 0.360 | 11,086 |
| contribution vote + token contribution agg, alpha0.25 | 0.031 | 0.406 | 0.524m | 0.343 | 64.5 | 0.334 | 8,023 |
| owner vote + contribution visibility center sample | 0.156 | 0.500 | 0.496m | 0.308 | 32.2 | 0.129 | 1,838 |
| contribution vote + contribution visibility center sample | 0.125 | 0.531 | 0.453m | 0.350 | 43.5 | 0.182 | 2,868 |
| all-view owner current-code baseline | 0.469 | 0.938 | 0.263m | 0.419 | 118.5 | 0.360 | 14,085 |

## Interpretation

The first contribution-aware implementation is not a positive replacement for owner-visible aggregation.

Evidence:

- top-contributor token averaging reduces PnP inlier count and precision;
- contribution visibility center sampling is too strict and collapses visible landmark coverage;
- contribution voting alone does not rescue localization;
- alpha gating further reduces recall.

The likely root cause is that this approximate token contribution map is still not the actual 3DGS/2DGS raster contribution used by ULF-Loc. It binds a VFM patch token to a single dominant Gaussian on a coarse token grid, while VFM tokens are patch-level descriptors and the trained Gaussian surface is much denser than the token grid.

## Current Conclusion

Do not promote this approximate contribution-aware path as the main map.

The useful implementation outcome is diagnostic:

1. the raw Gaussian source now exposes 2DGS geometry;
2. contribution maps can be exported and consumed;
3. approximate token top-contributor aggregation is empirically worse than owner-visible center projection;
4. the next correct direction is either exact renderer contribution or soft/multi-contributor support, not hard top-contributor token ownership.

## Next Step

The next map-side experiment should use one of:

- exact 2DGS/3DGS rasterizer contribution maps if available from the training code;
- soft multi-contributor aggregation where each token distributes feature to several high-contribution Gaussians;
- Gaussian-local support aggregation around reliable surfels rather than single top contributor per token.

Until that exists, the strongest raw Gaussian baseline remains the all-view owner-visible center-sampled map.
