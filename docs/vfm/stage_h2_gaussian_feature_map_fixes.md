# Stage H2 Gaussian Feature Map Fixes

## What Was Wrong

The earlier H2 Gaussian map was not a proper ULF-Loc-style Gaussian landmark
feature map. Two implementation problems were confirmed:

- Feature aggregation counted any in-image Gaussian projection as an observation.
  It did not require the Gaussian to be the token owner / front visible anchor.
- The exported `SemiDenseAnchorMap` did not record per-anchor observation image
  ids and wrote `feature_variances=0`, so downstream map-quality diagnostics were
  invalid for Gaussian anchors.

The previous map also selected many weak anchors:

- `min_votes=1`
- 19914 exported anchors
- 68.2% of sampled anchors had exactly one vote
- aggregation support count median was 64, but these were projected views, not
  owner-visible observations

## Fixes

- `aggregate_raw_vfm_features_to_gaussian_anchors` now records real support
  image ids for every retained Gaussian anchor.
- It computes feature variance from the normalized multi-view observations.
- H2 builder now defaults to owner-visible aggregation.
- H2 builder records sampled vote statistics in the summary JSON.
- `_project_xyz_to_grid` now honors COLMAP SIMPLE_RADIAL/RADIAL/OPENCV camera
  distortion before scaling to the VFM token grid.
- H2 builder defaults were tightened for new runs:
  - `--max_views 0` means all views unless explicitly capped.
  - `--min_votes 3` by default.
  - token-owner visibility is enabled by default; use
    `--disable_token_owner_visibility` only for ablations.

## OldHospital V96 Diagnostics

All rows are OldHospital first 32 test queries, raw VFM patch-to-Gaussian MNN,
EPNP + LM, `0.75 * stride` RANSAC threshold.

| Map | Submap | Anchors | S@25cm/10deg | S@50cm/10deg | S@1m/10deg | Median t | Inlier Patch@1 | Inlier GT@stride |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old no-owner min1 | all-map | 19914 | 0.219 | 0.625 | 0.813 | 0.377m | 0.324 | 0.703 |
| owner-visible min1 | all-map | 17231 | 0.313 | 0.750 | 1.000 | 0.311m | 0.356 | 0.775 |
| owner-visible min2 | all-map | 6330 | 0.156 | 0.594 | 0.906 | 0.472m | 0.329 | 0.715 |
| owner-visible min3 | all-map | 3126 | 0.094 | 0.625 | 0.906 | 0.466m | 0.296 | 0.692 |
| old no-owner min1 | GT-visible | 19914 | 0.281 | 0.688 | 0.844 | 0.300m | 0.300 | 0.690 |
| owner-visible min1 | GT-visible | 17231 | 0.219 | 0.781 | 1.000 | 0.371m | 0.369 | 0.775 |
| owner-visible min2 | GT-visible | 6330 | 0.156 | 0.781 | 1.000 | 0.409m | 0.378 | 0.803 |
| owner-visible min3 | GT-visible | 3126 | 0.156 | 0.563 | 0.875 | 0.437m | 0.296 | 0.696 |

## Interpretation

The main implementation bug was aggregation visibility, not the Gaussian-anchor
idea itself. Keeping coverage (`min_votes=1`) but requiring owner-visible
observations improves the q32 all-map result from `0.219/0.625/0.813` to
`0.313/0.750/1.000`.

Tightening votes to `min_votes=2/3` improves some inlier-correctness diagnostics
under GT-visible submaps, but it removes too much coverage with only 96 views and
hurts final pose. For OldHospital V96, `owner-visible min1` is the current
default diagnostic map.

Frozen C2.5 descriptors applied to the corrected Gaussian map improve inlier
correctness but do not improve the q32 final pose over raw1280. This is expected:
that C2.5 checkpoint was trained on SfM-track landmarks, not Gaussian anchors.
