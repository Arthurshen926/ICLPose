# Stage H2 Gaussian Raw Map Representation Diagnostic

## Scope

This pass does not touch selector training or learned descriptor inference. It fixes and validates the raw Gaussian map representation used before feature selection.

Implemented changes:

- `GaussianVFMSource` now preserves optional `scale_xyz`, `rotation`, and `normal`.
- 3-scale 3DGS PLY files use the smallest-scale rotated axis as the normal.
- 2-scale 2DGS PLY files preserve in-plane anisotropy and use the rotated third local axis as the surfel normal.
- Stage H2 raw-map summaries now include sampled/source and final-anchor geometry stats.

Relevant files:

- `feature_extract/vfm/gaussian_vfm_field.py`
- `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py`
- `tests/test_vfm_gaussian_vfm_field.py`
- `tests/test_vfm_gaussian_raw_landmarks.py`

## Key Finding

The OldHospital Gaussian PLY is 2DGS-style:

- fields: `scale_0`, `scale_1`, `rot_0..3`
- `nx/ny/nz` exist but are all zero

Before this fix, the loader only treated 3-scale Gaussians as anisotropic. Therefore the OldHospital Gaussian map was incorrectly diagnosed as:

- anisotropy ratio median: `1.0`
- normal available: `false`

After the 2DGS fix, the same sampled anchors are correctly diagnosed as:

- anisotropy ratio median: `2.712`
- anisotropy ratio p90: `8.066`
- anisotropy ratio p99: `22.170`
- normal available fraction: `1.0`

This means earlier Gaussian-map quality analysis missed a major geometry signal. In particular, view-angle weighting and surfel-orientation filtering were not possible with the previous source representation.

## Raw Map Diagnostic

Artifact:

`output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/geometry_diagnostic_raw1280_v2_2dgs/`

Configuration:

- OldHospital
- raw RADIO `radio_final`, 1280D
- all 895 train reference views
- VFM norm top5 token vote
- owner-visible raw aggregation
- max anchors: 20k
- min votes: 2
- no selector

Map stats:

| metric | value |
|---|---:|
| source Gaussians | 504,352 |
| sampled Gaussians | 20,000 |
| final raw anchors | 19,984 |
| feature dim | 1280 |
| median opacity | 0.514 |
| median mean scale | 0.087m |
| p90 mean scale | 0.387m |
| p99 mean scale | 1.342m |
| median anisotropy | 2.712 |
| p99 anisotropy | 22.170 |

## q32 No-Selector Evaluation

Artifact:

`output/vfm/stage_h2_gaussian_mainline/oldhospital/geometry_diagnostic_raw1280_v2_2dgs/`

Protocol:

- q32 OldHospital test queries
- GT-visible submap
- patch-level MNN top1
- raw1280 Gaussian anchor descriptors
- PnP threshold: `0.75 * stride`
- no selector

Current-code repeat-stable result:

| method | S@25cm/10deg | S@50cm/10deg | median t | inlier Patch@1 | matches | inliers |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian raw1280 v2 | 0.469 | 0.938 | 0.263m | 0.419 | 520.7 | 118.5 |

The raw map arrays are identical to the previous owner-visible min-votes-2 map. Therefore this should not be claimed as a feature-map improvement. The representation fix improves diagnostics and enables geometry-aware sampling/aggregation; it does not by itself change raw features.

## Camera-View Visualizations

Artifact:

`output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/geometry_diagnostic_raw1280_v2_2dgs/camera_view_diagnostics/`

Generated files include:

- `*_h2_gaussian_landmark_overlay.png`
- `*_3dgs_pseudo_rgb.png`
- `gaussian_sampling_distributions.png`

For the first four OldHospital query views:

- visible all Gaussians: about 365k-412k
- visible selected anchors: about 12k-15k
- selected visible ratio: about 3.35%-3.60%
- pseudo visible pixel fraction: about 0.11

The selected anchors are dense enough in camera view, but the current sampler is still not contribution-aware.

## Conclusion

The previous statement "Gaussian worse than SfM" was too coarse. The corrected diagnosis is:

1. The Gaussian centers are geometrically viable under oracle correspondences.
2. The current Gaussian map is still built with owner-projection fallback, not true render contribution.
3. The source representation previously dropped valid 2DGS anisotropy and normals.
4. The current anchor sampler still selects by VFM token saliency and frontmost center owner, which is not equivalent to ULF-Loc render-visible/keypoint-consensus sampling.

Next implementation target:

- implement contribution-aware observation export for the 2DGS/3DGS map;
- use normal/view-angle, alpha contribution, scale, anisotropy, opacity, support, variance, and ambiguity for raw Gaussian landmark sampling;
- compare raw1280 Gaussian vs SfM under the same no-selector q32/full protocol before returning to selector training.
