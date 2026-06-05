# Stage H: Gaussian Consensus VFM Landmarks

## Goal

Evaluate whether ULF-Loc-style sparse Gaussian landmarks are a better VFM anchor
set than the current SfM-track landmark map for patch-to-3D matching + PnP.

## ULF-Loc Reference

The local ULF-Loc code in `/root/ULF-Loc` uses:

- a separately trained 3DGS map;
- keypoint-consensus landmark sampling from visible Gaussians projected into
  reference images;
- multi-view Gaussian feature fusion with view-angle/visibility weights;
- sparse 2D keypoint to Gaussian landmark matching followed by PnP.

This project now has a Stage H side path that approximates the first sparse
landmark part without changing the canonical SfM pipeline.

## Implemented

- `GaussianConsensusAnchorConfig` and `build_gaussian_consensus_anchor_map` in
  `feature_extract/vfm/semidense_anchor_map.py`.
- `feature_extract/tools/vfm/build_stage_h_gaussian_consensus_anchor_map.py`
  for quality/support/opacity/scale-based Gaussian landmark sampling.
- `feature_extract/tools/vfm/build_stage_h_keypoint_consensus_anchor_map.py`
  for approximate SIFT keypoint-consensus voting.
- Camera-view visualizations comparing SfM sparse anchors and Gaussian anchors.

## Validation Artifacts

- Structured report:
  `output/vfm/stage_h_gaussian_consensus/stage_h_validation_report.json`
- OldHospital quality map:
  `output/vfm/stage_h_gaussian_consensus/oldhospital/full_v96_top20000/anchor_map.npz`
- OldHospital keypoint-consensus map:
  `output/vfm/stage_h_gaussian_consensus/oldhospital/keypoint_sift80_r4_top20000/anchor_map.npz`
- Camera-view visualizations:
  `output/vfm/stage_h_gaussian_consensus/visualizations/`

## Results

OldHospital q32, all-map, C2.5 128D query descriptors:

| map | S@25cm/10deg | S@50cm/10deg | median t | inlier Patch@1 |
| --- | ---: | ---: | ---: | ---: |
| SfM sparse | 0.844 | 1.000 | 0.171 m | 0.673 |
| Gaussian quality | 0.094 | 0.438 | 0.546 m | 0.444 |
| Gaussian keypoint-consensus | 0.031 | 0.375 | 0.582 m | 0.439 |

OldHospital q32, GT-visible oracle submap:

| map | S@25cm/10deg | S@50cm/10deg | median t | inlier Patch@1 |
| --- | ---: | ---: | ---: | ---: |
| SfM sparse | 0.844 | 1.000 | 0.181 m | 0.679 |
| Gaussian quality | 0.063 | 0.344 | 0.537 m | 0.460 |
| Gaussian keypoint-consensus | 0.063 | 0.375 | 0.570 m | 0.457 |

ShopFacade q16, all-map:

| map | S@25cm/10deg | S@50cm/10deg | median t | inlier Patch@1 |
| --- | ---: | ---: | ---: | ---: |
| SfM sparse | 1.000 | 1.000 | 0.122 m | 0.623 |
| Gaussian quality | 1.000 | 1.000 | 0.116 m | 0.334 |

ShopFacade pose accuracy did not collapse, but correspondence correctness is
much lower. Its Gaussian field coordinates also have an abnormal range, so this
scene is not a clean positive result for Gaussian anchors.

## Conclusion

The current Gaussian sparse landmark branch does not improve localization over
the SfM-track landmark map. OldHospital remains clearly worse even under
GT-visible oracle submaps, so the main bottleneck is not only candidate-pool
selection. The evidence points to Gaussian feature/geometry attribution mismatch:
the selected Gaussians are visually dense and project onto facade structure, but
their VFM descriptors produce less GT-correct patch correspondences than
multi-view SfM-track aggregated descriptors.

This does not invalidate ULF-Loc's full design. The current implementation lacks
two important ULF-Loc ingredients: rendered Gaussian visibility masks during
keypoint voting and true 3DGS-native feature fusion tied to selected Gaussian
landmarks. Until those are available, the SfM landmark map should remain the
main localization anchor set.
