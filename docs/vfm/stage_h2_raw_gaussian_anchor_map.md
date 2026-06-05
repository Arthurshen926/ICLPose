# Stage H2: Raw VFM Gaussian Anchor Map

## Purpose

Stage H2 fixes the earlier Gaussian-anchor experiment order.

The intended flow is:

1. sample reliable Gaussian anchors from the trained 3DGS map;
2. aggregate raw high-dimensional VFM features onto those anchors from posed reference views;
3. only then apply or train a selector/projection;
4. evaluate query tokens and map anchors in the same descriptor space.

This differs from the failed Stage H path, where selected descriptors were attached to Gaussians before validating raw VFM aggregation.

## Implementation

Core module:

- `feature_extract/vfm/gaussian_raw_landmarks.py`

CLIs:

- `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py`
- `feature_extract/tools/vfm/project_stage_h2_gaussian_anchor_map.py`
- `feature_extract/tools/vfm/apply_transform_to_stage_h2_gaussian_map_and_queries.py`

The Gaussian anchor proposal is VFM-native. Since RADIO/VFM does not provide a keypoint detector like SuperPoint, Stage H2 uses token saliency voting:

- project Gaussians into each posed reference token grid;
- compute token saliency from raw VFM features;
- vote for the frontmost Gaussian owner of salient tokens;
- sample voted Gaussians with vote count, opacity, and voxel NMS;
- aggregate raw 1280D VFM observations by bilinear sampling from reference token maps.

The output map is a `SemiDenseAnchorMap` with `source_type=gaussian_raw_vfm`, raw high-dimensional features, observation counts, visibility counts, and feature variance diagnostics.

## OldHospital Q32 Validation

Protocol:

- scene: OldHospital
- query subset: first 32 test queries
- map: trained 3DGS PLY
- raw VFM: RADIO `radio_final`, 1280D
- matching: patch-level MNN top1
- PnP: stride-aware threshold 0.75, EPNP, LM refinement

Main H2 map:

- reference views: 96 uniform train views
- source Gaussians: 504,352
- sampled anchors: 20,000
- valid aggregated anchors: 19,991
- aggregation: raw VFM first, selector/projection second

Results:

| Method | Submap | S@25cm/10deg | S@50cm/10deg | Median t | Median r | PnP solve | Inlier Patch@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage H selected-before-Gaussian | all-map | 0.094 | 0.438 | 0.546m | n/a | n/a | n/a |
| H2 raw Gaussian, 7.5k anchors | all-map | 0.188 | 0.594 | 0.443m | 0.746deg | 1.000 | 0.345 |
| H2 old C2.5 selector, 7.5k anchors | all-map | 0.094 | 0.594 | 0.461m | 0.833deg | 1.000 | 0.341 |
| H2 old C2.5 selector, 20k anchors | all-map | 0.312 | 0.688 | 0.355m | 0.565deg | 1.000 | 0.317 |
| H2 old C2.5 selector, 20k anchors | GT-visible | 0.281 | 0.750 | 0.324m | 0.462deg | 1.000 | 0.332 |
| SfM sparse selected128 baseline | all-map | 0.844 | 1.000 | 0.171m | n/a | n/a | 0.673 |

The corrected order gives a clear positive signal over Stage H. Increasing view count and anchors helps. However, H2 still trails the SfM sparse map by a large margin, mostly through lower patch correctness and lower PnP-inlier correctness.

## Selector Smoke

`train_stage_c1_patch_selector.py` now supports `--semidense_anchor_npz`, so the same patch-positive training path can be used with H2 Gaussian anchors.

A tiny smoke run was executed:

- train queries: 8
- samples: 1,024
- steps: 80
- output dim: 128
- raw eval top1: 0.088
- learned eval top1: 0.559

The transform was then applied to both the raw Gaussian anchor map and OldHospital test query tokens. Q32 localization was poor:

| Method | Submap | S@25cm/10deg | S@50cm/10deg | Median t | PnP solve | Inlier Patch@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2 Gaussian-trained C1 smoke | all-map | 0.000 | 0.077 | 4.444m | 0.406 | 0.091 |
| H2 Gaussian-trained C1 smoke | GT-visible | 0.000 | 0.000 | 2.711m | 0.656 | 0.072 |

This only validates the train/project/evaluate plumbing. It is not a usable selector checkpoint.

## Interpretation

The corrected Stage H2 flow is better aligned with ULF-Loc-style landmark construction:

- anchor sampling is performed on 3D geometry first;
- raw descriptors are aggregated after anchor selection;
- learned selectors are applied after raw map construction;
- VFM saliency replaces detector/keypoint voting for the first version.

The remaining gap is not just implementation order. The Gaussian anchors selected by simple VFM saliency are still less geometrically reliable than SfM landmarks for PnP. They contain more repeated-structure ambiguities and have weaker patch-to-landmark correctness.

The next useful experiment is not another tiny selector smoke. It is a full H2 selector training run with enough query coverage, plus a stronger Gaussian-anchor reliability filter that uses multi-view support, depth/visibility consistency, feature variance, and local ambiguity before localization.
