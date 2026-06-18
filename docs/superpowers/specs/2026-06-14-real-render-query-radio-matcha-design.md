# Real Render-Query RADIO-MATCHA Mainline Design

## Status

Accepted direction for the next phase: keep 2DGS Synthetic RADIO-MATCHA as the
pretraining stage, then move the main validation burden to real-image
render-query training and full-test evaluation.

The current synthetic stage shows that RADIO/VFM tokens can be trained for
render-to-render geometry matching. It does not yet prove real-image
generalization, robust initialization, no-match reasoning, or localization
verification. This design makes those missing pieces explicit.

## Research Claim Boundary

The next phase may claim only the following if its gates pass:

> A RADIO/VFM feature mapping and geometry matcher trained with synthetic
> render-to-render pretraining plus real-image render-query supervision can
> improve pose-conditioned correspondence and PnP under controlled render-pose
> initialization errors.

It must not claim compact 3D anchor mapability, interpretability, or final
retrieval-free localization until those are evaluated separately.

## Design Goals

1. Scale synthetic pretraining beyond 64 pairs per pose bin.
2. Add all available real training images as real-query / 2DGS-render pairs.
3. Train across GT render, pose-perturbed render, and reference-candidate render
   distributions without mixing their claims.
4. Add no-match and occlusion-aware supervision so the matcher does not assume
   every query token has a valid rendered counterpart.
5. Freeze a single recipe before full-test reporting.
6. Evaluate on the full real test split across multiple render-pose initializers.

## Non-Goals

- Do not implement compact 3D anchor aggregation in this phase.
- Do not claim final global localization without retrieval or pose candidates.
- Do not tune fine-head choices on the final test matrix.
- Do not report synthetic centimeter-level pose as real localization accuracy.

## Data Protocols

### Protocol A: Synthetic Render-Render

Purpose: pretrain geometry correspondence in a clean simulator.

Required scale:

- at least 1024 train pairs per scene per bin for micro/small/medium
- at least 128 held-out pairs per scene across micro/small/medium
- train and validation query IDs must be disjoint
- report pair counts and bin counts in every summary

### Protocol B: Real Query / GT Render

Purpose: measure real-image to render domain transfer without pose-init error.

For every training image:

- query side is the real RGB image and extracted RADIO/VFM tokens
- render side is the 2DGS render at the GT pose
- rendered depth and known camera geometry provide supervision labels
- GT pose is a label source only, never an input feature

For OldHospital this means 895 train queries and 182 test queries.

### Protocol C: Real Query / Perturbed Render

Purpose: train robustness to realistic pose initialization error.

For every training image, generate render poses from a frozen offset grid:

- translation magnitudes: 0.05m, 0.10m, 0.25m, 0.50m
- rotation magnitudes: 1deg, 3deg, 6deg, 10deg
- deterministic seeds and metadata for every perturbation
- no-match labels for query/render regions that fail visibility or roundtrip

### Protocol D: Real Query / Reference Candidate Render

Purpose: train and evaluate deployment-like candidate verification.

Inputs:

- existing reference/retrieval candidate banks
- top1/top5/top10 render candidates
- hard negatives where candidate pose is wrong but descriptor similarity is high

This protocol is not allowed to use oracle candidate choice during evaluation.

## Manifest Changes

Add an explicit render-query manifest schema on top of the existing
`MatchaStreamingPairManifest` records.

Metadata must include:

- `pair_source`: one of `2dgs_synthetic`, `real_gt_render`,
  `real_perturbed_render`, `real_reference_render`
- `source_query_manifest`
- `query_pose_file`
- `candidate_bank` when reference candidates are used
- `pose_bin_policy`
- `train_query_count`
- `validation_query_count`
- `train_validation_query_overlap_count`
- `pair_type_counts`

The current thin manifest can still store records, but the builder must reject
ambiguous metadata. A record must not silently switch between real and synthetic
semantics based only on `pair_type`.

## Training Architecture

The pipeline remains:

```text
query image/tokens + rendered image/tokens
  -> RADIO/VFM feature mapping
  -> coarse descriptor matching
  -> local fine refinement
  -> confidence/no-match heads
  -> PnP using rendered depth
```

### Feature Mapping

Keep the `radio_dual_attention` model as the baseline, but the recipe must
separate:

- descriptor projection loss
- heatmap/keypoint loss
- fine-refinement loss
- no-match/confidence loss

Descriptor freezing and head-only training are allowed only as named ablations.
The main recipe must state whether the descriptor projection is trainable.

### Geometry Matching

The main recipe should use local-window fine matching as the stable default.
Patch-correlation fine matching may be evaluated only after it passes a frozen
validation gate. It must not be selected by test-set performance.

### No-Match Supervision

No-match samples are required for real-query protocols:

- invalid render depth
- alpha below threshold
- failed roundtrip reprojection
- outside image after projection
- occluded or depth-inconsistent regions

Training summaries must report positive match count, no-match count, and
positive/no-match balance.

## Evaluation Matrix

Run one frozen checkpoint on the full real test split.

For OldHospital this means all 182 test images under:

- GT render
- GT render with 0.05m offset
- GT render with 0.10m offset
- GT render with 0.25m offset
- GT render with 0.50m offset
- GT render with 1deg rotation offset
- GT render with 3deg rotation offset
- GT render with 6deg rotation offset
- GT render with 10deg rotation offset
- reference top1
- reference top5
- reference top10

Required metrics:

- median translation error
- median rotation error
- 5cm/2deg success
- 10cm/5deg success
- 25cm/10deg success
- PnP solve rate
- mean match count
- mean PnP inlier count
- mean GT precision at 5/10/16/32px
- mean PnP-inlier GT precision at 5/10/16/32px
- failure-count table by render-pose initializer

Required baselines:

- raw RADIO descriptor matching
- current clean `A_gt`/`ABCD` checkpoints
- existing `reference_top1` and `reference_top5` reports
- a fixed HLoc or SuperPoint/SuperGlue baseline when available

## Acceptance Gates

### Gate 1: Synthetic Scale Gate

- at least 1024 train pairs per bin
- at least 128 held-out validation pairs
- train/validation query overlap count equals 0
- per-bin synthetic precision16 remains at or above the current small-scale
  trend after recipe freeze

### Gate 2: Real GT-Render Gate

On the full real test split:

- median translation error improves over the current `clean_abcd` GT-render
  median of 0.257m
- target threshold for strong promotion: median translation <= 0.10m
- 10cm/5deg success improves over the current `clean_abcd` 0.115
- PnP solve rate remains 1.0 or failures are explicitly counted

### Gate 3: Perturbed-Render Robustness Gate

On the full real test split:

- monotonic degradation is reported across increasing perturbation magnitudes
- 0.10m and 0.25m offsets improve over current clean protocol where comparable
- no-match count and inlier precision explain failures rather than hiding them

### Gate 4: Reference-Candidate Gate

On reference top1/top5/top10:

- improve over current reference-top1 median 0.426m and top5 median 0.448m
- report whether topK selection improves or hurts relative to top1
- no oracle candidate choice is allowed

## Implementation Units

1. Render-query manifest builder.
2. Real render-query streaming builder.
3. No-match and occlusion supervision extension.
4. Frozen recipe config and summary schema.
5. Full-test evaluation matrix runner.
6. Audit report generator that refuses to promote results when gates are not met.

## Risks

- Real-image to render domain gap may dominate even with all train images.
- The current fine heads may improve synthetic metrics without improving real
  PnP.
- Reference candidates may be too far from GT for local matching; in that case
  the method should be reframed as pose refinement within a bounded basin, not
  general localization.
- More data will not fix missing no-match supervision or poor pose-init
  robustness.

## Decision

Proceed in this order:

1. Implement protocol/manifest hygiene and full-test matrix tooling.
2. Add real GT-render training over all train images.
3. Add perturbed render training with no-match supervision.
4. Freeze a recipe.
5. Run full test matrix.
6. Only then revisit architecture changes beyond local-window fine matching.
