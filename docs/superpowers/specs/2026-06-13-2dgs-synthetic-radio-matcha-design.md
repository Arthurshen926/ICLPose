# 2DGS Synthetic RADIO-MATCHA Pair Training Design

## Status

Accepted direction for phase 1: train and validate MATCHA-like RADIO geometry on synthetic source/target pairs where both images are rendered from the same 2DGS scene. Real-image domain gap is intentionally deferred.

## Motivation

The previous GT-render diagnostic path reached centimeter-level pose error because the render/query pair was geometrically aligned and the local-window MATCHA path produced nearly perfect correspondences:

- `diagnostics_v1/eval_train_agt_only_default_1000_gt_full182`
- median translation `0.0408m`
- median rotation `0.086deg`
- `mean_gt_precision_16px = 1.000`
- `mean_pnp_inlier_gt_precision_16px = 1.000`

The latest `patch_corr_v1` path regressed because it made an unproven patch-correlation fine head the primary fine matcher:

- median translation `0.289m`
- median rotation `0.500deg`
- `mean_gt_precision_16px = 0.657`
- training `patch_corr_fine_acc = 0.0149`
- validation manifest empty

The next mainline should keep MATCHA-like RADIO matching, but remove the real-image/render domain gap and fixed perturbation buckets from the first phase. The goal is to prove that RADIO-MATCHA can learn clean geometric correspondence when 2DGS supplies both sides of the pair and exact geometry supervision.

## Goals

1. Generate deterministic synthetic source/target training pairs from 2DGS-rendered views.
2. Train the existing `radio_dual_attention` MATCHA-like model on these pairs without using fixed `A_gt/B025/C050/D_reference` bucket semantics.
3. Restore strong local-window/coarse-to-fine geometry supervision before using patch correlation as a primary fine path.
4. Add non-empty held-out validation for every synthetic training run.
5. Evaluate pair-level correspondence and PnP accuracy across continuous relative-pose bins before returning to real query/render evaluation.

## Non-Goals

- Do not solve the real-image to render domain gap in phase 1.
- Do not use reference retrieval candidates as a training distribution in phase 1.
- Do not claim full localization from GT-render or synthetic-pair results.
- Do not make `patch_corr_v1` the default evaluation path until its fine accuracy passes synthetic gates.

## Architecture

### Synthetic Pair Manifest

Add a synthetic manifest path alongside the current streaming manifest:

- Existing v1 records can still identify an anchor pose through `query_id`.
- The manifest metadata declares `pair_source = "2dgs_synthetic"`.
- Records use a synthetic pair type such as `S2DGS_RANDOM`; the numeric `pair_type_id` is kept only for logging.
- The record seed deterministically samples source pose, target pose, and optional source/target cache keys.

The synthetic manifest must include sampling configuration in metadata:

- source anchor pose file
- source jitter translation range
- source jitter rotation range
- target relative translation range
- target relative rotation range
- minimum overlap / visible-supervision count
- render size
- sampling seed

### Synthetic Pair Builder

Add a synthetic builder used by `train_matcha_joint_streaming_model.py` when manifest metadata has `pair_source = "2dgs_synthetic"`.

For each record:

1. Load the anchor camera pose from `query_pose_file`.
2. Sample `source_pose_w2c` around the anchor.
3. Sample `target_pose_w2c` relative to the source pose.
4. Render source RGB/depth/alpha from 2DGS.
5. Render target RGB/depth/alpha from 2DGS.
6. Extract RADIO dual features from both rendered RGB images.
7. Build supervision with existing geometry supervision code, treating source as query and target as render.
8. Build `MatchaJointTrainingSet` with source/target features, rendered RGBs, keypoint labels, offsets, no-match labels, and confidence targets.
9. Write diagnostic row fields for relative translation, relative rotation, overlap, roundtrip median, offset entropy, and supervision count.

This keeps the existing model and loss machinery while replacing only the data source.

### Training Recipe

Do not use the current `radio_matcha_patch_corr` preset as the first synthetic recipe.

Initial recipe:

- model: `radio_dual_attention`
- attention fusion: `matcha_original`
- dual-softmax weight: `1.0`
- offset loss weight: `0.25`
- pair confidence loss weight: `0.1`
- dense heatmap loss weight: `0.25`
- local-window fine loss weight: `0.5`
- patch-corr fine loss weight: `0.0`
- keypoint distillation: enabled on rendered RGB unless it destabilizes synthetic overfit

Patch correlation can be added only after the local-window synthetic baseline passes held-out validation. When added, it should be an auxiliary fine loss first, not the sole eval path.

### Sampling Curriculum

Use continuous relative-pose sampling rather than fixed named buckets.

Recommended bins:

- micro: `0-0.03m`, `0-1deg`
- small: `0.03-0.10m`, `1-3deg`
- medium: `0.10-0.25m`, `3-6deg`
- wide: `0.25-0.50m`, `6-10deg`

The first implementation should support:

- deterministic train and validation manifests
- per-record rejection/resampling when overlap or supervision count is too low
- summary counts per relative-pose bin

### Evaluation

Add a synthetic pair evaluation path before real query/render evaluation.

Required metrics:

- `mean_gt_precision_5px`
- `mean_gt_precision_10px`
- `mean_gt_precision_16px`
- `mean_pnp_inlier_gt_precision_16px`
- median PnP translation error between estimated and source/target ground truth geometry
- median PnP rotation error
- PnP solve rate
- mean PnP inlier count
- fine offset before/after median px
- fine improvement px
- metrics split by relative-pose bin

The first successful run must report both micro-overfit and held-out synthetic validation. A real-query GT-render run is only the next diagnostic after synthetic validation passes.

## Acceptance Gates

### Gate 1: Single-Pair Overfit

On one deterministic synthetic pair:

- `mean_gt_precision_16px >= 0.99`
- `mean_pnp_inlier_gt_precision_16px >= 0.99`
- PnP solve rate `1.0`
- median translation error `<= 0.05m`
- fine offset after must improve over before

### Gate 2: Held-Out Synthetic Small-Pose Validation

On at least 32 held-out synthetic pairs in micro/small bins:

- `mean_gt_precision_16px >= 0.95`
- `mean_pnp_inlier_gt_precision_16px >= 0.98`
- median translation error `<= 0.10m`
- median rotation error `<= 0.25deg`
- validation manifest must be non-empty and used for checkpoint selection

### Gate 3: Held-Out Synthetic Medium-Pose Validation

On at least 128 held-out synthetic pairs across micro/small/medium bins:

- report per-bin metrics
- no bin may have PnP solve rate below `0.95`
- medium-bin `mean_gt_precision_16px >= 0.85`
- medium-bin median translation error `<= 0.20m`

### Gate 4: Real GT-Render Diagnostic

Only after synthetic gates pass, evaluate real query image to GT 2DGS render:

- first target is to recover the previous local-window diagnostic scale, not to claim full localization
- compare against `diagnostics_v1/train_agt_only_default_1000`
- do not proceed to reference-topK unless GT-render correspondence is within striking distance of the previous diagnostic

## Implementation Boundaries

Likely files:

- `feature_extract/vfm/matcha_synthetic_pairs.py`: sampling config, deterministic source/target pose sampling, overlap bin labels.
- `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`: train/validation manifest creation.
- `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`: route `pair_source=2dgs_synthetic` to a synthetic builder.
- `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`: held-out synthetic pair eval.
- `tests/test_matcha_synthetic_pairs.py`: deterministic sampling, binning, manifest metadata.
- `tests/test_matcha_streaming_manifest.py`: CLI and builder routing smoke tests.

Existing files should keep their current real-query streaming behavior.

## Risks

- Synthetic-only training can overfit 2DGS render statistics and still fail on real images. This is acceptable in phase 1 because the phase goal is geometry cleanliness.
- If source/target views are too easy, the model may learn identity-like shortcuts. The sampling report must include relative pose and overlap distributions.
- If source/target views are too hard early, supervision will become sparse and unstable. The curriculum should start with micro/small bins.
- Patch-corr fine supervision may still fail even in synthetic mode. If so, keep local-window as the primary fine path and treat patch-corr as a separate research branch.

## Decision

Proceed with a 2DGS-render-to-2DGS-render synthetic MATCHA-like RADIO stage. Do not restore fixed `A_gt/B025/C050/D_reference` as the training distribution. Use 2DGS as the simulator for random source/target pairs, validate geometry matching in synthetic space, then address the real-image domain gap as a separate second phase.
