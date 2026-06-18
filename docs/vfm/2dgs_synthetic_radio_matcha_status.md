# 2DGS Synthetic RADIO-MATCHA Status

## Mainline Definition

The active mainline is phase-1 **2DGS Synthetic RADIO-MATCHA**.

The research question for this phase is not final visual localization. The
question is whether frozen RADIO/VFM dense tokens can be trained, in a clean
2DGS-rendered source/target setting, into a geometry-matchable feature mapping
that supports accurate correspondence and PnP.

This phase intentionally removes the real-image-to-render domain gap. Both
source and target images are rendered from the same 2DGS scene. Real query
images, reference retrieval candidates, compact 3D aggregation, and final
localization are later stages.

## Active Pipeline

```text
2DGS scene + anchor poses
  -> deterministic synthetic source/target pose sampling
  -> source and target 2DGS RGB/depth/alpha rendering
  -> RADIO/VFM dense token extraction
  -> MATCHA-style dual-view geometry matching
  -> fine correspondence refinement
  -> pair-level correspondence metrics
  -> PnP from predicted correspondences and rendered depth
```

## Implemented Components

- `feature_extract/vfm/matcha_synthetic_pairs.py`
  - Synthetic pair sampling config.
  - Deterministic source/target pose jitter.
  - Relative-pose bins: `micro`, `small`, `medium`, `wide`.

- `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`
  - Tensor-free synthetic train/validation manifest builder.
  - Manifest metadata declares `pair_source=2dgs_synthetic`.

- `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
  - Synthetic builder routing when manifest metadata declares
    `pair_source=2dgs_synthetic`.
  - `radio_matcha_2dgs_synthetic` preset keeps patch-correlation loss off by
    default.

- `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`
  - Synthetic pair evaluation.
  - Reports correspondence precision, fine-offset diagnostics, PnP solve rate,
    PnP inlier precision, translation error, and rotation error.

- `feature_extract/vfm/official_2dgs_renderer.py`
  - Official 2DGS RGB/depth rendering adapter.

## Current Experimental Results

The current documented results are OldHospital-only and simulator-domain only.
They must not be reported as real-image localization accuracy.

| stage | run | pairs | precision16 | PnP inlier precision16 | median t | median r | solve |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-pair overfit | `overfit_q1_micro_1000_cache/eval_train_q1_micro_fixed_pnp` | 1 | 1.0000 | 1.0000 | 0.0088m | 0.036deg | 1.000 |
| held-out micro | `random_q64_micro_1000_npz/eval_test_q16_micro_fixed_pnp` | 16 | 0.9691 | 0.9998 | 0.0837m | 0.168deg | 1.000 |
| held-out small | `random_q64_small_2000_costvol_headonly_npz/eval_val_q16_small_patch_corr_fineconf075_pnp_lm8_repeat` | 16 | 0.9469 | 1.0000 | 0.0125m | 0.035deg | 1.000 |
| held-out medium | `random_q64_medium_2500_costvol_desc005_ctx_npz/eval_val_q16_medium_patch_corr_fineconf025_top900_pnp_conf_lm6` | 16 | 0.9275 | 1.0000 | 0.0332m | 0.054deg | 1.000 |

Interpretation:

- Gate 1 is passed.
- Gate 2 is not passed. The held-out set has 16 pairs, below the required 32,
  and the documented small-bin precision16 is 0.9469, below the 0.95 threshold.
- Gate 3 is not passed. The held-out medium result has 16 pairs, below the
  required 128.
- The synthetic PnP numbers are promising but are still a simulator-domain
  geometry sanity check.

## Acceptance Gates

### Gate 1: Single-Pair Synthetic Overfit

Required on one deterministic synthetic pair:

- `mean_gt_precision_16px >= 0.99`
- `mean_pnp_inlier_gt_precision_16px >= 0.99`
- PnP solve rate `1.0`
- median translation error `<= 0.05m`
- fine offset after refinement improves over before refinement

Status: **passed** on the documented OldHospital micro overfit run.

### Gate 2: Held-Out Synthetic Micro/Small Validation

Required on at least 32 held-out synthetic pairs in micro/small bins:

- `mean_gt_precision_16px >= 0.95`
- `mean_pnp_inlier_gt_precision_16px >= 0.98`
- median translation error `<= 0.10m`
- median rotation error `<= 0.25deg`
- validation manifest is non-empty and used for checkpoint selection

Status: **not passed**. Current held-out reports use 16 pairs.

### Gate 3: Held-Out Synthetic Medium Validation

Required on at least 128 held-out synthetic pairs across micro/small/medium:

- per-bin metrics are reported
- no bin has PnP solve rate below `0.95`
- medium-bin `mean_gt_precision_16px >= 0.85`
- medium-bin median translation error `<= 0.20m`

Status: **not passed**. Current medium report uses 16 pairs.

### Gate 4: Real GT-Render Diagnostic

Allowed only after Gates 2 and 3 pass:

- real query image matched to a GT-pose 2DGS render
- no reference-topK or retrieval candidate claims
- compare against the previous strong local-window diagnostic scale

Status: **not active**.

### Gate 5: Real Candidate Localization

Allowed only after Gate 4 is credible:

- real reference/retrieval/init candidates
- fixed PnP and solver baselines
- no synthetic result may be mixed into this claim table

Status: **not active**.

## Historical Lines Kept For Reference

The previous VFM-MapLoc selector, selected-track aggregation, rendered selected
map verifier, SfM/Gaussian anchor aggregation, dynamic VPR, candidate reranking,
and hard-case utility work are not the active phase-1 narrative. They are kept
as reference infrastructure for future compact 3D feature aggregation after the
synthetic geometry-matching stage is stable.

## Current Verification Commands

Run from the repository root:

```bash
PYTHONPATH=. pytest tests/test_matcha_synthetic_pairs.py \
  tests/test_matcha_streaming_manifest.py \
  tests/test_matcha_joint_training.py \
  tests/test_matcha_coarse_supervision.py \
  tests/test_matcha_multiview_supervision.py -q

PYTHONPATH=. pytest tests/test_vfm_*.py -q

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```

`PYTHONPATH=.` is required in this checkout for direct pytest invocation.

## Next Experiments

1. Build held-out micro/small manifests with at least 32 pairs and rerun Gate 2
   without changing the threshold after seeing results.
2. Build a 128-pair mixed micro/small/medium validation set and report per-bin
   metrics for Gate 3.
3. Run at least 3-5 random seeds after the single best recipe is frozen.
4. Add explicit train/validation manifest overlap checks to every summary.
5. Only after Gates 2 and 3 pass, run real-query to GT-render diagnostics.
