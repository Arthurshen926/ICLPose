# Gaussian Feature Map Five-Step Plan

Objective: rebuild the Gaussian VFM feature-map branch from raw RADIO features first, then evaluate and retrain selectors on the corrected Gaussian map instead of reusing SfM-specific checkpoints.

## Step 1: Render-Contribution Visibility

Status: done in this round.

Deliverables:

- Added `GaussianTokenContributionView`.
- Added `aggregate_raw_vfm_features_from_token_contributions`.
- Added contribution gates to `RawGaussianFeatureAggregationConfig`:
  - `min_contribution_alpha`
  - `max_contribution_entropy`
- Added `--contribution_map_dir` support to `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py`.
- The builder can now consume per-reference NPZ files with:
  - `top_contributor`
  - optional `top_alpha`
  - optional `alpha_entropy`
- No contribution directory means the builder keeps using the current owner-projection fallback.

Verification:

```bash
PYTHONPATH=. pytest -q tests/test_vfm_gaussian_raw_landmarks.py tests/test_vfm_semidense_anchor_map.py -q
PYTHONPATH=. python -m py_compile feature_extract/vfm/gaussian_raw_landmarks.py feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py
```

Result:

- `22 passed`
- `py_compile` passed

## Step 2: Full-View Owner-Visible Aggregation

Status: done.

Tasks:

- Build all-train-view OldHospital Gaussian raw maps.
- Sweep `min_votes=1/2/3`.
- Report coverage, support count, vote distribution, and q32 localization.

Deliverables:

- Added `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map_sweep.py`.
- Added `subset_gaussian_anchor_map_by_source_indices` and regression coverage.
- Built all-view OldHospital raw Gaussian maps under:
  `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/full_owner_visible_allviews_top20k/`
- Ran q32 allmap and GT-visible localization for min_votes 1/2/3 under:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/full_owner_visible_allviews_top20k/`
- Wrote report:
  `docs/vfm/stage_h2_full_view_owner_visible_step2.md`

Main result:

- Full-view map uses 895 views.
- all895 min1/min2: 19,984 anchors, median support 52.
- q32 allmap: `S@25=0.3125`, `S@50=0.7812`, median t `0.337m`.
- q32 GT-visible: `S@25=0.3750`, `S@50=0.8125`, median t `0.317m`.
- min_votes=3 is worse for allmap.

## Step 3: Sampling Ablation

Status: done.

Tasks:

- Compare saliency norm top5%, local contrast top5%, top10%, mask-filtered saliency, keypoint vote, and render contribution vote where available.
- Keep raw RADIO aggregation fixed.

Deliverables:

- Added `vote_gaussians_from_vfm_token_saliency_configs` and regression coverage.
- Added `feature_extract/tools/vfm/build_stage_h2_sampling_ablation_sweep.py`.
- Built all-view OldHospital sampling ablation maps under:
  `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/sampling_ablation_allviews_top20k/`
- Ran q32 allmap and GT-visible localization for:
  - `norm_top5`
  - `norm_top10`
  - `local_contrast_top5`
  - `sift_keypoint_vote`
- Wrote report:
  `docs/vfm/stage_h2_sampling_ablation_step3.md`

Main result:

- `norm_top5` remains the canonical sampling strategy.
- `norm_top10` and `local_contrast_top5` increase support/inliers but reduce S@25.
- `sift_keypoint_vote` is a negative diagnostic: only 3,646 anchors survive raw RADIO owner-visible aggregation, and q32 allmap S@25 drops to 0.
- Mask-filtered saliency and render-contribution vote were marked unavailable because the required per-view mask / contribution NPZ files are not present.

## Step 4: Raw Gaussian Map Evaluation

Status: done.

Tasks:

- Evaluate raw Gaussian maps on q32/full182.
- Compare against old H2 and SfM raw baseline.
- Do not use trained selectors in this stage.

Deliverables:

- Ran canonical raw1280 MNN top1, 20k submap cap, 0.75-stride EPNP+LM protocol on OldHospital full182 for:
  - new all-view owner-visible Gaussian raw map, `norm_top5`
  - old H2 no-owner v96 Gaussian raw map
  - SfM raw balanced300k landmark bank
- Wrote unified table:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/raw_eval_canonical/step4_raw_eval_summary_table.json`
- Wrote report:
  `docs/vfm/stage_h2_raw_map_evaluation_step4.md`

Main result:

- The corrected Gaussian raw map improves over old H2 on q32 and GT-visible full182 S@25, and improves PnP-inlier patch correctness.
- The SfM raw landmark bank is still the stronger raw anchor baseline under the same protocol: full182 GT-visible S@25 is `0.269` for SfM raw vs `0.176` for the corrected Gaussian map, with PnP-inlier Patch@1 `0.600` vs `0.407`.
- This confirms the Gaussian branch should not reuse the SfM-trained selector, but Step 5 must train Gaussian-specific descriptors and test whether selector learning can close the raw-anchor gap.

## Step 5: Gaussian-Specific Selector Retraining

Status: done in smoke form.

Tasks:

- Train Gaussian-specific C1/C2 selectors from corrected raw Gaussian maps.
- Compare against raw, PCA, random, and SfM-trained selector transfer.
- Verify on OldHospital and ShopFacade.

Deliverables:

- Built an OldHospital corrected-Gaussian training cache:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/gaussian_specific_norm_top5_allviews/sample_cache_q128_8k_trainq64_seed0.npz`
- Trained Gaussian-specific:
  - C1 learned128
  - C1 learned64
  - conservative C2-safe128 initialized from learned128
- Built Gaussian-map fitted C0 baselines:
  - random128
  - PCA128
- Evaluated direct SfM-trained selector transfer using:
  `output/vfm/stage_c25_pose_refinement/oldhospital/c25_margin03_seed0/selector.pt`
- Wrote unified table:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/gaussian_specific_norm_top5_allviews/step5_gaussian_specific_selector_summary_table.json`
- Wrote report:
  `docs/vfm/stage_h2_gaussian_specific_selector_step5.md`

Main result:

- Gaussian-specific learned128 gives the best S@25 among 128D compressed variants and beats PCA/random/transfer on full182 S@25.
- The gain is limited: full182 GT-visible S@25 improves from raw `0.176` to learned128 `0.192`, while median translation worsens from `0.471m` to `0.534m` and S@50 drops from `0.538` to `0.478`.
- learned64 is not stable enough, and C2-safe128 improves sample-level top1 but hurts localization.
- SfM-trained C2.5 selector transfer is not a replacement for Gaussian-specific training; it does not improve S@25.
- The all-895-query 20k sample cache path was too slow without early-stop/streaming; current Step 5 is a smoke validation, not a final all-data training result.
