# Stage H2 Step 5: Gaussian-Specific Selector Retraining

## Protocol

This step tests whether the corrected Gaussian raw map can benefit from a selector trained on Gaussian anchors rather than reusing the SfM mainline selector.

Map:

`output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/sampling_ablation_allviews_top20k/norm_top5/raw_anchor_map.npz`

Training cache:

- train split source: OldHospital train manifest
- cache size: 64 train queries, 8k samples
- positives: GT patch-positive Gaussian anchors
- negatives: raw hard negatives over the corrected Gaussian map
- artifact: `output/vfm/stage_h2_gaussian_mainline/oldhospital/gaussian_specific_norm_top5_allviews/sample_cache_q128_8k_trainq64_seed0.npz`

This is a smoke training result. An all-895-query / 20k cache attempt was stopped because the current CPU hard-negative sample builder did not write any artifact after a long run. The training pipeline needs early-stop or streaming negative mining before final full-data selector training.

Evaluation:

- scene: OldHospital
- split: full182 plus q32 derived from the first 32 rows of the same full182 jsonl
- matching: patch-level MNN top1
- submap cap: 20k landmarks
- PnP: 0.75-stride EPNP + LM
- no dense render, no context reranker

Unified artifact:

`output/vfm/stage_h2_gaussian_mainline/oldhospital/gaussian_specific_norm_top5_allviews/step5_gaussian_specific_selector_summary_table.json`

## Training Diagnostics

| Model | Output dim | Eval top1 | Raw eval top1 | Notes |
|---|---:|---:|---:|---|
| learned128 | 128 | 0.4825 | 0.1000 | strong sample-level improvement |
| learned64 | 64 | 0.4888 | 0.1000 | sample-level similar to 128D |
| C2-safe128 | 128 | 0.5175 | 0.1000 | better sample-level score, worse pose |

## full182 Results

| Method | Submap | S@10cm/5deg | S@25cm/10deg | S@50cm/10deg | Median t | Median r | Inlier Patch@1 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw1280 | all-map | 0.005 | 0.126 | 0.451 | 0.532m | 0.845deg | 0.405 |
| raw1280 | GT-visible | 0.005 | 0.176 | 0.538 | 0.471m | 0.689deg | 0.407 |
| random128 | all-map | 0.011 | 0.099 | 0.401 | 0.591m | 0.997deg | 0.383 |
| random128 | GT-visible | 0.022 | 0.148 | 0.456 | 0.525m | 0.870deg | 0.385 |
| PCA128 | all-map | 0.005 | 0.099 | 0.374 | 0.673m | 1.056deg | 0.355 |
| PCA128 | GT-visible | 0.011 | 0.099 | 0.418 | 0.585m | 0.961deg | 0.363 |
| learned128 | all-map | 0.027 | 0.143 | 0.445 | 0.639m | 1.223deg | 0.438 |
| learned128 | GT-visible | 0.022 | 0.192 | 0.478 | 0.534m | 0.957deg | 0.443 |
| learned64 | all-map | 0.016 | 0.137 | 0.401 | 0.659m | 1.377deg | 0.414 |
| learned64 | GT-visible | 0.011 | 0.115 | 0.451 | 0.577m | 1.192deg | 0.422 |
| C2-safe128 | all-map | 0.005 | 0.121 | 0.335 | 0.764m | 1.524deg | 0.423 |
| C2-safe128 | GT-visible | 0.038 | 0.187 | 0.401 | 0.658m | 1.354deg | 0.440 |
| SfM C2.5 transfer | all-map | 0.011 | 0.115 | 0.429 | 0.568m | 0.788deg | 0.429 |
| SfM C2.5 transfer | GT-visible | 0.005 | 0.099 | 0.489 | 0.502m | 0.722deg | 0.428 |

## q32 Results

| Method | Submap | S@25cm/10deg | S@50cm/10deg | Median t | Inlier Patch@1 |
|---|---|---:|---:|---:|---:|
| raw1280 | all-map | 0.312 | 0.781 | 0.337m | 0.423 |
| raw1280 | GT-visible | 0.375 | 0.812 | 0.317m | 0.400 |
| random128 | all-map | 0.219 | 0.625 | 0.428m | 0.351 |
| random128 | GT-visible | 0.219 | 0.625 | 0.429m | 0.351 |
| PCA128 | all-map | 0.156 | 0.438 | 0.537m | 0.308 |
| PCA128 | GT-visible | 0.125 | 0.562 | 0.428m | 0.303 |
| learned128 | all-map | 0.406 | 0.719 | 0.277m | 0.502 |
| learned128 | GT-visible | 0.438 | 0.781 | 0.281m | 0.531 |
| learned64 | all-map | 0.344 | 0.594 | 0.350m | 0.501 |
| learned64 | GT-visible | 0.312 | 0.719 | 0.343m | 0.510 |
| C2-safe128 | all-map | 0.188 | 0.562 | 0.396m | 0.489 |
| C2-safe128 | GT-visible | 0.344 | 0.625 | 0.334m | 0.525 |
| SfM C2.5 transfer | all-map | 0.281 | 0.625 | 0.366m | 0.401 |
| SfM C2.5 transfer | GT-visible | 0.188 | 0.750 | 0.336m | 0.424 |

## Conclusion

Gaussian-specific learned128 has a real but limited positive signal. It beats random128, PCA128, and direct SfM C2.5 transfer on full182 S@25, and it improves PnP-inlier Patch@1 from `0.407` to `0.443` on GT-visible.

The result is not yet strong enough for a final method claim. Compared with raw1280, learned128 improves full182 GT-visible S@25 from `0.176` to `0.192`, but worsens median translation from `0.471m` to `0.534m` and S@50 from `0.538` to `0.478`.

C2-safe128 is a negative result in this smoke setting: it improves sample-level eval top1 but hurts localization. This means the current C2 objective overfits the sampled correspondence task and does not yet produce safer pose estimates for Gaussian anchors.

Next engineering step for a final selector run:

- add early-stop/streaming to Gaussian sample-cache construction;
- run all-895-query train cache with 20k-50k samples;
- train learned128 over 3-5 seeds;
- add confidence/coverage-aware filtering before revisiting C2-safe.
