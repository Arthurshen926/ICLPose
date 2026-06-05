# Stage H2 Gaussian Mainline Baseline

## Goal

This run treats the optimized Stage H2 Gaussian anchor map as a new landmark distribution and reruns the local descriptor-selection baseline instead of only reusing the old selector.

Map:

- `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/vfm_alpha_op01_score1_noowner_v96_top20k/raw_anchor_map.npz`
- sampling: `vote_min_owner_opacity=0.1`, `sample_opacity_power=1.0`, `sample_min_opacity=0.05`
- anchors: 19,914
- raw feature dim: 1280

## Training Runs

Two 128-query sample-cache variants were tested.

First-128 cache:

- manifest: `output/vfm_tokens_radio/OldHospital/train_manifest.json`
- query selection: first 128 records
- samples: 12,000
- cache: `output/vfm/stage_h2_gaussian_mainline/oldhospital/alpha_op01_score1_noowner_top20k/sample_cache_q128_12k_seed0.npz`
- cache build time: 313.5s

Uniform-128 cache:

- manifest: `output/vfm/stage_h2_gaussian_mainline/oldhospital/alpha_op01_score1_noowner_top20k/train_manifest_uniform128.json`
- query selection: 128 records uniformly sampled over the train manifest order
- samples: 12,000
- cache: `output/vfm/stage_h2_gaussian_mainline/oldhospital/alpha_op01_score1_noowner_top20k/sample_cache_uniform128_12k_seed0.npz`
- cache build time: 303.1s

Training:

- C1 learned128 linear projection, 400 steps, batch 2048
- C2 safe residual-gated selector, initialized from C1, 400 steps, batch 2048

Training metrics:

| Run | raw eval top1 | eval top1 | train top1 | inlier eval acc |
| --- | ---: | ---: | ---: | ---: |
| C1 first-128 | 0.122 | 0.416 | 0.477 | n/a |
| C2 first-128 | 0.122 | 0.513 | 0.621 | 0.789 |
| C1 uniform-128 | 0.113 | 0.353 | 0.382 | n/a |

The C2 training loss improves patch-ranking diagnostics, but final pose does not improve.

## Q32 Localization

Protocol:

- OldHospital first 32 test queries
- patch-level MNN top1
- PnP threshold: `0.75 * stride`
- EPNP + LM refinement

All-map:

| Descriptor | S@25 | S@50 | S@1m | Median t | Median r | matches | Patch@1 | PnP-inlier Patch@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw1280 | 0.219 | 0.625 | 0.813 | 0.377m | 0.542deg | 779 | 0.088 | 0.324 |
| frozen old C2.5 selector | 0.313 | 0.781 | 0.906 | 0.342m | 0.642deg | 1000 | 0.097 | 0.333 |
| C1 first-128 retrain | 0.063 | 0.406 | 0.938 | 0.540m | 1.090deg | 547 | 0.094 | 0.381 |
| C2 first-128 retrain | 0.125 | 0.406 | 0.781 | 0.560m | 1.048deg | 764 | 0.081 | 0.389 |
| C1 uniform-128 retrain | 0.031 | 0.063 | 0.375 | 1.264m | 2.378deg | 289 | 0.039 | 0.173 |
| C2 uniform-128 retrain | 0.031 | 0.281 | 0.781 | 0.704m | 1.330deg | 650 | 0.062 | 0.402 |

GT-visible:

| Descriptor | S@25 | S@50 | S@1m | Median t | Median r | matches | Patch@1 | PnP-inlier Patch@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw1280 | 0.281 | 0.688 | 0.844 | 0.300m | 0.585deg | 752 | 0.092 | 0.300 |
| frozen old C2.5 selector | 0.281 | 0.781 | 0.969 | 0.328m | 0.528deg | 996 | 0.104 | 0.307 |
| C1 first-128 retrain | 0.156 | 0.469 | 0.969 | 0.516m | 0.906deg | 483 | 0.109 | 0.424 |
| C2 first-128 retrain | 0.063 | 0.469 | 0.875 | 0.510m | 0.813deg | 665 | 0.097 | 0.369 |
| C1 uniform-128 retrain | 0.000 | 0.063 | 0.344 | 1.264m | 2.666deg | 272 | 0.042 | 0.189 |
| C2 uniform-128 retrain | 0.125 | 0.313 | 0.781 | 0.707m | 1.303deg | 587 | 0.069 | 0.371 |

## Interpretation

The retrained selectors improve some training-set patch ranking and, in a few cases, PnP-inlier Patch@1. They do not improve final pose. The failure mode is consistent:

- match count and spatial coverage drop;
- descriptor becomes sharper but less useful for broad PnP coverage;
- high PnP-inlier Patch@1 alone is insufficient if inliers cover too few or too biased regions;
- uniform-128 is worse than first-128, which indicates the current training objective is sensitive to query distribution and not yet a stable Gaussian-map descriptor learner.

The frozen old C2.5 selector remains the best q32 descriptor for this optimized Gaussian map. That result should be reported as a diagnostic, not as a final method claim, because it was trained on the older SfM-oriented distribution.

## Decision

Do not scale this exact retraining setup to full182 or more seeds yet. The q32 gate is negative.

The next correction should be in the training data and objective:

1. build a faster and more balanced sample-cache builder, so full train coverage is feasible;
2. use submap/visibility-aware sampling instead of all-map positives for every train query;
3. preserve broad match coverage with soft-mutual/topK or coverage-aware sampling during training;
4. use owner-visible observations as reliability labels or auxiliary supervision, not as the only feature aggregation path;
5. report spatial coverage and degeneracy during selector training, not only top1 patch accuracy.

Only after q32 retrained selector beats raw1280 and frozen old selector should full182 be run.
