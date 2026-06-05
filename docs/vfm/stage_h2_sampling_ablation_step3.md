# Stage H2 Step 3: Gaussian Landmark Sampling Ablation

This step compares Gaussian landmark sampling strategies while keeping the downstream map feature construction fixed:

- raw RADIO `radio_final`, 1280D
- all 895 OldHospital train reference views
- owner-visible raw feature aggregation
- min observations 2
- max 20k sampled Gaussian anchors
- q32 patch-to-3D MNN top1 localization with EPNP + LM

## Artifacts

- Builder: `feature_extract/tools/vfm/build_stage_h2_sampling_ablation_sweep.py`
- Raw maps: `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/sampling_ablation_allviews_top20k/`
- Eval outputs: `output/vfm/stage_h2_gaussian_mainline/oldhospital/sampling_ablation_allviews_top20k/`
- Summary table: `output/vfm/stage_h2_gaussian_mainline/oldhospital/sampling_ablation_allviews_top20k/step3_sampling_ablation_summary_table.json`

## Availability

| sampling source | status |
|---|---|
| VFM saliency norm top5% | run |
| VFM saliency norm top10% | run |
| VFM local-contrast top5% | run |
| SIFT keypoint vote | run |
| mask-filtered saliency | unavailable: no per-view mask files found |
| render-contribution vote | unavailable: no per-view `top_contributor/top_alpha/alpha_entropy` NPZ files found |

## Map Statistics

| sampling | anchors | missing after raw aggregation | vote median | support median | support p90 |
|---|---:|---:|---:|---:|---:|
| norm_top5 | 19,984 | 16 | 6 | 52 | 224 |
| norm_top10 | 19,999 | 1 | 13 | 67 | 253 |
| local_contrast_top5 | 20,000 | 0 | 7 | 77 | 276 |
| sift_keypoint_vote | 3,646 | 16,354 | 40 | 6 | 45 |

SIFT keypoint vote is a negative diagnostic: many keypoint-voted Gaussians do not survive raw RADIO owner-visible aggregation, leaving only 3,646 usable anchors from a 20k sampled set. This supports the earlier suspicion that classical keypoint sampling is not aligned with VFM token ownership on the current Gaussian map.

## q32 Localization

| sampling | submap | S@25cm/10deg | S@50cm/10deg | median t | median r | inlier Patch@1 | matches | inliers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| norm_top5 | allmap | 0.3125 | 0.7812 | 0.337m | 0.751deg | 0.423 | 553.8 | 106.1 |
| norm_top5 | gtvisible | 0.3750 | 0.8125 | 0.317m | 0.588deg | 0.400 | 520.7 | 108.8 |
| norm_top10 | allmap | 0.1875 | 0.8125 | 0.362m | 0.708deg | 0.391 | 587.8 | 120.8 |
| norm_top10 | gtvisible | 0.1562 | 0.7500 | 0.368m | 0.565deg | 0.395 | 559.9 | 123.4 |
| local_contrast_top5 | allmap | 0.1875 | 0.6875 | 0.344m | 0.701deg | 0.423 | 659.5 | 138.4 |
| local_contrast_top5 | gtvisible | 0.2812 | 0.7188 | 0.348m | 0.568deg | 0.437 | 639.0 | 137.6 |
| sift_keypoint_vote | allmap | 0.0000 | 0.0625 | 1.771m | 1.196deg | 0.327 | 279.6 | 24.2 |
| sift_keypoint_vote | gtvisible | 0.0312 | 0.1562 | 1.652m | 1.292deg | 0.371 | 268.9 | 25.1 |

## Conclusion

`norm_top5` remains the default Gaussian sampling strategy for the raw VFM branch.

The useful signal is not simply more support or more matches:

- `norm_top10` increases support and inlier count, but reduces S@25 sharply.
- `local_contrast_top5` gives the highest support and inlier count, but broad recall and S@25 both drop.
- `sift_keypoint_vote` is not compatible with the current raw VFM Gaussian aggregation path; its surviving anchor coverage is too sparse and localization fails.

For the next stages:

- Use `norm_top5` as the canonical raw Gaussian map for q32/full182 raw evaluation.
- Keep `norm_top10` only as a broad-recall diagnostic, not the main map.
- Do not use SIFT keypoint vote as a main sampling method unless the Gaussian feature assignment is redesigned around keypoint support.
- Render-contribution vote remains the right target once true per-view raster contribution maps are available.
