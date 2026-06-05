# Stage H2 Step 2: Full-View Owner-Visible Gaussian Raw Map

This step rebuilds the OldHospital Gaussian raw RADIO map with all train reference views instead of the previous `v96` subset.

## Protocol

- Scene: OldHospital
- Reference views: all 895 train views
- Gaussian source: `result/result/feature_gaussian/joint_rgb_geometry_cambridge_oldhospital_processed_rebuild_v4_4gpu_1280_safe_rgbft_30k/point_cloud/best/point_cloud.ply`
- Reference tokens: `output/vfm_tokens_radio/OldHospital/train_manifest.json`
- Query tokens: `output/vfm_tokens_radio/OldHospital/test_manifest.json`
- Feature: raw RADIO `radio_final`, 1280D
- Sampling: VFM token saliency, norm top 5%
- Owner visibility: enabled
- Owner opacity threshold: 0.1
- Gaussian sampling: max 20k, opacity power 1.0, min opacity 0.05, NMS voxel 0.03m
- Aggregation: L2-normalized observations, L2-normalized mean feature, min observations 2
- Evaluation: q32, MNN top1, all-map / GT-visible, EPNP + LM, 0.75 stride PnP threshold

## Artifacts

- Sweep builder: `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map_sweep.py`
- Sweep output: `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/full_owner_visible_allviews_top20k/`
- Eval output: `output/vfm/stage_h2_gaussian_mainline/oldhospital/full_owner_visible_allviews_top20k/`
- Summary table: `output/vfm/stage_h2_gaussian_mainline/oldhospital/full_owner_visible_allviews_top20k/step2_summary_table.json`

## Coverage And Support

| map | views | anchors | coverage | selected vote median | support median | support p90 |
|---|---:|---:|---:|---:|---:|---:|
| v96 min1 | 96 | 17,231 | 3.42% | 1 | 8 | 28 |
| v96 min2 | 96 | 6,330 | 1.26% | 2 | 11 | 34 |
| v96 min3 | 96 | 3,126 | 0.62% | 4 | 15 | 39 |
| all895 min1 | 895 | 19,984 | 3.96% | 6 | 52 | 224 |
| all895 min2 | 895 | 19,984 | 3.96% | 6 | 52 | 224 |
| all895 min3 | 895 | 19,999 | 3.97% | 7 | 59 | 243 |

With all train views, `min_votes=1` and `min_votes=2` select the same top-20k set: every selected Gaussian already has at least two saliency votes. Full-view aggregation substantially increases observation support per anchor.

## q32 Localization

| map | submap | S@25cm/10deg | S@50cm/10deg | median t | median r | inlier Patch@1 | matches | inliers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v96 min1 | allmap | 0.3125 | 0.7500 | 0.311m | 0.700deg | 0.356 | 588.6 | 112.0 |
| v96 min1 | gtvisible | 0.2188 | 0.7812 | 0.371m | 0.664deg | 0.369 | 557.6 | 114.2 |
| all895 min1 | allmap | 0.3125 | 0.7812 | 0.337m | 0.751deg | 0.423 | 553.8 | 106.1 |
| all895 min1 | gtvisible | 0.3750 | 0.8125 | 0.317m | 0.588deg | 0.400 | 520.7 | 108.8 |
| all895 min2 | allmap | 0.3125 | 0.7812 | 0.337m | 0.751deg | 0.423 | 553.8 | 106.1 |
| all895 min2 | gtvisible | 0.3750 | 0.8125 | 0.317m | 0.588deg | 0.400 | 520.7 | 108.8 |
| all895 min3 | allmap | 0.2188 | 0.6875 | 0.400m | 0.622deg | 0.391 | 559.7 | 108.9 |
| all895 min3 | gtvisible | 0.3125 | 0.7812 | 0.319m | 0.560deg | 0.403 | 533.9 | 114.2 |

## Conclusion

Full-view aggregation is positive for map support and GT-visible localization:

- Anchor support increases from median 8 to 52 observations for the comparable top20k map.
- GT-visible q32 improves from v96 min1 `S@25=0.2188` to all895 min1/min2 `S@25=0.3750`.
- All-map q32 broad recall improves from `S@50=0.7500` to `0.7812`, but median translation is slightly worse than v96 min1.
- `min_votes=3` is too strict for all-map localization despite high support; it hurts coverage/descriptor diversity enough to lower S@25 and S@50.

Default for the next sampling ablation should be:

`all895 + min_votes=1/2 + owner-visible raw RADIO top20k`

The remaining bottleneck is not raw observation support. The all-map gap versus GT-visible indicates that false matches / global distractors still dominate without better sampling, map reliability, or learned Gaussian-specific selection.
