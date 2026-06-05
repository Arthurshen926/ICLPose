# Stage H2 Step 4: Raw Gaussian Map Evaluation

## Protocol

Scene: OldHospital.

This step evaluates raw RADIO descriptors only. No trained selector, PCA, random projection, context reranker, dense render, or SfM-trained selector is used.

Canonical local matching protocol:

- descriptor: raw1280 RADIO
- query: all query RADIO tokens, `query_token_step=1`
- matching: patch-level MNN top1, ratio test disabled, `min_similarity=0`
- submap cap: `max_submap_landmarks=20000`
- PnP: EPNP, threshold `0.75 * token_stride`, LM refinement enabled
- camera: COLMAP intrinsics from `/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/model_train`
- splits: `q32_from_full182_first32` and `full182`, both derived from the same full182 JSONL files

Compared maps:

- `gaussian_new_norm_top5_allviews_owner_visible`: corrected all-view owner-visible Gaussian raw map.
- `gaussian_old_h2_noowner_v96`: previous H2 Gaussian map built from 96 views with no owner-visible filtering.
- `sfm_raw_balanced300k_minobs2`: SfM raw RADIO landmark bank, evaluated with the same 20k per-query cap.

Artifacts:

- Unified table: `output/vfm/stage_h2_gaussian_mainline/oldhospital/raw_eval_canonical/step4_raw_eval_summary_table.json`
- New Gaussian full182 summaries:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/raw_eval_canonical/norm_top5/`
- Old H2 full182 summaries:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/raw_eval_canonical/old_h2_noowner_v96/`
- SfM raw full182 summaries:
  `output/vfm/stage_h2_gaussian_mainline/oldhospital/raw_eval_canonical/sfm_raw/`

## q32 Results

| Method | Submap | S@25cm/10deg | S@50cm/10deg | Median t | Median r | Inlier Patch@1 | Inlier GT@stride |
|---|---:|---:|---:|---:|---:|---:|---:|
| New Gaussian raw | all-map | 0.312 | 0.781 | 0.337m | 0.751deg | 0.423 | 0.836 |
| New Gaussian raw | GT-visible | 0.375 | 0.812 | 0.317m | 0.588deg | 0.400 | 0.779 |
| Old H2 Gaussian raw | all-map | 0.250 | 0.656 | 0.323m | 0.549deg | 0.305 | 0.700 |
| Old H2 Gaussian raw | GT-visible | 0.250 | 0.594 | 0.415m | 0.583deg | 0.303 | 0.682 |
| SfM raw | all-map | 0.688 | 0.969 | 0.175m | 0.451deg | 0.670 | 0.966 |
| SfM raw | GT-visible | 0.719 | 1.000 | 0.151m | 0.357deg | 0.678 | 0.976 |

## full182 Results

| Method | Submap | S@25cm/10deg | S@50cm/10deg | Median t | Median r | Inlier Patch@1 | Inlier GT@stride |
|---|---:|---:|---:|---:|---:|---:|---:|
| New Gaussian raw | all-map | 0.126 | 0.451 | 0.532m | 0.845deg | 0.405 | 0.815 |
| New Gaussian raw | GT-visible | 0.176 | 0.538 | 0.471m | 0.689deg | 0.407 | 0.814 |
| Old H2 Gaussian raw | all-map | 0.115 | 0.516 | 0.495m | 0.690deg | 0.341 | 0.756 |
| Old H2 Gaussian raw | GT-visible | 0.115 | 0.434 | 0.549m | 0.762deg | 0.337 | 0.749 |
| SfM raw | all-map | 0.225 | 0.484 | 0.532m | 0.983deg | 0.575 | 0.900 |
| SfM raw | GT-visible | 0.269 | 0.588 | 0.437m | 0.773deg | 0.600 | 0.932 |

## Conclusion

The corrected Gaussian raw map is a real improvement over the old H2 map for q32 and for GT-visible full182 S@25. It also improves PnP-inlier patch correctness from roughly `0.34` to `0.41`.

However, raw Gaussian anchors are still weaker than the SfM raw landmark bank under the same matching protocol. The gap is largest in PnP-inlier correctness and q32 S@25. This means the Gaussian branch is not ready to claim a better raw anchor map.

The next step should not apply the old SfM-trained selector to the Gaussian map. Step 5 should train Gaussian-specific descriptors on the corrected raw Gaussian anchors and compare against raw Gaussian, PCA/random Gaussian baselines, and SfM-trained selector transfer.
