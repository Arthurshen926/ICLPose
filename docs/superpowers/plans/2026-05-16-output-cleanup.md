# 2026-05-16 Output Cleanup

Removed only regenerated/negative artifacts that are outside the active POFD single-localization-feature mainline.

UTC time: 2026-05-16 07:20:34

## Removed
5.9G	result/result/feature_extract/fine_selector_cache
609M	result/result/feature_extract/fine_selector_caches
15M	result/result/feature_extract/pofd_stage3_rich_selector_q50_shuf_from_mixed1025_b12_s60_20260516
24K	result/result/feature_extract/pofd_stage3_rich_selector_q10q25_shuf_from_mixed1025_b12_s40_20260516

Reason: fine-selector cache route and rich selector replacement were negative diagnostics; latest ChatGPT-ICLPose.md resets the mainline to POFD single localization feature / pose-observability refinement.

## Removed Interrupted Diagnostic
80K	result/result/feature_extract/pofd_stage3_residual_selector_q50_shuf_from_mixed1025_b12_s60_20260516
Reason: interrupted after 9 train rows because it belongs to the old residual topK selector branch; latest mainline prioritizes POFD single localization feature experiments.
