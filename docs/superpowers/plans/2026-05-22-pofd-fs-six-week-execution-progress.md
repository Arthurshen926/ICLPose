# 2026-05-22 POFD-FS Six-Week Execution Progress

## Iteration 1: Baseline Audit

- Read `deep-research-report (1).md` and `docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md`.
- Confirmed the six-week scope:
  1. schema/tests/scorer/leakage,
  2. q50 randomized/jittered banks,
  3. ShopFacade controlled bank,
  4. OldHospital + ShopFacade scorer runs,
  5. near-identity hard negatives,
  6. mapability/risk/reporting.
- Verified initial localizability test baseline:
  `47 passed`.

## Iteration 2: Week 1 Engineering Fixes

Implemented:

- `feature_extract/localizability/bank_schema.py`
  - `CandidateRow`
  - JSONL row loader
  - forbidden training-input guard for GT/oracle/retrieval/PnP leakage fields
- `PoseHypothesisScorer` zero-mask fix:
  - `valid_mask` now uses raw weight sum, not clamped denominator.
- Python 3.8 CLI compatibility:
  - `augment_pose_candidate_cache.py`
  - `select_score_hard_candidate_cache.py`
  - `train_localizability_selector_stream.py`
  - `train_localizability_pairmatcher_stream.py`
- `build_localizability_banks.py`
  - `--pose-candidate-cache` now clears plural cache config.
  - retrieval scores are excluded by default to avoid controlled-bank leakage.

Verification:

```text
pytest -q tests/test_localizability_core.py tests/test_localizability_tools.py tests/test_localizability_risk_report.py tests/test_pose_cache_report.py
52 passed, 2 warnings
```

## Iteration 3: Week 2/5 q50 Candidate and Hard-Negative Work

Generated artifacts:

- `result/result/feature_extract/pose_init_exports/oldhospital_q50_train_scorehard_oracle1_hard7_from_top25_20260522.npz`
- `result/result/feature_extract/pose_init_exports/oldhospital_local_lattice_renderloftr_q50cm10deg_top16_plus_nearinit9_val128_shuf20260522.npz`
- `result/result/feature_extract/pose_init_exports/oldhospital_local_lattice_renderloftr_q50cm10deg_top16_plus_nearinit49_train256_shuf20260522.npz`
- `feature_extract/configs/pofd_stage3a_multiscale_score_q50_val128_top65_eval.yaml`
- `result/result/feature_extract/localizability_banks/oldhospital_q50_train256_nearinit65_standard_20260522.npz`

The top65 bank is clean for training:

```text
samples: 96
candidates: 65
retrieval_scores_candidates: absent
```

Hard7 pairmatcher training result:

| setup | pred | top1 | gap | Spearman | conclusion |
|---|---:|---:|---:|---:|---|
| hard7, batch1, 80 steps | 0.2227 | 0.7188 | 0.0933 | 0.584 | no improvement |

Calibrator sweep:

| run | pred | top1 | gap | Spearman | conclusion |
|---|---:|---:|---:|---:|---|
| seed17 | 0.2332 | 0.6953 | 0.1039 | 0.587 | worse than raw |
| seed18 | 0.2076 | 0.7734 | 0.0783 | 0.505 | passes pred/top1/gap, fails Spearman stability |

Interpretation:

- Scalar calibrator can cross top1/pred thresholds, but still damages score surface stability.
- Static hard7 replay does not move the frozen pair-matcher ranking.
- Next q50 work should score the new top65 pool and mine score-high wrong candidates from that broader pool, not keep compressing top25.

## Iteration 4: Week 3/4 ShopFacade Controlled Bank and Scoring

Generated ShopFacade q50 controlled cache:

- `result/result/feature_extract/pose_init_exports/shopfacade_local_lattice_renderloftr_q50cm10deg_top16_noexact_val103_20260522.npz`
- `feature_extract/configs/pofd_shopfacade_q50_controlled_val_eval.yaml`
- `result/result/feature_extract/localizability_banks/shopfacade_q50_val103_standard_20260522.npz`

Cache quality:

```text
queries: 103
candidates/query: 16
candidate LoFTR/PnP success rate: 0.719
queries with at least one successful candidate: 1.000
oracle median translation: 0.125 m
basin_any@16: 1.000
retrieval_scores_candidates: absent in standardized bank
```

Frozen ShopFacade pair-matcher score table:

- `result/result/feature_extract/pofd_fs_scoretable_shopfacade_pairmatcher_r16_q50_val103_20260522/`

| scene | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade q50 val103 | 0.3376 | 0.1294 | 0.2083 | 0.4175 | 0.3879 | 0.4660 | 0.8350 |

Interpretation:

- This is not a strong second-scene result yet.
- It does satisfy the immediate gate that ShopFacade is no longer a degenerate all-zero top1 case.
- The next action is to train/calibrate on a ShopFacade train controlled bank, then evaluate on this val103 bank.

## Iteration 5: Week 6/Public Diagnostics

Risk coverage artifacts:

- `result/result/feature_extract/pofd_fs_risk_coverage_q50_val128_margin_20260522.json`
- `result/result/feature_extract/pofd_fs_risk_coverage_shopfacade_q50_val103_margin_20260522.json`

OldHospital q50 margin risk:

```text
risk@25: 0.0938
risk@50: 0.1719
risk@100: 0.2578
```

ShopFacade q50 margin risk:

```text
risk@25: 0.3462
risk@50: 0.4808
risk@100: 0.5340
```

Reference-pose patch-topk16 sweep:

| scene | pred | top1 | Spearman | conclusion |
|---|---:|---:|---:|---|
| GreatCourt | 18.265 | 0.067 | -0.110 | worse than retrieval order |
| KingsCollege | 4.914 | 0.090 | -0.096 | worse than retrieval order |
| OldHospital | 4.858 | 0.115 | 0.069 | worse than retrieval order |
| ShopFacade | 2.343 | 0.155 | 0.081 | worse than retrieval order |
| StMarysChurch | 4.809 | 0.094 | 0.002 | worse than retrieval order |

Interpretation:

- Global/patch query-student descriptor scoring remains a negative public reference-pose baseline.
- The public positive path must use learned localizability scoring or rendered-map hypothesis evidence.

## Next Automatic Task Allocation

1. Build ShopFacade q50 train controlled cache and standardized bank.
2. Score OldHospital top65 pool with frozen pair-matcher and mine oracle+score-hard candidates from the broader pool.
3. Train a small calibrator with a Spearman constraint or early-stop rule, because seed18 crossed pred/top1 while failing Spearman.
4. Train/evaluate ShopFacade scorer or calibrator on the new train/val controlled banks.
5. Export actual compact POFD-FS selector descriptors and rerun mapability on ShopFacade, not only OldHospital query-student descriptors.

## Continuation: 2026-05-22 Execution Round 2

### Spearman-Aware Calibrator Selection

Implemented a selection-only Spearman gate in:

- `feature_extract/tools/train_localizability_score_calibrator.py`
- `tests/test_localizability_tools.py`

New option:

```text
--selection-spearman-min
```

Default behavior is unchanged.  When set, a checkpoint is saved only if it
improves validation `pred_cost_m` and satisfies the requested validation
Spearman floor.

Verification:

```text
tests/test_localizability_tools.py::test_score_calibrator_cli_accepts_selection_spearman_min
tests/test_localizability_tools.py::test_score_calibrator_selection_gate_rejects_low_spearman_checkpoint
tests/test_localizability_tools.py::test_score_calibrator_eval_row_marks_selection_eligibility_before_logging
tests/test_localizability_tools.py::test_load_checkpoint_for_rgb_export_falls_back_for_trusted_legacy_checkpoint
```

OldHospital q50 seed18 gated reruns:

| run | best step | pred | top1 | gap | Spearman | conclusion |
|---|---:|---:|---:|---:|---:|---|
| Spearman >= 0.55 | 400 | 0.2214 | 0.7344 | 0.0921 | 0.5668 | stable, but weak gain |
| Spearman >= 0.53 | 500 | 0.2135 | 0.7578 | 0.0841 | 0.5314 | useful tradeoff, below strict 0.55 |

### OldHospital Top65 Stress Test

Scored the broader q50 top65 pool:

- `result/result/feature_extract/pofd_fs_scoretable_pose_adapter_pairmatcher_r16_q50_train256_top65_cvd1_20260522/`

| pool | samples | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OldHospital train top65 | 96 | 0.3325 | 0.1269 | 0.2057 | 0.4375 | 0.2159 | 0.5104 | 0.9271 |

Risk report:

- `result/result/feature_extract/pofd_fs_risk_coverage_q50_train_top65_margin_20260522.json`

```text
risk@25: 0.5833
risk@50: 0.4583
risk@100: 0.4896
```

Hard mining from top65:

- `result/result/feature_extract/pofd_fs_scorehard_pairs_q50_train_top65_20260522/`
- `result/result/feature_extract/pose_init_exports/oldhospital_q50_train_scorehard_oracle1_hard7_from_top65_20260522.npz`

```text
samples with score-hard pair: 96/96
near-identity negative pairs: 75/96
mean cost gap: 0.3949 m
```

Pair-matcher training on the top65-mined hard7 cache:

- `result/result/feature_extract/pofd_fs_pairmatcher_scorehard_oracle1_hard7_top65_q50_s100_b1_20260522/`

| step | pred | top1 | gap | Spearman | basin@1 | conclusion |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 0.2227 | 0.7188 | 0.0933 | 0.5829 | 0.7422 | no gain over raw baseline |

Interpretation:

- Top65 is a useful failure stress test: it exposes many near-identity
  high-score negatives.
- Compressing those failures back into hard7 and training only the pair
  matcher still does not improve q50 validation.
- The next lever is not more hard7 replay alone.  The model needs either a
  stronger score head/calibrator that preserves Spearman, or a candidate pool
  curriculum that keeps the broader top65 evidence visible during training.

### ShopFacade Train Controlled Bank

Generated the q50 controlled train cache:

- `result/result/feature_extract/pose_init_exports/shopfacade_local_lattice_renderloftr_q50cm10deg_top16_noexact_train231_20260522.npz`

Cache quality:

```text
queries: 231
candidates/query: 16
candidate LoFTR/PnP success rate: 0.7992
queries with at least one successful candidate: 0.9913
best candidate translation median: 0.2513 m
best candidate rotation median: 7.50 deg
```

Built the leakage-safe standardized bank:

- `result/result/feature_extract/localizability_banks/shopfacade_q50_train231_standard_20260522.npz`

```text
shape: 231 x 16
valid fraction: 1.0
oracle median translation: 0.1250 m
retrieval_scores_candidates: absent
```

Frozen pair-matcher train score table:

- `result/result/feature_extract/pofd_fs_scoretable_shopfacade_pairmatcher_r16_q50_train231_20260522/`

| split | samples | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade train231 | 231 | 0.3709 | 0.1294 | 0.2415 | 0.3664 | 0.2857 | 0.4224 | 0.8578 |
| ShopFacade val103 | 103 | 0.3376 | 0.1294 | 0.2083 | 0.4175 | 0.3879 | 0.4660 | 0.8350 |

ShopFacade calibrator:

| run | best step | pred | top1 | gap | Spearman | conclusion |
|---|---:|---:|---:|---:|---:|---|
| Spearman >= 0.35 | none | n/a | n/a | n/a | n/a | no eligible checkpoint |
| Spearman >= 0.30 | none | n/a | n/a | n/a | n/a | no eligible checkpoint |
| no gate | 1100 | 0.2197 | 0.8544 | 0.0903 | 0.1210 | rank surface collapses |

Interpretation:

- A scalar calibrator can make ShopFacade top1 and pred much better, but it
  destroys Spearman.
- This should be treated as a reranking/selection upper bound, not a stable
  localizability scorer.

### ShopFacade Mapability

Fixed descriptor export for trusted legacy local checkpoints by adding a
limited `weights_only=False` fallback only in descriptor export.

Exported query-adaptive dense maps:

- `result/result/feature_extract/pofd_fs_descriptors/shopfacade_query_adaptive_fine_mapability160_20260522.pt`
- `result/result/feature_extract/features_query_adaptive_v7_locaware_fineloc_v1_mapability160/cambridge_shopfacade/fine_geo/`

Mapability:

| feature | used images | missing features | track variance |
|---|---:|---:|---:|
| raw RADIO fine | 80 | 0 | 0.0777 |
| query-adaptive fine | 80 | 138 | 0.00221 |

Interpretation:

- The query-adaptive feature is much more consistent over COLMAP tracks on
  ShopFacade than raw RADIO fine.
- This is a useful mapability diagnostic, but it does not overturn the negative
  reference-pose ranking results.  It should be reported as evidence that the
  feature is track-stable while still insufficient as a simple retrieval scorer.

## Next Automatic Task Allocation After Round 2

1. Add a Spearman-preserving objective or multi-objective checkpoint selector
   for ShopFacade calibration.  Current scalar calibration improves top1 but
   collapses the rank surface.
2. Train on a broader visible candidate pool instead of compressing top65 to
   hard7.  The top65 stress test shows the failures, but hard7 replay alone did
   not teach the pair matcher to fix them.
3. Build a second-scene learned scorer with explicit train/val promotion gates:
   pred, top1, Spearman, basin@1, and risk@25 must all be reported together.
4. Finish a public-scene positive path using rendered-map hypothesis evidence,
   not pooled or patch-only query-student descriptors.

## Continuation: 2026-05-22 Execution Round 3

### Protocol Code Added

Implemented the remaining lightweight protocol hooks requested by the research
plan:

- `feature_extract/localizability/mapability.py`
  - `observation_track_feature_separability()`
  - `render_query_feature_consistency()`
- `feature_extract/tools/eval_feature_track_mapability.py`
  - now reports `within_track_variance`, `between_track_distance`,
    `track_separability_ratio`, and `mapability_valid`
- `feature_extract/localizability/reference_pose_scoring.py`
  - `project_descriptor_bank_pca()`
- `feature_extract/tools/eval_reference_pose_feature_ranking.py`
  - new `--pca-out-dim` option for public reference-pose PCA baselines

Unit coverage was added in `tests/test_localizability_core.py` for PCA
descriptor projection, track separability, and dense render-query consistency.

### Public Reference-Pose PCA-64 Baseline

Ran PCA-64 descriptor reranking on all five Cambridge reference-pose banks.

| scene | PCA pred | PCA top1 | PCA Spearman | retrieval pred | retrieval top1 | retrieval Spearman | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| GreatCourt | 19.127 | 0.072 | -0.120 | 14.089 | 0.121 | 0.166 | PCA worse |
| KingsCollege | 5.238 | 0.058 | -0.097 | 3.444 | 0.149 | 0.321 | PCA worse |
| OldHospital | 4.988 | 0.121 | 0.134 | 4.186 | 0.181 | 0.316 | PCA worse |
| ShopFacade | 2.344 | 0.136 | 0.089 | 1.649 | 0.320 | 0.399 | PCA worse |
| StMarysChurch | 5.078 | 0.079 | -0.037 | 3.746 | 0.174 | 0.269 | PCA worse |

Artifacts:

- `result/result/feature_extract/reference_pose_feature_ranking/greatcourt_query_student_fine_pca64_top10_20260522.json`
- `result/result/feature_extract/reference_pose_feature_ranking/kingscollege_query_student_fine_pca64_top10_20260522.json`
- `result/result/feature_extract/reference_pose_feature_ranking/oldhospital_query_adaptive_fine_pca64_top10_20260522.json`
- `result/result/feature_extract/reference_pose_feature_ranking/shopfacade_query_student_fine_pca64_top10_20260522.json`
- `result/result/feature_extract/reference_pose_feature_ranking/stmaryschurch_query_student_fine_pca64_top10_20260522.json`

Interpretation:

- PCA-64 is now a reproducible compression baseline.
- It does not solve public reference-pose reranking.  This strengthens the
  negative evidence that simple compact descriptor similarity is not enough.
- The public positive path still needs rendered-map hypothesis evidence or a
  learned localizability scorer, not global descriptor compression.

### ShopFacade Mapability Separability

Reran ShopFacade fine mapability with track separability:

| feature | used images | missing features | dim | track variance | between distance | separability ratio |
|---|---:|---:|---:|---:|---:|---:|
| raw RADIO fine | 80 | 0 | 64 | 0.08199 | 10.8966 | 132.90 |
| query-adaptive fine | 80 | 138 | 96 | 0.00236 | 0.6071 | 256.81 |

Artifacts:

- `result/result/feature_extract/pofd_fs_mapability/shopfacade_raw_radio_finegeo_tracks80_separability_20260522.json`
- `result/result/feature_extract/pofd_fs_mapability/shopfacade_query_adaptive_finegeo_tracks80_separability_20260522.json`

Interpretation:

- Query-adaptive features retain much lower within-track variance.
- Separability ratio is also higher despite lower absolute between-track
  distance, because within-track variance drops far more.
- This supports the "mapable feature" claim, but still remains a diagnostic
  result rather than a public reranking win.

### Handoff Protocol Rerun

Reran solver-free handoff selection over the q50 val128 candidate table:

- `result/result/feature_extract/pofd_fs_handoff_q50_val128_pnp_cached_rerun_20260522/summary.json`

Key rows:

| topK | selection | cost mean | trans mean | success@25cm/10deg |
|---:|---|---:|---:|---:|
| 1 | POFD score | 0.2227 | 0.2176 | 0.7422 |
| 4 | PnP inliers | 0.2995 | 0.2946 | 0.5703 |
| 8 | oracle pose | 0.1294 | 0.1250 | 1.0000 |
| 16 | PnP reproj median | 0.4630 | 0.4459 | 0.1953 |

Reran the real-pose cache report:

- `result/result/feature_extract/pofd_fs_solver_handoff_report_rerun_20260522/summary.json`
- `result/result/feature_extract/pofd_fs_solver_handoff_report_rerun_20260522/report.md`

| cache | trans median | trans mean | R@5deg/250mm |
|---|---:|---:|---:|
| POFD top1 identity | 125.0 mm | 217.6 mm | 73.4 |
| POFD top4 PnP-quality identity | 125.0 mm | 294.6 mm | 54.7 |

Interpretation:

- Identity POFD top1 remains the clean solver-free protocol.
- PnP-quality selection inside the POFD topK hurts this q50 controlled cache.
- Render-LoFTR/PNP and PnP-quality handoff should remain diagnostic baselines,
  not the primary claim.

### Selector Group-Gate vs Identity/Utility Probe

Started two GPU selector-stream runs on both GPUs, then stopped at step 100
after both were clearly below the existing q50 identity/utility evidence.

Artifacts:

- `result/result/feature_extract/pofd_fs_selector_groupgate_q50_s200_20260522/`
- `result/result/feature_extract/pofd_fs_selector_identity_utility_q50_s200_20260522/`

| run | best observed step | pred | top1 | gap | Spearman | basin@1 | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| group-gate trainable | 100 | 0.2757 | 0.6094 | 0.1463 | 0.7192 | 0.6094 | worse than baseline |
| identity / utility-only | 25-100 | 0.2835 | 0.5781 | 0.1542 | 0.7185 | 0.5938 | flat, worse |

Interpretation:

- The current streaming selector setup does not improve q50 ranking.
- Group-gate is slightly better than identity/utility in this run, but still
  far below the controlled pair-matcher/calibrator path.
- The selector claim should not be promoted until trained against the stronger
  pair-matcher evidence or a rendered-map localizability score, rather than
  this local-corr stream probe.

## Next Automatic Task Allocation After Round 3

1. Build the public positive path around rendered-map hypothesis evidence:
   reuse reference-pose banks, but score rendered/query evidence instead of
   global descriptors/PCA.
2. Add a report aggregator for public baselines so retrieval-order, descriptor,
   patch, PCA, and future POFD-FS rows are summarized from JSON without manual
   table assembly.
3. Keep identity top1 as the official solver-free handoff protocol; demote
   PnP-quality and render-LoFTR to diagnostic baselines unless a fixed external
   solver beats identity without hurting median translation.
4. Revisit selector training only after the scorer target is upgraded.  The
   current local-corr stream probe is a negative ablation, not a promotion path.

## Round 4 - Guarded Refinement and Selected-Feature Map Stitching

### Code Delivered

Added a guarded refinement policy and cache exporter:

- `feature_extract/localizability/refinement_policy.py`
- `feature_extract/tools/export_guarded_refinement_cache.py`
- `tests/test_refinement_policy.py`

Added the selected-feature map stitching interfaces requested for the next
pipeline stage:

- `feature_extract/localizability/selected_feature_map.py`
- `tests/test_selected_feature_map.py`

The new selected-feature module now covers the six requested links:

1. Localization pose cost to feature utility:
   `localization_feature_utility`.
2. Hard hypothesis ranking targets:
   `hard_hypothesis_selection_targets`, plus existing
   `online_score_hard_negative_loss`.
3. Explicit 3D track / visibility / geometry filtering:
   `aggregate_selected_track_features(..., visibility=..., geometry_valid=...)`.
4. Selected observations to 3D map feature bank:
   `SelectedTrackFeatureBank`.
5. Same selector for query and rendered selected map feature scoring:
   `score_query_with_selected_map_features`.
6. Weak joint selector + map adapter + scorer objective:
   `weak_joint_selected_feature_loss`.

### Guarded Refinement Experiment

Built guarded render-LoFTR refinement caches from the existing OldHospital q50
POFD top1 handoff and top4-PnP-inlier refinement outputs.

Artifacts:

- `result/result/feature_extract/pofd_fs_handoff_solver/oldhospital_q50_top4pnpinliers_guarded_renderloftr_final_i1500_d075_20260522.npz`
- `result/result/feature_extract/pofd_fs_handoff_solver/oldhospital_q50_top4pnpinliers_guarded_renderloftr_final_i1500_d075_20260522.diagnostics.json`
- `result/result/feature_extract/pofd_fs_solver_handoff_guarded_refine_20260522/final_summary.json`
- `result/result/feature_extract/pofd_fs_solver_handoff_guarded_refine_20260522/final_report.md`
- `result/result/feature_extract/pofd_fs_solver_handoff_guarded_refine_20260522/top4_guard_sweep.json`

Best guard found in the small threshold sweep:

- refinement source: `oldhospital_q50_top4pnpinliers_renderloftr_full128_20260521.npz`
- `min_inliers=1500`
- `max_delta_trans_m=0.75`
- `max_delta_rot_deg=5.0`
- accepted refined poses: 87 / 128
- identity fallbacks: 41 / 128

Final report:

| cache | rot median | trans median | trans mean | R@1deg/100mm | R@5deg/250mm |
|---|---:|---:|---:|---:|---:|
| POFD top1 identity | 2.500 deg | 125.0 mm | 217.6 mm | 0.0 | 73.4 |
| top4pnp render-LoFTR | 0.190 deg | 146.0 mm | 218.7 mm | 20.3 | 75.8 |
| final guarded top4pnp | 0.223 deg | 129.4 mm | 183.9 mm | 19.5 | 82.0 |

Interpretation:

- Direct render-LoFTR improves rotation sharply but leaves translation median
  worse than POFD top1.
- The guarded final cache keeps the rotation gain, improves translation mean,
  and raises the broad localization recall from 73.4 to 82.0 R@5deg/250mm.
- This is the current best practical final定位 cache for OldHospital q50 val128
  among the tested handoff/refinement variants.

### Verification

- `python -m py_compile feature_extract/localizability/refinement_policy.py feature_extract/localizability/selected_feature_map.py feature_extract/tools/export_guarded_refinement_cache.py`
- `pytest -q tests/test_refinement_policy.py tests/test_selected_feature_map.py tests/test_localizability_core.py`

## Next Automatic Task Allocation After Round 4

1. Add a selector-aware COLMAP track-mapability tool that applies a trained
   `LocalizationFeatureSelector` before sampling observations, then writes a
   persistent `SelectedTrackFeatureBank`.
2. Feed the persisted selected 3D map feature bank into a rendered-map scoring
   smoke path so query and rendered features use the exact same selector.
3. Add a tiny joint-training smoke test where selector, map adapter, and learned
   scorer all receive gradients through `weak_joint_selected_feature_loss`.
4. Promote the final guarded top4pnp cache as the current practical localization
   handoff output, while keeping POFD top1 identity as the clean solver-free
   baseline.

## Round 5 - Persisted Track Map Feature Bank and Weak-Joint Smoke Path

### Code Delivered

Extended `feature_extract/localizability/selected_feature_map.py` with:

- `save_selected_track_feature_bank`
- `load_selected_track_feature_bank`
- optional `map_adapter` support in `score_query_with_selected_map_features`

Extended `feature_extract/tools/eval_feature_track_mapability.py` so the real
COLMAP track audit can now:

- optionally load a `LocalizationFeatureSelector` checkpoint,
- apply the selector before 2D observation sampling,
- aggregate selected observations by explicit COLMAP track id,
- save a persistent selected 3D map feature bank as NPZ.

Added tests for:

- selected map bank NPZ round-trip,
- selector + map adapter + scorer gradients through the weak joint loss.

### Real Track-Bank Smoke Run

Ran a real OldHospital COLMAP track aggregation smoke test from existing raw
RADIO dense features:

- `result/result/feature_extract/pofd_fs_mapability/oldhospital_raw_radio_track_bank_smoke_20260522.json`
- `result/result/feature_extract/pofd_fs_mapability/oldhospital_raw_radio_track_bank_smoke_20260522.npz`

Smoke summary:

| used images | missing features | observations | tracks | feature dim | track variance | separability |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0 | 323 | 143 | 64 | 0.00313 | 1678.46 |

Interpretation:

- The requested 3D track / visibility / geometry stage now produces a concrete
  map-feature artifact, not just a diagnostic scalar.
- Query/render scoring and weak joint optimization now have a stable interface
  to consume the same selected-map representation.

### Verification

- `python -m py_compile feature_extract/localizability/selected_feature_map.py feature_extract/tools/eval_feature_track_mapability.py`
- `pytest -q tests/test_selected_feature_map.py tests/test_localizability_tools.py`

## Next Automatic Task Allocation After Round 5

1. Replace the smoke raw feature bank with a selector-checkpoint bank once the
   preferred selector checkpoint is chosen.
2. Add a rendered-bank scorer smoke that renders or projects selected 3D map
   features into candidate hypotheses and feeds them through
   `score_query_with_selected_map_features`.
3. Keep using the final guarded top4pnp cache as the current practical final
   localization output for OldHospital q50 val128.

## Round 6 - Protocol Controls, Hard Cases, and Projected Selected-Map Scoring

### Code Delivered

Implemented the protocol and anti-shortcut interfaces needed to turn POFD-FS
from a plain hypothesis-ranking experiment into a protocol-aware localization
evidence benchmark:

- `feature_extract/localizability/protocol.py`
  - protocol labels: `controlled_lattice`, `reference_pose`, `real_retrieval`
  - GT-usage, input-field, solver-conditioned, and deployment-claim metadata
  - claim-mixing guard so controlled and real retrieval rows are not silently
    summarized as one deployment claim
- `feature_extract/localizability/controls.py`
  - metadata-only baselines for candidate rank, retrieval score, PnP inliers,
    reprojection median, delta pose, and score margin
  - query/candidate feature shuffle controls
  - channel and spatial counterfactual masking for high-/low-utility evidence
- `feature_extract/localizability/hard_cases.py`
  - hard-case masks for score-top1 false accepts, retrieval-top1-wrong with
    topK basin available, near-identity false positives, and PnP-high-score
    wrong candidates
- `feature_extract/localizability/rendered_map_scoring.py`
  - projects a persistent `SelectedTrackFeatureBank` into candidate pose
    feature maps using explicit 3D track xyz, camera intrinsics, and w2c poses
  - calls the same selector/scorer path used by query/rendered selected map
    features
- `feature_extract/localizability/reporting.py`
  - protocol-aware summary rows and markdown tables with deployment-claim
    markers

Added coverage in:

- `tests/test_localizability_protocol_controls.py`

### Verification

- `pytest -q tests/test_localizability_protocol_controls.py`
- `pytest -q tests/test_localizability_protocol_controls.py tests/test_selected_feature_map.py tests/test_localizability_core.py tests/test_localizability_tools.py tests/test_refinement_policy.py`
- `python -m py_compile feature_extract/localizability/protocol.py feature_extract/localizability/controls.py feature_extract/localizability/hard_cases.py feature_extract/localizability/rendered_map_scoring.py feature_extract/localizability/reporting.py feature_extract/localizability/__init__.py`

### Next Automatic Task Allocation After Round 6

1. Run metadata-only baselines on OldHospital q50, ShopFacade q50, and the
   real retrieval top20 candidate table; require POFD to beat these before any
   feature-selection claim is promoted.
2. Run feature-shuffle and high-/low-utility counterfactual reports on the
   strongest selected-feature checkpoint once selected dense caches are
   exported.
3. Replace the projected-map scoring unit smoke with a real selected
   `SelectedTrackFeatureBank` generated from selector outputs, then compare
   rendered-map ranking retention against the 2D selected-feature scorer.

## Round 7 - Anti-Shortcut Controls, Hard-Case Subsets, and Selector Track Banks

### Code Delivered

Added protocol-control tooling:

- `feature_extract/tools/eval_localizability_protocol_controls.py`
  - evaluates POFD score tables against metadata-only controls,
  - exports JSON and markdown protocol summaries,
  - records `controlled_lattice` vs `real_retrieval` deployment-claim status,
  - warns when `score_rank` would leak model output into candidate-rank
    baselines,
  - warns when `retrieval_score` is oracle-like under GT-centered controlled
    protocols.
- `feature_extract/tools/build_localizability_hard_cases.py`
  - exports full candidate-table JSONL subsets for score-top1 false accepts,
    retrieval-top1-wrong/topK-basin, near-identity false positives, and
    PnP-high-score wrong cases.

Extended `feature_extract/tools/eval_feature_track_mapability.py`:

- carries COLMAP `xyz` into the persisted `SelectedTrackFeatureBank`,
- adds `lookup_point_xyz` to align observations to `points3D.bin`,
- loads local selector checkpoints that are not compatible with
  `weights_only=True`,
- infers whether checkpoint state contains an uncertainty head before loading.

### Protocol-Control Results

Artifacts:

- `result/result/feature_extract/pofd_fs_protocol_controls_20260523/oldhospital_q50_val128_controls.json`
- `result/result/feature_extract/pofd_fs_protocol_controls_20260523/shopfacade_q50_val103_controls.json`
- `result/result/feature_extract/pofd_fs_protocol_controls_20260523/oldhospital_real_top20_controls.json`

| protocol | POFD pred | POFD top1 | POFD Spearman | candidate-rank pred/top1 | retrieval-score note |
|---|---:|---:|---:|---:|---|
| OldHospital q50 controlled val128 | 0.2227 | 0.7188 | 0.5864 | 0.5167 / 0.0313 | oracle-like; not clean |
| ShopFacade q50 controlled val103 | 0.3376 | 0.4175 | 0.3879 | 0.5168 / 0.0000 | oracle-like; not clean |
| OldHospital real retrieval full182 | 0.3360 | 0.2033 | 0.5062 | 0.4302 / 0.1154 | clean baseline: 0.4006 / 0.0824 |

Interpretation:

- The real retrieval result now has a clean anti-shortcut baseline: POFD beats
  candidate order, retrieval score, PnP inliers, and reprojection median on
  top1 and median predicted cost.
- Controlled q50 still cannot be described as deployment accuracy. The tool now
  explicitly marks it non-deployable and flags the GT-centered retrieval-score
  shortcut.

### Hard-Case Subsets

Artifacts:

- `result/result/feature_extract/pofd_fs_hard_cases_20260523/oldhospital_real_top20/`
- `result/result/feature_extract/pofd_fs_hard_cases_20260523/oldhospital_q50_val128/`
- `result/result/feature_extract/pofd_fs_hard_cases_20260523/shopfacade_q50_val103/`

Real retrieval full182 hard cases:

| case | queries | fraction |
|---|---:|---:|
| score-top1 false accept | 45 | 0.2473 |
| retrieval-top1 wrong but topK basin exists | 50 | 0.2747 |
| near-identity false positive | 9 | 0.0495 |
| PnP-high-score wrong | 54 | 0.2967 |
| has basin candidate | 150 | 0.8242 |

### Selected 3D Mapability Results

Artifacts:

- `result/result/feature_extract/pofd_fs_mapability/oldhospital_raw_finegeo_tracks80_20260523.json`
- `result/result/feature_extract/pofd_fs_mapability/oldhospital_selected_qstudent_selector_finegeo_tracks80_20260523.json`
- `result/result/feature_extract/pofd_fs_mapability/oldhospital_selected_utilityonly_selector_finegeo_tracks80_20260523.json`
- matching `.npz` track banks with `xyz`

| feature | tracks | observations | track variance | separability ratio | bank xyz |
|---|---:|---:|---:|---:|---|
| raw fine_geo | 19345 | 45109 | 0.006712 | 658.50 | yes |
| selected qstudent | 19345 | 45109 | 0.000274 | 639.52 | yes |
| selected utility-only | 19345 | 45109 | 0.000350 | 658.41 | yes |

Interpretation:

- Selector checkpoints now generate persistent selected 3D track banks with
  explicit `xyz`, so projected rendered-map scoring can consume them.
- Selected features substantially reduce within-track variance. The
  utility-only selector keeps the raw separability trend while cutting variance,
  which is a positive mapability signal.
- A real projection smoke using the utility-only bank rendered
  `(1, 1, 64, 272, 480)` features with `13205` valid pixels in the first
  OldHospital COLMAP view.

### Verification

- `pytest -q tests/test_localizability_protocol_controls.py::test_protocol_controls_candidate_rank_uses_candidate_idx_not_model_score_rank tests/test_localizability_protocol_controls.py::test_protocol_controls_summary_compares_pofd_to_available_metadata_baselines tests/test_localizability_protocol_controls.py::test_metadata_baseline_scores_support_rank_inliers_reprojection_and_delta_pose`
- `pytest -q tests/test_localizability_protocol_controls.py::test_protocol_controls_warns_when_controlled_retrieval_score_is_oracle_like`
- `pytest -q tests/test_localizability_protocol_controls.py::test_build_hard_case_subset_rows_exports_full_candidate_groups`
- `pytest -q tests/test_localizability_tools.py::test_lookup_point_xyz_aligns_observation_tracks_to_colmap_points`

## Next Automatic Task Allocation After Round 7

1. Use the exported hard-case subsets for downstream identity POFD, guarded
   refinement, and fixed-solver handoff reports.
2. Launch Phase 2 selector retraining seeds on OldHospital and ShopFacade once
   the existing script flags are confirmed; gate checkpoints on Spearman and
   metadata-control margins, not just top1.
3. Add selected-feature counterfactual experiments using high-/low-utility
   channel and spatial masks.
4. Run rendered selected-map ranking retention against the 2D selected-feature
   scorer using the new `xyz` track banks.

## Round 8 - Phase 2 Selector+r16 Scorer Wiring and Hard-Case Handoff Reports

### Code Delivered

Extended `feature_extract/tools/train_localizability_selector_stream.py` so the
Phase 2 mainline can actually train a 64-d selector through a frozen
`pair_matcher_local` backend:

- added `--pose-adapter-checkpoint` / `--pose-adapter-strict`,
- added pair-matcher runtime knobs for stride, point chunking, candidate
  chunking, offset chunking, score mode, and score channel,
- added `_build_selector_stream_scorer`,
- require a loaded pair matcher when `--score-mode pair_matcher_local`,
- load the pair matcher from the existing pose-adapter checkpoint and keep it
  frozen while selector parameters receive gradients.

Extended hard-case downstream reporting:

- `feature_extract/localizability/pose_cache_report.py`
  - added `query_names_from_candidate_table` for hard-case JSONL subsets.
- `feature_extract/tools/report_solver_handoff_localization.py`
  - added `--query-table` so handoff reports can evaluate exactly the exported
    hard-case subsets.

### Hard-Case Downstream Results

Artifacts:

- `result/result/feature_extract/pofd_fs_hardcase_reports_20260523/oldhospital_real_score_top1_false_accept_report.json`
- `result/result/feature_extract/pofd_fs_hardcase_reports_20260523/oldhospital_real_retrieval_top1_wrong_topk_basin_report.json`

On real retrieval queries where retrieval top1 is wrong but topK contains a
basin candidate (`n=50`):

| method | trans median mm | R@5deg/250mm |
|---|---:|---:|
| real retrieval top20-quality | 415.3 | 0.0 |
| POFD score top1 | 309.2 | 28.0 |
| POFD top1 render-LoFTR | 274.5 | 44.0 |
| POFD top4 PnP-inliers | 283.4 | 30.0 |
| POFD top4 render-LoFTR | 272.4 | 48.0 |
| HLoc SP+SG | 231.8 | 54.0 |
| HLoc render-LoFTR | 258.9 | 38.0 |

Interpretation:

- This is a positive hard-case result for the new project framing: POFD
  evidence rescues a subset where the real retrieval baseline has zero
  `R@5deg/250mm`.
- POFD top4 render-LoFTR beats HLoc render-LoFTR on this subset, but still does
  not beat HLoc SP+SG. The claim remains hard-case verification utility, not
  global SOTA final localization.
- On POFD's own score-top1 false-accept subset (`n=45`), identity POFD is bad by
  construction and render-LoFTR only partially recovers it; this subset is now
  explicitly used as a failure/risk diagnostic rather than a success claim.

### Phase 2 Selector+r16 Smoke and Short Seeds

Smoke artifacts:

- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_smoke_20260523/`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_smoke_20260523/`

Short dual-GPU seed artifacts:

- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_oh_s20260523/`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_s20260523/`

OldHospital q50 eval64, frozen r16 scorer, 40 steps:

| step | pred | top1 | gap | Spearman | basin@5 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.1901 | 0.8281 | 0.0607 | 0.6130 | 1.0000 |
| 20 | 0.1901 | 0.8281 | 0.0607 | 0.6130 | 1.0000 |
| 30 | 0.1901 | 0.8281 | 0.0607 | 0.6128 | 1.0000 |
| 40 | 0.1901 | 0.8281 | 0.0607 | 0.6128 | 1.0000 |

This passes the OldHospital controlled q50 promotion gate on the eval64 short
run (`pred<=0.21`, `top1>=0.75`, `gap<=0.09`, `Spearman>=0.55`).

ShopFacade q50 eval64, same recipe, 40 steps:

| step | pred | top1 | gap | Spearman | basin@5 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.4197 | 0.2031 | 0.2903 | 0.2292 | 0.7031 |
| 20 | 0.4197 | 0.2031 | 0.2903 | 0.2302 | 0.7031 |
| 30 | 0.4197 | 0.2031 | 0.2903 | 0.2314 | 0.7031 |
| 40 | 0.4197 | 0.2031 | 0.2903 | 0.2345 | 0.7188 |

Interpretation:

- OldHospital has a clear positive signal under the new selector+r16 mainline.
- ShopFacade does not transfer with the same hyperparameters and needs a
  separate recipe or stronger scene-specific initialization.
- The r16 pair matcher uses about 20GB on OldHospital, so `batch_size=1` is the
  practical stable setting for this backend on a 24GB RTX 3090.

### Verification

- `pytest -q tests/test_localizability_tools.py::test_selector_stream_pair_matcher_mode_requires_loaded_pair_matcher`
- `pytest -q tests/test_localizability_tools.py::test_query_names_from_candidate_table_deduplicates_in_order`
- `pytest -q tests/test_localizability_tools.py::test_selector_stream_pair_matcher_mode_requires_loaded_pair_matcher tests/test_localizability_tools.py::test_query_names_from_candidate_table_deduplicates_in_order tests/test_localizability_protocol_controls.py`
- `python -m py_compile feature_extract/tools/train_localizability_selector_stream.py feature_extract/tools/report_solver_handoff_localization.py feature_extract/localizability/pose_cache_report.py`

## Next Automatic Task Allocation After Round 8

1. Promote OldHospital selector+r16 to a 5-seed controlled run and evaluate
   full val128, then compare against metadata-only controls and real retrieval
   transfer.
2. For ShopFacade, sweep a scene-specific recipe: lower LR, utility-only
   selector, frozen projection/channel gate, and/or smaller radius before
   claiming selector transfer.
3. Export selected dense/utility maps from the best OldHospital selector+r16
   checkpoint and run high-/low-utility channel and spatial counterfactuals.
4. Use the selected `xyz` track banks to run rendered selected-map ranking
   retention against the 2D selected scorer.

## Round 9 - Full-Val Selector Audit, Counterfactuals, and ShopFacade Sweeps

### Code Delivered

Added a seed/run summary helper:

- `feature_extract/tools/summarize_selector_seed_runs.py`
- `tests/test_selector_seed_summary.py`

Added selector counterfactual audits:

- `feature_extract/tools/eval_localizability_selector_audits.py`
- `tests/test_selector_audits.py`

The audit CLI now supports:

- spatial high-/low-utility removal over scorer score maps,
- channel-group high-/low-importance removal by recomputing hypothesis scores,
- optional selected feature return from
  `feature_extract/tools/train_localizability_selector_stream.py` only when
  audits request it.

### OldHospital Full-Val Selector+r16 Audit

Artifacts:

- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_oh_seed_summary_20260523.json`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_oh_fullval_s20260524/`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_oh_fullval_s20260525/`

Summary over the previous eval64 short seed plus two full-val128 short seeds:

| runs | pass fraction | pred | top1 | gap | Spearman | basin@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1/3 | 0.2180 | 0.7292 | 0.0886 | 0.5963 | 0.9948 |

Full val128 seeds both landed at:

| pred | top1 | gap | Spearman | basin@5 |
|---:|---:|---:|---:|---:|
| 0.2319 | 0.6797 | 0.1026 | 0.5876 | 0.9922 |

Interpretation:

- The eval64 result from Round 8 was a real positive smoke signal, but it was
  over-optimistic.
- OldHospital full-val128 does not pass the current promotion gate
  (`pred<=0.21`, `top1>=0.75`, `gap<=0.09`, `Spearman>=0.55`), although
  Spearman remains above gate and basin@5 is strong.
- The correct claim is now: controlled q50 selector+r16 has partial predictive
  utility, not a frozen promotion result.

### Selector Counterfactual Audit

Artifact:

- `result/result/feature_extract/pofd_fs_audits/oldhospital_pm_r16_spatial_channel_counterfactual_max8_20260523.json`

OldHospital q50 max8, selected r16 checkpoint:

| audit | base pred | drop high pred | drop low pred | high delta | low delta |
|---|---:|---:|---:|---:|---:|
| spatial utility | 0.1294 | 0.1294 | 0.2229 | 0.0000 | 0.0936 |
| channel gate | 0.1294 | 0.1294 | 0.1762 | 0.0000 | 0.0468 |

Interpretation:

- This is a negative Gate 2 result. Removing high-utility/high-gate evidence
  did not hurt ranking on this audit subset; removing low-utility/low-gate
  evidence hurt more.
- The selector's current utility/gate heads cannot yet be used as causal
  evidence for "selected localization features".
- Until this is fixed, the main claim should stay at predictive verification
  evidence plus mapability diagnostics, not interpretable causal selection.

### Selected 3D Track Bank Mapability

Artifact:

- `result/result/feature_extract/pofd_fs_mapability/oldhospital_selected_pm_r16_tracks80_20260523.json`
- `result/result/feature_extract/pofd_fs_mapability/oldhospital_selected_pm_r16_track_bank_tracks80_20260523.npz`

OldHospital selected r16 track-bank export:

| images | observations | tracks | track variance | between distance | separability | xyz |
|---:|---:|---:|---:|---:|---:|---|
| 80 | 13903 | 6350 | 0.000325 | 0.2301 | 708.16 | yes |

Interpretation:

- This is a positive mapability smoke for the selected r16 feature space.
- The persistent bank now carries `xyz`, so rendered/projected selected-map
  scoring can be implemented without falling back to a 2D-only cache claim.
- The next missing proof is rendered-map ranking retention against the same
  query selector/scorer.

### ShopFacade Sweep

Artifacts:

- `result/result/feature_extract/pofd_fs_phase2_shop_selector_sweep_summary_20260523.json`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_s20260523/`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_utilonly_lr5e6_s80_20260523/`
- `result/result/feature_extract/pofd_fs_phase2_selector_localcorr_r4_shop_gateutil_lr1e5_s80_20260523/`
- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_pairflow_s40_20260523/`

Gate used for this sweep was "beat the current raw/first64 ShopFacade
diagnostic" (`pred<0.3506`, `top1>0.375`, `gap<0.22`, `Spearman>0.410`).

| variant | pred | top1 | gap | Spearman | basin@5 | pass |
|---|---:|---:|---:|---:|---:|---|
| r16 selector baseline | 0.4197 | 0.2031 | 0.2903 | 0.2292 | 0.7031 | no |
| r16 utility-only | 0.4197 | 0.2031 | 0.2903 | 0.2278 | 0.7031 | no |
| local-corr r4 gate+utility | 0.5410 | 0.2031 | 0.4117 | -0.0462 | 0.4688 | no |
| pair-flow r16 gate+utility | 0.4018 | 0.2188 | 0.2724 | 0.2378 | 0.7031 | no |

Interpretation:

- ShopFacade does not currently support a cross-scene selector claim.
- Pair-flow r16 is the least bad of the failed sweeps, but it still misses the
  gate by a large margin.
- The next ShopFacade work should shift from light LR sweeps to protocol/data
  checks: feature source alignment, candidate label distribution, and whether
  the q50 controlled bank has enough visual evidence for the selected scorer.

### Verification

- `pytest -q tests/test_selector_seed_summary.py`
- `pytest -q tests/test_selector_audits.py`
- `python -m py_compile feature_extract/tools/eval_localizability_selector_audits.py feature_extract/tools/train_localizability_selector_stream.py feature_extract/tools/summarize_selector_seed_runs.py`

## Next Automatic Task Allocation After Round 9

1. Implement rendered selected-map scoring retention using the selected track
   bank with `xyz`, and compare it against the 2D selected-feature scorer.
2. Redesign selector utility supervision so the high/low utility and channel
   counterfactuals become meaningful; current entropy-only utility is not
   sufficient.
3. Run OldHospital full-val with stronger utility losses or detached scorer
   evidence maps, then rerun the spatial/channel counterfactual audit.
4. Diagnose ShopFacade before more tuning: report candidate-cost histograms,
   oracle basin distribution, feature-source mismatch, and metadata-only vs
   feature evidence on the same q50 bank.

## Round 10 - Candidate-Generator Geometry Diagnostics

### Code Delivered

Added candidate-cache geometry diagnostics:

- `feature_extract/localizability/pose_cache_report.py`
  - `summarize_pose_candidate_cache_geometry`
  - `pose_candidate_cache_diagnostics_to_markdown`
- `feature_extract/tools/report_pose_candidate_cache_diagnostics.py`
- `tests/test_localizability_tools.py::test_summarize_pose_candidate_cache_geometry_reports_oracle_basin`

The diagnostic reports candidate generator quality without learned scorer
outputs:

- top1 pose distribution by original cache order,
- oracle pose distribution over the candidate set,
- basin recall by original cache order,
- oracle rank in cache order,
- metadata fields available in the cache.

### OldHospital q50 Candidate Bank

Artifact:

- `result/result/feature_extract/pofd_fs_candidate_diagnostics/oldhospital_q50_val128_20260523.json`

| n | K | top1 trans q50 | oracle trans q50 | oracle cost mean | basin@1 | basin@5 | basin@10 | basin@16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 499.4 mm | 125.0 mm | 0.1294 m | 0.109 | 0.563 | 0.883 | 1.000 |

Oracle rank by cache order has median 9 and q90 15. This validates the shuffled
OldHospital q50 bank as a genuine hard ranking problem: the correct candidate is
not mostly encoded by a fixed small index.

### ShopFacade q50 Candidate Bank

Artifact:

- `result/result/feature_extract/pofd_fs_candidate_diagnostics/shopfacade_q50_val103_20260523.json`

| n | K | top1 trans q50 | oracle trans q50 | oracle cost mean | basin@1 | basin@5 | basin@10 | basin@16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 103 | 16 | 499.4 mm | 125.0 mm | 0.1294 m | 0.000 | 1.000 | 1.000 | 1.000 |

Oracle rank by cache order has median 2 and q90 2. This is a protocol warning:
ShopFacade q50 val103 is not a good cross-scene proof bank in its current form,
because the correct/basin candidate is almost completely determined by cache
construction order. POFD-FS should not use this bank alone to claim learned
localization evidence.

### Interpretation

- The OldHospital shuffled q50 bank remains useful for controlled hard ranking.
- The ShopFacade q50 bank explains why lightweight selector sweeps are
  inconclusive: the bank has a very strong order prior and top5 oracle is
  saturated.
- Next ShopFacade evidence should use a shuffled candidate bank, real retrieval,
  or an explicitly order-balanced hard-negative export before more selector
  training.

### Verification

- `pytest -q tests/test_localizability_tools.py::test_append_near_identity_candidate_arrays_extends_candidate_axis tests/test_localizability_tools.py::test_summarize_pose_candidate_cache_geometry_reports_oracle_basin`
- `python -m py_compile feature_extract/tools/report_pose_candidate_cache_diagnostics.py feature_extract/localizability/pose_cache_report.py`

## Next Automatic Task Allocation After Round 10

1. Create an order-balanced/shuffled ShopFacade candidate cache and rerun
   metadata-only, selected scorer, and selector training on that protocol.
2. Implement rendered selected-map ranking retention from the `xyz` selected
   track bank.
3. Replace entropy-only utility training with explicit evidence supervision
   from candidate-wise score contribution maps, then rerun high/low
   spatial/channel counterfactuals.
4. Keep final claims separated into protocol categories:
   `controlled_lattice`, `reference_pose`, and `real_retrieval`.

## Round 11 - Balanced ShopFacade Protocol and Projected Selected-Map Retention

### Code Delivered

Extended candidate-cache shuffle tooling:

- `feature_extract/tools/shuffle_pose_candidate_cache.py`
  - added `permute_pose_candidate_cache_arrays`,
  - switched default generated permutations to
    `balanced_cyclic_random_base`,
  - composes existing `candidate_permutation` provenance when present,
  - writes `candidate_order_permutation` plus shuffle metadata.
- `tests/test_localizability_tools.py`
  - added metadata-alignment and balanced-position tests.

Added projected selected-track bank retention tooling:

- `feature_extract/tools/eval_projected_selected_track_bank_retention.py`
  - converts map-renderer intrinsics to scorer-grid `K`,
  - projects an `xyz` selected track bank into candidate views,
  - scores query selected features against already-selected projected map
    features without applying the selector to the bank a second time,
  - reports ranking metrics and projected-map coverage.
- `tests/test_projected_selected_track_bank_retention.py`

### ShopFacade Balanced q50 Candidate Cache

Artifacts:

- `result/result/feature_extract/pose_init_exports/shopfacade_local_lattice_renderloftr_q50cm10deg_top16_noexact_val103_balancedshuf20260523.npz`
- `result/result/feature_extract/pofd_fs_candidate_diagnostics/shopfacade_q50_val103_original_vs_balancedshuf_20260523.json`

Permutation sanity:

- shape: `103 x 16`
- every row is a full candidate permutation,
- global rank/original-index counts are balanced: min `6`, max `7`.

Geometry diagnostic:

| cache | oracle cost | oracle rank q50 | basin@1 | basin@5 | basin@10 | basin@16 |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.1294 | 2 | 0.000 | 1.000 | 1.000 | 1.000 |
| balanced shuffled | 0.1294 | 8 | 0.126 | 0.631 | 0.922 | 1.000 |

Interpretation:

- The selected candidate set is unchanged, so oracle difficulty is preserved.
- The original ShopFacade q50 order prior is removed; the balanced cache is a
  better protocol for ranking evidence than the previous val103 cache.

### ShopFacade Balanced Selector Smoke

Artifact:

- `result/result/feature_extract/pofd_fs_phase2_selector_pm_r16_shop_pairflow_balancedshuf_s20_20260523/`

Pair-flow r16 selector, eval max32 on balanced-shuffled q50:

| step | pred | top1 | gap | Spearman | basin@1 | basin@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.3444 | 0.2813 | 0.2151 | 0.3176 | 0.4063 | 0.7500 |
| 20 | 0.3444 | 0.2813 | 0.2151 | 0.3176 | 0.4063 | 0.7500 |

Interpretation:

- This is better than the unshuffled ShopFacade selector sweeps on median
  predicted cost, but it still fails the protocol gate because top1 and
  Spearman remain weak.
- ShopFacade remains a diagnostic scene, not a positive cross-scene selector
  claim.

### Projected Selected-Map Retention Smoke

Artifact:

- `result/result/feature_extract/projected_selected_bank_retention/oldhospital_pm_r16_bank_localcorr_max8_20260523/metrics.json`

OldHospital selected r16 track bank projected into q50 candidates, max8:

| pred | top1 | gap | Spearman | basin@1 | basin@5 | valid pixel frac | valid px/candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6093 | 0.1250 | 0.4800 | 0.0827 | 0.1250 | 0.5000 | 0.0081 | 16.6 |

Interpretation:

- The selected 3D bank can now be projected and scored end-to-end, but sparse
  nearest-pixel projection does not retain 2D selected-feature ranking.
- Coverage is the likely bottleneck: only about 0.8% of scorer-grid pixels have
  projected selected tracks.
- Mapability remains positive as a track-bank statistic, but rendered/projected
  selected-map ranking retention is not established. The next method step
  should add splatting/densification and visibility-aware weighting before
  using this as a paper claim.

### Verification

- `pytest -q tests/test_localizability_tools.py::test_permute_pose_candidate_cache_arrays_keeps_candidate_metadata_aligned tests/test_localizability_tools.py::test_permute_pose_candidate_cache_arrays_balances_generated_candidate_positions`
- `pytest -q tests/test_projected_selected_track_bank_retention.py`
- `python -m py_compile feature_extract/tools/shuffle_pose_candidate_cache.py`

## Next Automatic Task Allocation After Round 11

1. Add projected-bank densification/splatting so sparse selected tracks cover
   enough scorer-grid pixels for retention to be meaningful.
2. Evaluate balanced ShopFacade with metadata-only controls and selected scorer
   tables, not just the selector training smoke.
3. Replace entropy-only utility with explicit evidence-map supervision, then
   rerun spatial/channel counterfactuals.
4. Keep negative projected-retention and ShopFacade transfer results in the
   report until a stronger method clears the gates.

## Round 12 - Fast Projected-Map Densification

### Code Delivered

Extended projected selected-map rendering:

- `feature_extract/localizability/rendered_map_scoring.py`
  - added `splat_radius` support to `render_selected_track_feature_maps`,
  - added fast `densify_projected_feature_maps` using local valid-neighbor
    averaging.
- `feature_extract/tools/eval_projected_selected_track_bank_retention.py`
  - added `--splat-radius` and `--densify-radius`.
- `tests/test_projected_selected_track_bank_retention.py`
  - added coverage for splatting and fast densification.

The first direct `splat_radius=2` experiment was interrupted because the
per-track Python neighborhood loop was too slow. The fast densification path
keeps the projection sparse and expands it with pooled local averages, which is
stable enough for repeated sweeps.

### OldHospital Projected Selected-Map Retention Sweep

Artifacts:

- `result/result/feature_extract/projected_selected_bank_retention/oldhospital_pm_r16_bank_localcorr_max8_20260523/metrics.json`
- `result/result/feature_extract/projected_selected_bank_retention/oldhospital_pm_r16_bank_localcorr_densify2_max8_20260523/metrics.json`
- `result/result/feature_extract/projected_selected_bank_retention/oldhospital_pm_r16_bank_localcorr_densify4_max8_20260523/metrics.json`
- `result/result/feature_extract/projected_selected_bank_retention/oldhospital_pm_r16_bank_localcorr_densify6_max8_20260523/metrics.json`

OldHospital q50 max8, selected r16 track bank:

| densify | pred | top1 | gap | Spearman | basin@1 | basin@5 | valid px frac | valid px/cand |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6093 | 0.125 | 0.4800 | 0.0827 | 0.125 | 0.500 | 0.0081 | 16.6 |
| 2 | 0.3420 | 0.250 | 0.2126 | 0.1846 | 0.500 | 0.625 | 0.0210 | 42.8 |
| 4 | 0.3411 | 0.125 | 0.2117 | 0.3625 | 0.375 | 0.750 | 0.0364 | 74.2 |
| 6 | 0.3856 | 0.000 | 0.2563 | 0.2496 | 0.250 | 0.750 | 0.0557 | 113.7 |

Interpretation:

- Densification produces a clear positive retention trend compared with sparse
  nearest-pixel projection: median predicted cost drops by about 27 cm and
  Spearman improves from 0.08 to 0.36 at radius 4.
- This is still not enough to claim selected 3D map scoring is solved. Top1 is
  unstable and much weaker than 2D selected-feature scoring.
- The current best projected-map setting is `densify_radius=4` for ranking
  trend retention, while `densify_radius=2` is best on basin@1/top1 in this
  tiny max8 smoke.
- The next mapability step should replace uniform local averaging with
  geometry-aware splatting: depth-aware weights, visibility confidence, and
  utility-weighted track features.

### Verification

- `pytest -q tests/test_projected_selected_track_bank_retention.py`

## Next Automatic Task Allocation After Round 12

1. Add geometry/utility-weighted projected-bank densification instead of uniform
   neighbor averaging.
2. Run projected-map retention on a larger OldHospital subset once the
   densifier is less noisy.
3. Build balanced ShopFacade selected-score tables and metadata controls.
4. Add explicit evidence-map utility supervision so utility/gate
   counterfactuals can become causal rather than diagnostic.
