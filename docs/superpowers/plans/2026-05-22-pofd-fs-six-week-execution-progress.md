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
