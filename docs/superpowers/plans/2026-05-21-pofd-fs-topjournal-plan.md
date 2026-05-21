# 2026-05-21 POFD-FS Top-Journal Mainline

## Position

The project should be written as a feature-centric localization paper:

> Which foundation features localize?

The main contribution is a benchmark and method for selecting compact,
interpretable, mapable localization subspaces from frozen dense foundation
features.  The primary task is **hypothesis-localizability**:

```text
score(query, map/reference, candidate hypothesis)
```

The paper should not claim SOTA continuous camera pose refinement unless the
POFD-FS path itself improves real-init final pose.  Stage4 continuous CPR,
rendered-RGB LoFTR refinement, WLS/GN, and feature-metric update branches remain
diagnostics or external baselines.

## Architecture

```text
frozen foundation feature
  -> localization feature selector / pose-adapted feature
  -> compact localization feature + utility / uncertainty
  -> hypothesis scorer
  -> pose-distance and basin-aware ranking
```

Current implementation boundary:

```text
feature_extract/localizability/
  selector.py
  scorer.py
  losses.py
  failure_replay.py
  handoff_cache.py
  reference_pose_bank.py
  candidate_bank.py
  metrics.py
  mapability.py
  visualization.py
```

## Current Evidence

OldHospital controlled val128, no external score labels:

| feature / scorer | q10 pred | q25 pred | q50 pred | q50 top1 | q50 gap | q50 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw RADIO + local r4 | 0.086 | 0.189 | 0.300 | 0.508 | 0.171 | 0.664 |
| query student + local r4 | 0.059 | 0.140 | 0.274 | 0.594 | 0.144 | 0.714 |
| pose-adapted + pair matcher r16 | 0.045 | 0.107 | 0.223 | 0.719 | 0.093 | 0.586 |

This supports the POFD-FS claim on q10/q25 and partially on q50.  q50 is close
but does not pass the promotion threshold:

```text
target: pred <= 0.21, top1 >= 0.75, gap <= 0.09, Spearman >= 0.55
current: pred 0.223, top1 0.719, gap 0.093, Spearman 0.586
```

Important interpretation:

- `pred` / `pred_cost_m` is currently the pose error of the top1 ranked
  candidate.  It is not a refined final pose.
- `basin_recall@K` is currently a geometric proxy: whether the topK contains a
  candidate under the configured translation/rotation threshold.
- This proxy is useful for feature-selection analysis, but a final localization
  claim needs a solver-specific handoff protocol.

## Handoff Solver Protocol

The paper should report two separate result types:

1. **Solver-free localizability**
   - final pose is the selected candidate pose;
   - main metrics are ranking metrics: top1/topK, Spearman, NDCG, oracle gap,
     and geometric basin recall;
   - this is the cleanest evidence that the feature subspace localizes
     hypotheses.

2. **Solver-conditioned localization**
   - POFD-FS ranks candidates;
   - a fixed external solver receives top1 or topK candidates;
   - final metrics are the solver's refined pose errors;
   - basin is defined by the actual solver success criterion, not only by a
     hand-picked geometric threshold.

Recommended handoff solvers:

| solver | role | train-free | note |
| --- | --- | --- | --- |
| Top1 identity | minimal baseline | yes | final pose is POFD-FS top1 candidate |
| Existing render-RGB LoFTR+PnP | diagnostic baseline | yes | already implemented; unstable on strong HLoc init |
| GS-SMC-style RGB rendered-view refinement | main handoff candidate | yes | uses existing 3DGS/DCFF RGB render and epipolar/multi-view geometry; should be evaluated as a fixed external solver |
| GS-CPR/MASt3R-style method | strong external baseline | mostly | useful for comparison, not POFD-FS method |

For solver-conditioned tables, basin should be calibrated by measurement:

```text
For each candidate error bucket, run the solver.
Candidate is in basin if final pose improves and lands below the target
translation/rotation threshold without using GT at selection time.
```

Until that table exists, `basin@K` should be described as "geometric basin
proxy", not as guaranteed solver success.

## 2026-05-21 Cached Solver Handoff Result

Implemented a generic handoff evaluator:

```text
feature_extract/localizability/solver_handoff.py
feature_extract/tools/eval_localizability_solver_handoff.py
```

Input:

```text
POFD-FS candidate_table.jsonl
```

Output:

```text
top1 / topK selected final pose metrics
```

For the OldHospital q50 controlled cache, the candidates already include
render-LoFTR/PnP metadata:

```text
retrieval_pnp_success_candidates
retrieval_pnp_num_inliers_candidates
retrieval_pnp_reproj_median_candidates
retrieval_pnp_inlier_ratio_candidates
```

This means the current q50 result should be interpreted as selecting among
cached solver-produced candidates, not as running a new continuous solver after
ranking.

Handoff results on q50 val128:

| topK | selector within POFD topK | mean cost | median trans | success 25cm/10deg |
| ---: | --- | ---: | ---: | ---: |
| 1 | POFD score | 0.2227 | 0.1250 | 0.742 |
| 4 | PNP inliers | 0.2995 | 0.1250 | 0.570 |
| 4 | PNP reproj median | 0.3488 | 0.3747 | 0.453 |
| 4 | oracle within topK | 0.1337 | 0.1250 | 0.977 |
| 8 | PNP inliers | 0.2900 | 0.1250 | 0.578 |
| 8 | PNP reproj median | 0.3745 | 0.3747 | 0.258 |
| 8 | oracle within topK | 0.1294 | 0.1250 | 1.000 |
| 16 | oracle within topK | 0.1294 | 0.1250 | 1.000 |

Conclusion:

- POFD top1 is the current deployable selection result.
- There is strong topK headroom: top8 oracle reaches the cache oracle.
- Existing PNP-quality fields are not reliable handoff selectors inside POFD
  topK.  Inliers and reprojection median make q50 worse.
- The next solver-conditioned improvement should not be a naive PNP-quality
  fallback.  It should be either a learned hypothesis calibrator over richer
  candidate evidence or a genuinely independent RGB/3DGS refinement method
  such as a GS-SMC-style solver.

## 2026-05-21 Explicit Init-Cache Handoff

Implemented a cache export layer:

```text
feature_extract/localizability/handoff_cache.py
feature_extract/tools/export_localizability_handoff_cache.py
feature_extract/configs/pofd_fs_oldhospital_render_handoff.yaml
```

This converts a POFD-FS `candidate_table.jsonl` and the original candidate
pose cache into a standard one-candidate init cache.  It lets the downstream
solver consume exactly the candidate selected by a controlled handoff policy.

Exported q50 val128 init caches:

```text
result/result/feature_extract/pofd_fs_handoff_exports/
  oldhospital_q50_val128_pofd_top1_20260521.npz
  oldhospital_q50_val128_pnpinliers_top4_20260521.npz
  oldhospital_q50_val128_oracle_top8_20260521.npz
```

Ran fixed render-at-init RGB LoFTR+PnP refinement on the exported caches:

| init cache | init mean | init median | refined mean | refined median | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| POFD top1 | 217.6mm | 125.0mm | 219.0mm | 146.0mm | no improvement |
| POFD top4, PNP-inlier choice | 294.6mm | 125.0mm | 218.7mm | 146.0mm | better mean than its bad init, still worse than POFD top1 |
| POFD top8 oracle diagnostic | 129.4mm | 125.0mm | 207.1mm | 143.7mm | solver degrades even oracle candidates |
| POFD top1, 2 refinement iterations | 217.6mm | 125.0mm | 221.2mm | 148.7mm | second iteration does not help |

Interpretation:

- For the current q50 controlled cache, the final localization result should
  remain **selected candidate pose = POFD top1**.
- A naive extra render-LoFTR refinement step is not a reliable handoff solver;
  it succeeds numerically on all rows but degrades median pose accuracy.
- The basin metric must be reported as a geometric proxy until an external
  solver proves its own acceptance basin.  A GS-SMC-style train-free solver is
  still worth evaluating, but it must beat the top1 identity handoff under the
  same exported-cache protocol.

## 2026-05-21 Cambridge Reference-Pose Banks

Implemented public multi-scene candidate-bank tooling:

```text
feature_extract/localizability/reference_pose_bank.py
feature_extract/tools/build_reference_pose_candidate_bank.py
```

Built HLoc/NetVLAD top10 reference-pose banks for Cambridge scenes:

```text
result/result/feature_extract/localizability_banks/cambridge_reference_pose/
```

Current retrieval-order baseline:

| scene | queries | top1 median trans | oracle@10 median trans | note |
| --- | ---: | ---: | ---: | --- |
| GreatCourt | 760 | 6.724m | 2.869m | large rerank headroom |
| KingsCollege | 343 | 2.909m | 1.985m | moderate rerank headroom |
| OldHospital | 182 | 3.978m | 2.276m | large rerank headroom |
| ShopFacade | 103 | 1.377m | 0.736m | usable second scene |
| StMarysChurch | 530 | 2.839m | 1.458m | large rerank headroom |

This is not yet a POFD-FS result.  It is the public benchmark substrate needed
to test raw RADIO / query-student / POFD-FS reranking without relying on DCFF
rendering or OldHospital-only evidence.

## 2026-05-21 Q50 Failure Replay Check

Train q50 dump:

```text
rows: 96
wrong top1: 55
wrong fraction: 57.3%
wrong outside basin: 29
wrong-row selected mean cost: 0.333m
wrong-row oracle mean cost: 0.204m
wrong-row mean gap: 0.130m
oracle_gap >= 0.05m replay rows: 34
```

Implemented:

- `feature_extract/localizability/failure_replay.py`
- `--failure-replay-rows`
- `--failure-replay-mode all|failure_only`
- `--failure-pair-weight`
- cached selected-wrong vs oracle margin loss

Validation:

```text
pytest -q tests/test_localizability_core.py tests/test_localizability_tools.py
14 passed
```

Small replay training result:

| run | step | pred | gap | top1 | Spearman | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| failure-only, pair loss w2, lr5e-5 | 60 | 0.2227 | 0.0934 | 0.7188 | 0.580 | no improvement |
| all-train, pair loss w4, lr2e-5 | 60 | 0.2227 | 0.0934 | 0.7188 | 0.583 | no improvement |

Conclusion: direct backprop into the heavy pair-matcher score path is not the
next likely source of q50 improvement.  It is memory-heavy, needs batch size 1,
and did not move validation selection even when cached failure loss was active.

## 2026-05-21 Q50 Score-Calibrator Check

Implemented a lightweight frozen-feature hypothesis-score calibrator:

```text
feature_extract/localizability/score_calibrator.py
feature_extract/tools/train_localizability_score_calibrator.py
```

It uses only cached candidate evidence from the POFD-FS table, not external
LoFTR/PnP scores.  Supported evidence includes:

```text
score
score_margin_to_top1
score_rank_norm
score_zscore
delta_trans_m
delta_rot_deg
```

Best balanced q50 val128 result:

```text
run: pofd_fs_scorecal_q50_mlp_h8_l1_rel_resid2_20260521
step: 700
raw: pred 0.2227, top1 0.7188, gap 0.0933, Spearman 0.5864
cal: pred 0.2129, top1 0.7422, gap 0.0835, Spearman 0.5911
```

This improves the original q50 ranking without breaking global ordering, but
it still does not strictly pass promotion:

```text
target: pred <= 0.2100, top1 >= 0.7500, gap <= 0.0900, Spearman >= 0.5500
best:   pred 0.2129, top1 0.7422, gap 0.0835, Spearman 0.5911
```

The result is close: pred is short by about 2.9mm mean error, and top1 is short
by one exact-oracle row on 128 validation samples.  However, larger MLPs are not
a clean fix.  One larger MLP reached top1 0.7578 and gap 0.0899, but Spearman
fell to 0.4856, so it improves exact top1 by damaging the full ranking surface.

Per-row analysis of the best balanced calibrator:

```text
changed rows: 4 / 128
changed rows better: 4 / 4
raw exact-oracle rows: 92 / 128
cal exact-oracle rows: 95 / 128
remaining non-oracle selected mean cost: 0.453m
```

Interpretation:

- The calibrator is not damaging good rows; it only makes a few safe positive
  flips.
- The remaining q50 gap is dominated by unchanged high-cost rows where the
  original pair-matcher score strongly prefers a near-identity candidate.
- Scalar candidate evidence is near saturation.  Further gains need richer
  candidate evidence or a better q50 training distribution, not just a larger
  table MLP.

Motion-prior probe:

```text
raw + train-selected motion prior:
  train pred 0.2606, val pred 0.2178

hand near-identity switch, not train-selected:
  train pred worsens to about 0.269-0.270
  val pred improves to 0.160-0.194
```

This is an important negative control.  The hand rule shows that many q50 val
failures are near-identity-bias failures, but the rule does not improve train
and must not be promoted as a valid method.  It points to a split/lattice
coverage problem: the training q50 rows do not expose the same near-identity
failure distribution strongly enough.

Actionable conclusion:

```text
Do not continue increasing calibrator capacity.
Next q50 work should generate broader randomized/jittered q50 candidate banks
and online score-hard negatives that specifically include near-identity
false-positive candidates, then re-train the score surface.
```

## 2026-05-21 Reference-Pose Feature Scoring Check

Implemented a reference-pose feature-ranking evaluator:

```text
feature_extract/localizability/reference_pose_scoring.py
feature_extract/tools/eval_reference_pose_feature_ranking.py
```

It loads dense feature caches, pools descriptors, and reranks Cambridge
reference-pose candidates using feature similarity.  This is a first public
multi-scene audit, not the final learned POFD-FS reference scorer.

Raw RADIO pooled-feature results on available cached scenes:

| scene | feature key | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | fine_geo | 5.147m | 2.646m | 2.500m | 0.137 | 0.051 | 0.494 | 0.770 |
| OldHospital | coarse_sem | 5.187m | 2.646m | 2.541m | 0.088 | 0.127 | 0.425 | 0.851 |
| ShopFacade | fine_geo | 2.254m | 0.933m | 1.321m | 0.155 | 0.160 | 0.400 | 0.815 |
| ShopFacade | coarse_sem | 2.048m | 0.933m | 1.115m | 0.184 | 0.211 | 0.354 | 0.785 |

Retrieval-order baseline on the same top10 banks:

| scene | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 4.186m | 2.646m | 1.540m | 0.181 | 0.316 | 0.644 | 0.897 |
| ShopFacade | 1.649m | 0.933m | 0.716m | 0.320 | 0.399 | 0.569 | 0.908 |

Conclusion:

- Naive pooled raw RADIO dense features are not a useful public reference-pose
  reranker; they are worse than the original retrieval order.
- This supports the main claim that raw foundation feature similarity is not
  enough for localization utility.
- The public benchmark substrate is ready, but the positive multi-scene result
  still requires a learned selector/scorer or another localization-specific
  feature path.

Implemented a reproducible retrieval-order baseline mode:

```text
feature_extract/localizability/reference_pose_scoring.py::retrieval_order_scores
feature_extract/tools/eval_reference_pose_feature_ranking.py --score-mode retrieval_order
```

Five-scene Cambridge retrieval-order baseline:

| scene | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | 14.089m | 5.860m | 8.229m | 0.121 | 0.166 | 0.288 | 0.742 |
| KingsCollege | 3.444m | 2.194m | 1.250m | 0.149 | 0.321 | 0.418 | 0.821 |
| OldHospital | 4.186m | 2.646m | 1.540m | 0.181 | 0.316 | 0.655 | 0.897 |
| ShopFacade | 1.649m | 0.933m | 0.716m | 0.320 | 0.399 | 0.800 | 0.975 |
| StMarysChurch | 3.746m | 1.952m | 1.794m | 0.174 | 0.269 | 0.488 | 0.826 |

Available feature-cache status:

```text
raw RADIO cache:
  OldHospital: available
  ShopFacade: available
  KingsCollege/GreatCourt/StMarysChurch: not available

query-adaptive cache:
  OldHospital: debug export only, limit=2
  other scenes: not available
```

This blocks a fair "3/5 scenes positive" POFD-FS reference-pose claim today.
The next implementation step is descriptor-only export for query-adapted /
POFD-FS features, because full dense export is large and unnecessary for
reference-pose ranking.

## 2026-05-21 Q50 Failure-Mode Clustering

Implemented reproducible candidate-table failure clustering:

```text
feature_extract/localizability/failure_replay.py::summarize_candidate_failure_modes
result/result/feature_extract/pofd_fs_failure_modes_20260521/summary.json
```

Result:

| split | samples | wrong top1 | near-identity wrong | outside-basin wrong | high-margin wrong | wrong selected mean | wrong oracle mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train q50 | 96 | 55 | 5 | 29 | 0 | 0.333m | 0.204m |
| val q50 | 128 | 36 | 26 | 33 | 9 | 0.461m | 0.129m |

Interpretation:

- The q50 validation failure is dominated by near-identity false positives.
- The q50 training table does not expose the same failure distribution
  strongly enough, which explains why hand near-identity rules improve val but
  worsen train.
- This makes the current q50 gap a data/candidate-bank coverage problem, not a
  scalar calibrator capacity problem.

Required next q50 experiment:

```text
Generate jittered/randomized q50 training banks with explicit near-identity
false positives, then train online score-hard negatives against those banks.
Do not train on q50 val failure rows.
```

## 2026-05-21 Mapability Baseline

Implemented mapability diagnostics:

```text
feature_extract/localizability/mapability.py::observation_track_feature_variance
feature_extract/tools/eval_feature_track_mapability.py
```

OldHospital sparse model has zero stored 2D point observations in `images.bin`,
so the evaluator falls back to projecting `points3D.bin` into cameras and
sampling the feature maps.

Raw RADIO projected-track baseline:

| feature | images | observations | tracks | dim | variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse_sem | 40 | 1236 | 567 | 64 | 1.4404 |
| fine_geo | 24 | 163 | 59 | 64 | 0.0033 |

Interpretation:

- The mapability protocol is now executable and records feature dimension,
  track count, observation count, and within-track variance.
- This is only a raw RADIO baseline.  It does not yet prove selected POFD-FS
  mapability because full selected/query-adaptive dense caches are not exported.
- The next efficient step is descriptor/track-observation export for selected
  features rather than writing full dense feature maps for every Cambridge
  image.

## 2026-05-21 Interpretability Interface

Implemented counterfactual diagnostics:

```text
feature_extract/localizability/interpretability.py
  channel_group_counterfactual_drop
  spatial_utility_counterfactual_drop
```

These report how much selected pose cost changes when removing channel groups
or high/low utility spatial regions.  Tests verify that:

- removing high-utility channel groups increases selected pose cost more than
  removing low-impact groups;
- removing high-utility spatial evidence can flip the selected candidate while
  removing low-utility evidence leaves it unchanged.

This makes the expert-requested interpretability ablation implementable, but it
still needs exported selected-feature score maps and utility maps to become a
paper result.

## 2026-05-21 Query-Student Descriptor Reference Ranking

Implemented a lightweight descriptor export path:

```text
feature_extract/tools/export_query_student_descriptors.py
```

Important engineering corrections:

- Descriptor export now runs RGB-only and no longer requires the historical
  teacher feature cache.  This avoids blocking on missing
  `features_radio_adaptive_v7` cache files.
- The tool uses the student export resolution from config
  (`student_feature_hw`, `student_coarse_feature_hw`) so checkpoint shapes match.
- Legacy `fine_loc_*` checkpoint keys are explicitly dropped because the current
  `RadioQueryStudent` no longer instantiates or forwards that old head.  The
  exported descriptor is therefore the current-code-compatible base `fine`
  feature, not the deprecated `fine_loc`.

Verification:

```text
OldHospital descriptor export:
  records: 1077
  descriptor: current query-student fine, 96D
  output: result/result/feature_extract/pofd_fs_descriptors/oldhospital_query_adaptive_fine_all_20260521.pt
```

Reference-pose top10 ranking on OldHospital:

| Method | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retrieval order | 4.186 | 2.646 | 1.540 | 0.181 | 0.316 | 0.655 | 0.897 |
| Raw RADIO fine pooled | 5.147 | 2.646 | 2.500 | 0.137 | 0.051 | 0.494 | 0.770 |
| Raw RADIO coarse pooled | 5.187 | 2.646 | 2.541 | 0.088 | 0.127 | 0.425 | 0.851 |
| Query-student fine pooled | 5.195 | 2.646 | 2.548 | 0.093 | 0.065 | 0.345 | 0.690 |

Conclusion:

- Pooled query-student descriptors do **not** improve reference-pose ranking.
- This is a negative result against an image-level global descriptor claim.
- The current positive evidence remains the hypothesis-ranking path with
  rendered candidates and low-resolution large-radius pair-matcher scoring.
- Next reference-pose work should not rely on simple global average pooling.
  It needs a learned localizability selector/scorer or patch-level evidence,
  otherwise retrieval-order is stronger.

## 2026-05-21 Q50 Calibrator Stability Check

Repeated the best small-capacity q50 calibrator family with four additional
seeds, keeping the same cached evidence and feature set:

```text
features:
  score, score_margin_to_top1, score_rank_norm, score_zscore,
  delta_trans_m, delta_rot_deg
model:
  MLP hidden=8, layers=1, score_residual_weight=2.0
```

| Run | pred | top1 | gap | Spearman | basin@1 |
|---|---:|---:|---:|---:|---:|
| previous best seed12 | 0.2129 | 0.7422 | 0.0835 | 0.5911 | 0.7734 |
| seed13 | 0.2137 | 0.7422 | 0.0844 | 0.5763 | 0.7734 |
| seed14 | 0.2177 | 0.7344 | 0.0884 | 0.5734 | 0.7578 |
| seed15 | 0.2206 | 0.7266 | 0.0913 | 0.5909 | 0.7500 |
| seed16 | 0.2208 | 0.7188 | 0.0915 | 0.5852 | 0.7422 |

Conclusion:

- The small calibrator consistently improves q50 over raw cached scores
  (`0.2227m -> ~0.213-0.221m`) and improves basin@1 in the best runs.
- It does **not** reliably pass the strict promotion threshold
  (`pred <= 0.21`, `top1 >= 0.75`) across seeds.
- Larger/scalar-only calibration is not the right next lever.  The remaining
  q50 gap is a training-distribution problem: val failures are dominated by
  near-identity false positives, while the current train table does not expose
  enough of those hard negatives.
- Next q50 work should generate randomized/jittered q50 train candidate banks
  and re-score them with the validated low-resolution pair-matcher path.

## 2026-05-21 Near-Init Hard-Negative Cache Probe

Implemented:

```text
feature_extract/tools/augment_pose_candidate_cache.py
feature_extract/configs/pofd_stage3a_multiscale_score_q50_val128_top25_eval.yaml
```

Purpose:

- Append identity and near-init jitter candidates to q50 train cache.
- Make the q50 train distribution contain the val-like near-identity false
  positives seen in failure analysis.

Generated cache:

```text
input:
  oldhospital_local_lattice_renderloftr_q50cm10deg_top16_noexact_train256_shuf20260515.npz
output:
  oldhospital_local_lattice_renderloftr_q50cm10deg_top16_plus_nearinit9_train256_shuf20260521.npz
added:
  identity + 8 jitter candidates
  trans_cm = [2.5, 5, 10]
  rot_deg = [0.5, 1, 2]
```

Important implementation detail:

- The existing eval config silently truncated candidate caches to top16.
- A top25 config was required to actually score the appended candidates.
- The verified top25 candidate table contains `96 * 25 = 2400` rows.

Raw top25 train score-table result:

| Train cache | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| q50 top16 train | 0.269 | 0.195 | 0.074 | 0.427 | 0.764 | 0.844 | 0.984 |
| q50 top25 near-init train | 0.244 | 0.161 | 0.082 | 0.344 | 0.753 | 0.740 | 0.973 |

Training the same small calibrator on top25 near-init train did **not** improve
q50 val:

| Train table | best val pred | top1 | gap | Spearman | basin@1 |
|---|---:|---:|---:|---:|---:|
| top16 train, seed13 | 0.2137 | 0.7422 | 0.0844 | 0.5763 | 0.7734 |
| top25 near-init train, seed13 | 0.2206 | 0.7266 | 0.0913 | 0.5854 | 0.7500 |

Conclusion:

- Naively appending near-init candidates is not sufficient and can degrade the
  train/val alignment.
- The remaining q50 issue is not solved by simply adding identity-like rows.
  The additional candidates need to be sampled so that they are
  score-hard wrong candidates under the current model, not just geometrically
  near the initializer.
- Next q50 probe should mine online score-high wrong candidates from the
  scorer itself and build a replay buffer/cache from those, rather than adding
  generic near-init jitter.

## 2026-05-21 Score-Hard Pair Mining And Replay Probe

Implemented:

```text
feature_extract/localizability/failure_replay.py
  mine_score_hard_candidate_pairs
  load_mined_score_hard_pairs

feature_extract/tools/mine_score_hard_candidate_pairs.py

feature_extract/tools/train_localizability_score_calibrator.py
  --replay-pairs
  --replay-weight
  --replay-margin
```

Definition:

For each query, choose:

```text
positive = lowest pose-cost candidate
negative = highest-score candidate with cost >= positive_cost + cost_gap
```

This mines the candidates that the current score surface actually likes among
geometrically worse hypotheses.

Mined-pair summaries:

| Table | samples | pairs | near-identity neg | outside-basin neg | neg score > pos |
|---|---:|---:|---:|---:|---:|
| q50 val top16 | 128 | 128 | 107 | 117 | 36 |
| q50 train top16 | 96 | 96 | 4 | 73 | 33 |
| q50 train top25 near-init | 96 | 96 | 4 | 72 | 42 |

Key finding:

- q50 validation hard negatives are overwhelmingly near-identity
  (`107/128`), while current train hard negatives are not (`4/96`).
- The top25 near-init augmentation did not create train hard negatives that
  look like val failures under the current scorer; even after appending
  near-init candidates, only `4/96` mined negatives are near-identity.

Replay-loss experiment:

| Train setup | best val pred | top1 | gap | Spearman | basin@1 |
|---|---:|---:|---:|---:|---:|
| top16, no replay seed13 | 0.2137 | 0.7422 | 0.0844 | 0.5763 | 0.7734 |
| top16 + mined replay | 0.2158 | 0.7344 | 0.0864 | 0.5833 | 0.7656 |
| top25 near-init + mined replay | 0.2197 | 0.7266 | 0.0904 | 0.5679 | 0.7500 |

Conclusion:

- Static replay on current train hard pairs does not cross promotion.
- The failure is now pinned down more tightly: the validation failures are
  near-identity score traps that the current train candidate distribution does
  not reproduce.
- The next useful experiment is not more scalar calibration.  It is to build a
  q50 train cache by **mining candidates that the frozen scorer scores highly**
  from a larger random/jittered candidate pool, then train against those mined
  candidates.

## Next Mainline Steps

1. **Broader q50 score-surface training**
   - Keep foundation/query/map features frozen.
   - Generate randomized/jittered q50 candidate banks so the correct correction
     is not tied to a fixed lattice artifact.
   - Force online score-hard negatives to include near-identity false positives
     and repeated-structure false positives.
   - Re-train the hypothesis scorer with richer candidate evidence; do not just
     increase scalar calibrator capacity.

2. **Public reference-pose localizability benchmark**
   - Use the newly built Cambridge reference-pose banks for:
     `ShopFacade`, `OldHospital`, `KingsCollege`, `GreatCourt`,
     `StMarysChurch`.
   - Raw RADIO pooled scoring is now a negative baseline.
   - Add query-student / learned POFD-FS reference-pose feature scoring.
   - Labels are geometric pose distance and basin membership only.
   - Report top1/top5, Spearman, NDCG, oracle gap, basin recall@K.

3. **Mapability validation**
   - Compare raw RADIO, query student, and POFD-FS selected feature.
   - Report track/primitive variance, rendered ranking retention, and feature
     storage dimension.

4. **Interpretability**
   - Channel-group counterfactual removal.
   - Spatial utility top/bottom masking.
   - Required evidence: removing high-utility channels/regions hurts ranking
     more than removing low-utility ones.

## Promote Conditions

- OldHospital q10/q25 improve at least 20% over raw/query/reconstruction.
- OldHospital q50 reaches:
  `pred <= 0.21`, `top1 >= 0.75`, `gap <= 0.09`, `Spearman >= 0.55`.
- At least 3/5 Cambridge scenes show nontrivial reference-pose ranking
  improvement over raw/query baselines.
- Selected feature is mapable: lower 3D aggregation variance at <=64D.
- Real-init stress improves basin recall@K or risk coverage without claiming
  SOTA CPR.

## Stop Conditions

- If q10/q25 cannot beat raw/query baselines, the selection claim fails.
- If q50 only improves through external LoFTR/PnP score leakage, POFD-FS is not
  a valid main method.
- If second-scene/reference-pose ranking is negative across Cambridge, pivot to
  a benchmark/diagnostic paper.
- If mapability collapses, the method is only a 2D scorer and should not be
  advertised as a visual localization feature for maps.
