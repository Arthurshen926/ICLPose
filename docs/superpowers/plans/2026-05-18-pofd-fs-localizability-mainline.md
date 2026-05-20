# 2026-05-18 POFD-FS Localizability Mainline

## Decision

The radio branch should stop presenting itself as a continuous camera pose
refinement method. The mainline is now **POFD-FS: Pose-Observable Foundation
Feature Selection**.

The central claim is:

> Frozen foundation dense features contain a compact, mapable,
> localization-usable subspace. The selected feature should optimize
> basin-aware pose-hypothesis ranking rather than direct sparse correspondence
> or continuous feature-metric pose updates.

## What Changes

- Stage4 continuous CPR is demoted to a no-go diagnostic.
- LoFTR / MASt3R / GS-CPR-style solvers are external baselines, teachers, or
  stress-test comparators, not the POFD-FS main method.
- The primary task is Level-2 hypothesis utility:
  `score(query, map, candidate_pose)` should rank candidates by geometric
  usefulness and solver basin membership.
- RADIO and DCFF geometry remain frozen in early phases.

## New Code Boundary

New package:

```text
feature_extract/localizability/
  selector.py
  scorer.py
  losses.py
  candidate_bank.py
  metrics.py
  mapability.py
  visualization.py
```

New tools:

```text
feature_extract/tools/build_localizability_banks.py
feature_extract/tools/eval_localizability_audit.py
feature_extract/tools/train_localizability_selector.py
feature_extract/tools/eval_localizability_feature_score.py
feature_extract/tools/train_localizability_selector_stream.py
feature_extract/tools/report_localizability.py
```

The new package is intentionally separate from
`train_nvs_pose_feature_adapter.py` so future changes do not keep expanding the
old CPR script.

## Initial Implementation Status

Implemented:

- `LocalizationFeatureSelector`: channel group gate, 1x1 compact projection,
  spatial utility head, uncertainty head, L2-normalized selected feature.
- `PoseHypothesisScorer`: same-pixel, local-correlation, and true
  `pair_matcher_local` scoring. The pair-matcher path now uses the existing
  pair-conditioned center-offset evidence instead of plain shift-invariant
  max-correlation.
- Rank losses: pose-distance listwise KL, basin BCE, online score-hard negative,
  channel sparsity, spatial utility entropy.
- Metrics: top1, oracle cost, oracle gap, Spearman, NDCG, basin recall@K.
- Candidate-bank dataclass and NPZ loader for standardized banks.
- Mapability diagnostic: track-level feature variance.
- Minimal audit/report/train tools for standardized candidate banks and
  pre-rendered query/candidate feature tensors.
- Streaming audit/training tools that avoid writing large candidate feature
  tensors to disk.
- Stage3a `PoseFeatureDomainAdapter + PairConditionedLocalMatcher` loading in
  the new POFD-FS audit path, so raw/query/adapted localization features can be
  compared under the same report interface.

Generated smoke/validation artifacts:

```text
result/result/feature_extract/localizability_banks/oldhospital_q10_val128_standard.npz
result/result/feature_extract/localizability_banks/oldhospital_q25_val128_standard.npz
result/result/feature_extract/localizability_banks/oldhospital_q50_val128_standard.npz
result/result/feature_extract/pofd_fs_audit_retrieval_val128_20260518.md
result/result/feature_extract/pofd_fs_audit_raw_query_poseadapter_val128_20260518.md
result/result/feature_extract/pofd_fs_audit_poseadapter_q50_scoremode_20260518.md
result/result/feature_extract/pofd_fs_pairmatcher_geom_train_q50_20260518.md
result/result/feature_extract/pofd_fs_audit_pose_adapter_pairmatcher_r16_q50_val128_dump_20260518/rows.jsonl
result/result/feature_extract/pofd_fs_audit_pose_adapter_pairmatcher_r16_q50_val128_shuf_dump_20260518/rows.jsonl
result/result/feature_extract/pofd_fs_shopfacade_controlled_q25_top64_audit16_20260518/summary.json
result/result/feature_extract/pofd_fs_shopfacade_controlled_q50_audit16_20260518/summary.json
```

Important caveat:

The existing `retrieval_scores_candidates` in the controlled local-lattice
cache are not a valid POFD-FS training signal or primary baseline. On q10/q25/q50
val128 they select the geometric oracle candidate with top1=1.0 and Spearman=1.0.
This is useful as a leakage check, but it must not be used as feature-selection
supervision.

## OldHospital Controlled Ranking, 2026-05-18

All rows use val128 controlled local-lattice candidates, frozen DCFF geometry,
and no external score labels. `raw` and `query_student` use simple
34x60/r4 local correlation. `pose_adapter` uses the Stage3a adapted feature and
true 34x60/r16 pair-matcher center-offset score path.

| feature / scorer | q10 pred | q10 gap | q10 top1 | q25 pred | q25 gap | q25 top1 | q50 pred | q50 gap | q50 top1 | q50 spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw RADIO + local r4 | 0.086 | 0.060 | 0.227 | 0.189 | 0.124 | 0.273 | 0.300 | 0.171 | 0.508 | 0.664 |
| query student + local r4 | 0.059 | 0.033 | 0.219 | 0.140 | 0.075 | 0.594 | 0.274 | 0.144 | 0.594 | 0.714 |
| pose-adapted + pair matcher r16 | 0.045 | 0.019 | 0.750 | 0.107 | 0.043 | 0.773 | 0.223 | 0.093 | 0.719 | 0.586 |

Conclusions:

- The new POFD-FS entry point preserves the Stage3a positive result rather than
  falling back to the weaker local-corr path.
- The feature-adapted pair-matcher path clearly beats raw RADIO and query
  student on q10/q25/q50 pred cost and oracle gap.
- q10/q25 satisfy the 20% improvement requirement over raw/query baselines.
- q50 is close but not yet promoted: `pred_cost_m=0.223` is above the
  `0.21` target, `top1=0.719` is below `0.75`, and `oracle_gap=0.093` is just
  above `0.09`. Spearman already passes the `0.55` threshold.

q50 score-mode diagnostics:

| adapted scorer | q50 pred | q50 gap | top1 | spearman | basin@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| r12 `logprob + margin` | 0.225 | 0.095 | 0.711 | 0.571 | 0.977 |
| r16 `logprob + margin` | 0.223 | 0.093 | 0.719 | 0.586 | 0.984 |
| r16 `center_logprob` only | 0.223 | 0.093 | 0.719 | 0.586 | 0.984 |
| r16 `center_margin` only | 0.224 | 0.094 | 0.719 | 0.593 | 0.984 |

This rules out an easy promotion by radius or fixed channel selection. The
remaining q50 gap is not a cosmetic aggregation issue; it needs score-surface
training or better hard-negative supervision.

Geometry-only q50 fine-tuning:

| run | best pred | best gap | best top1 | note |
| --- | ---: | ---: | ---: | --- |
| pair matcher only, lr 2e-5 | 0.223 | 0.093 | 0.719 | unchanged |
| adapter + pair matcher, lr 5e-6 | 0.223 | 0.093 | 0.719 | unchanged |
| pair matcher only, lr 1e-4 | 0.223 | 0.093 | 0.719 | step20 worsened to 0.225 |
| adapter + pair matcher, lr 2e-5 | 0.223 | 0.093 | 0.719 | unchanged |

These runs use only pose-cost listwise ranking, basin BCE, and online
score-hard negatives. They do not use LoFTR/PnP candidate-quality labels. The
negative result is useful: simply continuing Stage3a with geometry-only
short-horizon hard-negative training does not move the q50 discrete selection.

q50 failure dump:

- 128 rows, 36 wrong top1 selections (`28.1%`).
- All 33 selected candidates outside the 25cm/10deg basin are wrong selections.
- Wrong selected mean cost is `0.461m`; oracle mean remains `0.129m`, so wrong
  rows carry a large `0.332m` mean selected-oracle gap.
- On the unshuffled cache the oracle is always index 1 and wrong selections are
  mostly fixed indices 5/13/7. On the shuffled cache the same metrics hold, but
  oracle/selected indices are distributed across all 16 positions. Therefore
  the q50 gap is not just candidate index leakage; it is a real score ordering
  failure on about one quarter of rows.

## ShopFacade Probe, 2026-05-18

Controlled lattice probes on ShopFacade are not yet positive:

- q50/top16 has poor candidate coverage: oracle mean is `0.333m`, so the table
  is not suitable for judging feature ranking.
- q25/top64 has usable oracle mean `0.103m`, but all tested score spaces have
  top1 `0.0` on the first 16 validation images. The adapted feature improves
  Spearman/AUC over raw/query in that small probe, but selected pose cost is
  still around `0.286m`, worse than the oracle and not a second-scene success.

Conclusion: the OldHospital controlled evidence is real, but the second-scene
requirement is not met. ShopFacade needs either a proper controlled candidate
bank/export protocol or scene-specific POFD-FS training before it can support
the paper claim.

Selector training note:

- Naive identity-initialized selector training on top of query-student features
  did not improve q50; projection training degraded ranking, while utility-only
  training stayed at the baseline. The next optimization should therefore
  target q50 score ordering in the pair-matcher/adapted-feature path rather than
  training a free projection mask on the weaker local-corr score.

## Next Experiments

1. Optimize q50 within the adapted-feature pair-matcher score path:
   - move beyond short-horizon fine-tuning, because r12/r16/channel selection
     and 20-step geometry-only fine-tuning did not improve q50;
   - add diagnostics for which q50 rows are still score-dominant wrong
     candidates, then train with those failure modes explicitly represented;
   - keep geometry frozen and keep external score labels out of the objective.
2. Build a clean ShopFacade controlled candidate bank with adequate oracle
   coverage, then repeat raw/query/adapted ranking on that bank.
3. Add mapability validation before any low-LR map-side adaptation.
4. Add real-init stress only after q50 controlled ranking clears the promotion
   threshold or a clear stop condition is reached.

## Promotion Thresholds

- OldHospital q10/q25: selected feature improves pred cost or oracle gap by at
  least 20% over raw RADIO or reconstruction baseline.
- OldHospital q50 val128: `pred_cost_m <= 0.21`, `top1 >= 0.75`,
  `spearman >= 0.55`, `oracle_gap_m <= 0.09`.
- At least one second scene shows nontrivial improvement.
- Selected feature has lower 3D aggregation variance than raw/reconstruction.
- Real-init stress improves basin recall@K or risk coverage without catastrophic
  degradation.

## Current Caveat

The current evidence supports the new feature-selection claim on OldHospital
controlled ranking, especially q10/q25. The remaining blocker is q50
score-ordering: the adapted feature contains useful signal, but the selected
candidate still misses the topK oracle often enough to keep pred cost above the
paper threshold. Second-scene and mapability evidence are still missing.
