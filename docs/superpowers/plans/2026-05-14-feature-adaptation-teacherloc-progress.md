# 2026-05-14 Feature Adaptation / TeacherLoc Progress

## Context

After `ChatGPT-特征训练与可微检索 (3).md`, the mainline is no longer
RADIO/DCFF scalar similarity tuning.  The active question is whether RADIO can
serve as a base anchor while a matching/localization teacher reshapes the final
loc feature.

## Implemented

- NVS adapter training now consumes offline LoFTR/render-PnP teacher
  correspondences.
- Added `nvs_teacher_correspondence_loss`:
  - sparse symmetric InfoNCE on teacher query-map matches;
  - local patch supervision around teacher matches;
  - metrics exposed as `nvs_teacher_corr_*` and `nvs_teacher_patch_*`.
- `eval_cpr_buckets.build_model_and_data` now loads
  `TeacherCorrespondenceStore`, so NVS train/eval loaders can carry teacher
  matches.
- Fixed NVS config precedence:
  - `nvs_pose_feature_adapter.*` now overrides generic `pose_energy.*`;
  - this fixed `synthetic_ratio: 0.0` being incorrectly overwritten by the
    Stage-1 default `0.5`.
- Added RGB/local texture context to `PoseFeatureDomainAdapter`:
  - optional query/render RGB stems;
  - adapter residual can condition on local RGB context as well as RADIO/DCFF
    base feature.
- Added clean config:
  - `feature_extract/configs/nvs_pose_feature_adapter_stage1_teacherloc.yaml`
- Added `PairConditionedLocalMatcher`:
  - starts from normalized dot-product prior;
  - learns a pair-conditioned residual over query point, render patch,
    relative offset, and local score context;
  - trained by `nvs_teacher_pair_match_loss`.
- Added RoMa-style RGB fine/local texture branch to `PoseFeatureDomainAdapter`:
  - RADIO/DCFF remains the base anchor;
  - optional query/render RGB ConvNet branch injects localizable texture
    evidence into the final loc feature.
- Added candidate scoring from pair-conditioned heatmaps:
  - `score_mode: pair_matcher_local`;
  - sparse grid heatmaps score each pose hypothesis by center probability and
    center-vs-hard-negative margin;
  - this avoids relying only on hand-written cosine statistics.
- Added foundation-guided matching/localization feature head:
  - `texture_fusion_mode: replace`;
  - RADIO/DCFF base is only an anchor/input via `base_anchor_weight`;
  - RGB/matching branch can dominate the final localization feature.
- Added configs:
  - `feature_extract/configs/nvs_pose_feature_adapter_stage1_teacherloc_adapteronly.yaml`
  - `feature_extract/configs/nvs_pose_feature_adapter_stage1_teacherloc_direction_rank.yaml`
  - `feature_extract/configs/nvs_pose_feature_adapter_stage1_teacherloc_pairscore.yaml`

## Verification

- `pytest tests/test_pose_energy_net.py tests/test_nvs_pose_feature_adapter.py ...`: 67 passed.
- `py_compile` passed for:
  - `feature_extract/students/pose_energy_net.py`
  - `feature_extract/tools/train_pose_energy.py`
  - `feature_extract/tools/train_nvs_pose_feature_adapter.py`
  - `feature_extract/tools/eval_cpr_buckets.py`

## Experiments

All runs used real query teacher correspondences only (`synthetic_fraction=0.0`).

| Run | step | corr gap | corr acc | patch gap | patch acc | pred cost | selected cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacherloc no-RGB corr+patch | 10 | -0.0535 | 0.4157 | -0.0748 | 0.2072 | 0.2719 | -0.4151 |
| teacherloc no-RGB corr+patch | 20 | -0.0528 | 0.4196 | -0.0742 | 0.2106 | 0.2830 | -0.0781 |
| teacherloc no-RGB patch-strong | 10 | -0.0535 | 0.4162 | -0.0748 | 0.2077 | 0.2719 | -0.4151 |
| teacherloc no-RGB patch-strong | 20 | -0.0529 | 0.4196 | -0.0739 | 0.2135 | 0.2830 | -0.0781 |
| teacherloc RGB corr+patch | 10 | -0.0535 | 0.4157 | -0.0749 | 0.2077 | 0.2719 | -0.4151 |
| teacherloc RGB corr+patch | 20 | -0.0528 | 0.4196 | -0.0743 | 0.2096 | 0.2830 | -0.0781 |
| teacherloc RGB patch-strong | 10 | -0.0535 | 0.4152 | -0.0748 | 0.2072 | 0.2719 | -0.4151 |
| teacherloc RGB patch-strong | 20 | -0.0529 | 0.4196 | -0.0740 | 0.2116 | 0.2830 | -0.0781 |

Second-round Stage-1 runs after implementing pair-conditioned matching:

| Run | step | eval patch gap | eval patch acc | eval pair gap | eval pair acc | pred trans | selected cos | Spearman | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pair matcher, no score context | 20 | -0.0418 | 0.2547 | -0.0397 | 0.2726 | 0.2832 | -0.3138 | -0.0612 | pair residual too weak |
| pair matcher + score context | 20 | -0.0418 | 0.2547 | -0.0180 | 0.4876 | 0.2832 | -0.3138 | -0.0612 | heatmap decoder learns teacher patch, pose scorer unchanged |
| adapter-only stronger residual | 20 | -0.0168 | 0.3891 | n/a | n/a | 0.2649 | -0.2565 | 0.0664 | feature moves, still not pose-sortable |
| adapter + RGB texture branch | 20 | -0.0173 | 0.3875 | n/a | n/a | 0.2649 | -0.2565 | 0.0716 | texture branch did not solve dot-product localizability |
| direction-rank feature score | 20 | -0.0180 | 0.3774 | n/a | n/a | 0.2492 | -0.1467 | 0.0761 | train hard-negative signal strong, eval still negative |
| pairscore, directed train | 20 | -0.0202 | 0.3393 | -0.0081 | 0.4973 | 0.2381 | -0.1385 | 0.1113 | best pose-direction signal so far, still below go condition |
| pairscore, adaptive train | 20 | -0.0223 | 0.3391 | -0.0114 | 0.4662 | 0.2460 | -0.1829 | 0.1061 | matching train/eval candidate distribution did not help |
| matchingloc main | 20 | -0.0091 | 0.0498 | 0.0178 | 0.9845 | 0.2257 | 0.0225 | 0.0814 | first independent eval with positive selected direction; AUC 0.75 |
| matchingloc main | 40 | -0.0060 | 0.0628 | 0.1198 | 1.0000 | 0.2258 | 0.0285 | 0.1458 | best selected direction but still far below go condition |
| matchingloc main | 60 | -0.0157 | 0.2701 | 0.2457 | 0.9999 | 0.2266 | 0.0099 | 0.0859 | pair matcher saturated; candidate score still not discriminative |

## Conclusion

The teacher correspondence path is now real and measurable, and the
pair-conditioned heatmap decoder is the first component that clearly improves a
feature-localization metric:

```text
eval pair acc: 0.2726 -> 0.4876
eval pair gap: -0.0397 -> -0.0180
```

However, the current adapted feature is still not pose-sortable enough:

```text
best selected correction cosine remains negative (-0.1385)
Stage-1 go condition requires > 0.50
```

Pure feature residuals remain too weak:

```text
RADIO/DCFF base + residual adapter (+ optional RGB stem)
```

Texture-branch and stronger residual variants move features but do not create a
stable dot-product pose signal.  This supports the expert warning: the final
localization feature cannot be just RADIO/DCFF plus a shallow projector.

The reset mainline is now more faithful to the expert response:

```text
RADIO/DCFF = frozen context/base anchor
RGB + matching teacher = final localization feature
pair-conditioned heatmap = candidate evidence
hard-direction ranking = Stage-1 pose sorting objective
PoseEnergy/map adaptation = disabled until Stage-1 passes
```

This changed behavior: independent eval selected correction cosine is no longer
strongly negative, and Stage-1 AUC reached 0.75 at step 20.  But it still does
not satisfy the Stage-1 gate:

```text
target: selected_correction_cos >= 0.50, Spearman >= 0.30, trans <= 150mm
best observed: selected_correction_cos 0.0285, Spearman 0.1458, trans 225.8mm
```

Failure mode after the reset:

```text
pair matcher learns center heatmaps for all candidates
-> local pair accuracy saturates
-> candidate score becomes weak/flat
-> top1 pose direction remains unreliable
```

## Next Architecture Step

Do not spend more time on teacher-corr weight tuning or raw dot-product feature
visual appearance.  The next useful change is:

```text
Q_base/M_base/RGB -> pair-conditioned local matcher / heatmap decoder
                  -> candidate score / fine selector
```

The pair-conditioned evidence should become the main localization feature for
Stage 1/2.  If this still cannot make independent selected correction cosine
positive with a larger teacher/candidate cache, the project should promote
LoFTR/RoMa/MASt3R-derived matching features as the final localization feature
teacher, with RADIO kept only as context/base anchor.

Immediate next step is not another residual tweak.  Build or cache
candidate-level teacher quality:

```text
for each query and pose candidate:
  render RGB/depth
  run LoFTR/RoMa/MASt3R teacher
  store inlier count / confidence / pair quality / hard false positives
train pairscore on this cache with listwise and hard-negative losses
```

This directly implements the expert `soft match heatmap + match confidence +
candidate pair quality + hard false-positive` recommendation.

## Update: local-lattice teacher-quality cache

Implemented the next architecture step directly:

- Added `feature_retrieval/local_lattice_render_loftr_quality_export.py`.
- It exports a CPR-local cache where:
  - `pose_init` is the external T0;
  - `pose_init_candidates` are local hypotheses around T0;
  - render-LoFTR/PnP inliers, match count, confidence, reprojection quality are
    teacher targets for each candidate.
- Added same-magnitude opposite-direction hard candidates.
- Added no-exact cache mode by excluding correction fraction `1.0`, avoiding
  fixed-lattice 0mm oracle artifacts.
- Added configs:
  - `nvs_pose_feature_adapter_stage1_matchingloc_teacherquality_cache.yaml`
  - `nvs_pose_feature_adapter_stage1_matchingloc_teacherquality_local_noexact.yaml`
  - `nvs_pose_feature_adapter_stage1_matchingloc_teacherquality_local_noexact_256.yaml`

Verification:

```text
py_compile passed:
  feature_retrieval/local_lattice_render_loftr_quality_export.py
  feature_extract/tools/eval_cpr_buckets.py

pytest:
  tests/test_local_lattice_render_loftr_quality_export.py
  tests/test_nvs_pose_feature_adapter.py::test_candidate_teacher_quality_scores_from_render_loftr_pnp_fields
  tests/test_adaptive_joint_query_map.py::test_joint_radio_dataset_loads_pose_candidate_quality_fields
  -> 6 passed
```

Teacher-cache audit:

| Cache | Queries | Candidates | LoFTR/PnP success | Oracle cost | Teacher-selected cost |
|---|---:|---:|---:|---:|---:|
| local exact smoke | 4 | 64 | 100% | ~0mm | ~0mm |
| local no-exact train64 | 64 | 1024 | 100% | 62.7mm | 177.0mm |
| local no-exact val32 | 32 | 512 | 100% | 62.7mm | 115.5mm |
| local no-exact train256 | 256 | 4096 | 100% | ~125mm median | n/a |
| local no-exact val128 | 128 | 2048 | 100% | ~125mm median | n/a |

Training results:

| Run | Eval step | selected cos | pred trans | pred rot | Spearman | top1 | identity | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retrieval-cache teacher quality | 10 | 0.3939 | 89.9mm | 0.210deg | 0.402 | 0.075 | 0.20 | 60.0mm |
| local no-exact 64/32 | 40 | 0.6945 | 117.2mm | 1.531deg | 0.346 | 0.700 | 0.00 | 55.2mm |
| local no-exact pose-heavy resume | 20 | 0.5732 | 140.6mm | 1.656deg | 0.374 | 0.575 | 0.00 | 78.8mm |
| local no-exact 256/128 resume | 30 | 0.6021 | 135.9mm | 1.625deg | 0.375 | 0.600 | 0.00 | 74.1mm |

Current conclusion:

```text
The expert-guided local teacher cache direction is valid.
It is the first route to pass selected_correction_cos > 0.50
without identity collapse.
```

Compared with the old matchingloc mainline:

```text
selected correction cosine: ~0.03 -> 0.60-0.69
identity collapse: removed on local no-exact runs
selected trans on local no-exact: 117-136mm, close to but not yet <100mm
```

Remaining bottleneck:

```text
The local teacher quality is useful but not identical to GT pose cost.
No-exact candidates have oracle around 62-125mm depending on split/cache,
while the learned selector still leaves a 55-74mm oracle gap.
```

Do not return to WLS, map candidate-gradient, or raw RADIO cosine tuning.
The next useful experiments are:

1. Improve candidate-bank design:
   - mix 10cm/2deg and 25cm/5deg local caches;
   - avoid exact inverse candidates;
   - ensure every query has symmetric good/near-bad/opposite candidates.
2. Train selector with a two-target schedule:
   - first GT pose/correction ranking;
   - then add LoFTR/PnP quality as a reliability/confidence auxiliary.
3. Add a dedicated selector head over pair heatmaps instead of using the same
   scalar pair-matcher score for all objectives.
4. Only after selector reliably reaches <100mm on 25cm/5deg should map-side
   loc-adapter fine-tuning resume.
